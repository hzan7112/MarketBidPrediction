#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04b_train_template_parameter_models_frozen_v9.py

Feasibility version:
    Direct 21-point price-curve residual prediction
    + quantity specialist prediction

Price target:
    P_j - lt_bid_level, j=0..20

The 21 price points correspond exactly to the Stage2 normalized quantity grid:
    x = 0, 0.05, ..., 1.00

Quantity target:
    q_base_mw, q_span_mw, q1..q5

Feature policy:
    Z_base + Z_tr + M + U + current-template one-hot condition

Direct participant historical bid state H remains excluded.

This version deliberately abandons low-dimensional price theta such as
p_base/alpha/beta or PCA coefficients. The purpose is to test whether the
available strategy/profile/market/unit features can predict the curve itself.

Run:
python scripts/bidprediction/04b_train_template_parameter_models_frozen_v9.py --year 2025
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
TEMPLATES = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
T2I = {t: i for i, t in enumerate(TEMPLATES)}

LT = [
    "lt_bid_level",
    "lt_adjustment_magnitude",
    "lt_strategy_persistence",
    "lt_quantity_hhi",
    "lt_effective_segment_count",
    "lt_flat_curve_rate",
    "lt_tail_uplift_ratio",
    "lt_curve_bend_ratio",
    "lt_shape_variability",
]
ST = [
    "st_bid_level_z",
    "st_adjustment_bias_z",
    "st_adjustment_magnitude_z",
    "st_quantity_hhi_z",
    "st_effective_segment_count_z",
    "st_flat_curve_rate_z",
    "st_tail_uplift_ratio_z",
    "st_curve_bend_ratio_z",
    "st_shape_shift",
]
BR = [
    "break_bid_level",
    "break_adjustment_bias",
    "break_adjustment_magnitude",
    "break_quantity_hhi",
    "break_effective_segment_count",
    "break_flat_curve_rate",
    "break_tail_uplift_ratio",
    "break_curve_bend_ratio",
]

EXPECTED_GROUP_DIMS = {
    "Z_base": 26,
    "Z_tr": 26,
    "M": 8,
    "U": 12,
}

EXCLUDE = {
    "rolling_lt_ready_flag",
    "st_ready_flag",
    "profile_ready_flag",
    "market_ready_flag",
    "unit_state_ready_flag",
    "prediction_ready_flag",
    "parameter_label_available_flag",
    "parameter_target_valid_flag",
    "parameter_template_match_flag",
    "parameter_ready_flag",
    "theta_target_valid_flag",
    "theta_ready_flag",
    "theta_quantity_sentinel_flag",
    "market_nonmissing_count",
    "rolling_lt_nonmissing_count",
    "hist_prev_available_flag",
    "tr_ready_flag",
}

PRICE_CANDIDATES = [
    "Ridge",
    "RandomForest",
    "RandomForest_weighted",
    "ExtraTrees",
    "ExtraTrees_weighted",
]

Q_CANDIDATES = [
    "Ridge",
    "RandomForest",
    "ExtraTrees",
]

Q_SCALE_MODES = [
    "absolute",
    "unit_lag1_max_ecomax",
    "unit_lag1_avg_ecomax",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def norm(s):
    return s.astype("string").str.strip()


def load_manifest(frozen: Path):
    p = frozen / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(z):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.expm1(np.abs(z))


def shape_cols(header):
    exact = [f"shape_v{i:02d}" for i in range(21)]
    return exact if all(c in header for c in exact) else None


def month_key(path: Path):
    m = re.search(r"(\d{4})[_-](\d{2})", path.stem)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def discover_curve_samples(year_dir: Path):
    root = year_dir / "curve_samples"
    if not root.exists():
        raise FileNotFoundError(root)

    out = {}
    for p in sorted(root.rglob("*.csv")):
        try:
            h = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue

        if "sample_id" not in h:
            continue

        sc = shape_cols(h)
        if sc is None:
            continue

        mk = month_key(p)
        if mk:
            out.setdefault(mk, []).append(p)

    if not out:
        raise FileNotFoundError(
            f"No Stage2 curve samples with sample_id + shape_v00..20 under {root}"
        )

    return out


def load_month_shapes(files, wanted_ids=None):
    wanted = None
    if wanted_ids is not None:
        wanted = set(
            pd.Series(wanted_ids, dtype="string")
            .astype("string")
            .str.strip()
            .tolist()
        )

    blocks = []

    for p in files:
        h = pd.read_csv(p, nrows=0).columns.tolist()
        sc = shape_cols(h)
        if sc is None:
            continue

        d = pd.read_csv(
            p,
            usecols=["sample_id", *sc],
            low_memory=False,
        )

        d["sample_id"] = norm(d["sample_id"])

        if wanted is not None:
            d = d.loc[d["sample_id"].isin(wanted)].copy()

        if d.empty:
            continue

        d = d.rename(
            columns={
                c: f"shape_v{i:02d}"
                for i, c in enumerate(sc)
            }
        )

        d["__stage2_row_found"] = 1
        blocks.append(d)

    if not blocks:
        return pd.DataFrame(
            columns=[
                "sample_id",
                *SHAPE_COLS,
                "__stage2_row_found",
            ]
        )

    out = pd.concat(blocks, ignore_index=True)

    if out["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in Stage2 curve samples.")

    return out


def attach_shapes(d: pd.DataFrame, sample_files: dict):
    if all(c in d.columns for c in SHAPE_COLS):
        out = d.copy()
        flat = norm(out["y_template_id"]).str.upper().eq("FLAT")
        if flat.any():
            out.loc[flat, SHAPE_COLS] = 0.0
        return out

    base = d.copy()
    base["sample_id"] = norm(base["sample_id"])
    base["__order"] = np.arange(len(base), dtype=np.int64)

    months = (
        pd.to_datetime(
            base["local_date"],
            errors="coerce",
        )
        .dt.strftime("%Y-%m")
    )

    if months.isna().any():
        raise ValueError("Invalid local_date in frozen rows.")

    blocks = []

    for mk in sorted(months.unique()):
        if mk not in sample_files:
            raise FileNotFoundError(
                f"No Stage2 curve samples for month {mk}"
            )

        ids = base.loc[months.eq(mk), "sample_id"]

        blocks.append(
            load_month_shapes(
                sample_files[mk],
                wanted_ids=ids,
            )
        )

    shapes = pd.concat(blocks, ignore_index=True)

    out = base.merge(
        shapes,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    out = (
        out.sort_values("__order")
        .reset_index(drop=True)
    )

    joined = (
        num(out["__stage2_row_found"])
        .fillna(0)
        .eq(1)
    )

    if not joined.all():
        bad = int((~joined).sum())
        ex = out.loc[~joined, "sample_id"].head(10).tolist()
        raise ValueError(
            f"{bad:,} rows have no Stage2 sample_id match. Examples: {ex}"
        )

    flat = (
        norm(out["y_template_id"])
        .str.upper()
        .eq("FLAT")
    )

    if flat.any():
        out.loc[flat, SHAPE_COLS] = 0.0

    nonflat = ~flat

    ok = out.loc[
        nonflat,
        SHAPE_COLS,
    ].notna().all(axis=1)

    if not ok.all():
        bad_index = ok.index[~ok]
        ex = out.loc[
            bad_index,
            "sample_id",
        ].head(10).tolist()

        raise ValueError(
            f"{len(bad_index):,} non-FLAT rows have incomplete shape. "
            f"Examples: {ex}"
        )

    return out.drop(
        columns=[
            "__order",
            "__stage2_row_found",
        ]
    )


def build_features(schema):
    s = schema[
        schema["role"]
        .astype(str)
        .str.lower()
        .eq("feature")
    ].copy()

    fmap = dict(
        zip(
            s["column"].astype(str),
            s["feature_group"].astype(str),
        )
    )

    avail = set(fmap) - EXCLUDE

    zb = LT + ST + BR

    missing = [
        c for c in zb
        if c not in avail
    ]

    if missing:
        raise KeyError(
            f"Missing Z_base columns: {missing}"
        )

    ztr = [
        c for c, g in fmap.items()
        if g == "transition_strategy_profile"
        and c in avail
    ]

    M = [
        c for c, g in fmap.items()
        if g == "market_environment"
        and c in avail
    ]

    U = [
        c for c, g in fmap.items()
        if g == "unit_state_proxy"
        and c in avail
    ]

    H = [
        c for c, g in fmap.items()
        if g == "participant_history"
        and c in avail
    ]

    groups = {
        "Z_base": zb,
        "Z_tr": ztr,
        "M": M,
        "U": U,
        "H_excluded": H,
    }

    for name, expected in EXPECTED_GROUP_DIMS.items():
        if len(groups[name]) != expected:
            raise ValueError(
                f"{name}: expected {expected}, got {len(groups[name])}"
            )

    features = zb + ztr + M + U

    if any(c in H for c in features):
        raise RuntimeError("participant_history leakage detected.")

    return features, groups


def prepare_X(d, features, template=None):
    out = {
        c: num(d[c]).astype("float32")
        for c in features
    }

    if template is None:
        t = norm(d["y_template_id"])
    else:
        if isinstance(template, pd.Series):
            t = norm(template)
        else:
            t = norm(
                pd.Series(
                    template,
                    index=d.index,
                    dtype="string",
                )
            )

    ti = t.map(T2I)

    if ti.isna().any():
        bad = t[ti.isna()].unique().tolist()
        raise ValueError(
            f"Unknown template condition: {bad[:20]}"
        )

    a = ti.to_numpy(np.int16)

    for j, name in enumerate(TEMPLATES):
        out[f"cond_template_{name}"] = (
            a == j
        ).astype("float32")

    return pd.DataFrame(
        out,
        index=d.index,
    )


def fit_preprocess(train, features):
    xdf = prepare_X(train, features)

    imp = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X = imp.fit_transform(xdf).astype(
        np.float32,
        copy=False,
    )

    scaler = StandardScaler()

    Xs = scaler.fit_transform(X).astype(
        np.float32,
        copy=False,
    )

    return imp, scaler, X, Xs


def uses_scaled_x(candidate):
    return candidate == "Ridge"


def make_model(candidate, args):
    base = candidate.replace("_weighted", "")

    if base == "Ridge":
        return Ridge(
            alpha=args.ridge_alpha,
            solver="lsqr",
        )

    if base == "RandomForest":
        return RandomForestRegressor(
            n_estimators=args.rf_trees,
            max_depth=args.rf_depth,
            min_samples_leaf=args.rf_leaf,
            max_features=args.rf_max_features,
            n_jobs=-1,
            random_state=args.seed,
        )

    if base == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=args.et_trees,
            max_depth=args.et_depth,
            min_samples_leaf=args.et_leaf,
            max_features=args.et_max_features,
            n_jobs=-1,
            random_state=args.seed,
        )

    raise KeyError(candidate)


def empirical_rank_weight(x):
    r = (
        pd.Series(np.asarray(x, dtype=float))
        .rank(method="average", pct=True)
        .to_numpy(float)
    )

    w = 1.0 + 2.0 * r * r

    return w / np.mean(w)


def predict_model(model, candidate, X, Xs):
    Xin = Xs if uses_scaled_x(candidate) else X

    y = np.asarray(
        model.predict(Xin),
        dtype=float,
    )

    if y.ndim == 1:
        y = y[:, None]

    return y


def true_price_curve(d):
    pa = num(d["p_anchor"]).to_numpy(float)
    ps = num(d["p_span"]).to_numpy(float)

    S = (
        d[SHAPE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    flat = (
        norm(d["y_template_id"])
        .str.upper()
        .eq("FLAT")
        .to_numpy()
    )

    if flat.any():
        S[flat] = 0.0
        ps = ps.copy()
        ps[flat] = 0.0

    P = pa[:, None] + ps[:, None] * S

    valid = (
        np.isfinite(pa)
        & np.isfinite(ps)
        & np.isfinite(S).all(axis=1)
        & np.isfinite(P).all(axis=1)
    )

    return P, S, valid


def fit_price_state(train):
    baseline = num(train["lt_bid_level"])

    finite = baseline[np.isfinite(baseline)]

    if finite.empty:
        raise ValueError(
            "lt_bid_level has no finite training values."
        )

    fill = float(finite.median())

    P, S, valid = true_price_curve(train)

    if not valid.all():
        raise ValueError(
            "Invalid training price-curve target."
        )

    b = baseline.to_numpy(float)
    b = np.where(
        np.isfinite(b),
        b,
        fill,
    )

    R = P - b[:, None]

    mean = R.mean(axis=0)
    std = R.std(axis=0)
    std = np.where(std > 1e-10, std, 1.0)

    Z = (R - mean[None, :]) / std[None, :]

    return {
        "baseline_column": "lt_bid_level",
        "baseline_fill": fill,
        "target_mean": mean,
        "target_std": std,
        "Z_train": Z,
        "true_price_train": P,
    }


def reconstruct_price(d, state, standardized_pred):
    Z = np.asarray(standardized_pred, dtype=float)

    R = (
        Z
        * np.asarray(
            state["target_std"],
            dtype=float,
        )[None, :]
        + np.asarray(
            state["target_mean"],
            dtype=float,
        )[None, :]
    )

    b = num(
        d[state["baseline_column"]]
    ).to_numpy(float)

    b = np.where(
        np.isfinite(b),
        b,
        float(state["baseline_fill"]),
    )

    return b[:, None] + R


def fit_price_model(
    candidate,
    X,
    Xs,
    state,
    args,
):
    model = make_model(
        candidate,
        args,
    )

    Xin = Xs if uses_scaled_x(candidate) else X

    kwargs = {}

    if candidate.endswith("_weighted"):
        scale = np.mean(
            np.abs(
                state["true_price_train"]
            ),
            axis=1,
        )

        kwargs["sample_weight"] = empirical_rank_weight(scale)

    model.fit(
        Xin,
        state["Z_train"],
        **kwargs,
    )

    return model


def evaluate_price(
    paths,
    candidate,
    model,
    features,
    imputer,
    scaler,
    state,
    sample_files,
):
    rows = 0
    abs_sum = 0.0
    sq_sum = 0.0
    true_abs_sum = 0.0
    smape_sum = 0.0
    smape_count = 0

    curve_mae = []

    for p in paths:
        d = pd.read_pickle(p)

        if d.empty:
            continue

        d = attach_shapes(
            d,
            sample_files,
        )

        X = imputer.transform(
            prepare_X(d, features)
        ).astype(
            np.float32,
            copy=False,
        )

        Xs = scaler.transform(X).astype(
            np.float32,
            copy=False,
        )

        z = predict_model(
            model,
            candidate,
            X,
            Xs,
        )

        pp = reconstruct_price(
            d,
            state,
            z,
        )

        pt, _, valid = true_price_curve(d)

        if not valid.all():
            pp = pp[valid]
            pt = pt[valid]

        e = pp - pt
        ae = np.abs(e)

        abs_sum += float(np.sum(ae))
        sq_sum += float(np.sum(e * e))
        true_abs_sum += float(
            np.sum(np.abs(pt))
        )

        den = np.abs(pt) + np.abs(pp)
        smv = den > 1e-8

        if smv.any():
            smape_sum += float(
                np.sum(
                    2.0
                    * ae[smv]
                    / den[smv]
                )
            )
            smape_count += int(smv.sum())

        curve_mae.extend(
            np.mean(ae, axis=1).tolist()
        )

        rows += len(pt)

        del d, X, Xs, z, pp, pt
        gc.collect()

    c = np.asarray(curve_mae, dtype=float)

    return {
        "rows": rows,
        "price_mae": (
            abs_sum / (rows * 21)
        ),
        "price_rmse": np.sqrt(
            sq_sum / (rows * 21)
        ),
        "price_wape_pct": (
            100.0
            * abs_sum
            / max(true_abs_sum, 1e-12)
        ),
        "price_smape_pct": (
            100.0
            * smape_sum
            / smape_count
            if smape_count
            else np.nan
        ),
        "curve_mae_p50": (
            float(np.quantile(c, 0.50))
            if len(c)
            else np.nan
        ),
        "curve_mae_p90": (
            float(np.quantile(c, 0.90))
            if len(c)
            else np.nan
        ),
        "curve_mae_p95": (
            float(np.quantile(c, 0.95))
            if len(c)
            else np.nan
        ),
        "curve_mae_le_20_share": (
            float(np.mean(c <= 20.0))
            if len(c)
            else np.nan
        ),
    }


# ---------------------------------------------------------------------
# Quantity
# ---------------------------------------------------------------------

def fit_q_spec(train, mode):
    spec = {
        "q_scale_mode": mode,
        "q_scale_column": None,
        "q_scale_fill": 1.0,
        "q_scale_floor": 1.0,
    }

    if mode != "absolute":
        x = num(train[mode])

        vals = x[
            np.isfinite(x)
            & (x > 1.0)
        ]

        if vals.empty:
            raise ValueError(
                f"{mode} has no positive train values."
            )

        spec["q_scale_column"] = mode
        spec["q_scale_fill"] = float(
            vals.median()
        )

    return spec


def q_scale_value(d, spec):
    col = spec["q_scale_column"]

    if col is None:
        return np.ones(
            len(d),
            dtype=float,
        )

    x = num(d[col]).to_numpy(float)

    fill = float(
        spec["q_scale_fill"]
    )

    floor = float(
        spec["q_scale_floor"]
    )

    x = np.where(
        np.isfinite(x)
        & (x > floor),
        x,
        fill,
    )

    return np.maximum(x, floor)


def q_latent(d, spec):
    scale = q_scale_value(d, spec)

    qb = (
        num(d["theta_q_base_mw"]).to_numpy(float)
        / scale
    )

    qs = (
        np.maximum(
            num(d["theta_q_span_mw"]).to_numpy(float),
            1e-8,
        )
        / scale
    )

    Q = (
        d[
            [
                "theta_q1",
                "theta_q2",
                "theta_q3",
                "theta_q4",
                "theta_q5",
            ]
        ]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    Q = np.maximum(Q, 1e-8)

    Q /= Q.sum(
        axis=1,
        keepdims=True,
    )

    logits = np.log(
        Q[:, :4]
        / Q[:, 4, None]
    )

    return np.c_[
        signed_log1p(qb),
        np.log1p(qs),
        logits,
    ]


def fit_q_state(train, spec):
    Y = q_latent(train, spec)

    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    std = np.where(std > 1e-10, std, 1.0)

    Z = (Y - mean[None, :]) / std[None, :]

    return {
        "mean": mean,
        "std": std,
        "Z_train": Z,
    }


def inverse_q(d, spec, state, z_scale, z_shape):
    Z = np.zeros(
        (len(d), 6),
        dtype=float,
    )

    Z[:, :2] = z_scale
    Z[:, 2:6] = z_shape

    Y = (
        Z
        * state["std"][None, :]
        + state["mean"][None, :]
    )

    scale = q_scale_value(d, spec)

    qb = signed_expm1(Y[:, 0]) * scale

    qs = (
        np.maximum(
            np.expm1(Y[:, 1]),
            1e-8,
        )
        * scale
    )

    L = np.c_[
        Y[:, 2:6],
        np.zeros(len(Y)),
    ]

    L -= L.max(
        axis=1,
        keepdims=True,
    )

    E = np.exp(L)

    Q = E / E.sum(
        axis=1,
        keepdims=True,
    )

    return qb, qs, Q


def fit_q_model(candidate, X, Xs, Y, args):
    model = make_model(candidate, args)

    Xin = Xs if uses_scaled_x(candidate) else X

    model.fit(Xin, Y)

    return model


def evaluate_qscale(
    paths,
    candidate,
    model,
    features,
    imputer,
    scaler,
    spec,
    state,
):
    rows = 0

    qa_abs = qs_abs = qm_abs = 0.0
    qa_true = qs_true = qm_true = 0.0

    for p in paths:
        d = pd.read_pickle(p)

        if d.empty:
            continue

        X = imputer.transform(
            prepare_X(d, features)
        ).astype(np.float32)

        Xs = scaler.transform(X).astype(
            np.float32
        )

        z = predict_model(
            model,
            candidate,
            X,
            Xs,
        )

        full_scale = z[:, :2]

        true_lat = q_latent(d, spec)

        true_shape_z = (
            true_lat[:, 2:6]
            - state["mean"][None, 2:6]
        ) / state["std"][None, 2:6]

        qb_p, qs_p, _ = inverse_q(
            d,
            spec,
            state,
            full_scale,
            true_shape_z,
        )

        qb_t = num(
            d["theta_q_base_mw"]
        ).to_numpy(float)

        qs_t = num(
            d["theta_q_span_mw"]
        ).to_numpy(float)

        qm_t = qb_t + qs_t
        qm_p = qb_p + qs_p

        qa_abs += float(
            np.sum(np.abs(qb_p - qb_t))
        )

        qs_abs += float(
            np.sum(np.abs(qs_p - qs_t))
        )

        qm_abs += float(
            np.sum(np.abs(qm_p - qm_t))
        )

        qa_true += float(
            np.sum(np.abs(qb_t))
        )

        qs_true += float(
            np.sum(np.abs(qs_t))
        )

        qm_true += float(
            np.sum(np.abs(qm_t))
        )

        rows += len(d)

    qa_w = 100.0 * qa_abs / max(qa_true, 1e-12)
    qs_w = 100.0 * qs_abs / max(qs_true, 1e-12)
    qm_w = 100.0 * qm_abs / max(qm_true, 1e-12)

    return {
        "rows": rows,
        "q_anchor_mae_mw": qa_abs / rows,
        "q_span_mae_mw": qs_abs / rows,
        "q_max_mae_mw": qm_abs / rows,
        "q_anchor_wape_pct": qa_w,
        "q_span_wape_pct": qs_w,
        "q_max_wape_pct": qm_w,
        "selection_score": float(
            np.mean([qa_w, qs_w, qm_w])
        ),
    }


def evaluate_qshape(
    paths,
    candidate,
    model,
    features,
    imputer,
    scaler,
    state,
):
    rows = 0
    qshare_abs = 0.0
    break_abs = 0.0

    for p in paths:
        d = pd.read_pickle(p)

        if d.empty:
            continue

        X = imputer.transform(
            prepare_X(d, features)
        ).astype(np.float32)

        Xs = scaler.transform(X).astype(
            np.float32
        )

        z = predict_model(
            model,
            candidate,
            X,
            Xs,
        )

        Y = (
            z
            * state["std"][None, 2:6]
            + state["mean"][None, 2:6]
        )

        L = np.c_[
            Y,
            np.zeros(len(Y)),
        ]

        L -= L.max(
            axis=1,
            keepdims=True,
        )

        E = np.exp(L)

        Qp = E / E.sum(
            axis=1,
            keepdims=True,
        )

        Qt = (
            d[
                [
                    "theta_q1",
                    "theta_q2",
                    "theta_q3",
                    "theta_q4",
                    "theta_q5",
                ]
            ]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )

        Qt = np.maximum(Qt, 1e-8)

        Qt /= Qt.sum(
            axis=1,
            keepdims=True,
        )

        qshare_abs += float(
            np.sum(np.abs(Qp - Qt))
        )

        bp = np.cumsum(Qp, axis=1)[:, :4]
        bt = np.cumsum(Qt, axis=1)[:, :4]

        break_abs += float(
            np.sum(np.abs(bp - bt))
        )

        rows += len(d)

    qmae = qshare_abs / (rows * 5)
    bmae = break_abs / (rows * 4)

    return {
        "rows": rows,
        "q_share_mae": qmae,
        "breakpoint_fraction_mae": bmae,
        "selection_score": 0.5 * (qmae + bmae),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--bidtemplate-root",
        default="data/processed/bidtemplate",
    )

    ap.add_argument("--ridge-alpha", type=float, default=10.0)

    ap.add_argument("--rf-trees", type=int, default=72)
    ap.add_argument("--rf-depth", type=int, default=18)
    ap.add_argument("--rf-leaf", type=int, default=8)
    ap.add_argument(
        "--rf-max-features",
        type=float,
        default=0.7,
    )

    ap.add_argument("--et-trees", type=int, default=72)
    ap.add_argument("--et-depth", type=int, default=18)
    ap.add_argument("--et-leaf", type=int, default=8)
    ap.add_argument(
        "--et-max-features",
        type=float,
        default=0.7,
    )

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    frozen = base / "frozen_modeling_dataset"

    manifest = load_manifest(frozen)

    schema = pd.read_csv(
        frozen / "feature_schema.csv"
    )

    features, groups = build_features(schema)

    train = pd.read_pickle(
        frozen / manifest["train_sample_file"]
    )

    val_parts = [
        frozen / p
        for p in manifest["parts"]["val"]
    ]

    test_parts = [
        frozen / p
        for p in manifest["parts"]["test_curve"]
    ]

    sample_files = discover_curve_samples(
        Path(args.bidtemplate_root)
        / str(args.year)
    )

    print("=" * 80)
    print(
        "Template parameter regression - direct 21-point price curve"
    )
    print("=" * 80)
    print(
        f"Frozen train sample: {len(train):,}"
    )
    print(
        "Feature policy: Z_base + Z_tr + M + U + template condition"
    )
    print(
        "H / participant direct historical bid state: disabled"
    )
    print()

    print(
        "Joining Stage2 shapes to fixed train sample...",
        flush=True,
    )

    train_curve = attach_shapes(
        train,
        sample_files,
    )

    flat_count = int(
        norm(train_curve["y_template_id"])
        .str.upper()
        .eq("FLAT")
        .sum()
    )

    print(
        f"Stage2 shape join complete: {len(train_curve):,}/{len(train):,}"
    )
    print(
        f"FLAT zero-shape rows: {flat_count:,}"
    )
    print()

    imputer, scaler, X, Xs = fit_preprocess(
        train,
        features,
    )

    # ---------------------------------------------------------------
    # Direct price curve.
    # ---------------------------------------------------------------

    price_state = fit_price_state(
        train_curve
    )

    price_rows = []
    price_models = {}

    for candidate in PRICE_CANDIDATES:
        print(
            f"[price] {candidate}",
            flush=True,
        )

        model = fit_price_model(
            candidate,
            X,
            Xs,
            price_state,
            args,
        )

        metrics = evaluate_price(
            val_parts,
            candidate,
            model,
            features,
            imputer,
            scaler,
            price_state,
            sample_files,
        )

        price_rows.append(
            {
                "candidate": candidate,
                **metrics,
            }
        )

        price_models[candidate] = model

    price_df = (
        pd.DataFrame(price_rows)
        .sort_values(
            [
                "price_wape_pct",
                "price_mae",
                "curve_mae_p90",
            ]
        )
        .reset_index(drop=True)
    )

    best_price_name = str(
        price_df.iloc[0]["candidate"]
    )

    best_price_model = price_models[
        best_price_name
    ]

    # ---------------------------------------------------------------
    # Quantity scale.
    # ---------------------------------------------------------------

    qscale_rows = []
    qscale_models = {}

    for mode in Q_SCALE_MODES:
        spec = fit_q_spec(train, mode)
        state = fit_q_state(train, spec)

        zscale = state["Z_train"][:, :2]

        for candidate in Q_CANDIDATES:
            print(
                f"[q_scale] mode={mode} / {candidate}",
                flush=True,
            )

            model = fit_q_model(
                candidate,
                X,
                Xs,
                zscale,
                args,
            )

            metrics = evaluate_qscale(
                val_parts,
                candidate,
                model,
                features,
                imputer,
                scaler,
                spec,
                state,
            )

            qscale_rows.append(
                {
                    "q_scale_mode": mode,
                    "candidate": candidate,
                    **metrics,
                }
            )

            qscale_models[
                (mode, candidate)
            ] = (
                spec,
                state,
                model,
            )

    qscale_df = (
        pd.DataFrame(qscale_rows)
        .sort_values("selection_score")
        .reset_index(drop=True)
    )

    best_q_mode = str(
        qscale_df.iloc[0]["q_scale_mode"]
    )

    best_qscale_name = str(
        qscale_df.iloc[0]["candidate"]
    )

    q_spec, q_state, q_scale_model = (
        qscale_models[
            (
                best_q_mode,
                best_qscale_name,
            )
        ]
    )

    # ---------------------------------------------------------------
    # Quantity shape.
    # ---------------------------------------------------------------

    zshape = q_state["Z_train"][:, 2:6]

    qshape_rows = []
    qshape_models = {}

    for candidate in Q_CANDIDATES:
        print(
            f"[q_shape] {candidate}",
            flush=True,
        )

        model = fit_q_model(
            candidate,
            X,
            Xs,
            zshape,
            args,
        )

        metrics = evaluate_qshape(
            val_parts,
            candidate,
            model,
            features,
            imputer,
            scaler,
            q_state,
        )

        qshape_rows.append(
            {
                "candidate": candidate,
                **metrics,
            }
        )

        qshape_models[candidate] = model

    qshape_df = (
        pd.DataFrame(qshape_rows)
        .sort_values("selection_score")
        .reset_index(drop=True)
    )

    best_qshape_name = str(
        qshape_df.iloc[0]["candidate"]
    )

    q_shape_model = qshape_models[
        best_qshape_name
    ]

    # ---------------------------------------------------------------
    # Save final bundle.
    # ---------------------------------------------------------------

    bundle = {
        "version": "04b-direct-price-v9",
        "model_mode": "direct_21_price_q_specialist",
        "feature_policy": "Z_base + Z_tr + M + U",
        "features": features,
        "history_features_used": False,
        "templates": TEMPLATES,
        "imputer": imputer,
        "x_scaler": scaler,
        "price_state": {
            k: v
            for k, v in price_state.items()
            if k != "Z_train"
            and k != "true_price_train"
        },
        "price_model": {
            "candidate": best_price_name,
            "uses_scaled_x": uses_scaled_x(
                best_price_name
            ),
            "model": best_price_model,
        },
        "q_spec": q_spec,
        "q_state": {
            k: v
            for k, v in q_state.items()
            if k != "Z_train"
        },
        "q_scale_model": {
            "candidate": best_qscale_name,
            "uses_scaled_x": uses_scaled_x(
                best_qscale_name
            ),
            "model": q_scale_model,
        },
        "q_shape_model": {
            "candidate": best_qshape_name,
            "uses_scaled_x": uses_scaled_x(
                best_qshape_name
            ),
            "model": q_shape_model,
        },
    }

    out = base / "template_parameter_models"

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mdir = out / "models"

    mdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for stale in mdir.glob("*.joblib"):
        stale.unlink()

    fs = "Z_base_Z_tr_M_U"

    joblib.dump(
        bundle,
        mdir / f"{fs}.joblib",
        compress=3,
    )

    price_df.to_csv(
        out / "price_model_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qscale_df.to_csv(
        out / "q_scale_model_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qshape_df.to_csv(
        out / "q_shape_model_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Test metrics for the selected direct-price model.
    test_price = evaluate_price(
        test_parts,
        best_price_name,
        best_price_model,
        features,
        imputer,
        scaler,
        price_state,
        sample_files,
    )

    selection = {
        "selected_feature_set": fs,
        "selected_model": "Direct21PriceSpecialist",
        "participant_history_used": False,
        "price_representation": (
            "21-point price residual to lt_bid_level"
        ),
        "price_model": best_price_name,
        "q_scale_mode": best_q_mode,
        "q_scale_model": best_qscale_name,
        "q_shape_model": best_qshape_name,
        "selected_val_score": float(
            price_df.iloc[0]["price_wape_pct"]
        ),
        "selected_test_score": float(
            test_price["price_wape_pct"]
        ),
    }

    pd.DataFrame(
        [selection]
    ).to_csv(
        out
        / "selected_template_parameter_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cfg = {
        "version": "04b-direct-price-v9",
        "year": args.year,
        "feature_policy": (
            "Z_base + Z_tr + M + U + template one-hot"
        ),
        "price_target": (
            "P_j - lt_bid_level for j=0..20"
        ),
        "price_candidates": PRICE_CANDIDATES,
        "quantity_scale_modes": Q_SCALE_MODES,
        "quantity_candidates": Q_CANDIDATES,
        "generalization_policy": {
            "participant_specific_rule": False,
            "template_specific_exception": False,
            "fixed_price_threshold": False,
            "direct_history_features_used": False,
        },
    }

    (
        out / "config.json"
    ).write_text(
        json.dumps(
            cfg,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        (
            "Template parameter regression - "
            f"direct 21-point price curve - {args.year}"
        ),
        "=" * 80,
        "",
        f"Frozen train sample: {len(train):,}",
        (
            "Feature policy: "
            "Z_base + Z_tr + M + U + template condition"
        ),
        (
            "H / participant direct historical bid state: disabled"
        ),
        "",
        "Price representation:",
        (
            "  21-point price residual relative to lt_bid_level"
        ),
        (
            f"  selected model = {best_price_name}"
        ),
        "",
        "Validation price:",
        (
            f"  MAE={price_df.iloc[0]['price_mae']:.6f}"
        ),
        (
            f"  WAPE={price_df.iloc[0]['price_wape_pct']:.4f}%"
        ),
        (
            f"  sMAPE={price_df.iloc[0]['price_smape_pct']:.4f}%"
        ),
        (
            f"  curve P50/P90/P95="
            f"{price_df.iloc[0]['curve_mae_p50']:.3f} / "
            f"{price_df.iloc[0]['curve_mae_p90']:.3f} / "
            f"{price_df.iloc[0]['curve_mae_p95']:.3f}"
        ),
        (
            f"  MAE<=20 share="
            f"{price_df.iloc[0]['curve_mae_le_20_share']:.2%}"
        ),
        "",
        "TEST price:",
        (
            f"  MAE={test_price['price_mae']:.6f}"
        ),
        (
            f"  WAPE={test_price['price_wape_pct']:.4f}%"
        ),
        (
            f"  sMAPE={test_price['price_smape_pct']:.4f}%"
        ),
        (
            f"  curve P50/P90/P95="
            f"{test_price['curve_mae_p50']:.3f} / "
            f"{test_price['curve_mae_p90']:.3f} / "
            f"{test_price['curve_mae_p95']:.3f}"
        ),
        (
            f"  MAE<=20 share="
            f"{test_price['curve_mae_le_20_share']:.2%}"
        ),
        "",
        "Quantity:",
        (
            f"  q scale mode = {best_q_mode}"
        ),
        (
            f"  q scale model = {best_qscale_name}"
        ),
        (
            f"  q shape model = {best_qshape_name}"
        ),
    ]

    summary = "\n".join(lines)

    (
        out / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
