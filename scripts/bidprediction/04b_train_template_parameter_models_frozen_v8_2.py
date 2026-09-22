#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04b_train_template_parameter_models_frozen.py

Stage 3 / 04b - final feasibility version:
    template + direct price scale + low-dimensional residual-shape basis

Why this version
----------------
The previous p_base / alpha / beta representation had good representation
capacity but poor predictability. This version changes the PRICE parameterization
instead of continuing to tune the same targets.

Price representation:
    P(x) = p_anchor + p_span * S_hat(x)

    S_hat(x) = template_center(x)
             + PCA_residual(x)

The PCA basis is learned ONLY from the frozen training sample residual shapes:
    residual_shape = true_normalized_shape - template_center

The number of PCA components is selected automatically from training explained
variance, subject only to generic min/max bounds.

Quantity representation is retained:
    q_base_mw, q_span_mw, q1..q5

Feature policy remains:
    Z_base + Z_tr + M + U + current-template condition

H / direct participant historical bid-state features remain excluded.

No dataset-specific price threshold, participant rule, or template exception is
used.

Run:
python scripts/bidprediction/04b_train_template_parameter_models_frozen.py --year 2025
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

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Fixed project schema
# ---------------------------------------------------------------------

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

TEMPLATES = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
T2I = {t: i for i, t in enumerate(TEMPLATES)}

GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]

RAW_Q = [
    "theta_q_base_mw",
    "theta_q_span_mw",
    "theta_q1",
    "theta_q2",
    "theta_q3",
    "theta_q4",
    "theta_q5",
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
    "HGB_squared",
    "HGB_absolute",
    "HGB_weighted",
    "RandomForest",
]

Q_CANDIDATES = [
    "Ridge",
    "HGB_squared",
    "HGB_absolute",
    "RandomForest",
]

P_ANCHOR_MODES = [
    "absolute",
    "residual_lt_bid_level",
]

Q_SCALE_MODES = [
    "absolute",
    "unit_lag1_max_ecomax",
    "unit_lag1_avg_ecomax",
]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def num(s):
    return pd.to_numeric(s, errors="coerce")


def norm(s):
    return s.astype("string").str.strip()


def load_manifest(frozen: Path):
    p = frozen / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p}\nRun 04a3_freeze_modeling_dataset.py first."
        )
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


def normalize_template_id(v):
    s = str(v).strip()

    if s.upper() == "FLAT":
        return "FLAT"

    if s.upper().startswith("T"):
        s = s[1:]

    try:
        i = int(float(s))
        if 0 <= i <= 11:
            return f"T{i:02d}"
    except Exception:
        pass

    return None


def discover_centers(year_dir: Path):
    preferred = (
        year_dir
        / "template_library"
        / "curve_template_library.csv"
    )

    candidates = [preferred] if preferred.exists() else []

    if not candidates:
        candidates = sorted(year_dir.rglob("*.csv"))

    for p in candidates:
        try:
            h = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue

        sc = shape_cols(h)

        id_col = next(
            (
                c
                for c in [
                    "template_id",
                    "template",
                    "cluster_id",
                    "cluster_label",
                    "label",
                ]
                if c in h
            ),
            None,
        )

        if sc is None or id_col is None:
            continue

        d = pd.read_csv(
            p,
            usecols=[id_col] + sc,
        )

        if len(d) > 100:
            continue

        centers = {}

        for _, r in d.iterrows():
            t = normalize_template_id(r[id_col])

            if t is None:
                continue

            a = pd.to_numeric(
                r[sc],
                errors="coerce",
            ).to_numpy(float)

            if np.isfinite(a).all():
                centers[t] = a

        centers["FLAT"] = np.zeros(21)

        if all(t in centers for t in TEMPLATES):
            return centers, p

    raise FileNotFoundError(
        "Cannot locate Stage2 curve template centers."
    )



def month_key_from_path(path: Path):
    m = re.search(
        r"(\d{4})[_-](\d{2})",
        path.stem,
    )
    return (
        f"{m.group(1)}-{m.group(2)}"
        if m
        else None
    )


def discover_curve_samples(year_dir: Path):
    """
    Discover Stage2 curve_samples grouped by YYYY-MM.
    These files are the authoritative source of shape_v00..shape_v20.
    """
    root = (
        year_dir
        / "curve_samples"
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Stage2 curve samples not found: {root}"
        )

    out = {}

    for p in sorted(
        root.rglob("*.csv")
    ):
        try:
            h = pd.read_csv(
                p,
                nrows=0,
            ).columns.tolist()
        except Exception:
            continue

        if "sample_id" not in h:
            continue

        sc = shape_cols(h)

        if sc is None:
            continue

        mk = month_key_from_path(
            p
        )

        if mk:
            out.setdefault(
                mk,
                [],
            ).append(
                p
            )

    if not out:
        raise FileNotFoundError(
            "No Stage2 curve sample files with "
            "sample_id + 21 shape columns were found."
        )

    return out


def load_month_shapes(
    files,
    wanted_ids=None,
):
    blocks = []

    wanted = None

    if wanted_ids is not None:
        wanted = set(
            pd.Series(
                wanted_ids,
                dtype="string",
            )
            .astype("string")
            .str.strip()
            .tolist()
        )

    for p in files:
        h = pd.read_csv(
            p,
            nrows=0,
        ).columns.tolist()

        sc = shape_cols(
            h
        )

        if sc is None:
            continue

        d = pd.read_csv(
            p,
            usecols=[
                "sample_id",
                *sc,
            ],
            low_memory=False,
        )

        d[
            "sample_id"
        ] = (
            d[
                "sample_id"
            ]
            .astype("string")
            .str.strip()
        )

        if wanted is not None:
            d = d.loc[
                d[
                    "sample_id"
                ].isin(
                    wanted
                )
            ].copy()

        if d.empty:
            continue

        d = d.rename(
            columns={
                c: f"shape_v{i:02d}"
                for i, c in enumerate(
                    sc
                )
            }
        )

        blocks.append(
            d
        )

    if not blocks:
        return pd.DataFrame(
            columns=[
                "sample_id",
                *SHAPE_COLS,
            ]
        )

    out = pd.concat(
        blocks,
        ignore_index=True,
    )

    if out[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate sample_id in Stage2 curve samples."
        )

    return out


def attach_stage2_shapes(
    d: pd.DataFrame,
    sample_files: dict,
):
    """
    Restore shape_v00..shape_v20 to frozen train/val/test rows.

    04a3 intentionally stores those columns only in test_curve_parts.
    For train/val, this function joins them from Stage2 curve_samples
    using sample_id and local_date month. The merge is strict one-to-one
    and original row order is preserved.
    """
    if all(
        c in d.columns
        for c in SHAPE_COLS
    ):
        return d

    if "sample_id" not in d.columns:
        raise KeyError(
            "sample_id is required to join Stage2 shapes."
        )

    if "local_date" not in d.columns:
        raise KeyError(
            "local_date is required to join Stage2 shapes."
        )

    base = d.copy()

    base[
        "sample_id"
    ] = (
        base[
            "sample_id"
        ]
        .astype("string")
        .str.strip()
    )

    base[
        "__shape_join_order"
    ] = np.arange(
        len(base),
        dtype=np.int64,
    )

    month = (
        pd.to_datetime(
            base[
                "local_date"
            ],
            errors="coerce",
        )
        .dt.strftime(
            "%Y-%m"
        )
    )

    if month.isna().any():
        raise ValueError(
            "Invalid local_date while joining Stage2 shapes."
        )

    shape_blocks = []

    for mk in sorted(
        month.unique()
    ):
        if mk not in sample_files:
            raise FileNotFoundError(
                f"No Stage2 curve samples found for month {mk}."
            )

        ids = base.loc[
            month.eq(
                mk
            ),
            "sample_id",
        ]

        s = load_month_shapes(
            sample_files[
                mk
            ],
            wanted_ids=ids,
        )

        shape_blocks.append(
            s
        )

    shapes = pd.concat(
        shape_blocks,
        ignore_index=True,
    )

    if shapes[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate sample_id after Stage2 shape collection."
        )

    # Distinguish "sample_id did not join" from the legitimate
    # Stage2 FLAT representation. In Stage2, flat curves are explicit
    # template_family="flat" rows and shape_v00..shape_v20 are NaN by
    # design because no endpoint normalization is performed for them.
    shapes = shapes.copy()
    shapes["__stage2_shape_row_found"] = 1

    out = base.merge(
        shapes,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    out = (
        out
        .sort_values(
            "__shape_join_order"
        )
        .reset_index(
            drop=True
        )
    )

    joined = (
        pd.to_numeric(
            out[
                "__stage2_shape_row_found"
            ],
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
    )

    if not joined.all():
        bad = int(
            (~joined).sum()
        )

        examples = (
            out.loc[
                ~joined,
                "sample_id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{bad:,} frozen rows truly have no Stage2 sample_id match. "
            f"Examples: {examples}"
        )

    template = (
        out[
            "y_template_id"
        ]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    flat = template.eq(
        "FLAT"
    )

    # Stage2 semantics:
    # - FLAT has no normalized shape by construction.
    # - For modeling/reconstruction its normalized residual shape is
    #   exactly the zero vector.
    if flat.any():
        out.loc[
            flat,
            SHAPE_COLS,
        ] = 0.0

    nonflat = ~flat

    nonflat_ok = out.loc[
        nonflat,
        SHAPE_COLS,
    ].notna().all(
        axis=1
    )

    if not nonflat_ok.all():
        bad_index = nonflat_ok.index[
            ~nonflat_ok
        ]

        examples = (
            out.loc[
                bad_index,
                "sample_id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{len(bad_index):,} non-FLAT rows have incomplete Stage2 shape. "
            f"Examples: {examples}"
        )

    flat_count = int(
        flat.sum()
    )

    out = out.drop(
        columns=[
            "__shape_join_order",
            "__stage2_shape_row_found",
        ]
    )

    out.attrs[
        "stage2_flat_shape_zero_filled"
    ] = flat_count

    return out


def build_features(schema: pd.DataFrame):
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

    features = (
        groups["Z_base"]
        + groups["Z_tr"]
        + groups["M"]
        + groups["U"]
    )

    leaked = [
        c for c in features
        if c in set(H)
    ]

    if leaked:
        raise RuntimeError(
            f"participant_history leakage detected: {leaked}"
        )

    return features, groups


def prepare_X(
    d: pd.DataFrame,
    features: list[str],
    template=None,
):
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
            t = pd.Series(
                template,
                index=d.index,
                dtype="string",
            )
            t = norm(t)

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


def fit_preprocess(
    train: pd.DataFrame,
    features: list[str],
):
    xdf = prepare_X(
        train,
        features,
    )

    imp = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X = imp.fit_transform(
        xdf
    ).astype(
        np.float32,
        copy=False,
    )

    scaler = StandardScaler()

    Xs = scaler.fit_transform(
        X
    ).astype(
        np.float32,
        copy=False,
    )

    return imp, scaler, X, Xs


def uses_scaled_x(candidate):
    return candidate == "Ridge"


def make_model(candidate, args):
    if candidate == "Ridge":
        return Ridge(
            alpha=args.ridge_alpha,
            solver="lsqr",
        )

    if candidate == "HGB_squared":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=args.hgb_lr,
                max_iter=args.hgb_iter,
                max_leaf_nodes=args.hgb_leaves,
                min_samples_leaf=args.hgb_leaf,
                l2_regularization=args.hgb_l2,
                random_state=args.seed,
            ),
            n_jobs=1,
        )

    if candidate == "HGB_absolute":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=args.hgb_lr,
                max_iter=args.hgb_iter,
                max_leaf_nodes=args.hgb_leaves,
                min_samples_leaf=args.hgb_leaf,
                l2_regularization=args.hgb_l2,
                random_state=args.seed,
            ),
            n_jobs=1,
        )

    if candidate == "HGB_weighted":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=args.hgb_lr,
                max_iter=args.hgb_iter,
                max_leaf_nodes=args.hgb_leaves,
                min_samples_leaf=args.hgb_leaf,
                l2_regularization=args.hgb_l2,
                random_state=args.seed,
            ),
            n_jobs=1,
        )

    if candidate == "RandomForest":
        return RandomForestRegressor(
            n_estimators=args.rf_trees,
            max_depth=args.rf_depth,
            min_samples_leaf=args.rf_leaf,
            max_features=args.rf_max_features,
            n_jobs=-1,
            random_state=args.seed,
        )

    raise KeyError(candidate)


def predict_standardized(
    model,
    candidate,
    X,
    Xs,
):
    Xin = (
        Xs
        if uses_scaled_x(candidate)
        else X
    )

    y = np.asarray(
        model.predict(Xin),
        dtype=float,
    )

    if y.ndim == 1:
        y = y[:, None]

    return y


def empirical_rank_weight(x):
    r = (
        pd.Series(
            np.asarray(x, dtype=float)
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(float)
    )

    w = 1.0 + 2.0 * r * r

    return w / np.mean(w)


# ---------------------------------------------------------------------
# True price labels and PCA residual basis
# ---------------------------------------------------------------------

def true_price_components(
    d: pd.DataFrame,
    centers: dict,
):
    pa = num(
        d["p_anchor"]
    ).to_numpy(float)

    ps = num(
        d["p_span"]
    ).to_numpy(float)

    S = d[
        SHAPE_COLS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)

    tid = norm(
        d["y_template_id"]
    ).to_numpy(object)

    flat = (
        tid == "FLAT"
    )

    S[flat] = 0.0

    C = np.vstack(
        [
            centers[str(t)]
            for t in tid
        ]
    )

    R = S - C

    R[flat] = 0.0

    P = (
        pa[:, None]
        + ps[:, None] * S
    )

    valid = (
        np.isfinite(pa)
        & np.isfinite(ps)
        & np.isfinite(S).all(axis=1)
        & np.isfinite(R).all(axis=1)
        & np.isfinite(P).all(axis=1)
    )

    return (
        pa,
        ps,
        S,
        R,
        P,
        tid,
        flat,
        valid,
    )


def fit_price_basis(
    train,
    centers,
    args,
):
    (
        pa,
        ps,
        S,
        R,
        P,
        tid,
        flat,
        valid,
    ) = true_price_components(
        train,
        centers,
    )

    fit_rows = (
        valid
        & (~flat)
    )

    if fit_rows.sum() < 100:
        raise ValueError(
            "Too few non-flat rows to fit price residual PCA."
        )

    max_comp = min(
        args.price_pca_max,
        R.shape[1],
        int(fit_rows.sum()) - 1,
    )

    if max_comp < 1:
        raise ValueError(
            "Invalid PCA component count."
        )

    pca = PCA(
        n_components=max_comp,
        svd_solver="randomized",
        random_state=args.seed,
    )

    pca.fit(
        R[fit_rows]
    )

    cum = np.cumsum(
        pca.explained_variance_ratio_
    )

    k = int(
        np.searchsorted(
            cum,
            args.price_pca_variance,
        )
        + 1
    )

    k = max(
        args.price_pca_min,
        k,
    )

    k = min(
        k,
        max_comp,
    )

    return {
        "mean": pca.mean_.astype(float),
        "components": pca.components_[:k].astype(float),
        "explained_variance_ratio": (
            pca.explained_variance_ratio_[:k].astype(float)
        ),
        "explained_variance_cumulative": float(
            cum[k - 1]
        ),
        "n_components": int(k),
    }


def p_anchor_baseline(
    d,
    state,
):
    if (
        state["anchor_mode"]
        == "absolute"
    ):
        return np.zeros(
            len(d),
            dtype=float,
        )

    x = num(
        d["lt_bid_level"]
    ).to_numpy(float)

    fill = float(
        state["baseline_fill"]
    )

    return np.where(
        np.isfinite(x),
        x,
        fill,
    )


def build_price_targets(
    d,
    centers,
    basis,
    anchor_mode,
    baseline_fill,
):
    (
        pa,
        ps,
        S,
        R,
        P,
        tid,
        flat,
        valid,
    ) = true_price_components(
        d,
        centers,
    )

    temp_state = {
        "anchor_mode": anchor_mode,
        "baseline_fill": baseline_fill,
    }

    anchor_component = (
        pa
        - p_anchor_baseline(
            d,
            temp_state,
        )
    )

    coeff = (
        R - basis["mean"][None, :]
    ) @ basis["components"].T

    coeff[flat] = 0.0

    Y = np.c_[
        anchor_component,
        signed_log1p(ps),
        coeff,
    ]

    return Y, P, S, tid, flat, valid


def fit_price_target_state(
    train,
    centers,
    basis,
    anchor_mode,
    clip_quantile,
):
    if (
        anchor_mode
        == "residual_lt_bid_level"
    ):
        z = num(
            train["lt_bid_level"]
        )
        v = z[
            np.isfinite(z)
        ]

        if v.empty:
            raise ValueError(
                "lt_bid_level has no finite train values."
            )

        baseline_fill = float(
            v.median()
        )

    elif anchor_mode == "absolute":
        baseline_fill = 0.0

    else:
        raise KeyError(anchor_mode)

    Y, _, _, _, _, valid = (
        build_price_targets(
            train,
            centers,
            basis,
            anchor_mode,
            baseline_fill,
        )
    )

    if not valid.all():
        raise ValueError(
            "Frozen train sample contains invalid price targets."
        )

    mean = Y.mean(axis=0)

    std = Y.std(axis=0)

    std = np.where(
        std > 1e-10,
        std,
        1.0,
    )

    q = clip_quantile

    lo = np.quantile(
        Y,
        q,
        axis=0,
    )

    hi = np.quantile(
        Y,
        1.0 - q,
        axis=0,
    )

    Z = (
        Y - mean
    ) / std

    return {
        "anchor_mode": anchor_mode,
        "baseline_fill": baseline_fill,
        "target_mean": mean,
        "target_std": std,
        "clip_lo": lo,
        "clip_hi": hi,
        "Z_train": Z,
    }


def reconstruct_price_from_target(
    d,
    template,
    centers,
    basis,
    state,
    standardized_pred,
):
    Y = (
        np.asarray(
            standardized_pred,
            dtype=float,
        )
        * state["target_std"][None, :]
        + state["target_mean"][None, :]
    )

    Y = np.clip(
        Y,
        state["clip_lo"][None, :],
        state["clip_hi"][None, :],
    )

    anchor = (
        Y[:, 0]
        + p_anchor_baseline(
            d,
            state,
        )
    )

    span = signed_expm1(
        Y[:, 1]
    )

    coeff = Y[:, 2:]

    residual = (
        basis["mean"][None, :]
        + coeff @ basis["components"]
    )

    if isinstance(template, pd.Series):
        tid = norm(template).to_numpy(object)
    else:
        tid = np.asarray(
            template,
            dtype=object,
        )

    C = np.vstack(
        [
            centers[str(t)]
            for t in tid
        ]
    )

    shape = (
        C + residual
    )

    flat = (
        tid == "FLAT"
    )

    shape[flat] = 0.0
    span[flat] = 0.0

    nonflat = ~flat

    if nonflat.any():
        shape[
            nonflat,
            0,
        ] = 0.0

        shape[
            nonflat,
            -1,
        ] = 1.0

    price = (
        anchor[:, None]
        + span[:, None] * shape
    )

    return {
        "p_anchor": anchor,
        "p_span": span,
        "shape": shape,
        "price": price,
    }


def price_metrics(
    true_price,
    pred_price,
    true_shape,
    pred_shape,
):
    err = (
        pred_price
        - true_price
    )

    ae = np.abs(err)

    mae = float(
        np.mean(ae)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                err * err
            )
        )
    )

    den = float(
        np.sum(
            np.abs(true_price)
        )
    )

    wape = (
        100.0
        * float(np.sum(ae))
        / max(den, 1e-12)
    )

    sden = (
        np.abs(true_price)
        + np.abs(pred_price)
    )

    valid = (
        sden > 1e-8
    )

    smape = (
        100.0
        * float(
            np.mean(
                2.0
                * ae[valid]
                / sden[valid]
            )
        )
        if valid.any()
        else np.nan
    )

    se = (
        pred_shape
        - true_shape
    )

    shape_mae = float(
        np.mean(
            np.abs(se)
        )
    )

    shape_rmse = float(
        np.sqrt(
            np.mean(
                se * se
            )
        )
    )

    return {
        "price_mae": mae,
        "price_rmse": rmse,
        "price_wape_pct": wape,
        "price_smape_pct": smape,
        "shape_mae": shape_mae,
        "shape_rmse": shape_rmse,
    }


def fit_price_candidate(
    candidate,
    X,
    Xs,
    state,
    train_curve_scale,
    args,
):
    model = make_model(
        candidate,
        args,
    )

    Xin = (
        Xs
        if uses_scaled_x(candidate)
        else X
    )

    kwargs = {}

    if candidate == "HGB_weighted":
        kwargs["sample_weight"] = (
            empirical_rank_weight(
                train_curve_scale
            )
        )

    model.fit(
        Xin,
        state["Z_train"],
        **kwargs,
    )

    return model


def evaluate_price_candidate(
    paths,
    candidate,
    model,
    features,
    imputer,
    scaler,
    centers,
    basis,
    state,
    sample_files,
):
    rows = 0

    abs_sum = 0.0
    sq_sum = 0.0
    true_abs_sum = 0.0
    smape_sum = 0.0
    smape_count = 0

    shape_abs = 0.0
    shape_sq = 0.0
    shape_count = 0

    for p in paths:
        d = pd.read_pickle(p)

        if d.empty:
            continue

        d = attach_stage2_shapes(
            d,
            sample_files,
        )

        X = imputer.transform(
            prepare_X(
                d,
                features,
            )
        ).astype(
            np.float32,
            copy=False,
        )

        Xs = scaler.transform(
            X
        ).astype(
            np.float32,
            copy=False,
        )

        z = predict_standardized(
            model,
            candidate,
            X,
            Xs,
        )

        pred = reconstruct_price_from_target(
            d,
            norm(d["y_template_id"]),
            centers,
            basis,
            state,
            z,
        )

        (
            pa,
            ps,
            S,
            R,
            P,
            tid,
            flat,
            valid,
        ) = true_price_components(
            d,
            centers,
        )

        if not valid.all():
            keep = valid
            P = P[keep]
            S = S[keep]
            pp = pred["price"][keep]
            ss = pred["shape"][keep]
        else:
            pp = pred["price"]
            ss = pred["shape"]

        e = (
            pp - P
        )

        ae = np.abs(e)

        abs_sum += float(
            np.sum(ae)
        )

        sq_sum += float(
            np.sum(
                e * e
            )
        )

        true_abs_sum += float(
            np.sum(
                np.abs(P)
            )
        )

        den = (
            np.abs(P)
            + np.abs(pp)
        )

        sm_valid = (
            den > 1e-8
        )

        if sm_valid.any():
            smape_sum += float(
                np.sum(
                    2.0
                    * ae[sm_valid]
                    / den[sm_valid]
                )
            )
            smape_count += int(
                sm_valid.sum()
            )

        se = (
            ss - S
        )

        shape_abs += float(
            np.sum(
                np.abs(se)
            )
        )

        shape_sq += float(
            np.sum(
                se * se
            )
        )

        shape_count += int(
            se.size
        )

        rows += len(P)

        del d, X, Xs, z, pred, P, S
        gc.collect()

    points = rows * 21

    return {
        "rows": rows,
        "price_mae": (
            abs_sum / points
        ),
        "price_rmse": (
            np.sqrt(
                sq_sum / points
            )
        ),
        "price_wape_pct": (
            100.0
            * abs_sum
            / max(
                true_abs_sum,
                1e-12,
            )
        ),
        "price_smape_pct": (
            100.0
            * smape_sum
            / smape_count
            if smape_count
            else np.nan
        ),
        "shape_mae": (
            shape_abs
            / shape_count
        ),
        "shape_rmse": (
            np.sqrt(
                shape_sq
                / shape_count
            )
        ),
    }


# ---------------------------------------------------------------------
# Quantity targets
# ---------------------------------------------------------------------

def fit_q_spec(
    train,
    q_scale_mode,
):
    spec = {
        "q_scale_mode": q_scale_mode,
        "q_scale_column": None,
        "q_scale_fill": 1.0,
        "q_scale_floor": 1.0,
    }

    if q_scale_mode != "absolute":
        s = num(
            train[q_scale_mode]
        )

        vals = s[
            np.isfinite(s)
            & (s > 1.0)
        ]

        if vals.empty:
            raise ValueError(
                f"{q_scale_mode} has no positive train values."
            )

        spec[
            "q_scale_column"
        ] = q_scale_mode

        spec[
            "q_scale_fill"
        ] = float(
            vals.median()
        )

    return spec


def q_scale_value(
    d,
    spec,
):
    col = spec[
        "q_scale_column"
    ]

    if col is None:
        return np.ones(
            len(d),
            dtype=float,
        )

    x = num(
        d[col]
    ).to_numpy(float)

    fill = float(
        spec[
            "q_scale_fill"
        ]
    )

    floor = float(
        spec[
            "q_scale_floor"
        ]
    )

    x = np.where(
        np.isfinite(x)
        & (x > floor),
        x,
        fill,
    )

    return np.maximum(
        x,
        floor,
    )


def q_latent(
    d,
    spec,
):
    scale = q_scale_value(
        d,
        spec,
    )

    qb = (
        num(
            d["theta_q_base_mw"]
        ).to_numpy(float)
        / scale
    )

    qs = (
        np.maximum(
            num(
                d["theta_q_span_mw"]
            ).to_numpy(float),
            1e-8,
        )
        / scale
    )

    Q = d[
        [
            "theta_q1",
            "theta_q2",
            "theta_q3",
            "theta_q4",
            "theta_q5",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)

    Q = np.maximum(
        Q,
        1e-8,
    )

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


def fit_q_state(
    train,
    spec,
    clip_quantile,
):
    Y = q_latent(
        train,
        spec,
    )

    mean = Y.mean(
        axis=0
    )

    std = Y.std(
        axis=0
    )

    std = np.where(
        std > 1e-10,
        std,
        1.0,
    )

    q = clip_quantile

    lo = np.quantile(
        Y,
        q,
        axis=0,
    )

    hi = np.quantile(
        Y,
        1.0 - q,
        axis=0,
    )

    Z = (
        Y - mean
    ) / std

    return {
        "mean": mean,
        "std": std,
        "clip_lo": lo,
        "clip_hi": hi,
        "Z_train": Z,
    }


def inverse_q_latent(
    d,
    spec,
    state,
    standardized,
):
    Y = (
        np.asarray(
            standardized,
            dtype=float,
        )
        * state["std"][None, :]
        + state["mean"][None, :]
    )

    Y = np.clip(
        Y,
        state["clip_lo"][None, :],
        state["clip_hi"][None, :],
    )

    scale = q_scale_value(
        d,
        spec,
    )

    qb = (
        signed_expm1(
            Y[:, 0]
        )
        * scale
    )

    qs = (
        np.maximum(
            np.expm1(
                Y[:, 1]
            ),
            1e-8,
        )
        * scale
    )

    L = np.c_[
        Y[:, 2:6],
        np.zeros(
            len(Y)
        ),
    ]

    L -= L.max(
        axis=1,
        keepdims=True,
    )

    E = np.exp(L)

    Q = (
        E
        / E.sum(
            axis=1,
            keepdims=True,
        )
    )

    return qb, qs, Q


def fit_q_candidate(
    candidate,
    X,
    Xs,
    Z,
    args,
):
    model = make_model(
        candidate,
        args,
    )

    Xin = (
        Xs
        if uses_scaled_x(candidate)
        else X
    )

    model.fit(
        Xin,
        Z,
    )

    return model


def evaluate_q_scale_candidate(
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

    qa_abs = 0.0
    qs_abs = 0.0
    qm_abs = 0.0

    qa_true_abs = 0.0
    qs_true_abs = 0.0
    qm_true_abs = 0.0

    for p in paths:
        d = pd.read_pickle(p)

        if d.empty:
            continue

        X = imputer.transform(
            prepare_X(
                d,
                features,
            )
        ).astype(
            np.float32,
            copy=False,
        )

        Xs = scaler.transform(
            X
        ).astype(
            np.float32,
            copy=False,
        )

        z = predict_standardized(
            model,
            candidate,
            X,
            Xs,
        )

        # q-scale candidate predicts only first two q latent dimensions.
        full_z = np.zeros(
            (
                len(d),
                6,
            ),
            dtype=float,
        )

        full_z[
            :,
            :2,
        ] = z

        # True q-shape logits keep q-shape neutral for scale evaluation.
        true_lat = q_latent(
            d,
            spec,
        )

        full_z[
            :,
            2:6,
        ] = (
            true_lat[
                :,
                2:6,
            ]
            - state[
                "mean"
            ][
                None,
                2:6,
            ]
        ) / state[
            "std"
        ][
            None,
            2:6,
        ]

        qb_p, qspan_p, Qp = inverse_q_latent(
            d,
            spec,
            state,
            full_z,
        )

        qb_t = num(
            d["theta_q_base_mw"]
        ).to_numpy(float)

        qspan_t = num(
            d["theta_q_span_mw"]
        ).to_numpy(float)

        qmax_t = (
            qb_t + qspan_t
        )

        qmax_p = (
            qb_p + qspan_p
        )

        qa_abs += float(
            np.sum(
                np.abs(
                    qb_p - qb_t
                )
            )
        )

        qs_abs += float(
            np.sum(
                np.abs(
                    qspan_p - qspan_t
                )
            )
        )

        qm_abs += float(
            np.sum(
                np.abs(
                    qmax_p - qmax_t
                )
            )
        )

        qa_true_abs += float(
            np.sum(
                np.abs(qb_t)
            )
        )

        qs_true_abs += float(
            np.sum(
                np.abs(qspan_t)
            )
        )

        qm_true_abs += float(
            np.sum(
                np.abs(qmax_t)
            )
        )

        rows += len(d)

        del d, X, Xs, z
        gc.collect()

    qa_wape = (
        100.0
        * qa_abs
        / max(
            qa_true_abs,
            1e-12,
        )
    )

    qs_wape = (
        100.0
        * qs_abs
        / max(
            qs_true_abs,
            1e-12,
        )
    )

    qm_wape = (
        100.0
        * qm_abs
        / max(
            qm_true_abs,
            1e-12,
        )
    )

    return {
        "rows": rows,
        "q_anchor_mae_mw": (
            qa_abs / rows
        ),
        "q_span_mae_mw": (
            qs_abs / rows
        ),
        "q_max_mae_mw": (
            qm_abs / rows
        ),
        "q_anchor_wape_pct": qa_wape,
        "q_span_wape_pct": qs_wape,
        "q_max_wape_pct": qm_wape,
        "selection_score": float(
            np.mean(
                [
                    qa_wape,
                    qs_wape,
                    qm_wape,
                ]
            )
        ),
    }


def evaluate_q_shape_candidate(
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

    qshare_abs = 0.0
    break_abs = 0.0

    for p in paths:
        d = pd.read_pickle(p)

        if d.empty:
            continue

        X = imputer.transform(
            prepare_X(
                d,
                features,
            )
        ).astype(
            np.float32,
            copy=False,
        )

        Xs = scaler.transform(
            X
        ).astype(
            np.float32,
            copy=False,
        )

        z = predict_standardized(
            model,
            candidate,
            X,
            Xs,
        )

        Y = (
            z
            * state["std"][
                None,
                2:6,
            ]
            + state["mean"][
                None,
                2:6,
            ]
        )

        Y = np.clip(
            Y,
            state["clip_lo"][
                None,
                2:6,
            ],
            state["clip_hi"][
                None,
                2:6,
            ],
        )

        L = np.c_[
            Y,
            np.zeros(
                len(Y)
            ),
        ]

        L -= L.max(
            axis=1,
            keepdims=True,
        )

        E = np.exp(L)

        Qp = (
            E
            / E.sum(
                axis=1,
                keepdims=True,
            )
        )

        Qt = d[
            [
                "theta_q1",
                "theta_q2",
                "theta_q3",
                "theta_q4",
                "theta_q5",
            ]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(float)

        Qt = np.maximum(
            Qt,
            1e-8,
        )

        Qt /= Qt.sum(
            axis=1,
            keepdims=True,
        )

        qshare_abs += float(
            np.sum(
                np.abs(
                    Qp - Qt
                )
            )
        )

        bp = np.cumsum(
            Qp,
            axis=1,
        )[
            :,
            :4,
        ]

        bt = np.cumsum(
            Qt,
            axis=1,
        )[
            :,
            :4,
        ]

        break_abs += float(
            np.sum(
                np.abs(
                    bp - bt
                )
            )
        )

        rows += len(d)

        del d, X, Xs, z
        gc.collect()

    qshare_mae = (
        qshare_abs
        / (rows * 5)
    )

    break_mae = (
        break_abs
        / (rows * 4)
    )

    return {
        "rows": rows,
        "q_share_mae": qshare_mae,
        "breakpoint_fraction_mae": break_mae,
        "selection_score": float(
            0.5
            * (
                qshare_mae
                + break_mae
            )
        ),
    }


# ---------------------------------------------------------------------
# Combined bundle evaluation
# ---------------------------------------------------------------------

def predict_bundle(
    bundle,
    d,
    template,
):
    X = bundle[
        "imputer"
    ].transform(
        prepare_X(
            d,
            bundle["features"],
            template,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    Xs = bundle[
        "x_scaler"
    ].transform(
        X
    ).astype(
        np.float32,
        copy=False,
    )

    price_info = bundle[
        "price_model"
    ]

    zp = predict_standardized(
        price_info["model"],
        price_info["candidate"],
        X,
        Xs,
    )

    price = reconstruct_price_from_target(
        d,
        template,
        bundle["template_centers"],
        bundle["price_basis"],
        bundle["price_target_state"],
        zp,
    )

    qscale_info = bundle[
        "q_scale_model"
    ]

    zqs = predict_standardized(
        qscale_info["model"],
        qscale_info["candidate"],
        X,
        Xs,
    )

    qshape_info = bundle[
        "q_shape_model"
    ]

    zqh = predict_standardized(
        qshape_info["model"],
        qshape_info["candidate"],
        X,
        Xs,
    )

    full_z = np.zeros(
        (
            len(d),
            6,
        ),
        dtype=float,
    )

    full_z[
        :,
        :2,
    ] = zqs

    full_z[
        :,
        2:6,
    ] = zqh

    qb, qspan, Q = inverse_q_latent(
        d,
        bundle["q_spec"],
        bundle["q_state"],
        full_z,
    )

    return {
        "price": price,
        "q_base": qb,
        "q_span": qspan,
        "q_share": Q,
    }


def evaluate_combined(
    paths,
    bundle,
    sample_files,
):
    rows = 0

    price_abs = 0.0
    price_sq = 0.0
    price_true_abs = 0.0

    q_anchor_abs = 0.0
    q_span_abs = 0.0
    q_max_abs = 0.0

    q_anchor_true_abs = 0.0
    q_span_true_abs = 0.0
    q_max_true_abs = 0.0

    qshare_abs = 0.0
    break_abs = 0.0

    for p in paths:
        d = pd.read_pickle(p)

        if d.empty:
            continue

        d = attach_stage2_shapes(
            d,
            sample_files,
        )

        pred = predict_bundle(
            bundle,
            d,
            norm(
                d["y_template_id"]
            ),
        )

        (
            pa,
            ps,
            S,
            R,
            P,
            tid,
            flat,
            valid,
        ) = true_price_components(
            d,
            bundle[
                "template_centers"
            ],
        )

        pe = (
            pred["price"]["price"]
            - P
        )

        price_abs += float(
            np.sum(
                np.abs(pe)
            )
        )

        price_sq += float(
            np.sum(
                pe * pe
            )
        )

        price_true_abs += float(
            np.sum(
                np.abs(P)
            )
        )

        qb_t = num(
            d["theta_q_base_mw"]
        ).to_numpy(float)

        qs_t = num(
            d["theta_q_span_mw"]
        ).to_numpy(float)

        qm_t = (
            qb_t + qs_t
        )

        qb_p = pred[
            "q_base"
        ]

        qs_p = pred[
            "q_span"
        ]

        qm_p = (
            qb_p + qs_p
        )

        q_anchor_abs += float(
            np.sum(
                np.abs(
                    qb_p - qb_t
                )
            )
        )

        q_span_abs += float(
            np.sum(
                np.abs(
                    qs_p - qs_t
                )
            )
        )

        q_max_abs += float(
            np.sum(
                np.abs(
                    qm_p - qm_t
                )
            )
        )

        q_anchor_true_abs += float(
            np.sum(
                np.abs(qb_t)
            )
        )

        q_span_true_abs += float(
            np.sum(
                np.abs(qs_t)
            )
        )

        q_max_true_abs += float(
            np.sum(
                np.abs(qm_t)
            )
        )

        Qt = d[
            [
                "theta_q1",
                "theta_q2",
                "theta_q3",
                "theta_q4",
                "theta_q5",
            ]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(float)

        Qt = np.maximum(
            Qt,
            1e-8,
        )

        Qt /= Qt.sum(
            axis=1,
            keepdims=True,
        )

        Qp = pred[
            "q_share"
        ]

        qshare_abs += float(
            np.sum(
                np.abs(
                    Qp - Qt
                )
            )
        )

        break_abs += float(
            np.sum(
                np.abs(
                    np.cumsum(
                        Qp,
                        axis=1,
                    )[
                        :,
                        :4,
                    ]
                    - np.cumsum(
                        Qt,
                        axis=1,
                    )[
                        :,
                        :4,
                    ]
                )
            )
        )

        rows += len(d)

        del d, pred
        gc.collect()

    return {
        "rows": rows,
        "price_mae": (
            price_abs
            / (rows * 21)
        ),
        "price_rmse": (
            np.sqrt(
                price_sq
                / (rows * 21)
            )
        ),
        "price_wape_pct": (
            100.0
            * price_abs
            / max(
                price_true_abs,
                1e-12,
            )
        ),
        "q_anchor_mae_mw": (
            q_anchor_abs / rows
        ),
        "q_span_mae_mw": (
            q_span_abs / rows
        ),
        "q_max_mae_mw": (
            q_max_abs / rows
        ),
        "q_anchor_wape_pct": (
            100.0
            * q_anchor_abs
            / max(
                q_anchor_true_abs,
                1e-12,
            )
        ),
        "q_span_wape_pct": (
            100.0
            * q_span_abs
            / max(
                q_span_true_abs,
                1e-12,
            )
        ),
        "q_max_wape_pct": (
            100.0
            * q_max_abs
            / max(
                q_max_true_abs,
                1e-12,
            )
        ),
        "q_share_mae": (
            qshare_abs
            / (rows * 5)
        ),
        "breakpoint_fraction_mae": (
            break_abs
            / (rows * 4)
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )

    ap.add_argument(
        "--bidtemplate-root",
        default="data/processed/bidtemplate",
    )

    ap.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
    )

    ap.add_argument(
        "--hgb-lr",
        type=float,
        default=0.06,
    )

    ap.add_argument(
        "--hgb-iter",
        type=int,
        default=120,
    )

    ap.add_argument(
        "--hgb-leaves",
        type=int,
        default=15,
    )

    ap.add_argument(
        "--hgb-leaf",
        type=int,
        default=80,
    )

    ap.add_argument(
        "--hgb-l2",
        type=float,
        default=2.0,
    )

    ap.add_argument(
        "--rf-trees",
        type=int,
        default=96,
    )

    ap.add_argument(
        "--rf-depth",
        type=int,
        default=18,
    )

    ap.add_argument(
        "--rf-leaf",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--rf-max-features",
        type=float,
        default=0.7,
    )

    ap.add_argument(
        "--price-pca-variance",
        type=float,
        default=0.95,
    )

    ap.add_argument(
        "--price-pca-min",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--price-pca-max",
        type=int,
        default=6,
    )

    ap.add_argument(
        "--target-clip-quantile",
        type=float,
        default=0.001,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    base = (
        Path(args.root)
        / str(args.year)
    )

    frozen = (
        base
        / "frozen_modeling_dataset"
    )

    manifest = load_manifest(
        frozen
    )

    schema = pd.read_csv(
        frozen
        / "feature_schema.csv"
    )

    features, groups = build_features(
        schema
    )

    train_file = (
        frozen
        / manifest[
            "train_sample_file"
        ]
    )

    train = pd.read_pickle(
        train_file
    )

    val_parts = [
        frozen / p
        for p in manifest[
            "parts"
        ][
            "val"
        ]
    ]

    test_parts = [
        frozen / p
        for p in manifest[
            "parts"
        ][
            "test_curve"
        ]
    ]

    bidtemplate_year = (
        Path(args.bidtemplate_root)
        / str(args.year)
    )

    centers, center_file = discover_centers(
        bidtemplate_year
    )

    sample_files = discover_curve_samples(
        bidtemplate_year
    )

    imputer, scaler, X, Xs = fit_preprocess(
        train,
        features,
    )

    print("=" * 80)
    print(
        "Template parameter regression - PCA residual price representation"
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
    print(
        f"Template centers: {center_file}"
    )
    print()

    # ---------------------------------------------------------------
    # 1) Restore Stage2 price shapes for the FIXED frozen train sample,
    #    then learn the price residual basis from TRAIN only.
    # ---------------------------------------------------------------

    print(
        "Joining Stage2 shapes to frozen train sample...",
        flush=True,
    )

    train_price = attach_stage2_shapes(
        train,
        sample_files,
    )

    flat_zero_filled = int(
        (
            train_price[
                "y_template_id"
            ]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq("FLAT")
        ).sum()
    )

    print(
        f"Stage2 shape join complete: {len(train_price):,}/{len(train):,}",
        flush=True,
    )
    print(
        f"FLAT rows represented by zero normalized shape: "
        f"{flat_zero_filled:,}",
        flush=True,
    )

    basis = fit_price_basis(
        train_price,
        centers,
        args,
    )

    print(
        "Price representation:"
    )
    print(
        "  P(x) = p_anchor + p_span * "
        "[template_center + PCA residual]"
    )
    print(
        f"  PCA components = {basis['n_components']}"
    )
    print(
        "  cumulative explained variance = "
        f"{basis['explained_variance_cumulative']:.4%}"
    )
    print()

    (
        pa_t,
        ps_t,
        S_t,
        R_t,
        P_t,
        tid_t,
        flat_t,
        valid_t,
    ) = true_price_components(
        train_price,
        centers,
    )

    train_curve_scale = np.mean(
        np.abs(P_t),
        axis=1,
    )

    # ---------------------------------------------------------------
    # 2) Price model + anchor transform search.
    # ---------------------------------------------------------------

    price_rows = []
    fitted_price = {}

    for anchor_mode in P_ANCHOR_MODES:
        state = fit_price_target_state(
            train_price,
            centers,
            basis,
            anchor_mode,
            args.target_clip_quantile,
        )

        for candidate in PRICE_CANDIDATES:
            print(
                f"[price] anchor={anchor_mode} / model={candidate}",
                flush=True,
            )

            model = fit_price_candidate(
                candidate,
                X,
                Xs,
                state,
                train_curve_scale,
                args,
            )

            metrics = evaluate_price_candidate(
                val_parts,
                candidate,
                model,
                features,
                imputer,
                scaler,
                centers,
                basis,
                state,
                sample_files,
            )

            price_rows.append(
                {
                    "anchor_mode": anchor_mode,
                    "candidate": candidate,
                    **metrics,
                }
            )

            fitted_price[
                (
                    anchor_mode,
                    candidate,
                )
            ] = (
                state,
                model,
            )

    price_df = pd.DataFrame(
        price_rows
    ).sort_values(
        [
            "price_wape_pct",
            "shape_mae",
            "price_mae",
        ]
    ).reset_index(
        drop=True
    )

    best_price = price_df.iloc[0]

    best_anchor_mode = str(
        best_price[
            "anchor_mode"
        ]
    )

    best_price_candidate = str(
        best_price[
            "candidate"
        ]
    )

    price_state, price_model = (
        fitted_price[
            (
                best_anchor_mode,
                best_price_candidate,
            )
        ]
    )

    # ---------------------------------------------------------------
    # 3) q-scale model + transform search.
    # ---------------------------------------------------------------

    qscale_rows = []
    fitted_qscale = {}

    for q_mode in Q_SCALE_MODES:
        spec = fit_q_spec(
            train,
            q_mode,
        )

        state = fit_q_state(
            train,
            spec,
            args.target_clip_quantile,
        )

        Z_scale = state[
            "Z_train"
        ][
            :,
            :2,
        ]

        for candidate in Q_CANDIDATES:
            print(
                f"[q_scale] mode={q_mode} / model={candidate}",
                flush=True,
            )

            model = fit_q_candidate(
                candidate,
                X,
                Xs,
                Z_scale,
                args,
            )

            metrics = evaluate_q_scale_candidate(
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
                    "q_scale_mode": q_mode,
                    "candidate": candidate,
                    **metrics,
                }
            )

            fitted_qscale[
                (
                    q_mode,
                    candidate,
                )
            ] = (
                spec,
                state,
                model,
            )

    qscale_df = pd.DataFrame(
        qscale_rows
    ).sort_values(
        "selection_score"
    ).reset_index(
        drop=True
    )

    best_qs = qscale_df.iloc[0]

    best_q_mode = str(
        best_qs[
            "q_scale_mode"
        ]
    )

    best_qscale_candidate = str(
        best_qs[
            "candidate"
        ]
    )

    q_spec, q_state, q_scale_model = (
        fitted_qscale[
            (
                best_q_mode,
                best_qscale_candidate,
            )
        ]
    )

    # ---------------------------------------------------------------
    # 4) q-shape model.
    # ---------------------------------------------------------------

    Z_shape = q_state[
        "Z_train"
    ][
        :,
        2:6,
    ]

    qshape_rows = []
    fitted_qshape = {}

    for candidate in Q_CANDIDATES:
        print(
            f"[q_shape] model={candidate}",
            flush=True,
        )

        model = fit_q_candidate(
            candidate,
            X,
            Xs,
            Z_shape,
            args,
        )

        metrics = evaluate_q_shape_candidate(
            val_parts,
            candidate,
            model,
            features,
            imputer,
            scaler,
            q_spec,
            q_state,
        )

        qshape_rows.append(
            {
                "candidate": candidate,
                **metrics,
            }
        )

        fitted_qshape[
            candidate
        ] = model

    qshape_df = pd.DataFrame(
        qshape_rows
    ).sort_values(
        "selection_score"
    ).reset_index(
        drop=True
    )

    best_qshape_candidate = str(
        qshape_df.iloc[0][
            "candidate"
        ]
    )

    q_shape_model = fitted_qshape[
        best_qshape_candidate
    ]

    # ---------------------------------------------------------------
    # Final bundle.
    # ---------------------------------------------------------------

    bundle = {
        "version": "04b-pca-price-v8.2",
        "model_mode": "pca_price_q_specialist",
        "feature_policy": "Z_base + Z_tr + M + U",
        "features": features,
        "history_features_used": False,
        "templates": TEMPLATES,
        "imputer": imputer,
        "x_scaler": scaler,
        "template_centers": centers,
        "price_basis": basis,
        "price_target_state": price_state,
        "price_model": {
            "candidate": best_price_candidate,
            "uses_scaled_x": uses_scaled_x(
                best_price_candidate
            ),
            "model": price_model,
        },
        "q_spec": q_spec,
        "q_state": q_state,
        "q_scale_model": {
            "candidate": best_qscale_candidate,
            "uses_scaled_x": uses_scaled_x(
                best_qscale_candidate
            ),
            "model": q_scale_model,
        },
        "q_shape_model": {
            "candidate": best_qshape_candidate,
            "uses_scaled_x": uses_scaled_x(
                best_qshape_candidate
            ),
            "model": q_shape_model,
        },
    }

    val_summary = evaluate_combined(
        val_parts,
        bundle,
        sample_files,
    )

    test_summary = evaluate_combined(
        test_parts,
        bundle,
        sample_files,
    )

    out = (
        base
        / "template_parameter_models"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mdir = out / "models"

    mdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for stale in mdir.glob(
        "*.joblib"
    ):
        stale.unlink()

    feature_set_name = (
        "Z_base_Z_tr_M_U"
    )

    joblib.dump(
        bundle,
        mdir
        / f"{feature_set_name}.joblib",
        compress=3,
    )

    price_df.to_csv(
        out
        / "price_model_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qscale_df.to_csv(
        out
        / "q_scale_model_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qshape_df.to_csv(
        out
        / "q_shape_model_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "selected_feature_set": feature_set_name,
                "selected_model": "PCAResidualSpecialist",
                "feature_policy": (
                    "Z_base + Z_tr + M + U (fixed)"
                ),
                "participant_history_used": False,
                "price_representation": (
                    "p_anchor + p_span * "
                    "(template_center + PCA residual)"
                ),
                "price_anchor_mode": best_anchor_mode,
                "price_model": best_price_candidate,
                "price_pca_components": basis[
                    "n_components"
                ],
                "price_pca_explained_variance": basis[
                    "explained_variance_cumulative"
                ],
                "q_scale_mode": best_q_mode,
                "q_scale_model": best_qscale_candidate,
                "q_shape_model": best_qshape_candidate,
                "selected_val_score": val_summary[
                    "price_wape_pct"
                ],
                "selected_test_score": test_summary[
                    "price_wape_pct"
                ],
            }
        ]
    ).to_csv(
        out
        / "selected_template_parameter_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "split": "val",
                **val_summary,
            },
            {
                "split": "test",
                **test_summary,
            },
        ]
    ).to_csv(
        out
        / "parameter_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cfg = {
        "version": "04b-pca-price-v8.2",
        "year": args.year,
        "feature_policy": {
            "Z_base": len(
                groups["Z_base"]
            ),
            "Z_tr": len(
                groups["Z_tr"]
            ),
            "M": len(
                groups["M"]
            ),
            "U": len(
                groups["U"]
            ),
            "H_excluded": len(
                groups["H_excluded"]
            ),
            "final": (
                "Z_base + Z_tr + M + U"
            ),
        },
        "price_representation": {
            "anchor_modes": P_ANCHOR_MODES,
            "model_candidates": PRICE_CANDIDATES,
            "pca_variance_target": args.price_pca_variance,
            "pca_min": args.price_pca_min,
            "pca_max": args.price_pca_max,
            "selected_components": basis[
                "n_components"
            ],
            "selected_explained_variance": basis[
                "explained_variance_cumulative"
            ],
        },
        "quantity_representation": {
            "q_scale_modes": Q_SCALE_MODES,
            "model_candidates": Q_CANDIDATES,
            "q_shape": "5-simplex ALR logits",
        },
        "generalization_policy": {
            "dataset_specific_price_threshold": False,
            "participant_specific_rule": False,
            "template_specific_exception": False,
            "price_basis_source": (
                "Stage2 curve_samples joined to the fixed frozen "
                "training sample by sample_id; PCA fitted on train only"
            ),
        },
    }

    (
        out
        / "config.json"
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
            f"PCA residual price representation - {args.year}"
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
        (
            "Price-shape labels: Stage2 curve_samples joined "
            "strictly by sample_id"
        ),
        "",
        "Price representation:",
        (
            "  P(x) = p_anchor + p_span * "
            "[template_center(x) + residual_PCA(x)]"
        ),
        (
            f"  PCA components = "
            f"{basis['n_components']}"
        ),
        (
            "  cumulative explained variance = "
            f"{basis['explained_variance_cumulative']:.4%}"
        ),
        (
            f"  p_anchor mode = "
            f"{best_anchor_mode}"
        ),
        (
            f"  price model = "
            f"{best_price_candidate}"
        ),
        "",
        "Quantity representation:",
        (
            f"  q scale mode = "
            f"{best_q_mode}"
        ),
        (
            f"  q scale model = "
            f"{best_qscale_candidate}"
        ),
        (
            f"  q shape model = "
            f"{best_qshape_candidate}"
        ),
        "",
        "Combined validation:",
        (
            f"  price MAE="
            f"{val_summary['price_mae']:.6f}"
        ),
        (
            f"  price WAPE="
            f"{val_summary['price_wape_pct']:.4f}%"
        ),
        (
            f"  q_anchor MAE="
            f"{val_summary['q_anchor_mae_mw']:.6f} MW"
        ),
        (
            f"  q_span MAE="
            f"{val_summary['q_span_mae_mw']:.6f} MW"
        ),
        (
            f"  q_max MAE="
            f"{val_summary['q_max_mae_mw']:.6f} MW"
        ),
        (
            f"  q_share MAE="
            f"{val_summary['q_share_mae']:.6f}"
        ),
        "",
        "Combined TEST:",
        (
            f"  price MAE="
            f"{test_summary['price_mae']:.6f}"
        ),
        (
            f"  price WAPE="
            f"{test_summary['price_wape_pct']:.4f}%"
        ),
        (
            f"  q_anchor MAE="
            f"{test_summary['q_anchor_mae_mw']:.6f} MW"
        ),
        (
            f"  q_span MAE="
            f"{test_summary['q_span_mae_mw']:.6f} MW"
        ),
        (
            f"  q_max MAE="
            f"{test_summary['q_max_mae_mw']:.6f} MW"
        ),
        (
            f"  q_share MAE="
            f"{test_summary['q_share_mae']:.6f}"
        ),
        "",
        (
            "This version changes the price parameterization itself; "
            "it is not a local p_base tuning patch."
        ),
    ]

    summary = "\n".join(
        lines
    )

    (
        out
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(
        f"Outputs: {out}"
    )


if __name__ == "__main__":
    main()
