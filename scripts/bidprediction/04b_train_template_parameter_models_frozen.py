#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04b_train_template_parameter_models.py

Frozen-data version.

This script no longer rebuilds train/val/test or resamples training rows.
It reads only:
data/processed/bidprediction/<year>/frozen_modeling_dataset/

Expected frozen inputs:
    feature_schema.csv
    manifest.json
    train_sample_300000.pkl
    val_parts/*.pkl
    test_parts/*.pkl

Current target parameterization:
    p_base                  -> absolute
    alpha, beta             -> log1p nonnegative
    q_max=q_base+q_span     -> log1p absolute MW
    r_base=q_base/q_max     -> logit
    q1..q5                  -> ALR logits -> softmax

Models:
    LinearRegression
    Ridge
    small HistGradientBoostingRegressor

Feature sets:
    Z_base + H
    Z_base + M + U + H

Run:
python scripts/bidprediction/04b_train_template_parameter_models.py --year 2025
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


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
MODE2I = {"flat": 0, "block": 1, "sloped": 2}

RAW_THETA = [
    "theta_p_base",
    "theta_alpha",
    "theta_beta",
    "theta_q_base_mw",
    "theta_q_span_mw",
    "theta_q1",
    "theta_q2",
    "theta_q3",
    "theta_q4",
    "theta_q5",
]

LATENT = [
    "p_base",
    "log1p_alpha",
    "log1p_beta",
    "log1p_q_max_mw",
    "q_base_fraction_logit",
    "qlogit1",
    "qlogit2",
    "qlogit3",
    "qlogit4",
]

MODELS = [
    "LinearRegression",
    "Ridge",
    "HistGradientBoostingRegressor",
]

MODEL_COMPLEXITY = {
    "LinearRegression": 0,
    "Ridge": 1,
    "HistGradientBoostingRegressor": 2,
}

FEATURE_COMPLEXITY = {
    "Z_base_H": 0,
    "Z_base_M_U_H": 1,
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
    "market_nonmissing_count",
    "rolling_lt_nonmissing_count",
    "hist_prev_available_flag",
    "tr_ready_flag",
}

SELECTION_CORE_TARGETS = [
    "theta_p_base",
    "theta_alpha",
    "theta_beta",
    "theta_q_base_mw",
    "theta_q_span_mw",
]


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def uniq(xs):
    return list(dict.fromkeys(xs))


def load_manifest(frozen: Path):
    p = frozen / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p}\nRun 04a3_freeze_modeling_dataset.py first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def feature_sets(schema: pd.DataFrame):
    s = schema[
        schema["role"].astype(str).str.lower().eq("feature")
    ].copy()

    fmap = dict(
        zip(
            s["column"].astype(str),
            s["feature_group"].astype(str),
        )
    )

    avail = set(fmap) - EXCLUDE

    zb = LT + ST + BR

    miss = [c for c in zb if c not in avail]
    if miss:
        raise KeyError(f"Missing Z_base columns: {miss}")

    ztr = [
        c
        for c, g in fmap.items()
        if g == "transition_strategy_profile"
        and c in avail
    ]

    M = [
        c
        for c, g in fmap.items()
        if g == "market_environment"
        and c in avail
    ]

    U = [
        c
        for c, g in fmap.items()
        if g == "unit_state_proxy"
        and c in avail
    ]

    H = [
        c
        for c, g in fmap.items()
        if g == "participant_history"
        and c in avail
    ]

    fs = {
        "Z_base_H": uniq(zb + H),
        "Z_base_M_U_H": uniq(zb + M + U + H),
    }

    return (
        fs,
        {
            "Z_base": zb,
            "Z_tr_unused": ztr,
            "M": M,
            "U": U,
            "H": H,
        },
        fmap,
    )


def enc_col(s: pd.Series, col: str) -> pd.Series:
    if col in {
        "hist_lag1_template_id",
        "hist30_dominant_template_id",
    }:
        return (
            s.astype("string")
            .str.strip()
            .map(T2I)
            .astype("float32")
        )

    if col == "hist_lag1_curve_mode":
        return (
            s.astype("string")
            .str.strip()
            .str.lower()
            .map(MODE2I)
            .astype("float32")
        )

    return num(s).astype("float32")


def Xframe(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = {
        c: enc_col(d[c], c)
        for c in features
    }

    t = (
        d["y_template_id"]
        .astype("string")
        .str.strip()
        .map(T2I)
    )

    if t.isna().any():
        bad = (
            d.loc[t.isna(), "y_template_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Unknown template condition: {bad[:20]}"
        )

    a = t.to_numpy(np.int16)

    for j, z in enumerate(TEMPLATES):
        out[f"cond_template_{z}"] = (
            a == j
        ).astype("float32")

    return pd.DataFrame(
        out,
        index=d.index,
    )


def raw_to_latent(d: pd.DataFrame) -> np.ndarray:
    p = num(d["theta_p_base"]).to_numpy(float)

    a = np.maximum(
        num(d["theta_alpha"]).to_numpy(float),
        0.0,
    )

    b = np.maximum(
        num(d["theta_beta"]).to_numpy(float),
        0.0,
    )

    qb = np.maximum(
        num(d["theta_q_base_mw"]).to_numpy(float),
        0.0,
    )

    qs = np.maximum(
        num(d["theta_q_span_mw"]).to_numpy(float),
        1e-8,
    )

    qmax = np.maximum(
        qb + qs,
        1e-8,
    )

    rbase = np.clip(
        qb / qmax,
        1e-6,
        1.0 - 1e-6,
    )

    rbase_logit = np.log(
        rbase
        / (1.0 - rbase)
    )

    Q = d[
        [f"theta_q{i}" for i in range(1, 6)]
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
        p,
        np.log1p(a),
        np.log1p(b),
        np.log1p(qmax),
        rbase_logit,
        logits,
    ]


def latent_to_raw(
    Z: np.ndarray,
    clip_lo: np.ndarray,
    clip_hi: np.ndarray,
) -> np.ndarray:
    Z = np.asarray(
        Z,
        dtype=float,
    )

    Z = np.clip(
        Z,
        clip_lo[None, :],
        clip_hi[None, :],
    )

    out = np.empty(
        (len(Z), 10),
        dtype=float,
    )

    out[:, 0] = Z[:, 0]
    out[:, 1] = np.maximum(
        np.expm1(Z[:, 1]),
        0.0,
    )
    out[:, 2] = np.maximum(
        np.expm1(Z[:, 2]),
        0.0,
    )

    qmax = np.maximum(
        np.expm1(Z[:, 3]),
        1e-8,
    )

    zr = np.clip(
        Z[:, 4],
        -30.0,
        30.0,
    )

    rbase = 1.0 / (
        1.0 + np.exp(-zr)
    )

    rbase = np.clip(
        rbase,
        1e-6,
        1.0 - 1e-6,
    )

    out[:, 3] = qmax * rbase
    out[:, 4] = qmax * (1.0 - rbase)

    L = np.c_[
        Z[:, 5:9],
        np.zeros(len(Z)),
    ]

    L -= L.max(
        axis=1,
        keepdims=True,
    )

    E = np.exp(L)

    out[:, 5:10] = (
        E
        / E.sum(
            axis=1,
            keepdims=True,
        )
    )

    return out


def fit_bundle(
    train: pd.DataFrame,
    features: list[str],
    args,
):
    xd = Xframe(
        train,
        features,
    )

    imp = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X = imp.fit_transform(
        xd
    ).astype("float32")

    xs = StandardScaler()

    Xs = xs.fit_transform(
        X
    ).astype("float32")

    Y = raw_to_latent(
        train
    )

    q = args.latent_clip_quantile

    clip_lo = np.quantile(
        Y,
        q,
        axis=0,
    )

    clip_hi = np.quantile(
        Y,
        1.0 - q,
        axis=0,
    )

    ym = Y.mean(axis=0)
    ys = Y.std(axis=0)

    ys = np.where(
        ys > 1e-10,
        ys,
        1.0,
    )

    Z = (
        Y - ym
    ) / ys

    models = {}

    print(
        "  [fit] LinearRegression",
        flush=True,
    )

    models["LinearRegression"] = (
        LinearRegression(
            n_jobs=-1,
        ).fit(
            Xs,
            Z,
        )
    )

    print(
        "  [fit] Ridge",
        flush=True,
    )

    models["Ridge"] = (
        Ridge(
            alpha=args.ridge_alpha,
            solver="lsqr",
        ).fit(
            Xs,
            Z,
        )
    )

    print(
        "  [fit] small HistGradientBoostingRegressor",
        flush=True,
    )

    base = HistGradientBoostingRegressor(
        learning_rate=args.hgb_lr,
        max_iter=args.hgb_iter,
        max_leaf_nodes=args.hgb_leaves,
        min_samples_leaf=args.hgb_leaf,
        l2_regularization=args.hgb_l2,
        random_state=args.seed,
    )

    models[
        "HistGradientBoostingRegressor"
    ] = MultiOutputRegressor(
        base,
        n_jobs=1,
    ).fit(
        X,
        Z,
    )

    return {
        "features": features,
        "imputer": imp,
        "x_scaler": xs,
        "latent_mean": ym,
        "latent_std": ys,
        "latent_clip_lo": clip_lo,
        "latent_clip_hi": clip_hi,
        "models": models,
        "templates": TEMPLATES,
        "raw_theta": RAW_THETA,
        "latent_targets": LATENT,
        "latent_clip_quantile": q,
        "frozen_training": True,
    }


def predict(
    bundle,
    d: pd.DataFrame,
):
    X = bundle[
        "imputer"
    ].transform(
        Xframe(
            d,
            bundle["features"],
        )
    ).astype("float32")

    Xs = bundle[
        "x_scaler"
    ].transform(
        X
    ).astype("float32")

    out = {}

    for name, model in bundle["models"].items():
        zin = (
            Xs
            if name in {
                "LinearRegression",
                "Ridge",
            }
            else X
        )

        z = np.asarray(
            model.predict(zin),
            dtype=float,
        )

        z = (
            z
            * bundle["latent_std"]
            + bundle["latent_mean"]
        )

        out[name] = latent_to_raw(
            z,
            bundle["latent_clip_lo"],
            bundle["latent_clip_hi"],
        )

    return out


def train_scales(
    train: pd.DataFrame,
):
    R = train[
        RAW_THETA
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)

    scales = []

    for j in range(5):
        s = (
            np.quantile(
                R[:, j],
                0.95,
            )
            - np.quantile(
                R[:, j],
                0.05,
            )
        )

        scales.append(
            max(
                float(s),
                1e-8,
            )
        )

    return np.asarray(
        scales
    )


def evaluate_parts(
    paths,
    split_name,
    fsname,
    bundle,
    scales,
):
    acc = {
        model: {
            "n": 0,
            "ae": np.zeros(5),
            "se": np.zeros(5),
            "qae": 0.0,
            "qpoints": 0,
        }
        for model in MODELS
    }

    for i, p in enumerate(paths, 1):
        print(
            f"[eval {fsname}/{split_name} {i}/{len(paths)}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(
            p
        )

        if d.empty:
            continue

        true = d[
            RAW_THETA
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(float)

        P = predict(
            bundle,
            d,
        )

        for model, pr in P.items():
            st = acc[model]
            st["n"] += len(d)

            e = (
                pr[:, :5]
                - true[:, :5]
            )

            st["ae"] += np.abs(
                e
            ).sum(axis=0)

            st["se"] += np.square(
                e
            ).sum(axis=0)

            st["qae"] += np.abs(
                pr[:, 5:10]
                - true[:, 5:10]
            ).sum()

            st["qpoints"] += (
                len(d)
                * 5
            )

        del d, true, P
        gc.collect()

    rows = []

    selection_idx = [
        RAW_THETA.index(t)
        for t in SELECTION_CORE_TARGETS
    ]

    for model, st in acc.items():
        if not st["n"]:
            continue

        mae = (
            st["ae"]
            / st["n"]
        )

        rmse = np.sqrt(
            st["se"]
            / st["n"]
        )

        nmae = (
            mae
            / scales
        )

        qmae = (
            st["qae"]
            / st["qpoints"]
        )

        core = float(
            np.mean(
                nmae[
                    selection_idx
                ]
            )
        )

        row = {
            "split": split_name,
            "feature_set": fsname,
            "model": model,
            "rows": st["n"],
            "q_share_mae": qmae,
            "mean_core_nmae": core,
            "selection_score": float(
                core + qmae
            ),
        }

        for j, target in enumerate(
            RAW_THETA[:5]
        ):
            row[
                f"{target}_mae"
            ] = mae[j]

            row[
                f"{target}_rmse"
            ] = rmse[j]

            row[
                f"{target}_nmae"
            ] = nmae[j]

        rows.append(
            row
        )

    return rows


def select_simple_model(
    metrics: pd.DataFrame,
    tolerance: float,
):
    val = metrics[
        metrics["split"].eq("val")
    ].copy()

    val = val.sort_values(
        "selection_score"
    ).reset_index(drop=True)

    absolute_best = val.iloc[0]

    threshold = float(
        absolute_best[
            "selection_score"
        ]
    ) * (
        1.0 + tolerance
    )

    candidates = val[
        val[
            "selection_score"
        ]
        <= threshold
    ].copy()

    candidates[
        "model_complexity_rank"
    ] = candidates[
        "model"
    ].map(
        MODEL_COMPLEXITY
    )

    candidates[
        "feature_complexity_rank"
    ] = candidates[
        "feature_set"
    ].map(
        FEATURE_COMPLEXITY
    )

    candidates = candidates.sort_values(
        [
            "model_complexity_rank",
            "feature_complexity_rank",
            "selection_score",
        ]
    )

    return (
        absolute_best,
        candidates.iloc[0],
        threshold,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--feature-set",
        choices=[
            "Z_base_H",
            "Z_base_M_U_H",
            "both",
        ],
        default="both",
    )

    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--hgb-lr", type=float, default=0.06)
    ap.add_argument("--hgb-iter", type=int, default=120)
    ap.add_argument("--hgb-leaves", type=int, default=15)
    ap.add_argument("--hgb-leaf", type=int, default=80)
    ap.add_argument("--hgb-l2", type=float, default=2.0)
    ap.add_argument("--simple-tolerance", type=float, default=0.02)
    ap.add_argument("--latent-clip-quantile", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    frozen = base / "frozen_modeling_dataset"

    manifest = load_manifest(
        frozen
    )

    schema = pd.read_csv(
        frozen / "feature_schema.csv"
    )

    fs, comp, fmap = feature_sets(
        schema
    )

    selected_sets = (
        list(fs)
        if args.feature_set == "both"
        else [args.feature_set]
    )

    train_sample_file = (
        frozen
        / manifest[
            "train_sample_file"
        ]
    )

    train = pd.read_pickle(
        train_sample_file
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
            "test"
        ]
    ]

    tr = pd.Timestamp(
        manifest["train_end"]
    )

    vs = pd.Timestamp(
        manifest[
            "validation_start"
        ]
    )

    ve = pd.Timestamp(
        manifest[
            "validation_end"
        ]
    )

    ts = pd.Timestamp(
        manifest[
            "test_start"
        ]
    )

    out = mkdir(
        base
        / "template_parameter_models"
    )

    mdir = mkdir(
        out
        / "models"
    )

    for stale in mdir.glob("*.joblib"):
        stale.unlink()

    print("=" * 80)
    print("Template parameter regression - frozen dataset")
    print("=" * 80)
    print(f"Year:                 {args.year}")
    print(f"Frozen train sample:  {len(train):,}")
    print(f"Train <=              {tr.date()}")
    print(f"Validation:           {vs.date()} .. {ve.date()}")
    print(f"Test >=               {ts.date()}")
    print(f"Z_base:               {len(comp['Z_base'])}")
    print(f"M:                    {len(comp['M'])}")
    print(f"U:                    {len(comp['U'])}")
    print(f"H:                    {len(comp['H'])}")
    print(f"Z_tr excluded:        {len(comp['Z_tr_unused'])}")
    print(f"Feature sets:         {selected_sets}")
    print()

    scales = train_scales(
        train
    )

    pd.DataFrame(
        {
            "target": RAW_THETA[:5],
            "train_nmae_scale_p95_minus_p05": scales,
        }
    ).to_csv(
        out
        / "target_train_scales.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []

    for i, z in enumerate(
        selected_sets,
        1,
    ):
        print(
            f"\n[feature set {i}/{len(selected_sets)}] {z}",
            flush=True,
        )

        bundle = fit_bundle(
            train,
            fs[z],
            args,
        )

        joblib.dump(
            bundle,
            mdir
            / f"{z}.joblib",
            compress=3,
        )

        rows += evaluate_parts(
            val_parts,
            "val",
            z,
            bundle,
            scales,
        )

        rows += evaluate_parts(
            test_parts,
            "test",
            z,
            bundle,
            scales,
        )

        del bundle
        gc.collect()

    metrics = pd.DataFrame(
        rows
    )

    metrics.to_csv(
        out
        / "parameter_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    absolute_best, selected, threshold = (
        select_simple_model(
            metrics,
            args.simple_tolerance,
        )
    )

    selected_fs = str(
        selected[
            "feature_set"
        ]
    )

    selected_model = str(
        selected[
            "model"
        ]
    )

    selected_test = metrics[
        (metrics["split"] == "test")
        & (
            metrics["feature_set"]
            == selected_fs
        )
        & (
            metrics["model"]
            == selected_model
        )
    ].iloc[0]

    selection_row = {
        "selected_feature_set": selected_fs,
        "selected_model": selected_model,
        "selection_rule": (
            "simplest configuration within relative validation tolerance "
            "of absolute best"
        ),
        "simple_tolerance": args.simple_tolerance,
        "selection_metric": (
            "mean NMAE(p_base,alpha,beta,q_base,q_span) + q_share_MAE"
        ),
        "absolute_best_feature_set": str(
            absolute_best["feature_set"]
        ),
        "absolute_best_model": str(
            absolute_best["model"]
        ),
        "absolute_best_val_score": float(
            absolute_best["selection_score"]
        ),
        "simplicity_threshold": float(
            threshold
        ),
        "selected_val_score": float(
            selected["selection_score"]
        ),
        "selected_test_score": float(
            selected_test["selection_score"]
        ),
        "q_base_source": "Regression",
        "q_span_source": "Regression",
        "frozen_dataset": True,
    }

    pd.DataFrame(
        [selection_row]
    ).to_csv(
        out
        / "selected_template_parameter_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cfg = {
        "year": args.year,
        "frozen_dataset": str(frozen),
        "train_sample_file": str(train_sample_file),
        "train_end": str(tr.date()),
        "validation": [
            str(vs.date()),
            str(ve.date()),
        ],
        "test_start": str(ts.date()),
        "feature_sets": selected_sets,
        "z_tr_used": False,
        "models": MODELS,
        "target_parameterization": {
            "p_base": "absolute",
            "alpha_beta": "log1p",
            "q_max_mw": "log1p(q_base+q_span)",
            "q_base_fraction": "logit(q_base/q_max)",
            "qshares": "ALR logits -> softmax",
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

    summary = "\n".join(
        [
            f"Template parameter regression - frozen dataset - {args.year}",
            "=" * 80,
            "",
            f"Frozen train sample: {len(train):,}",
            f"Train <= {tr.date()}",
            f"Validation = {vs.date()} .. {ve.date()}",
            f"Test >= {ts.date()}",
            "",
            "Final continuous theta:",
            "  p_base, alpha, beta",
            "  q_base_mw, q_span_mw",
            "  q1..q5",
            "",
            "Frozen data:",
            "  No split rebuilding.",
            "  No training resampling.",
            "  No Stage2 rejoin in 04b.",
            "",
            "Feature policy:",
            "  Z_tr excluded from parameter regression.",
            "  Compare Z_base+H vs Z_base+M+U+H.",
            "",
            "Models:",
            "  LinearRegression / Ridge / small HistGradientBoostingRegressor",
            "",
            f"Absolute validation best: {absolute_best['model']} / {absolute_best['feature_set']}",
            f"  score={absolute_best['selection_score']:.6f}",
            f"Simplicity threshold (+{args.simple_tolerance:.2%}): {threshold:.6f}",
            f"Selected lightweight configuration: {selected_model} / {selected_fs}",
            f"  VAL score={selected['selection_score']:.6f}",
            f"  TEST score={selected_test['selection_score']:.6f}",
            "",
            "q_base/q_span are absolute regression targets.",
            "Lag1 replacement is disabled.",
        ]
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
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
