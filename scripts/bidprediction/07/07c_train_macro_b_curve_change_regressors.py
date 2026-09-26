#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07c_train_macro_b_curve_change_regressors.py

Train the SAME lightweight model family as 06 on the stable Macro-B subset:
    zero-change
    Ridge
    Spline-GAM
    Random Forest

No new model structure is introduced.

Run:
python scripts/bidprediction/07c_train_macro_b_curve_change_regressors.py --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
CURVE_COLS = [
    *SHAPE_COLS,
    "p_anchor",
    "p_span",
    "q_anchor_mw",
    "q_span_mw",
]

RAW_PREV_COLS = [
    *[
        f"_curvevec_p{i:02d}_lag1"
        for i in range(21)
    ],
    "_curvevec_q_anchor_lag1",
    "_curvevec_log_q_span_lag1",
]

META_COLS = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
]


def files_from_manifest(manifest, split):
    return [
        x["file"] if isinstance(x, dict) else x
        for x in manifest["parts"][split]
    ]


def num_frame(d, cols):
    return d[cols].apply(
        pd.to_numeric,
        errors="coerce",
    )


def zero_latent(bundle, k):
    scaler = bundle["scaler"]
    pca = bundle["pca"]

    raw_zero = np.zeros(
        (1, 23),
        dtype=np.float64,
    )

    return pca.transform(
        scaler.transform(
            raw_zero
        )
    )[0, :k]


def load_train_sample(
    dataset,
    manifest,
    features,
    targets,
    max_rows,
    seed,
):
    files = files_from_manifest(
        manifest,
        "train",
    )

    per_part = max(
        1,
        int(
            math.ceil(
                max_rows
                / max(
                    len(files),
                    1,
                )
            )
        ),
    )

    blocks = []

    for i, rel in enumerate(files, 1):
        p = dataset / rel
        print(
            f"[train {i}/{len(files)}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(p)

        if d.empty:
            continue

        n = min(
            per_part,
            len(d),
        )

        if n < len(d):
            d = d.sample(
                n=n,
                random_state=(
                    seed
                    + 7919 * i
                ),
            )

        blocks.append(
            d[
                [
                    *features,
                    *targets,
                ]
            ].copy()
        )

        del d
        gc.collect()

    if not blocks:
        raise ValueError(
            "No Macro-B TRAIN rows."
        )

    train = pd.concat(
        blocks,
        ignore_index=True,
    )

    if len(train) > max_rows:
        train = (
            train.sample(
                n=max_rows,
                random_state=seed,
            )
            .reset_index(drop=True)
        )

    return train


def top_context_by_spearman(
    d,
    context_features,
    target,
    top_k,
):
    y = pd.to_numeric(
        d[target],
        errors="coerce",
    )

    scores = []

    for c in context_features:
        x = pd.to_numeric(
            d[c],
            errors="coerce",
        )

        valid = (
            x.notna()
            & y.notna()
        )

        if valid.sum() < 100:
            continue

        if x.loc[valid].nunique() <= 1:
            continue

        rho = x.loc[valid].corr(
            y.loc[valid],
            method="spearman",
        )

        if pd.notna(rho):
            scores.append(
                (
                    c,
                    float(abs(rho)),
                    float(rho),
                )
            )

    scores.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return scores[:top_k]


def fit_ridge(
    train,
    features,
    targets,
    alpha,
):
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(alpha=alpha),
            ),
        ]
    )

    model.fit(
        num_frame(
            train,
            features,
        ),
        num_frame(
            train,
            targets,
        ).to_numpy(np.float64),
    )

    return model


def fit_gam(
    train,
    targets,
    source_features,
    history_features,
    top_context,
    knots,
    degree,
    alpha,
):
    all_lag1 = [
        f"{z}_lag1"
        for z in targets
    ]

    models = {}
    selection = {}

    for i, target in enumerate(targets, 1):
        print(
            f"[GAM {i}/{len(targets)}] {target}",
            flush=True,
        )

        own = [
            c
            for c in history_features
            if c.startswith(
                f"{target}_"
            )
        ]

        top = top_context_by_spearman(
            train,
            source_features,
            target,
            top_context,
        )

        selected_context = [
            x[0]
            for x in top
        ]

        features = list(
            dict.fromkeys(
                [
                    *all_lag1,
                    *own,
                    *selected_context,
                ]
            )
        )

        model = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=knots,
                        degree=degree,
                        knots="quantile",
                        extrapolation="constant",
                        include_bias=False,
                        sparse_output=True,
                    ),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=alpha,
                        solver="lsqr",
                    ),
                ),
            ]
        )

        model.fit(
            num_frame(
                train,
                features,
            ),
            pd.to_numeric(
                train[target],
                errors="coerce",
            ).to_numpy(np.float64),
        )

        models[target] = {
            "features": features,
            "model": model,
        }

        selection[target] = {
            "features": features,
            "top_context": [
                {
                    "feature": c,
                    "abs_spearman": a,
                    "spearman": r,
                }
                for c, a, r in top
            ],
        }

    return models, selection


def fit_rf(
    train,
    features,
    targets,
    args,
):
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=args.rf_trees,
                    max_depth=args.rf_max_depth,
                    min_samples_leaf=args.rf_min_leaf,
                    max_features=args.rf_max_features,
                    n_jobs=args.n_jobs,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    model.fit(
        num_frame(
            train,
            features,
        ),
        num_frame(
            train,
            targets,
        ).to_numpy(np.float32),
    )

    return model


def predict_gam(
    d,
    models,
    targets,
):
    out = np.empty(
        (
            len(d),
            len(targets),
        ),
        dtype=np.float32,
    )

    for j, target in enumerate(targets):
        spec = models[target]

        out[:, j] = (
            spec["model"]
            .predict(
                num_frame(
                    d,
                    spec["features"],
                )
            )
            .astype(np.float32)
        )

    return out


def latent_metrics(
    true,
    predictions,
):
    rows = []

    for name, pred in predictions.items():
        err = pred - true

        rows.append(
            {
                "model": name,
                "rows": int(len(true)),
                "delta_latent_mae_mean": float(
                    np.mean(
                        np.abs(err)
                    )
                ),
                "delta_latent_rmse_mean": float(
                    np.sqrt(
                        np.mean(
                            err**2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def predict_split(
    dataset,
    manifest,
    split,
    out_dir,
    features,
    targets,
    zero_z,
    ridge,
    gam,
    rf,
):
    files = files_from_manifest(
        manifest,
        split,
    )

    pred_dir = (
        out_dir
        / "prediction_parts"
        / split
    )
    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = []
    metric_parts = []

    for i, rel in enumerate(files, 1):
        p = dataset / rel
        print(
            f"[predict {split} {i}/{len(files)}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(p)

        if d.empty:
            continue

        X = num_frame(
            d,
            features,
        )

        true = num_frame(
            d,
            targets,
        ).to_numpy(np.float32)

        zero = np.repeat(
            zero_z[None, :],
            len(d),
            axis=0,
        ).astype(np.float32)

        pred_ridge = ridge.predict(
            X
        ).astype(np.float32)

        pred_gam = predict_gam(
            d,
            gam,
            targets,
        )

        pred_rf = rf.predict(
            X
        ).astype(np.float32)

        predictions = {
            "zero_change": zero,
            "ridge": pred_ridge,
            "spline_gam": pred_gam,
            "random_forest": pred_rf,
        }

        metric_parts.append(
            latent_metrics(
                true,
                predictions,
            )
        )

        keep = [
            c
            for c in [
                *META_COLS,
                *CURVE_COLS,
                *RAW_PREV_COLS,
                "y_template_id",
            ]
            if c in d.columns
        ]

        out_d = d[keep].copy()

        for j, z in enumerate(targets):
            out_d[
                f"true_{z}"
            ] = true[:, j]

            for name, pred in predictions.items():
                out_d[
                    f"pred_{name}_{z}"
                ] = pred[:, j]

        name = (
            f"{split}_macro_b_predictions_"
            f"{i:04d}.pkl"
        )

        path = pred_dir / name
        out_d.to_pickle(path)

        written.append(
            {
                "file": str(
                    path.relative_to(
                        out_dir
                    )
                ),
                "rows": int(len(out_d)),
            }
        )

        del (
            d,
            X,
            true,
            zero,
            pred_ridge,
            pred_gam,
            pred_rf,
            out_d,
        )
        gc.collect()

    metrics = pd.concat(
        metric_parts,
        ignore_index=True,
    )

    agg = []

    for model, g in metrics.groupby(
        "model",
        sort=False,
    ):
        w = g["rows"].to_numpy(float)

        agg.append(
            {
                "model": model,
                "rows": int(w.sum()),
                "delta_latent_mae_mean": float(
                    np.average(
                        g[
                            "delta_latent_mae_mean"
                        ],
                        weights=w,
                    )
                ),
                "delta_latent_rmse_mean": float(
                    np.average(
                        g[
                            "delta_latent_rmse_mean"
                        ],
                        weights=w,
                    )
                ),
                "split": split,
            }
        )

    return written, pd.DataFrame(agg)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--dataset-dir",
        default="macro_b_curve_change_latent_dataset",
    )
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=300_000,
    )
    ap.add_argument(
        "--gam-train-rows",
        type=int,
        default=150_000,
    )
    ap.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
    )
    ap.add_argument(
        "--gam-alpha",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--gam-top-context",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--gam-knots",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--gam-degree",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--rf-trees",
        type=int,
        default=96,
    )
    ap.add_argument(
        "--rf-max-depth",
        type=int,
        default=18,
    )
    ap.add_argument(
        "--rf-min-leaf",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--rf-max-features",
        type=float,
        default=0.40,
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=8,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    dataset = base / args.dataset_dir

    manifest = json.loads(
        (
            dataset
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    bundle = joblib.load(
        dataset
        / "delta_pca_bundle.joblib"
    )

    targets = list(
        manifest["delta_columns"]
    )

    features = list(
        manifest["model_features"]
    )

    source_features = list(
        manifest["source_model_features"]
    )

    history_features = list(
        manifest["delta_history_features"]
    )

    zero_z = zero_latent(
        bundle,
        len(targets),
    )

    out_dir = (
        base
        / "macro_b_curve_change_regression_models"
    )

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out_dir} exists. Use --overwrite."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"07c Macro-B lightweight regressors - {args.year}"
    )
    print("=" * 80)
    print(
        f"Delta latent dimension = {len(targets)}"
    )
    print(
        f"Model features = {len(features)}"
    )

    train = load_train_sample(
        dataset,
        manifest,
        features,
        targets,
        args.max_train_rows,
        args.seed,
    )

    print(
        f"Shared train sample = {len(train):,}"
    )

    ridge = fit_ridge(
        train,
        features,
        targets,
        args.ridge_alpha,
    )

    if len(train) > args.gam_train_rows:
        gam_train = (
            train.sample(
                n=args.gam_train_rows,
                random_state=args.seed + 17,
            )
            .reset_index(drop=True)
        )
    else:
        gam_train = train

    gam, gam_selection = fit_gam(
        gam_train,
        targets,
        source_features,
        history_features,
        args.gam_top_context,
        args.gam_knots,
        args.gam_degree,
        args.gam_alpha,
    )

    rf = fit_rf(
        train,
        features,
        targets,
        args,
    )

    joblib.dump(
        {
            "model": ridge,
            "features": features,
            "targets": targets,
        },
        out_dir
        / "ridge_model.joblib",
        compress=3,
    )

    joblib.dump(
        {
            "models": gam,
            "targets": targets,
            "selection": gam_selection,
        },
        out_dir
        / "spline_gam_models.joblib",
        compress=3,
    )

    joblib.dump(
        {
            "model": rf,
            "features": features,
            "targets": targets,
        },
        out_dir
        / "random_forest_model.joblib",
        compress=3,
    )

    (
        out_dir
        / "spline_gam_feature_selection.json"
    ).write_text(
        json.dumps(
            gam_selection,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pred_manifest = {
        "year": args.year,
        "source_dataset": str(dataset),
        "delta_columns": targets,
        "models": [
            "zero_change",
            "ridge",
            "spline_gam",
            "random_forest",
        ],
        "parts": {},
    }

    metric_tables = []

    for split in ["val", "test"]:
        written, metrics = predict_split(
            dataset,
            manifest,
            split,
            out_dir,
            features,
            targets,
            zero_z,
            ridge,
            gam,
            rf,
        )

        pred_manifest["parts"][split] = written
        metric_tables.append(metrics)

        metrics.to_csv(
            out_dir
            / f"{split}_latent_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    (
        out_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            pred_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    all_metrics = pd.concat(
        metric_tables,
        ignore_index=True,
    )

    summary = "\n".join(
        [
            (
                f"07c Macro-B lightweight regressors - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                f"Delta latent dimension = "
                f"{len(targets)}"
            ),
            (
                f"Model features = "
                f"{len(features)}"
            ),
            (
                f"Shared train sample = "
                f"{len(train):,}"
            ),
            (
                f"Spline-GAM train sample = "
                f"{len(gam_train):,}"
            ),
            "",
            "Latent diagnostics:",
            all_metrics.to_string(
                index=False
            ),
            "",
            (
                "Final model selection is deferred to 07d and uses "
                "Macro-B validation reconstructed curve WAPE."
            ),
        ]
    )

    (
        out_dir
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
