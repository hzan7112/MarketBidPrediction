#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06b_train_curve_change_regressors.py

Train lightweight regressors for curve-change latent coordinates DeltaZ.

Models:
0) zero_change baseline: DeltaV = 0 (equivalent to raw curve persistence)
1) Ridge
2) Spline additive model (GAM-style, sklearn only)
3) Random Forest

No neural network is used.

The script writes VAL/TEST DeltaZ predictions. Final model selection is done
in 06c using reconstructed full-curve price WAPE on VALIDATION only.

Run
---
python scripts/bidprediction/06b_train_curve_change_regressors.py --year 2025 --overwrite
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
from sklearn.preprocessing import StandardScaler, SplineTransformer


SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
CURVE_TARGET_COLS = [
    *SHAPE_COLS,
    "p_anchor",
    "p_span",
    "q_anchor_mw",
    "q_span_mw",
]
RAW_PREV_COLS = [
    *[f"_curvevec_p{i:02d}_lag1" for i in range(21)],
    "_curvevec_q_anchor_lag1",
    "_curvevec_log_q_span_lag1",
]
META_CANDIDATES = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
]


def part_files(manifest, split):
    out = []
    for item in manifest["parts"][split]:
        out.append(item["file"] if isinstance(item, dict) else item)
    return out


def num_frame(d, cols):
    return d[cols].apply(pd.to_numeric, errors="coerce")


def load_train_sample(
    dataset_dir,
    manifest,
    feature_cols,
    target_cols,
    max_rows,
    seed,
):
    files = part_files(manifest, "train")
    per_part = max(1, int(math.ceil(max_rows / len(files))))
    blocks = []

    for i, rel in enumerate(files, 1):
        path = dataset_dir / rel
        print(
            f"[train sample {i}/{len(files)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)
        if d.empty:
            continue

        n = min(per_part, len(d))
        if n < len(d):
            d = d.sample(n=n, random_state=seed + 7919 * i)

        blocks.append(d[[*feature_cols, *target_cols]].copy())
        del d
        gc.collect()

    if not blocks:
        raise ValueError("No TRAIN rows found.")

    out = pd.concat(blocks, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return out


def top_context_by_spearman(d, features, target, top_k):
    if top_k <= 0:
        return []

    y = pd.to_numeric(d[target], errors="coerce")
    scores = []

    for c in features:
        x = pd.to_numeric(d[c], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 100:
            continue
        if x.loc[valid].nunique(dropna=True) <= 1:
            continue

        rho = x.loc[valid].corr(y.loc[valid], method="spearman")
        if pd.notna(rho):
            scores.append((c, float(abs(rho)), float(rho)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def fit_ridge(train, features, targets, alpha):
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

    X = num_frame(train, features)
    Y = num_frame(train, targets).to_numpy(np.float64)
    model.fit(X, Y)
    return model


def fit_spline_gam(
    train,
    delta_cols,
    delta_history_features,
    source_features,
    top_context,
    n_knots,
    degree,
    alpha,
):
    all_delta_lag1 = [f"{z}_lag1" for z in delta_cols]
    models = {}
    selection = {}

    for i, target in enumerate(delta_cols, 1):
        print(
            f"[spline GAM {i}/{len(delta_cols)}] {target}",
            flush=True,
        )

        own_history = [
            c
            for c in delta_history_features
            if c.startswith(f"{target}_")
        ]

        top = top_context_by_spearman(
            train,
            source_features,
            target,
            top_context,
        )
        context_selected = [x[0] for x in top]

        features = list(
            dict.fromkeys(
                [
                    *all_delta_lag1,
                    *own_history,
                    *context_selected,
                ]
            )
        )

        pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=n_knots,
                        degree=degree,
                        knots="quantile",
                        extrapolation="constant",
                        include_bias=False,
                        sparse_output=True,
                    ),
                ),
                (
                    "ridge",
                    Ridge(alpha=alpha, solver="lsqr"),
                ),
            ]
        )

        X = num_frame(train, features)
        y = pd.to_numeric(train[target], errors="coerce").to_numpy(np.float64)
        pipe.fit(X, y)

        models[target] = {
            "features": features,
            "model": pipe,
        }
        selection[target] = {
            "all_delta_lag1": all_delta_lag1,
            "own_delta_history": own_history,
            "top_source_features": [
                {
                    "feature": c,
                    "abs_spearman": a,
                    "spearman": r,
                }
                for c, a, r in top
            ],
            "final_features": features,
        }

    return models, selection


def fit_random_forest(
    train,
    features,
    targets,
    trees,
    max_depth,
    min_leaf,
    max_features,
    n_jobs,
    seed,
):
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=trees,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    max_features=max_features,
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            ),
        ]
    )

    X = num_frame(train, features)
    Y = num_frame(train, targets).to_numpy(np.float32)
    model.fit(X, Y)
    return model


def predict_spline_gam(d, models, delta_cols):
    out = np.empty((len(d), len(delta_cols)), dtype=np.float32)

    for j, target in enumerate(delta_cols):
        spec = models[target]
        X = num_frame(d, spec["features"])
        out[:, j] = spec["model"].predict(X).astype(np.float32)

    return out


def latent_metrics(y_true, predictions, delta_cols):
    rows = []
    for model_name, pred in predictions.items():
        err = pred - y_true
        row = {
            "model": model_name,
            "rows": int(len(y_true)),
            "delta_latent_mae_mean": float(np.mean(np.abs(err))),
            "delta_latent_rmse_mean": float(np.sqrt(np.mean(err**2))),
        }
        for j, z in enumerate(delta_cols):
            row[f"{z}_mae"] = float(np.mean(np.abs(err[:, j])))
        rows.append(row)
    return pd.DataFrame(rows)


def zero_change_latent(delta_bundle, k):
    scaler = delta_bundle["scaler"]
    pca = delta_bundle["pca"]
    zero = np.zeros((1, 23), dtype=np.float64)
    return pca.transform(scaler.transform(zero))[0, :k].astype(np.float32)


def evaluate_and_write_split(
    dataset_dir,
    manifest,
    split,
    out_dir,
    feature_cols,
    delta_cols,
    ridge_model,
    gam_models,
    rf_model,
    zero_latent,
):
    files = part_files(manifest, split)
    pred_dir = out_dir / "prediction_parts" / split
    pred_dir.mkdir(parents=True, exist_ok=True)

    written = []
    metric_blocks = []

    for i, rel in enumerate(files, 1):
        path = dataset_dir / rel
        print(
            f"[predict {split} {i}/{len(files)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(path).reset_index(drop=True)
        if d.empty:
            continue

        X = num_frame(d, feature_cols)
        y_true = num_frame(d, delta_cols).to_numpy(np.float32)

        zero_pred = np.repeat(
            zero_latent[None, :],
            len(d),
            axis=0,
        )
        ridge = ridge_model.predict(X).astype(np.float32)
        gam = predict_spline_gam(d, gam_models, delta_cols)
        rf = rf_model.predict(X).astype(np.float32)

        predictions = {
            "zero_change": zero_pred,
            "ridge": ridge,
            "spline_gam": gam,
            "random_forest": rf,
        }

        metric_blocks.append(
            latent_metrics(y_true, predictions, delta_cols)
        )

        keep = [
            c
            for c in [
                *META_CANDIDATES,
                *CURVE_TARGET_COLS,
                *RAW_PREV_COLS,
                "y_template_id",
            ]
            if c in d.columns
        ]
        out_d = d[keep].copy()

        for j, z in enumerate(delta_cols):
            out_d[f"true_{z}"] = y_true[:, j]
            for name, pred in predictions.items():
                out_d[f"pred_{name}_{z}"] = pred[:, j]

        out_name = f"{split}_curve_change_predictions_{i:04d}.pkl"
        out_path = pred_dir / out_name
        out_d.to_pickle(out_path)

        written.append(
            {
                "file": str(out_path.relative_to(out_dir)),
                "rows": int(len(out_d)),
            }
        )

        del d, X, y_true, zero_pred, ridge, gam, rf, out_d
        gc.collect()

    metrics = (
        pd.concat(metric_blocks, ignore_index=True)
        if metric_blocks
        else pd.DataFrame()
    )

    if not metrics.empty:
        numeric_cols = [c for c in metrics.columns if c not in {"model", "rows"}]
        agg_rows = []
        for model_name, g in metrics.groupby("model", sort=False):
            weights = g["rows"].to_numpy(float)
            row = {"model": model_name, "rows": int(weights.sum())}
            for c in numeric_cols:
                row[c] = float(
                    np.average(g[c].to_numpy(float), weights=weights)
                )
            agg_rows.append(row)
        metrics = pd.DataFrame(agg_rows)

    return written, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--root", default="data/processed/bidprediction")
    ap.add_argument("--dataset-dir", default="curve_change_latent_dataset")
    ap.add_argument("--max-train-rows", type=int, default=300_000)
    ap.add_argument("--gam-train-rows", type=int, default=150_000)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--gam-alpha", type=float, default=1.0)
    ap.add_argument("--gam-top-context", type=int, default=20)
    ap.add_argument("--gam-knots", type=int, default=5)
    ap.add_argument("--gam-degree", type=int, default=2)
    ap.add_argument("--rf-trees", type=int, default=64)
    ap.add_argument("--rf-max-depth", type=int, default=16)
    ap.add_argument("--rf-min-leaf", type=int, default=12)
    ap.add_argument("--rf-max-features", type=float, default=0.35)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    dataset_dir = base / args.dataset_dir
    manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    delta_bundle = joblib.load(dataset_dir / "delta_pca_bundle.joblib")

    delta_cols = list(manifest["delta_columns"])
    delta_history_features = list(manifest["delta_history_features"])
    source_features = list(manifest["source_model_features"])
    feature_cols = list(manifest["model_features"])

    out = base / "curve_change_regression_models"
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} already exists. Use --overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Lightweight curve-change regressors - {args.year}")
    print("=" * 80)
    print(f"Delta latent dimension = {len(delta_cols)}")
    print(f"Model features = {len(feature_cols)}")
    print("Models = Zero-change / Ridge / Spline-GAM / Random Forest")
    print()

    train = load_train_sample(
        dataset_dir,
        manifest,
        feature_cols,
        delta_cols,
        args.max_train_rows,
        args.seed,
    )
    print(f"Shared train sample = {len(train):,}")

    ridge_model = fit_ridge(
        train, feature_cols, delta_cols, args.ridge_alpha
    )

    if len(train) > args.gam_train_rows:
        gam_train = train.sample(
            n=args.gam_train_rows,
            random_state=args.seed + 17,
        ).reset_index(drop=True)
    else:
        gam_train = train

    gam_models, gam_selection = fit_spline_gam(
        gam_train,
        delta_cols,
        delta_history_features,
        source_features,
        args.gam_top_context,
        args.gam_knots,
        args.gam_degree,
        args.gam_alpha,
    )

    rf_model = fit_random_forest(
        train,
        feature_cols,
        delta_cols,
        args.rf_trees,
        args.rf_max_depth,
        args.rf_min_leaf,
        args.rf_max_features,
        args.n_jobs,
        args.seed,
    )

    joblib.dump(
        {
            "version": "curve-change-ridge-v1",
            "features": feature_cols,
            "targets": delta_cols,
            "alpha": args.ridge_alpha,
            "model": ridge_model,
        },
        out / "ridge_model.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "version": "curve-change-spline-gam-v1",
            "targets": delta_cols,
            "models": gam_models,
            "selection": gam_selection,
            "n_knots": args.gam_knots,
            "degree": args.gam_degree,
            "alpha": args.gam_alpha,
        },
        out / "spline_gam_models.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "version": "curve-change-random-forest-v1",
            "features": feature_cols,
            "targets": delta_cols,
            "model": rf_model,
            "config": {
                "trees": args.rf_trees,
                "max_depth": args.rf_max_depth,
                "min_leaf": args.rf_min_leaf,
                "max_features": args.rf_max_features,
            },
        },
        out / "random_forest_model.joblib",
        compress=3,
    )

    ridge_core = ridge_model.named_steps["ridge"]
    ridge_coef = pd.DataFrame(
        ridge_core.coef_,
        index=delta_cols,
        columns=feature_cols,
    )
    ridge_long = (
        ridge_coef.reset_index(names="target")
        .melt(
            id_vars="target",
            var_name="feature",
            value_name="coefficient",
        )
    )
    ridge_long["abs_coefficient"] = ridge_long["coefficient"].abs()
    ridge_long.to_csv(
        out / "ridge_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rf_core = rf_model.named_steps["rf"]
    rf_imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf_core.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    rf_imp.to_csv(
        out / "random_forest_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (out / "spline_gam_feature_selection.json").write_text(
        json.dumps(gam_selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    zero_latent = zero_change_latent(delta_bundle, len(delta_cols))

    pred_manifest = {
        "year": int(args.year),
        "source_dataset": str(dataset_dir),
        "delta_columns": delta_cols,
        "models": [
            "zero_change",
            "ridge",
            "spline_gam",
            "random_forest",
        ],
        "parts": {},
    }

    all_metrics = []

    for split in ["val", "test"]:
        written, metrics = evaluate_and_write_split(
            dataset_dir,
            manifest,
            split,
            out,
            feature_cols,
            delta_cols,
            ridge_model,
            gam_models,
            rf_model,
            zero_latent,
        )
        pred_manifest["parts"][split] = written
        metrics["split"] = split
        all_metrics.append(metrics)
        metrics.to_csv(
            out / f"{split}_delta_latent_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    (out / "manifest.json").write_text(
        json.dumps(pred_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_metrics = pd.concat(all_metrics, ignore_index=True)
    summary = "\n".join(
        [
            f"Lightweight curve-change regressors - {args.year}",
            "=" * 80,
            "",
            f"Delta latent dimension = {len(delta_cols)}",
            f"Model features = {len(feature_cols)}",
            f"Shared train sample = {len(train):,}",
            f"Spline-GAM train sample = {len(gam_train):,}",
            "",
            "Latent-space diagnostics:",
            all_metrics.to_string(index=False),
            "",
            (
                "Final selection is deferred to 06c and uses validation "
                "reconstructed curve WAPE. Zero-change is an eligible winner."
            ),
        ]
    )
    (out / "summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
