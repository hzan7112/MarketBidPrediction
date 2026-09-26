#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11c_monthly_rolling_price_bounds_validation.py

2025 monthly expanding-window validation for bid price bounds.

Purpose
-------
Keep EVERYTHING from the current simple price-bound task unchanged:
    83 frozen feature-only inputs
        -> RandomForestRegressor
        -> [p_min, p_max]

Only change the TIME EVALUATION scheme.

Default rolling folds
---------------------
Train Jan-Jun -> evaluate Jul
Train Jan-Jul -> evaluate Aug
Train Jan-Aug -> evaluate Sep
Train Jan-Sep -> evaluate Oct
Train Jan-Oct -> evaluate Nov
Train Jan-Nov -> evaluate Dec

No PCA.
No template routing.
No Prediction Family.
No historical raw bid curve.
No new features.
No model search.

The RF hyperparameters are the same as 11_validate_bid_price_bounds.py:
    n_estimators=128
    max_depth=18
    min_samples_leaf=8
    max_features=0.5
    max_train_rows=300000
    random_state=42

Input
-----
data/processed/bidprediction/<year>/final_feature_only_dataset/

Output
------
data/processed/bidprediction/<year>/monthly_rolling_price_bounds/
    monthly_metrics.csv
    monthly_target_distribution.csv
    monthly_summary.csv
    monthly_wape.png
    monthly_mae.png
    monthly_r2.png
    summary.txt

Run
---
python scripts/bidprediction/11c_monthly_rolling_price_bounds_validation.py --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
CURVE_COLS = [*SHAPE_COLS, "p_anchor", "p_span"]


def all_part_files(manifest):
    out = []
    seen = set()

    for split in ["train", "val", "test"]:
        for x in manifest.get("parts", {}).get(split, []):
            rel = x["file"] if isinstance(x, dict) else x
            if rel not in seen:
                seen.add(rel)
                out.append(rel)

    if not out:
        raise RuntimeError("No dataset parts found in manifest.")

    return out


def numeric_frame(d, cols):
    return d[cols].apply(pd.to_numeric, errors="coerce")


def row_month(d):
    """
    Return integer month 1..12 for every row.
    Prefer local_date because Stage-10 evaluation is defined in local market time.
    """
    if "local_date" in d.columns:
        dt = pd.to_datetime(d["local_date"], errors="coerce")
    elif "timestamp_local" in d.columns:
        dt = pd.to_datetime(d["timestamp_local"], errors="coerce")
    else:
        raise KeyError(
            "Neither local_date nor timestamp_local is available; "
            "cannot perform monthly rolling validation."
        )

    return dt.dt.month.to_numpy(dtype=np.float64)


def compute_price_bounds(d):
    missing = [c for c in CURVE_COLS if c not in d.columns]
    if missing:
        raise KeyError("Missing curve columns: " + ", ".join(missing))

    shape = numeric_frame(d, SHAPE_COLS).to_numpy(np.float64)

    p_anchor = pd.to_numeric(
        d["p_anchor"],
        errors="coerce",
    ).to_numpy(np.float64)

    p_span = pd.to_numeric(
        d["p_span"],
        errors="coerce",
    ).to_numpy(np.float64)

    flat = np.abs(p_span) <= 1e-12

    if flat.any():
        shape[flat, :] = np.nan_to_num(
            shape[flat, :],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    price = p_anchor[:, None] + p_span[:, None] * shape

    valid = np.isfinite(price).all(axis=1)

    p_min = np.full(len(d), np.nan, dtype=np.float64)
    p_max = np.full(len(d), np.nan, dtype=np.float64)

    if valid.any():
        p_min[valid] = np.min(price[valid, :], axis=1)
        p_max[valid] = np.max(price[valid, :], axis=1)

    return p_min, p_max, valid


def metric_row(
    eval_month,
    model_name,
    target,
    y_true,
    y_pred,
):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "eval_month": int(eval_month),
            "model": model_name,
            "target": target,
            "rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "wape_pct": np.nan,
            "smape_pct": np.nan,
            "r2": np.nan,
        }

    err = y_pred - y_true
    ae = np.abs(err)

    smape_den = np.abs(y_true) + np.abs(y_pred)
    smape_mask = smape_den > 1e-8

    if smape_mask.any():
        smape = float(
            100.0
            * np.mean(
                2.0
                * ae[smape_mask]
                / smape_den[smape_mask]
            )
        )
    else:
        smape = 0.0

    return {
        "eval_month": int(eval_month),
        "model": model_name,
        "target": target,
        "rows": int(len(y_true)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "wape_pct": float(
            100.0
            * np.sum(ae)
            / max(np.sum(np.abs(y_true)), 1e-12)
        ),
        "smape_pct": smape,
        "r2": float(r2_score(y_true, y_pred))
        if len(y_true) >= 2
        else np.nan,
    }


def target_distribution_row(eval_month, target, x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "eval_month": int(eval_month),
            "target": target,
            "rows": 0,
        }

    q = np.quantile(
        x,
        [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )

    return {
        "eval_month": int(eval_month),
        "target": target,
        "rows": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(np.max(x)),
    }


def build_train_reservoir(
    dataset,
    part_files,
    features,
    eval_month,
    max_train_rows,
    seed,
):
    """
    Exact random-priority reservoir over all eligible rows before eval_month.

    Every eligible training row receives an iid random priority. Keeping the
    smallest max_train_rows priorities gives a uniform random sample without
    loading all historical rows into memory.
    """
    rng = np.random.default_rng(seed + eval_month * 10007)

    reservoir = None
    eligible_rows = 0

    for i, rel in enumerate(part_files, 1):
        path = dataset / rel

        print(
            f"[month {eval_month:02d} TRAIN scan {i}/{len(part_files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)

        if d.empty:
            continue

        months = row_month(d)

        time_mask = (
            np.isfinite(months)
            & (months < eval_month)
        )

        if not time_mask.any():
            del d, months, time_mask
            gc.collect()
            continue

        p_min, p_max, curve_valid = compute_price_bounds(d)

        valid = (
            time_mask
            & curve_valid
            & np.isfinite(p_min)
            & np.isfinite(p_max)
        )

        idx = np.flatnonzero(valid)

        if len(idx) == 0:
            del d, months, time_mask, p_min, p_max, curve_valid, valid, idx
            gc.collect()
            continue

        eligible_rows += len(idx)

        X = numeric_frame(
            d.iloc[idx],
            features,
        ).reset_index(drop=True)

        block = X.copy()
        block["_target_p_min"] = p_min[idx]
        block["_target_p_max"] = p_max[idx]
        block["_priority"] = rng.random(len(block))

        if reservoir is None:
            reservoir = block
        else:
            reservoir = pd.concat(
                [reservoir, block],
                ignore_index=True,
            )

        if len(reservoir) > max_train_rows:
            keep_idx = np.argpartition(
                reservoir["_priority"].to_numpy(np.float64),
                max_train_rows - 1,
            )[:max_train_rows]

            reservoir = (
                reservoir.iloc[keep_idx]
                .copy()
                .reset_index(drop=True)
            )

        del (
            d,
            months,
            time_mask,
            p_min,
            p_max,
            curve_valid,
            valid,
            idx,
            X,
            block,
        )

        gc.collect()

    if reservoir is None or reservoir.empty:
        raise RuntimeError(
            f"No valid training rows found before month {eval_month}."
        )

    reservoir = (
        reservoir
        .sort_values("_priority", kind="mergesort")
        .head(max_train_rows)
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )

    return reservoir, eligible_rows


def evaluate_month(
    dataset,
    part_files,
    features,
    eval_month,
    model,
    train_mean,
):
    store = {
        "true_p_min": [],
        "true_p_max": [],
        "pred_rf_p_min": [],
        "pred_rf_p_max": [],
    }

    inversion_rows = 0
    total_rows = 0

    for i, rel in enumerate(part_files, 1):
        path = dataset / rel

        print(
            f"[month {eval_month:02d} EVAL scan {i}/{len(part_files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)

        if d.empty:
            continue

        months = row_month(d)

        month_mask = (
            np.isfinite(months)
            & (months == eval_month)
        )

        if not month_mask.any():
            del d, months, month_mask
            gc.collect()
            continue

        p_min, p_max, curve_valid = compute_price_bounds(d)

        valid = (
            month_mask
            & curve_valid
            & np.isfinite(p_min)
            & np.isfinite(p_max)
        )

        idx = np.flatnonzero(valid)

        if len(idx) == 0:
            del d, months, month_mask, p_min, p_max, curve_valid, valid, idx
            gc.collect()
            continue

        X = numeric_frame(
            d.iloc[idx],
            features,
        )

        pred = model.predict(X).astype(np.float64)

        inversion_rows += int(
            (pred[:, 0] > pred[:, 1]).sum()
        )

        total_rows += len(idx)

        store["true_p_min"].append(
            p_min[idx].astype(np.float32)
        )

        store["true_p_max"].append(
            p_max[idx].astype(np.float32)
        )

        store["pred_rf_p_min"].append(
            pred[:, 0].astype(np.float32)
        )

        store["pred_rf_p_max"].append(
            pred[:, 1].astype(np.float32)
        )

        del (
            d,
            months,
            month_mask,
            p_min,
            p_max,
            curve_valid,
            valid,
            idx,
            X,
            pred,
        )

        gc.collect()

    if total_rows == 0:
        raise RuntimeError(
            f"No valid evaluation rows found for month {eval_month}."
        )

    arrays = {
        k: np.concatenate(v)
        for k, v in store.items()
    }

    true_min = arrays["true_p_min"]
    true_max = arrays["true_p_max"]
    pred_min = arrays["pred_rf_p_min"]
    pred_max = arrays["pred_rf_p_max"]

    mean_min = np.full(
        len(true_min),
        train_mean[0],
        dtype=np.float64,
    )

    mean_max = np.full(
        len(true_max),
        train_mean[1],
        dtype=np.float64,
    )

    rows = []

    for model_name, pmin_pred, pmax_pred in [
        ("train_mean", mean_min, mean_max),
        ("random_forest", pred_min, pred_max),
    ]:
        rows.append(
            metric_row(
                eval_month,
                model_name,
                "p_min",
                true_min,
                pmin_pred,
            )
        )

        rows.append(
            metric_row(
                eval_month,
                model_name,
                "p_max",
                true_max,
                pmax_pred,
            )
        )

        rows.append(
            metric_row(
                eval_month,
                model_name,
                "p_span",
                true_max - true_min,
                pmax_pred - pmin_pred,
            )
        )

    distribution = pd.DataFrame(
        [
            target_distribution_row(
                eval_month,
                "p_min",
                true_min,
            ),
            target_distribution_row(
                eval_month,
                "p_max",
                true_max,
            ),
            target_distribution_row(
                eval_month,
                "p_span",
                true_max - true_min,
            ),
        ]
    )

    details = {
        "eval_month": int(eval_month),
        "eval_rows": int(total_rows),
        "raw_bound_inversion_rows": int(inversion_rows),
        "raw_bound_inversion_rate": float(
            inversion_rows / max(total_rows, 1)
        ),
    }

    return pd.DataFrame(rows), distribution, details


def plot_monthly_metric(
    metrics,
    metric_name,
    out_path,
):
    rf = metrics.loc[
        metrics["model"].eq("random_forest")
        & metrics["target"].isin(["p_min", "p_max"])
    ].copy()

    if rf.empty:
        return

    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(1, 1, 1)

    for target in ["p_min", "p_max"]:
        g = (
            rf.loc[rf["target"].eq(target)]
            .sort_values("eval_month")
        )

        if not g.empty:
            ax.plot(
                g["eval_month"],
                g[metric_name],
                marker="o",
                linewidth=2,
                label=target,
            )

    ax.set_xlabel("Evaluation month")
    ax.set_ylabel(metric_name)
    ax.set_xticks(
        sorted(
            rf["eval_month"].unique().tolist()
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(
        f"2025 expanding-window monthly validation: {metric_name}"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


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
        "--dataset-dir",
        default="final_feature_only_dataset",
    )

    ap.add_argument(
        "--start-test-month",
        type=int,
        default=7,
    )

    ap.add_argument(
        "--end-test-month",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=300_000,
    )

    ap.add_argument(
        "--rf-trees",
        type=int,
        default=128,
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
        default=0.5,
    )

    ap.add_argument(
        "--n-jobs",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = ap.parse_args()

    if not (
        2 <= args.start_test_month <= 12
        and args.start_test_month <= args.end_test_month <= 12
    ):
        raise ValueError(
            "Require 2 <= start-test-month <= end-test-month <= 12."
        )

    base = (
        Path(args.root)
        / str(args.year)
    )

    dataset = (
        base
        / args.dataset_dir
    )

    manifest_path = (
        dataset
        / "manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    features = list(
        manifest["model_features"]
    )

    if len(features) != 83:
        print(
            f"[warning] model feature count is {len(features)}, "
            f"not 83. The script will use the manifest exactly as-is.",
            flush=True,
        )

    files = all_part_files(
        manifest
    )

    out = (
        base
        / "monthly_rolling_price_bounds"
    )

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} exists. Use --overwrite."
            )
        shutil.rmtree(out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        f"2025 monthly rolling bid-price-bound validation - "
        f"{args.year}"
    )
    print("=" * 100)
    print(f"Feature count = {len(features)}")
    print(
        "Model = same RandomForestRegressor as the current "
        "simple bounds experiment"
    )
    print(
        f"Monthly folds = {args.start_test_month:02d} "
        f"through {args.end_test_month:02d}"
    )
    print(
        f"Max TRAIN rows per fold = {args.max_train_rows:,}"
    )
    print()

    metric_tables = []
    distribution_tables = []
    fold_rows = []

    for eval_month in range(
        args.start_test_month,
        args.end_test_month + 1,
    ):
        print()
        print("#" * 100)
        print(
            f"FOLD: train months 01-{eval_month - 1:02d} "
            f"-> evaluate month {eval_month:02d}"
        )
        print("#" * 100)

        train, eligible_train_rows = build_train_reservoir(
            dataset=dataset,
            part_files=files,
            features=features,
            eval_month=eval_month,
            max_train_rows=args.max_train_rows,
            seed=args.seed,
        )

        X_train = train[features]

        y_train = train[
            [
                "_target_p_min",
                "_target_p_max",
            ]
        ].to_numpy(np.float32)

        train_mean = np.mean(
            y_train,
            axis=0,
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

        print(
            f"[fit month {eval_month:02d}] "
            f"sampled TRAIN rows = {len(train):,} "
            f"from eligible historical rows = {eligible_train_rows:,}",
            flush=True,
        )

        model.fit(
            X_train,
            y_train,
        )

        metrics, distribution, details = evaluate_month(
            dataset=dataset,
            part_files=files,
            features=features,
            eval_month=eval_month,
            model=model,
            train_mean=train_mean,
        )

        metrics["train_month_end"] = eval_month - 1
        metrics["sampled_train_rows"] = len(train)
        metrics["eligible_train_rows"] = eligible_train_rows

        distribution["train_month_end"] = eval_month - 1

        metric_tables.append(metrics)
        distribution_tables.append(distribution)

        rf_min = metrics.loc[
            metrics["model"].eq("random_forest")
            & metrics["target"].eq("p_min")
        ].iloc[0]

        rf_max = metrics.loc[
            metrics["model"].eq("random_forest")
            & metrics["target"].eq("p_max")
        ].iloc[0]

        mean_min = metrics.loc[
            metrics["model"].eq("train_mean")
            & metrics["target"].eq("p_min")
        ].iloc[0]

        mean_max = metrics.loc[
            metrics["model"].eq("train_mean")
            & metrics["target"].eq("p_max")
        ].iloc[0]

        dist_min = distribution.loc[
            distribution["target"].eq("p_min")
        ].iloc[0]

        dist_max = distribution.loc[
            distribution["target"].eq("p_max")
        ].iloc[0]

        fold_rows.append(
            {
                "eval_month": eval_month,
                "train_month_end": eval_month - 1,
                "sampled_train_rows": int(len(train)),
                "eligible_train_rows": int(eligible_train_rows),
                "eval_rows": details["eval_rows"],
                "p_min_true_mean": dist_min["mean"],
                "p_min_true_median": dist_min["p50"],
                "p_min_true_p90": dist_min["p90"],
                "p_max_true_mean": dist_max["mean"],
                "p_max_true_median": dist_max["p50"],
                "p_max_true_p90": dist_max["p90"],
                "p_min_rf_mae": rf_min["mae"],
                "p_min_rf_wape_pct": rf_min["wape_pct"],
                "p_min_rf_r2": rf_min["r2"],
                "p_max_rf_mae": rf_max["mae"],
                "p_max_rf_wape_pct": rf_max["wape_pct"],
                "p_max_rf_r2": rf_max["r2"],
                "p_min_train_mean_wape_pct": mean_min["wape_pct"],
                "p_max_train_mean_wape_pct": mean_max["wape_pct"],
                "raw_bound_inversion_rate": details[
                    "raw_bound_inversion_rate"
                ],
            }
        )

        del (
            train,
            X_train,
            y_train,
            train_mean,
            model,
            metrics,
            distribution,
        )
        gc.collect()

    all_metrics = pd.concat(
        metric_tables,
        ignore_index=True,
    )

    all_distribution = pd.concat(
        distribution_tables,
        ignore_index=True,
    )

    monthly_summary = pd.DataFrame(
        fold_rows
    ).sort_values(
        "eval_month"
    ).reset_index(
        drop=True
    )

    all_metrics.to_csv(
        out / "monthly_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_distribution.to_csv(
        out / "monthly_target_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    monthly_summary.to_csv(
        out / "monthly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_monthly_metric(
        all_metrics,
        "wape_pct",
        out / "monthly_wape.png",
    )

    plot_monthly_metric(
        all_metrics,
        "mae",
        out / "monthly_mae.png",
    )

    plot_monthly_metric(
        all_metrics,
        "r2",
        out / "monthly_r2.png",
    )

    lines = [
        f"Monthly rolling bid-price-bound validation - {args.year}",
        "=" * 110,
        "",
        (
            f"Frozen input features = {len(features)} "
            f"(used exactly from final_feature_only_dataset manifest)"
        ),
        (
            "Frozen model = RandomForestRegressor("
            f"n_estimators={args.rf_trees}, "
            f"max_depth={args.rf_max_depth}, "
            f"min_samples_leaf={args.rf_min_leaf}, "
            f"max_features={args.rf_max_features})"
        ),
        (
            f"TRAIN sample cap per fold = {args.max_train_rows:,}"
        ),
        "",
        "EXPANDING-WINDOW RESULTS",
        "-" * 110,
    ]

    for _, r in monthly_summary.iterrows():
        lines.append(
            f"Train 01-{int(r['train_month_end']):02d} "
            f"-> Eval {int(r['eval_month']):02d} | "
            f"p_min MAE={r['p_min_rf_mae']:.3f}, "
            f"WAPE={r['p_min_rf_wape_pct']:.3f}%, "
            f"R2={r['p_min_rf_r2']:.4f} | "
            f"p_max MAE={r['p_max_rf_mae']:.3f}, "
            f"WAPE={r['p_max_rf_wape_pct']:.3f}%, "
            f"R2={r['p_max_rf_r2']:.4f} | "
            f"true median p_min/p_max="
            f"{r['p_min_true_median']:.3f}/"
            f"{r['p_max_true_median']:.3f}"
        )

    lines.extend(
        [
            "",
            "MONTHLY TARGET LEVELS",
            "-" * 110,
        ]
    )

    for _, r in monthly_summary.iterrows():
        lines.append(
            f"Month {int(r['eval_month']):02d}: "
            f"p_min mean/median/P90="
            f"{r['p_min_true_mean']:.3f}/"
            f"{r['p_min_true_median']:.3f}/"
            f"{r['p_min_true_p90']:.3f}; "
            f"p_max mean/median/P90="
            f"{r['p_max_true_mean']:.3f}/"
            f"{r['p_max_true_median']:.3f}/"
            f"{r['p_max_true_p90']:.3f}"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION RULE",
            "-" * 110,
            (
                "Do not change features or model from this script alone. "
                "First inspect whether monthly WAPE/MAE deterioration moves "
                "together with monthly p_min/p_max level changes."
            ),
            (
                "If errors are low in some months and rise sharply when target "
                "price levels shift, temporal/seasonal state coverage is a "
                "credible problem."
            ),
            (
                "If errors remain similarly large across most months, the "
                "current 83-feature representation itself is insufficient for "
                "accurate price-bound prediction."
            ),
            "",
            f"Outputs: {out}",
        ]
    )

    summary = "\n".join(lines)

    (
        out / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)


if __name__ == "__main__":
    main()
