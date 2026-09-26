#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11b_diagnose_price_bound_failure.py

Purpose
-------
Diagnose WHY the current 83-feature Random Forest still predicts p_min / p_max
poorly. This script does NOT train a new model and does NOT change the task.

It checks only five things:
1) Are each participant's price bounds intrinsically stable?
   -> compare global train-mean vs participant train-mean / train-median baselines.
2) Which of the 83 features are actually related to p_min / p_max?
   -> RF feature importance + TRAIN Spearman correlation.
3) How much do price-bound distributions drift from VAL to TEST?
4) Are TEST errors widespread, or concentrated in a few participants / high-price rows?
5) Did the compressed strategy features lose absolute price-level information?
   -> compare the current RF against simple participant historical price statistics.

Inputs
------
data/processed/bidprediction/<year>/final_feature_only_dataset/
data/processed/bidprediction/<year>/price_bounds_validation/random_forest_bounds.joblib

Outputs
-------
data/processed/bidprediction/<year>/price_bounds_diagnostics/
    01_participant_train_stability.csv
    01_history_baseline_metrics.csv
    02_feature_relevance.csv
    03_target_distribution_drift.csv
    04_test_error_by_price_bin.csv
    04_test_error_by_participant.csv
    04_test_error_concentration.csv
    summary.txt

Run
---
python scripts/bidprediction/11b_diagnose_price_bound_failure.py --year 2025 --overwrite
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

from sklearn.metrics import r2_score


SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
CURVE_COLS = [*SHAPE_COLS, "p_anchor", "p_span"]


def part_files(manifest, split):
    vals = manifest.get("parts", {}).get(split, [])
    return [
        x["file"] if isinstance(x, dict) else x
        for x in vals
    ]


def numeric_frame(d, cols):
    return d[cols].apply(pd.to_numeric, errors="coerce")


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


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "wape_pct": np.nan,
            "r2": np.nan,
        }

    e = y_pred - y_true
    ae = np.abs(e)

    return {
        "rows": int(len(y_true)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "wape_pct": float(
            100.0
            * np.sum(ae)
            / max(np.sum(np.abs(y_true)), 1e-12)
        ),
        "r2": float(r2_score(y_true, y_pred))
        if len(y_true) >= 2
        else np.nan,
    }


def target_distribution_row(split, target, x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    qs = np.quantile(
        x,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )

    return {
        "split": split,
        "target": target,
        "rows": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p10": float(qs[2]),
        "p25": float(qs[3]),
        "p50": float(qs[4]),
        "p75": float(qs[5]),
        "p90": float(qs[6]),
        "p95": float(qs[7]),
        "p99": float(qs[8]),
        "max": float(np.max(x)),
    }


def choose_feature_sample_indices(n, take, seed):
    if n <= take:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    return rng.choice(
        n,
        size=take,
        replace=False,
    )


def read_train(
    dataset,
    manifest,
    features,
    feature_sample_rows,
    seed,
):
    files = part_files(manifest, "train")
    if not files:
        raise RuntimeError("No TRAIN parts found.")

    compact_blocks = []
    feature_blocks = []

    per_part_sample = max(
        1,
        int(math.ceil(feature_sample_rows / len(files))),
    )

    for i, rel in enumerate(files, 1):
        path = dataset / rel
        print(
            f"[TRAIN {i}/{len(files)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)
        if d.empty:
            continue

        p_min, p_max, valid = compute_price_bounds(d)

        idx_valid = np.flatnonzero(valid)
        if len(idx_valid) == 0:
            continue

        participant = (
            d["participant_id"].astype(str).to_numpy()
            if "participant_id" in d.columns
            else np.asarray(["UNKNOWN"] * len(d))
        )

        compact_blocks.append(
            pd.DataFrame(
                {
                    "participant_id": participant[idx_valid],
                    "p_min": p_min[idx_valid].astype(np.float32),
                    "p_max": p_max[idx_valid].astype(np.float32),
                }
            )
        )

        take_idx_local = choose_feature_sample_indices(
            len(idx_valid),
            min(per_part_sample, len(idx_valid)),
            seed + i * 911,
        )
        take_idx = idx_valid[take_idx_local]

        f = numeric_frame(
            d.iloc[take_idx],
            features,
        ).reset_index(drop=True)

        f["_target_p_min"] = p_min[take_idx]
        f["_target_p_max"] = p_max[take_idx]

        feature_blocks.append(f)

        del d, p_min, p_max, valid, idx_valid, participant, f
        gc.collect()

    if not compact_blocks:
        raise RuntimeError("No valid TRAIN rows found.")

    compact = pd.concat(
        compact_blocks,
        ignore_index=True,
    )

    feature_sample = pd.concat(
        feature_blocks,
        ignore_index=True,
    )

    if len(feature_sample) > feature_sample_rows:
        feature_sample = feature_sample.sample(
            n=feature_sample_rows,
            random_state=seed,
        ).reset_index(drop=True)

    return compact, feature_sample


def build_participant_stats(train_compact):
    grouped = train_compact.groupby(
        "participant_id",
        sort=False,
        observed=True,
    )

    stats = grouped.agg(
        rows=("p_min", "size"),
        p_min_mean=("p_min", "mean"),
        p_min_median=("p_min", "median"),
        p_min_std=("p_min", "std"),
        p_min_q25=("p_min", lambda x: x.quantile(0.25)),
        p_min_q75=("p_min", lambda x: x.quantile(0.75)),
        p_max_mean=("p_max", "mean"),
        p_max_median=("p_max", "median"),
        p_max_std=("p_max", "std"),
        p_max_q25=("p_max", lambda x: x.quantile(0.25)),
        p_max_q75=("p_max", lambda x: x.quantile(0.75)),
    ).reset_index()

    stats["p_min_iqr"] = (
        stats["p_min_q75"]
        - stats["p_min_q25"]
    )

    stats["p_max_iqr"] = (
        stats["p_max_q75"]
        - stats["p_max_q25"]
    )

    stats["p_min_abs_level"] = np.maximum(
        np.abs(stats["p_min_median"]),
        1.0,
    )

    stats["p_max_abs_level"] = np.maximum(
        np.abs(stats["p_max_median"]),
        1.0,
    )

    stats["p_min_iqr_to_level"] = (
        stats["p_min_iqr"]
        / stats["p_min_abs_level"]
    )

    stats["p_max_iqr_to_level"] = (
        stats["p_max_iqr"]
        / stats["p_max_abs_level"]
    )

    return stats


def build_feature_relevance(
    feature_sample,
    features,
    model_bundle,
):
    model = model_bundle["model"]

    rf = (
        model.named_steps["rf"]
        if hasattr(model, "named_steps")
        and "rf" in model.named_steps
        else model
    )

    importances = getattr(
        rf,
        "feature_importances_",
        np.full(
            len(features),
            np.nan,
        ),
    )

    rows = []

    y_min = pd.to_numeric(
        feature_sample["_target_p_min"],
        errors="coerce",
    )

    y_max = pd.to_numeric(
        feature_sample["_target_p_max"],
        errors="coerce",
    )

    for j, feature in enumerate(features):
        x = pd.to_numeric(
            feature_sample[feature],
            errors="coerce",
        )

        corr_min = x.corr(
            y_min,
            method="spearman",
        )

        corr_max = x.corr(
            y_max,
            method="spearman",
        )

        rows.append(
            {
                "feature": feature,
                "rf_importance": float(importances[j])
                if j < len(importances)
                and np.isfinite(importances[j])
                else np.nan,
                "spearman_p_min": corr_min,
                "abs_spearman_p_min": abs(corr_min)
                if pd.notna(corr_min)
                else np.nan,
                "spearman_p_max": corr_max,
                "abs_spearman_p_max": abs(corr_max)
                if pd.notna(corr_max)
                else np.nan,
                "max_abs_spearman": np.nanmax(
                    [
                        abs(corr_min)
                        if pd.notna(corr_min)
                        else np.nan,
                        abs(corr_max)
                        if pd.notna(corr_max)
                        else np.nan,
                    ]
                ),
            }
        )

    out = pd.DataFrame(rows)

    out = out.sort_values(
        [
            "rf_importance",
            "max_abs_spearman",
        ],
        ascending=[
            False,
            False,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return out


def eval_split(
    dataset,
    manifest,
    split,
    features,
    model_bundle,
    participant_stats,
    global_mean,
):
    model = model_bundle["model"]

    pstats = participant_stats.set_index(
        "participant_id"
    )

    arrays = {
        "true_p_min": [],
        "true_p_max": [],
        "pred_rf_p_min": [],
        "pred_rf_p_max": [],
        "pred_participant_mean_p_min": [],
        "pred_participant_mean_p_max": [],
        "pred_participant_median_p_min": [],
        "pred_participant_median_p_max": [],
        "participant_id": [],
    }

    for i, rel in enumerate(
        part_files(manifest, split),
        1,
    ):
        path = dataset / rel

        print(
            f"[{split.upper()} {i}/{len(part_files(manifest, split))}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)
        if d.empty:
            continue

        p_min, p_max, valid = compute_price_bounds(d)
        if not valid.any():
            continue

        d = d.loc[valid].copy().reset_index(drop=True)
        p_min = p_min[valid]
        p_max = p_max[valid]

        X = numeric_frame(d, features)
        pred_rf = model.predict(X).astype(np.float64)

        participant = (
            d["participant_id"].astype(str).to_numpy()
            if "participant_id" in d.columns
            else np.asarray(["UNKNOWN"] * len(d))
        )

        s = pd.Series(
            participant,
            name="participant_id",
        )

        mean_min = (
            s.map(pstats["p_min_mean"])
            .fillna(global_mean[0])
            .to_numpy(np.float64)
        )

        mean_max = (
            s.map(pstats["p_max_mean"])
            .fillna(global_mean[1])
            .to_numpy(np.float64)
        )

        median_min = (
            s.map(pstats["p_min_median"])
            .fillna(global_mean[0])
            .to_numpy(np.float64)
        )

        median_max = (
            s.map(pstats["p_max_median"])
            .fillna(global_mean[1])
            .to_numpy(np.float64)
        )

        arrays["true_p_min"].append(
            p_min.astype(np.float32)
        )
        arrays["true_p_max"].append(
            p_max.astype(np.float32)
        )
        arrays["pred_rf_p_min"].append(
            pred_rf[:, 0].astype(np.float32)
        )
        arrays["pred_rf_p_max"].append(
            pred_rf[:, 1].astype(np.float32)
        )
        arrays["pred_participant_mean_p_min"].append(
            mean_min.astype(np.float32)
        )
        arrays["pred_participant_mean_p_max"].append(
            mean_max.astype(np.float32)
        )
        arrays["pred_participant_median_p_min"].append(
            median_min.astype(np.float32)
        )
        arrays["pred_participant_median_p_max"].append(
            median_max.astype(np.float32)
        )
        arrays["participant_id"].append(participant)

        del (
            d,
            p_min,
            p_max,
            valid,
            X,
            pred_rf,
            participant,
            s,
            mean_min,
            mean_max,
            median_min,
            median_max,
        )
        gc.collect()

    for k, vals in arrays.items():
        if not vals:
            continue

        arrays[k] = np.concatenate(vals)

    return arrays


def history_baseline_table(
    split,
    arrays,
    global_mean,
):
    rows = []

    y_min = arrays["true_p_min"]
    y_max = arrays["true_p_max"]

    pred_global_min = np.full(
        len(y_min),
        global_mean[0],
        dtype=np.float64,
    )

    pred_global_max = np.full(
        len(y_max),
        global_mean[1],
        dtype=np.float64,
    )

    models = {
        "global_train_mean": (
            pred_global_min,
            pred_global_max,
        ),
        "participant_train_mean": (
            arrays["pred_participant_mean_p_min"],
            arrays["pred_participant_mean_p_max"],
        ),
        "participant_train_median": (
            arrays["pred_participant_median_p_min"],
            arrays["pred_participant_median_p_max"],
        ),
        "current_83feature_rf": (
            arrays["pred_rf_p_min"],
            arrays["pred_rf_p_max"],
        ),
    }

    for model_name, (pred_min, pred_max) in models.items():
        for target, yt, yp in [
            ("p_min", y_min, pred_min),
            ("p_max", y_max, pred_max),
        ]:
            m = metrics(yt, yp)
            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "target": target,
                    **m,
                }
            )

    return pd.DataFrame(rows)


def error_concentration_rows(
    target,
    y_true,
    y_pred,
):
    ae = np.abs(
        np.asarray(y_pred, dtype=np.float64)
        - np.asarray(y_true, dtype=np.float64)
    )

    ae = ae[np.isfinite(ae)]

    if len(ae) == 0:
        return []

    ae_sorted = np.sort(ae)[::-1]
    total = max(float(ae_sorted.sum()), 1e-12)

    rows = []

    for frac in [
        0.01,
        0.05,
        0.10,
        0.20,
        0.50,
    ]:
        k = max(
            1,
            int(
                math.ceil(
                    len(ae_sorted)
                    * frac
                )
            ),
        )

        rows.append(
            {
                "target": target,
                "top_row_fraction": frac,
                "rows": int(k),
                "absolute_error_share": float(
                    ae_sorted[:k].sum()
                    / total
                ),
            }
        )

    qs = np.quantile(
        ae,
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
    )

    rows.append(
        {
            "target": target,
            "top_row_fraction": np.nan,
            "rows": int(len(ae)),
            "absolute_error_share": np.nan,
            "error_p50": float(qs[0]),
            "error_p75": float(qs[1]),
            "error_p90": float(qs[2]),
            "error_p95": float(qs[3]),
            "error_p99": float(qs[4]),
        }
    )

    return rows


def error_by_price_bin(
    target,
    y_true,
    y_pred,
):
    d = pd.DataFrame(
        {
            "true": np.asarray(
                y_true,
                dtype=np.float64,
            ),
            "pred": np.asarray(
                y_pred,
                dtype=np.float64,
            ),
        }
    )

    d = d.loc[
        np.isfinite(d["true"])
        & np.isfinite(d["pred"])
    ].copy()

    if d.empty:
        return pd.DataFrame()

    try:
        d["price_bin"] = pd.qcut(
            d["true"],
            q=10,
            duplicates="drop",
        )
    except Exception:
        d["price_bin"] = "all"

    rows = []

    for bin_name, g in d.groupby(
        "price_bin",
        observed=True,
        sort=True,
    ):
        m = metrics(
            g["true"].to_numpy(),
            g["pred"].to_numpy(),
        )

        rows.append(
            {
                "target": target,
                "price_bin": str(bin_name),
                "true_mean": float(
                    g["true"].mean()
                ),
                "true_median": float(
                    g["true"].median()
                ),
                **m,
            }
        )

    return pd.DataFrame(rows)


def error_by_participant(arrays):
    d = pd.DataFrame(
        {
            "participant_id": arrays["participant_id"],
            "true_p_min": arrays["true_p_min"],
            "true_p_max": arrays["true_p_max"],
            "pred_p_min": arrays["pred_rf_p_min"],
            "pred_p_max": arrays["pred_rf_p_max"],
        }
    )

    d["ae_p_min"] = np.abs(
        d["pred_p_min"]
        - d["true_p_min"]
    )

    d["ae_p_max"] = np.abs(
        d["pred_p_max"]
        - d["true_p_max"]
    )

    g = d.groupby(
        "participant_id",
        observed=True,
        sort=False,
    )

    out = g.agg(
        rows=("participant_id", "size"),
        p_min_mae=("ae_p_min", "mean"),
        p_max_mae=("ae_p_max", "mean"),
        p_min_abs_error_sum=("ae_p_min", "sum"),
        p_max_abs_error_sum=("ae_p_max", "sum"),
        true_p_min_mean=("true_p_min", "mean"),
        true_p_max_mean=("true_p_max", "mean"),
    ).reset_index()

    total_min = max(
        float(out["p_min_abs_error_sum"].sum()),
        1e-12,
    )

    total_max = max(
        float(out["p_max_abs_error_sum"].sum()),
        1e-12,
    )

    out["p_min_error_share"] = (
        out["p_min_abs_error_sum"]
        / total_min
    )

    out["p_max_error_share"] = (
        out["p_max_abs_error_sum"]
        / total_max
    )

    out["combined_mae"] = (
        out["p_min_mae"]
        + out["p_max_mae"]
    ) / 2.0

    return out.sort_values(
        "combined_mae",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


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
        "--bounds-dir",
        default="price_bounds_validation",
    )

    ap.add_argument(
        "--feature-sample-rows",
        type=int,
        default=200_000,
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

    base = (
        Path(args.root)
        / str(args.year)
    )

    dataset = (
        base
        / args.dataset_dir
    )

    bounds_dir = (
        base
        / args.bounds_dir
    )

    manifest = json.loads(
        (
            dataset
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    model_bundle = joblib.load(
        bounds_dir
        / "random_forest_bounds.joblib"
    )

    features = list(
        model_bundle["features"]
    )

    out = (
        base
        / "price_bounds_diagnostics"
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
        f"11b diagnose price-bound prediction failure - "
        f"{args.year}"
    )
    print("=" * 100)
    print(
        "No new model is trained. "
        "Only diagnose the existing 83-feature RF."
    )
    print(
        f"Feature count = {len(features)}"
    )
    print()

    train_compact, feature_sample = read_train(
        dataset,
        manifest,
        features,
        args.feature_sample_rows,
        args.seed,
    )

    global_mean = np.asarray(
        [
            train_compact["p_min"].mean(),
            train_compact["p_max"].mean(),
        ],
        dtype=np.float64,
    )

    participant_stats = build_participant_stats(
        train_compact
    )

    participant_stats.to_csv(
        out
        / "01_participant_train_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_relevance = build_feature_relevance(
        feature_sample,
        features,
        model_bundle,
    )

    feature_relevance.to_csv(
        out
        / "02_feature_relevance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_arrays = {}
    history_tables = []
    drift_rows = [
        target_distribution_row(
            "train",
            "p_min",
            train_compact["p_min"].to_numpy(),
        ),
        target_distribution_row(
            "train",
            "p_max",
            train_compact["p_max"].to_numpy(),
        ),
    ]

    for split in [
        "val",
        "test",
    ]:
        arrays = eval_split(
            dataset,
            manifest,
            split,
            features,
            model_bundle,
            participant_stats,
            global_mean,
        )

        split_arrays[split] = arrays

        history_tables.append(
            history_baseline_table(
                split,
                arrays,
                global_mean,
            )
        )

        drift_rows.extend(
            [
                target_distribution_row(
                    split,
                    "p_min",
                    arrays["true_p_min"],
                ),
                target_distribution_row(
                    split,
                    "p_max",
                    arrays["true_p_max"],
                ),
            ]
        )

    history_metrics = pd.concat(
        history_tables,
        ignore_index=True,
    )

    history_metrics.to_csv(
        out
        / "01_history_baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    drift = pd.DataFrame(
        drift_rows
    )

    drift.to_csv(
        out
        / "03_target_distribution_drift.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test = split_arrays["test"]

    price_bins = pd.concat(
        [
            error_by_price_bin(
                "p_min",
                test["true_p_min"],
                test["pred_rf_p_min"],
            ),
            error_by_price_bin(
                "p_max",
                test["true_p_max"],
                test["pred_rf_p_max"],
            ),
        ],
        ignore_index=True,
    )

    price_bins.to_csv(
        out
        / "04_test_error_by_price_bin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    participant_error = error_by_participant(
        test
    )

    participant_error.to_csv(
        out
        / "04_test_error_by_participant.csv",
        index=False,
        encoding="utf-8-sig",
    )

    concentration = pd.DataFrame(
        [
            *error_concentration_rows(
                "p_min",
                test["true_p_min"],
                test["pred_rf_p_min"],
            ),
            *error_concentration_rows(
                "p_max",
                test["true_p_max"],
                test["pred_rf_p_max"],
            ),
        ]
    )

    concentration.to_csv(
        out
        / "04_test_error_concentration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------
    # Compact textual diagnosis
    # -------------------------

    stable_cohort = participant_stats.loc[
        participant_stats["rows"] >= 30
    ]

    pmin_iqr_med = float(
        stable_cohort["p_min_iqr"].median()
    )

    pmax_iqr_med = float(
        stable_cohort["p_max_iqr"].median()
    )

    pmin_rel_iqr_med = float(
        stable_cohort["p_min_iqr_to_level"].median()
    )

    pmax_rel_iqr_med = float(
        stable_cohort["p_max_iqr_to_level"].median()
    )

    def hm(split, model, target, col):
        q = history_metrics.loc[
            history_metrics["split"].eq(split)
            & history_metrics["model"].eq(model)
            & history_metrics["target"].eq(target),
            col,
        ]

        return float(q.iloc[0]) if not q.empty else np.nan

    def drift_value(split, target, col):
        q = drift.loc[
            drift["split"].eq(split)
            & drift["target"].eq(target),
            col,
        ]
        return float(q.iloc[0]) if not q.empty else np.nan

    top_features = feature_relevance.head(15)

    conc_lines = []
    for target in ["p_min", "p_max"]:
        q = concentration.loc[
            concentration["target"].eq(target)
            & concentration["top_row_fraction"].notna()
        ].copy()

        if not q.empty:
            q10 = q.loc[
                np.isclose(
                    q["top_row_fraction"],
                    0.10,
                )
            ]

            if not q10.empty:
                conc_lines.append(
                    f"{target}: top 10% rows contribute "
                    f"{100.0 * float(q10.iloc[0]['absolute_error_share']):.2f}% "
                    f"of total absolute error"
                )

    top10_part_min = float(
        participant_error.head(
            max(
                1,
                int(
                    math.ceil(
                        len(participant_error)
                        * 0.10
                    )
                ),
            )
        )["p_min_error_share"].sum()
    )

    top10_part_max = float(
        participant_error.head(
            max(
                1,
                int(
                    math.ceil(
                        len(participant_error)
                        * 0.10
                    )
                ),
            )
        )["p_max_error_share"].sum()
    )

    lines = [
        f"11b price-bound failure diagnosis - {args.year}",
        "=" * 100,
        "",
        "1) PARTICIPANT-LEVEL STABILITY",
        "-" * 100,
        f"Participants with >=30 TRAIN rows = {len(stable_cohort):,}",
        f"Median within-participant p_min IQR = {pmin_iqr_med:.4f}",
        f"Median within-participant p_max IQR = {pmax_iqr_med:.4f}",
        f"Median p_min IQR / |participant median level| = {pmin_rel_iqr_med:.4f}",
        f"Median p_max IQR / |participant median level| = {pmax_rel_iqr_med:.4f}",
        "",
        "Historical-statistic baselines on TEST:",
        (
            f"p_min WAPE: global_mean={hm('test','global_train_mean','p_min','wape_pct'):.3f}% | "
            f"participant_mean={hm('test','participant_train_mean','p_min','wape_pct'):.3f}% | "
            f"participant_median={hm('test','participant_train_median','p_min','wape_pct'):.3f}% | "
            f"83-feature RF={hm('test','current_83feature_rf','p_min','wape_pct'):.3f}%"
        ),
        (
            f"p_max WAPE: global_mean={hm('test','global_train_mean','p_max','wape_pct'):.3f}% | "
            f"participant_mean={hm('test','participant_train_mean','p_max','wape_pct'):.3f}% | "
            f"participant_median={hm('test','participant_train_median','p_max','wape_pct'):.3f}% | "
            f"83-feature RF={hm('test','current_83feature_rf','p_max','wape_pct'):.3f}%"
        ),
        "",
        "2) FEATURE RELEVANCE",
        "-" * 100,
    ]

    for _, r in top_features.iterrows():
        lines.append(
            f"{r['feature']}: "
            f"RF importance={r['rf_importance']:.6f}, "
            f"Spearman(p_min)={r['spearman_p_min']:.4f}, "
            f"Spearman(p_max)={r['spearman_p_max']:.4f}"
        )

    lines.extend(
        [
            "",
            "3) VAL -> TEST TARGET DISTRIBUTION DRIFT",
            "-" * 100,
            (
                f"p_min mean: {drift_value('val','p_min','mean'):.3f} -> "
                f"{drift_value('test','p_min','mean'):.3f}"
            ),
            (
                f"p_min median: {drift_value('val','p_min','p50'):.3f} -> "
                f"{drift_value('test','p_min','p50'):.3f}"
            ),
            (
                f"p_min P90: {drift_value('val','p_min','p90'):.3f} -> "
                f"{drift_value('test','p_min','p90'):.3f}"
            ),
            (
                f"p_max mean: {drift_value('val','p_max','mean'):.3f} -> "
                f"{drift_value('test','p_max','mean'):.3f}"
            ),
            (
                f"p_max median: {drift_value('val','p_max','p50'):.3f} -> "
                f"{drift_value('test','p_max','p50'):.3f}"
            ),
            (
                f"p_max P90: {drift_value('val','p_max','p90'):.3f} -> "
                f"{drift_value('test','p_max','p90'):.3f}"
            ),
            "",
            "4) TEST ERROR CONCENTRATION",
            "-" * 100,
            *conc_lines,
            (
                f"Top 10% highest-error participants contribute "
                f"{100.0 * top10_part_min:.2f}% of p_min absolute error "
                f"(participants sorted by combined MAE)"
            ),
            (
                f"Top 10% highest-error participants contribute "
                f"{100.0 * top10_part_max:.2f}% of p_max absolute error "
                f"(participants sorted by combined MAE)"
            ),
            "",
            "5) ABSOLUTE PRICE-LEVEL INFORMATION CHECK",
            "-" * 100,
            (
                "Compare participant_train_mean / participant_train_median "
                "against the current 83-feature RF above."
            ),
            (
                "If simple participant historical price statistics are much "
                "better than the RF, the current compressed feature set is "
                "missing participant-specific absolute price-level information."
            ),
            (
                "If the RF is already clearly better, the main issue is not "
                "simply missing participant price anchors; inspect drift and "
                "error concentration instead."
            ),
            "",
            f"Outputs: {out}",
        ]
    )

    summary = "\n".join(lines)

    (
        out
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)


if __name__ == "__main__":
    main()
