#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11d_diagnose_monthly_price_bound_errors.py

Diagnostic only. Do NOT change features, targets, model or rolling scheme.

Same experiment as 11c:
    83 frozen feature-only inputs
        -> RandomForestRegressor
        -> [p_min, p_max]

Monthly expanding-window:
    Jan-Jun -> Jul
    Jan-Jul -> Aug
    ...
    Jan-Nov -> Dec

This script answers only:
1) Does each month show systematic over/under prediction?
2) Are absolute errors concentrated in a small fraction of rows?
3) Are large errors concentrated in rows whose current price bounds changed
   strongly relative to the available lag-1 reference curve?

The lag-1 curve is used ONLY for diagnostics, never as a model input.

Outputs
-------
data/processed/bidprediction/<year>/monthly_price_bound_error_diagnostics/
    monthly_error_summary.csv
    monthly_error_quantiles.csv
    monthly_error_concentration.csv
    monthly_error_by_change_bin.csv
    summary.txt

Run
---
python scripts/bidprediction/11d_diagnose_monthly_price_bound_errors.py --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
CURVE_COLS = [*SHAPE_COLS, "p_anchor", "p_span"]

REF_PRICE_COLS = [
    f"reference_curvevec_p{i:02d}_lag1"
    for i in range(21)
]


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
    if "local_date" in d.columns:
        dt = pd.to_datetime(d["local_date"], errors="coerce")
    elif "timestamp_local" in d.columns:
        dt = pd.to_datetime(d["timestamp_local"], errors="coerce")
    else:
        raise KeyError(
            "Neither local_date nor timestamp_local is available."
        )

    return dt.dt.month.to_numpy(dtype=np.float64)


def compute_price_bounds(d):
    missing = [c for c in CURVE_COLS if c not in d.columns]

    if missing:
        raise KeyError(
            "Missing curve columns: "
            + ", ".join(missing)
        )

    shape = numeric_frame(
        d,
        SHAPE_COLS,
    ).to_numpy(np.float64)

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

    price = (
        p_anchor[:, None]
        + p_span[:, None] * shape
    )

    valid = np.isfinite(price).all(axis=1)

    p_min = np.full(
        len(d),
        np.nan,
        dtype=np.float64,
    )

    p_max = np.full(
        len(d),
        np.nan,
        dtype=np.float64,
    )

    if valid.any():
        p_min[valid] = np.min(
            price[valid, :],
            axis=1,
        )

        p_max[valid] = np.max(
            price[valid, :],
            axis=1,
        )

    return p_min, p_max, valid


def reference_bounds(d):
    if not all(
        c in d.columns
        for c in REF_PRICE_COLS
    ):
        return (
            np.full(len(d), np.nan),
            np.full(len(d), np.nan),
            np.zeros(len(d), dtype=bool),
        )

    p = numeric_frame(
        d,
        REF_PRICE_COLS,
    ).to_numpy(np.float64)

    valid = np.isfinite(p).all(axis=1)

    p_min = np.full(
        len(d),
        np.nan,
        dtype=np.float64,
    )

    p_max = np.full(
        len(d),
        np.nan,
        dtype=np.float64,
    )

    if valid.any():
        p_min[valid] = np.min(
            p[valid, :],
            axis=1,
        )

        p_max[valid] = np.max(
            p[valid, :],
            axis=1,
        )

    return p_min, p_max, valid


def build_train_reservoir(
    dataset,
    files,
    features,
    eval_month,
    max_train_rows,
    seed,
):
    rng = np.random.default_rng(
        seed + eval_month * 10007
    )

    reservoir = None
    eligible_rows = 0

    for i, rel in enumerate(files, 1):
        path = dataset / rel

        print(
            f"[month {eval_month:02d} TRAIN "
            f"{i}/{len(files)}] {path.name}",
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
            del (
                d,
                months,
                time_mask,
                p_min,
                p_max,
                curve_valid,
                valid,
                idx,
            )
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
            keep = np.argpartition(
                reservoir["_priority"].to_numpy(
                    np.float64
                ),
                max_train_rows - 1,
            )[:max_train_rows]

            reservoir = (
                reservoir.iloc[keep]
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
            f"No valid training rows before month {eval_month}."
        )

    reservoir = (
        reservoir
        .sort_values(
            "_priority",
            kind="mergesort",
        )
        .head(max_train_rows)
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )

    return reservoir, eligible_rows


def collect_eval_month(
    dataset,
    files,
    features,
    eval_month,
    model,
):
    blocks = []

    for i, rel in enumerate(files, 1):
        path = dataset / rel

        print(
            f"[month {eval_month:02d} EVAL "
            f"{i}/{len(files)}] {path.name}",
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

        ref_min, ref_max, ref_valid = reference_bounds(d)

        valid = (
            month_mask
            & curve_valid
            & np.isfinite(p_min)
            & np.isfinite(p_max)
        )

        idx = np.flatnonzero(valid)

        if len(idx) == 0:
            del (
                d,
                months,
                month_mask,
                p_min,
                p_max,
                curve_valid,
                ref_min,
                ref_max,
                ref_valid,
                valid,
                idx,
            )
            gc.collect()
            continue

        X = numeric_frame(
            d.iloc[idx],
            features,
        )

        pred = model.predict(
            X
        ).astype(np.float64)

        block = pd.DataFrame(
            {
                "true_p_min": p_min[idx],
                "true_p_max": p_max[idx],
                "pred_p_min": pred[:, 0],
                "pred_p_max": pred[:, 1],
                "ref_p_min": ref_min[idx],
                "ref_p_max": ref_max[idx],
                "ref_valid": ref_valid[idx],
            }
        )

        block["err_p_min"] = (
            block["pred_p_min"]
            - block["true_p_min"]
        )

        block["err_p_max"] = (
            block["pred_p_max"]
            - block["true_p_max"]
        )

        block["ae_p_min"] = np.abs(
            block["err_p_min"]
        )

        block["ae_p_max"] = np.abs(
            block["err_p_max"]
        )

        block["delta_p_min"] = (
            block["true_p_min"]
            - block["ref_p_min"]
        )

        block["delta_p_max"] = (
            block["true_p_max"]
            - block["ref_p_max"]
        )

        block["abs_delta_p_min"] = np.abs(
            block["delta_p_min"]
        )

        block["abs_delta_p_max"] = np.abs(
            block["delta_p_max"]
        )

        blocks.append(block)

        del (
            d,
            months,
            month_mask,
            p_min,
            p_max,
            curve_valid,
            ref_min,
            ref_max,
            ref_valid,
            valid,
            idx,
            X,
            pred,
            block,
        )

        gc.collect()

    if not blocks:
        raise RuntimeError(
            f"No valid rows for evaluation month {eval_month}."
        )

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def summary_row(
    month,
    target,
    true,
    pred,
):
    true = np.asarray(
        true,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(true)
        & np.isfinite(pred)
    )

    true = true[mask]
    pred = pred[mask]

    err = pred - true
    ae = np.abs(err)

    return {
        "eval_month": int(month),
        "target": target,
        "rows": int(len(true)),
        "true_mean": float(np.mean(true)),
        "true_median": float(np.median(true)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "signed_bias_mean": float(np.mean(err)),
        "signed_bias_median": float(np.median(err)),
        "underprediction_rate": float(
            np.mean(err < 0.0)
        ),
        "overprediction_rate": float(
            np.mean(err > 0.0)
        ),
        "mae": float(np.mean(ae)),
        "wape_pct": float(
            100.0
            * np.sum(ae)
            / max(
                np.sum(np.abs(true)),
                1e-12,
            )
        ),
    }


def quantile_row(
    month,
    target,
    ae,
):
    ae = np.asarray(
        ae,
        dtype=np.float64,
    )

    ae = ae[
        np.isfinite(ae)
    ]

    q = np.quantile(
        ae,
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
    )

    return {
        "eval_month": int(month),
        "target": target,
        "rows": int(len(ae)),
        "ae_p50": float(q[0]),
        "ae_p75": float(q[1]),
        "ae_p90": float(q[2]),
        "ae_p95": float(q[3]),
        "ae_p99": float(q[4]),
        "ae_max": float(np.max(ae)),
    }


def concentration_rows(
    month,
    target,
    ae,
):
    ae = np.asarray(
        ae,
        dtype=np.float64,
    )

    ae = ae[
        np.isfinite(ae)
    ]

    ae = np.sort(ae)[::-1]

    total = max(
        float(ae.sum()),
        1e-12,
    )

    rows = []

    for frac in [
        0.01,
        0.05,
        0.10,
        0.20,
        0.50,
    ]:
        n = max(
            1,
            int(
                math.ceil(
                    len(ae) * frac
                )
            ),
        )

        rows.append(
            {
                "eval_month": int(month),
                "target": target,
                "top_row_fraction": frac,
                "rows_in_top_group": int(n),
                "absolute_error_share": float(
                    ae[:n].sum()
                    / total
                ),
            }
        )

    return rows


def change_bin_rows(
    month,
    target,
    true,
    pred,
    abs_delta,
    ref_valid,
):
    true = np.asarray(
        true,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    abs_delta = np.asarray(
        abs_delta,
        dtype=np.float64,
    )

    ref_valid = np.asarray(
        ref_valid,
        dtype=bool,
    )

    valid = (
        ref_valid
        & np.isfinite(true)
        & np.isfinite(pred)
        & np.isfinite(abs_delta)
    )

    true = true[valid]
    pred = pred[valid]
    abs_delta = abs_delta[valid]

    if len(true) == 0:
        return []

    # Quantile bins of |current bound - lag1 reference bound|.
    # These are diagnostic only and are calculated independently within month.
    quantiles = np.quantile(
        abs_delta,
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
    )

    edges = [
        -np.inf,
        quantiles[0],
        quantiles[1],
        quantiles[2],
        quantiles[3],
        quantiles[4],
        np.inf,
    ]

    labels = [
        "Q00-50",
        "Q50-75",
        "Q75-90",
        "Q90-95",
        "Q95-99",
        "Q99-100",
    ]

    bin_id = pd.cut(
        abs_delta,
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    d = pd.DataFrame(
        {
            "true": true,
            "pred": pred,
            "abs_delta": abs_delta,
            "change_bin": bin_id,
        }
    )

    rows = []

    for label, g in d.groupby(
        "change_bin",
        observed=True,
        sort=True,
    ):
        err = (
            g["pred"].to_numpy(np.float64)
            - g["true"].to_numpy(np.float64)
        )

        ae = np.abs(err)

        rows.append(
            {
                "eval_month": int(month),
                "target": target,
                "change_bin": str(label),
                "rows": int(len(g)),
                "abs_change_mean": float(
                    g["abs_delta"].mean()
                ),
                "abs_change_median": float(
                    g["abs_delta"].median()
                ),
                "signed_bias_mean": float(
                    np.mean(err)
                ),
                "mae": float(
                    np.mean(ae)
                ),
                "wape_pct": float(
                    100.0
                    * np.sum(ae)
                    / max(
                        np.sum(
                            np.abs(
                                g["true"].to_numpy(
                                    np.float64
                                )
                            )
                        ),
                        1e-12,
                    )
                ),
            }
        )

    return rows


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

    base = (
        Path(args.root)
        / str(args.year)
    )

    dataset = (
        base
        / args.dataset_dir
    )

    manifest = json.loads(
        (
            dataset
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    features = list(
        manifest["model_features"]
    )

    files = all_part_files(
        manifest
    )

    out = (
        base
        / "monthly_price_bound_error_diagnostics"
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
        f"11d monthly price-bound error diagnosis - "
        f"{args.year}"
    )
    print("=" * 100)
    print(
        f"Frozen feature count = {len(features)}"
    )
    print(
        "Frozen model and rolling scheme are identical to 11c."
    )
    print(
        "Lag-1 price bounds are diagnostic-only and are NOT model inputs."
    )
    print()

    summary_rows = []
    quantile_rows = []
    concentration = []
    change_rows = []

    for eval_month in range(
        args.start_test_month,
        args.end_test_month + 1,
    ):
        print()
        print("#" * 100)
        print(
            f"Train 01-{eval_month - 1:02d} "
            f"-> evaluate {eval_month:02d}"
        )
        print("#" * 100)

        train, eligible_train_rows = build_train_reservoir(
            dataset=dataset,
            files=files,
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
            f"[fit] sampled TRAIN={len(train):,}, "
            f"eligible history={eligible_train_rows:,}",
            flush=True,
        )

        model.fit(
            X_train,
            y_train,
        )

        d = collect_eval_month(
            dataset=dataset,
            files=files,
            features=features,
            eval_month=eval_month,
            model=model,
        )

        for target in [
            "p_min",
            "p_max",
        ]:
            true = d[
                f"true_{target}"
            ].to_numpy(np.float64)

            pred = d[
                f"pred_{target}"
            ].to_numpy(np.float64)

            ae = d[
                f"ae_{target}"
            ].to_numpy(np.float64)

            abs_delta = d[
                f"abs_delta_{target}"
            ].to_numpy(np.float64)

            summary_rows.append(
                {
                    **summary_row(
                        eval_month,
                        target,
                        true,
                        pred,
                    ),
                    "sampled_train_rows": int(
                        len(train)
                    ),
                    "eligible_train_rows": int(
                        eligible_train_rows
                    ),
                    "lag1_reference_coverage": float(
                        d["ref_valid"].mean()
                    ),
                }
            )

            quantile_rows.append(
                quantile_row(
                    eval_month,
                    target,
                    ae,
                )
            )

            concentration.extend(
                concentration_rows(
                    eval_month,
                    target,
                    ae,
                )
            )

            change_rows.extend(
                change_bin_rows(
                    eval_month,
                    target,
                    true,
                    pred,
                    abs_delta,
                    d[
                        "ref_valid"
                    ].to_numpy(bool),
                )
            )

        del (
            train,
            X_train,
            y_train,
            model,
            d,
        )

        gc.collect()

    monthly = pd.DataFrame(
        summary_rows
    )

    quantiles = pd.DataFrame(
        quantile_rows
    )

    concentration_df = pd.DataFrame(
        concentration
    )

    change_df = pd.DataFrame(
        change_rows
    )

    monthly.to_csv(
        out
        / "monthly_error_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quantiles.to_csv(
        out
        / "monthly_error_quantiles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    concentration_df.to_csv(
        out
        / "monthly_error_concentration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    change_df.to_csv(
        out
        / "monthly_error_by_change_bin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        f"11d monthly price-bound error diagnosis - {args.year}",
        "=" * 112,
        "",
        "MONTHLY BIAS / ERROR",
        "-" * 112,
    ]

    for month in range(
        args.start_test_month,
        args.end_test_month + 1,
    ):
        for target in [
            "p_min",
            "p_max",
        ]:
            r = monthly.loc[
                monthly["eval_month"].eq(month)
                & monthly["target"].eq(target)
            ].iloc[0]

            q = quantiles.loc[
                quantiles["eval_month"].eq(month)
                & quantiles["target"].eq(target)
            ].iloc[0]

            c10 = concentration_df.loc[
                concentration_df["eval_month"].eq(month)
                & concentration_df["target"].eq(target)
                & np.isclose(
                    concentration_df[
                        "top_row_fraction"
                    ],
                    0.10,
                )
            ].iloc[0]

            lines.append(
                f"Month {month:02d} {target}: "
                f"bias={r['signed_bias_mean']:+.3f}, "
                f"under={100.0*r['underprediction_rate']:.2f}%, "
                f"over={100.0*r['overprediction_rate']:.2f}%, "
                f"MAE={r['mae']:.3f}, "
                f"WAPE={r['wape_pct']:.3f}%, "
                f"AE P50/P90/P95/P99="
                f"{q['ae_p50']:.3f}/"
                f"{q['ae_p90']:.3f}/"
                f"{q['ae_p95']:.3f}/"
                f"{q['ae_p99']:.3f}, "
                f"top10% error share="
                f"{100.0*c10['absolute_error_share']:.2f}%"
            )

    lines.extend(
        [
            "",
            "ERROR VS LAG-1 BOUND CHANGE",
            "-" * 112,
            (
                "Q00-50 ... Q99-100 are monthly quantile bins of "
                "|current bound - lag1 reference bound|."
            ),
        ]
    )

    for month in range(
        args.start_test_month,
        args.end_test_month + 1,
    ):
        for target in [
            "p_min",
            "p_max",
        ]:
            g = change_df.loc[
                change_df["eval_month"].eq(month)
                & change_df["target"].eq(target)
            ].copy()

            if g.empty:
                lines.append(
                    f"Month {month:02d} {target}: "
                    f"no usable lag1 reference."
                )
                continue

            text_parts = []

            for _, r in g.iterrows():
                text_parts.append(
                    f"{r['change_bin']}: "
                    f"Δ={r['abs_change_median']:.2f}, "
                    f"MAE={r['mae']:.2f}, "
                    f"WAPE={r['wape_pct']:.2f}%"
                )

            lines.append(
                f"Month {month:02d} {target} | "
                + " | ".join(text_parts)
            )

    lines.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 112,
            (
                "1. signed_bias_mean < 0 means systematic underprediction; "
                "> 0 means overprediction."
            ),
            (
                "2. If top 10% rows still contribute roughly half of total "
                "absolute error, average WAPE is being strongly driven by a "
                "small difficult subset."
            ),
            (
                "3. If MAE/WAPE rises sharply from Q00-50 to Q95-99/Q99-100, "
                "the main failure is concentrated in sudden bound changes."
            ),
            (
                "4. If error stays large even in Q00-50, the current 83 "
                "features are insufficient even for relatively stable rows."
            ),
            "",
            f"Outputs: {out}",
        ]
    )

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


if __name__ == "__main__":
    main()
