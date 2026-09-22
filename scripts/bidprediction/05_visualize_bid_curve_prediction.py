#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_visualize_bid_curve_prediction.py

Visualize current full-pipeline bid-curve predictions.

The script compares:
    Stage2 ground-truth 21-point bid curve
vs.
    current full-pipeline predicted 21-point bid curve

It produces two figures per selected sample:

1) Physical quantity curve
       x = MW
       y = $/MWh
   This shows the combined effect of price prediction + quantity-axis prediction.

2) Normalized quantity curve
       x = 0..1
       y = $/MWh
   This aligns both curves on the same normalized quantity grid and isolates
   price-level / price-shape differences.

Default case selection:
    scan a deterministic sample from the frozen TEST set,
    compute current full-pipeline curve MAE,
    and plot representative error quantiles:
        P10 / P50 / P75 / P90 / P95

You can also request exact sample IDs.

Examples
--------
# Representative cases
python scripts/bidprediction/05_visualize_bid_curve_prediction.py --year 2025

# Scan more TEST rows before selecting representative cases
python scripts/bidprediction/05_visualize_bid_curve_prediction.py \
    --year 2025 --scan-rows 20000

# Plot one exact sample
python scripts/bidprediction/05_visualize_bid_curve_prediction.py \
    --year 2025 \
    --sample-id "energy_market_offers_2025_11:123456"

Outputs
-------
data/processed/bidprediction/<year>/curve_visualization/
    selected_cases.csv
    case_*.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRID = np.linspace(0.0, 1.0, 21)


def load_reconstruction_module(script_dir: Path):
    """
    Reuse the exact prediction logic from the current 04c script.
    Prefer v10, then canonical unversioned file, then v9.
    """
    candidates = [
        script_dir / "04c_reconstruct_bid_curves_frozen_v10.py",
        script_dir / "04c_reconstruct_bid_curves_frozen.py",
        script_dir / "04c_reconstruct_bid_curves_frozen_v9.py",
    ]

    for path in candidates:
        if not path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            "bid_curve_reconstruct_runtime",
            path,
        )

        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        required = [
            "actual_curve",
            "predict_template",
            "v9_predict_curve",
            "norm",
            "TARGET_TEMPLATE",
            "ORIGIN_TEMPLATE",
            "TEMPLATES",
        ]

        if all(hasattr(module, x) for x in required):
            return module, path

    raise FileNotFoundError(
        "Cannot find a compatible 04c reconstruction script in "
        f"{script_dir}"
    )


def load_manifest(frozen: Path):
    p = frozen / "manifest.json"

    if not p.exists():
        raise FileNotFoundError(p)

    return json.loads(
        p.read_text(
            encoding="utf-8",
        )
    )


def rowwise_interp_to_true_q(
    q_true,
    q_pred,
    p_pred,
):
    """
    Interpolate each predicted curve onto the true physical quantity grid.
    This matches the final 04c price-evaluation logic conceptually.
    """
    q_true = np.asarray(
        q_true,
        dtype=float,
    )

    q_pred = np.asarray(
        q_pred,
        dtype=float,
    )

    p_pred = np.asarray(
        p_pred,
        dtype=float,
    )

    out = np.empty_like(
        q_true,
        dtype=float,
    )

    for i in range(
        len(q_true)
    ):
        order = np.argsort(
            q_pred[i],
            kind="mergesort",
        )

        q = q_pred[
            i,
            order,
        ]

        p = p_pred[
            i,
            order,
        ]

        uq, first = np.unique(
            q,
            return_index=True,
        )

        up = p[first]

        if len(uq) == 1:
            out[i, :] = up[0]
        else:
            out[i, :] = np.interp(
                q_true[i],
                uq,
                up,
                left=up[0],
                right=up[-1],
            )

    return out


def sample_test_rows(
    parts,
    target_rows,
    seed,
):
    """
    Deterministically sample across all TEST parts instead of taking only
    the first month/file.
    """
    if not parts:
        raise ValueError(
            "No frozen test_curve parts."
        )

    per_part = max(
        1,
        math.ceil(
            target_rows
            / len(parts)
        ),
    )

    blocks = []

    for i, p in enumerate(
        parts
    ):
        d = pd.read_pickle(
            p
        )

        if d.empty:
            continue

        n = min(
            per_part,
            len(d),
        )

        if n < len(d):
            d = d.sample(
                n=n,
                random_state=seed + i,
            )

        blocks.append(
            d
        )

    if not blocks:
        raise ValueError(
            "No rows loaded from frozen TEST."
        )

    out = pd.concat(
        blocks,
        ignore_index=True,
    )

    if len(out) > target_rows:
        out = out.sample(
            n=target_rows,
            random_state=seed,
        ).reset_index(
            drop=True
        )

    return out


def find_exact_sample(
    parts,
    sample_id,
):
    key = str(
        sample_id
    ).strip()

    for p in parts:
        d = pd.read_pickle(
            p
        )

        if d.empty:
            continue

        sid = (
            d[
                "sample_id"
            ]
            .astype("string")
            .str.strip()
        )

        hit = sid.eq(
            key
        )

        if hit.any():
            return (
                d.loc[
                    hit
                ]
                .head(1)
                .copy()
                .reset_index(
                    drop=True
                )
            )

    raise KeyError(
        f"sample_id not found in frozen TEST: {key}"
    )


def run_full_pipeline(
    d,
    runtime,
    parameter_bundle,
    template_bundle,
):
    (
        valid,
        q_true,
        p_true,
        true_template,
    ) = runtime.actual_curve(
        d
    )

    d = (
        d.loc[
            valid
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    q_true = q_true[
        valid
    ]

    p_true = p_true[
        valid
    ]

    true_template = true_template[
        valid
    ]

    if d.empty:
        raise ValueError(
            "No valid curve rows after actual_curve filtering."
        )

    origin_valid = (
        runtime.norm(
            d[
                runtime.ORIGIN_TEMPLATE
            ]
        )
        .isin(
            runtime.TEMPLATES
        )
        .to_numpy()
    )

    d = (
        d.loc[
            origin_valid
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    q_true = q_true[
        origin_valid
    ]

    p_true = p_true[
        origin_valid
    ]

    true_template = true_template[
        origin_valid
    ]

    if d.empty:
        raise ValueError(
            "No rows have a valid origin template."
        )

    pred_template, p_switch = (
        runtime.predict_template(
            d,
            template_bundle,
        )
    )

    pred_template_s = pd.Series(
        pred_template,
        index=d.index,
        dtype="string",
    )

    pred = runtime.v9_predict_curve(
        parameter_bundle,
        d,
        pred_template_s,
    )

    q_pred = np.asarray(
        pred[
            "q"
        ],
        dtype=float,
    )

    p_pred = np.asarray(
        pred[
            "p"
        ],
        dtype=float,
    )

    p_pred_on_true_q = (
        rowwise_interp_to_true_q(
            q_true,
            q_pred,
            p_pred,
        )
    )

    physical_curve_mae = (
        np.mean(
            np.abs(
                p_pred_on_true_q
                - p_true
            ),
            axis=1,
        )
    )

    normalized_price_mae = (
        np.mean(
            np.abs(
                p_pred
                - p_true
            ),
            axis=1,
        )
    )

    q_anchor_true = (
        q_true[
            :,
            0,
        ]
    )

    q_anchor_pred = (
        q_pred[
            :,
            0,
        ]
    )

    q_span_true = (
        q_true[
            :,
            -1,
        ]
        - q_true[
            :,
            0,
        ]
    )

    q_span_pred = (
        q_pred[
            :,
            -1,
        ]
        - q_pred[
            :,
            0,
        ]
    )

    meta = pd.DataFrame(
        {
            "sample_id": (
                d[
                    "sample_id"
                ]
                .astype("string")
                .to_numpy()
            ),
            "participant_id": (
                d[
                    "participant_id"
                ]
                .astype("string")
                .to_numpy()
            ),
            "local_date": (
                d[
                    "local_date"
                ]
                .astype(str)
                .to_numpy()
            ),
            "true_template_id": (
                true_template
            ),
            "pred_template_id": (
                pred_template
            ),
            "switch_probability": (
                p_switch
            ),
            "physical_curve_mae": (
                physical_curve_mae
            ),
            "normalized_price_mae": (
                normalized_price_mae
            ),
            "q_anchor_true_mw": (
                q_anchor_true
            ),
            "q_anchor_pred_mw": (
                q_anchor_pred
            ),
            "q_anchor_abs_error_mw": (
                np.abs(
                    q_anchor_pred
                    - q_anchor_true
                )
            ),
            "q_span_true_mw": (
                q_span_true
            ),
            "q_span_pred_mw": (
                q_span_pred
            ),
            "q_span_abs_error_mw": (
                np.abs(
                    q_span_pred
                    - q_span_true
                )
            ),
        }
    )

    return {
        "data": d,
        "meta": meta,
        "q_true": q_true,
        "p_true": p_true,
        "q_pred": q_pred,
        "p_pred": p_pred,
        "p_pred_on_true_q": (
            p_pred_on_true_q
        ),
    }


def quantile_cases(
    meta,
    quantiles,
):
    error = (
        meta[
            "physical_curve_mae"
        ]
        .to_numpy(float)
    )

    selected = []

    for q in quantiles:
        target = float(
            np.quantile(
                error,
                q,
            )
        )

        idx = int(
            np.argmin(
                np.abs(
                    error
                    - target
                )
            )
        )

        selected.append(
            (
                q,
                idx,
                target,
            )
        )

    # Prevent duplicate case rows when the error distribution has ties.
    unique = []
    seen = set()

    for item in selected:
        if item[1] in seen:
            continue

        seen.add(
            item[1]
        )

        unique.append(
            item
        )

    return unique


def safe_name(v):
    s = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(v),
    )

    return s[
        :100
    ]


def plot_physical_case(
    row,
    q_true,
    p_true,
    q_pred,
    p_pred,
    out_path,
):
    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.4,
        )
    )

    ax.plot(
        q_true,
        p_true,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="True curve",
    )

    ax.plot(
        q_pred,
        p_pred,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Predicted curve",
    )

    ax.set_xlabel(
        "Quantity (MW)"
    )

    ax.set_ylabel(
        "Bid price ($/MWh)"
    )

    ax.set_title(
        (
            f"{row['sample_id']} | "
            f"true={row['true_template_id']} "
            f"pred={row['pred_template_id']}\n"
            f"physical curve MAE="
            f"{row['physical_curve_mae']:.2f} $/MWh, "
            f"q_span error="
            f"{row['q_span_abs_error_mw']:.2f} MW"
        )
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_normalized_case(
    row,
    p_true,
    p_pred,
    out_path,
):
    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.4,
        )
    )

    ax.plot(
        GRID,
        p_true,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="True price",
    )

    ax.plot(
        GRID,
        p_pred,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Predicted price",
    )

    ax.set_xlabel(
        "Normalized quantity"
    )

    ax.set_ylabel(
        "Bid price ($/MWh)"
    )

    ax.set_title(
        (
            f"{row['sample_id']} | normalized quantity comparison\n"
            f"price-only MAE="
            f"{row['normalized_price_mae']:.2f} $/MWh"
        )
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

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
        default=(
            "data/processed/"
            "bidprediction"
        ),
    )

    ap.add_argument(
        "--scan-rows",
        type=int,
        default=5000,
        help=(
            "Number of TEST rows used to choose "
            "representative quantile cases."
        ),
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--sample-id",
        default=None,
        help=(
            "Exact frozen TEST sample_id. "
            "If set, quantile selection is skipped."
        ),
    )

    ap.add_argument(
        "--quantiles",
        default=(
            "0.10,0.50,0.75,0.90,0.95"
        ),
        help=(
            "Curve-MAE quantiles to visualize."
        ),
    )

    args = ap.parse_args()

    root = Path(
        args.root
    )

    base = (
        root
        / str(args.year)
    )

    frozen = (
        base
        / "frozen_modeling_dataset"
    )

    manifest = load_manifest(
        frozen
    )

    test_parts = [
        frozen / p
        for p in manifest[
            "parts"
        ][
            "test_curve"
        ]
    ]

    script_dir = Path(
        __file__
    ).resolve().parent

    runtime, runtime_file = (
        load_reconstruction_module(
            script_dir
        )
    )

    parameter_dir = (
        base
        / "template_parameter_models"
    )

    selected = pd.read_csv(
        parameter_dir
        / "selected_template_parameter_model.csv"
    ).iloc[0]

    fs = str(
        selected[
            "selected_feature_set"
        ]
    )

    parameter_bundle = joblib.load(
        parameter_dir
        / "models"
        / f"{fs}.joblib"
    )

    template_bundle = joblib.load(
        base
        / "final_template_predictor"
        / "final_template_model.joblib"
    )

    if args.sample_id:
        test_rows = find_exact_sample(
            test_parts,
            args.sample_id,
        )
    else:
        test_rows = sample_test_rows(
            test_parts,
            args.scan_rows,
            args.seed,
        )

    result = run_full_pipeline(
        test_rows,
        runtime,
        parameter_bundle,
        template_bundle,
    )

    meta = result[
        "meta"
    ]

    if args.sample_id:
        selected_cases = [
            (
                np.nan,
                0,
                float(
                    meta.iloc[0][
                        "physical_curve_mae"
                    ]
                ),
            )
        ]
    else:
        quantiles = [
            float(x)
            for x in str(
                args.quantiles
            ).split(",")
            if str(x).strip()
        ]

        for q in quantiles:
            if not (
                0.0 <= q <= 1.0
            ):
                raise ValueError(
                    f"Invalid quantile: {q}"
                )

        selected_cases = quantile_cases(
            meta,
            quantiles,
        )

    out = (
        base
        / "curve_visualization"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for case_no, (
        quantile,
        idx,
        target_error,
    ) in enumerate(
        selected_cases,
        1,
    ):
        row = meta.iloc[
            idx
        ].to_dict()

        qlabel = (
            "exact"
            if np.isnan(
                quantile
            )
            else (
                f"p{int(round(100 * quantile)):02d}"
            )
        )

        stem = (
            f"case_{case_no:02d}_"
            f"{qlabel}_"
            f"{safe_name(row['sample_id'])}"
        )

        physical_path = (
            out
            / f"{stem}_physical.png"
        )

        normalized_path = (
            out
            / f"{stem}_normalized.png"
        )

        plot_physical_case(
            row,
            result[
                "q_true"
            ][idx],
            result[
                "p_true"
            ][idx],
            result[
                "q_pred"
            ][idx],
            result[
                "p_pred"
            ][idx],
            physical_path,
        )

        plot_normalized_case(
            row,
            result[
                "p_true"
            ][idx],
            result[
                "p_pred"
            ][idx],
            normalized_path,
        )

        row[
            "selection_quantile"
        ] = (
            quantile
            if not np.isnan(
                quantile
            )
            else ""
        )

        row[
            "physical_figure"
        ] = str(
            physical_path
        )

        row[
            "normalized_figure"
        ] = str(
            normalized_path
        )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        out
        / "selected_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scan_summary = {
        "year": args.year,
        "runtime_file": str(
            runtime_file
        ),
        "scan_rows_requested": (
            None
            if args.sample_id
            else args.scan_rows
        ),
        "scan_rows_valid": int(
            len(meta)
        ),
        "sample_id_mode": (
            args.sample_id
            if args.sample_id
            else None
        ),
        "scanned_curve_mae": {
            "mean": float(
                meta[
                    "physical_curve_mae"
                ].mean()
            ),
            "p50": float(
                meta[
                    "physical_curve_mae"
                ].quantile(
                    0.50
                )
            ),
            "p75": float(
                meta[
                    "physical_curve_mae"
                ].quantile(
                    0.75
                )
            ),
            "p90": float(
                meta[
                    "physical_curve_mae"
                ].quantile(
                    0.90
                )
            ),
            "p95": float(
                meta[
                    "physical_curve_mae"
                ].quantile(
                    0.95
                )
            ),
        },
    }

    (
        out
        / "visualization_summary.json"
    ).write_text(
        json.dumps(
            scan_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "Bid-curve prediction visualization"
    )
    print("=" * 80)
    print(
        f"Runtime 04c: {runtime_file}"
    )
    print(
        f"Rows evaluated: {len(meta):,}"
    )
    print(
        f"Cases plotted: {len(summary):,}"
    )
    print()

    if not summary.empty:
        cols = [
            "sample_id",
            "true_template_id",
            "pred_template_id",
            "physical_curve_mae",
            "normalized_price_mae",
            "q_span_abs_error_mw",
        ]

        print(
            summary[
                cols
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Outputs: {out}"
    )


if __name__ == "__main__":
    main()
