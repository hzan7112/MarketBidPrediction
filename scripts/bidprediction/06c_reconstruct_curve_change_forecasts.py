#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06c_reconstruct_curve_change_forecasts.py

Final evaluation for the curve-persistence + predicted-change route.

For learned models:
    predicted DeltaZ
        -> inverse PCA
        -> predicted DeltaV
        -> V_prev + predicted DeltaV
        -> predicted bid curve

For zero_change:
    predicted DeltaV = 0
    -> predicted curve = previous raw observed curve

VALIDATION selects the winner by reconstructed full-curve PRICE WAPE.
TEST is evaluated only after that choice is frozen.

Candidates:
- zero_change  (raw curve persistence)
- ridge
- spline_gam
- random_forest

Diagnostic oracle:
- delta_latent_oracle: true DeltaZ passed through the selected PCA basis;
  this is the curve-change representation floor, not a forecast.

Run
---
python scripts/bidprediction/06c_reconstruct_curve_change_forecasts.py --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
RAW_PREV_COLS = [
    *[f"_curvevec_p{i:02d}_lag1" for i in range(21)],
    "_curvevec_q_anchor_lag1",
    "_curvevec_log_q_span_lag1",
]
CANDIDATES = [
    "zero_change",
    "ridge",
    "spline_gam",
    "random_forest",
]


def prediction_files(manifest, split):
    out = []
    for item in manifest["parts"][split]:
        out.append(item["file"] if isinstance(item, dict) else item)
    return out


def num(s):
    return pd.to_numeric(s, errors="coerce")


def true_curve(d):
    shape = (
        d[SHAPE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    p_anchor = num(d["p_anchor"]).to_numpy(np.float64)
    p_span = num(d["p_span"]).to_numpy(np.float64)

    zero = np.abs(p_span) <= 1e-12
    if zero.any():
        shape[zero, :] = np.nan_to_num(
            shape[zero, :], nan=0.0, posinf=0.0, neginf=0.0
        )

    price = p_anchor[:, None] + p_span[:, None] * shape
    q_anchor = num(d["q_anchor_mw"]).to_numpy(np.float64)
    q_span = num(d["q_span_mw"]).to_numpy(np.float64)
    quantity = q_anchor[:, None] + q_span[:, None] * GRID[None, :]

    return quantity, price, q_anchor, q_span


def previous_vector(d):
    return np.column_stack(
        [
            *[
                num(d[f"_curvevec_p{i:02d}_lag1"]).to_numpy(np.float64)
                for i in range(21)
            ],
            num(d["_curvevec_q_anchor_lag1"]).to_numpy(np.float64),
            num(d["_curvevec_log_q_span_lag1"]).to_numpy(np.float64),
        ]
    )


def latent_matrix(d, prefix, delta_cols):
    return np.column_stack(
        [
            num(d[f"{prefix}{z}"]).to_numpy(np.float64)
            for z in delta_cols
        ]
    )


def decode_delta(z, delta_bundle):
    z = np.asarray(z, dtype=np.float64)
    scaler = delta_bundle["scaler"]
    pca = delta_bundle["pca"]

    full = np.zeros(
        (len(z), int(pca.n_components_)),
        dtype=np.float64,
    )
    full[:, : z.shape[1]] = z

    standardized = pca.inverse_transform(full)
    return scaler.inverse_transform(standardized)


def vector_to_curve(v):
    v = np.asarray(v, dtype=np.float64)
    price = v[:, :21]
    q_anchor = v[:, 21]
    q_span = np.exp(np.clip(v[:, 22], -20.0, 20.0))
    return price, q_anchor, q_span


def pred_price_on_true_q(
    true_q,
    pred_price,
    pred_q_anchor,
    pred_q_span,
):
    q_span = np.maximum(pred_q_span, 1e-8)
    x = (
        true_q
        - pred_q_anchor[:, None]
    ) / q_span[:, None]

    pos = np.clip(x, 0.0, 1.0) * 20.0
    lo = np.floor(pos).astype(np.int16)
    hi = np.minimum(lo + 1, 20)
    f = pos - lo

    p_lo = np.take_along_axis(pred_price, lo, axis=1)
    p_hi = np.take_along_axis(pred_price, hi, axis=1)
    return p_lo + f * (p_hi - p_lo)


def empty_state():
    return {
        "rows": 0,
        "price_ae": 0.0,
        "price_se": 0.0,
        "price_abs_true": 0.0,
        "price_points": 0,
        "smape_sum": 0.0,
        "smape_points": 0,
        "curve_mae": [],
        "q_anchor_ae": 0.0,
        "q_anchor_abs_true": 0.0,
        "q_span_ae": 0.0,
        "q_span_abs_true": 0.0,
    }


def update_state(
    state,
    true_q,
    true_p,
    true_qa,
    true_qs,
    pred_p,
    pred_qa,
    pred_qs,
):
    pred_on_true = pred_price_on_true_q(
        true_q,
        pred_p,
        pred_qa,
        pred_qs,
    )

    err = pred_on_true - true_p
    ae = np.abs(err)
    denom = np.abs(pred_on_true) + np.abs(true_p)
    smape = np.divide(
        2.0 * ae,
        denom,
        out=np.zeros_like(ae),
        where=denom > 1e-9,
    )

    state["rows"] += int(len(true_p))
    state["price_ae"] += float(ae.sum())
    state["price_se"] += float(np.square(err).sum())
    state["price_abs_true"] += float(np.abs(true_p).sum())
    state["price_points"] += int(ae.size)
    state["smape_sum"] += float(smape.sum())
    state["smape_points"] += int(smape.size)
    state["curve_mae"].append(ae.mean(axis=1).astype(np.float32))
    state["q_anchor_ae"] += float(np.abs(pred_qa - true_qa).sum())
    state["q_anchor_abs_true"] += float(np.abs(true_qa).sum())
    state["q_span_ae"] += float(np.abs(pred_qs - true_qs).sum())
    state["q_span_abs_true"] += float(np.abs(true_qs).sum())


def finalize_state(state):
    curve = (
        np.concatenate(state["curve_mae"])
        if state["curve_mae"]
        else np.empty(0)
    )

    return {
        "rows": int(state["rows"]),
        "price_mae": state["price_ae"] / max(state["price_points"], 1),
        "price_rmse": float(
            np.sqrt(state["price_se"] / max(state["price_points"], 1))
        ),
        "price_wape_pct": (
            100.0
            * state["price_ae"]
            / max(state["price_abs_true"], 1e-12)
        ),
        "price_smape_pct": (
            100.0
            * state["smape_sum"]
            / max(state["smape_points"], 1)
        ),
        "curve_mae_p50": (
            float(np.quantile(curve, 0.50)) if len(curve) else np.nan
        ),
        "curve_mae_p90": (
            float(np.quantile(curve, 0.90)) if len(curve) else np.nan
        ),
        "curve_mae_p95": (
            float(np.quantile(curve, 0.95)) if len(curve) else np.nan
        ),
        "curve_mae_le20_share": (
            float(np.mean(curve <= 20.0)) if len(curve) else np.nan
        ),
        "q_anchor_wape_pct": (
            100.0
            * state["q_anchor_ae"]
            / max(state["q_anchor_abs_true"], 1e-12)
        ),
        "q_span_wape_pct": (
            100.0
            * state["q_span_ae"]
            / max(state["q_span_abs_true"], 1e-12)
        ),
    }


def evaluate_split(
    pred_dir,
    manifest,
    split,
    delta_bundle,
    delta_cols,
    modes,
):
    states = {name: empty_state() for name in modes}
    files = prediction_files(manifest, split)

    for i, rel in enumerate(files, 1):
        path = pred_dir / rel
        print(
            f"[{split.upper()} {i}/{len(files)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)
        if d.empty:
            continue

        true_q, true_p, true_qa, true_qs = true_curve(d)
        prev_v = previous_vector(d)

        if "zero_change" in states:
            p, qa, qs = vector_to_curve(prev_v)
            update_state(
                states["zero_change"],
                true_q,
                true_p,
                true_qa,
                true_qs,
                p,
                qa,
                qs,
            )

        for name in ["ridge", "spline_gam", "random_forest"]:
            if name not in states:
                continue

            z = latent_matrix(
                d,
                f"pred_{name}_",
                delta_cols,
            )
            delta_pred = decode_delta(z, delta_bundle)
            v_pred = prev_v + delta_pred
            p, qa, qs = vector_to_curve(v_pred)

            update_state(
                states[name],
                true_q,
                true_p,
                true_qa,
                true_qs,
                p,
                qa,
                qs,
            )

        if "delta_latent_oracle" in states:
            z_true = latent_matrix(
                d,
                "true_",
                delta_cols,
            )
            delta_oracle = decode_delta(z_true, delta_bundle)
            v_oracle = prev_v + delta_oracle
            p, qa, qs = vector_to_curve(v_oracle)

            update_state(
                states["delta_latent_oracle"],
                true_q,
                true_p,
                true_qa,
                true_qs,
                p,
                qa,
                qs,
            )

        del d, true_q, true_p, true_qa, true_qs, prev_v
        gc.collect()

    rows = []
    for name, state in states.items():
        row = finalize_state(state)
        row["model"] = name
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--root", default="data/processed/bidprediction")
    ap.add_argument("--delta-dir", default="curve_change_latent_dataset")
    ap.add_argument("--prediction-dir", default="curve_change_regression_models")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    delta_dir = base / args.delta_dir
    pred_dir = base / args.prediction_dir

    delta_bundle = joblib.load(delta_dir / "delta_pca_bundle.joblib")
    manifest = json.loads(
        (pred_dir / "manifest.json").read_text(encoding="utf-8")
    )
    delta_cols = list(manifest["delta_columns"])

    out = base / "curve_change_forecast_evaluation"
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} already exists. Use --overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Curve-change forecast evaluation - {args.year}")
    print("=" * 80)
    print(f"Delta latent dimension = {len(delta_cols)}")
    print()

    validation = evaluate_split(
        pred_dir,
        manifest,
        "val",
        delta_bundle,
        delta_cols,
        [*CANDIDATES, "delta_latent_oracle"],
    )
    validation.to_csv(
        out / "validation_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selectable = (
        validation.loc[validation["model"].isin(CANDIDATES)]
        .sort_values(
            ["price_wape_pct", "price_mae"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    if selectable.empty:
        raise ValueError("No selectable validation model.")

    selected_model = str(selectable.iloc[0]["model"])
    selected_validation = selectable.iloc[0].to_dict()

    selection = {
        "selection_split": "validation",
        "selection_metric": "reconstructed full-curve price WAPE",
        "selected_model": selected_model,
        "selected_validation_metrics": selected_validation,
        "candidate_models": CANDIDATES,
        "zero_change_definition": (
            "previous raw observed curve; exactly raw-curve persistence"
        ),
        "test_used_for_selection": False,
    }
    (out / "selected_model.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Validation curve metrics:")
    print(validation.to_string(index=False))
    print()
    print(f"Selected model from VALIDATION = {selected_model}")
    print()

    test_modes = list(
        dict.fromkeys(
            [
                "zero_change",
                selected_model,
                "delta_latent_oracle",
            ]
        )
    )

    test = evaluate_split(
        pred_dir,
        manifest,
        "test",
        delta_bundle,
        delta_cols,
        test_modes,
    )
    test.to_csv(
        out / "test_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def get_wape(table, model):
        x = table.loc[
            table["model"].eq(model),
            "price_wape_pct",
        ]
        return float(x.iloc[0]) if len(x) else np.nan

    baseline_wape = get_wape(test, "zero_change")
    selected_wape = get_wape(test, selected_model)
    oracle_wape = get_wape(test, "delta_latent_oracle")

    relative = (
        100.0
        * (baseline_wape - selected_wape)
        / baseline_wape
        if np.isfinite(baseline_wape) and baseline_wape != 0
        else np.nan
    )

    summary = "\n".join(
        [
            f"Curve-change forecast evaluation - {args.year}",
            "=" * 80,
            "",
            f"Delta latent dimension = {len(delta_cols)}",
            f"Selected model from VALIDATION = {selected_model}",
            "Selection metric = reconstructed full-curve price WAPE",
            "",
            "Validation:",
            validation.to_string(index=False),
            "",
            "TEST (frozen validation choice):",
            test.to_string(index=False),
            "",
            f"TEST zero-change/raw-persistence WAPE = {baseline_wape:.6f}%",
            f"TEST selected-model WAPE = {selected_wape:.6f}%",
            f"TEST delta-latent oracle WAPE = {oracle_wape:.6f}%",
            (
                "Relative improvement vs raw persistence = "
                f"{relative:.4f}%"
            ),
            "",
            (
                "If validation selects zero_change, the learned correction "
                "does not yet provide reliable incremental value."
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
