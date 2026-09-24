#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05a_build_curve_latent_representation.py

NEW MAIN ROUTE - Step 05a
=========================

Construct a continuous low-dimensional bid-curve representation using PCA.

Main forecasting route:
    historical bid curves
        -> continuous latent representation z
        -> lightweight regression
        -> future z
        -> bid curve

The existing 13 templates and interpretable theta are NOT used as forecasting
targets here. They remain available later as an explanation layer.

Continuous curve vector
-----------------------
For each valid bid curve:
    v = [P(u_00), ..., P(u_20), q_anchor_mw, log(q_span_mw)]

where:
    P(u_i) = p_anchor + p_span * shape_vXX
    u_i = 0, 0.05, ..., 1.0

This representation:
- retains absolute price level and curve shape;
- retains quantity interval;
- is continuous across former template boundaries;
- supports negative electricity prices;
- does not depend on template labels.

Flat-safe validity
------------------
For p_span == 0, P(u) == p_anchor regardless of normalized shape. Therefore
missing shape values are allowed for zero-price-span curves. This is a generic
mathematical rule, not a dataset-specific FLAT-template exception.

Leakage control
---------------
- StandardScaler and PCA are fitted ONLY on train data.
- Validation/test are transform/evaluation only.
- Latent dimension is chosen from TRAIN cumulative explained variance.
- No current template label or current theta is used by the encoder.

Outputs
-------
data/processed/bidprediction/<year>/curve_latent_pca/
    pca_bundle.joblib
    latent_dimension_metrics.csv
    selected_latent_dimension.json
    curve_validity_diagnostics.json
    feature_schema.csv
    latent_parts/
        train/*.pkl
        val/*.pkl
        test/*.pkl
    manifest.json
    summary.txt

Run
---
python scripts/bidprediction/05a_build_curve_latent_representation.py --year 2025 --overwrite
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
CORE_CURVE_COLS = [
    "p_anchor",
    "p_span",
    "q_anchor_mw",
    "q_span_mw",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def get_parts(manifest, split):
    parts = manifest.get("parts", {}).get(split)
    if parts:
        return list(parts)

    if split == "train" and manifest.get("train_sample_file"):
        return [manifest["train_sample_file"]]

    raise KeyError(f"No files found for split={split}")


def curve_validity(d):
    missing = [
        c
        for c in [*CORE_CURVE_COLS, *SHAPE_COLS]
        if c not in d.columns
    ]
    if missing:
        raise KeyError(f"Missing curve columns: {missing}")

    core = d[CORE_CURVE_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    core_finite = core.notna().all(axis=1)

    p_span = num(d["p_span"])
    q_span = num(d["q_span_mw"])

    zero_price_span = p_span.abs() <= 1e-12
    positive_q_span = q_span > 0.0

    shape = d[SHAPE_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    shape_complete = shape.notna().all(axis=1)

    valid = (
        core_finite
        & positive_q_span
        & (
            zero_price_span
            | shape_complete
        )
    )

    reason = pd.Series(
        "valid",
        index=d.index,
        dtype="string",
    )

    reason.loc[~core_finite] = "invalid_core_numeric"
    reason.loc[
        core_finite
        & (~positive_q_span)
    ] = "nonpositive_q_span"
    reason.loc[
        core_finite
        & positive_q_span
        & (~zero_price_span)
        & (~shape_complete)
    ] = "missing_shape_nonzero_price_span"
    reason.loc[
        valid
        & zero_price_span
        & (~shape_complete)
    ] = "valid_zero_price_span_missing_shape"

    return valid, reason


def curve_vector(d):
    shape = (
        d[SHAPE_COLS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(np.float64)
    )

    p_anchor = num(d["p_anchor"]).to_numpy(np.float64)
    p_span = num(d["p_span"]).to_numpy(np.float64)
    q_anchor = num(d["q_anchor_mw"]).to_numpy(np.float64)
    q_span = num(d["q_span_mw"]).to_numpy(np.float64)

    zero_price_span = np.abs(p_span) <= 1e-12

    if zero_price_span.any():
        shape[zero_price_span, :] = np.nan_to_num(
            shape[zero_price_span, :],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    if not np.isfinite(shape).all():
        raise ValueError(
            "Nonfinite shape remains on nonzero-price-span rows."
        )

    price = (
        p_anchor[:, None]
        + p_span[:, None] * shape
    )

    return np.c_[
        price,
        q_anchor,
        np.log(
            np.maximum(
                q_span,
                1e-8,
            )
        ),
    ]


def vector_to_curve(v):
    v = np.asarray(v, dtype=np.float64)
    price = v[:, :21]
    q_anchor = v[:, 21]
    q_span = np.exp(
        np.clip(
            v[:, 22],
            -20.0,
            20.0,
        )
    )
    return price, q_anchor, q_span


def load_fit_sample(
    frozen,
    manifest,
    max_rows,
    seed,
):
    parts = get_parts(
        manifest,
        "train",
    )

    per_part = max(
        1,
        int(
            math.ceil(
                max_rows
                / len(parts)
            )
        ),
    )

    blocks = []

    for i, rel in enumerate(parts, 1):
        path = frozen / rel

        print(
            f"[fit {i}/{len(parts)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)
        ready, _ = curve_validity(d)

        d = d.loc[ready].copy()

        if d.empty:
            continue

        n = min(
            per_part,
            len(d),
        )

        if n < len(d):
            d = d.sample(
                n=n,
                random_state=seed + 1009 * i,
            )

        blocks.append(
            curve_vector(d)
        )

        del d
        gc.collect()

    if not blocks:
        raise ValueError(
            "No valid train curves found."
        )

    X = np.vstack(blocks)

    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            len(X),
            size=max_rows,
            replace=False,
        )
        X = X[idx]

    return X


def empty_stats():
    return {
        "rows": 0,
        "price_ae_sum": 0.0,
        "price_abs_sum": 0.0,
        "price_se_sum": 0.0,
        "price_points": 0,
        "curve_mae_values": [],
        "q_anchor_ae_sum": 0.0,
        "q_anchor_abs_sum": 0.0,
        "q_span_ae_sum": 0.0,
        "q_span_abs_sum": 0.0,
    }


def reconstruction_stats(
    true_v,
    reconstructed_v,
):
    true_p, true_qa, true_qs = vector_to_curve(true_v)
    pred_p, pred_qa, pred_qs = vector_to_curve(reconstructed_v)

    ae_p = np.abs(pred_p - true_p)

    return {
        "rows": int(len(true_v)),
        "price_ae_sum": float(ae_p.sum()),
        "price_abs_sum": float(np.abs(true_p).sum()),
        "price_se_sum": float(
            np.square(
                pred_p - true_p
            ).sum()
        ),
        "price_points": int(ae_p.size),
        "curve_mae_values": ae_p.mean(axis=1).astype(np.float32),
        "q_anchor_ae_sum": float(
            np.abs(
                pred_qa - true_qa
            ).sum()
        ),
        "q_anchor_abs_sum": float(
            np.abs(true_qa).sum()
        ),
        "q_span_ae_sum": float(
            np.abs(
                pred_qs - true_qs
            ).sum()
        ),
        "q_span_abs_sum": float(
            np.abs(true_qs).sum()
        ),
    }


def merge_stats(state, part):
    state["rows"] += part["rows"]

    for k in [
        "price_ae_sum",
        "price_abs_sum",
        "price_se_sum",
        "price_points",
        "q_anchor_ae_sum",
        "q_anchor_abs_sum",
        "q_span_ae_sum",
        "q_span_abs_sum",
    ]:
        state[k] += part[k]

    state["curve_mae_values"].append(
        part["curve_mae_values"]
    )


def finalize_stats(state):
    curve = (
        np.concatenate(
            state["curve_mae_values"]
        )
        if state["curve_mae_values"]
        else np.empty(0)
    )

    return {
        "rows": int(state["rows"]),
        "price_mae": (
            state["price_ae_sum"]
            / max(
                state["price_points"],
                1,
            )
        ),
        "price_rmse": float(
            np.sqrt(
                state["price_se_sum"]
                / max(
                    state["price_points"],
                    1,
                )
            )
        ),
        "price_wape_pct": (
            100.0
            * state["price_ae_sum"]
            / max(
                state["price_abs_sum"],
                1e-12,
            )
        ),
        "curve_mae_p50": (
            float(
                np.quantile(
                    curve,
                    0.50,
                )
            )
            if len(curve)
            else np.nan
        ),
        "curve_mae_p90": (
            float(
                np.quantile(
                    curve,
                    0.90,
                )
            )
            if len(curve)
            else np.nan
        ),
        "curve_mae_p95": (
            float(
                np.quantile(
                    curve,
                    0.95,
                )
            )
            if len(curve)
            else np.nan
        ),
        "q_anchor_wape_pct": (
            100.0
            * state["q_anchor_ae_sum"]
            / max(
                state["q_anchor_abs_sum"],
                1e-12,
            )
        ),
        "q_span_wape_pct": (
            100.0
            * state["q_span_ae_sum"]
            / max(
                state["q_span_abs_sum"],
                1e-12,
            )
        ),
    }


def truncate_inverse(
    z_full,
    k,
    pca,
    scaler,
):
    z = np.zeros_like(z_full)
    z[:, :k] = z_full[:, :k]

    return scaler.inverse_transform(
        pca.inverse_transform(z)
    )


def evaluate_dimensions(
    frozen,
    manifest,
    split,
    scaler,
    pca,
    candidate_dims,
):
    states = {
        int(k): empty_stats()
        for k in candidate_dims
    }

    reasons = {}
    parts = get_parts(
        manifest,
        split,
    )

    for i, rel in enumerate(parts, 1):
        path = frozen / rel

        print(
            f"[{split} {i}/{len(parts)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)
        ready, reason = curve_validity(d)

        vc = reason.value_counts(
            dropna=False
        )

        for key, value in vc.items():
            reasons[str(key)] = (
                reasons.get(
                    str(key),
                    0,
                )
                + int(value)
            )

        d = d.loc[ready].copy()

        if d.empty:
            continue

        V = curve_vector(d)
        Z = pca.transform(
            scaler.transform(V)
        )

        for k in candidate_dims:
            Vhat = truncate_inverse(
                Z,
                int(k),
                pca,
                scaler,
            )

            merge_stats(
                states[int(k)],
                reconstruction_stats(
                    V,
                    Vhat,
                ),
            )

        del d, V, Z
        gc.collect()

    cumulative = np.cumsum(
        pca.explained_variance_ratio_
    )

    invalid_rows = sum(
        count
        for name, count in reasons.items()
        if not name.startswith("valid")
    )

    rows = []

    for k in candidate_dims:
        row = finalize_stats(
            states[int(k)]
        )
        row["split"] = split
        row["latent_dim"] = int(k)
        row[
            "cumulative_explained_variance"
        ] = float(
            cumulative[
                int(k) - 1
            ]
        )
        row[
            "dropped_invalid_curve_rows"
        ] = int(invalid_rows)
        rows.append(row)

    return pd.DataFrame(rows), reasons


def write_latent_parts(
    frozen,
    manifest,
    out_dir,
    split,
    scaler,
    pca,
    selected_k,
):
    out_part_dir = (
        out_dir
        / "latent_parts"
        / split
    )
    out_part_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = []
    source_parts = get_parts(
        manifest,
        split,
    )

    for i, rel in enumerate(source_parts, 1):
        src = frozen / rel

        print(
            f"[write {split} {i}/{len(source_parts)}] {src.name}",
            flush=True,
        )

        d = pd.read_pickle(src)
        ready, _ = curve_validity(d)

        d = (
            d.loc[ready]
            .copy()
            .reset_index(drop=True)
        )

        if d.empty:
            continue

        V = curve_vector(d)

        Z = pca.transform(
            scaler.transform(V)
        )[
            :,
            :selected_k,
        ].astype(
            np.float32
        )

        latent = pd.DataFrame(
            {
                f"latent_z{i+1:02d}": Z[:, i]
                for i in range(selected_k)
            }
        )

        out_d = pd.concat(
            [
                d.reset_index(drop=True),
                latent,
            ],
            axis=1,
        )

        out_name = (
            f"{split}_curve_latent_"
            f"{i:04d}.pkl"
        )
        out_path = out_part_dir / out_name

        out_d.to_pickle(out_path)

        written.append(
            {
                "file": str(
                    out_path.relative_to(
                        out_dir
                    )
                ),
                "source_file": str(rel),
                "rows": int(len(out_d)),
            }
        )

        del d, V, Z, latent, out_d
        gc.collect()

    return written


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
        default="frozen_stable_theta_continuous_dataset",
    )
    ap.add_argument(
        "--max-fit-rows",
        type=int,
        default=500_000,
    )
    ap.add_argument(
        "--max-components",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--variance-target",
        type=float,
        default=0.995,
    )
    ap.add_argument(
        "--candidate-dims",
        default="2,3,4,5,6,8,10,12,16,20",
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
    frozen = (
        base
        / args.dataset_dir
    )

    manifest_file = (
        frozen
        / "manifest.json"
    )
    if not manifest_file.exists():
        raise FileNotFoundError(
            manifest_file
        )

    manifest = json.loads(
        manifest_file.read_text(
            encoding="utf-8"
        )
    )

    out = (
        base
        / "curve_latent_pca"
    )

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} already exists. "
                f"Use --overwrite."
            )
        shutil.rmtree(out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_components = min(
        int(args.max_components),
        23,
    )

    candidate_dims = sorted(
        set(
            int(x.strip())
            for x in args.candidate_dims.split(",")
            if x.strip()
        )
    )

    candidate_dims = [
        k
        for k in candidate_dims
        if 1 <= k <= max_components
    ]

    if max_components not in candidate_dims:
        candidate_dims.append(
            max_components
        )

    print("=" * 80)
    print(
        f"Continuous bid-curve latent representation - {args.year}"
    )
    print("=" * 80)
    print(
        "Curve vector = 21 absolute price ordinates "
        "+ q_anchor + log(q_span)"
    )
    print(
        f"Fit sample cap = {args.max_fit_rows:,}"
    )
    print(
        f"Variance target = {args.variance_target:.4f}"
    )
    print()

    Xfit = load_fit_sample(
        frozen,
        manifest,
        args.max_fit_rows,
        args.seed,
    )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xfit)

    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=args.seed,
    )
    pca.fit(Xs)

    cumulative = np.cumsum(
        pca.explained_variance_ratio_
    )

    reached = np.where(
        cumulative
        >= args.variance_target
    )[0]

    selected_k = (
        int(
            reached[0]
            + 1
        )
        if len(reached)
        else max_components
    )

    candidate_dims = sorted(
        set(
            [
                *candidate_dims,
                selected_k,
            ]
        )
    )

    print(
        f"Fit rows = {len(Xfit):,}"
    )
    print(
        f"Selected latent dimension = {selected_k}"
    )
    print(
        f"Cumulative explained variance = "
        f"{cumulative[selected_k - 1]:.6f}"
    )
    print()

    metric_tables = []
    validity = {}

    for split in ["val", "test"]:
        table, reasons = evaluate_dimensions(
            frozen,
            manifest,
            split,
            scaler,
            pca,
            candidate_dims,
        )
        metric_tables.append(table)
        validity[split] = reasons

    metrics = pd.concat(
        metric_tables,
        ignore_index=True,
    )

    metrics.to_csv(
        out
        / "latent_dimension_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_info = {
        "selection_rule": (
            "smallest PCA dimension reaching "
            "train cumulative explained-variance target"
        ),
        "variance_target": float(
            args.variance_target
        ),
        "selected_latent_dim": int(
            selected_k
        ),
        "selected_cumulative_explained_variance": float(
            cumulative[
                selected_k - 1
            ]
        ),
    }

    (
        out
        / "selected_latent_dimension.json"
    ).write_text(
        json.dumps(
            selected_info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        out
        / "curve_validity_diagnostics.json"
    ).write_text(
        json.dumps(
            validity,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    joblib.dump(
        {
            "version": "continuous-bid-curve-pca-v2-flat-safe",
            "year": int(args.year),
            "curve_vector_definition": (
                "[P(u00)..P(u20), "
                "q_anchor_mw, log(q_span_mw)]"
            ),
            "grid": GRID,
            "shape_columns": SHAPE_COLS,
            "scaler": scaler,
            "pca": pca,
            "selected_latent_dim": int(
                selected_k
            ),
            "variance_target": float(
                args.variance_target
            ),
            "max_components": int(
                max_components
            ),
        },
        out
        / "pca_bundle.joblib",
        compress=3,
    )

    source_schema = (
        frozen
        / "feature_schema.csv"
    )
    if source_schema.exists():
        shutil.copy2(
            source_schema,
            out
            / "feature_schema.csv",
        )

    latent_manifest = {
        "year": int(args.year),
        "source_dataset": str(frozen),
        "selected_latent_dim": int(
            selected_k
        ),
        "parts": {},
    }

    for split in [
        "train",
        "val",
        "test",
    ]:
        latent_manifest[
            "parts"
        ][
            split
        ] = write_latent_parts(
            frozen,
            manifest,
            out,
            split,
            scaler,
            pca,
            selected_k,
        )

    (
        out
        / "manifest.json"
    ).write_text(
        json.dumps(
            latent_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selected_metrics = (
        metrics.loc[
            metrics["latent_dim"]
            .eq(selected_k)
        ]
        .sort_values("split")
    )

    summary = "\n".join(
        [
            (
                f"Continuous bid-curve latent representation - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                "Curve vector = "
                "21 absolute price ordinates "
                "+ q_anchor + log(q_span)"
            ),
            f"Fit rows = {len(Xfit):,}",
            (
                f"Selected latent dimension = "
                f"{selected_k}"
            ),
            (
                "Cumulative train explained variance = "
                f"{cumulative[selected_k - 1]:.6f}"
            ),
            "",
            "Curve validity diagnostics:",
            json.dumps(
                validity,
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Selected-dimension reconstruction:",
            selected_metrics.to_string(
                index=False
            ),
            "",
            "All candidate dimensions:",
            metrics.to_string(
                index=False
            ),
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
    print(
        f"Outputs: {out}"
    )


if __name__ == "__main__":
    main()
