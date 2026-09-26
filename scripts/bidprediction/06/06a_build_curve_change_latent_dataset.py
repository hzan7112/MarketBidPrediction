#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06a_build_curve_change_latent_dataset.py

Main route: curve persistence + predicted curve change.

For every history-ready sample:
    V_t     = [P_t(u00)..P_t(u20), q_anchor_t, log(q_span_t)]
    V_prev  = previous observed curve vector for the same participant + slot
    DeltaV  = V_t - V_prev

Fit StandardScaler + PCA on TRAIN DeltaV only:
    DeltaV -> DeltaZ

Then build leakage-free historical DeltaZ features:
    lag1 / lag2 / lag7
    mean7 / std7
    mean30 / std30
    delta1 / delta7
    mean_gap_7_30

Outputs
-------
data/processed/bidprediction/<year>/curve_change_latent_dataset/

Run
---
python scripts/bidprediction/06a_build_curve_change_latent_dataset.py --year 2025 --overwrite
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
RAW_PREV_COLS = [
    *[f"_curvevec_p{i:02d}_lag1" for i in range(21)],
    "_curvevec_q_anchor_lag1",
    "_curvevec_log_q_span_lag1",
]
CURVE_TARGET_COLS = [
    *SHAPE_COLS,
    "p_anchor",
    "p_span",
    "q_anchor_mw",
    "q_span_mw",
]
META_CANDIDATES = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
    "prediction_cutoff_utc",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def part_files(manifest, split):
    out = []
    for item in manifest["parts"][split]:
        out.append(item["file"] if isinstance(item, dict) else item)
    return out


def current_curve_vector(d):
    shape = d[SHAPE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    p_anchor = num(d["p_anchor"]).to_numpy(np.float64)
    p_span = num(d["p_span"]).to_numpy(np.float64)

    zero = np.abs(p_span) <= 1e-12
    if zero.any():
        shape[zero, :] = np.nan_to_num(
            shape[zero, :], nan=0.0, posinf=0.0, neginf=0.0
        )

    if not np.isfinite(shape).all():
        raise ValueError("Nonfinite current shape on a nonzero-price-span row.")

    price = p_anchor[:, None] + p_span[:, None] * shape
    q_anchor = num(d["q_anchor_mw"]).to_numpy(np.float64)
    q_span = num(d["q_span_mw"]).to_numpy(np.float64)

    return np.c_[price, q_anchor, np.log(np.maximum(q_span, 1e-8))]


def previous_curve_vector(d):
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


def valid_change_rows(d):
    ready = d["latent_history_ready_flag"].eq(1)
    prev = d[RAW_PREV_COLS].apply(pd.to_numeric, errors="coerce")
    ready &= prev.notna().all(axis=1)

    current_core = d[
        ["p_anchor", "p_span", "q_anchor_mw", "q_span_mw"]
    ].apply(pd.to_numeric, errors="coerce")
    ready &= current_core.notna().all(axis=1)
    ready &= num(d["q_span_mw"]) > 0.0
    return ready


def load_fit_sample(source_dir, manifest, max_rows, seed):
    files = part_files(manifest, "train")
    per_part = max(1, int(math.ceil(max_rows / len(files))))
    blocks = []

    for i, rel in enumerate(files, 1):
        path = source_dir / rel
        print(f"[fit {i}/{len(files)}] {path.name}", flush=True)

        d = pd.read_pickle(path)
        d = d.loc[valid_change_rows(d)].copy()
        if d.empty:
            continue

        n = min(per_part, len(d))
        if n < len(d):
            d = d.sample(n=n, random_state=seed + 1009 * i)

        blocks.append(current_curve_vector(d) - previous_curve_vector(d))
        del d
        gc.collect()

    if not blocks:
        raise ValueError("No valid TRAIN curve-change rows.")

    X = np.vstack(blocks)
    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), size=max_rows, replace=False)]
    return X


def evaluate_dimensions(source_dir, manifest, split, scaler, pca, dims):
    sums = {
        k: {
            "rows": 0,
            "vec_abs": 0.0,
            "vec_count": 0,
            "price_abs": 0.0,
            "price_sq": 0.0,
            "price_count": 0,
            "qa_abs": 0.0,
            "lqs_abs": 0.0,
        }
        for k in dims
    }

    files = part_files(manifest, split)

    for i, rel in enumerate(files, 1):
        path = source_dir / rel
        print(f"[{split} {i}/{len(files)}] {path.name}", flush=True)

        d = pd.read_pickle(path)
        d = d.loc[valid_change_rows(d)].copy()
        if d.empty:
            continue

        delta = current_curve_vector(d) - previous_curve_vector(d)
        z_full = pca.transform(scaler.transform(delta))

        for k in dims:
            z = np.zeros_like(z_full)
            z[:, :k] = z_full[:, :k]
            rec = scaler.inverse_transform(pca.inverse_transform(z))
            err = rec - delta
            ae = np.abs(err)

            st = sums[k]
            st["rows"] += len(delta)
            st["vec_abs"] += float(ae.sum())
            st["vec_count"] += int(ae.size)
            st["price_abs"] += float(ae[:, :21].sum())
            st["price_sq"] += float(np.square(err[:, :21]).sum())
            st["price_count"] += int(err[:, :21].size)
            st["qa_abs"] += float(ae[:, 21].sum())
            st["lqs_abs"] += float(ae[:, 22].sum())

        del d, delta, z_full
        gc.collect()

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    rows = []

    for k in dims:
        st = sums[k]
        n = max(st["rows"], 1)
        rows.append(
            {
                "split": split,
                "latent_dim": int(k),
                "rows": int(st["rows"]),
                "cumulative_explained_variance": float(cumulative[k - 1]),
                "delta_vector_mae": st["vec_abs"] / max(st["vec_count"], 1),
                "delta_price_mae": st["price_abs"] / max(st["price_count"], 1),
                "delta_price_rmse": float(
                    np.sqrt(st["price_sq"] / max(st["price_count"], 1))
                ),
                "delta_q_anchor_mae": st["qa_abs"] / n,
                "delta_log_q_span_mae": st["lqs_abs"] / n,
            }
        )

    return pd.DataFrame(rows)


def resolve_time_col(d):
    if "timestamp_utc" in d.columns:
        return "timestamp_utc"
    if "timestamp_local" in d.columns:
        return "timestamp_local"
    if "local_date" in d.columns:
        return "local_date"
    raise KeyError("Need timestamp_utc, timestamp_local, or local_date.")


def normalize_time(s, col):
    return pd.to_datetime(s, errors="coerce", utc=(col == "timestamp_utc"))


def grouped_rolling(frame, shifted, keys, window, stat):
    grouped = shifted.groupby([frame[k] for k in keys], sort=False)
    roll = grouped.rolling(window=window, min_periods=2)
    if stat == "mean":
        out = roll.mean()
    elif stat == "std":
        out = roll.std(ddof=0)
    else:
        raise ValueError(stat)
    return out.reset_index(level=list(range(len(keys))), drop=True)


def add_delta_history(
    current,
    tail,
    delta_cols,
    max_history,
):
    """
    Build leakage-free DeltaZ history features without expanding the history
    tail to the full current-row schema.

    The history state contains ONLY:
        participant_id
        local_slot_seconds
        chronological key
        current DeltaZ coordinates

    This avoids DataFrame fragmentation and unnecessary all-NA columns.
    """
    keys = [
        "participant_id",
        "local_slot_seconds",
    ]

    current = current.copy().reset_index(drop=True)

    # Preserve original row order so computed history features can be attached
    # back to the full current dataframe in one concat operation.
    current["_current_row_id"] = np.arange(
        len(current),
        dtype=np.int64,
    )

    time_col = resolve_time_col(
        current
    )

    history_time = normalize_time(
        current[time_col],
        time_col,
    )

    if history_time.isna().any():
        raise ValueError(
            f"Cannot parse chronological key {time_col}."
        )

    # Minimal state frame used only for temporal feature construction.
    current_state = pd.DataFrame(
        {
            "participant_id": current[
                "participant_id"
            ].to_numpy(),
            "local_slot_seconds": current[
                "local_slot_seconds"
            ].to_numpy(),
            "_history_time": history_time.to_numpy(),
            "_is_current": np.ones(
                len(current),
                dtype=np.int8,
            ),
            "_current_row_id": current[
                "_current_row_id"
            ].to_numpy(),
        }
    )

    delta_block = current[
        delta_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).astype(
        np.float32
    )

    current_state = pd.concat(
        [
            current_state,
            delta_block.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    state_cols = [
        *keys,
        "_history_time",
        *delta_cols,
    ]

    if tail is None:
        combined = current_state
    else:
        hist = tail.copy()

        hist_state = hist[
            state_cols
        ].copy()

        hist_state[
            "_is_current"
        ] = np.zeros(
            len(hist_state),
            dtype=np.int8,
        )

        hist_state[
            "_current_row_id"
        ] = np.full(
            len(hist_state),
            -1,
            dtype=np.int64,
        )

        # Both frames now have exactly the same compact schema.
        hist_state = hist_state[
            current_state.columns
        ]

        combined = pd.concat(
            [
                hist_state,
                current_state,
            ],
            ignore_index=True,
            copy=False,
        )

    combined = (
        combined.sort_values(
            [
                *keys,
                "_history_time",
                "_is_current",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    g = combined.groupby(
        keys,
        sort=False,
    )

    history_feature_data = {}

    for z in delta_cols:
        lag1 = g[z].shift(1)
        lag2 = g[z].shift(2)
        lag7 = g[z].shift(7)

        mean7 = grouped_rolling(
            combined,
            lag1,
            keys,
            7,
            "mean",
        )

        std7 = grouped_rolling(
            combined,
            lag1,
            keys,
            7,
            "std",
        )

        mean30 = grouped_rolling(
            combined,
            lag1,
            keys,
            30,
            "mean",
        )

        std30 = grouped_rolling(
            combined,
            lag1,
            keys,
            30,
            "std",
        )

        history_feature_data[
            f"{z}_lag1"
        ] = lag1.astype(
            np.float32
        )

        history_feature_data[
            f"{z}_lag2"
        ] = lag2.astype(
            np.float32
        )

        history_feature_data[
            f"{z}_lag7"
        ] = lag7.astype(
            np.float32
        )

        history_feature_data[
            f"{z}_mean7"
        ] = mean7.astype(
            np.float32
        )

        history_feature_data[
            f"{z}_std7"
        ] = std7.astype(
            np.float32
        )

        history_feature_data[
            f"{z}_mean30"
        ] = mean30.astype(
            np.float32
        )

        history_feature_data[
            f"{z}_std30"
        ] = std30.astype(
            np.float32
        )

        history_feature_data[
            f"{z}_delta1"
        ] = (
            lag1
            - lag2
        ).astype(
            np.float32
        )

        history_feature_data[
            f"{z}_delta7"
        ] = (
            lag1
            - lag7
        ).astype(
            np.float32
        )

        history_feature_data[
            f"{z}_mean_gap_7_30"
        ] = (
            mean7
            - mean30
        ).astype(
            np.float32
        )

    history_features = pd.DataFrame(
        history_feature_data,
        index=combined.index,
    )

    history_cols = list(
        history_features.columns
    )

    combined = pd.concat(
        [
            combined,
            history_features,
        ],
        axis=1,
        copy=False,
    )

    # Extract only current rows, restore their original order, and attach the
    # computed history columns to the untouched full current dataframe.
    current_hist = (
        combined.loc[
            combined[
                "_is_current"
            ].eq(
                1
            ),
            [
                "_current_row_id",
                *history_cols,
            ],
        ]
        .sort_values(
            "_current_row_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        current_hist
    ) != len(
        current
    ):
        raise RuntimeError(
            "History feature row count does not match current rows."
        )

    enriched = pd.concat(
        [
            current.drop(
                columns=[
                    "_current_row_id"
                ]
            ).reset_index(
                drop=True
            ),
            current_hist[
                history_cols
            ].reset_index(
                drop=True
            ),
        ],
        axis=1,
        copy=False,
    )

    # Carry only compact temporal state forward.
    new_tail = (
        combined[
            state_cols
        ]
        .groupby(
            keys,
            sort=False,
            as_index=False,
            group_keys=False,
        )
        .tail(
            max_history
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        enriched,
        new_tail,
        history_cols,
    )


def write_dataset(
    source_dir,
    source_manifest,
    out,
    scaler,
    pca,
    selected_k,
    max_history,
):
    delta_cols = [f"delta_latent_z{i+1:02d}" for i in range(selected_k)]
    source_features = list(source_manifest["model_features"])

    tail = None
    output_parts = {}
    split_stats = {}
    delta_history_cols = None

    for split in ["train", "val", "test"]:
        files = part_files(source_manifest, split)
        split_dir = out / "parts" / split
        split_dir.mkdir(parents=True, exist_ok=True)

        written = []
        total = 0

        for i, rel in enumerate(files, 1):
            path = source_dir / rel
            print(f"[write {split} {i}/{len(files)}] {path.name}", flush=True)

            d = pd.read_pickle(path)
            d = d.loc[valid_change_rows(d)].copy().reset_index(drop=True)
            if d.empty:
                continue

            current_v = current_curve_vector(d)
            prev_v = previous_curve_vector(d)
            delta = current_v - prev_v
            z = pca.transform(scaler.transform(delta))[:, :selected_k].astype(
                np.float32
            )

            for j, c in enumerate(delta_cols):
                d[c] = z[:, j]

            enriched, tail, hist_cols = add_delta_history(
                d, tail, delta_cols, max_history
            )
            if delta_history_cols is None:
                delta_history_cols = hist_cols

            meta_cols = [c for c in META_CANDIDATES if c in enriched.columns]
            keep = list(
                dict.fromkeys(
                    [
                        *meta_cols,
                        *source_features,
                        *delta_history_cols,
                        *delta_cols,
                        *CURVE_TARGET_COLS,
                        *RAW_PREV_COLS,
                    ]
                )
            )
            if "y_template_id" in enriched.columns:
                keep.append("y_template_id")

            out_d = enriched[keep].copy()
            out_name = f"{split}_curve_change_{i:04d}.pkl"
            out_path = split_dir / out_name
            out_d.to_pickle(out_path)

            written.append(
                {
                    "file": str(out_path.relative_to(out)),
                    "rows": int(len(out_d)),
                }
            )
            total += int(len(out_d))

            del d, current_v, prev_v, delta, z, enriched, out_d
            gc.collect()

        output_parts[split] = written
        split_stats[split] = {"rows": int(total)}

    model_features = list(dict.fromkeys([*source_features, *delta_history_cols]))

    return {
        "parts": output_parts,
        "split_stats": split_stats,
        "delta_columns": delta_cols,
        "delta_history_features": delta_history_cols,
        "source_model_features": source_features,
        "model_features": model_features,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--root", default="data/processed/bidprediction")
    ap.add_argument("--source-dir", default="latent_forecasting_dataset")
    ap.add_argument("--max-fit-rows", type=int, default=500_000)
    ap.add_argument("--max-components", type=int, default=20)
    ap.add_argument("--variance-target", type=float, default=0.995)
    ap.add_argument("--candidate-dims", default="2,3,4,5,6,8,10,12,16,20")
    ap.add_argument("--max-history", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    source_dir = base / args.source_dir
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )

    out = base / "curve_change_latent_dataset"
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} already exists. Use --overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Curve-change latent dataset - {args.year}")
    print("=" * 80)
    print("Target = current raw curve vector - previous raw curve vector")
    print()

    Xfit = load_fit_sample(
        source_dir, source_manifest, args.max_fit_rows, args.seed
    )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xfit)

    max_components = min(int(args.max_components), Xfit.shape[1])
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=args.seed,
    )
    pca.fit(Xs)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    reached = np.where(cumulative >= args.variance_target)[0]
    selected_k = int(reached[0] + 1) if len(reached) else max_components

    dims = sorted(
        set(
            int(x.strip())
            for x in args.candidate_dims.split(",")
            if x.strip()
        )
    )
    dims = [k for k in dims if 1 <= k <= max_components]
    dims = sorted(set([*dims, selected_k, max_components]))

    print(f"Fit rows = {len(Xfit):,}")
    print(f"Selected delta latent dimension = {selected_k}")
    print(
        "Cumulative explained variance = "
        f"{cumulative[selected_k - 1]:.6f}"
    )
    print()

    metrics = pd.concat(
        [
            evaluate_dimensions(
                source_dir, source_manifest, split, scaler, pca, dims
            )
            for split in ["val", "test"]
        ],
        ignore_index=True,
    )
    metrics.to_csv(
        out / "delta_latent_dimension_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        {
            "version": "curve-change-pca-v1",
            "year": int(args.year),
            "vector_definition": (
                "[Delta P(u00)..Delta P(u20), Delta q_anchor, "
                "Delta log(q_span)]"
            ),
            "scaler": scaler,
            "pca": pca,
            "selected_delta_latent_dim": int(selected_k),
            "variance_target": float(args.variance_target),
        },
        out / "delta_pca_bundle.joblib",
        compress=3,
    )

    selected_info = {
        "selection_rule": (
            "smallest TRAIN PCA dimension reaching the configured "
            "cumulative explained-variance target"
        ),
        "variance_target": float(args.variance_target),
        "selected_delta_latent_dim": int(selected_k),
        "selected_cumulative_explained_variance": float(
            cumulative[selected_k - 1]
        ),
    }
    (out / "selected_delta_latent_dimension.json").write_text(
        json.dumps(selected_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ds_info = write_dataset(
        source_dir,
        source_manifest,
        out,
        scaler,
        pca,
        selected_k,
        args.max_history,
    )

    manifest = {
        "year": int(args.year),
        "source_dataset": str(source_dir),
        "selected_delta_latent_dim": int(selected_k),
        **ds_info,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    schema_rows = []
    for c in ds_info["model_features"]:
        schema_rows.append(
            {
                "column": c,
                "role": "feature",
                "feature_group": (
                    "delta_latent_history"
                    if c in ds_info["delta_history_features"]
                    else "source_feature"
                ),
            }
        )
    for c in ds_info["delta_columns"]:
        schema_rows.append(
            {
                "column": c,
                "role": "target",
                "feature_group": "delta_latent_target",
            }
        )
    pd.DataFrame(schema_rows).to_csv(
        out / "feature_schema.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_metrics = metrics.loc[
        metrics["latent_dim"].eq(selected_k)
    ].sort_values("split")

    summary = "\n".join(
        [
            f"Curve-change latent dataset - {args.year}",
            "=" * 80,
            "",
            f"Fit rows = {len(Xfit):,}",
            f"Selected delta latent dimension = {selected_k}",
            (
                "Cumulative TRAIN explained variance = "
                f"{cumulative[selected_k - 1]:.6f}"
            ),
            f"Source model features = {len(ds_info['source_model_features'])}",
            f"Delta-history features = {len(ds_info['delta_history_features'])}",
            f"Total model features = {len(ds_info['model_features'])}",
            "",
            "Selected-dimension delta reconstruction:",
            selected_metrics.to_string(index=False),
            "",
            "All candidate dimensions:",
            metrics.to_string(index=False),
        ]
    )
    (out / "summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
