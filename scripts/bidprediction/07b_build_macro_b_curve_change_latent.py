#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07b_build_macro_b_curve_change_latent.py

Refit DeltaCurve PCA ONLY on the 07a stable Macro-B subset and rebuild
Macro-B-specific historical DeltaZ features.

Source:
    macro_b_filtered_dataset/

Output:
    macro_b_curve_change_latent_dataset/

Run:
python scripts/bidprediction/07b_build_macro_b_curve_change_latent.py --year 2025 --overwrite
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


SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
PREV_PRICE_COLS = [f"_curvevec_p{i:02d}_lag1" for i in range(21)]
RAW_PREV_COLS = [
    *PREV_PRICE_COLS,
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
    "prediction_cutoff_utc",
]

CURVE_COLS = [
    *SHAPE_COLS,
    "p_anchor",
    "p_span",
    "q_anchor_mw",
    "q_span_mw",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def files_from_manifest(manifest, split):
    return [
        x["file"] if isinstance(x, dict) else x
        for x in manifest["parts"][split]
    ]


def current_vector(d):
    shape = d[SHAPE_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(np.float64)

    pa = num(d["p_anchor"]).to_numpy(np.float64)
    ps = num(d["p_span"]).to_numpy(np.float64)

    zero = np.abs(ps) <= 1e-12
    if zero.any():
        shape[zero] = np.nan_to_num(
            shape[zero],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    price = (
        pa[:, None]
        + ps[:, None] * shape
    )

    qa = num(d["q_anchor_mw"]).to_numpy(np.float64)
    qs = num(d["q_span_mw"]).to_numpy(np.float64)

    return np.c_[
        price,
        qa,
        np.log(
            np.maximum(
                qs,
                1e-8,
            )
        ),
    ]


def previous_vector(d):
    return np.column_stack(
        [
            *[
                num(d[c]).to_numpy(np.float64)
                for c in PREV_PRICE_COLS
            ],
            num(
                d["_curvevec_q_anchor_lag1"]
            ).to_numpy(np.float64),
            num(
                d["_curvevec_log_q_span_lag1"]
            ).to_numpy(np.float64),
        ]
    )


def load_fit_sample(
    source,
    manifest,
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
        p = source / rel

        print(
            f"[fit {i}/{len(files)}] {p.name}",
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
                    + 1009 * i
                ),
            )

        delta = (
            current_vector(d)
            - previous_vector(d)
        )

        blocks.append(delta)

        del d, delta
        gc.collect()

    if not blocks:
        raise ValueError(
            "No Macro-B TRAIN rows."
        )

    x = np.vstack(blocks)

    if len(x) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            len(x),
            size=max_rows,
            replace=False,
        )
        x = x[idx]

    return x


def eval_dims(
    source,
    manifest,
    split,
    scaler,
    pca,
    dims,
):
    state = {
        k: {
            "rows": 0,
            "price_ae": 0.0,
            "price_se": 0.0,
            "price_n": 0,
            "vector_ae": 0.0,
            "vector_n": 0,
        }
        for k in dims
    }

    files = files_from_manifest(
        manifest,
        split,
    )

    for i, rel in enumerate(files, 1):
        p = source / rel
        print(
            f"[{split} {i}/{len(files)}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(p)

        if d.empty:
            continue

        delta = (
            current_vector(d)
            - previous_vector(d)
        )

        zfull = pca.transform(
            scaler.transform(delta)
        )

        for k in dims:
            z = np.zeros_like(zfull)
            z[:, :k] = zfull[:, :k]

            rec = scaler.inverse_transform(
                pca.inverse_transform(z)
            )

            err = rec - delta
            ae = np.abs(err)

            st = state[k]
            st["rows"] += len(delta)
            st["price_ae"] += float(
                ae[:, :21].sum()
            )
            st["price_se"] += float(
                np.square(
                    err[:, :21]
                ).sum()
            )
            st["price_n"] += int(
                err[:, :21].size
            )
            st["vector_ae"] += float(
                ae.sum()
            )
            st["vector_n"] += int(
                ae.size
            )

        del d, delta, zfull
        gc.collect()

    cum = np.cumsum(
        pca.explained_variance_ratio_
    )

    rows = []

    for k in dims:
        st = state[k]

        rows.append(
            {
                "split": split,
                "latent_dim": k,
                "rows": st["rows"],
                "cumulative_explained_variance": float(
                    cum[k - 1]
                ),
                "delta_vector_mae": (
                    st["vector_ae"]
                    / max(
                        st["vector_n"],
                        1,
                    )
                ),
                "delta_price_mae": (
                    st["price_ae"]
                    / max(
                        st["price_n"],
                        1,
                    )
                ),
                "delta_price_rmse": float(
                    np.sqrt(
                        st["price_se"]
                        / max(
                            st["price_n"],
                            1,
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def time_col(d):
    for c in [
        "timestamp_utc",
        "timestamp_local",
        "local_date",
    ]:
        if c in d.columns:
            return c

    raise KeyError(
        "Need timestamp_utc, timestamp_local, or local_date."
    )


def time_values(d, c):
    return pd.to_datetime(
        d[c],
        errors="coerce",
        utc=(c == "timestamp_utc"),
    )


def grouped_rolling(
    frame,
    shifted,
    keys,
    window,
    stat,
):
    g = shifted.groupby(
        [
            frame[k]
            for k in keys
        ],
        sort=False,
    )

    r = g.rolling(
        window=window,
        min_periods=2,
    )

    out = (
        r.mean()
        if stat == "mean"
        else r.std(ddof=0)
    )

    return out.reset_index(
        level=list(
            range(
                len(keys)
            )
        ),
        drop=True,
    )


def add_history(
    current,
    tail,
    delta_cols,
    max_history,
):
    keys = [
        "participant_id",
        "local_slot_seconds",
    ]

    current = (
        current.copy()
        .reset_index(drop=True)
    )

    current["_rowid"] = np.arange(
        len(current),
        dtype=np.int64,
    )

    tc = time_col(current)

    state = pd.DataFrame(
        {
            "participant_id": current[
                "participant_id"
            ].to_numpy(),
            "local_slot_seconds": current[
                "local_slot_seconds"
            ].to_numpy(),
            "_time": time_values(
                current,
                tc,
            ).to_numpy(),
            "_is_current": np.ones(
                len(current),
                dtype=np.int8,
            ),
            "_rowid": current[
                "_rowid"
            ].to_numpy(),
        }
    )

    state = pd.concat(
        [
            state,
            current[
                delta_cols
            ].astype(
                np.float32
            ).reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    carry_cols = [
        *keys,
        "_time",
        *delta_cols,
    ]

    if tail is None:
        combined = state
    else:
        hist = tail[
            carry_cols
        ].copy()

        hist["_is_current"] = 0
        hist["_rowid"] = -1

        hist = hist[
            state.columns
        ]

        combined = pd.concat(
            [
                hist,
                state,
            ],
            ignore_index=True,
            copy=False,
        )

    combined = (
        combined.sort_values(
            [
                *keys,
                "_time",
                "_is_current",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    g = combined.groupby(
        keys,
        sort=False,
    )

    data = {}

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

        data.update(
            {
                f"{z}_lag1": lag1.astype(
                    np.float32
                ),
                f"{z}_lag2": lag2.astype(
                    np.float32
                ),
                f"{z}_lag7": lag7.astype(
                    np.float32
                ),
                f"{z}_mean7": mean7.astype(
                    np.float32
                ),
                f"{z}_std7": std7.astype(
                    np.float32
                ),
                f"{z}_mean30": mean30.astype(
                    np.float32
                ),
                f"{z}_std30": std30.astype(
                    np.float32
                ),
                f"{z}_delta1": (
                    lag1 - lag2
                ).astype(np.float32),
                f"{z}_delta7": (
                    lag1 - lag7
                ).astype(np.float32),
                f"{z}_mean_gap_7_30": (
                    mean7 - mean30
                ).astype(np.float32),
            }
        )

    h = pd.DataFrame(
        data,
        index=combined.index,
    )

    combined = pd.concat(
        [
            combined,
            h,
        ],
        axis=1,
        copy=False,
    )

    hist_cols = list(
        h.columns
    )

    current_hist = (
        combined.loc[
            combined[
                "_is_current"
            ].eq(1),
            [
                "_rowid",
                *hist_cols,
            ],
        ]
        .sort_values("_rowid")
        .reset_index(drop=True)
    )

    enriched = pd.concat(
        [
            current.drop(
                columns=["_rowid"]
            ).reset_index(
                drop=True
            ),
            current_hist[
                hist_cols
            ],
        ],
        axis=1,
        copy=False,
    )

    new_tail = (
        combined[
            carry_cols
        ]
        .groupby(
            keys,
            sort=False,
            as_index=False,
            group_keys=False,
        )
        .tail(max_history)
        .copy()
        .reset_index(drop=True)
    )

    return enriched, new_tail, hist_cols


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--source-dir",
        default="macro_b_filtered_dataset",
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
        "--max-history",
        type=int,
        default=30,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    source = base / args.source_dir

    manifest = json.loads(
        (
            source
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    out_dir = (
        base
        / "macro_b_curve_change_latent_dataset"
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
        f"07b Macro-B DeltaCurve PCA - {args.year}"
    )
    print("=" * 80)

    xfit = load_fit_sample(
        source,
        manifest,
        args.max_fit_rows,
        args.seed,
    )

    scaler = StandardScaler()
    xs = scaler.fit_transform(xfit)

    max_components = min(
        args.max_components,
        xfit.shape[1],
    )

    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=args.seed,
    )
    pca.fit(xs)

    cum = np.cumsum(
        pca.explained_variance_ratio_
    )

    hit = np.where(
        cum >= args.variance_target
    )[0]

    selected_k = (
        int(hit[0] + 1)
        if len(hit)
        else max_components
    )

    dims = sorted(
        {
            *[
                int(x)
                for x in args.candidate_dims.split(",")
                if x.strip()
            ],
            selected_k,
            max_components,
        }
    )

    dims = [
        k
        for k in dims
        if 1 <= k <= max_components
    ]

    metrics = pd.concat(
        [
            eval_dims(
                source,
                manifest,
                split,
                scaler,
                pca,
                dims,
            )
            for split in [
                "val",
                "test",
            ]
        ],
        ignore_index=True,
    )

    metrics.to_csv(
        out_dir
        / "delta_latent_dimension_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        {
            "version": "macro-b-delta-pca-v1",
            "scaler": scaler,
            "pca": pca,
            "selected_delta_latent_dim": selected_k,
            "variance_target": args.variance_target,
        },
        out_dir
        / "delta_pca_bundle.joblib",
        compress=3,
    )

    source_features = list(
        manifest[
            "source_model_features"
        ]
    )

    delta_cols = [
        f"delta_latent_z{i+1:02d}"
        for i in range(selected_k)
    ]

    tail = None
    out_parts = {}
    split_stats = {}
    history_cols = None

    for split in [
        "train",
        "val",
        "test",
    ]:
        files = files_from_manifest(
            manifest,
            split,
        )

        split_dir = (
            out_dir
            / "parts"
            / split
        )
        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        written = []
        total = 0

        for i, rel in enumerate(files, 1):
            p = source / rel

            print(
                f"[write {split} {i}/{len(files)}] {p.name}",
                flush=True,
            )

            d = pd.read_pickle(p)

            if d.empty:
                continue

            delta = (
                current_vector(d)
                - previous_vector(d)
            )

            z = pca.transform(
                scaler.transform(delta)
            )[
                :,
                :selected_k,
            ].astype(np.float32)

            for j, c in enumerate(delta_cols):
                d[c] = z[:, j]

            enriched, tail, hc = add_history(
                d,
                tail,
                delta_cols,
                args.max_history,
            )

            if history_cols is None:
                history_cols = hc

            meta = [
                c
                for c in META_CANDIDATES
                if c in enriched.columns
            ]

            keep = list(
                dict.fromkeys(
                    [
                        *meta,
                        *source_features,
                        *history_cols,
                        *delta_cols,
                        *CURVE_COLS,
                        *RAW_PREV_COLS,
                    ]
                )
            )

            if "y_template_id" in enriched.columns:
                keep.append("y_template_id")

            out_d = enriched[
                keep
            ].copy()

            name = (
                f"{split}_macro_b_delta_"
                f"{i:04d}.pkl"
            )

            path = split_dir / name
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

            total += len(out_d)

            del d, delta, z, enriched, out_d
            gc.collect()

        out_parts[split] = written
        split_stats[split] = {
            "rows": int(total)
        }

    model_features = list(
        dict.fromkeys(
            [
                *source_features,
                *history_cols,
            ]
        )
    )

    out_manifest = {
        "year": args.year,
        "source_dataset": str(source),
        "selected_delta_latent_dim": selected_k,
        "delta_columns": delta_cols,
        "source_model_features": source_features,
        "delta_history_features": history_cols,
        "model_features": model_features,
        "parts": out_parts,
        "split_stats": split_stats,
    }

    (
        out_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            out_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selected = metrics.loc[
        metrics["latent_dim"].eq(
            selected_k
        )
    ]

    summary = "\n".join(
        [
            (
                f"07b Macro-B DeltaCurve PCA - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            f"Fit rows = {len(xfit):,}",
            (
                f"Selected delta latent dimension = "
                f"{selected_k}"
            ),
            (
                "TRAIN cumulative explained variance = "
                f"{cum[selected_k - 1]:.6f}"
            ),
            (
                f"Source features = "
                f"{len(source_features)}"
            ),
            (
                f"Macro-B delta-history features = "
                f"{len(history_cols)}"
            ),
            (
                f"Total model features = "
                f"{len(model_features)}"
            ),
            "",
            "Selected dimension reconstruction:",
            selected.to_string(index=False),
            "",
            "All candidate dimensions:",
            metrics.to_string(index=False),
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
