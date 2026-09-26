#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09a_diagnose_absolute_prediction_families.py

Purpose
-------
Diagnose whether a small number of coarse "prediction families" can partition
the Macro-B ABSOLUTE bid-curve target space into substantially more homogeneous
sub-populations.

This script DOES NOT train the final bid predictor.

Workflow
--------
1) Read the Macro-B absolute-curve PCA representation from 08a.
2) Transform TRAIN Macro-B curves into absolute latent Z.
3) Fit TRAIN-only MiniBatchKMeans for K = 2..8 (configurable).
4) Freeze each clustering model.
5) Assign TRAIN / VAL / TEST curves to the nearest TRAIN centroid using the
   realized TRUE absolute latent vector. This is therefore an ORACLE-family
   diagnostic, not deployable routing.
6) Measure:
   - family sample balance;
   - within-family latent dispersion / global latent dispersion;
   - latent quantization RMSE;
   - oracle-family-mean reconstructed curve WAPE / MAE / RMSE;
   - standard unsupervised clustering indices on the TRAIN fit sample.
7) Decode every family centroid back to an absolute bid curve so each family
   can be inspected in physical price / quantity coordinates.

Important interpretation
------------------------
This is NOT:
    X -> family -> curve prediction

It is only:
    TRUE current curve -> nearest TRAIN family centroid

The question is:
    "Does a coarse partition of absolute curve space materially reduce target
     heterogeneity enough to justify training a future family classifier and
     family-specific regressors?"

Source
------
data/processed/bidprediction/<year>/
    macro_b_filtered_dataset/
    macro_b_absolute_curve_representation/

Outputs
-------
data/processed/bidprediction/<year>/absolute_prediction_family_diagnostics/
    k_summary.csv
    family_statistics.csv
    family_centroids.csv
    train_fit_cluster_indices.csv
    cluster_models.joblib
    manifest.json
    summary.txt

Run
---
python scripts/bidprediction/09a_diagnose_absolute_prediction_families.py \
    --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]


def num(s):
    return pd.to_numeric(
        s,
        errors="coerce",
    )


def part_files(manifest, split):
    return [
        x["file"]
        if isinstance(x, dict)
        else x
        for x in manifest["parts"][split]
    ]


def current_vector(d):
    """
    Same absolute vector definition as 08a:
        [21 absolute prices, q_anchor_mw, log(q_span_mw)]
    """
    shape = (
        d[SHAPE_COLS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            np.float64
        )
    )

    p_anchor = num(
        d["p_anchor"]
    ).to_numpy(
        np.float64
    )

    p_span = num(
        d["p_span"]
    ).to_numpy(
        np.float64
    )

    flat = (
        np.abs(
            p_span
        )
        <= 1e-12
    )

    if flat.any():
        shape[
            flat,
            :,
        ] = np.nan_to_num(
            shape[
                flat,
                :,
            ],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    price = (
        p_anchor[:, None]
        + p_span[:, None]
        * shape
    )

    q_anchor = num(
        d[
            "q_anchor_mw"
        ]
    ).to_numpy(
        np.float64
    )

    q_span = num(
        d[
            "q_span_mw"
        ]
    ).to_numpy(
        np.float64
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


def true_quantity_from_vector(v):
    qa = v[:, 21]

    qs = np.exp(
        np.clip(
            v[:, 22],
            -20.0,
            20.0,
        )
    )

    return (
        qa[:, None]
        + qs[:, None]
        * GRID[None, :]
    )


def unpack_vector(v):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    return (
        v[:, :21],
        v[:, 21],
        np.exp(
            np.clip(
                v[:, 22],
                -20.0,
                20.0,
            )
        ),
    )


def price_on_true_q(
    true_q,
    pred_price,
    pred_q_anchor,
    pred_q_span,
):
    x = (
        true_q
        - pred_q_anchor[:, None]
    ) / np.maximum(
        pred_q_span[:, None],
        1e-8,
    )

    pos = (
        np.clip(
            x,
            0.0,
            1.0,
        )
        * 20.0
    )

    lo = np.floor(
        pos
    ).astype(
        np.int16
    )

    hi = np.minimum(
        lo + 1,
        20,
    )

    frac = (
        pos
        - lo
    )

    plo = np.take_along_axis(
        pred_price,
        lo,
        axis=1,
    )

    phi = np.take_along_axis(
        pred_price,
        hi,
        axis=1,
    )

    return (
        plo
        + frac
        * (
            phi
            - plo
        )
    )


def to_latent(
    v,
    bundle,
    latent_dim,
):
    return (
        bundle[
            "pca"
        ]
        .transform(
            bundle[
                "scaler"
            ].transform(
                v
            )
        )[
            :,
            :latent_dim,
        ]
    )


def decode_latent(
    z,
    bundle,
):
    z = np.asarray(
        z,
        dtype=np.float64,
    )

    pca = bundle["pca"]
    scaler = bundle["scaler"]

    full = np.zeros(
        (
            len(z),
            int(
                pca.n_components_
            ),
        ),
        dtype=np.float64,
    )

    full[
        :,
        :z.shape[1],
    ] = z

    return scaler.inverse_transform(
        pca.inverse_transform(
            full
        )
    )


def load_train_fit_sample(
    source_dir,
    source_manifest,
    bundle,
    latent_dim,
    max_rows,
    seed,
):
    files = part_files(
        source_manifest,
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

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            source_dir
            / rel
        )

        print(
            f"[TRAIN fit sample {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        n = min(
            len(d),
            per_part,
        )

        if n < len(d):
            d = d.sample(
                n=n,
                random_state=(
                    seed
                    + 1009 * i
                ),
            )

        v = current_vector(
            d
        )

        z = np.asarray(
            to_latent(
                v,
                bundle,
                latent_dim,
            ),
            dtype=np.float64,
        )

        blocks.append(
            z
        )

        del d, v, z
        gc.collect()

    if not blocks:
        raise ValueError(
            "No TRAIN rows available."
        )

    x = np.vstack(
        blocks
    )

    if len(x) > max_rows:
        rng = np.random.default_rng(
            seed
        )

        idx = rng.choice(
            len(x),
            size=max_rows,
            replace=False,
        )

        x = x[
            idx
        ]

    return x


def safe_cluster_indices(
    x,
    labels,
    silhouette_rows,
    seed,
):
    n_clusters = len(
        np.unique(
            labels
        )
    )

    out = {
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }

    if (
        len(x) <= n_clusters
        or n_clusters < 2
    ):
        return out

    try:
        out[
            "calinski_harabasz"
        ] = float(
            calinski_harabasz_score(
                x,
                labels,
            )
        )
    except Exception:
        pass

    try:
        out[
            "davies_bouldin"
        ] = float(
            davies_bouldin_score(
                x,
                labels,
            )
        )
    except Exception:
        pass

    try:
        sample_size = min(
            int(
                silhouette_rows
            ),
            len(x),
        )

        if sample_size >= 100:
            out[
                "silhouette"
            ] = float(
                silhouette_score(
                    x,
                    labels,
                    metric="euclidean",
                    sample_size=sample_size,
                    random_state=seed,
                )
            )
    except Exception:
        pass

    return out


def fit_cluster_models(
    x,
    k_values,
    args,
):
    models = {}
    index_rows = []

    for k in k_values:
        print(
            f"[fit K={k}]",
            flush=True,
        )

        model = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=args.n_init,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            reassignment_ratio=0.01,
            random_state=args.seed,
        )

        x64 = np.asarray(
            x,
            dtype=np.float64,
        )

        labels = model.fit_predict(
            x64
        )

        indices = safe_cluster_indices(
            x64,
            labels,
            args.silhouette_rows,
            args.seed,
        )

        counts = np.bincount(
            labels,
            minlength=k,
        )

        index_rows.append(
            {
                "k": int(k),
                "fit_rows": int(
                    len(x)
                ),
                "inertia": float(
                    model.inertia_
                ),
                "min_fit_cluster_rows": int(
                    counts.min()
                ),
                "min_fit_cluster_share": float(
                    counts.min()
                    / len(x)
                ),
                "max_fit_cluster_share": float(
                    counts.max()
                    / len(x)
                ),
                **indices,
            }
        )

        models[
            int(k)
        ] = model

    return (
        models,
        pd.DataFrame(
            index_rows
        ),
    )


def global_latent_train_stats(
    source_dir,
    source_manifest,
    bundle,
    latent_dim,
):
    n = 0
    sum_z = np.zeros(
        latent_dim,
        dtype=np.float64,
    )
    sum_sq = 0.0

    files = part_files(
        source_manifest,
        "train",
    )

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            source_dir
            / rel
        )

        print(
            f"[global TRAIN stats {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        v = current_vector(
            d
        )

        z = to_latent(
            v,
            bundle,
            latent_dim,
        )

        n += len(z)
        sum_z += z.sum(
            axis=0
        )
        sum_sq += float(
            np.square(
                z
            ).sum()
        )

        del d, v, z
        gc.collect()

    if n <= 0:
        raise ValueError(
            "No TRAIN rows for global stats."
        )

    mean = (
        sum_z
        / n
    )

    sse = (
        sum_sq
        - n
        * float(
            np.square(
                mean
            ).sum()
        )
    )

    sse = max(
        sse,
        0.0,
    )

    rms = float(
        np.sqrt(
            sse
            / n
        )
    )

    return {
        "rows": int(n),
        "mean": mean,
        "sse": float(
            sse
        ),
        "rms_distance": rms,
    }


def new_eval_state(k):
    return {
        "rows": 0,
        "latent_sq_error": 0.0,
        "latent_points": 0,
        "price_ae": 0.0,
        "price_se": 0.0,
        "price_abs_true": 0.0,
        "price_points": 0,
        "q_anchor_ae": 0.0,
        "q_anchor_abs_true": 0.0,
        "q_span_ae": 0.0,
        "q_span_abs_true": 0.0,
        "cluster_rows": np.zeros(
            k,
            dtype=np.int64,
        ),
        "cluster_latent_sse": np.zeros(
            k,
            dtype=np.float64,
        ),
    }


def evaluate_split_for_all_k(
    source_dir,
    source_manifest,
    split,
    bundle,
    latent_dim,
    models,
    global_train_rms,
):
    states = {
        k: new_eval_state(
            k
        )
        for k in models
    }

    files = part_files(
        source_manifest,
        split,
    )

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            source_dir
            / rel
        )

        print(
            f"[evaluate {split} {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        true_v = current_vector(
            d
        )

        true_z = np.asarray(
            to_latent(
                true_v,
                bundle,
                latent_dim,
            ),
            dtype=np.float64,
        )

        true_p = true_v[
            :,
            :21,
        ]

        true_q = true_quantity_from_vector(
            true_v
        )

        true_qa = true_v[
            :,
            21,
        ]

        true_qs = np.exp(
            np.clip(
                true_v[
                    :,
                    22,
                ],
                -20.0,
                20.0,
            )
        )

        for k, model in models.items():
            predict_z = np.asarray(
                true_z,
                dtype=model.cluster_centers_.dtype,
            )

            labels = model.predict(
                predict_z
            )

            centers = model.cluster_centers_[
                labels
            ]

            latent_diff = (
                predict_z
                - centers
            )

            latent_row_sse = np.square(
                latent_diff
            ).sum(
                axis=1
            )

            pred_v = decode_latent(
                centers,
                bundle,
            )

            pred_p, pred_qa, pred_qs = unpack_vector(
                pred_v
            )

            pred_on_true = price_on_true_q(
                true_q,
                pred_p,
                pred_qa,
                pred_qs,
            )

            err = (
                pred_on_true
                - true_p
            )

            ae = np.abs(
                err
            )

            st = states[
                k
            ]

            st[
                "rows"
            ] += len(
                d
            )

            st[
                "latent_sq_error"
            ] += float(
                latent_row_sse.sum()
            )

            st[
                "latent_points"
            ] += int(
                len(d)
                * latent_dim
            )

            st[
                "price_ae"
            ] += float(
                ae.sum()
            )

            st[
                "price_se"
            ] += float(
                np.square(
                    err
                ).sum()
            )

            st[
                "price_abs_true"
            ] += float(
                np.abs(
                    true_p
                ).sum()
            )

            st[
                "price_points"
            ] += int(
                ae.size
            )

            st[
                "q_anchor_ae"
            ] += float(
                np.abs(
                    pred_qa
                    - true_qa
                ).sum()
            )

            st[
                "q_anchor_abs_true"
            ] += float(
                np.abs(
                    true_qa
                ).sum()
            )

            st[
                "q_span_ae"
            ] += float(
                np.abs(
                    pred_qs
                    - true_qs
                ).sum()
            )

            st[
                "q_span_abs_true"
            ] += float(
                np.abs(
                    true_qs
                ).sum()
            )

            st[
                "cluster_rows"
            ] += np.bincount(
                labels,
                minlength=k,
            )

            for c in range(
                k
            ):
                mask = (
                    labels
                    == c
                )

                if mask.any():
                    st[
                        "cluster_latent_sse"
                    ][
                        c
                    ] += float(
                        latent_row_sse[
                            mask
                        ].sum()
                    )

        del (
            d,
            true_v,
            true_z,
            true_p,
            true_q,
            true_qa,
            true_qs,
        )
        gc.collect()

    summary_rows = []
    family_rows = []

    for k, st in states.items():
        row_rms = float(
            np.sqrt(
                st[
                    "latent_sq_error"
                ]
                / max(
                    st[
                        "rows"
                    ],
                    1,
                )
            )
        )

        summary_rows.append(
            {
                "split": split,
                "k": int(
                    k
                ),
                "rows": int(
                    st[
                        "rows"
                    ]
                ),
                "active_families": int(
                    (
                        st[
                            "cluster_rows"
                        ]
                        > 0
                    ).sum()
                ),
                "min_family_rows": int(
                    st[
                        "cluster_rows"
                    ].min()
                ),
                "min_family_share": float(
                    st[
                        "cluster_rows"
                    ].min()
                    / max(
                        st[
                            "rows"
                        ],
                        1,
                    )
                ),
                "max_family_share": float(
                    st[
                        "cluster_rows"
                    ].max()
                    / max(
                        st[
                            "rows"
                        ],
                        1,
                    )
                ),
                "latent_quantization_rmse_per_dimension": float(
                    np.sqrt(
                        st[
                            "latent_sq_error"
                        ]
                        / max(
                            st[
                                "latent_points"
                            ],
                            1,
                        )
                    )
                ),
                "latent_rms_distance_to_family_center": row_rms,
                "within_family_dispersion_ratio_vs_global_train": float(
                    row_rms
                    / global_train_rms
                )
                if global_train_rms > 0
                else np.nan,
                "oracle_family_mean_price_mae": float(
                    st[
                        "price_ae"
                    ]
                    / max(
                        st[
                            "price_points"
                        ],
                        1,
                    )
                ),
                "oracle_family_mean_price_rmse": float(
                    np.sqrt(
                        st[
                            "price_se"
                        ]
                        / max(
                            st[
                                "price_points"
                            ],
                            1,
                        )
                    )
                ),
                "oracle_family_mean_price_wape_pct": float(
                    100.0
                    * st[
                        "price_ae"
                    ]
                    / max(
                        st[
                            "price_abs_true"
                        ],
                        1e-12,
                    )
                ),
                "oracle_family_mean_q_anchor_wape_pct": float(
                    100.0
                    * st[
                        "q_anchor_ae"
                    ]
                    / max(
                        st[
                            "q_anchor_abs_true"
                        ],
                        1e-12,
                    )
                ),
                "oracle_family_mean_q_span_wape_pct": float(
                    100.0
                    * st[
                        "q_span_ae"
                    ]
                    / max(
                        st[
                            "q_span_abs_true"
                        ],
                        1e-12,
                    )
                ),
            }
        )

        for c in range(
            k
        ):
            n = int(
                st[
                    "cluster_rows"
                ][
                    c
                ]
            )

            family_rms = (
                float(
                    np.sqrt(
                        st[
                            "cluster_latent_sse"
                        ][
                            c
                        ]
                        / n
                    )
                )
                if n > 0
                else np.nan
            )

            family_rows.append(
                {
                    "split": split,
                    "k": int(
                        k
                    ),
                    "family_id": (
                        f"F{c:02d}"
                    ),
                    "rows": n,
                    "share": float(
                        n
                        / max(
                            st[
                                "rows"
                            ],
                            1,
                        )
                    ),
                    "latent_rms_distance_to_center": family_rms,
                    "dispersion_ratio_vs_global_train": (
                        float(
                            family_rms
                            / global_train_rms
                        )
                        if (
                            n > 0
                            and global_train_rms > 0
                        )
                        else np.nan
                    ),
                }
            )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.DataFrame(
            family_rows
        ),
    )


def centroid_table(
    models,
    bundle,
    latent_dim,
):
    rows = []

    for k, model in models.items():
        centers = (
            model.cluster_centers_[
                :,
                :latent_dim,
            ]
        )

        decoded = decode_latent(
            centers,
            bundle,
        )

        p, qa, qs = unpack_vector(
            decoded
        )

        for c in range(
            k
        ):
            row = {
                "k": int(
                    k
                ),
                "family_id": (
                    f"F{c:02d}"
                ),
                "centroid_price_mean": float(
                    p[
                        c
                    ].mean()
                ),
                "centroid_price_min": float(
                    p[
                        c
                    ].min()
                ),
                "centroid_price_max": float(
                    p[
                        c
                    ].max()
                ),
                "centroid_price_span": float(
                    p[
                        c
                    ].max()
                    - p[
                        c
                    ].min()
                ),
                "centroid_q_anchor_mw": float(
                    qa[
                        c
                    ]
                ),
                "centroid_q_span_mw": float(
                    qs[
                        c
                    ]
                ),
            }

            for j in range(
                latent_dim
            ):
                row[
                    f"centroid_z{j+1:02d}"
                ] = float(
                    centers[
                        c,
                        j,
                    ]
                )

            for j in range(
                21
            ):
                row[
                    f"centroid_price_u{j:02d}"
                ] = float(
                    p[
                        c,
                        j,
                    ]
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def add_k_diagnostics(
    k_summary,
    cluster_indices,
):
    out = k_summary.merge(
        cluster_indices[
            [
                "k",
                "inertia",
                "min_fit_cluster_rows",
                "min_fit_cluster_share",
                "max_fit_cluster_share",
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
            ]
        ],
        on="k",
        how="left",
        validate="many_to_one",
    )

    out = out.sort_values(
        [
            "split",
            "k",
        ]
    ).reset_index(
        drop=True
    )

    out[
        "oracle_family_wape_gain_vs_previous_k_pp"
    ] = np.nan

    for split in out[
        "split"
    ].unique():
        idx = out[
            "split"
        ].eq(
            split
        )

        temp = out.loc[
            idx
        ].sort_values(
            "k"
        )

        gain = (
            temp[
                "oracle_family_mean_price_wape_pct"
            ]
            .shift(
                1
            )
            - temp[
                "oracle_family_mean_price_wape_pct"
            ]
        )

        out.loc[
            temp.index,
            "oracle_family_wape_gain_vs_previous_k_pp",
        ] = gain.to_numpy()

    return out


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
        "--source-dir",
        default="macro_b_filtered_dataset",
    )

    ap.add_argument(
        "--representation-dir",
        default="macro_b_absolute_curve_representation",
    )

    ap.add_argument(
        "--k-values",
        default="2,3,4,5,6,7,8",
    )

    ap.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help=(
            "Default: use 08a selected latent dimension."
        ),
    )

    ap.add_argument(
        "--max-fit-rows",
        type=int,
        default=500_000,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )

    ap.add_argument(
        "--n-init",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--max-iter",
        type=int,
        default=300,
    )

    ap.add_argument(
        "--silhouette-rows",
        type=int,
        default=3000,
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
        Path(
            args.root
        )
        / str(
            args.year
        )
    )

    source_dir = (
        base
        / args.source_dir
    )

    representation_dir = (
        base
        / args.representation_dir
    )

    source_manifest = json.loads(
        (
            source_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    rep_manifest = json.loads(
        (
            representation_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    bundle = joblib.load(
        representation_dir
        / "absolute_pca_bundle.joblib"
    )

    selected_dim = int(
        rep_manifest[
            "selected_latent_dim"
        ]
    )

    latent_dim = (
        int(
            args.latent_dim
        )
        if args.latent_dim
        is not None
        else selected_dim
    )

    if (
        latent_dim < 1
        or latent_dim
        > int(
            bundle[
                "pca"
            ].n_components_
        )
    ):
        raise ValueError(
            f"Invalid latent_dim={latent_dim}."
        )

    k_values = sorted(
        {
            int(
                x.strip()
            )
            for x in args.k_values.split(",")
            if x.strip()
        }
    )

    if not k_values:
        raise ValueError(
            "No K values requested."
        )

    if min(
        k_values
    ) < 2:
        raise ValueError(
            "K must be >= 2."
        )

    out_dir = (
        base
        / "absolute_prediction_family_diagnostics"
    )

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out_dir} exists. "
                "Use --overwrite."
            )

        shutil.rmtree(
            out_dir
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"09a absolute prediction-family diagnostics - "
        f"{args.year}"
    )
    print("=" * 80)
    print(
        f"Absolute latent dimension = {latent_dim}"
    )
    print(
        f"K candidates = {k_values}"
    )
    print(
        "VAL/TEST family assignment uses TRUE current latent "
        "(oracle diagnostic only)."
    )
    print()

    x_fit = load_train_fit_sample(
        source_dir,
        source_manifest,
        bundle,
        latent_dim,
        args.max_fit_rows,
        args.seed,
    )

    print(
        f"TRAIN clustering fit sample = "
        f"{len(x_fit):,}"
    )
    print()

    models, cluster_indices = fit_cluster_models(
        x_fit,
        k_values,
        args,
    )

    cluster_indices.to_csv(
        out_dir
        / "train_fit_cluster_indices.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_stats = global_latent_train_stats(
        source_dir,
        source_manifest,
        bundle,
        latent_dim,
    )

    print()
    print(
        "Global TRAIN latent RMS distance to global mean = "
        f"{global_stats['rms_distance']:.6f}"
    )
    print()

    all_k_summary = []
    all_family_stats = []

    for split in [
        "train",
        "val",
        "test",
    ]:
        summary, family = evaluate_split_for_all_k(
            source_dir,
            source_manifest,
            split,
            bundle,
            latent_dim,
            models,
            global_stats[
                "rms_distance"
            ],
        )

        all_k_summary.append(
            summary
        )

        all_family_stats.append(
            family
        )

    k_summary = pd.concat(
        all_k_summary,
        ignore_index=True,
    )

    family_stats = pd.concat(
        all_family_stats,
        ignore_index=True,
    )

    k_summary = add_k_diagnostics(
        k_summary,
        cluster_indices,
    )

    k_summary.to_csv(
        out_dir
        / "k_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family_stats.to_csv(
        out_dir
        / "family_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    centroids = centroid_table(
        models,
        bundle,
        latent_dim,
    )

    centroids.to_csv(
        out_dir
        / "family_centroids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        {
            "version": "absolute-prediction-family-diagnostic-v1",
            "year": int(
                args.year
            ),
            "latent_dim": int(
                latent_dim
            ),
            "k_values": k_values,
            "models": models,
            "global_train_latent_mean": global_stats[
                "mean"
            ],
            "global_train_latent_rms_distance": global_stats[
                "rms_distance"
            ],
            "note": (
                "Clustering fit on TRAIN only. VAL/TEST labels are oracle "
                "nearest-centroid assignments from realized true current latent."
            ),
        },
        out_dir
        / "cluster_models.joblib",
        compress=3,
    )

    manifest = {
        "year": int(
            args.year
        ),
        "source_dataset": str(
            source_dir
        ),
        "representation_dir": str(
            representation_dir
        ),
        "latent_dim": int(
            latent_dim
        ),
        "k_values": k_values,
        "fit_rows": int(
            len(
                x_fit
            )
        ),
        "clustering": "MiniBatchKMeans",
        "fit_scope": "TRAIN only",
        "assignment_scope": (
            "TRAIN/VAL/TEST assigned by nearest TRAIN centroid using "
            "realized true current absolute latent; oracle diagnostic"
        ),
    }

    (
        out_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    display_cols = [
        "split",
        "k",
        "rows",
        "min_family_share",
        "max_family_share",
        "within_family_dispersion_ratio_vs_global_train",
        "oracle_family_mean_price_wape_pct",
        "oracle_family_mean_price_mae",
        "latent_quantization_rmse_per_dimension",
        "silhouette",
        "davies_bouldin",
        "oracle_family_wape_gain_vs_previous_k_pp",
    ]

    test_rows = (
        k_summary.loc[
            k_summary[
                "split"
            ].eq(
                "test"
            ),
            display_cols,
        ]
        .sort_values(
            "k"
        )
    )

    val_rows = (
        k_summary.loc[
            k_summary[
                "split"
            ].eq(
                "val"
            ),
            display_cols,
        ]
        .sort_values(
            "k"
        )
    )

    train_rows = (
        k_summary.loc[
            k_summary[
                "split"
            ].eq(
                "train"
            ),
            display_cols,
        ]
        .sort_values(
            "k"
        )
    )

    summary_txt = "\n".join(
        [
            (
                f"09a absolute prediction-family diagnostics - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                f"Absolute latent dimension = "
                f"{latent_dim}"
            ),
            (
                f"TRAIN clustering fit rows = "
                f"{len(x_fit):,}"
            ),
            (
                "Global TRAIN latent RMS distance to global mean = "
                f"{global_stats['rms_distance']:.6f}"
            ),
            "",
            (
                "IMPORTANT: VAL/TEST families use TRUE current latent "
                "for nearest-centroid assignment. These are oracle "
                "diagnostics, not deployable forecasts."
            ),
            "",
            "TRAIN:",
            train_rows.to_string(
                index=False
            ),
            "",
            "VALIDATION:",
            val_rows.to_string(
                index=False
            ),
            "",
            "TEST:",
            test_rows.to_string(
                index=False
            ),
            "",
            (
                "Primary evidence to inspect: "
                "(1) within_family_dispersion_ratio_vs_global_train, "
                "(2) oracle_family_mean_price_wape_pct, "
                "(3) min_family_share / max_family_share, and "
                "(4) marginal WAPE gain when K increases."
            ),
            "",
            (
                "A useful coarse family partition should materially reduce "
                "dispersion and oracle-family-mean WAPE without creating "
                "tiny families. This script intentionally does not auto-select "
                "K because K should be chosen from that trade-off, not by "
                "blindly minimizing oracle WAPE."
            ),
        ]
    )

    (
        out_dir
        / "summary.txt"
    ).write_text(
        summary_txt,
        encoding="utf-8",
    )

    print()
    print(
        summary_txt
    )
    print()
    print(
        f"Outputs: {out_dir}"
    )


if __name__ == "__main__":
    main()
