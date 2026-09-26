#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10b_build_final_absolute_curve_latent.py

Final absolute bid-curve representation for the full eligible population.

Curve vector:
    V = [P(u_0), ..., P(u_20), q_anchor_mw, log(q_span_mw)]

Representation:
    TRAIN-only StandardScaler + PCA

FINAL dimension:
    fixed at 8D by the frozen Stage-3 route.
    This script does NOT re-select dimension from the current data.

For audit, reconstruction metrics are also reported for several candidate
dimensions, but final_latent_dim remains fixed at --latent-dim (default 8).

Input
-----
data/processed/bidprediction/<year>/final_feature_only_dataset/

Output
------
data/processed/bidprediction/<year>/final_absolute_curve_dataset/
    parts/{train,val,test}/*.pkl
    absolute_pca_bundle.joblib
    representation_metrics.csv
    manifest.json
    summary.txt

Run
---
python scripts/bidprediction/10b_build_final_absolute_curve_latent.py \
    --year 2025 --overwrite
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
    return pd.to_numeric(
        s,
        errors="coerce",
    )


def parts(manifest, split):
    return [
        x["file"]
        if isinstance(x, dict)
        else x
        for x in manifest["parts"][split]
    ]


def curve_vector(d):
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

    q_anchor = num(
        d["q_anchor_mw"]
    ).to_numpy(
        np.float64
    )

    q_span = num(
        d["q_span_mw"]
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

    if not np.isfinite(
        shape
    ).all():
        raise ValueError(
            "Nonfinite shape remains on non-FLAT rows."
        )

    price = (
        p_anchor[:, None]
        + p_span[:, None]
        * shape
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


def true_quantity(v):
    qa = v[
        :,
        21,
    ]

    qs = np.exp(
        np.clip(
            v[
                :,
                22,
            ],
            -20.0,
            20.0,
        )
    )

    return (
        qa[:, None]
        + qs[:, None]
        * GRID[None, :]
    )


def unpack(v):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    return (
        v[
            :,
            :21,
        ],
        v[
            :,
            21,
        ],
        np.exp(
            np.clip(
                v[
                    :,
                    22,
                ],
                -20.0,
                20.0,
            )
        ),
    )


def price_on_true_q(
    true_q,
    pred_p,
    pred_qa,
    pred_qs,
):
    pos = np.clip(
        (
            true_q
            - pred_qa[:, None]
        )
        / np.maximum(
            pred_qs[:, None],
            1e-8,
        ),
        0.0,
        1.0,
    ) * 20.0

    lo = np.floor(
        pos
    ).astype(
        np.int16
    )

    hi = np.minimum(
        lo + 1,
        20,
    )

    frac = pos - lo

    plo = np.take_along_axis(
        pred_p,
        lo,
        axis=1,
    )

    phi = np.take_along_axis(
        pred_p,
        hi,
        axis=1,
    )

    return (
        plo
        + frac
        * (
            phi - plo
        )
    )


def load_train_fit_sample(
    source,
    manifest,
    max_rows,
    seed,
):
    fs = parts(
        manifest,
        "train",
    )

    per_part = max(
        1,
        int(
            math.ceil(
                max_rows
                / max(
                    len(fs),
                    1,
                )
            )
        ),
    )

    blocks = []

    for i, rel in enumerate(
        fs,
        1,
    ):
        path = (
            source
            / rel
        )

        print(
            f"[fit TRAIN {i}/{len(fs)}] {path.name}",
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

        blocks.append(
            curve_vector(
                d
            )
        )

        del d
        gc.collect()

    if not blocks:
        raise ValueError(
            "No TRAIN rows for PCA fit."
        )

    x = np.vstack(
        blocks
    )

    if len(
        x
    ) > max_rows:
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


def decode_dim(
    zfull,
    k,
    pca,
    scaler,
):
    z = np.zeros_like(
        zfull
    )

    z[
        :,
        :k,
    ] = zfull[
        :,
        :k,
    ]

    return scaler.inverse_transform(
        pca.inverse_transform(
            z
        )
    )


def evaluate_reconstruction(
    source,
    manifest,
    split,
    scaler,
    pca,
    dims,
):
    stats = {
        k: {
            "rows": 0,
            "ae": 0.0,
            "se": 0.0,
            "abst": 0.0,
            "n": 0,
            "qa": 0.0,
            "qaa": 0.0,
            "qs": 0.0,
            "qsa": 0.0,
        }
        for k in dims
    }

    fs = parts(
        manifest,
        split,
    )

    for i, rel in enumerate(
        fs,
        1,
    ):
        path = (
            source
            / rel
        )

        print(
            f"[representation {split} {i}/{len(fs)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        true_v = curve_vector(
            d
        )

        true_p = true_v[
            :,
            :21,
        ]

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

        tq = true_quantity(
            true_v
        )

        zfull = pca.transform(
            scaler.transform(
                true_v
            )
        )

        for k in dims:
            rec = decode_dim(
                zfull,
                k,
                pca,
                scaler,
            )

            rp, rqa, rqs = unpack(
                rec
            )

            pred = price_on_true_q(
                tq,
                rp,
                rqa,
                rqs,
            )

            err = (
                pred
                - true_p
            )

            ae = np.abs(
                err
            )

            st = stats[
                k
            ]

            st[
                "rows"
            ] += len(
                d
            )

            st[
                "ae"
            ] += float(
                ae.sum()
            )

            st[
                "se"
            ] += float(
                np.square(
                    err
                ).sum()
            )

            st[
                "abst"
            ] += float(
                np.abs(
                    true_p
                ).sum()
            )

            st[
                "n"
            ] += int(
                ae.size
            )

            st[
                "qa"
            ] += float(
                np.abs(
                    rqa
                    - true_qa
                ).sum()
            )

            st[
                "qaa"
            ] += float(
                np.abs(
                    true_qa
                ).sum()
            )

            st[
                "qs"
            ] += float(
                np.abs(
                    rqs
                    - true_qs
                ).sum()
            )

            st[
                "qsa"
            ] += float(
                np.abs(
                    true_qs
                ).sum()
            )

        del (
            d,
            true_v,
            true_p,
            true_qa,
            true_qs,
            tq,
            zfull,
        )
        gc.collect()

    cum = np.cumsum(
        pca.explained_variance_ratio_
    )

    rows = []

    for k in dims:
        st = stats[
            k
        ]

        rows.append(
            {
                "split": split,
                "latent_dim": int(
                    k
                ),
                "rows": int(
                    st[
                        "rows"
                    ]
                ),
                "cumulative_explained_variance": float(
                    cum[
                        k - 1
                    ]
                ),
                "price_mae": float(
                    st[
                        "ae"
                    ]
                    / max(
                        st[
                            "n"
                        ],
                        1,
                    )
                ),
                "price_rmse": float(
                    np.sqrt(
                        st[
                            "se"
                        ]
                        / max(
                            st[
                                "n"
                            ],
                            1,
                        )
                    )
                ),
                "price_wape_pct": float(
                    100.0
                    * st[
                        "ae"
                    ]
                    / max(
                        st[
                            "abst"
                        ],
                        1e-12,
                    )
                ),
                "q_anchor_wape_pct": float(
                    100.0
                    * st[
                        "qa"
                    ]
                    / max(
                        st[
                            "qaa"
                        ],
                        1e-12,
                    )
                ),
                "q_span_wape_pct": float(
                    100.0
                    * st[
                        "qs"
                    ]
                    / max(
                        st[
                            "qsa"
                        ],
                        1e-12,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def write_latent_dataset(
    source,
    source_manifest,
    out,
    scaler,
    pca,
    latent_dim,
):
    targets = [
        f"absolute_latent_z{i+1:02d}"
        for i in range(
            latent_dim
        )
    ]

    output_parts = {}
    split_stats = {}

    for split in [
        "train",
        "val",
        "test",
    ]:
        fs = parts(
            source_manifest,
            split,
        )

        split_dir = (
            out
            / "parts"
            / split
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        written = []
        total = 0

        for i, rel in enumerate(
            fs,
            1,
        ):
            path = (
                source
                / rel
            )

            print(
                f"[write {split} {i}/{len(fs)}] "
                f"{path.name}",
                flush=True,
            )

            d = pd.read_pickle(
                path
            )

            if d.empty:
                continue

            v = curve_vector(
                d
            )

            z = (
                pca.transform(
                    scaler.transform(
                        v
                    )
                )[
                    :,
                    :latent_dim,
                ]
                .astype(
                    np.float32
                )
            )

            out_d = d.copy()

            for j, c in enumerate(
                targets
            ):
                out_d[
                    c
                ] = z[
                    :,
                    j,
                ]

            name = (
                f"{split}_final_absolute_"
                f"{i:04d}.pkl"
            )

            out_path = (
                split_dir
                / name
            )

            out_d.to_pickle(
                out_path,
                protocol=5,
            )

            written.append(
                {
                    "file": str(
                        out_path.relative_to(
                            out
                        )
                    ),
                    "rows": int(
                        len(
                            out_d
                        )
                    ),
                }
            )

            total += len(
                out_d
            )

            del d, v, z, out_d
            gc.collect()

        output_parts[
            split
        ] = written

        split_stats[
            split
        ] = {
            "rows": int(
                total
            )
        }

    return (
        targets,
        output_parts,
        split_stats,
    )


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
        default="final_feature_only_dataset",
    )
    ap.add_argument(
        "--latent-dim",
        type=int,
        default=8,
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
        "--candidate-dims",
        default="8",
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

    source = (
        base
        / args.source_dir
    )

    source_manifest = json.loads(
        (
            source
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    max_components = min(
        int(
            args.max_components
        ),
        23,
    )

    if not (
        1
        <= args.latent_dim
        <= max_components
    ):
        raise ValueError(
            "latent_dim must be within fitted PCA components."
        )

    dims = sorted(
        {
            int(
                x.strip()
            )
            for x
            in args.candidate_dims.split(
                ","
            )
            if x.strip()
        }
        | {
            int(
                args.latent_dim
            )
        }
    )

    dims = [
        k
        for k in dims
        if 1
        <= k
        <= max_components
    ]

    out = (
        base
        / "final_absolute_curve_dataset"
    )

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} exists. Use --overwrite."
            )

        shutil.rmtree(
            out
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"10b Final absolute curve latent - {args.year}"
    )
    print("=" * 80)
    print(
        "Curve vector = 21 absolute prices + q_anchor + log(q_span)"
    )
    print(
        f"FINAL latent dimension = {args.latent_dim} (fixed)"
    )
    print(
        f"TRAIN PCA fit cap = {args.max_fit_rows:,}"
    )
    print()

    fit_x = load_train_fit_sample(
        source,
        source_manifest,
        args.max_fit_rows,
        args.seed,
    )

    scaler = StandardScaler()

    fit_scaled = scaler.fit_transform(
        fit_x
    )

    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=args.seed,
    )

    pca.fit(
        fit_scaled
    )

    representation = pd.concat(
        [
            evaluate_reconstruction(
                source,
                source_manifest,
                split,
                scaler,
                pca,
                dims,
            )
            for split
            in [
                "val",
                "test",
            ]
        ],
        ignore_index=True,
    )

    representation.to_csv(
        out
        / "representation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    targets, output_parts, split_stats = write_latent_dataset(
        source,
        source_manifest,
        out,
        scaler,
        pca,
        args.latent_dim,
    )

    joblib.dump(
        {
            "version": "final-absolute-pca-v1",
            "scaler": scaler,
            "pca": pca,
            "final_latent_dim": int(
                args.latent_dim
            ),
            "vector_definition": (
                "[21 absolute prices, q_anchor_mw, log(q_span_mw)]"
            ),
            "fit_scope": "TRAIN only",
            "fit_rows": int(
                len(
                    fit_x
                )
            ),
        },
        out
        / "absolute_pca_bundle.joblib",
        compress=3,
    )

    out_manifest = {
        "version": "final-absolute-curve-dataset-v1",
        "year": int(
            args.year
        ),
        "source_dataset": str(
            source
        ),
        "model_features": source_manifest[
            "model_features"
        ],
        "model_feature_count": int(
            len(
                source_manifest[
                    "model_features"
                ]
            )
        ),
        "latent_columns": targets,
        "final_latent_dim": int(
            args.latent_dim
        ),
        "representation_rule": (
            "TRAIN-only StandardScaler + PCA; final dimension fixed before "
            "final model training"
        ),
        "curve_target_columns": source_manifest[
            "curve_target_columns"
        ],
        "reference_only_columns": source_manifest.get(
            "reference_only_columns",
            [],
        ),
        "parts": output_parts,
        "split_stats": split_stats,
    }

    (
        out
        / "manifest.json"
    ).write_text(
        json.dumps(
            out_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selected = representation.loc[
        representation[
            "latent_dim"
        ].eq(
            args.latent_dim
        )
    ]

    cumulative = np.cumsum(
        pca.explained_variance_ratio_
    )

    summary = "\n".join(
        [
            (
                f"10b Final absolute curve latent - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                f"TRAIN PCA fit rows = "
                f"{len(fit_x):,}"
            ),
            (
                f"FINAL latent dimension = "
                f"{args.latent_dim}"
            ),
            (
                "TRAIN cumulative explained variance at final dim = "
                f"{cumulative[args.latent_dim - 1]:.6f}"
            ),
            "",
            "FINAL-dimension reconstruction:",
            selected.to_string(
                index=False
            ),
            "",
            "Audit across candidate dimensions:",
            representation.to_string(
                index=False
            ),
            "",
            (
                "Dimension is NOT selected from these validation/test "
                "metrics; final dimension is frozen at 8D by the final route."
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
    print(
        summary
    )
    print()
    print(
        f"Outputs: {out}"
    )


if __name__ == "__main__":
    main()
