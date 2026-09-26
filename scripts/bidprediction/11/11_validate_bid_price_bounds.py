#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_validate_bid_price_bounds.py

Single-script diagnostic:
Can the current 83 feature-only inputs predict the bid price lower/upper bounds?

Task
----
X = frozen feature-only inputs from Stage 10
y = [p_min, p_max]

No PCA.
No template routing.
No Prediction Family.
No previous raw bid curve.
No latent / theta history.

Model
-----
- Train-mean baseline
- Random Forest (primary simple diagnostic)

Data split
----------
Use the already frozen chronological train / val / test split from:
data/processed/bidprediction/<year>/final_feature_only_dataset/

Outputs
-------
data/processed/bidprediction/<year>/price_bounds_validation/
    random_forest_bounds.joblib
    metrics.csv
    test_prediction_sample.csv
    pmin_true_vs_pred.png
    pmax_true_vs_pred.png
    pspan_true_vs_pred.png
    summary.txt

Run
---
python scripts/bidprediction/11_validate_bid_price_bounds.py --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import joblib
import matplotlib

# Headless-safe on Windows / remote environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


SHAPE_COLS = [
    f"shape_v{i:02d}"
    for i in range(21)
]

REQUIRED_CURVE_COLS = [
    *SHAPE_COLS,
    "p_anchor",
    "p_span",
]


def part_files(manifest, split):
    vals = manifest.get(
        "parts",
        {},
    ).get(
        split,
        [],
    )

    return [
        x["file"]
        if isinstance(x, dict)
        else x
        for x in vals
    ]


def numeric_frame(d, cols):
    return d[
        cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )


def compute_price_bounds(d):
    """
    Recover the 21 absolute price points exactly as in the curve definition,
    then calculate p_min / p_max.

    FLAT rows can have NaN shape values because p_span == 0. In that case
    the whole absolute-price curve equals p_anchor.
    """
    missing = [
        c
        for c in REQUIRED_CURVE_COLS
        if c not in d.columns
    ]

    if missing:
        raise KeyError(
            "Missing curve columns: "
            + ", ".join(
                missing
            )
        )

    shape = (
        numeric_frame(
            d,
            SHAPE_COLS,
        )
        .to_numpy(
            np.float64
        )
    )

    p_anchor = pd.to_numeric(
        d[
            "p_anchor"
        ],
        errors="coerce",
    ).to_numpy(
        np.float64
    )

    p_span = pd.to_numeric(
        d[
            "p_span"
        ],
        errors="coerce",
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
        p_anchor[
            :,
            None,
        ]
        + p_span[
            :,
            None,
        ]
        * shape
    )

    finite = np.isfinite(
        price
    ).all(
        axis=1
    )

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

    if finite.any():
        p_min[
            finite
        ] = np.min(
            price[
                finite,
                :,
            ],
            axis=1,
        )

        p_max[
            finite
        ] = np.max(
            price[
                finite,
                :,
            ],
            axis=1,
        )

    return (
        p_min,
        p_max,
        finite,
    )


def load_train_sample(
    dataset,
    manifest,
    features,
    max_rows,
    seed,
):
    files = part_files(
        manifest,
        "train",
    )

    if not files:
        raise RuntimeError(
            "No TRAIN parts found."
        )

    per_part = max(
        1,
        int(
            math.ceil(
                max_rows
                / len(
                    files
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
            dataset
            / rel
        )

        print(
            f"[TRAIN {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        p_min, p_max, valid = compute_price_bounds(
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

        p_min = p_min[
            valid
        ]

        p_max = p_max[
            valid
        ]

        if d.empty:
            continue

        if len(
            d
        ) > per_part:
            rng = np.random.default_rng(
                seed
                + i * 1009
            )

            idx = rng.choice(
                len(d),
                size=per_part,
                replace=False,
            )

            d = (
                d.iloc[
                    idx
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            p_min = p_min[
                idx
            ]

            p_max = p_max[
                idx
            ]

        x = numeric_frame(
            d,
            features,
        )

        block = x.copy()

        block[
            "_target_p_min"
        ] = p_min

        block[
            "_target_p_max"
        ] = p_max

        blocks.append(
            block
        )

        del (
            d,
            x,
            block,
            p_min,
            p_max,
            valid,
        )

        gc.collect()

    if not blocks:
        raise RuntimeError(
            "No valid TRAIN rows."
        )

    train = pd.concat(
        blocks,
        ignore_index=True,
    )

    if len(
        train
    ) > max_rows:
        train = (
            train.sample(
                n=max_rows,
                random_state=seed,
            )
            .reset_index(
                drop=True
            )
        )

    return train


def metric_row(
    split,
    model,
    target,
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(
            y_true
        )
        & np.isfinite(
            y_pred
        )
    )

    y_true = y_true[
        mask
    ]

    y_pred = y_pred[
        mask
    ]

    if len(
        y_true
    ) == 0:
        return {
            "split": split,
            "model": model,
            "target": target,
            "rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "wape_pct": np.nan,
            "smape_pct": np.nan,
            "r2": np.nan,
        }

    err = (
        y_pred
        - y_true
    )

    ae = np.abs(
        err
    )

    smape_den = (
        np.abs(
            y_true
        )
        + np.abs(
            y_pred
        )
    )

    smape_mask = (
        smape_den
        > 1e-8
    )

    if smape_mask.any():
        smape = float(
            100.0
            * np.mean(
                2.0
                * ae[
                    smape_mask
                ]
                / smape_den[
                    smape_mask
                ]
            )
        )
    else:
        smape = 0.0

    return {
        "split": split,
        "model": model,
        "target": target,
        "rows": int(
            len(
                y_true
            )
        ),
        "mae": float(
            np.mean(
                ae
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        err
                    )
                )
            )
        ),
        "wape_pct": float(
            100.0
            * np.sum(
                ae
            )
            / max(
                np.sum(
                    np.abs(
                        y_true
                    )
                ),
                1e-12,
            )
        ),
        "smape_pct": smape,
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        )
        if len(
            y_true
        ) >= 2
        else np.nan,
    }


def evaluate_split(
    dataset,
    manifest,
    split,
    features,
    model,
    mean_targets,
    sample_limit,
    seed,
):
    files = part_files(
        manifest,
        split,
    )

    metric_store = {
        "train_mean": {
            "p_min_true": [],
            "p_min_pred": [],
            "p_max_true": [],
            "p_max_pred": [],
        },
        "random_forest": {
            "p_min_true": [],
            "p_min_pred": [],
            "p_max_true": [],
            "p_max_pred": [],
        },
    }

    raw_inversions = 0
    total_rows = 0

    sample_rows = []

    rng = np.random.default_rng(
        seed
        + (
            100
            if split == "val"
            else 200
        )
    )

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            dataset
            / rel
        )

        print(
            f"[{split.upper()} {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        p_min, p_max, valid = compute_price_bounds(
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

        p_min = p_min[
            valid
        ]

        p_max = p_max[
            valid
        ]

        if d.empty:
            continue

        X = numeric_frame(
            d,
            features,
        )

        pred_rf = model.predict(
            X
        ).astype(
            np.float64
        )

        pred_mean = np.repeat(
            mean_targets[
                None,
                :,
            ],
            len(
                d
            ),
            axis=0,
        )

        raw_inversions += int(
            (
                pred_rf[
                    :,
                    0
                ]
                > pred_rf[
                    :,
                    1
                ]
            ).sum()
        )

        total_rows += len(
            d
        )

        for model_name, pred in [
            (
                "train_mean",
                pred_mean,
            ),
            (
                "random_forest",
                pred_rf,
            ),
        ]:
            st = metric_store[
                model_name
            ]

            st[
                "p_min_true"
            ].append(
                p_min.astype(
                    np.float32
                )
            )

            st[
                "p_min_pred"
            ].append(
                pred[
                    :,
                    0
                ].astype(
                    np.float32
                )
            )

            st[
                "p_max_true"
            ].append(
                p_max.astype(
                    np.float32
                )
            )

            st[
                "p_max_pred"
            ].append(
                pred[
                    :,
                    1
                ].astype(
                    np.float32
                )
            )

        # Keep only a limited random sample for plots / CSV.
        if sample_limit > 0:
            take = min(
                len(
                    d
                ),
                max(
                    1,
                    int(
                        math.ceil(
                            sample_limit
                            / max(
                                len(
                                    files
                                ),
                                1,
                            )
                        )
                    ),
                ),
            )

            if take < len(
                d
            ):
                idx = rng.choice(
                    len(
                        d
                    ),
                    size=take,
                    replace=False,
                )
            else:
                idx = np.arange(
                    len(
                        d
                    )
                )

            meta = pd.DataFrame(
                {
                    "split": split,
                    "sample_id": (
                        d[
                            "sample_id"
                        ].astype(
                            str
                        ).iloc[
                            idx
                        ].to_numpy()
                        if "sample_id"
                        in d.columns
                        else np.asarray(
                            [
                                ""
                            ]
                            * len(
                                idx
                            )
                        )
                    ),
                    "participant_id": (
                        d[
                            "participant_id"
                        ].astype(
                            str
                        ).iloc[
                            idx
                        ].to_numpy()
                        if "participant_id"
                        in d.columns
                        else np.asarray(
                            [
                                ""
                            ]
                            * len(
                                idx
                            )
                        )
                    ),
                    "local_date": (
                        d[
                            "local_date"
                        ].astype(
                            str
                        ).iloc[
                            idx
                        ].to_numpy()
                        if "local_date"
                        in d.columns
                        else np.asarray(
                            [
                                ""
                            ]
                            * len(
                                idx
                            )
                        )
                    ),
                    "true_p_min": p_min[
                        idx
                    ],
                    "pred_p_min": pred_rf[
                        idx,
                        0,
                    ],
                    "true_p_max": p_max[
                        idx
                    ],
                    "pred_p_max": pred_rf[
                        idx,
                        1,
                    ],
                }
            )

            meta[
                "true_p_span"
            ] = (
                meta[
                    "true_p_max"
                ]
                - meta[
                    "true_p_min"
                ]
            )

            meta[
                "pred_p_span"
            ] = (
                meta[
                    "pred_p_max"
                ]
                - meta[
                    "pred_p_min"
                ]
            )

            sample_rows.append(
                meta
            )

        del (
            d,
            X,
            p_min,
            p_max,
            valid,
            pred_rf,
            pred_mean,
        )

        gc.collect()

    metrics = []

    all_arrays = {}

    for model_name, st in metric_store.items():
        if not st[
            "p_min_true"
        ]:
            continue

        p_min_true = np.concatenate(
            st[
                "p_min_true"
            ]
        )

        p_min_pred = np.concatenate(
            st[
                "p_min_pred"
            ]
        )

        p_max_true = np.concatenate(
            st[
                "p_max_true"
            ]
        )

        p_max_pred = np.concatenate(
            st[
                "p_max_pred"
            ]
        )

        p_span_true = (
            p_max_true
            - p_min_true
        )

        p_span_pred = (
            p_max_pred
            - p_min_pred
        )

        metrics.extend(
            [
                metric_row(
                    split,
                    model_name,
                    "p_min",
                    p_min_true,
                    p_min_pred,
                ),
                metric_row(
                    split,
                    model_name,
                    "p_max",
                    p_max_true,
                    p_max_pred,
                ),
                metric_row(
                    split,
                    model_name,
                    "p_span",
                    p_span_true,
                    p_span_pred,
                ),
            ]
        )

        all_arrays[
            model_name
        ] = {
            "p_min_true": p_min_true,
            "p_min_pred": p_min_pred,
            "p_max_true": p_max_true,
            "p_max_pred": p_max_pred,
            "p_span_true": p_span_true,
            "p_span_pred": p_span_pred,
        }

    sample = (
        pd.concat(
            sample_rows,
            ignore_index=True,
        )
        if sample_rows
        else pd.DataFrame()
    )

    if len(
        sample
    ) > sample_limit:
        sample = (
            sample.sample(
                n=sample_limit,
                random_state=seed,
            )
            .reset_index(
                drop=True
            )
        )

    return (
        pd.DataFrame(
            metrics
        ),
        all_arrays,
        sample,
        {
            "rows": int(
                total_rows
            ),
            "raw_bound_inversion_rows": int(
                raw_inversions
            ),
            "raw_bound_inversion_rate": float(
                raw_inversions
                / max(
                    total_rows,
                    1,
                )
            ),
        },
    )


def plot_true_vs_pred(
    y_true,
    y_pred,
    title,
    xlabel,
    ylabel,
    out_path,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(
            y_true
        )
        & np.isfinite(
            y_pred
        )
    )

    y_true = y_true[
        mask
    ]

    y_pred = y_pred[
        mask
    ]

    if len(
        y_true
    ) == 0:
        return

    lo = float(
        min(
            np.min(
                y_true
            ),
            np.min(
                y_pred
            ),
        )
    )

    hi = float(
        max(
            np.max(
                y_true
            ),
            np.max(
                y_pred
            ),
        )
    )

    fig = plt.figure(
        figsize=(
            7,
            7,
        )
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    ax.scatter(
        y_true,
        y_pred,
        s=10,
        alpha=0.25,
    )

    ax.plot(
        [
            lo,
            hi,
        ],
        [
            lo,
            hi,
        ],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def lookup_metric(
    metrics,
    split,
    model,
    target,
    col,
):
    q = metrics.loc[
        metrics[
            "split"
        ].eq(
            split
        )
        & metrics[
            "model"
        ].eq(
            model
        )
        & metrics[
            "target"
        ].eq(
            target
        ),
        col,
    ]

    if q.empty:
        return np.nan

    return float(
        q.iloc[
            0
        ]
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
        "--dataset-dir",
        default="final_feature_only_dataset",
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
        "--plot-sample",
        type=int,
        default=20_000,
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

    dataset = (
        base
        / args.dataset_dir
    )

    manifest_file = (
        dataset
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

    features = list(
        manifest[
            "model_features"
        ]
    )

    if not features:
        raise RuntimeError(
            "No model_features found."
        )

    out = (
        base
        / "price_bounds_validation"
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

    print(
        "="
        * 80
    )

    print(
        f"Simple bid price bounds validation - "
        f"{args.year}"
    )

    print(
        "="
        * 80
    )

    print(
        f"Input features = "
        f"{len(features)}"
    )

    print(
        "Targets = p_min, p_max"
    )

    print(
        "Models = train_mean + RandomForest"
    )

    print(
        "No PCA / Template / Family / historical raw bid curve."
    )

    print()

    train = load_train_sample(
        dataset,
        manifest,
        features,
        args.max_train_rows,
        args.seed,
    )

    print(
        f"\nTRAIN sample rows = "
        f"{len(train):,}"
    )

    X_train = train[
        features
    ]

    y_train = train[
        [
            "_target_p_min",
            "_target_p_max",
        ]
    ].to_numpy(
        np.float32
    )

    mean_targets = np.mean(
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
        "\n[fit] Random Forest...",
        flush=True,
    )

    model.fit(
        X_train,
        y_train,
    )

    joblib.dump(
        {
            "model": model,
            "features": features,
            "targets": [
                "p_min",
                "p_max",
            ],
            "train_mean_targets": mean_targets,
        },
        out
        / "random_forest_bounds.joblib",
        compress=3,
    )

    metric_tables = []
    split_details = {}
    test_arrays = None
    test_sample = None

    for split in [
        "val",
        "test",
    ]:
        (
            metrics,
            arrays,
            sample,
            details,
        ) = evaluate_split(
            dataset,
            manifest,
            split,
            features,
            model,
            mean_targets,
            args.plot_sample,
            args.seed,
        )

        metric_tables.append(
            metrics
        )

        split_details[
            split
        ] = details

        if split == "test":
            test_arrays = arrays
            test_sample = sample

    metrics = pd.concat(
        metric_tables,
        ignore_index=True,
    )

    metrics.to_csv(
        out
        / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if (
        test_sample
        is not None
        and not test_sample.empty
    ):
        test_sample.to_csv(
            out
            / "test_prediction_sample.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if (
        test_arrays
        is not None
        and "random_forest"
        in test_arrays
    ):
        arr = test_arrays[
            "random_forest"
        ]

        # Plot only the already-limited TEST sample if available.
        if (
            test_sample
            is not None
            and not test_sample.empty
        ):
            plot_true_vs_pred(
                test_sample[
                    "true_p_min"
                ].to_numpy(),
                test_sample[
                    "pred_p_min"
                ].to_numpy(),
                "TEST: true vs predicted p_min",
                "True p_min",
                "Predicted p_min",
                out
                / "pmin_true_vs_pred.png",
            )

            plot_true_vs_pred(
                test_sample[
                    "true_p_max"
                ].to_numpy(),
                test_sample[
                    "pred_p_max"
                ].to_numpy(),
                "TEST: true vs predicted p_max",
                "True p_max",
                "Predicted p_max",
                out
                / "pmax_true_vs_pred.png",
            )

            plot_true_vs_pred(
                test_sample[
                    "true_p_span"
                ].to_numpy(),
                test_sample[
                    "pred_p_span"
                ].to_numpy(),
                "TEST: true vs predicted price span",
                "True p_max - p_min",
                "Predicted p_max - p_min",
                out
                / "pspan_true_vs_pred.png",
            )

    def metric_line(
        split,
        model_name,
        target,
    ):
        mae = lookup_metric(
            metrics,
            split,
            model_name,
            target,
            "mae",
        )

        rmse = lookup_metric(
            metrics,
            split,
            model_name,
            target,
            "rmse",
        )

        wape = lookup_metric(
            metrics,
            split,
            model_name,
            target,
            "wape_pct",
        )

        smape = lookup_metric(
            metrics,
            split,
            model_name,
            target,
            "smape_pct",
        )

        r2 = lookup_metric(
            metrics,
            split,
            model_name,
            target,
            "r2",
        )

        return (
            f"{split.upper():4s} "
            f"{model_name:13s} "
            f"{target:6s} | "
            f"MAE={mae:10.4f} "
            f"RMSE={rmse:10.4f} "
            f"WAPE={wape:9.3f}% "
            f"sMAPE={smape:9.3f}% "
            f"R2={r2:8.4f}"
        )

    lines = [
        (
            f"Simple bid price bounds validation - "
            f"{args.year}"
        ),
        "="
        * 100,
        "",
        (
            f"Input features = "
            f"{len(features)}"
        ),
        (
            f"TRAIN sample rows = "
            f"{len(train):,}"
        ),
        (
            "Targets = p_min, p_max "
            "(p_span is derived only for evaluation)"
        ),
        (
            "Primary diagnostic model = "
            "RandomForestRegressor"
        ),
        "",
        "METRICS",
        "-" * 100,
    ]

    for split in [
        "val",
        "test",
    ]:
        for model_name in [
            "train_mean",
            "random_forest",
        ]:
            for target in [
                "p_min",
                "p_max",
                "p_span",
            ]:
                lines.append(
                    metric_line(
                        split,
                        model_name,
                        target,
                    )
                )

        lines.append(
            ""
        )

    test_rf_min_wape = lookup_metric(
        metrics,
        "test",
        "random_forest",
        "p_min",
        "wape_pct",
    )

    test_rf_max_wape = lookup_metric(
        metrics,
        "test",
        "random_forest",
        "p_max",
        "wape_pct",
    )

    test_rf_min_r2 = lookup_metric(
        metrics,
        "test",
        "random_forest",
        "p_min",
        "r2",
    )

    test_rf_max_r2 = lookup_metric(
        metrics,
        "test",
        "random_forest",
        "p_max",
        "r2",
    )

    test_mean_min_wape = lookup_metric(
        metrics,
        "test",
        "train_mean",
        "p_min",
        "wape_pct",
    )

    test_mean_max_wape = lookup_metric(
        metrics,
        "test",
        "train_mean",
        "p_max",
        "wape_pct",
    )

    min_improvement = (
        100.0
        * (
            test_mean_min_wape
            - test_rf_min_wape
        )
        / max(
            test_mean_min_wape,
            1e-12,
        )
    )

    max_improvement = (
        100.0
        * (
            test_mean_max_wape
            - test_rf_max_wape
        )
        / max(
            test_mean_max_wape,
            1e-12,
        )
    )

    # This verdict is deliberately only about whether X has useful
    # out-of-sample predictive signal, not whether the absolute engineering
    # accuracy is already sufficient.
    signal_pass = bool(
        np.isfinite(
            test_rf_min_r2
        )
        and np.isfinite(
            test_rf_max_r2
        )
        and test_rf_min_r2
        > 0.0
        and test_rf_max_r2
        > 0.0
        and test_rf_min_wape
        < test_mean_min_wape
        and test_rf_max_wape
        < test_mean_max_wape
    )

    lines.extend(
        [
            "TEST SIMPLE DIAGNOSTIC",
            "-" * 100,
            (
                f"p_min WAPE improvement vs train-mean = "
                f"{min_improvement:.2f}%"
            ),
            (
                f"p_max WAPE improvement vs train-mean = "
                f"{max_improvement:.2f}%"
            ),
            (
                f"RF raw p_min > p_max inversion rate = "
                f"{split_details['test']['raw_bound_inversion_rate']:.6f}"
            ),
            "",
            (
                "FEATURE_SIGNAL = PASS"
                if signal_pass
                else "FEATURE_SIGNAL = FAIL"
            ),
            (
                "PASS here means only: on TEST, RF beats the train-mean "
                "baseline for BOTH bounds and both R2 values are positive."
            ),
            (
                "Whether the absolute errors are accurate enough for the "
                "project should be judged from TEST p_min/p_max MAE, WAPE, "
                "R2 and the three scatter plots."
            ),
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
    print(
        summary
    )

    print()
    print(
        f"Outputs: {out}"
    )


if __name__ == "__main__":
    main()
