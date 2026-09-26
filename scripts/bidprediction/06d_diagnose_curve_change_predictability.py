#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06d_diagnose_curve_change_predictability.py

Pure diagnostic for the existing 06a/06b/06c curve-change pipeline.

NO model is trained and NO forecasting structure is changed.

Questions answered
------------------
1) Is failure caused by overfitting / temporal generalization?
   -> TRAIN-sample / VAL / TEST model metrics.

2) Are regressors useful only when the true curve change is large?
   -> Curve WAPE by fixed TRAIN-derived true-change magnitude bins.

3) Do predicted Delta-latents contain direction information?
   -> Pearson/Spearman correlation and sign accuracy relative to the
      zero-change latent coordinate.

4) Do regressors systematically over/under-correct?
   -> Predicted correction norm vs true change norm.

5) Does the DeltaCurve / DeltaZ distribution drift over time?
   -> TRAIN / VAL / TEST distribution comparison.

Important
---------
The PCA latent coordinates are centered/scaled coordinates. The raw "no curve
change" DeltaV = 0 generally maps to a NONZERO PCA coordinate z_zero.
Therefore direction/sign diagnostics use:

    effective_true_delta_z = true_z - z_zero
    effective_pred_delta_z = pred_z - z_zero

rather than taking the sign of PCA coordinates directly.

True curve-change magnitude
---------------------------
The primary magnitude used for binning is the row-level price MAE of the
zero-change/raw-curve-persistence forecast, evaluated on the CURRENT true
quantity grid. This is directly aligned with the final forecasting task.

Fixed magnitude bins
--------------------
Thresholds are estimated from the reconstructed TRAIN sample:
    0-50%, 50-75%, 75-90%, 90-95%, 95-99%, 99-100%

The same absolute thresholds are then applied to VAL and TEST, so the rows in
each bin are directly comparable over time.

Outputs
-------
data/processed/bidprediction/<year>/curve_change_predictability_diagnostics/
    split_model_overall_metrics.csv
    change_magnitude_train_quantiles.csv
    change_magnitude_bin_curve_metrics.csv
    delta_latent_direction_diagnostics.csv
    correction_norm_diagnostics.csv
    split_change_distribution.csv
    delta_latent_distribution_by_split.csv
    model_benefit_by_change_bin.csv
    summary.txt
    config.json

Run
---
python scripts/bidprediction/06d_diagnose_curve_change_predictability.py --year 2025

If 06b was trained with non-default sampling arguments, pass the same:
    --max-train-rows ...
    --seed ...
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from dataclasses import dataclass, field
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

CURVE_TARGET_COLS = [
    *SHAPE_COLS,
    "p_anchor",
    "p_span",
    "q_anchor_mw",
    "q_span_mw",
]

MODELS = [
    "zero_change",
    "ridge",
    "spline_gam",
    "random_forest",
]

LEARNED_MODELS = [
    "ridge",
    "spline_gam",
    "random_forest",
]

BIN_LABELS = [
    "Q00-Q50",
    "Q50-Q75",
    "Q75-Q90",
    "Q90-Q95",
    "Q95-Q99",
    "Q99-Q100",
]

BIN_QUANTILES = [
    0.00,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]


# =============================================================================
# Generic helpers
# =============================================================================

def num(s):
    return pd.to_numeric(
        s,
        errors="coerce",
    )


def num_frame(d, cols):
    return d[
        cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )


def part_files(manifest, split):
    out = []

    for item in manifest[
        "parts"
    ][
        split
    ]:
        out.append(
            item["file"]
            if isinstance(
                item,
                dict,
            )
            else item
        )

    return out


def prediction_files(manifest, split):
    return part_files(
        manifest,
        split,
    )


def safe_corr(x, y, method="pearson"):
    x = np.asarray(
        x,
        dtype=np.float64,
    )
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if valid.sum() < 3:
        return np.nan

    xv = x[
        valid
    ]
    yv = y[
        valid
    ]

    if (
        np.nanstd(xv) <= 1e-15
        or np.nanstd(yv) <= 1e-15
    ):
        return np.nan

    if method == "pearson":
        return float(
            np.corrcoef(
                xv,
                yv,
            )[
                0,
                1,
            ]
        )

    if method == "spearman":
        return float(
            pd.Series(
                xv
            ).corr(
                pd.Series(
                    yv
                ),
                method="spearman",
            )
        )

    raise ValueError(
        method
    )


# =============================================================================
# Curve representation / reconstruction
# =============================================================================

def true_curve(d):
    shape = (
        d[
            SHAPE_COLS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            np.float64
        )
    )

    p_anchor = num(
        d[
            "p_anchor"
        ]
    ).to_numpy(
        np.float64
    )

    p_span = num(
        d[
            "p_span"
        ]
    ).to_numpy(
        np.float64
    )

    zero = (
        np.abs(
            p_span
        )
        <= 1e-12
    )

    if zero.any():
        shape[
            zero,
            :
        ] = np.nan_to_num(
            shape[
                zero,
                :
            ],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    if not np.isfinite(
        shape
    ).all():
        raise ValueError(
            "Nonfinite true shape on nonzero-price-span row."
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

    quantity = (
        q_anchor[
            :,
            None,
        ]
        + q_span[
            :,
            None,
        ]
        * GRID[
            None,
            :,
        ]
    )

    return (
        quantity,
        price,
        q_anchor,
        q_span,
    )


def previous_vector(d):
    return np.column_stack(
        [
            *[
                num(
                    d[
                        f"_curvevec_p{i:02d}_lag1"
                    ]
                ).to_numpy(
                    np.float64
                )
                for i in range(
                    21
                )
            ],
            num(
                d[
                    "_curvevec_q_anchor_lag1"
                ]
            ).to_numpy(
                np.float64
            ),
            num(
                d[
                    "_curvevec_log_q_span_lag1"
                ]
            ).to_numpy(
                np.float64
            ),
        ]
    )


def current_vector(d):
    _, price, q_anchor, q_span = true_curve(
        d
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
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    price = v[
        :,
        :21,
    ]

    q_anchor = v[
        :,
        21,
    ]

    q_span = np.exp(
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
        price,
        q_anchor,
        q_span,
    )


def pred_price_on_true_q(
    true_q,
    pred_price,
    pred_q_anchor,
    pred_q_span,
):
    q_span = np.maximum(
        pred_q_span,
        1e-8,
    )

    x = (
        true_q
        - pred_q_anchor[
            :,
            None,
        ]
    ) / q_span[
        :,
        None,
    ]

    pos = np.clip(
        x,
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

    frac = (
        pos
        - lo
    )

    p_lo = np.take_along_axis(
        pred_price,
        lo,
        axis=1,
    )

    p_hi = np.take_along_axis(
        pred_price,
        hi,
        axis=1,
    )

    return (
        p_lo
        + frac
        * (
            p_hi
            - p_lo
        )
    )


def decode_delta(
    z,
    delta_bundle,
):
    z = np.asarray(
        z,
        dtype=np.float64,
    )

    scaler = delta_bundle[
        "scaler"
    ]

    pca = delta_bundle[
        "pca"
    ]

    full = np.zeros(
        (
            len(
                z
            ),
            int(
                pca.n_components_
            ),
        ),
        dtype=np.float64,
    )

    full[
        :,
        :z.shape[
            1
        ],
    ] = z

    standardized = pca.inverse_transform(
        full
    )

    return scaler.inverse_transform(
        standardized
    )


def zero_change_latent(
    delta_bundle,
    k,
):
    scaler = delta_bundle[
        "scaler"
    ]
    pca = delta_bundle[
        "pca"
    ]

    zero = np.zeros(
        (
            1,
            23,
        ),
        dtype=np.float64,
    )

    return (
        pca.transform(
            scaler.transform(
                zero
            )
        )[
            0,
            :k,
        ]
        .astype(
            np.float64
        )
    )


def curve_row_errors(
    d,
    predicted_vectors,
):
    true_q, true_p, _, _ = true_curve(
        d
    )

    out = {}

    for name, pred_v in predicted_vectors.items():
        pred_p, pred_qa, pred_qs = vector_to_curve(
            pred_v
        )

        pred_on_true = pred_price_on_true_q(
            true_q,
            pred_p,
            pred_qa,
            pred_qs,
        )

        ae = np.abs(
            pred_on_true
            - true_p
        )

        out[
            name
        ] = {
            "row_ae_sum": ae.sum(
                axis=1
            ),
            "row_curve_mae": ae.mean(
                axis=1
            ),
        }

    true_abs = np.abs(
        true_p
    ).sum(
        axis=1
    )

    return (
        out,
        true_abs,
    )


# =============================================================================
# Model prediction
# =============================================================================

def predict_spline_gam(
    d,
    gam_models,
    delta_cols,
):
    out = np.empty(
        (
            len(
                d
            ),
            len(
                delta_cols
            ),
        ),
        dtype=np.float64,
    )

    for j, target in enumerate(
        delta_cols
    ):
        spec = gam_models[
            target
        ]

        X = num_frame(
            d,
            spec[
                "features"
            ],
        )

        out[
            :,
            j,
        ] = (
            spec[
                "model"
            ]
            .predict(
                X
            )
        )

    return out


def predict_models_on_source(
    d,
    delta_cols,
    feature_cols,
    zero_z,
    ridge_model,
    gam_models,
    rf_model,
):
    X = num_frame(
        d,
        feature_cols,
    )

    y_true = num_frame(
        d,
        delta_cols,
    ).to_numpy(
        np.float64
    )

    zero = np.repeat(
        zero_z[
            None,
            :
        ],
        len(
            d
        ),
        axis=0,
    )

    ridge = (
        ridge_model.predict(
            X
        )
        .astype(
            np.float64
        )
    )

    gam = predict_spline_gam(
        d,
        gam_models,
        delta_cols,
    )

    rf = (
        rf_model.predict(
            X
        )
        .astype(
            np.float64
        )
    )

    return (
        y_true,
        {
            "zero_change": zero,
            "ridge": ridge,
            "spline_gam": gam,
            "random_forest": rf,
        },
    )


def latent_matrix(
    d,
    prefix,
    delta_cols,
):
    return np.column_stack(
        [
            num(
                d[
                    f"{prefix}{z}"
                ]
            ).to_numpy(
                np.float64
            )
            for z in delta_cols
        ]
    )


def predictions_from_prediction_part(
    d,
    delta_cols,
):
    true_z = latent_matrix(
        d,
        "true_",
        delta_cols,
    )

    predictions = {
        model: latent_matrix(
            d,
            f"pred_{model}_",
            delta_cols,
        )
        for model in MODELS
    }

    return (
        true_z,
        predictions,
    )


# =============================================================================
# TRAIN sample reconstruction
# =============================================================================

def load_train_sample(
    dataset_dir,
    manifest,
    required_cols,
    max_rows,
    seed,
):
    """
    Reproduce the same shared TRAIN sampling rule used in 06b by default.
    """
    files = part_files(
        manifest,
        "train",
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
            dataset_dir
            / rel
        )

        print(
            f"[TRAIN sample {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        missing = [
            c
            for c in required_cols
            if c not in d.columns
        ]

        if missing:
            raise KeyError(
                f"{path.name}: missing required columns "
                f"{missing[:20]}"
            )

        n = min(
            per_part,
            len(
                d
            ),
        )

        if n < len(
            d
        ):
            d = d.sample(
                n=n,
                random_state=(
                    seed
                    + 7919
                    * i
                ),
            )

        blocks.append(
            d[
                required_cols
            ].copy()
        )

        del d
        gc.collect()

    if not blocks:
        raise ValueError(
            "No TRAIN rows available."
        )

    out = pd.concat(
        blocks,
        ignore_index=True,
    )

    if len(
        out
    ) > max_rows:
        out = (
            out.sample(
                n=max_rows,
                random_state=seed,
            )
            .reset_index(
                drop=True
            )
        )

    return out


# =============================================================================
# Accumulators
# =============================================================================

@dataclass
class OverallAgg:
    rows: int = 0
    latent_ae: float = 0.0
    latent_se: float = 0.0
    latent_count: int = 0
    curve_ae: float = 0.0
    curve_true_abs: float = 0.0
    curve_points: int = 0
    curve_se: float = 0.0
    benefit_rows: int = 0


@dataclass
class BinAgg:
    rows: int = 0
    true_abs_sum: float = 0.0
    model_ae_sum: dict = field(
        default_factory=lambda: {
            m: 0.0
            for m in MODELS
        }
    )
    benefit_rows: dict = field(
        default_factory=lambda: {
            m: 0
            for m in LEARNED_MODELS
        }
    )


@dataclass
class NormAgg:
    true_norm: list = field(
        default_factory=list
    )
    pred_norm: dict = field(
        default_factory=lambda: {
            m: []
            for m in LEARNED_MODELS
        }
    )


def update_overall(
    agg,
    true_z,
    pred_z,
    row_ae,
    true_abs,
    zero_row_ae,
):
    latent_err = (
        pred_z
        - true_z
    )

    agg.rows += int(
        len(
            true_z
        )
    )

    agg.latent_ae += float(
        np.abs(
            latent_err
        ).sum()
    )

    agg.latent_se += float(
        np.square(
            latent_err
        ).sum()
    )

    agg.latent_count += int(
        latent_err.size
    )

    agg.curve_ae += float(
        row_ae.sum()
    )

    agg.curve_true_abs += float(
        true_abs.sum()
    )

    agg.curve_points += int(
        len(
            row_ae
        )
        * 21
    )

    if zero_row_ae is not None:
        agg.benefit_rows += int(
            np.sum(
                row_ae
                < zero_row_ae
            )
        )


def finalize_overall(
    split,
    model,
    agg,
):
    return {
        "split": split,
        "model": model,
        "rows": int(
            agg.rows
        ),
        "delta_latent_mae_mean": (
            agg.latent_ae
            / max(
                agg.latent_count,
                1,
            )
        ),
        "delta_latent_rmse_mean": float(
            np.sqrt(
                agg.latent_se
                / max(
                    agg.latent_count,
                    1,
                )
            )
        ),
        "price_wape_pct": (
            100.0
            * agg.curve_ae
            / max(
                agg.curve_true_abs,
                1e-12,
            )
        ),
        "share_rows_better_than_zero_change": (
            agg.benefit_rows
            / agg.rows
            if (
                agg.rows
                and model
                != "zero_change"
            )
            else np.nan
        ),
    }


# =============================================================================
# Diagnostic calculations
# =============================================================================

def derive_train_bin_edges(
    train_change_magnitude,
):
    q = np.quantile(
        train_change_magnitude,
        BIN_QUANTILES,
    )

    # Searchsorted/binning needs strict ordering. Ties at zero are common.
    # Preserve the empirical thresholds but make subsequent identical edges
    # infinitesimally increasing for deterministic assignment.
    edges = q.astype(
        np.float64
    ).copy()

    for i in range(
        1,
        len(
            edges
        )
    ):
        if edges[
            i
        ] <= edges[
            i - 1
        ]:
            edges[
                i
            ] = np.nextafter(
                edges[
                    i - 1
                ],
                np.inf,
            )

    # Include all future values in final bin.
    edges[
        0
    ] = -np.inf

    edges[
        -1
    ] = np.inf

    return (
        q,
        edges,
    )


def assign_bins(
    magnitude,
    edges,
):
    idx = np.searchsorted(
        edges,
        magnitude,
        side="right",
    ) - 1

    return np.clip(
        idx,
        0,
        len(
            BIN_LABELS
        )
        - 1,
    ).astype(
        np.int16
    )


def update_bin_aggs(
    aggs,
    bin_index,
    true_abs,
    curve_errors,
):
    zero_ae = curve_errors[
        "zero_change"
    ][
        "row_ae_sum"
    ]

    for b in np.unique(
        bin_index
    ):
        mask = (
            bin_index
            == b
        )

        agg = aggs[
            int(
                b
            )
        ]

        agg.rows += int(
            mask.sum()
        )

        agg.true_abs_sum += float(
            true_abs[
                mask
            ].sum()
        )

        for model in MODELS:
            ae = curve_errors[
                model
            ][
                "row_ae_sum"
            ][
                mask
            ]

            agg.model_ae_sum[
                model
            ] += float(
                ae.sum()
            )

            if model in LEARNED_MODELS:
                agg.benefit_rows[
                    model
                ] += int(
                    np.sum(
                        ae
                        < zero_ae[
                            mask
                        ]
                    )
                )


def finalize_bin_aggs(
    split,
    aggs,
    empirical_quantiles,
):
    rows = []

    for i, agg in enumerate(
        aggs
    ):
        for model in MODELS:
            rows.append(
                {
                    "split": split,
                    "change_bin": BIN_LABELS[
                        i
                    ],
                    "train_quantile_low": BIN_QUANTILES[
                        i
                    ],
                    "train_quantile_high": BIN_QUANTILES[
                        i + 1
                    ],
                    "train_threshold_low_curve_mae": float(
                        empirical_quantiles[
                            i
                        ]
                    ),
                    "train_threshold_high_curve_mae": float(
                        empirical_quantiles[
                            i + 1
                        ]
                    ),
                    "rows": int(
                        agg.rows
                    ),
                    "model": model,
                    "price_wape_pct": (
                        100.0
                        * agg.model_ae_sum[
                            model
                        ]
                        / max(
                            agg.true_abs_sum,
                            1e-12,
                        )
                        if agg.rows
                        else np.nan
                    ),
                    "share_rows_better_than_zero_change": (
                        agg.benefit_rows[
                            model
                        ]
                        / agg.rows
                        if (
                            agg.rows
                            and model
                            in LEARNED_MODELS
                        )
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def direction_rows(
    split,
    true_z,
    predictions,
    zero_z,
    delta_cols,
):
    rows = []

    true_eff = (
        true_z
        - zero_z[
            None,
            :
        ]
    )

    eps_by_dim = np.maximum(
        np.nanquantile(
            np.abs(
                true_eff
            ),
            0.10,
            axis=0,
        )
        * 0.05,
        1e-8,
    )

    for model in LEARNED_MODELS:
        pred_eff = (
            predictions[
                model
            ]
            - zero_z[
                None,
                :
            ]
        )

        for j, col in enumerate(
            delta_cols
        ):
            t = true_eff[
                :,
                j,
            ]
            p = pred_eff[
                :,
                j,
            ]

            eps = eps_by_dim[
                j
            ]

            active = (
                np.abs(
                    t
                )
                > eps
            )

            sign_acc = (
                float(
                    np.mean(
                        np.sign(
                            p[
                                active
                            ]
                        )
                        == np.sign(
                            t[
                                active
                            ]
                        )
                    )
                )
                if active.any()
                else np.nan
            )

            rows.append(
                {
                    "split": split,
                    "model": model,
                    "delta_latent": col,
                    "rows": int(
                        len(
                            t
                        )
                    ),
                    "active_rows_for_sign": int(
                        active.sum()
                    ),
                    "zero_reference_latent": float(
                        zero_z[
                            j
                        ]
                    ),
                    "pearson_corr_effective_delta": safe_corr(
                        t,
                        p,
                        "pearson",
                    ),
                    "spearman_corr_effective_delta": safe_corr(
                        t,
                        p,
                        "spearman",
                    ),
                    "sign_accuracy_nontrivial_change": sign_acc,
                    "true_effective_mae": float(
                        np.mean(
                            np.abs(
                                t
                            )
                        )
                    ),
                    "pred_effective_mae": float(
                        np.mean(
                            np.abs(
                                p
                            )
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def norm_rows(
    split,
    true_delta_v,
    pred_delta_v,
    curve_errors,
):
    # Primary norm is mean absolute PRICE correction across 21 points.
    true_norm = np.mean(
        np.abs(
            true_delta_v[
                :,
                :21,
            ]
        ),
        axis=1,
    )

    rows = []

    zero_row_ae = curve_errors[
        "zero_change"
    ][
        "row_ae_sum"
    ]

    for model in LEARNED_MODELS:
        pred_norm = np.mean(
            np.abs(
                pred_delta_v[
                    model
                ][
                    :,
                    :21,
                ]
            ),
            axis=1,
        )

        ratio = np.divide(
            pred_norm,
            true_norm,
            out=np.full_like(
                pred_norm,
                np.nan,
            ),
            where=true_norm
            > 1e-8,
        )

        benefit = (
            curve_errors[
                model
            ][
                "row_ae_sum"
            ]
            < zero_row_ae
        )

        rows.append(
            {
                "split": split,
                "model": model,
                "rows": int(
                    len(
                        true_norm
                    )
                ),
                "true_price_change_mae_mean": float(
                    np.mean(
                        true_norm
                    )
                ),
                "true_price_change_mae_p50": float(
                    np.quantile(
                        true_norm,
                        0.50,
                    )
                ),
                "true_price_change_mae_p90": float(
                    np.quantile(
                        true_norm,
                        0.90,
                    )
                ),
                "true_price_change_mae_p95": float(
                    np.quantile(
                        true_norm,
                        0.95,
                    )
                ),
                "true_price_change_mae_p99": float(
                    np.quantile(
                        true_norm,
                        0.99,
                    )
                ),
                "pred_price_correction_mae_mean": float(
                    np.mean(
                        pred_norm
                    )
                ),
                "pred_price_correction_mae_p50": float(
                    np.quantile(
                        pred_norm,
                        0.50,
                    )
                ),
                "pred_price_correction_mae_p90": float(
                    np.quantile(
                        pred_norm,
                        0.90,
                    )
                ),
                "pred_price_correction_mae_p95": float(
                    np.quantile(
                        pred_norm,
                        0.95,
                    )
                ),
                "pred_price_correction_mae_p99": float(
                    np.quantile(
                        pred_norm,
                        0.99,
                    )
                ),
                "corr_true_vs_pred_correction_norm": safe_corr(
                    true_norm,
                    pred_norm,
                    "pearson",
                ),
                "spearman_true_vs_pred_correction_norm": safe_corr(
                    true_norm,
                    pred_norm,
                    "spearman",
                ),
                "median_pred_to_true_norm_ratio_nonzero_true": (
                    float(
                        np.nanmedian(
                            ratio
                        )
                    )
                ),
                "share_pred_norm_gt_true_norm": float(
                    np.mean(
                        pred_norm
                        > true_norm
                    )
                ),
                "share_rows_curve_error_better_than_zero_change": float(
                    np.mean(
                        benefit
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def split_distribution_row(
    split,
    magnitude,
    true_delta_v,
):
    price_abs = np.abs(
        true_delta_v[
            :,
            :21,
        ]
    )

    return {
        "split": split,
        "rows": int(
            len(
                magnitude
            )
        ),
        "zero_change_curve_mae_mean": float(
            np.mean(
                magnitude
            )
        ),
        "zero_change_curve_mae_p50": float(
            np.quantile(
                magnitude,
                0.50,
            )
        ),
        "zero_change_curve_mae_p75": float(
            np.quantile(
                magnitude,
                0.75,
            )
        ),
        "zero_change_curve_mae_p90": float(
            np.quantile(
                magnitude,
                0.90,
            )
        ),
        "zero_change_curve_mae_p95": float(
            np.quantile(
                magnitude,
                0.95,
            )
        ),
        "zero_change_curve_mae_p99": float(
            np.quantile(
                magnitude,
                0.99,
            )
        ),
        "zero_change_curve_mae_max": float(
            np.max(
                magnitude
            )
        ),
        "raw_delta_price_abs_mean": float(
            price_abs.mean()
        ),
        "raw_delta_price_abs_p95": float(
            np.quantile(
                price_abs,
                0.95,
            )
        ),
        "raw_delta_q_anchor_abs_mean": float(
            np.mean(
                np.abs(
                    true_delta_v[
                        :,
                        21,
                    ]
                )
            )
        ),
        "raw_delta_log_q_span_abs_mean": float(
            np.mean(
                np.abs(
                    true_delta_v[
                        :,
                        22,
                    ]
                )
            )
        ),
    }


def latent_distribution_rows(
    split,
    true_z,
    zero_z,
    delta_cols,
):
    eff = (
        true_z
        - zero_z[
            None,
            :
        ]
    )

    rows = []

    for j, col in enumerate(
        delta_cols
    ):
        x = eff[
            :,
            j,
        ]

        rows.append(
            {
                "split": split,
                "delta_latent": col,
                "rows": int(
                    len(
                        x
                    )
                ),
                "effective_mean": float(
                    np.mean(
                        x
                    )
                ),
                "effective_std": float(
                    np.std(
                        x
                    )
                ),
                "effective_abs_mean": float(
                    np.mean(
                        np.abs(
                            x
                        )
                    )
                ),
                "effective_p05": float(
                    np.quantile(
                        x,
                        0.05,
                    )
                ),
                "effective_p50": float(
                    np.quantile(
                        x,
                        0.50,
                    )
                ),
                "effective_p95": float(
                    np.quantile(
                        x,
                        0.95,
                    )
                ),
                "zero_reference_latent": float(
                    zero_z[
                        j
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Split evaluation
# =============================================================================

def evaluate_block(
    split,
    d,
    true_z,
    predictions,
    zero_z,
    delta_bundle,
    delta_cols,
    bin_edges,
    empirical_quantiles,
):
    prev_v = previous_vector(
        d
    )

    current_v = current_vector(
        d
    )

    true_delta_v = (
        current_v
        - prev_v
    )

    pred_delta_v = {
        model: decode_delta(
            pred_z,
            delta_bundle,
        )
        for model, pred_z in predictions.items()
    }

    predicted_vectors = {
        model: (
            prev_v
            + pred_delta_v[
                model
            ]
        )
        for model in MODELS
    }

    curve_errors, true_abs = curve_row_errors(
        d,
        predicted_vectors,
    )

    true_change_magnitude = curve_errors[
        "zero_change"
    ][
        "row_curve_mae"
    ]

    bin_index = assign_bins(
        true_change_magnitude,
        bin_edges,
    )

    bin_aggs = [
        BinAgg()
        for _ in BIN_LABELS
    ]

    update_bin_aggs(
        bin_aggs,
        bin_index,
        true_abs,
        curve_errors,
    )

    bin_table = finalize_bin_aggs(
        split,
        bin_aggs,
        empirical_quantiles,
    )

    overall_rows = []

    zero_row_ae = curve_errors[
        "zero_change"
    ][
        "row_ae_sum"
    ]

    for model in MODELS:
        agg = OverallAgg()

        update_overall(
            agg,
            true_z,
            predictions[
                model
            ],
            curve_errors[
                model
            ][
                "row_ae_sum"
            ],
            true_abs,
            (
                None
                if model
                == "zero_change"
                else zero_row_ae
            ),
        )

        overall_rows.append(
            finalize_overall(
                split,
                model,
                agg,
            )
        )

    direction = direction_rows(
        split,
        true_z,
        predictions,
        zero_z,
        delta_cols,
    )

    norms = norm_rows(
        split,
        true_delta_v,
        pred_delta_v,
        curve_errors,
    )

    distribution = pd.DataFrame(
        [
            split_distribution_row(
                split,
                true_change_magnitude,
                true_delta_v,
            )
        ]
    )

    latent_distribution = latent_distribution_rows(
        split,
        true_z,
        zero_z,
        delta_cols,
    )

    return {
        "overall": pd.DataFrame(
            overall_rows
        ),
        "bins": bin_table,
        "direction": direction,
        "norms": norms,
        "distribution": distribution,
        "latent_distribution": latent_distribution,
        "change_magnitude": true_change_magnitude,
    }


def concatenate_split_results(
    blocks,
):
    keys = [
        "overall",
        "bins",
        "direction",
        "norms",
        "distribution",
        "latent_distribution",
    ]

    return {
        key: pd.concat(
            [
                b[
                    key
                ]
                for b in blocks
            ],
            ignore_index=True,
        )
        for key in keys
    }


# =============================================================================
# Main
# =============================================================================

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
        default="curve_change_latent_dataset",
    )
    ap.add_argument(
        "--model-dir",
        default="curve_change_regression_models",
    )
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=300_000,
        help=(
            "Use the same value used by 06b to reproduce "
            "the shared TRAIN sample."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Use the same seed used by 06b to reproduce "
            "the shared TRAIN sample."
        ),
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

    dataset_dir = (
        base
        / args.dataset_dir
    )

    model_dir = (
        base
        / args.model_dir
    )

    dataset_manifest = json.loads(
        (
            dataset_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    prediction_manifest = json.loads(
        (
            model_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    delta_bundle = joblib.load(
        dataset_dir
        / "delta_pca_bundle.joblib"
    )

    ridge_bundle = joblib.load(
        model_dir
        / "ridge_model.joblib"
    )

    gam_bundle = joblib.load(
        model_dir
        / "spline_gam_models.joblib"
    )

    rf_bundle = joblib.load(
        model_dir
        / "random_forest_model.joblib"
    )

    delta_cols = list(
        dataset_manifest[
            "delta_columns"
        ]
    )

    feature_cols = list(
        dataset_manifest[
            "model_features"
        ]
    )

    zero_z = zero_change_latent(
        delta_bundle,
        len(
            delta_cols
        ),
    )

    ridge_model = ridge_bundle[
        "model"
    ]

    gam_models = gam_bundle[
        "models"
    ]

    rf_model = rf_bundle[
        "model"
    ]

    out = (
        base
        / "curve_change_predictability_diagnostics"
    )

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} already exists. "
                "Use --overwrite."
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
        f"Curve-change predictability diagnostics - "
        f"{args.year}"
    )
    print("=" * 80)
    print(
        f"Delta latent dimension = {len(delta_cols)}"
    )
    print(
        f"Model features = {len(feature_cols)}"
    )
    print(
        "No model is retrained."
    )
    print()

    # ---------------------------------------------------------------------
    # TRAIN sample: reproduce 06b shared training sample.
    # ---------------------------------------------------------------------
    train_required = list(
        dict.fromkeys(
            [
                *feature_cols,
                *delta_cols,
                *CURVE_TARGET_COLS,
                *RAW_PREV_COLS,
            ]
        )
    )

    train = load_train_sample(
        dataset_dir,
        dataset_manifest,
        train_required,
        args.max_train_rows,
        args.seed,
    )

    train_true_z, train_predictions = predict_models_on_source(
        train,
        delta_cols,
        feature_cols,
        zero_z,
        ridge_model,
        gam_models,
        rf_model,
    )

    # Derive fixed magnitude thresholds from TRAIN sample before diagnostics.
    train_prev_v = previous_vector(
        train
    )
    train_current_v = current_vector(
        train
    )

    train_true_q, train_true_p, _, _ = true_curve(
        train
    )

    train_prev_p, train_prev_qa, train_prev_qs = vector_to_curve(
        train_prev_v
    )

    train_prev_on_true = pred_price_on_true_q(
        train_true_q,
        train_prev_p,
        train_prev_qa,
        train_prev_qs,
    )

    train_change_magnitude = np.mean(
        np.abs(
            train_prev_on_true
            - train_true_p
        ),
        axis=1,
    )

    empirical_quantiles, bin_edges = derive_train_bin_edges(
        train_change_magnitude
    )

    quantile_table = pd.DataFrame(
        {
            "quantile": BIN_QUANTILES,
            "train_curve_change_mae_threshold": empirical_quantiles,
        }
    )

    quantile_table.to_csv(
        out
        / "change_magnitude_train_quantiles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "TRAIN-derived curve-change MAE thresholds:"
    )
    print(
        quantile_table.to_string(
            index=False
        )
    )
    print()

    train_result = evaluate_block(
        "train_sample",
        train,
        train_true_z,
        train_predictions,
        zero_z,
        delta_bundle,
        delta_cols,
        bin_edges,
        empirical_quantiles,
    )

    del (
        train_true_z,
        train_predictions,
        train_prev_v,
        train_current_v,
        train_true_q,
        train_true_p,
        train_prev_p,
        train_prev_qa,
        train_prev_qs,
        train_prev_on_true,
    )
    gc.collect()

    # ---------------------------------------------------------------------
    # VAL / TEST: use frozen 06b prediction files.
    # ---------------------------------------------------------------------
    split_results = [
        train_result
    ]

    for split in [
        "val",
        "test",
    ]:
        files = prediction_files(
            prediction_manifest,
            split,
        )

        print(
            f"Processing {split.upper()} "
            f"prediction parts = {len(files)}",
            flush=True,
        )

        block_results = []

        for i, rel in enumerate(
            files,
            1,
        ):
            path = (
                model_dir
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

            true_z, predictions = predictions_from_prediction_part(
                d,
                delta_cols,
            )

            block_results.append(
                evaluate_block(
                    split,
                    d,
                    true_z,
                    predictions,
                    zero_z,
                    delta_bundle,
                    delta_cols,
                    bin_edges,
                    empirical_quantiles,
                )
            )

            del d, true_z, predictions
            gc.collect()

        if not block_results:
            raise ValueError(
                f"No {split} prediction rows."
            )

        # Need exact weighted aggregation across files. Re-run concatenated
        # tables only where metrics are row-additive; distribution/correlation
        # is not row-additive, so for those diagnostics concatenate source
        # prediction parts below into a bounded in-memory representation.
        #
        # To keep the diagnostic exact, load the prediction parts once into
        # compact arrays rather than approximating correlations from blocks.
        frames = []

        for rel in files:
            path = (
                model_dir
                / rel
            )

            d = pd.read_pickle(
                path
            )

            if d.empty:
                continue

            needed = list(
                dict.fromkeys(
                    [
                        *CURVE_TARGET_COLS,
                        *RAW_PREV_COLS,
                        *[
                            f"true_{z}"
                            for z in delta_cols
                        ],
                        *[
                            f"pred_{m}_{z}"
                            for m in MODELS
                            for z in delta_cols
                        ],
                    ]
                )
            )

            frames.append(
                d[
                    needed
                ].copy()
            )

            del d
            gc.collect()

        full = pd.concat(
            frames,
            ignore_index=True,
        )

        true_z, predictions = predictions_from_prediction_part(
            full,
            delta_cols,
        )

        exact = evaluate_block(
            split,
            full,
            true_z,
            predictions,
            zero_z,
            delta_bundle,
            delta_cols,
            bin_edges,
            empirical_quantiles,
        )

        split_results.append(
            exact
        )

        del frames, full, true_z, predictions, block_results
        gc.collect()

    # ---------------------------------------------------------------------
    # Save diagnostics.
    # ---------------------------------------------------------------------
    all_results = concatenate_split_results(
        split_results
    )

    overall = all_results[
        "overall"
    ]

    bins = all_results[
        "bins"
    ]

    direction = all_results[
        "direction"
    ]

    norms = all_results[
        "norms"
    ]

    distribution = all_results[
        "distribution"
    ]

    latent_distribution = all_results[
        "latent_distribution"
    ]

    overall.to_csv(
        out
        / "split_model_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bins.to_csv(
        out
        / "change_magnitude_bin_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    direction.to_csv(
        out
        / "delta_latent_direction_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    norms.to_csv(
        out
        / "correction_norm_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    distribution.to_csv(
        out
        / "split_change_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    latent_distribution.to_csv(
        out
        / "delta_latent_distribution_by_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Compact table: learned model advantage vs zero-change by magnitude bin.
    zero_bin = (
        bins.loc[
            bins[
                "model"
            ].eq(
                "zero_change"
            ),
            [
                "split",
                "change_bin",
                "price_wape_pct",
            ],
        ]
        .rename(
            columns={
                "price_wape_pct": (
                    "zero_change_wape_pct"
                )
            }
        )
    )

    learned = bins.loc[
        bins[
            "model"
        ].isin(
            LEARNED_MODELS
        )
    ].copy()

    benefit = learned.merge(
        zero_bin,
        on=[
            "split",
            "change_bin",
        ],
        how="left",
    )

    benefit[
        "wape_improvement_vs_zero_change_pp"
    ] = (
        benefit[
            "zero_change_wape_pct"
        ]
        - benefit[
            "price_wape_pct"
        ]
    )

    benefit[
        "relative_wape_improvement_vs_zero_change_pct"
    ] = (
        100.0
        * benefit[
            "wape_improvement_vs_zero_change_pp"
        ]
        / benefit[
            "zero_change_wape_pct"
        ].replace(
            0.0,
            np.nan,
        )
    )

    benefit.to_csv(
        out
        / "model_benefit_by_change_bin.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "year": int(
            args.year
        ),
        "dataset_dir": str(
            dataset_dir
        ),
        "model_dir": str(
            model_dir
        ),
        "max_train_rows": int(
            args.max_train_rows
        ),
        "seed": int(
            args.seed
        ),
        "models": MODELS,
        "delta_latent_dimension": int(
            len(
                delta_cols
            )
        ),
        "change_magnitude_definition": (
            "row-level mean absolute price error of raw-curve "
            "persistence evaluated on the current true quantity grid"
        ),
        "change_bins": {
            label: [
                float(
                    BIN_QUANTILES[
                        i
                    ]
                ),
                float(
                    BIN_QUANTILES[
                        i + 1
                    ]
                ),
            ]
            for i, label in enumerate(
                BIN_LABELS
            )
        },
        "train_magnitude_thresholds": [
            float(
                x
            )
            for x in empirical_quantiles
        ],
        "direction_reference": (
            "PCA DeltaZ relative to the latent coordinate of raw DeltaV=0"
        ),
    }

    (
        out
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # Human-readable summary.
    # ---------------------------------------------------------------------
    summary_lines = [
        (
            f"Curve-change predictability diagnostics - "
            f"{args.year}"
        ),
        "=" * 80,
        "",
        (
            "No model was retrained. TRAIN uses the reconstructed "
            "shared 06b training sample; VAL/TEST use frozen 06b predictions."
        ),
        "",
        "1. Overall TRAIN / VAL / TEST metrics",
        overall.to_string(
            index=False
        ),
        "",
        "2. True curve-change distribution",
        distribution.to_string(
            index=False
        ),
        "",
        "3. TRAIN-derived magnitude thresholds",
        quantile_table.to_string(
            index=False
        ),
        "",
        "4. Model benefit by true change magnitude bin",
        benefit[
            [
                "split",
                "change_bin",
                "rows",
                "model",
                "zero_change_wape_pct",
                "price_wape_pct",
                "wape_improvement_vs_zero_change_pp",
                "share_rows_better_than_zero_change",
            ]
        ].to_string(
            index=False
        ),
        "",
        "5. Correction magnitude diagnostics",
        norms.to_string(
            index=False
        ),
        "",
        (
            "6. Delta-latent direction diagnostics are saved in "
            "delta_latent_direction_diagnostics.csv."
        ),
        (
            "7. Per-latent temporal distribution diagnostics are saved in "
            "delta_latent_distribution_by_split.csv."
        ),
    ]

    summary = "\n".join(
        summary_lines
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
