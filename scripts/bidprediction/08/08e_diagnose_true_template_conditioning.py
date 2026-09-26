#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08e_diagnose_true_template_conditioning.py

Purpose
-------
Diagnose whether the existing 13 template labels meaningfully partition the
Macro-B feature-only absolute-curve prediction problem.

NO regression model is retrained.

The script performs three complementary checks:

1) Existing-model error attribution by TRUE y_template_id
   - train_mean
   - Ridge
   - Spline-GAM
   - Random Forest
   - absolute-latent oracle
   - persistence reference

2) Oracle-template mean baseline
   Using TRAIN only, compute the mean absolute latent vector separately for
   each TRUE template:
       mean_z(template) = E_train[Z | template]
   Then on VAL/TEST, assume the TRUE template is known and reconstruct the
   curve from that template-specific mean latent vector.

   This is intentionally simple. It answers:
       "Does knowing the template alone materially shrink the target space?"

3) Template target dispersion
   Compare within-template absolute-latent dispersion against the global
   train-mean latent baseline.

Interpretation
--------------
- If oracle-template mean is much better than global train_mean, template
  labels carry strong absolute-curve information.
- If existing RF errors differ strongly by template, the pooled regression
  problem is heterogeneous.
- If template-conditioned mean is strong AND some templates still benefit
  from features, the next justified experiment is template-specific
  regressors / routing.

Important
---------
This is an ORACLE-TEMPLATE diagnostic. It does NOT claim deployable
performance because VAL/TEST use the realized true template for grouping and
for the template-mean baseline.

Inputs
------
data/processed/bidprediction/<year>/
    macro_b_filtered_dataset/
    macro_b_absolute_curve_representation/
    macro_b_feature_only_regression_models/

Outputs
-------
data/processed/bidprediction/<year>/
    macro_b_true_template_diagnostics/
        template_train_statistics.csv
        template_curve_metrics.csv
        overall_oracle_template_metrics.csv
        template_latent_dispersion.csv
        template_diagnostic_summary.csv
        summary.txt

Run
---
python scripts/bidprediction/08e_diagnose_true_template_conditioning.py \
    --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]

FEATURE_ONLY_MODELS = [
    "train_mean",
    "ridge",
    "spline_gam",
    "random_forest",
]

ALL_MODELS = [
    *FEATURE_ONLY_MODELS,
    "oracle_template_mean",
    "absolute_latent_oracle",
    "persistence_reference",
]

REFERENCE_PRICE_COLS = [
    f"reference_curvevec_p{i:02d}_lag1"
    for i in range(21)
]

TEMPLATE_COL_CANDIDATES = [
    "y_template_id",
    "template_id",
]


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


def detect_template_col(columns):
    for c in TEMPLATE_COL_CANDIDATES:
        if c in columns:
            return c

    raise KeyError(
        "No template label found. Expected one of: "
        + ", ".join(
            TEMPLATE_COL_CANDIDATES
        )
    )


def current_vector(d):
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


def true_curve(d):
    v = current_vector(d)

    price = v[:, :21]
    q_anchor = v[:, 21]
    q_span = np.exp(
        np.clip(
            v[:, 22],
            -20.0,
            20.0,
        )
    )

    quantity = (
        q_anchor[:, None]
        + q_span[:, None]
        * GRID[None, :]
    )

    return (
        quantity,
        price,
        q_anchor,
        q_span,
        v,
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


def latent_matrix(
    d,
    prefix,
    latent_cols,
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
            for z in latent_cols
        ]
    )


def persistence_vector(d):
    required = [
        *REFERENCE_PRICE_COLS,
        "reference_curvevec_q_anchor_lag1",
        "reference_curvevec_log_q_span_lag1",
    ]

    missing = [
        c
        for c in required
        if c not in d.columns
    ]

    if missing:
        return None

    return np.column_stack(
        [
            *[
                num(
                    d[c]
                ).to_numpy(
                    np.float64
                )
                for c
                in REFERENCE_PRICE_COLS
            ],
            num(
                d[
                    "reference_curvevec_q_anchor_lag1"
                ]
            ).to_numpy(
                np.float64
            ),
            num(
                d[
                    "reference_curvevec_log_q_span_lag1"
                ]
            ).to_numpy(
                np.float64
            ),
        ]
    )


def build_template_lookup(
    source_dir,
    source_manifest,
    split,
):
    """
    Return sample_id -> template mapping.
    Fallback to composite key only when sample_id is unavailable.
    """
    frames = []
    join_mode = None

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
            f"[template map {split} {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        template_col = detect_template_col(
            d.columns
        )

        if "sample_id" in d.columns:
            join_mode = (
                join_mode
                or "sample_id"
            )

            if join_mode != "sample_id":
                raise RuntimeError(
                    "Template join mode changed between parts."
                )

            x = d[
                [
                    "sample_id",
                    template_col,
                ]
            ].copy()

            x = x.rename(
                columns={
                    template_col:
                    "_true_template_id"
                }
            )

        else:
            keys = [
                "participant_id",
                "local_date",
                "local_slot_seconds",
            ]

            missing = [
                k
                for k in keys
                if k not in d.columns
            ]

            if missing:
                raise KeyError(
                    "Cannot link templates to 08 predictions. "
                    "Need sample_id or composite keys. "
                    f"Missing: {missing}"
                )

            join_mode = (
                join_mode
                or "composite"
            )

            if join_mode != "composite":
                raise RuntimeError(
                    "Template join mode changed between parts."
                )

            x = d[
                [
                    *keys,
                    template_col,
                ]
            ].copy()

            x = x.rename(
                columns={
                    template_col:
                    "_true_template_id"
                }
            )

        frames.append(
            x
        )

        del d, x
        gc.collect()

    lookup = pd.concat(
        frames,
        ignore_index=True,
    )

    lookup[
        "_true_template_id"
    ] = (
        lookup[
            "_true_template_id"
        ]
        .astype(
            "string"
        )
        .fillna(
            "UNMAPPED"
        )
    )

    if join_mode == "sample_id":
        duplicates = (
            lookup[
                "sample_id"
            ]
            .duplicated(
                keep=False
            )
        )

        if duplicates.any():
            check = (
                lookup.loc[
                    duplicates
                ]
                .groupby(
                    "sample_id"
                )[
                    "_true_template_id"
                ]
                .nunique(
                    dropna=False
                )
            )

            conflicts = check[
                check > 1
            ]

            if len(conflicts):
                raise RuntimeError(
                    "Conflicting y_template_id for duplicated sample_id. "
                    f"Conflict count={len(conflicts):,}"
                )

            lookup = (
                lookup.drop_duplicates(
                    "sample_id"
                )
                .reset_index(
                    drop=True
                )
            )

    else:
        keys = [
            "participant_id",
            "local_date",
            "local_slot_seconds",
        ]

        duplicates = (
            lookup.duplicated(
                keys,
                keep=False,
            )
        )

        if duplicates.any():
            check = (
                lookup.loc[
                    duplicates
                ]
                .groupby(
                    keys
                )[
                    "_true_template_id"
                ]
                .nunique(
                    dropna=False
                )
            )

            conflicts = check[
                check > 1
            ]

            if len(conflicts):
                raise RuntimeError(
                    "Conflicting y_template_id for composite key. "
                    f"Conflict count={len(conflicts):,}"
                )

            lookup = (
                lookup.drop_duplicates(
                    keys
                )
                .reset_index(
                    drop=True
                )
            )

    return (
        lookup,
        join_mode,
    )


def attach_templates(
    d,
    lookup,
    join_mode,
):
    if join_mode == "sample_id":
        if "sample_id" not in d.columns:
            raise KeyError(
                "08 prediction part has no sample_id."
            )

        out = d.merge(
            lookup,
            on="sample_id",
            how="left",
            validate="many_to_one",
        )

    else:
        keys = [
            "participant_id",
            "local_date",
            "local_slot_seconds",
        ]

        out = d.merge(
            lookup,
            on=keys,
            how="left",
            validate="many_to_one",
        )

    out[
        "_true_template_id"
    ] = (
        out[
            "_true_template_id"
        ]
        .astype(
            "string"
        )
        .fillna(
            "UNMAPPED"
        )
    )

    return out


def fit_template_latent_means(
    source_dir,
    source_manifest,
    bundle,
    latent_dim,
):
    """
    TRAIN-only template means in absolute latent space.
    Also produce global/train template statistics and within-template
    latent dispersion.
    """
    sums = {}
    sumsq = {}
    counts = defaultdict(int)

    global_sum = np.zeros(
        latent_dim,
        dtype=np.float64,
    )

    global_sumsq = np.zeros(
        latent_dim,
        dtype=np.float64,
    )

    global_count = 0

    files = part_files(
        source_manifest,
        "train",
    )

    pca = bundle["pca"]
    scaler = bundle["scaler"]

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            source_dir
            / rel
        )

        print(
            f"[template mean TRAIN {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        template_col = detect_template_col(
            d.columns
        )

        template = (
            d[
                template_col
            ]
            .astype(
                "string"
            )
            .fillna(
                "UNMAPPED"
            )
            .to_numpy()
        )

        v = current_vector(
            d
        )

        z = pca.transform(
            scaler.transform(
                v
            )
        )[
            :,
            :latent_dim,
        ]

        global_sum += z.sum(
            axis=0
        )

        global_sumsq += np.square(
            z
        ).sum(
            axis=0
        )

        global_count += len(z)

        for t in pd.unique(
            template
        ):
            mask = (
                template
                == t
            )

            n = int(
                mask.sum()
            )

            if t not in sums:
                sums[t] = np.zeros(
                    latent_dim,
                    dtype=np.float64,
                )

                sumsq[t] = np.zeros(
                    latent_dim,
                    dtype=np.float64,
                )

            sums[t] += z[
                mask
            ].sum(
                axis=0
            )

            sumsq[t] += np.square(
                z[
                    mask
                ]
            ).sum(
                axis=0
            )

            counts[t] += n

        del d, v, z
        gc.collect()

    if global_count <= 0:
        raise ValueError(
            "No TRAIN rows for template mean."
        )

    global_mean = (
        global_sum
        / global_count
    )

    global_var = np.maximum(
        global_sumsq
        / global_count
        - np.square(
            global_mean
        ),
        0.0,
    )

    global_rmse_from_mean = float(
        np.sqrt(
            global_var.mean()
        )
    )

    means = {}
    rows = []

    for t in sorted(
        counts,
        key=str,
    ):
        n = counts[t]

        mean = (
            sums[t]
            / n
        )

        var = np.maximum(
            sumsq[t]
            / n
            - np.square(
                mean
            ),
            0.0,
        )

        rmse = float(
            np.sqrt(
                var.mean()
            )
        )

        means[str(t)] = mean

        rows.append(
            {
                "template_id": str(
                    t
                ),
                "train_rows": int(
                    n
                ),
                "train_share": float(
                    n
                    / global_count
                ),
                "within_template_latent_rmse_from_mean": rmse,
                "global_latent_rmse_from_global_mean": (
                    global_rmse_from_mean
                ),
                "dispersion_ratio_vs_global": float(
                    rmse
                    / global_rmse_from_mean
                )
                if global_rmse_from_mean > 0
                else np.nan,
            }
        )

    return (
        means,
        global_mean,
        pd.DataFrame(
            rows
        ),
    )


def new_state():
    return {
        "rows": 0,
        "ae": 0.0,
        "se": 0.0,
        "abs_true": 0.0,
        "points": 0,
        "smape": 0.0,
        "curve_mae": [],
        "qa_ae": 0.0,
        "qa_abs": 0.0,
        "qs_ae": 0.0,
        "qs_abs": 0.0,
    }


def update_state(
    st,
    true_q,
    true_p,
    true_qa,
    true_qs,
    pred_v,
):
    p, qa, qs = unpack_vector(
        pred_v
    )

    pred = price_on_true_q(
        true_q,
        p,
        qa,
        qs,
    )

    err = (
        pred
        - true_p
    )

    ae = np.abs(
        err
    )

    denom = (
        np.abs(
            pred
        )
        + np.abs(
            true_p
        )
    )

    smape = np.divide(
        2.0 * ae,
        denom,
        out=np.zeros_like(
            ae
        ),
        where=(
            denom
            > 1e-9
        ),
    )

    st["rows"] += len(
        true_p
    )

    st["ae"] += float(
        ae.sum()
    )

    st["se"] += float(
        np.square(
            err
        ).sum()
    )

    st["abs_true"] += float(
        np.abs(
            true_p
        ).sum()
    )

    st["points"] += int(
        ae.size
    )

    st["smape"] += float(
        smape.sum()
    )

    st[
        "curve_mae"
    ].append(
        ae.mean(
            axis=1
        ).astype(
            np.float32
        )
    )

    st["qa_ae"] += float(
        np.abs(
            qa
            - true_qa
        ).sum()
    )

    st["qa_abs"] += float(
        np.abs(
            true_qa
        ).sum()
    )

    st["qs_ae"] += float(
        np.abs(
            qs
            - true_qs
        ).sum()
    )

    st["qs_abs"] += float(
        np.abs(
            true_qs
        ).sum()
    )


def finish_state(
    st,
    split,
    template_id,
    model,
):
    curves = (
        np.concatenate(
            st[
                "curve_mae"
            ]
        )
        if st[
            "curve_mae"
        ]
        else np.empty(
            0,
            dtype=np.float32,
        )
    )

    return {
        "split": split,
        "template_id": str(
            template_id
        ),
        "model": model,
        "rows": int(
            st["rows"]
        ),
        "price_mae": float(
            st["ae"]
            / max(
                st["points"],
                1,
            )
        ),
        "price_rmse": float(
            np.sqrt(
                st["se"]
                / max(
                    st["points"],
                    1,
                )
            )
        ),
        "price_wape_pct": float(
            100.0
            * st["ae"]
            / max(
                st["abs_true"],
                1e-12,
            )
        ),
        "price_smape_pct": float(
            100.0
            * st["smape"]
            / max(
                st["points"],
                1,
            )
        ),
        "curve_mae_p50": float(
            np.quantile(
                curves,
                0.50,
            )
        )
        if len(
            curves
        )
        else np.nan,
        "curve_mae_p90": float(
            np.quantile(
                curves,
                0.90,
            )
        )
        if len(
            curves
        )
        else np.nan,
        "curve_mae_p95": float(
            np.quantile(
                curves,
                0.95,
            )
        )
        if len(
            curves
        )
        else np.nan,
        "q_anchor_wape_pct": float(
            100.0
            * st["qa_ae"]
            / max(
                st["qa_abs"],
                1e-12,
            )
        ),
        "q_span_wape_pct": float(
            100.0
            * st["qs_ae"]
            / max(
                st["qs_abs"],
                1e-12,
            )
        ),
    }


def evaluate_split(
    prediction_dir,
    prediction_manifest,
    split,
    bundle,
    latent_cols,
    template_lookup,
    join_mode,
    template_means,
    global_mean,
):
    states = defaultdict(
        new_state
    )

    overall = defaultdict(
        new_state
    )

    files = part_files(
        prediction_manifest,
        split,
    )

    mapped_rows = 0
    total_rows = 0

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            prediction_dir
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

        d = attach_templates(
            d,
            template_lookup,
            join_mode,
        )

        total_rows += len(
            d
        )

        mapped_rows += int(
            d[
                "_true_template_id"
            ].ne(
                "UNMAPPED"
            ).sum()
        )

        (
            true_q,
            true_p,
            true_qa,
            true_qs,
            _,
        ) = true_curve(
            d
        )

        templates = (
            d[
                "_true_template_id"
            ]
            .astype(
                "string"
            )
            .to_numpy()
        )

        pred_vectors = {}

        for model in FEATURE_ONLY_MODELS:
            z = latent_matrix(
                d,
                f"pred_{model}_",
                latent_cols,
            )

            pred_vectors[
                model
            ] = decode_latent(
                z,
                bundle,
            )

        true_z = latent_matrix(
            d,
            "true_",
            latent_cols,
        )

        pred_vectors[
            "absolute_latent_oracle"
        ] = decode_latent(
            true_z,
            bundle,
        )

        pv = persistence_vector(
            d
        )

        if pv is not None:
            pred_vectors[
                "persistence_reference"
            ] = pv

        tm_z = np.empty(
            (
                len(d),
                len(
                    latent_cols
                ),
            ),
            dtype=np.float64,
        )

        for t in pd.unique(
            templates
        ):
            mask = (
                templates
                == t
            )

            mean = template_means.get(
                str(t),
                global_mean,
            )

            tm_z[
                mask,
                :,
            ] = mean

        pred_vectors[
            "oracle_template_mean"
        ] = decode_latent(
            tm_z,
            bundle,
        )

        unique_templates = pd.unique(
            templates
        )

        for model, pred_v in pred_vectors.items():
            update_state(
                overall[
                    model
                ],
                true_q,
                true_p,
                true_qa,
                true_qs,
                pred_v,
            )

            for t in unique_templates:
                mask = (
                    templates
                    == t
                )

                update_state(
                    states[
                        (
                            str(
                                t
                            ),
                            model,
                        )
                    ],
                    true_q[
                        mask
                    ],
                    true_p[
                        mask
                    ],
                    true_qa[
                        mask
                    ],
                    true_qs[
                        mask
                    ],
                    pred_v[
                        mask
                    ],
                )

        del (
            d,
            true_q,
            true_p,
            true_qa,
            true_qs,
            pred_vectors,
            true_z,
            tm_z,
        )

        gc.collect()

    template_rows = [
        finish_state(
            st,
            split,
            template,
            model,
        )
        for (
            template,
            model,
        ), st
        in states.items()
    ]

    overall_rows = [
        finish_state(
            st,
            split,
            "ALL",
            model,
        )
        for model, st
        in overall.items()
    ]

    coverage = {
        "split": split,
        "rows": int(
            total_rows
        ),
        "template_mapped_rows": int(
            mapped_rows
        ),
        "template_mapping_coverage": float(
            mapped_rows
            / max(
                total_rows,
                1,
            )
        ),
    }

    return (
        pd.DataFrame(
            template_rows
        ),
        pd.DataFrame(
            overall_rows
        ),
        coverage,
    )


def build_template_summary(
    metrics,
    dispersion,
):
    base = (
        metrics.loc[
            metrics[
                "model"
            ].eq(
                "random_forest"
            ),
            [
                "split",
                "template_id",
                "rows",
                "price_wape_pct",
                "price_mae",
            ],
        ]
        .rename(
            columns={
                "price_wape_pct":
                "rf_wape_pct",
                "price_mae":
                "rf_price_mae",
            }
        )
        .copy()
    )

    def model_metric(
        model,
        value_name,
    ):
        x = (
            metrics.loc[
                metrics[
                    "model"
                ].eq(
                    model
                ),
                [
                    "split",
                    "template_id",
                    "price_wape_pct",
                ],
            ]
            .rename(
                columns={
                    "price_wape_pct":
                    value_name
                }
            )
        )

        return x

    for model, name in [
        (
            "train_mean",
            "global_train_mean_wape_pct",
        ),
        (
            "oracle_template_mean",
            "oracle_template_mean_wape_pct",
        ),
        (
            "spline_gam",
            "gam_wape_pct",
        ),
        (
            "ridge",
            "ridge_wape_pct",
        ),
        (
            "persistence_reference",
            "persistence_wape_pct",
        ),
        (
            "absolute_latent_oracle",
            "absolute_latent_oracle_wape_pct",
        ),
    ]:
        base = base.merge(
            model_metric(
                model,
                name,
            ),
            on=[
                "split",
                "template_id",
            ],
            how="left",
            validate="one_to_one",
        )

    base[
        "template_mean_improvement_vs_global_mean_pp"
    ] = (
        base[
            "global_train_mean_wape_pct"
        ]
        - base[
            "oracle_template_mean_wape_pct"
        ]
    )

    base[
        "rf_improvement_vs_global_mean_pp"
    ] = (
        base[
            "global_train_mean_wape_pct"
        ]
        - base[
            "rf_wape_pct"
        ]
    )

    base[
        "rf_gap_vs_persistence_pp"
    ] = (
        base[
            "rf_wape_pct"
        ]
        - base[
            "persistence_wape_pct"
        ]
    )

    base = base.merge(
        dispersion,
        on="template_id",
        how="left",
        validate="many_to_one",
    )

    total_by_split = (
        base.groupby(
            "split"
        )[
            "rows"
        ]
        .transform(
            "sum"
        )
    )

    base[
        "split_row_share"
    ] = (
        base[
            "rows"
        ]
        / total_by_split
    )

    return base


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
        "--model-dir",
        default="macro_b_feature_only_regression_models",
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

    model_dir = (
        base
        / args.model_dir
    )

    source_manifest = json.loads(
        (
            source_dir
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

    bundle = joblib.load(
        representation_dir
        / "absolute_pca_bundle.joblib"
    )

    latent_cols = list(
        prediction_manifest[
            "latent_columns"
        ]
    )

    latent_dim = len(
        latent_cols
    )

    out_dir = (
        base
        / "macro_b_true_template_diagnostics"
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
        f"08e TRUE-template conditioning diagnostics - "
        f"{args.year}"
    )
    print("=" * 80)
    print(
        f"Absolute latent dimension = {latent_dim}"
    )
    print()

    (
        template_means,
        global_mean,
        dispersion,
    ) = fit_template_latent_means(
        source_dir,
        source_manifest,
        bundle,
        latent_dim,
    )

    dispersion.to_csv(
        out_dir
        / "template_latent_dispersion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dispersion[
        [
            "template_id",
            "train_rows",
            "train_share",
        ]
    ].to_csv(
        out_dir
        / "template_train_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_template_metrics = []
    all_overall_metrics = []
    coverage_rows = []

    for split in [
        "val",
        "test",
    ]:
        lookup, join_mode = (
            build_template_lookup(
                source_dir,
                source_manifest,
                split,
            )
        )

        (
            template_metrics,
            overall_metrics,
            coverage,
        ) = evaluate_split(
            model_dir,
            prediction_manifest,
            split,
            bundle,
            latent_cols,
            lookup,
            join_mode,
            template_means,
            global_mean,
        )

        coverage[
            "join_mode"
        ] = join_mode

        coverage_rows.append(
            coverage
        )

        all_template_metrics.append(
            template_metrics
        )

        all_overall_metrics.append(
            overall_metrics
        )

        del lookup
        gc.collect()

    template_metrics = pd.concat(
        all_template_metrics,
        ignore_index=True,
    )

    overall_metrics = pd.concat(
        all_overall_metrics,
        ignore_index=True,
    )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    template_metrics.to_csv(
        out_dir
        / "template_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_metrics.to_csv(
        out_dir
        / "overall_oracle_template_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage_df.to_csv(
        out_dir
        / "template_mapping_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = build_template_summary(
        template_metrics,
        dispersion,
    )

    summary_df = summary_df.sort_values(
        [
            "split",
            "split_row_share",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    summary_df.to_csv(
        out_dir
        / "template_diagnostic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def overall_value(
        split,
        model,
        col="price_wape_pct",
    ):
        x = overall_metrics.loc[
            overall_metrics[
                "split"
            ].eq(
                split
            )
            & overall_metrics[
                "model"
            ].eq(
                model
            ),
            col,
        ]

        return (
            float(
                x.iloc[0]
            )
            if len(x)
            else np.nan
        )

    val_global_mean = overall_value(
        "val",
        "train_mean",
    )

    val_template_mean = overall_value(
        "val",
        "oracle_template_mean",
    )

    val_rf = overall_value(
        "val",
        "random_forest",
    )

    test_global_mean = overall_value(
        "test",
        "train_mean",
    )

    test_template_mean = overall_value(
        "test",
        "oracle_template_mean",
    )

    test_rf = overall_value(
        "test",
        "random_forest",
    )

    test_persistence = overall_value(
        "test",
        "persistence_reference",
    )

    test_oracle = overall_value(
        "test",
        "absolute_latent_oracle",
    )

    test_template_gain = (
        100.0
        * (
            test_global_mean
            - test_template_mean
        )
        / test_global_mean
        if (
            np.isfinite(
                test_global_mean
            )
            and test_global_mean != 0
        )
        else np.nan
    )

    dispersion_weighted = float(
        np.average(
            dispersion[
                "dispersion_ratio_vs_global"
            ],
            weights=dispersion[
                "train_rows"
            ],
        )
    )

    test_summary = summary_df.loc[
        summary_df[
            "split"
        ].eq(
            "test"
        )
    ].copy()

    display_cols = [
        "template_id",
        "rows",
        "split_row_share",
        "global_train_mean_wape_pct",
        "oracle_template_mean_wape_pct",
        "rf_wape_pct",
        "gam_wape_pct",
        "persistence_wape_pct",
        "absolute_latent_oracle_wape_pct",
        "dispersion_ratio_vs_global",
    ]

    summary = "\n".join(
        [
            (
                f"08e TRUE-template conditioning diagnostics - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                "No regression model was retrained. "
                "TRUE template labels are used only for oracle diagnosis."
            ),
            "",
            "Template mapping coverage:",
            coverage_df.to_string(
                index=False
            ),
            "",
            (
                "Weighted within-template latent dispersion / "
                "global latent dispersion = "
                f"{dispersion_weighted:.6f}"
            ),
            "",
            "OVERALL VALIDATION:",
            (
                f"global train-mean WAPE       = "
                f"{val_global_mean:.6f}%"
            ),
            (
                f"oracle-template mean WAPE   = "
                f"{val_template_mean:.6f}%"
            ),
            (
                f"global Random Forest WAPE   = "
                f"{val_rf:.6f}%"
            ),
            "",
            "OVERALL TEST:",
            (
                f"global train-mean WAPE       = "
                f"{test_global_mean:.6f}%"
            ),
            (
                f"oracle-template mean WAPE   = "
                f"{test_template_mean:.6f}%"
            ),
            (
                f"global Random Forest WAPE   = "
                f"{test_rf:.6f}%"
            ),
            (
                f"persistence reference WAPE  = "
                f"{test_persistence:.6f}%"
            ),
            (
                f"absolute-latent oracle WAPE = "
                f"{test_oracle:.6f}%"
            ),
            (
                "TEST oracle-template mean relative improvement "
                "vs global train mean = "
                f"{test_template_gain:.4f}%"
            ),
            "",
            "TEST per-template diagnostics:",
            test_summary[
                display_cols
            ].to_string(
                index=False
            ),
            "",
            (
                "Interpretation: strong reduction from global train_mean "
                "to oracle_template_mean, plus materially lower within-"
                "template latent dispersion, supports template-specific "
                "regression/routing as the next experiment."
            ),
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
    print(
        f"Outputs: {out_dir}"
    )


if __name__ == "__main__":
    main()
