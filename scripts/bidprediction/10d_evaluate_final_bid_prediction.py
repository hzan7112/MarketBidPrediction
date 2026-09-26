#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10d_evaluate_final_bid_prediction.py

FINAL full-population acceptance evaluation.

Primary final model:
    Random Forest (fixed in 10c)

Same-information audit baselines:
    train mean
    Ridge
    Spline-GAM

References:
    absolute-latent representation oracle (8D)
    previous raw curve Persistence (extra historical bid information)

Metrics
-------
- coverage
- price MAE / RMSE / MAPE / sMAPE / WAPE
- per-curve MAE P50/P90/P95
- representation-knot price MAE
  (also reported as breakpoint_proxy_price_mae; the final 21-point latent
   representation has knots, not explicit structural bid breakpoints)
- five equal-quantity segment midpoint price MAEs
- q_anchor MAE/WAPE
- q_span MAE/WAPE
- q_max MAE/WAPE
- normalized shape MAE
- template consistency/accuracy when the Stage-2 template library is found

RF-only decomposition:
- by FLAT vs SHAPE
- by true template
- by participant

No final model selection is performed. random_forest is the frozen primary
feature-only model.

Run
---
python scripts/bidprediction/10d_evaluate_final_bid_prediction.py \
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


GRID = np.linspace(
    0.0,
    1.0,
    21,
)

MID_U = np.asarray(
    [
        0.10,
        0.30,
        0.50,
        0.70,
        0.90,
    ],
    dtype=np.float64,
)

SHAPE_COLS = [
    f"shape_v{i:02d}"
    for i in range(21)
]

MODELS = [
    "train_mean",
    "ridge",
    "spline_gam",
    "random_forest",
]

REF_PRICE_COLS = [
    f"reference_curvevec_p{i:02d}_lag1"
    for i in range(21)
]

REF_QA = (
    "reference_curvevec_q_anchor_lag1"
)

REF_LOG_QS = (
    "reference_curvevec_log_q_span_lag1"
)


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


def normalize_template_id(x):
    s = str(
        x
    ).strip()

    if not s or s.lower() in {
        "nan",
        "none",
        "<na>",
    }:
        return None

    if s.upper() == "FLAT":
        return "FLAT"

    try:
        i = int(
            float(
                s.upper()
                .replace(
                    "T",
                    "",
                )
            )
        )

        return f"T{i:02d}"

    except Exception:
        return s


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
        shape,
        flat,
    )


def unpack_vector(v):
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


def price_on_quantity(
    q,
    pred_price,
    pred_q_anchor,
    pred_q_span,
):
    pos = np.clip(
        (
            q
            - pred_q_anchor[
                :,
                None,
            ]
        )
        / np.maximum(
            pred_q_span[
                :,
                None,
            ],
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

    pca = bundle[
        "pca"
    ]

    scaler = bundle[
        "scaler"
    ]

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
        :z.shape[
            1
        ],
    ] = z

    return scaler.inverse_transform(
        pca.inverse_transform(
            full
        )
    )


def latent_matrix(
    d,
    prefix,
    targets,
):
    return np.column_stack(
        [
            num(
                d[
                    f"{prefix}{t}"
                ]
            ).to_numpy(
                np.float64
            )
            for t in targets
        ]
    )


def persistence_vector(d):
    required = [
        *REF_PRICE_COLS,
        REF_QA,
        REF_LOG_QS,
    ]

    if not all(
        c in d.columns
        for c in required
    ):
        return (
            None,
            None,
        )

    raw = (
        d[
            required
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    ready = (
        raw.notna()
        .all(
            axis=1
        )
        .to_numpy()
    )

    if not ready.any():
        return (
            None,
            ready,
        )

    v = np.column_stack(
        [
            *[
                num(
                    d[
                        c
                    ]
                ).to_numpy(
                    np.float64
                )
                for c in REF_PRICE_COLS
            ],
            num(
                d[
                    REF_QA
                ]
            ).to_numpy(
                np.float64
            ),
            num(
                d[
                    REF_LOG_QS
                ]
            ).to_numpy(
                np.float64
            ),
        ]
    )

    return (
        v,
        ready,
    )


def predicted_shape(
    pred_price,
    relative_flat_tol,
):
    p0 = pred_price[
        :,
        0,
    ]

    span = (
        pred_price[
            :,
            -1
        ]
        - p0
    )

    scale = np.maximum(
        1.0,
        np.max(
            np.abs(
                pred_price
            ),
            axis=1,
        ),
    )

    flat = (
        np.abs(
            span
        )
        <= (
            relative_flat_tol
            * scale
        )
    )

    out = np.zeros_like(
        pred_price,
        dtype=np.float64,
    )

    nonflat = (
        ~flat
    )

    if nonflat.any():
        out[
            nonflat,
            :,
        ] = (
            pred_price[
                nonflat,
                :,
            ]
            - p0[
                nonflat,
                None,
            ]
        ) / span[
            nonflat,
            None,
        ]

    return (
        out,
        flat,
    )


def discover_template_centers(
    bidtemplate_year,
):
    preferred = (
        bidtemplate_year
        / "template_library"
        / "curve_template_library.csv"
    )

    candidates = (
        [
            preferred
        ]
        if preferred.exists()
        else sorted(
            bidtemplate_year.rglob(
                "*.csv"
            )
        )
    )

    for path in candidates:
        try:
            header = pd.read_csv(
                path,
                nrows=0,
            ).columns.tolist()

        except Exception:
            continue

        if not all(
            c in header
            for c in SHAPE_COLS
        ):
            continue

        id_col = next(
            (
                c
                for c in [
                    "template_id",
                    "template",
                    "cluster_id",
                    "cluster_label",
                    "label",
                ]
                if c in header
            ),
            None,
        )

        if id_col is None:
            continue

        d = pd.read_csv(
            path,
            usecols=[
                id_col,
                *SHAPE_COLS,
            ],
        )

        if len(
            d
        ) > 100:
            continue

        centers = {}

        for _, row in d.iterrows():
            tid = normalize_template_id(
                row[
                    id_col
                ]
            )

            if tid is None:
                continue

            shape = pd.to_numeric(
                row[
                    SHAPE_COLS
                ],
                errors="coerce",
            ).to_numpy(
                np.float64
            )

            if tid == "FLAT":
                shape = np.nan_to_num(
                    shape,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

            if not np.isfinite(
                shape
            ).all():
                continue

            centers[
                tid
            ] = shape

        if centers:
            return (
                centers,
                path,
            )

    return (
        {},
        None,
    )


def assign_predicted_template(
    pred_shape,
    pred_flat,
    centers,
):
    n = len(
        pred_shape
    )

    out = np.empty(
        n,
        dtype=object,
    )

    out[
        :
    ] = None

    if "FLAT" in centers:
        out[
            pred_flat
        ] = "FLAT"

    nonflat_idx = np.flatnonzero(
        ~pred_flat
    )

    nonflat_centers = [
        (
            tid,
            c
        )
        for tid, c
        in centers.items()
        if tid != "FLAT"
    ]

    if (
        len(
            nonflat_idx
        )
        and nonflat_centers
    ):
        best_d = np.full(
            len(
                nonflat_idx
            ),
            np.inf,
            dtype=np.float64,
        )

        best_t = np.empty(
            len(
                nonflat_idx
            ),
            dtype=object,
        )

        for tid, center in nonflat_centers:
            dist = np.mean(
                np.square(
                    pred_shape[
                        nonflat_idx,
                        :,
                    ]
                    - center[
                        None,
                        :,
                    ]
                ),
                axis=1,
            )

            better = (
                dist
                < best_d
            )

            if better.any():
                best_d[
                    better
                ] = dist[
                    better
                ]

                best_t[
                    better
                ] = tid

        out[
            nonflat_idx
        ] = best_t

    if (
        "FLAT"
        not in centers
        and pred_flat.any()
    ):
        out[
            pred_flat
        ] = "FLAT"

    return out


def new_state():
    return {
        "rows": 0,
        "price_ae": 0.0,
        "price_se": 0.0,
        "price_abs_true": 0.0,
        "price_points": 0,
        "mape_sum": 0.0,
        "mape_points": 0,
        "smape_sum": 0.0,
        "smape_points": 0,
        "curve_mae": [],
        "knot_ae": 0.0,
        "knot_points": 0,
        "mid_ae": np.zeros(
            5,
            dtype=np.float64,
        ),
        "mid_rows": 0,
        "qa_ae": 0.0,
        "qa_abs": 0.0,
        "qs_ae": 0.0,
        "qs_abs": 0.0,
        "qmax_ae": 0.0,
        "qmax_abs": 0.0,
        "shape_ae": 0.0,
        "shape_points": 0,
        "template_correct": 0,
        "template_total": 0,
        "template_nonflat_correct": 0,
        "template_nonflat_total": 0,
    }


def row_price_errors(
    true_q,
    true_p,
    pred_v,
):
    pred_p, pred_qa, pred_qs = unpack_vector(
        pred_v
    )

    pred_on_true = price_on_quantity(
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

    return (
        ae.sum(
            axis=1
        ),
        np.abs(
            true_p
        ).sum(
            axis=1
        ),
    )


def update_state(
    st,
    true_q,
    true_p,
    true_qa,
    true_qs,
    true_shape,
    true_flat,
    pred_v,
    mape_eps,
    relative_flat_tol,
    true_template=None,
    centers=None,
):
    pred_p, pred_qa, pred_qs = unpack_vector(
        pred_v
    )

    pred_on_true = price_on_quantity(
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

    st[
        "rows"
    ] += len(
        true_p
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

    mape_mask = (
        np.abs(
            true_p
        )
        > mape_eps
    )

    if mape_mask.any():
        st[
            "mape_sum"
        ] += float(
            (
                ae[
                    mape_mask
                ]
                / np.abs(
                    true_p[
                        mape_mask
                    ]
                )
            ).sum()
        )

        st[
            "mape_points"
        ] += int(
            mape_mask.sum()
        )

    smape_den = (
        np.abs(
            pred_on_true
        )
        + np.abs(
            true_p
        )
    )

    smape_mask = (
        smape_den
        > mape_eps
    )

    if smape_mask.any():
        st[
            "smape_sum"
        ] += float(
            (
                2.0
                * ae[
                    smape_mask
                ]
                / smape_den[
                    smape_mask
                ]
            ).sum()
        )

        st[
            "smape_points"
        ] += int(
            smape_mask.sum()
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

    knot_ae = np.abs(
        pred_p
        - true_p
    )

    st[
        "knot_ae"
    ] += float(
        knot_ae.sum()
    )

    st[
        "knot_points"
    ] += int(
        knot_ae.size
    )

    true_mid_q = (
        true_qa[
            :,
            None,
        ]
        + true_qs[
            :,
            None,
        ]
        * MID_U[
            None,
            :,
        ]
    )

    true_mid_pos = (
        MID_U
        * 20.0
    )

    mid_lo = np.floor(
        true_mid_pos
    ).astype(
        np.int16
    )

    mid_hi = np.minimum(
        mid_lo + 1,
        20,
    )

    mid_frac = (
        true_mid_pos
        - mid_lo
    )

    true_mid_p = (
        true_p[
            :,
            mid_lo,
        ]
        + (
            true_p[
                :,
                mid_hi,
            ]
            - true_p[
                :,
                mid_lo,
            ]
        )
        * mid_frac[
            None,
            :,
        ]
    )

    pred_mid_p = price_on_quantity(
        true_mid_q,
        pred_p,
        pred_qa,
        pred_qs,
    )

    st[
        "mid_ae"
    ] += np.abs(
        pred_mid_p
        - true_mid_p
    ).sum(
        axis=0
    )

    st[
        "mid_rows"
    ] += len(
        true_p
    )

    st[
        "qa_ae"
    ] += float(
        np.abs(
            pred_qa
            - true_qa
        ).sum()
    )

    st[
        "qa_abs"
    ] += float(
        np.abs(
            true_qa
        ).sum()
    )

    st[
        "qs_ae"
    ] += float(
        np.abs(
            pred_qs
            - true_qs
        ).sum()
    )

    st[
        "qs_abs"
    ] += float(
        np.abs(
            true_qs
        ).sum()
    )

    pred_qmax = (
        pred_qa
        + pred_qs
    )

    true_qmax = (
        true_qa
        + true_qs
    )

    st[
        "qmax_ae"
    ] += float(
        np.abs(
            pred_qmax
            - true_qmax
        ).sum()
    )

    st[
        "qmax_abs"
    ] += float(
        np.abs(
            true_qmax
        ).sum()
    )

    pshape, pflat = predicted_shape(
        pred_p,
        relative_flat_tol,
    )

    st[
        "shape_ae"
    ] += float(
        np.abs(
            pshape
            - true_shape
        ).sum()
    )

    st[
        "shape_points"
    ] += int(
        true_shape.size
    )

    if (
        true_template
        is not None
        and centers
    ):
        pred_template = assign_predicted_template(
            pshape,
            pflat,
            centers,
        )

        true_t = np.asarray(
            [
                normalize_template_id(
                    x
                )
                for x
                in true_template
            ],
            dtype=object,
        )

        valid = np.asarray(
            [
                x is not None
                for x in true_t
            ],
            dtype=bool,
        )

        pred_valid = np.asarray(
            [
                x is not None
                for x in pred_template
            ],
            dtype=bool,
        )

        valid &= pred_valid

        if valid.any():
            correct = (
                true_t[
                    valid
                ]
                == pred_template[
                    valid
                ]
            )

            st[
                "template_correct"
            ] += int(
                correct.sum()
            )

            st[
                "template_total"
            ] += int(
                valid.sum()
            )

            nonflat = (
                valid
                & (
                    true_t
                    != "FLAT"
                )
            )

            if nonflat.any():
                st[
                    "template_nonflat_correct"
                ] += int(
                    (
                        true_t[
                            nonflat
                        ]
                        == pred_template[
                            nonflat
                        ]
                    ).sum()
                )

                st[
                    "template_nonflat_total"
                ] += int(
                    nonflat.sum()
                )


def finish_state(
    st,
    split,
    model,
    eligible,
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

    row = {
        "split": split,
        "model": model,
        "eligible_for_feature_only_comparison": bool(
            eligible
        ),
        "rows": int(
            st[
                "rows"
            ]
        ),
        "price_mae": float(
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
        "price_rmse": float(
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
        "price_mape_pct": float(
            100.0
            * st[
                "mape_sum"
            ]
            / max(
                st[
                    "mape_points"
                ],
                1,
            )
        ),
        "mape_evaluated_point_share": float(
            st[
                "mape_points"
            ]
            / max(
                st[
                    "price_points"
                ],
                1,
            )
        ),
        "price_smape_pct": float(
            100.0
            * st[
                "smape_sum"
            ]
            / max(
                st[
                    "smape_points"
                ],
                1,
            )
        ),
        "price_wape_pct": float(
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
        "representation_knot_price_mae": float(
            st[
                "knot_ae"
            ]
            / max(
                st[
                    "knot_points"
                ],
                1,
            )
        ),
        "breakpoint_proxy_price_mae": float(
            st[
                "knot_ae"
            ]
            / max(
                st[
                    "knot_points"
                ],
                1,
            )
        ),
        "q_anchor_mae_mw": float(
            st[
                "qa_ae"
            ]
            / max(
                st[
                    "rows"
                ],
                1,
            )
        ),
        "q_anchor_wape_pct": float(
            100.0
            * st[
                "qa_ae"
            ]
            / max(
                st[
                    "qa_abs"
                ],
                1e-12,
            )
        ),
        "q_span_mae_mw": float(
            st[
                "qs_ae"
            ]
            / max(
                st[
                    "rows"
                ],
                1,
            )
        ),
        "q_span_wape_pct": float(
            100.0
            * st[
                "qs_ae"
            ]
            / max(
                st[
                    "qs_abs"
                ],
                1e-12,
            )
        ),
        "q_max_mae_mw": float(
            st[
                "qmax_ae"
            ]
            / max(
                st[
                    "rows"
                ],
                1,
            )
        ),
        "q_max_wape_pct": float(
            100.0
            * st[
                "qmax_ae"
            ]
            / max(
                st[
                    "qmax_abs"
                ],
                1e-12,
            )
        ),
        "normalized_shape_mae": float(
            st[
                "shape_ae"
            ]
            / max(
                st[
                    "shape_points"
                ],
                1,
            )
        ),
        "template_accuracy": (
            float(
                st[
                    "template_correct"
                ]
                / st[
                    "template_total"
                ]
            )
            if st[
                "template_total"
            ]
            else np.nan
        ),
        "template_accuracy_nonflat": (
            float(
                st[
                    "template_nonflat_correct"
                ]
                / st[
                    "template_nonflat_total"
                ]
            )
            if st[
                "template_nonflat_total"
            ]
            else np.nan
        ),
        "template_evaluated_rows": int(
            st[
                "template_total"
            ]
        ),
    }

    for j in range(
        5
    ):
        row[
            f"segment{j+1}_midpoint_price_mae"
        ] = float(
            st[
                "mid_ae"
            ][
                j
            ]
            / max(
                st[
                    "mid_rows"
                ],
                1,
            )
        )

    return row


def update_group_table(
    store,
    group_values,
    row_ae,
    row_abs_true,
):
    temp = pd.DataFrame(
        {
            "group": group_values,
            "ae": row_ae,
            "abst": row_abs_true,
        }
    )

    temp = (
        temp.dropna(
            subset=[
                "group"
            ]
        )
        .groupby(
            "group",
            sort=False,
            as_index=False,
        )
        .agg(
            rows=(
                "ae",
                "size",
            ),
            ae=(
                "ae",
                "sum",
            ),
            abst=(
                "abst",
                "sum",
            ),
        )
    )

    for _, row in temp.iterrows():
        key = str(
            row[
                "group"
            ]
        )

        st = store[
            key
        ]

        st[
            "rows"
        ] += int(
            row[
                "rows"
            ]
        )

        st[
            "ae"
        ] += float(
            row[
                "ae"
            ]
        )

        st[
            "abst"
        ] += float(
            row[
                "abst"
            ]
        )


def finalize_group_table(
    store,
    group_name,
):
    rows = []

    for group, st in store.items():
        rows.append(
            {
                group_name: group,
                "rows": int(
                    st[
                        "rows"
                    ]
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
                "price_ae_sum": float(
                    st[
                        "ae"
                    ]
                ),
                "price_abs_true_sum": float(
                    st[
                        "abst"
                    ]
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "rows",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
        if rows
        else pd.DataFrame()
    )


def evaluate_split(
    model_dir,
    manifest,
    split,
    bundle,
    targets,
    centers,
    mape_eps,
    relative_flat_tol,
):
    names = [
        *MODELS,
        "absolute_latent_oracle",
        "persistence_reference",
    ]

    states = {
        name: new_state()
        for name in names
    }

    by_template = defaultdict(
        lambda: {
            "rows": 0,
            "ae": 0.0,
            "abst": 0.0,
        }
    )

    by_participant = defaultdict(
        lambda: {
            "rows": 0,
            "ae": 0.0,
            "abst": 0.0,
        }
    )

    by_curve_type = defaultdict(
        lambda: {
            "rows": 0,
            "ae": 0.0,
            "abst": 0.0,
        }
    )

    files = parts(
        manifest,
        split,
    )

    total_rows = 0

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            model_dir
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

        total_rows += len(
            d
        )

        (
            true_q,
            true_p,
            true_qa,
            true_qs,
            true_shape,
            true_flat,
        ) = true_curve(
            d
        )

        true_template = (
            d[
                "y_template_id"
            ].to_numpy(
                object
            )
            if "y_template_id"
            in d.columns
            else None
        )

        for name in MODELS:
            pred_v = decode_latent(
                latent_matrix(
                    d,
                    f"pred_{name}_",
                    targets,
                ),
                bundle,
            )

            update_state(
                states[
                    name
                ],
                true_q,
                true_p,
                true_qa,
                true_qs,
                true_shape,
                true_flat,
                pred_v,
                mape_eps,
                relative_flat_tol,
                true_template,
                centers,
            )

            if (
                name
                == "random_forest"
            ):
                row_ae, row_abst = row_price_errors(
                    true_q,
                    true_p,
                    pred_v,
                )

                curve_type = np.where(
                    true_flat,
                    "FLAT",
                    "SHAPE",
                )

                update_group_table(
                    by_curve_type,
                    curve_type,
                    row_ae,
                    row_abst,
                )

                if (
                    true_template
                    is not None
                ):
                    update_group_table(
                        by_template,
                        [
                            normalize_template_id(
                                x
                            )
                            for x
                            in true_template
                        ],
                        row_ae,
                        row_abst,
                    )

                if (
                    "participant_id"
                    in d.columns
                ):
                    update_group_table(
                        by_participant,
                        d[
                            "participant_id"
                        ].astype(
                            "string"
                        ).to_numpy(),
                        row_ae,
                        row_abst,
                    )

            del pred_v

        oracle_v = decode_latent(
            latent_matrix(
                d,
                "true_",
                targets,
            ),
            bundle,
        )

        update_state(
            states[
                "absolute_latent_oracle"
            ],
            true_q,
            true_p,
            true_qa,
            true_qs,
            true_shape,
            true_flat,
            oracle_v,
            mape_eps,
            relative_flat_tol,
            true_template,
            centers,
        )

        prev_v, prev_ready = persistence_vector(
            d
        )

        if (
            prev_v
            is not None
            and prev_ready
            is not None
            and prev_ready.any()
        ):
            update_state(
                states[
                    "persistence_reference"
                ],
                true_q[
                    prev_ready
                ],
                true_p[
                    prev_ready
                ],
                true_qa[
                    prev_ready
                ],
                true_qs[
                    prev_ready
                ],
                true_shape[
                    prev_ready
                ],
                true_flat[
                    prev_ready
                ],
                prev_v[
                    prev_ready
                ],
                mape_eps,
                relative_flat_tol,
                (
                    true_template[
                        prev_ready
                    ]
                    if true_template
                    is not None
                    else None
                ),
                centers,
            )

        del (
            d,
            true_q,
            true_p,
            true_qa,
            true_qs,
            true_shape,
            true_flat,
            oracle_v,
            prev_v,
            prev_ready,
        )

        gc.collect()

    rows = []

    for name in names:
        if states[
            name
        ][
            "rows"
        ] == 0:
            continue

        row = finish_state(
            states[
                name
            ],
            split,
            name,
            name
            in MODELS,
        )

        row[
            "coverage_vs_full_split"
        ] = float(
            row[
                "rows"
            ]
            / max(
                total_rows,
                1,
            )
        )

        rows.append(
            row
        )

    metrics = pd.DataFrame(
        rows
    )

    return (
        metrics,
        finalize_group_table(
            by_curve_type,
            "curve_type",
        ),
        finalize_group_table(
            by_template,
            "template_id",
        ),
        finalize_group_table(
            by_participant,
            "participant_id",
        ),
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
        "--representation-dir",
        default="final_absolute_curve_dataset",
    )
    ap.add_argument(
        "--model-dir",
        default="final_curve_regression_models",
    )
    ap.add_argument(
        "--bidtemplate-root",
        default="data/processed/bidtemplate",
    )
    ap.add_argument(
        "--mape-eps",
        type=float,
        default=1e-6,
    )
    ap.add_argument(
        "--relative-flat-tol",
        type=float,
        default=1e-4,
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

    rep_dir = (
        base
        / args.representation_dir
    )

    model_dir = (
        base
        / args.model_dir
    )

    bundle = joblib.load(
        rep_dir
        / "absolute_pca_bundle.joblib"
    )

    manifest = json.loads(
        (
            model_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    targets = list(
        manifest[
            "latent_columns"
        ]
    )

    if manifest.get(
        "primary_model"
    ) != "random_forest":
        raise RuntimeError(
            "10d expects the frozen primary model to be random_forest."
        )

    bidtemplate_year = (
        Path(
            args.bidtemplate_root
        )
        / str(
            args.year
        )
    )

    centers, center_file = discover_template_centers(
        bidtemplate_year
    )

    out = (
        base
        / "final_bid_prediction_evaluation"
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
        f"10d FINAL bid-prediction evaluation - {args.year}"
    )
    print("=" * 80)
    print(
        "Primary model = random_forest"
    )
    print(
        f"Absolute latent dim = {len(targets)}"
    )
    print(
        (
            f"Template library = {center_file}"
            if center_file
            is not None
            else (
                "Template library = NOT FOUND; "
                "template consistency will be NaN"
            )
        )
    )
    print()

    all_metrics = []

    for split in [
        "val",
        "test",
    ]:
        (
            metrics,
            by_curve_type,
            by_template,
            by_participant,
        ) = evaluate_split(
            model_dir,
            manifest,
            split,
            bundle,
            targets,
            centers,
            args.mape_eps,
            args.relative_flat_tol,
        )

        metrics.to_csv(
            out
            / f"{split}_final_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        by_curve_type.to_csv(
            out
            / f"{split}_random_forest_by_curve_type.csv",
            index=False,
            encoding="utf-8-sig",
        )

        by_template.to_csv(
            out
            / f"{split}_random_forest_by_template.csv",
            index=False,
            encoding="utf-8-sig",
        )

        by_participant.to_csv(
            out
            / f"{split}_random_forest_by_participant.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_metrics.append(
            metrics
        )

    all_metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    all_metrics.to_csv(
        out
        / "all_final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_model = {
        "primary_model": "random_forest",
        "task": (
            "full-population feature-only absolute bid-curve prediction"
        ),
        "model_selection_status": (
            "frozen before final full-population evaluation"
        ),
        "test_used_for_model_selection": False,
        "latent_dim": int(
            len(
                targets
            )
        ),
        "direct_historical_bid_input": False,
        "persistence_reference_extra_information": True,
        "template_library": (
            str(
                center_file
            )
            if center_file
            is not None
            else None
        ),
        "breakpoint_metric_note": (
            "breakpoint_proxy_price_mae equals direct MAE at the 21 "
            "absolute-curve representation knots. The final latent method "
            "does not define five structural bid breakpoints."
        ),
    }

    (
        out
        / "final_model.json"
    ).write_text(
        json.dumps(
            final_model,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def row(
        split,
        model,
    ):
        x = all_metrics.loc[
            all_metrics[
                "split"
            ].eq(
                split
            )
            & all_metrics[
                "model"
            ].eq(
                model
            )
        ]

        return (
            x.iloc[
                0
            ]
            if len(
                x
            )
            else None
        )

    def get(
        r,
        c,
    ):
        return (
            float(
                r[
                    c
                ]
            )
            if r
            is not None
            and pd.notna(
                r[
                    c
                ]
            )
            else np.nan
        )

    val_rf = row(
        "val",
        "random_forest",
    )

    test_rf = row(
        "test",
        "random_forest",
    )

    test_mean = row(
        "test",
        "train_mean",
    )

    test_ridge = row(
        "test",
        "ridge",
    )

    test_gam = row(
        "test",
        "spline_gam",
    )

    test_oracle = row(
        "test",
        "absolute_latent_oracle",
    )

    test_persist = row(
        "test",
        "persistence_reference",
    )

    summary = "\n".join(
        [
            (
                f"10d FINAL bid-prediction evaluation - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            "FINAL TECHNICAL ROUTE:",
            (
                "83 feature-only inputs -> Random Forest -> "
                "8D absolute latent -> complete bid curve"
            ),
            (
                "No previous complete bid curve / historical latent / "
                "theta / template routing is used by the final model."
            ),
            "",
            "PRIMARY MODEL = random_forest",
            "TEST is not used for model selection.",
            "",
            "VALIDATION primary RF:",
            (
                f"price MAE   = "
                f"{get(val_rf, 'price_mae'):.6f}"
            ),
            (
                f"price RMSE  = "
                f"{get(val_rf, 'price_rmse'):.6f}"
            ),
            (
                f"price MAPE  = "
                f"{get(val_rf, 'price_mape_pct'):.6f}%"
            ),
            (
                f"price sMAPE = "
                f"{get(val_rf, 'price_smape_pct'):.6f}%"
            ),
            (
                f"price WAPE  = "
                f"{get(val_rf, 'price_wape_pct'):.6f}%"
            ),
            "",
            "TEST final comparison:",
            (
                f"train_mean WAPE       = "
                f"{get(test_mean, 'price_wape_pct'):.6f}%"
            ),
            (
                f"Ridge WAPE            = "
                f"{get(test_ridge, 'price_wape_pct'):.6f}%"
            ),
            (
                f"Spline-GAM WAPE       = "
                f"{get(test_gam, 'price_wape_pct'):.6f}%"
            ),
            (
                f"FINAL Random Forest   = "
                f"{get(test_rf, 'price_wape_pct'):.6f}%"
            ),
            (
                f"8D representation oracle = "
                f"{get(test_oracle, 'price_wape_pct'):.6f}%"
            ),
            (
                f"Persistence reference = "
                f"{get(test_persist, 'price_wape_pct'):.6f}%"
            ),
            (
                f"Persistence coverage   = "
                f"{get(test_persist, 'coverage_vs_full_split'):.6f}"
            ),
            "",
            "TEST final Random Forest acceptance metrics:",
            (
                f"coverage = "
                f"{get(test_rf, 'coverage_vs_full_split'):.6f}"
            ),
            (
                f"MAE = "
                f"{get(test_rf, 'price_mae'):.6f}"
            ),
            (
                f"RMSE = "
                f"{get(test_rf, 'price_rmse'):.6f}"
            ),
            (
                f"MAPE = "
                f"{get(test_rf, 'price_mape_pct'):.6f}%"
            ),
            (
                f"sMAPE = "
                f"{get(test_rf, 'price_smape_pct'):.6f}%"
            ),
            (
                f"WAPE = "
                f"{get(test_rf, 'price_wape_pct'):.6f}%"
            ),
            (
                f"representation-knot / breakpoint-proxy price MAE = "
                f"{get(test_rf, 'representation_knot_price_mae'):.6f}"
            ),
            (
                "segment midpoint price MAE = ["
                + ", ".join(
                    f"{get(test_rf, f'segment{i}_midpoint_price_mae'):.6f}"
                    for i in range(
                        1,
                        6,
                    )
                )
                + "]"
            ),
            (
                f"q_anchor MAE/WAPE = "
                f"{get(test_rf, 'q_anchor_mae_mw'):.6f} MW / "
                f"{get(test_rf, 'q_anchor_wape_pct'):.6f}%"
            ),
            (
                f"q_span MAE/WAPE = "
                f"{get(test_rf, 'q_span_mae_mw'):.6f} MW / "
                f"{get(test_rf, 'q_span_wape_pct'):.6f}%"
            ),
            (
                f"q_max MAE/WAPE = "
                f"{get(test_rf, 'q_max_mae_mw'):.6f} MW / "
                f"{get(test_rf, 'q_max_wape_pct'):.6f}%"
            ),
            (
                f"normalized shape MAE = "
                f"{get(test_rf, 'normalized_shape_mae'):.6f}"
            ),
            (
                f"template accuracy = "
                f"{get(test_rf, 'template_accuracy'):.6f}"
            ),
            (
                f"template accuracy non-FLAT = "
                f"{get(test_rf, 'template_accuracy_nonflat'):.6f}"
            ),
            "",
            (
                "Persistence is an extra-information reference only and "
                "is not a same-input competitor to the final feature-only RF."
            ),
            (
                "breakpoint_proxy_price_mae is the direct error at the "
                "21 representation knots; the final latent method does not "
                "impose artificial five-breakpoint parameters."
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
