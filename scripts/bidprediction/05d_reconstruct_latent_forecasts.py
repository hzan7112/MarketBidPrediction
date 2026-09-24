#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05d_reconstruct_latent_forecasts.py

NEW MAIN ROUTE - Step 05d
=========================

Decode latent forecasts back to bid curves and perform FINAL model selection.

Validation:
    compare latent_persistence / ridge / spline_gam / random_forest
    by reconstructed full-curve PRICE WAPE.

Selection:
    choose the smallest validation price WAPE.
    TEST is not used for model selection.

Test:
    evaluate only:
    - raw_curve_persistence:
        previous observed curve vector directly;
    - latent_persistence:
        previous latent z decoded through PCA;
    - selected lightweight regression model;
    - latent_oracle:
        current true latent z decoded through PCA
        (representation floor; not a forecast).

Curve metric
------------
Predicted price is evaluated on the TRUE quantity grid.

For predicted:
    q_pred(u) = q_anchor_pred + q_span_pred * u
    p_pred(u) = decoded PCA price ordinates

For every true quantity point, predicted price is linearly interpolated on
q_pred. This makes q_anchor/q_span errors affect the final price-curve metric
and keeps evaluation comparable to the previous template+theta pipeline.

Outputs
-------
data/processed/bidprediction/<year>/latent_curve_forecast_evaluation/
    validation_curve_metrics.csv
    selected_model.json
    test_curve_metrics.csv
    summary.txt

Run
---
python scripts/bidprediction/05d_reconstruct_latent_forecasts.py --year 2025 --overwrite
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

LEARNED_MODEL_CANDIDATES = [
    "latent_persistence",
    "ridge",
    "spline_gam",
    "random_forest",
]


def prediction_files(
    manifest,
    split,
):
    files = []

    for item in manifest[
        "parts"
    ][
        split
    ]:
        if isinstance(
            item,
            dict,
        ):
            files.append(
                item["file"]
            )
        else:
            files.append(
                item
            )

    return files


def true_curve(
    d,
):
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

    q_anchor = pd.to_numeric(
        d[
            "q_anchor_mw"
        ],
        errors="coerce",
    ).to_numpy(
        np.float64
    )

    q_span = pd.to_numeric(
        d[
            "q_span_mw"
        ],
        errors="coerce",
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


def decode_latent(
    z,
    pca_bundle,
):
    z = np.asarray(
        z,
        dtype=np.float64,
    )

    pca = pca_bundle[
        "pca"
    ]

    scaler = pca_bundle[
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
        :z.shape[1],
    ] = z

    vector = scaler.inverse_transform(
        pca.inverse_transform(
            full
        )
    )

    price = vector[
        :,
        :21,
    ]

    q_anchor = vector[
        :,
        21,
    ]

    q_span = np.exp(
        np.clip(
            vector[
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


def raw_persistence_curve(
    d,
):
    price = np.column_stack(
        [
            pd.to_numeric(
                d[
                    f"_curvevec_p{i:02d}_lag1"
                ],
                errors="coerce",
            ).to_numpy(
                np.float64
            )
            for i in range(
                21
            )
        ]
    )

    q_anchor = pd.to_numeric(
        d[
            "_curvevec_q_anchor_lag1"
        ],
        errors="coerce",
    ).to_numpy(
        np.float64
    )

    log_q_span = pd.to_numeric(
        d[
            "_curvevec_log_q_span_lag1"
        ],
        errors="coerce",
    ).to_numpy(
        np.float64
    )

    q_span = np.exp(
        np.clip(
            log_q_span,
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

    f = (
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
        + f
        * (
            p_hi
            - p_lo
        )
    )


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

    err = (
        pred_on_true
        - true_p
    )

    ae = np.abs(
        err
    )

    denom = (
        np.abs(
            pred_on_true
        )
        + np.abs(
            true_p
        )
    )

    smape = np.divide(
        2.0
        * ae,
        denom,
        out=np.zeros_like(
            ae
        ),
        where=denom > 1e-9,
    )

    state[
        "rows"
    ] += int(
        len(
            true_p
        )
    )

    state[
        "price_ae"
    ] += float(
        ae.sum()
    )

    state[
        "price_se"
    ] += float(
        np.square(
            err
        ).sum()
    )

    state[
        "price_abs_true"
    ] += float(
        np.abs(
            true_p
        ).sum()
    )

    state[
        "price_points"
    ] += int(
        ae.size
    )

    state[
        "smape_sum"
    ] += float(
        smape.sum()
    )

    state[
        "smape_points"
    ] += int(
        smape.size
    )

    state[
        "curve_mae"
    ].append(
        ae.mean(
            axis=1
        ).astype(
            np.float32
        )
    )

    state[
        "q_anchor_ae"
    ] += float(
        np.abs(
            pred_qa
            - true_qa
        ).sum()
    )

    state[
        "q_anchor_abs_true"
    ] += float(
        np.abs(
            true_qa
        ).sum()
    )

    state[
        "q_span_ae"
    ] += float(
        np.abs(
            pred_qs
            - true_qs
        ).sum()
    )

    state[
        "q_span_abs_true"
    ] += float(
        np.abs(
            true_qs
        ).sum()
    )


def finalize_state(
    state,
):
    curve = (
        np.concatenate(
            state[
                "curve_mae"
            ]
        )
        if state[
            "curve_mae"
        ]
        else np.empty(
            0
        )
    )

    return {
        "rows": int(
            state[
                "rows"
            ]
        ),
        "price_mae": (
            state[
                "price_ae"
            ]
            / max(
                state[
                    "price_points"
                ],
                1,
            )
        ),
        "price_rmse": float(
            np.sqrt(
                state[
                    "price_se"
                ]
                / max(
                    state[
                        "price_points"
                    ],
                    1,
                )
            )
        ),
        "price_wape_pct": (
            100.0
            * state[
                "price_ae"
            ]
            / max(
                state[
                    "price_abs_true"
                ],
                1e-12,
            )
        ),
        "price_smape_pct": (
            100.0
            * state[
                "smape_sum"
            ]
            / max(
                state[
                    "smape_points"
                ],
                1,
            )
        ),
        "curve_mae_p50": (
            float(
                np.quantile(
                    curve,
                    0.50,
                )
            )
            if len(
                curve
            )
            else np.nan
        ),
        "curve_mae_p90": (
            float(
                np.quantile(
                    curve,
                    0.90,
                )
            )
            if len(
                curve
            )
            else np.nan
        ),
        "curve_mae_p95": (
            float(
                np.quantile(
                    curve,
                    0.95,
                )
            )
            if len(
                curve
            )
            else np.nan
        ),
        "curve_mae_le20_share": (
            float(
                np.mean(
                    curve
                    <= 20.0
                )
            )
            if len(
                curve
            )
            else np.nan
        ),
        "q_anchor_wape_pct": (
            100.0
            * state[
                "q_anchor_ae"
            ]
            / max(
                state[
                    "q_anchor_abs_true"
                ],
                1e-12,
            )
        ),
        "q_span_wape_pct": (
            100.0
            * state[
                "q_span_ae"
            ]
            / max(
                state[
                    "q_span_abs_true"
                ],
                1e-12,
            )
        ),
    }


def latent_matrix(
    d,
    prefix,
    latent_cols,
):
    return np.column_stack(
        [
            pd.to_numeric(
                d[
                    f"{prefix}{z}"
                ],
                errors="coerce",
            ).to_numpy(
                np.float64
            )
            for z in latent_cols
        ]
    )


def evaluate_validation(
    pred_dir,
    manifest,
    pca_bundle,
    latent_cols,
):
    states = {
        name: empty_state()
        for name in [
            "raw_curve_persistence",
            "latent_persistence",
            "ridge",
            "spline_gam",
            "random_forest",
            "latent_oracle",
        ]
    }

    files = prediction_files(
        manifest,
        "val",
    )

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            pred_dir
            / rel
        )

        print(
            f"[VAL {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        (
            true_q,
            true_p,
            true_qa,
            true_qs,
        ) = true_curve(
            d
        )

        raw_p, raw_qa, raw_qs = raw_persistence_curve(
            d
        )

        update_state(
            states[
                "raw_curve_persistence"
            ],
            true_q,
            true_p,
            true_qa,
            true_qs,
            raw_p,
            raw_qa,
            raw_qs,
        )

        for name in [
            "latent_persistence",
            "ridge",
            "spline_gam",
            "random_forest",
        ]:
            z = latent_matrix(
                d,
                f"pred_{name}_",
                latent_cols,
            )

            p, qa, qs = decode_latent(
                z,
                pca_bundle,
            )

            update_state(
                states[
                    name
                ],
                true_q,
                true_p,
                true_qa,
                true_qs,
                p,
                qa,
                qs,
            )

        z_true = latent_matrix(
            d,
            "true_",
            latent_cols,
        )

        p, qa, qs = decode_latent(
            z_true,
            pca_bundle,
        )

        update_state(
            states[
                "latent_oracle"
            ],
            true_q,
            true_p,
            true_qa,
            true_qs,
            p,
            qa,
            qs,
        )

        del d
        gc.collect()

    rows = []

    for name, state in states.items():
        row = finalize_state(
            state
        )

        row[
            "model"
        ] = name

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def evaluate_test_selected(
    pred_dir,
    manifest,
    pca_bundle,
    latent_cols,
    selected_model,
):
    modes = [
        "raw_curve_persistence",
        "latent_persistence",
        selected_model,
        "latent_oracle",
    ]

    # Remove duplicate when selected model is latent_persistence.
    modes = list(
        dict.fromkeys(
            modes
        )
    )

    states = {
        name: empty_state()
        for name in modes
    }

    files = prediction_files(
        manifest,
        "test",
    )

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            pred_dir
            / rel
        )

        print(
            f"[TEST {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        (
            true_q,
            true_p,
            true_qa,
            true_qs,
        ) = true_curve(
            d
        )

        if (
            "raw_curve_persistence"
            in states
        ):
            raw_p, raw_qa, raw_qs = raw_persistence_curve(
                d
            )

            update_state(
                states[
                    "raw_curve_persistence"
                ],
                true_q,
                true_p,
                true_qa,
                true_qs,
                raw_p,
                raw_qa,
                raw_qs,
            )

        if (
            "latent_persistence"
            in states
        ):
            z = latent_matrix(
                d,
                "pred_latent_persistence_",
                latent_cols,
            )

            p, qa, qs = decode_latent(
                z,
                pca_bundle,
            )

            update_state(
                states[
                    "latent_persistence"
                ],
                true_q,
                true_p,
                true_qa,
                true_qs,
                p,
                qa,
                qs,
            )

        if (
            selected_model
            not in {
                "latent_persistence",
                "raw_curve_persistence",
            }
        ):
            z = latent_matrix(
                d,
                f"pred_{selected_model}_",
                latent_cols,
            )

            p, qa, qs = decode_latent(
                z,
                pca_bundle,
            )

            update_state(
                states[
                    selected_model
                ],
                true_q,
                true_p,
                true_qa,
                true_qs,
                p,
                qa,
                qs,
            )

        z_true = latent_matrix(
            d,
            "true_",
            latent_cols,
        )

        p, qa, qs = decode_latent(
            z_true,
            pca_bundle,
        )

        update_state(
            states[
                "latent_oracle"
            ],
            true_q,
            true_p,
            true_qa,
            true_qs,
            p,
            qa,
            qs,
        )

        del d
        gc.collect()

    rows = []

    for name, state in states.items():
        row = finalize_state(
            state
        )

        row[
            "model"
        ] = name

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
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
        "--latent-dir",
        default="curve_latent_pca",
    )
    ap.add_argument(
        "--prediction-dir",
        default="latent_regression_models",
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

    latent_dir = (
        base
        / args.latent_dir
    )

    pred_dir = (
        base
        / args.prediction_dir
    )

    pca_bundle = joblib.load(
        latent_dir
        / "pca_bundle.joblib"
    )

    manifest = json.loads(
        (
            pred_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    latent_cols = list(
        manifest[
            "latent_columns"
        ]
    )

    out = (
        base
        / "latent_curve_forecast_evaluation"
    )

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} already exists. "
                f"Use --overwrite."
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
        f"Latent bid-curve forecast evaluation - "
        f"{args.year}"
    )
    print("=" * 80)
    print(
        f"Latent dimension = {len(latent_cols)}"
    )
    print()

    validation = evaluate_validation(
        pred_dir,
        manifest,
        pca_bundle,
        latent_cols,
    )

    validation.to_csv(
        out
        / "validation_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selectable = (
        validation.loc[
            validation[
                "model"
            ].isin(
                LEARNED_MODEL_CANDIDATES
            )
        ]
        .sort_values(
            [
                "price_wape_pct",
                "price_mae",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    if selectable.empty:
        raise ValueError(
            "No selectable model in validation results."
        )

    selected_model = str(
        selectable.iloc[
            0
        ][
            "model"
        ]
    )

    selected_validation = (
        selectable.iloc[
            0
        ].to_dict()
    )

    selection = {
        "selection_split": "validation",
        "selection_metric": (
            "reconstructed full-curve price WAPE"
        ),
        "selected_model": selected_model,
        "selected_validation_metrics": (
            selected_validation
        ),
        "test_used_for_selection": False,
        "candidate_models": (
            LEARNED_MODEL_CANDIDATES
        ),
    }

    (
        out
        / "selected_model.json"
    ).write_text(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Validation curve metrics:"
    )
    print(
        validation.to_string(
            index=False
        )
    )
    print()
    print(
        f"Selected model from VALIDATION = "
        f"{selected_model}"
    )
    print()

    test = evaluate_test_selected(
        pred_dir,
        manifest,
        pca_bundle,
        latent_cols,
        selected_model,
    )

    test.to_csv(
        out
        / "test_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def get_wape(
        table,
        model,
    ):
        x = table.loc[
            table[
                "model"
            ].eq(
                model
            ),
            "price_wape_pct",
        ]

        return (
            float(
                x.iloc[
                    0
                ]
            )
            if len(
                x
            )
            else np.nan
        )

    test_raw = get_wape(
        test,
        "raw_curve_persistence",
    )

    test_latent_p = get_wape(
        test,
        "latent_persistence",
    )

    test_selected = get_wape(
        test,
        selected_model,
    )

    relative_to_raw = (
        100.0
        * (
            test_raw
            - test_selected
        )
        / test_raw
        if (
            np.isfinite(
                test_raw
            )
            and test_raw
            != 0
        )
        else np.nan
    )

    summary = "\n".join(
        [
            (
                f"Latent bid-curve forecast evaluation - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                f"Latent dimension = "
                f"{len(latent_cols)}"
            ),
            (
                f"Selected model from VALIDATION = "
                f"{selected_model}"
            ),
            (
                "Selection metric = reconstructed "
                "full-curve price WAPE"
            ),
            "",
            "Validation:",
            validation.to_string(
                index=False
            ),
            "",
            "TEST (frozen validation choice):",
            test.to_string(
                index=False
            ),
            "",
            (
                f"TEST raw-curve persistence WAPE = "
                f"{test_raw:.6f}%"
            ),
            (
                f"TEST latent persistence WAPE = "
                f"{test_latent_p:.6f}%"
            ),
            (
                f"TEST selected-model WAPE = "
                f"{test_selected:.6f}%"
            ),
            (
                "Relative improvement vs raw-curve "
                f"persistence = {relative_to_raw:.4f}%"
            ),
            "",
            (
                "latent_oracle is the PCA representation floor "
                "on the same evaluation rows; it is not a forecast."
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
