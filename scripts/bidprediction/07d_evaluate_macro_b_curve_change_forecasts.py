#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07d_evaluate_macro_b_curve_change_forecasts.py

Final Macro-B screening evaluation.

Decode predicted DeltaZ -> DeltaCurve, add it to the PREVIOUS RAW curve, and
evaluate price prediction on the CURRENT true quantity grid.

Validation selects the winner among:
    zero_change
    ridge
    spline_gam
    random_forest

TEST is never used for model selection.

The script reports all TEST models for diagnosis, but the frozen final choice
is the model selected on validation.

Run:
python scripts/bidprediction/07d_evaluate_macro_b_curve_change_forecasts.py --year 2025 --overwrite
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

MODELS = [
    "zero_change",
    "ridge",
    "spline_gam",
    "random_forest",
]

PREV_PRICE_COLS = [
    f"_curvevec_p{i:02d}_lag1"
    for i in range(21)
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def files_from_manifest(manifest, split):
    return [
        x["file"] if isinstance(x, dict) else x
        for x in manifest["parts"][split]
    ]


def true_curve(d):
    shape = (
        d[SHAPE_COLS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(np.float64)
    )

    pa = num(
        d["p_anchor"]
    ).to_numpy(np.float64)

    ps = num(
        d["p_span"]
    ).to_numpy(np.float64)

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

    qa = num(
        d["q_anchor_mw"]
    ).to_numpy(np.float64)

    qs = num(
        d["q_span_mw"]
    ).to_numpy(np.float64)

    q = (
        qa[:, None]
        + qs[:, None]
        * GRID[None, :]
    )

    return q, price, qa, qs


def previous_vector(d):
    return np.column_stack(
        [
            *[
                num(
                    d[c]
                ).to_numpy(np.float64)
                for c in PREV_PRICE_COLS
            ],
            num(
                d[
                    "_curvevec_q_anchor_lag1"
                ]
            ).to_numpy(np.float64),
            num(
                d[
                    "_curvevec_log_q_span_lag1"
                ]
            ).to_numpy(np.float64),
        ]
    )


def vector_to_curve(v):
    price = v[:, :21]
    qa = v[:, 21]
    qs = np.exp(
        np.clip(
            v[:, 22],
            -20.0,
            20.0,
        )
    )
    return price, qa, qs


def decode_delta(z, bundle):
    scaler = bundle["scaler"]
    pca = bundle["pca"]

    z = np.asarray(
        z,
        dtype=np.float64,
    )

    full = np.zeros(
        (
            len(z),
            int(pca.n_components_),
        ),
        dtype=np.float64,
    )

    full[:, :z.shape[1]] = z

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
                    f"{prefix}{z}"
                ]
            ).to_numpy(np.float64)
            for z in targets
        ]
    )


def price_on_true_q(
    true_q,
    pred_price,
    pred_qa,
    pred_qs,
):
    span = np.maximum(
        pred_qs,
        1e-8,
    )

    x = (
        true_q
        - pred_qa[:, None]
    ) / span[:, None]

    pos = np.clip(
        x,
        0.0,
        1.0,
    ) * 20.0

    lo = np.floor(
        pos
    ).astype(np.int16)

    hi = np.minimum(
        lo + 1,
        20,
    )

    frac = pos - lo

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
        * (phi - plo)
    )


def new_state():
    return {
        "rows": 0,
        "ae": 0.0,
        "se": 0.0,
        "abs_true": 0.0,
        "n": 0,
        "smape_sum": 0.0,
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
    pp, pqa, pqs = vector_to_curve(
        pred_v
    )

    pred = price_on_true_q(
        true_q,
        pp,
        pqa,
        pqs,
    )

    err = pred - true_p
    ae = np.abs(err)

    denom = (
        np.abs(pred)
        + np.abs(true_p)
    )

    smape = np.divide(
        2.0 * ae,
        denom,
        out=np.zeros_like(ae),
        where=denom > 1e-9,
    )

    st["rows"] += len(true_p)
    st["ae"] += float(ae.sum())
    st["se"] += float(
        np.square(err).sum()
    )
    st["abs_true"] += float(
        np.abs(true_p).sum()
    )
    st["n"] += int(ae.size)
    st["smape_sum"] += float(
        smape.sum()
    )
    st["curve_mae"].append(
        ae.mean(axis=1).astype(
            np.float32
        )
    )

    st["qa_ae"] += float(
        np.abs(
            pqa - true_qa
        ).sum()
    )
    st["qa_abs"] += float(
        np.abs(true_qa).sum()
    )
    st["qs_ae"] += float(
        np.abs(
            pqs - true_qs
        ).sum()
    )
    st["qs_abs"] += float(
        np.abs(true_qs).sum()
    )


def finalize(st, split, model):
    c = (
        np.concatenate(
            st["curve_mae"]
        )
        if st["curve_mae"]
        else np.empty(0)
    )

    return {
        "split": split,
        "model": model,
        "rows": int(st["rows"]),
        "price_mae": (
            st["ae"]
            / max(
                st["n"],
                1,
            )
        ),
        "price_rmse": float(
            np.sqrt(
                st["se"]
                / max(
                    st["n"],
                    1,
                )
            )
        ),
        "price_wape_pct": (
            100.0
            * st["ae"]
            / max(
                st["abs_true"],
                1e-12,
            )
        ),
        "price_smape_pct": (
            100.0
            * st["smape_sum"]
            / max(
                st["n"],
                1,
            )
        ),
        "curve_mae_p50": (
            float(
                np.quantile(
                    c,
                    0.50,
                )
            )
            if len(c)
            else np.nan
        ),
        "curve_mae_p90": (
            float(
                np.quantile(
                    c,
                    0.90,
                )
            )
            if len(c)
            else np.nan
        ),
        "curve_mae_p95": (
            float(
                np.quantile(
                    c,
                    0.95,
                )
            )
            if len(c)
            else np.nan
        ),
        "q_anchor_wape_pct": (
            100.0
            * st["qa_ae"]
            / max(
                st["qa_abs"],
                1e-12,
            )
        ),
        "q_span_wape_pct": (
            100.0
            * st["qs_ae"]
            / max(
                st["qs_abs"],
                1e-12,
            )
        ),
    }


def evaluate_split(
    model_dir,
    manifest,
    split,
    bundle,
    targets,
):
    states = {
        m: new_state()
        for m in [
            *MODELS,
            "delta_latent_oracle",
        ]
    }

    files = files_from_manifest(
        manifest,
        split,
    )

    for i, rel in enumerate(files, 1):
        p = model_dir / rel

        print(
            f"[{split} {i}/{len(files)}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(p)

        if d.empty:
            continue

        true_q, true_p, true_qa, true_qs = true_curve(
            d
        )

        prev = previous_vector(d)

        for model in MODELS:
            z = latent_matrix(
                d,
                f"pred_{model}_",
                targets,
            )

            delta = decode_delta(
                z,
                bundle,
            )

            pred_v = prev + delta

            update_state(
                states[model],
                true_q,
                true_p,
                true_qa,
                true_qs,
                pred_v,
            )

        true_z = latent_matrix(
            d,
            "true_",
            targets,
        )

        oracle_delta = decode_delta(
            true_z,
            bundle,
        )

        update_state(
            states[
                "delta_latent_oracle"
            ],
            true_q,
            true_p,
            true_qa,
            true_qs,
            prev + oracle_delta,
        )

        del d
        gc.collect()

    return pd.DataFrame(
        [
            finalize(
                st,
                split,
                model,
            )
            for model, st in states.items()
        ]
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--dataset-dir",
        default="macro_b_curve_change_latent_dataset",
    )
    ap.add_argument(
        "--model-dir",
        default="macro_b_curve_change_regression_models",
    )
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    dataset = base / args.dataset_dir
    model_dir = base / args.model_dir

    bundle = joblib.load(
        dataset
        / "delta_pca_bundle.joblib"
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
        manifest["delta_columns"]
    )

    out_dir = (
        base
        / "macro_b_curve_change_evaluation"
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
        f"07d Macro-B curve forecast evaluation - "
        f"{args.year}"
    )
    print("=" * 80)

    val = evaluate_split(
        model_dir,
        manifest,
        "val",
        bundle,
        targets,
    )

    candidates = (
        val.loc[
            val["model"].isin(
                MODELS
            )
        ]
        .sort_values(
            [
                "price_wape_pct",
                "price_mae",
            ]
        )
        .reset_index(drop=True)
    )

    if candidates.empty:
        raise ValueError(
            "No validation candidates."
        )

    selected = str(
        candidates.iloc[0]["model"]
    )

    val.to_csv(
        out_dir
        / "validation_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selection = {
        "selection_split": "validation",
        "selection_metric": (
            "reconstructed full-curve price WAPE"
        ),
        "selected_model": selected,
        "candidate_models": MODELS,
        "test_used_for_selection": False,
        "validation_selected_wape_pct": float(
            candidates.iloc[0][
                "price_wape_pct"
            ]
        ),
    }

    (
        out_dir
        / "selected_model.json"
    ).write_text(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    test = evaluate_split(
        model_dir,
        manifest,
        "test",
        bundle,
        targets,
    )

    test[
        "selected_by_validation"
    ] = test[
        "model"
    ].eq(selected)

    test.to_csv(
        out_dir
        / "test_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def wape(table, model):
        x = table.loc[
            table["model"].eq(model),
            "price_wape_pct",
        ]
        return (
            float(x.iloc[0])
            if len(x)
            else np.nan
        )

    val_zero = wape(
        val,
        "zero_change",
    )
    val_selected = wape(
        val,
        selected,
    )
    test_zero = wape(
        test,
        "zero_change",
    )
    test_selected = wape(
        test,
        selected,
    )

    improvement = (
        100.0
        * (
            test_zero
            - test_selected
        )
        / test_zero
        if (
            np.isfinite(test_zero)
            and test_zero != 0
        )
        else np.nan
    )

    summary = "\n".join(
        [
            (
                f"07d Macro-B curve forecast evaluation - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                "This is an oracle Macro-B screening experiment. "
                "Current and previous curves were both filtered as Macro-B."
            ),
            "",
            (
                f"Selected model from VALIDATION = "
                f"{selected}"
            ),
            "",
            "VALIDATION:",
            val.to_string(
                index=False
            ),
            "",
            "TEST:",
            test.to_string(
                index=False
            ),
            "",
            (
                f"VAL zero-change WAPE = "
                f"{val_zero:.6f}%"
            ),
            (
                f"VAL selected-model WAPE = "
                f"{val_selected:.6f}%"
            ),
            (
                f"TEST zero-change WAPE = "
                f"{test_zero:.6f}%"
            ),
            (
                f"TEST selected-model WAPE = "
                f"{test_selected:.6f}%"
            ),
            (
                "TEST relative improvement vs Macro-B persistence = "
                f"{improvement:.4f}%"
            ),
            "",
            (
                "Interpretation: a learned model beating zero_change on this "
                "homogeneous subset supports the hypothesis that pooled curve-"
                "family heterogeneity was suppressing regression performance."
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
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
