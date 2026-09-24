#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03e_diagnose_switch_threshold.py

Diagnose whether the current final-template predictor fails because of an
over-conservative switch threshold or because P(switch) itself loses
out-of-time discrimination.

No model is trained.

For each split (val/test), this script reports:
- true / predicted switch share
- switch Precision / Recall / F1
- switch Balanced Accuracy
- overall final-template accuracy
- exact current-template accuracy on true-switch rows
- destination accuracy conditional on detected true switches
- P(switch) quantiles for true-switch vs unchanged rows
- ROC-AUC and PR-AUC for switch probability
- threshold sweep

The destination model is held fixed. For a threshold tau:

    if P(switch) < tau:
        final_template = historical_template
    else:
        final_template = predicted_destination

Run
---
python scripts/bidprediction/03e_diagnose_switch_threshold.py --year 2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


TEMPLATES = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
T2I = {t: i for i, t in enumerate(TEMPLATES)}
I2T = np.asarray(TEMPLATES, dtype=object)
N_TEMPLATE = len(TEMPLATES)

MODE2I = {
    "flat": 0,
    "block": 1,
    "sloped": 2,
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def norm(s):
    return s.astype("string").str.strip()


def template_history_ready(d):
    if "hist_lag1_template_id" not in d.columns:
        return pd.Series(False, index=d.index)

    return norm(
        d["hist_lag1_template_id"]
    ).isin(TEMPLATES)


def template_encode_col(s, col):
    if col in {
        "hist_lag1_template_id",
        "hist30_dominant_template_id",
    }:
        return (
            norm(s)
            .map(T2I)
            .astype("float32")
        )

    if col == "hist_lag1_curve_mode":
        return (
            norm(s)
            .str.lower()
            .map(MODE2I)
            .astype("float32")
        )

    return num(s).astype("float32")


def expand_proba(model, raw_p):
    classes = np.asarray(
        model.classes_,
        dtype=np.int16,
    )

    full = np.zeros(
        (len(raw_p), N_TEMPLATE),
        dtype=float,
    )

    full[:, classes] = raw_p
    return full


def mask_destination_proba(
    p,
    origin,
    allowed,
):
    p = np.asarray(
        p,
        dtype=float,
    ).copy()

    for i in range(N_TEMPLATE):
        rows = np.where(
            origin == i
        )[0]

        if not len(rows):
            continue

        mask = allowed[i].copy()

        if not mask.any():
            mask[:] = True
            mask[i] = False

        p[
            np.ix_(
                rows,
                ~mask,
            )
        ] = 0.0

    den = p.sum(
        axis=1,
        keepdims=True,
    )

    bad = den[:, 0] <= 0

    if bad.any():
        for r in np.where(bad)[0]:
            p[r, :] = 1.0
            p[r, origin[r]] = 0.0

        den = p.sum(
            axis=1,
            keepdims=True,
        )

    p /= den
    return p


def predict_switch_and_destination(
    d,
    bundle,
):
    origin_s = (
        norm(
            d["hist_lag1_template_id"]
        )
        .map(T2I)
    )

    if origin_s.isna().any():
        raise ValueError(
            "Invalid hist_lag1_template_id after filtering."
        )

    origin = origin_s.to_numpy(
        np.int16
    )

    # -------------------------------------------------------------
    # Switch probability
    # -------------------------------------------------------------
    switch_features = bundle[
        "switch_features"
    ]

    Xs = pd.DataFrame(
        {
            c: template_encode_col(
                d[c],
                c,
            )
            for c in switch_features
        },
        index=d.index,
    )

    Xs = bundle[
        "switch_imputer"
    ].transform(
        Xs
    ).astype(
        np.float32
    )

    switch_model = bundle[
        "switch_model"
    ]

    raw_sw = switch_model.predict_proba(
        Xs
    )

    classes = list(
        switch_model.classes_
    )

    p_switch = (
        raw_sw[
            :,
            classes.index(1),
        ]
        if 1 in classes
        else np.zeros(
            len(d),
            dtype=float,
        )
    )

    # -------------------------------------------------------------
    # Destination probability
    # -------------------------------------------------------------
    destination_features = bundle[
        "destination_features"
    ]

    Xd = pd.DataFrame(
        {
            c: template_encode_col(
                d[c],
                c,
            )
            for c in destination_features
        },
        index=d.index,
    )

    Xd = bundle[
        "destination_imputer"
    ].transform(
        Xd
    ).astype(
        np.float32
    )

    Xdg = np.column_stack(
        [
            Xd,
            origin.astype(
                np.float32
            ),
        ]
    )

    destination_model = bundle[
        "destination_model"
    ]

    dest = expand_proba(
        destination_model,
        destination_model.predict_proba(
            Xdg
        ),
    )

    dest = mask_destination_proba(
        dest,
        origin,
        np.asarray(
            bundle[
                "allowed_destinations_train"
            ],
            dtype=bool,
        ),
    )

    destination = np.argmax(
        dest,
        axis=1,
    ).astype(
        np.int16
    )

    return {
        "origin": origin,
        "p_switch": p_switch.astype(
            np.float32
        ),
        "destination": destination,
        "destination_probability": dest.astype(
            np.float32
        ),
    }


def collect_split(
    frozen,
    manifest,
    split,
    bundle,
):
    origin_blocks = []
    true_template_blocks = []
    p_switch_blocks = []
    destination_blocks = []

    dropped_invalid_hist_template = 0

    parts = manifest[
        "parts"
    ][
        split
    ]

    for i, rel in enumerate(
        parts,
        1,
    ):
        path = frozen / rel

        print(
            f"[{split} {i}/{len(parts)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        ready = template_history_ready(
            d
        )

        dropped_invalid_hist_template += int(
            (~ready).sum()
        )

        d = (
            d.loc[
                ready
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if d.empty:
            continue

        pred = predict_switch_and_destination(
            d,
            bundle,
        )

        true_s = (
            norm(
                d[
                    "y_template_id"
                ]
            )
            .map(T2I)
        )

        valid_true = true_s.notna()

        if not valid_true.all():
            d = (
                d.loc[
                    valid_true
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            keep = valid_true.to_numpy()

            for k in list(
                pred.keys()
            ):
                pred[k] = pred[k][
                    keep
                ]

            true_s = true_s.loc[
                valid_true
            ]

        origin_blocks.append(
            pred[
                "origin"
            ]
        )

        true_template_blocks.append(
            true_s.to_numpy(
                np.int16
            )
        )

        p_switch_blocks.append(
            pred[
                "p_switch"
            ]
        )

        destination_blocks.append(
            pred[
                "destination"
            ]
        )

    if not origin_blocks:
        raise ValueError(
            f"No usable rows for split={split}."
        )

    return {
        "origin": np.concatenate(
            origin_blocks
        ),
        "true_template": np.concatenate(
            true_template_blocks
        ),
        "p_switch": np.concatenate(
            p_switch_blocks
        ),
        "destination": np.concatenate(
            destination_blocks
        ),
        "dropped_invalid_hist_template_rows": int(
            dropped_invalid_hist_template
        ),
    }


def safe_div(
    a,
    b,
):
    return float(
        a / b
    ) if b else np.nan


def threshold_metrics(
    data,
    threshold,
):
    origin = data[
        "origin"
    ]

    true_template = data[
        "true_template"
    ]

    p_switch = data[
        "p_switch"
    ]

    destination = data[
        "destination"
    ]

    true_switch = (
        true_template
        != origin
    )

    pred_switch = (
        p_switch
        >= threshold
    )

    final_template = origin.copy()

    final_template[
        pred_switch
    ] = destination[
        pred_switch
    ]

    tp = int(
        np.sum(
            true_switch
            & pred_switch
        )
    )

    fp = int(
        np.sum(
            (~true_switch)
            & pred_switch
        )
    )

    fn = int(
        np.sum(
            true_switch
            & (~pred_switch)
        )
    )

    tn = int(
        np.sum(
            (~true_switch)
            & (~pred_switch)
        )
    )

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            np.isfinite(
                precision
            )
            and np.isfinite(
                recall
            )
            and (
                precision
                + recall
            ) > 0
        )
        else np.nan
    )

    # Balanced accuracy computed directly for transparency.
    tpr = safe_div(
        tp,
        tp + fn,
    )

    tnr = safe_div(
        tn,
        tn + fp,
    )

    balanced_acc = (
        0.5
        * (
            tpr
            + tnr
        )
        if (
            np.isfinite(
                tpr
            )
            and np.isfinite(
                tnr
            )
        )
        else np.nan
    )

    exact_template_accuracy = float(
        np.mean(
            final_template
            == true_template
        )
    )

    true_switch_rows = int(
        true_switch.sum()
    )

    exact_true_switch = int(
        np.sum(
            true_switch
            & (
                final_template
                == true_template
            )
        )
    )

    detected_true_switch = (
        true_switch
        & pred_switch
    )

    destination_correct_detected = int(
        np.sum(
            detected_true_switch
            & (
                destination
                == true_template
            )
        )
    )

    return {
        "threshold": float(
            threshold
        ),
        "rows": int(
            len(origin)
        ),
        "true_switch_share": float(
            true_switch.mean()
        ),
        "pred_switch_share": float(
            pred_switch.mean()
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "switch_precision": precision,
        "switch_recall": recall,
        "switch_f1": f1,
        "switch_balanced_accuracy": balanced_acc,
        "final_template_accuracy": exact_template_accuracy,
        "exact_template_accuracy_on_true_switch": safe_div(
            exact_true_switch,
            true_switch_rows,
        ),
        "destination_accuracy_when_switch_detected": safe_div(
            destination_correct_detected,
            tp,
        ),
    }


def probability_summary(
    data,
):
    origin = data[
        "origin"
    ]

    true_template = data[
        "true_template"
    ]

    p_switch = data[
        "p_switch"
    ].astype(float)

    true_switch = (
        true_template
        != origin
    )

    rows = []

    quantiles = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    for flag, label in [
        (False, "unchanged"),
        (True, "switched"),
    ]:
        x = p_switch[
            true_switch
            == flag
        ]

        if not len(
            x
        ):
            continue

        row = {
            "true_state": label,
            "rows": int(
                len(
                    x
                )
            ),
            "mean": float(
                np.mean(
                    x
                )
            ),
            "std": float(
                np.std(
                    x
                )
            ),
        }

        for q in quantiles:
            row[
                f"q{int(round(q * 100)):02d}"
            ] = float(
                np.quantile(
                    x,
                    q,
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def discrimination_metrics(
    data,
):
    origin = data[
        "origin"
    ]

    true_template = data[
        "true_template"
    ]

    y = (
        true_template
        != origin
    ).astype(
        np.int8
    )

    p = data[
        "p_switch"
    ].astype(float)

    out = {
        "rows": int(
            len(y)
        ),
        "true_switch_share": float(
            y.mean()
        ),
    }

    if len(
        np.unique(
            y
        )
    ) == 2:
        out[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y,
                p,
            )
        )

        out[
            "pr_auc"
        ] = float(
            average_precision_score(
                y,
                p,
            )
        )
    else:
        out[
            "roc_auc"
        ] = np.nan

        out[
            "pr_auc"
        ] = np.nan

    return out


def parse_thresholds(
    spec,
    current_threshold,
):
    if spec:
        thresholds = [
            float(
                x
            )
            for x in spec.split(",")
            if x.strip()
        ]
    else:
        thresholds = list(
            np.round(
                np.arange(
                    0.01,
                    1.00,
                    0.01,
                ),
                2,
            )
        )

    thresholds.extend(
        [
            0.0,
            1.0,
            float(
                current_threshold
            ),
        ]
    )

    thresholds = sorted(
        set(
            round(
                float(
                    x
                ),
                6,
            )
            for x in thresholds
            if 0.0 <= float(
                x
            ) <= 1.0
        )
    )

    return thresholds


def summarize_best(
    sweep,
    current_threshold,
):
    def pick(
        col,
        ascending=False,
    ):
        valid = sweep.loc[
            sweep[
                col
            ].notna()
        ]

        if valid.empty:
            return {}

        row = (
            valid.sort_values(
                [
                    col,
                    "final_template_accuracy",
                    "threshold",
                ],
                ascending=[
                    ascending,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        return row.to_dict()

    current_idx = np.argmin(
        np.abs(
            sweep[
                "threshold"
            ].to_numpy(
                float
            )
            - float(
                current_threshold
            )
        )
    )

    current = sweep.iloc[
        current_idx
    ].to_dict()

    return {
        "current_threshold_row": current,
        "best_switch_f1_row": pick(
            "switch_f1",
            ascending=False,
        ),
        "best_balanced_accuracy_row": pick(
            "switch_balanced_accuracy",
            ascending=False,
        ),
        "best_final_template_accuracy_row": pick(
            "final_template_accuracy",
            ascending=False,
        ),
        "best_true_switch_exact_accuracy_row": pick(
            "exact_template_accuracy_on_true_switch",
            ascending=False,
        ),
    }


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
        default="frozen_stable_theta_continuous_dataset",
    )

    ap.add_argument(
        "--template-model-dir",
        default="final_template_predictor",
    )

    ap.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Comma-separated threshold list. "
            "Default scans 0.01..0.99 by 0.01, "
            "plus 0, 1 and the current hierarchy threshold."
        ),
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

    frozen = (
        base
        / args.dataset_dir
    )

    manifest_file = (
        frozen
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

    model_file = (
        base
        / args.template_model_dir
        / "final_template_model.joblib"
    )

    if not model_file.exists():
        raise FileNotFoundError(
            model_file
        )

    bundle = joblib.load(
        model_file
    )

    current_threshold = float(
        bundle[
            "hierarchy_threshold"
        ]
    )

    thresholds = parse_thresholds(
        args.thresholds,
        current_threshold,
    )

    out = (
        base
        / "switch_threshold_diagnostics"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"Switch-threshold diagnostics - {args.year}"
    )
    print("=" * 80)
    print(
        f"Current hierarchy threshold = "
        f"{current_threshold:.6f}"
    )
    print(
        f"Thresholds tested = {len(thresholds)}"
    )
    print()

    all_summary = {
        "year": int(
            args.year
        ),
        "current_hierarchy_threshold": current_threshold,
        "splits": {},
    }

    for split in [
        "val",
        "test",
    ]:
        print()
        print(
            f"Collecting {split} predictions..."
        )

        data = collect_split(
            frozen,
            manifest,
            split,
            bundle,
        )

        disc = discrimination_metrics(
            data
        )

        prob = probability_summary(
            data
        )

        sweep = pd.DataFrame(
            [
                threshold_metrics(
                    data,
                    tau,
                )
                for tau in thresholds
            ]
        )

        best = summarize_best(
            sweep,
            current_threshold,
        )

        split_dir = (
            out
            / split
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        sweep.to_csv(
            split_dir
            / "threshold_sweep.csv",
            index=False,
            encoding="utf-8-sig",
        )

        prob.to_csv(
            split_dir
            / "switch_probability_distribution.csv",
            index=False,
            encoding="utf-8-sig",
        )

        split_summary = {
            "dropped_invalid_hist_template_rows": int(
                data[
                    "dropped_invalid_hist_template_rows"
                ]
            ),
            **disc,
            **best,
        }

        (
            split_dir
            / "diagnostics.json"
        ).write_text(
            json.dumps(
                split_summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        all_summary[
            "splits"
        ][
            split
        ] = split_summary

        print()
        print(
            f"{split.upper()} switch discrimination:"
        )
        print(
            json.dumps(
                disc,
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print(
            f"{split.upper()} P(switch) distribution:"
        )
        print(
            prob.to_string(
                index=False
            )
        )

        print()
        print(
            f"{split.upper()} current threshold:"
        )

        current_row = pd.DataFrame(
            [
                best[
                    "current_threshold_row"
                ]
            ]
        )

        print(
            current_row.to_string(
                index=False
            )
        )

        print()
        print(
            f"{split.upper()} best by switch F1:"
        )

        print(
            pd.DataFrame(
                [
                    best[
                        "best_switch_f1_row"
                    ]
                ]
            ).to_string(
                index=False
            )
        )

        print()
        print(
            f"{split.upper()} best by balanced accuracy:"
        )

        print(
            pd.DataFrame(
                [
                    best[
                        "best_balanced_accuracy_row"
                    ]
                ]
            ).to_string(
                index=False
            )
        )

        print()
        print(
            f"{split.upper()} best by final template accuracy:"
        )

        print(
            pd.DataFrame(
                [
                    best[
                        "best_final_template_accuracy_row"
                    ]
                ]
            ).to_string(
                index=False
            )
        )

    (
        out
        / "summary.json"
    ).write_text(
        json.dumps(
            all_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Compact cross-split text summary.
    lines = [
        (
            f"Switch-threshold diagnostics - {args.year}"
        ),
        "=" * 80,
        "",
        (
            f"Current hierarchy threshold = "
            f"{current_threshold:.6f}"
        ),
    ]

    for split in [
        "val",
        "test",
    ]:
        s = all_summary[
            "splits"
        ][
            split
        ]

        current = s[
            "current_threshold_row"
        ]

        best_f1 = s[
            "best_switch_f1_row"
        ]

        best_bacc = s[
            "best_balanced_accuracy_row"
        ]

        best_acc = s[
            "best_final_template_accuracy_row"
        ]

        lines.extend(
            [
                "",
                f"[{split.upper()}]",
                (
                    f"ROC-AUC = "
                    f"{s['roc_auc']:.6f}"
                ),
                (
                    f"PR-AUC = "
                    f"{s['pr_auc']:.6f}"
                ),
                (
                    f"True switch share = "
                    f"{s['true_switch_share']:.6f}"
                ),
                "",
                (
                    "Current threshold: "
                    f"tau={current['threshold']:.3f}, "
                    f"pred_share={current['pred_switch_share']:.6f}, "
                    f"P={current['switch_precision']:.6f}, "
                    f"R={current['switch_recall']:.6f}, "
                    f"F1={current['switch_f1']:.6f}, "
                    f"BalAcc={current['switch_balanced_accuracy']:.6f}, "
                    f"TemplateAcc={current['final_template_accuracy']:.6f}"
                ),
                (
                    "Best F1: "
                    f"tau={best_f1['threshold']:.3f}, "
                    f"P={best_f1['switch_precision']:.6f}, "
                    f"R={best_f1['switch_recall']:.6f}, "
                    f"F1={best_f1['switch_f1']:.6f}, "
                    f"TemplateAcc={best_f1['final_template_accuracy']:.6f}"
                ),
                (
                    "Best balanced accuracy: "
                    f"tau={best_bacc['threshold']:.3f}, "
                    f"P={best_bacc['switch_precision']:.6f}, "
                    f"R={best_bacc['switch_recall']:.6f}, "
                    f"BalAcc={best_bacc['switch_balanced_accuracy']:.6f}, "
                    f"TemplateAcc={best_bacc['final_template_accuracy']:.6f}"
                ),
                (
                    "Best final template accuracy: "
                    f"tau={best_acc['threshold']:.3f}, "
                    f"P={best_acc['switch_precision']:.6f}, "
                    f"R={best_acc['switch_recall']:.6f}, "
                    f"F1={best_acc['switch_f1']:.6f}, "
                    f"TemplateAcc={best_acc['final_template_accuracy']:.6f}"
                ),
            ]
        )

    (
        out
        / "summary.txt"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))
    print()
    print(
        f"Outputs: {out}"
    )


if __name__ == "__main__":
    main()
