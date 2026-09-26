#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09e_summarize_prediction_family_experiment.py

Final diagnostic summary for 09a-09d.

No model training. Reads:
- 09a absolute prediction-family diagnostics
- 09b family classifier
- 09c oracle family experts
- 09d full hard-routing pipeline

Produces a compact decomposition:
1) classification quality;
2) oracle family-specific regression gain over global RF;
3) routing penalty from predicted family;
4) final deployable gain/loss versus global RF;
5) whether family-specific routing is worth retaining.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def one(df, **conds):
    x = df.copy()

    for col, val in conds.items():
        x = x.loc[
            x[col].eq(val)
        ]

    if x.empty:
        return None

    return x.iloc[0]


def val(row, key):
    if row is None:
        return np.nan

    return float(
        row[key]
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
        "--family-dir",
        default="absolute_prediction_family_diagnostics",
    )

    ap.add_argument(
        "--classifier-dir",
        default="prediction_family_classifier",
    )

    ap.add_argument(
        "--expert-dir",
        default="oracle_family_regressors",
    )

    ap.add_argument(
        "--full-dir",
        default="full_family_routing_evaluation",
    )

    ap.add_argument(
        "--k",
        type=int,
        default=5,
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

    fam_dir = (
        base
        / args.family_dir
    )

    cls_dir = (
        base
        / args.classifier_dir
    )

    exp_dir = (
        base
        / args.expert_dir
    )

    full_dir = (
        base
        / args.full_dir
    )

    out_dir = (
        base
        / "prediction_family_experiment_summary"
    )

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                out_dir
            )

        shutil.rmtree(
            out_dir
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    k_summary = pd.read_csv(
        fam_dir
        / "k_summary.csv"
    )

    cls = pd.read_csv(
        cls_dir
        / "classifier_metrics.csv"
    )

    oracle = pd.read_csv(
        exp_dir
        / "oracle_selected_expert_overall_metrics.csv"
    )

    full = pd.read_csv(
        full_dir
        / "full_pipeline_curve_metrics.csv"
    )

    conditions = pd.read_csv(
        full_dir
        / "routing_correctness_curve_metrics.csv"
    )

    cls_selection = json.loads(
        (
            cls_dir
            / "selected_classifier.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    expert_selection = json.loads(
        (
            exp_dir
            / "selected_experts.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    selected_classifier = cls_selection[
        "selected_model"
    ]

    rows = []

    for split in [
        "val",
        "test",
    ]:
        krow = one(
            k_summary,
            split=split,
            k=args.k,
        )

        crow = one(
            cls,
            split=split,
            model=selected_classifier,
        )

        orow = one(
            oracle,
            split=split,
        )

        grow = one(
            full,
            split=split,
            route="global_random_forest",
        )

        prow = one(
            full,
            split=split,
            route="predicted_family_experts",
        )

        frow = one(
            full,
            split=split,
            route="oracle_family_experts",
        )

        correct = one(
            conditions,
            split=split,
            routing_condition="correct_family",
        )

        wrong = one(
            conditions,
            split=split,
            routing_condition="wrong_family",
        )

        global_wape = val(
            grow,
            "price_wape_pct",
        )

        oracle_wape = val(
            frow,
            "price_wape_pct",
        )

        predicted_wape = val(
            prow,
            "price_wape_pct",
        )

        rows.append(
            {
                "split": split,
                "k": int(
                    args.k
                ),
                "selected_classifier": selected_classifier,
                "classifier_accuracy": val(
                    crow,
                    "accuracy",
                ),
                "classifier_balanced_accuracy": val(
                    crow,
                    "balanced_accuracy",
                ),
                "classifier_macro_f1": val(
                    crow,
                    "macro_f1",
                ),
                "oracle_family_mean_wape_pct": val(
                    krow,
                    "oracle_family_mean_price_wape_pct",
                ),
                "within_family_dispersion_ratio": val(
                    krow,
                    "within_family_dispersion_ratio_vs_global_train",
                ),
                "global_rf_wape_pct": global_wape,
                "oracle_family_expert_wape_pct": oracle_wape,
                "predicted_family_expert_wape_pct": predicted_wape,
                "oracle_expert_gain_vs_global_rf_pp": (
                    global_wape
                    - oracle_wape
                ),
                "routing_penalty_pp": (
                    predicted_wape
                    - oracle_wape
                ),
                "final_gain_vs_global_rf_pp": (
                    global_wape
                    - predicted_wape
                ),
                "correct_routing_wape_pct": val(
                    correct,
                    "price_wape_pct",
                ),
                "wrong_routing_wape_pct": val(
                    wrong,
                    "price_wape_pct",
                ),
                "oracle_selected_expert_wape_crosscheck": val(
                    orow,
                    "price_wape_pct",
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        out_dir
        / "experiment_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test = result.loc[
        result[
            "split"
        ].eq(
            "test"
        )
    ].iloc[
        0
    ]

    oracle_gain = float(
        test[
            "oracle_expert_gain_vs_global_rf_pp"
        ]
    )

    routing_penalty = float(
        test[
            "routing_penalty_pp"
        ]
    )

    final_gain = float(
        test[
            "final_gain_vs_global_rf_pp"
        ]
    )

    if final_gain > 0:
        conclusion = (
            "Predicted-family hard routing improves over the global RF on TEST."
        )
    elif oracle_gain > 0 and routing_penalty > 0:
        conclusion = (
            "Family-specific experts have oracle value, but family-classification "
            "errors erase that gain under hard routing."
        )
    elif oracle_gain <= 0:
        conclusion = (
            "Even oracle family-specific experts do not improve over the global RF; "
            "family-specific regression is not justified in its current form."
        )
    else:
        conclusion = (
            "The current hard-routing formulation does not improve the global RF."
        )

    summary = "\n".join(
        [
            (
                f"09e Prediction-Family experiment summary - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            f"K = {args.k}",
            (
                f"Selected classifier = "
                f"{selected_classifier}"
            ),
            (
                "Selected expert by family = "
                + json.dumps(
                    expert_selection[
                        "selected_expert_by_family"
                    ],
                    ensure_ascii=False,
                )
            ),
            "",
            "Error decomposition:",
            result.to_string(
                index=False
            ),
            "",
            "TEST interpretation:",
            (
                f"Oracle family-specific expert gain vs global RF = "
                f"{oracle_gain:.6f} percentage points"
            ),
            (
                f"Routing penalty = "
                f"{routing_penalty:.6f} percentage points"
            ),
            (
                f"Final predicted-family gain vs global RF = "
                f"{final_gain:.6f} percentage points"
            ),
            "",
            conclusion,
        ]
    )

    (
        out_dir
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print(
        summary
    )

    print()
    print(
        f"Outputs: {out_dir}"
    )


if __name__ == "__main__":
    main()
