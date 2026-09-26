#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10a_build_final_feature_only_dataset.py

Final Stage-3 dataset freeze for the full eligible bid-curve population.

FINAL TASK DEFINITION
---------------------
Inputs:
    strategy profile + strategy transition + market + unit + calendar/slot

Explicitly forbidden as model inputs:
    previous raw bid curve
    historical curve latent
    DeltaCurve / DeltaZ history
    historical theta
    template ID / template inertia
    any current bid-curve target field

Target curve is preserved only for 10b/10d.
Previous raw curve is preserved only under "reference_*" names for the
information-rich Persistence reference in 10d.

No Macro-B / Prediction-Family / template filtering is applied.

Default source
--------------
data/processed/bidprediction/<year>/latent_forecasting_dataset

This source already contains the frozen temporal split, current complete curve,
base context features and the previous same-slot raw curve. 10a deliberately
drops all historical latent features and uses the source manifest
"base_features" directly, excluding only hist_days_since_prev_same_slot.
The source base_features already contain the frozen calendar/slot variables.

Output
------
data/processed/bidprediction/<year>/final_feature_only_dataset/
    parts/{train,val,test}/*.pkl
    manifest.json
    model_features.csv
    summary.txt

Run
---
python scripts/bidprediction/10a_build_final_feature_only_dataset.py \
    --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import re
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
CURVE_COLS = [
    *SHAPE_COLS,
    "p_anchor",
    "p_span",
    "q_anchor_mw",
    "q_span_mw",
]
META_CANDIDATES = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
    "prediction_cutoff_utc",
    "y_template_id",
]

SOURCE_PREV_PRICE = [
    f"_curvevec_p{i:02d}_lag1"
    for i in range(21)
]
SOURCE_PREV_RAW = [
    *SOURCE_PREV_PRICE,
    "_curvevec_q_anchor_lag1",
    "_curvevec_log_q_span_lag1",
]

# 05b base_features already contain calendar/slot variables.
# This single history-recency field was added specifically for the
# latent-history forecasting route and is excluded from the final
# feature-only route, restoring the frozen 83-input definition.
FINAL_BASE_EXCLUDE = {
    "hist_days_since_prev_same_slot",
}

FORBIDDEN_INPUT_TOKENS = (
    "latent",
    "curvevec",
    "theta",
    "template",
    "shape_v",
    "p_anchor",
    "p_span",
    "q_anchor",
    "q_span",
)


def num(s):
    return pd.to_numeric(
        s,
        errors="coerce",
    )


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


def leakage_check(features):
    """
    Precise final-input leakage guard.

    Allowed examples:
      lt_shape_variability
      tr_template_entropy_7d
      tr_template_entropy_30d
      tr_unique_template_count_7d
      tr_unique_template_count_30d
      tr_lag1_daily_template_entropy
      tr_lag1_daily_unique_template_count

    These are aggregate strategy-profile / strategy-transition features and
    belong to the frozen feature set.

    Forbidden:
      raw current curve targets (shape_v00..20, p_anchor, p_span,
      q_anchor_mw, q_span_mw);
      raw historical curve vectors (curvevec);
      curve latent/history targets (latent);
      theta targets/history;
      raw/current/lagged template IDs (names ending in template_id);
      explicit template inertia fields.
    """
    exact_forbidden = {
        "p_anchor",
        "p_span",
        "q_anchor_mw",
        "q_span_mw",
        "y_template_id",
        "template_id",
    }

    bad = []

    for f in features:
        low = str(f).lower()

        is_raw_shape_target = bool(
            re.fullmatch(r"shape_v\d{2}", low)
        )

        is_template_id = (
            low.endswith("template_id")
            or low.endswith("_template_id")
        )

        is_direct_history_or_target = any(
            token in low
            for token in (
                "curvevec",
                "latent",
                "theta",
            )
        )

        is_template_inertia = (
            "template_inertia" in low
        )

        if (
            low in exact_forbidden
            or is_raw_shape_target
            or is_template_id
            or is_direct_history_or_target
            or is_template_inertia
        ):
            bad.append(f)

    if bad:
        raise RuntimeError(
            "Forbidden direct bid-history/target information entered "
            "final model_features: "
            + ", ".join(
                bad[:40]
            )
        )


def curve_valid_mask(d):
    missing = [
        c
        for c in CURVE_COLS
        if c not in d.columns
    ]

    if missing:
        raise KeyError(
            "Source part misses current curve fields: "
            + ", ".join(
                missing
            )
        )

    core = d[
        [
            "p_anchor",
            "p_span",
            "q_anchor_mw",
            "q_span_mw",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite_core = (
        core.notna()
        .all(
            axis=1
        )
    )

    positive_q_span = (
        num(
            d[
                "q_span_mw"
            ]
        )
        > 0.0
    )

    p_span = num(
        d[
            "p_span"
        ]
    )

    flat = (
        p_span.abs()
        <= 1e-12
    )

    shape_complete = (
        d[
            SHAPE_COLS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .notna()
        .all(
            axis=1
        )
    )

    return (
        finite_core
        & positive_q_span
        & (
            flat
            | shape_complete
        )
    )


def reference_name(source_col):
    return (
        "reference"
        + source_col
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
        default="latent_forecasting_dataset",
    )
    ap.add_argument(
        "--expected-feature-count",
        type=int,
        default=83,
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

    manifest_file = (
        source
        / "manifest.json"
    )

    if not manifest_file.exists():
        raise FileNotFoundError(
            manifest_file
        )

    source_manifest = json.loads(
        manifest_file.read_text(
            encoding="utf-8"
        )
    )

    source_base_features = source_manifest.get(
        "base_features"
    )

    if not isinstance(
        source_base_features,
        list,
    ) or not source_base_features:
        raise KeyError(
            "Source manifest must contain non-empty 'base_features'. "
            "10a must not fall back to the history-enriched model_features."
        )

    # Recover the frozen 08 feature-only input definition exactly:
    # 05b base_features already include profile + transition + market +
    # unit + calendar. The only extra 05b context feature is
    # hist_days_since_prev_same_slot, which belongs to history-recency
    # support for the latent-history route and is excluded here.
    base_features = [
        c
        for c in source_base_features
        if c not in FINAL_BASE_EXCLUDE
    ]

    excluded_base_features = [
        c
        for c in source_base_features
        if c in FINAL_BASE_EXCLUDE
    ]

    leakage_check(
        base_features
    )

    out = (
        base
        / "final_feature_only_dataset"
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

    output_parts = {}
    split_stats = {}
    final_features = None

    print("=" * 80)
    print(
        f"10a Final full feature-only dataset - {args.year}"
    )
    print("=" * 80)
    print(
        f"Source = {source}"
    )
    print(
        "No Macro-B / family / template filtering."
    )
    print(
        f"Source base features = {len(source_base_features)}"
    )
    print(
        f"Excluded history-recency features = {excluded_base_features}"
    )
    print(
        f"Frozen final feature-only inputs = {len(base_features)}"
    )
    print()

    for split in [
        "train",
        "val",
        "test",
    ]:
        files = part_files(
            source_manifest,
            split,
        )

        if not files:
            raise KeyError(
                f"No source files for split={split}"
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

        source_rows = 0
        valid_rows = 0
        invalid_rows = 0
        persistence_rows = 0
        template_rows = 0

        for i, rel in enumerate(
            files,
            1,
        ):
            src = (
                source
                / rel
            )

            print(
                f"[{split} {i}/{len(files)}] {src.name}",
                flush=True,
            )

            d = pd.read_pickle(
                src
            )

            source_rows += len(
                d
            )

            # Source base_features already contain the upstream calendar/slot
            # features. Do NOT add another calendar encoding here.
            model_features = list(
                base_features
            )

            leakage_check(
                model_features
            )

            if final_features is None:
                final_features = model_features

                if (
                    args.expected_feature_count
                    is not None
                    and len(
                        final_features
                    )
                    != args.expected_feature_count
                ):
                    raise RuntimeError(
                        "Final feature count drifted from the frozen route: "
                        f"expected {args.expected_feature_count}, "
                        f"got {len(final_features)}. "
                        "Inspect the source manifest instead of silently "
                        "changing the final method."
                    )

            elif model_features != final_features:
                raise RuntimeError(
                    "Final feature list changed between source parts."
                )

            missing_features = [
                c
                for c in final_features
                if c not in d.columns
            ]

            if missing_features:
                raise KeyError(
                    f"{src.name}: missing final input features: "
                    + ", ".join(
                        missing_features[:30]
                    )
                )

            valid = curve_valid_mask(
                d
            )

            invalid_rows += int(
                (
                    ~valid
                ).sum()
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

            if d.empty:
                continue

            valid_rows += len(
                d
            )

            keep = list(
                dict.fromkeys(
                    [
                        *[
                            c
                            for c
                            in META_CANDIDATES
                            if c
                            in d.columns
                        ],
                        *final_features,
                        *CURVE_COLS,
                    ]
                )
            )

            out_d = d[
                keep
            ].copy()

            reference_cols_written = []

            for c in SOURCE_PREV_RAW:
                if c in d.columns:
                    dst = reference_name(
                        c
                    )

                    out_d[
                        dst
                    ] = d[
                        c
                    ].to_numpy()

                    reference_cols_written.append(
                        dst
                    )

            if len(
                reference_cols_written
            ) == len(
                SOURCE_PREV_RAW
            ):
                ref_ready = (
                    out_d[
                        reference_cols_written
                    ]
                    .apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    .notna()
                    .all(
                        axis=1
                    )
                )

                persistence_rows += int(
                    ref_ready.sum()
                )

            if "y_template_id" in out_d.columns:
                template_rows += int(
                    out_d[
                        "y_template_id"
                    ]
                    .notna()
                    .sum()
                )

            name = (
                f"{split}_final_feature_only_"
                f"{i:04d}.pkl"
            )

            path = (
                split_dir
                / name
            )

            out_d.to_pickle(
                path,
                protocol=5,
            )

            written.append(
                {
                    "file": str(
                        path.relative_to(
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

            del d, out_d
            gc.collect()

        output_parts[
            split
        ] = written

        split_stats[
            split
        ] = {
            "source_rows": int(
                source_rows
            ),
            "curve_valid_rows": int(
                valid_rows
            ),
            "curve_invalid_rows": int(
                invalid_rows
            ),
            "curve_valid_share": float(
                valid_rows
                / max(
                    source_rows,
                    1,
                )
            ),
            "persistence_reference_ready_rows": int(
                persistence_rows
            ),
            "persistence_reference_ready_share": float(
                persistence_rows
                / max(
                    valid_rows,
                    1,
                )
            ),
            "template_label_rows": int(
                template_rows
            ),
        }

    leakage_check(
        final_features
    )

    feature_table = pd.DataFrame(
        {
            "feature": final_features,
            "role": "feature",
            "leakage_use": "model_input",
        }
    )

    feature_table.to_csv(
        out
        / "model_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reference_columns = [
        reference_name(
            c
        )
        for c in SOURCE_PREV_RAW
    ]

    out_manifest = {
        "version": "final-feature-only-full-population-v1",
        "year": int(
            args.year
        ),
        "source_dataset": str(
            source
        ),
        "population": (
            "all curve-valid source rows; no Macro-B, template or "
            "prediction-family filtering"
        ),
        "task": (
            "feature-only absolute bid-curve prediction without direct "
            "historical bid-curve input"
        ),
        "source_base_features": source_base_features,
        "excluded_source_base_features": excluded_base_features,
        "feature_definition": (
            "05b base_features minus hist_days_since_prev_same_slot; "
            "source base_features already include calendar/slot features"
        ),
        "model_features": final_features,
        "model_feature_count": int(
            len(
                final_features
            )
        ),
        "curve_target_columns": CURVE_COLS,
        "reference_only_columns": reference_columns,
        "reference_only_definition": (
            "previous same-slot raw bid curve; evaluation-only Persistence "
            "reference; never included in model_features"
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

    summary = "\n".join(
        [
            (
                f"10a Final full feature-only dataset - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                "Population = all curve-valid source rows "
                "(no Macro-B / template / family filtering)"
            ),
            (
                f"Source base features = "
                f"{len(source_base_features)}"
            ),
            (
                "Excluded source feature = "
                + ", ".join(excluded_base_features)
            ),
            (
                f"Model features = "
                f"{len(final_features)}"
            ),
            (
                "Feature rule = source base_features already contain calendar; "
                "exclude only history-recency field hist_days_since_prev_same_slot"
            ),
            "Historical bid leakage check = PASS",
            "",
            "Forbidden from model input:",
            (
                "previous raw curve / historical latent / DeltaZ / theta / "
                "template / current curve target"
            ),
            "",
            "Split statistics:",
            json.dumps(
                split_stats,
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Final feature list:",
            "\n".join(
                final_features
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
