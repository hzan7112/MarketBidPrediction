#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05b_build_latent_forecasting_dataset.py

NEW MAIN ROUTE - Step 05b
=========================

Build a leakage-free tabular forecasting dataset for future PCA latent bid
coordinates.

Target:
    current latent_z01 ... latent_zK

Predictors:
1) historical latent sequence summarized with interpretable lag/rolling features;
2) existing LT / ST / Break strategy profile;
3) transition-profile features;
4) market-environment features;
5) unit-state proxy features;
6) calendar features;
7) history-recency information.

No current template, current theta, current bid-curve point, or current latent
coordinate is used as a feature.

Historical latent features
--------------------------
For every latent dimension z_j and the same participant + local market slot:

    lag1
    lag2
    lag7
    mean7
    std7
    mean30
    std30
    delta1      = lag1 - lag2
    delta7      = lag1 - lag7
    mean_gap    = mean7 - mean30

All rolling statistics use SHIFT(1), so the current target never enters its own
feature vector.

A raw previous-curve vector is also stored ONLY for evaluation of a direct
curve-persistence baseline. Those columns are explicitly marked evaluation-only
and are never model features.

Temporal handling
-----------------
Parts are processed in:
    train -> validation -> test
order while carrying only past history forward. Chronological ordering uses
timestamp_utc when available, otherwise timestamp_local, otherwise local_date. Thus validation may use train
history, and test may use train/validation history, but no future row is used.

Outputs
-------
data/processed/bidprediction/<year>/latent_forecasting_dataset/
    parts/train/*.pkl
    parts/val/*.pkl
    parts/test/*.pkl
    feature_schema.csv
    manifest.json
    summary.txt

Run
---
python scripts/bidprediction/05b_build_latent_forecasting_dataset.py --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]

REQUIRED_META_COLS = [
    "participant_id",
    "local_date",
    "local_slot_seconds",
]

OPTIONAL_META_COLS = [
    "sample_id",
    "timestamp_utc",
    "timestamp_local",
    "prediction_cutoff_utc",
]

BASE_FEATURE_GROUPS = {
    "profile_LT",
    "profile_ST",
    "profile_Break",
    "transition_strategy_profile",
    "market_environment",
    "unit_state_proxy",
    "calendar",
}

EXCLUDE_BASE_FEATURES = {
    "rolling_lt_ready_flag",
    "st_ready_flag",
    "profile_ready_flag",
    "market_ready_flag",
    "unit_state_ready_flag",
    "prediction_ready_flag",
    "market_nonmissing_count",
    "rolling_lt_nonmissing_count",
    "hist_prev_available_flag",
    "tr_ready_flag",
}

CURVE_TARGET_COLS = [
    *SHAPE_COLS,
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


def latent_part_files(
    manifest,
    split,
):
    items = manifest[
        "parts"
    ][
        split
    ]

    files = []

    for item in items:
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


def available_meta_cols(d):
    """
    Return metadata columns actually present in the current source dataset.
    Only participant_id, local_date and local_slot_seconds are mandatory.
    """
    return [
        c
        for c in [
            *REQUIRED_META_COLS,
            *OPTIONAL_META_COLS,
        ]
        if c in d.columns
    ]


def resolve_time_order_column(d):
    """
    Choose the best available chronological key without assuming a specific
    upstream timestamp schema.

    Priority:
        timestamp_utc
        timestamp_local
        local_date

    Because history is grouped by participant_id + local_slot_seconds,
    local_date is sufficient as a chronological key when explicit timestamps
    are absent.
    """
    if "timestamp_utc" in d.columns:
        return "timestamp_utc"

    if "timestamp_local" in d.columns:
        return "timestamp_local"

    if "local_date" in d.columns:
        return "local_date"

    raise KeyError(
        "No usable chronological column found. "
        "Expected one of timestamp_utc, timestamp_local, local_date."
    )


def normalize_time_order(d, time_col):
    """
    Convert the selected chronological key to a sortable datetime.
    """
    if time_col == "local_date":
        return pd.to_datetime(
            d[time_col],
            errors="coerce",
        )

    return pd.to_datetime(
        d[time_col],
        errors="coerce",
        utc=(
            time_col
            == "timestamp_utc"
        ),
    )


def select_base_features(
    schema,
):
    s = schema.loc[
        schema[
            "role"
        ]
        .astype(str)
        .str.lower()
        .eq("feature")
    ].copy()

    out = []

    for row in s.itertuples(
        index=False
    ):
        col = str(
            row.column
        )
        group = str(
            row.feature_group
        )

        if (
            group
            in BASE_FEATURE_GROUPS
            and col
            not in EXCLUDE_BASE_FEATURES
        ):
            out.append(
                col
            )

    # History recency is useful and leakage-safe even though legacy
    # participant-history template/curve fields are deliberately excluded.
    available = set(
        s[
            "column"
        ].astype(str)
    )

    if (
        "hist_days_since_prev_same_slot"
        in available
    ):
        out.append(
            "hist_days_since_prev_same_slot"
        )

    return list(
        dict.fromkeys(
            out
        )
    )


def current_curve_vector(
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


def add_current_curve_vector(
    d,
):
    V = current_curve_vector(
        d
    )

    out = d.copy()

    for j in range(
        21
    ):
        out[
            f"_curvevec_p{j:02d}"
        ] = V[
            :,
            j,
        ].astype(
            np.float32
        )

    out[
        "_curvevec_q_anchor"
    ] = V[
        :,
        21,
    ].astype(
        np.float32
    )

    out[
        "_curvevec_log_q_span"
    ] = V[
        :,
        22,
    ].astype(
        np.float32
    )

    return out


def grouped_rolling(
    d,
    shifted,
    keys,
    window,
    stat,
):
    grouped = shifted.groupby(
        [
            d[
                k
            ]
            for k in keys
        ],
        sort=False,
    )

    roll = grouped.rolling(
        window=window,
        min_periods=2,
    )

    if stat == "mean":
        out = roll.mean()
    elif stat == "std":
        out = roll.std(
            ddof=0
        )
    else:
        raise ValueError(
            stat
        )

    return out.reset_index(
        level=list(
            range(
                len(
                    keys
                )
            )
        ),
        drop=True,
    )


def build_history_for_part(
    current,
    tail,
    latent_cols,
    max_history,
):
    keys = [
        "participant_id",
        "local_slot_seconds",
    ]

    current = current.copy()
    current[
        "_is_current"
    ] = 1

    time_col = resolve_time_order_column(
        current
    )

    current[
        "_history_time"
    ] = normalize_time_order(
        current,
        time_col,
    )

    if current[
        "_history_time"
    ].isna().any():
        raise ValueError(
            f"Unable to parse some {time_col} values "
            "for chronological history construction."
        )

    keep_state_cols = [
        *keys,
        "_history_time",
        *latent_cols,
        *[
            f"_curvevec_p{i:02d}"
            for i in range(
                21
            )
        ],
        "_curvevec_q_anchor",
        "_curvevec_log_q_span",
    ]

    if tail is None:
        combined = current.copy()
    else:
        hist = tail.copy()
        hist[
            "_is_current"
        ] = 0

        # Tail stores only generic history-state columns. Add any current-only
        # columns as NaN so concatenation is schema-safe.
        for c in current.columns:
            if c not in hist.columns:
                hist[
                    c
                ] = np.nan

        hist = hist[
            current.columns
        ]

        combined = pd.concat(
            [
                hist,
                current,
            ],
            ignore_index=True,
        )

    combined = (
        combined.sort_values(
            [
                *keys,
                "_history_time",
                "_is_current",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    g = combined.groupby(
        keys,
        sort=False,
    )

    history_feature_cols = []

    for z in latent_cols:
        lag1 = g[
            z
        ].shift(
            1
        )

        lag2 = g[
            z
        ].shift(
            2
        )

        lag7 = g[
            z
        ].shift(
            7
        )

        mean7 = grouped_rolling(
            combined,
            lag1,
            keys,
            7,
            "mean",
        )

        std7 = grouped_rolling(
            combined,
            lag1,
            keys,
            7,
            "std",
        )

        mean30 = grouped_rolling(
            combined,
            lag1,
            keys,
            30,
            "mean",
        )

        std30 = grouped_rolling(
            combined,
            lag1,
            keys,
            30,
            "std",
        )

        names = {
            f"{z}_lag1": lag1,
            f"{z}_lag2": lag2,
            f"{z}_lag7": lag7,
            f"{z}_mean7": mean7,
            f"{z}_std7": std7,
            f"{z}_mean30": mean30,
            f"{z}_std30": std30,
            f"{z}_delta1": lag1 - lag2,
            f"{z}_delta7": lag1 - lag7,
            f"{z}_mean_gap_7_30": (
                mean7
                - mean30
            ),
        }

        for name, values in names.items():
            combined[
                name
            ] = values.astype(
                np.float32
            )

            history_feature_cols.append(
                name
            )

    raw_cols = [
        *[
            f"_curvevec_p{i:02d}"
            for i in range(
                21
            )
        ],
        "_curvevec_q_anchor",
        "_curvevec_log_q_span",
    ]

    for c in raw_cols:
        combined[
            f"{c}_lag1"
        ] = g[
            c
        ].shift(
            1
        ).astype(
            np.float32
        )

    current_out = (
        combined.loc[
            combined[
                "_is_current"
            ].eq(
                1
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    lag1_cols = [
        f"{z}_lag1"
        for z in latent_cols
    ]

    current_out[
        "latent_history_ready_flag"
    ] = (
        current_out[
            lag1_cols
        ]
        .notna()
        .all(
            axis=1
        )
        .astype(
            np.int8
        )
    )

    new_tail = (
        combined[
            keep_state_cols
        ]
        .groupby(
            keys,
            sort=False,
            as_index=False,
            group_keys=False,
        )
        .tail(
            max_history
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        current_out,
        new_tail,
        history_feature_cols,
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
        "--max-history",
        type=int,
        default=30,
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

    manifest = json.loads(
        (
            latent_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    selected_k = int(
        manifest[
            "selected_latent_dim"
        ]
    )

    latent_cols = [
        f"latent_z{i+1:02d}"
        for i in range(
            selected_k
        )
    ]

    schema_file = (
        latent_dir
        / "feature_schema.csv"
    )

    if not schema_file.exists():
        raise FileNotFoundError(
            schema_file
        )

    source_schema = pd.read_csv(
        schema_file
    )

    base_features = select_base_features(
        source_schema
    )

    out = (
        base
        / "latent_forecasting_dataset"
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

    tail = None
    output_manifest = {
        "year": int(
            args.year
        ),
        "source_latent_dir": str(
            latent_dir
        ),
        "selected_latent_dim": int(
            selected_k
        ),
        "latent_columns": latent_cols,
        "base_features": base_features,
        "parts": {},
    }

    split_stats = {}
    history_feature_cols = None

    print("=" * 80)
    print(
        f"Latent forecasting dataset - {args.year}"
    )
    print("=" * 80)
    print(
        f"Latent dimension = {selected_k}"
    )
    print(
        f"Base context features = {len(base_features)}"
    )
    print(
        "History key = participant_id + local_slot_seconds"
    )
    print()

    for split in [
        "train",
        "val",
        "test",
    ]:
        part_files = latent_part_files(
            manifest,
            split,
        )

        split_out_dir = (
            out
            / "parts"
            / split
        )

        split_out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        written = []
        total_rows = 0
        ready_rows = 0

        for i, rel in enumerate(
            part_files,
            1,
        ):
            src = (
                latent_dir
                / rel
            )

            print(
                f"[{split} {i}/{len(part_files)}] "
                f"{src.name}",
                flush=True,
            )

            d = pd.read_pickle(
                src
            )

            required = [
                *REQUIRED_META_COLS,
                *base_features,
                *latent_cols,
                *CURVE_TARGET_COLS,
            ]

            missing = [
                c
                for c in required
                if c not in d.columns
            ]

            if missing:
                raise KeyError(
                    f"{src.name}: missing required columns "
                    f"{missing[:20]}"
                )

            meta_cols = available_meta_cols(
                d
            )

            d = add_current_curve_vector(
                d
            )

            enriched, tail, hist_cols = build_history_for_part(
                d,
                tail,
                latent_cols,
                args.max_history,
            )

            if history_feature_cols is None:
                history_feature_cols = hist_cols

            keep = [
                *meta_cols,
                *base_features,
                *latent_cols,
                *history_feature_cols,
                "latent_history_ready_flag",
                *CURVE_TARGET_COLS,
                *[
                    f"_curvevec_p{i:02d}_lag1"
                    for i in range(
                        21
                    )
                ],
                "_curvevec_q_anchor_lag1",
                "_curvevec_log_q_span_lag1",
            ]

            # Preserve template label only for later explanation/evaluation.
            if (
                "y_template_id"
                in enriched.columns
            ):
                keep.append(
                    "y_template_id"
                )

            out_d = enriched[
                keep
            ].copy()

            out_name = (
                f"{split}_latent_forecasting_"
                f"{i:04d}.pkl"
            )

            out_path = (
                split_out_dir
                / out_name
            )

            out_d.to_pickle(
                out_path
            )

            total_rows += int(
                len(
                    out_d
                )
            )

            ready_rows += int(
                out_d[
                    "latent_history_ready_flag"
                ].sum()
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
                    "history_ready_rows": int(
                        out_d[
                            "latent_history_ready_flag"
                        ].sum()
                    ),
                }
            )

            del d, enriched, out_d
            gc.collect()

        output_manifest[
            "parts"
        ][
            split
        ] = written

        split_stats[
            split
        ] = {
            "rows": int(
                total_rows
            ),
            "history_ready_rows": int(
                ready_rows
            ),
            "history_ready_share": float(
                ready_rows
                / max(
                    total_rows,
                    1,
                )
            ),
        }

    model_features = [
        *base_features,
        *history_feature_cols,
    ]

    rows = []

    schema_meta_cols = list(
        dict.fromkeys(
            [
                *REQUIRED_META_COLS,
                *OPTIONAL_META_COLS,
            ]
        )
    )

    for c in schema_meta_cols:
        rows.append(
            {
                "column": c,
                "role": "metadata",
                "feature_group": "metadata",
                "leakage_use": "audit_only",
            }
        )

    for c in base_features:
        group = (
            source_schema.loc[
                source_schema[
                    "column"
                ].astype(str).eq(c),
                "feature_group",
            ]
            .astype(str)
            .iloc[0]
        )

        rows.append(
            {
                "column": c,
                "role": "feature",
                "feature_group": group,
                "leakage_use": "feature",
            }
        )

    for c in history_feature_cols:
        rows.append(
            {
                "column": c,
                "role": "feature",
                "feature_group": "latent_history",
                "leakage_use": "feature",
            }
        )

    rows.append(
        {
            "column": "latent_history_ready_flag",
            "role": "quality_flag",
            "feature_group": "latent_history",
            "leakage_use": "filter_only",
        }
    )

    for c in latent_cols:
        rows.append(
            {
                "column": c,
                "role": "target",
                "feature_group": "latent_target",
                "leakage_use": "target_only",
            }
        )

    for c in [
        *[
            f"_curvevec_p{i:02d}_lag1"
            for i in range(
                21
            )
        ],
        "_curvevec_q_anchor_lag1",
        "_curvevec_log_q_span_lag1",
    ]:
        rows.append(
            {
                "column": c,
                "role": "evaluation_baseline",
                "feature_group": "raw_curve_history",
                "leakage_use": "evaluation_only",
            }
        )

    feature_schema = pd.DataFrame(
        rows
    )

    feature_schema.to_csv(
        out
        / "feature_schema.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_manifest[
        "model_features"
    ] = model_features

    output_manifest[
        "history_features"
    ] = history_feature_cols

    output_manifest[
        "split_stats"
    ] = split_stats

    (
        out
        / "manifest.json"
    ).write_text(
        json.dumps(
            output_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            (
                f"Latent forecasting dataset - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                f"Latent dimension = "
                f"{selected_k}"
            ),
            (
                f"Base context features = "
                f"{len(base_features)}"
            ),
            (
                f"Latent history features = "
                f"{len(history_feature_cols)}"
            ),
            (
                f"Total model features = "
                f"{len(model_features)}"
            ),
            "",
            "Split statistics:",
            json.dumps(
                split_stats,
                ensure_ascii=False,
                indent=2,
            ),
            "",
            (
                "No current latent target, current curve, "
                "current template or current theta is used as a feature."
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
