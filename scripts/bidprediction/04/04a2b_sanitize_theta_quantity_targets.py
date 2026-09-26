#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04a2b_sanitize_theta_quantity_targets.py

Remove sentinel-scale / placeholder quantity targets from the 04a2 output
WITHOUT rebuilding Stage2.

Why this exists
---------------
04e diagnosis found that a tiny number of rows carry values such as:
    q_base_mw = -999999
    q_span_mw ~= 1,000,000

These values are finite, so the original 04a2 validity rule accepted them.
They then dominate q_base/q_span MAE even though they are not meaningful
participant offer quantities.

This script does NOT clip targets and does NOT repair them.
It only marks such rows as invalid for theta modeling:
    theta_target_valid_flag = 0
    theta_ready_flag        = 0
    theta_quantity_sentinel_flag = 1

The source CSVs are updated in-place via a temporary file + atomic replace.
An audit CSV is written so every excluded row is traceable.

Default guard:
    abs(q_base_mw) >= 900000
 OR abs(q_span_mw) >= 900000
 OR abs(q_base_mw + q_span_mw) >= 900000

This is a sentinel-scale guard, not a physical unit-capacity constraint.

Run:
python scripts/bidprediction/04a2b_sanitize_theta_quantity_targets.py --year 2025
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = [
    "theta_q_base_mw",
    "theta_q_span_mw",
    "theta_target_valid_flag",
    "theta_ready_flag",
]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


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
        "--abs-mw-threshold",
        type=float,
        default=900_000.0,
        help=(
            "Sentinel-scale absolute MW guard. "
            "This is not a physical capacity limit."
        ),
    )

    ap.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
    )

    args = ap.parse_args()

    base = (
        Path(args.root)
        / str(args.year)
    )

    source_dir = (
        base
        / "template_adjustment_dataset"
    )

    part_dir = (
        source_dir
        / "dataset_parts"
    )

    parts = sorted(
        part_dir.glob(
            "template_adjustment_dataset_*.csv"
        )
    )

    if not parts:
        raise FileNotFoundError(
            f"No 04a2 dataset parts under {part_dir}"
        )

    threshold = float(
        args.abs_mw_threshold
    )

    audit_rows = []
    totals = {
        "rows": 0,
        "flagged_rows": 0,
        "previous_theta_ready_rows": 0,
        "new_theta_ready_rows": 0,
    }

    print("=" * 80)
    print("Sanitize sentinel-scale theta quantity targets")
    print("=" * 80)
    print(f"Year:                 {args.year}")
    print(f"Parts:                {len(parts)}")
    print(f"Absolute MW guard:    {threshold:,.1f}")
    print(
        "Rule: abs(q_base), abs(q_span), or abs(q_base+q_span) "
        ">= threshold"
    )
    print()

    for i, path in enumerate(
        parts,
        1,
    ):
        print(
            f"[part {i}/{len(parts)}] {path.name}",
            flush=True,
        )

        header = pd.read_csv(
            path,
            nrows=0,
        ).columns.tolist()

        missing = [
            c for c in REQUIRED
            if c not in header
        ]

        if missing:
            raise KeyError(
                f"{path.name} missing required columns: {missing}"
            )

        tmp = path.with_suffix(
            path.suffix + ".sanitize_tmp"
        )

        if tmp.exists():
            tmp.unlink()

        first = True
        part_rows = 0
        part_flagged = 0

        for chunk in pd.read_csv(
            path,
            chunksize=args.chunksize,
            low_memory=False,
        ):
            part_rows += len(chunk)
            totals["rows"] += len(chunk)

            qb = num(
                chunk["theta_q_base_mw"]
            ).to_numpy(float)

            qs = num(
                chunk["theta_q_span_mw"]
            ).to_numpy(float)

            qmax = qb + qs

            finite = (
                np.isfinite(qb)
                & np.isfinite(qs)
                & np.isfinite(qmax)
            )

            sentinel = (
                finite
                & (
                    (np.abs(qb) >= threshold)
                    | (np.abs(qs) >= threshold)
                    | (np.abs(qmax) >= threshold)
                )
            )

            old_ready = (
                num(
                    chunk["theta_ready_flag"]
                )
                .fillna(0)
                .eq(1)
                .to_numpy()
            )

            totals[
                "previous_theta_ready_rows"
            ] += int(
                old_ready.sum()
            )

            if (
                "theta_quantity_sentinel_flag"
                not in chunk.columns
            ):
                chunk[
                    "theta_quantity_sentinel_flag"
                ] = np.zeros(
                    len(chunk),
                    dtype=np.int8,
                )

            chunk.loc[
                sentinel,
                "theta_quantity_sentinel_flag",
            ] = 1

            chunk.loc[
                sentinel,
                "theta_target_valid_flag",
            ] = 0

            chunk.loc[
                sentinel,
                "theta_ready_flag",
            ] = 0

            new_ready = (
                num(
                    chunk["theta_ready_flag"]
                )
                .fillna(0)
                .eq(1)
                .to_numpy()
            )

            totals[
                "new_theta_ready_rows"
            ] += int(
                new_ready.sum()
            )

            nflag = int(
                sentinel.sum()
            )

            part_flagged += nflag
            totals[
                "flagged_rows"
            ] += nflag

            if nflag:
                audit_cols = [
                    c
                    for c in [
                        "sample_id",
                        "participant_id",
                        "local_date",
                        "y_template_id",
                        "theta_q_base_mw",
                        "theta_q_span_mw",
                        "unit_lag1_min_ecomax",
                        "unit_lag1_avg_ecomax",
                        "unit_lag1_max_ecomax",
                    ]
                    if c in chunk.columns
                ]

                audit = (
                    chunk.loc[
                        sentinel,
                        audit_cols,
                    ]
                    .copy()
                )

                audit[
                    "derived_q_max_mw"
                ] = qmax[
                    sentinel
                ]

                audit[
                    "source_part"
                ] = path.name

                audit_rows.append(
                    audit
                )

            chunk.to_csv(
                tmp,
                mode="w" if first else "a",
                header=first,
                index=False,
                encoding="utf-8-sig",
            )

            first = False

        os.replace(
            tmp,
            path,
        )

        print(
            f"  rows={part_rows:,}, "
            f"flagged={part_flagged:,}"
        )

    # ------------------------------------------------------------------
    # Audit outputs
    # ------------------------------------------------------------------

    audit_dir = (
        source_dir
        / "quantity_target_sanitization"
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if audit_rows:
        audit_df = pd.concat(
            audit_rows,
            ignore_index=True,
        )
    else:
        audit_df = pd.DataFrame(
            columns=[
                "sample_id",
                "participant_id",
                "local_date",
                "y_template_id",
                "theta_q_base_mw",
                "theta_q_span_mw",
                "derived_q_max_mw",
                "source_part",
            ]
        )

    audit_df.to_csv(
        audit_dir
        / "sentinel_quantity_targets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_participant = pd.DataFrame()

    if (
        not audit_df.empty
        and "participant_id"
        in audit_df.columns
    ):
        by_participant = (
            audit_df.groupby(
                "participant_id",
                dropna=False,
            )
            .agg(
                rows=(
                    "participant_id",
                    "size",
                ),
                q_base_min=(
                    "theta_q_base_mw",
                    "min",
                ),
                q_base_max=(
                    "theta_q_base_mw",
                    "max",
                ),
                q_span_min=(
                    "theta_q_span_mw",
                    "min",
                ),
                q_span_max=(
                    "theta_q_span_mw",
                    "max",
                ),
            )
            .reset_index()
            .sort_values(
                "rows",
                ascending=False,
            )
        )

    by_participant.to_csv(
        audit_dir
        / "sentinel_quantity_targets_by_participant.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Add audit flag to the 04a2 schema so later frozen metadata remains
    # explicit and traceable.
    # ------------------------------------------------------------------

    schema_file = (
        source_dir
        / f"template_adjustment_schema_{args.year}.csv"
    )

    if schema_file.exists():
        schema = pd.read_csv(
            schema_file
        )

        if (
            "theta_quantity_sentinel_flag"
            not in set(
                schema["column"].astype(str)
            )
        ):
            add = pd.DataFrame(
                [
                    {
                        "column": (
                            "theta_quantity_sentinel_flag"
                        ),
                        "role": "quality_flag",
                        "feature_group": (
                            "template_adjustment_quality"
                        ),
                        "source": (
                            "04a2b_sentinel_sanitization"
                        ),
                        "leakage_use": "audit_only",
                    }
                ]
            )

            schema = pd.concat(
                [
                    schema,
                    add,
                ],
                ignore_index=True,
            )

            schema.to_csv(
                schema_file,
                index=False,
                encoding="utf-8-sig",
            )

    cfg = {
        "version": "04a2b-sentinel-quantity-sanitization-v1",
        "year": args.year,
        "abs_mw_threshold": threshold,
        "rule": (
            "flag if abs(q_base_mw), abs(q_span_mw), "
            "or abs(q_base_mw+q_span_mw) >= threshold"
        ),
        "action": (
            "set theta_target_valid_flag=0 and "
            "theta_ready_flag=0; do not clip or repair"
        ),
        **{
            k: int(v)
            for k, v in totals.items()
        },
    }

    (
        audit_dir
        / "config.json"
    ).write_text(
        json.dumps(
            cfg,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    participant_count = (
        int(
            audit_df[
                "participant_id"
            ].nunique(
                dropna=True
            )
        )
        if (
            not audit_df.empty
            and "participant_id"
            in audit_df.columns
        )
        else 0
    )

    summary = "\n".join(
        [
            (
                "04a2b sentinel-scale quantity-target "
                f"sanitization - {args.year}"
            ),
            "=" * 80,
            "",
            f"Rows scanned:                {totals['rows']:,}",
            f"Flagged rows:                {totals['flagged_rows']:,}",
            f"Flagged participants:        {participant_count:,}",
            (
                "Theta-ready before:          "
                f"{totals['previous_theta_ready_rows']:,}"
            ),
            (
                "Theta-ready after:           "
                f"{totals['new_theta_ready_rows']:,}"
            ),
            (
                "Theta-ready rows removed:    "
                f"{totals['previous_theta_ready_rows'] - totals['new_theta_ready_rows']:,}"
            ),
            "",
            (
                f"Absolute MW sentinel guard: {threshold:,.1f}"
            ),
            (
                "No target is clipped or repaired; "
                "flagged rows are excluded from modeling."
            ),
            "",
            f"Audit outputs: {audit_dir}",
        ]
    )

    (
        audit_dir
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)


if __name__ == "__main__":
    main()
