#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_build_final_strategy_profile.py

Combine the historical multi-timescale subject strategy profile.

Subject-level final profile Z_i:
    LT
  + historical short-term tendency
  + seasonal summary
  + intraday summary
  + optional template-transition summary

The participant x month and participant x slot context tables are NOT widened
into Z_i. They remain separate historical lookup tables and should be joined by
the future target calendar (month / slot) in bidprediction.

This keeps the participant profile compact while preserving context-specific
historical behavior.

No KMeans is performed here. Template clustering belongs to bidtemplate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    LT_FEATURES,
    SHORT_TENDENCY_FEATURES,
    SEASONAL_SUMMARY_FEATURES,
    INTRADAY_SUMMARY_FEATURES,
    TRANSITION_FEATURES,
    empirical_percentile,
    ensure_dir,
)


GROUPS = {
    "long_term": LT_FEATURES,
    "short_tendency": SHORT_TENDENCY_FEATURES,
    "seasonal_summary": SEASONAL_SUMMARY_FEATURES,
    "intraday_summary": INTRADAY_SUMMARY_FEATURES,
    "transition": TRANSITION_FEATURES,
}


def merge_one(
    base,
    frame,
    label,
):
    if frame is None:
        return base

    if (
        "participant_id"
        not in frame.columns
    ):
        raise KeyError(
            f"{label} missing participant_id"
        )

    dup = (
        frame[
            "participant_id"
        ]
        .astype(str)
        .duplicated()
        .any()
    )

    if dup:
        raise ValueError(
            f"{label} has duplicate participant_id rows."
        )

    frame = frame.copy()

    frame[
        "participant_id"
    ] = (
        frame[
            "participant_id"
        ]
        .astype(str)
    )

    return base.merge(
        frame,
        on="participant_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            f"_{label}",
        ),
    )


def feature_dictionary(
    transition_available,
):
    rows = []

    for group, features in (
        GROUPS.items()
    ):
        if (
            group
            == "transition"
            and not transition_available
        ):
            continue

        for idx, f in enumerate(
            features,
            start=1,
        ):
            rows.append(
                {
                    "feature_group": group,
                    "group_order": idx,
                    "feature": f,
                    "model_input_default": 1,
                    "description": "",
                }
            )

    # Contextual historical lookup features are separate from subject-level Z_i.
    month_features = [
        "season_bid_level_delta",
        "season_adjustment_magnitude_delta",
        "season_quantity_hhi_delta",
        "season_effective_segment_count_delta",
        "season_flat_curve_rate_delta",
        "season_tail_uplift_ratio_delta",
        "season_curve_bend_ratio_delta",
        "season_shape_shift",
    ]

    for idx, f in enumerate(
        month_features,
        start=1,
    ):
        rows.append(
            {
                "feature_group": "month_context_lookup",
                "group_order": idx,
                "feature": f,
                "model_input_default": 1,
                "description": (
                    "Select from the historical participant x month table "
                    "using target calendar month."
                ),
            }
        )

    slot_features = [
        "intraday_bid_level_delta",
        "intraday_adjustment_magnitude_delta",
        "intraday_quantity_hhi_delta",
        "intraday_effective_segment_count_delta",
        "intraday_flat_curve_rate_delta",
        "intraday_tail_uplift_ratio_delta",
        "intraday_curve_bend_ratio_delta",
    ]

    for idx, f in enumerate(
        slot_features,
        start=1,
    ):
        rows.append(
            {
                "feature_group": "slot_context_lookup",
                "group_order": idx,
                "feature": f,
                "model_input_default": 1,
                "description": (
                    "Select from the historical participant x slot table "
                    "using target local time slot."
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    p.add_argument(
        "--root",
        default=(
            "data/processed/"
            "final_clean"
        ),
    )

    p.add_argument(
        "--out-root",
        default=(
            "data/processed/"
            "final_clean/strategy_profile"
        ),
    )

    p.add_argument(
        "--require-transition",
        action="store_true",
        help=(
            "Fail if template-transition profile is unavailable. "
            "Without this flag, a base profile is still produced."
        ),
    )

    args = p.parse_args()

    root = Path(
        args.root
    )

    lt_file = (
        root
        / "long_term"
        / str(args.year)
        / f"long_term_strategy_profile_{args.year}.csv"
    )

    short_file = (
        root
        / "short_term_history"
        / str(args.year)
        / f"historical_short_term_tendency_{args.year}.csv"
    )

    context_dir = (
        root
        / "context_profile"
        / str(args.year)
    )

    season_file = (
        context_dir
        / f"seasonal_strategy_summary_{args.year}.csv"
    )

    intraday_file = (
        context_dir
        / f"intraday_strategy_summary_{args.year}.csv"
    )

    month_lookup_file = (
        context_dir
        / f"participant_month_strategy_profile_{args.year}.csv"
    )

    slot_lookup_file = (
        context_dir
        / f"participant_slot_strategy_profile_{args.year}.csv"
    )

    transition_file = (
        root
        / "transition_profile"
        / str(args.year)
        / f"strategy_transition_profile_{args.year}.csv"
    )

    required = [
        lt_file,
        short_file,
        season_file,
        intraday_file,
        month_lookup_file,
        slot_lookup_file,
    ]

    for f in required:
        if not f.exists():
            raise FileNotFoundError(
                f
            )

    transition_available = (
        transition_file.exists()
    )

    if (
        args.require_transition
        and not transition_available
    ):
        raise FileNotFoundError(
            transition_file
        )

    lt = pd.read_csv(
        lt_file,
        low_memory=False,
    )

    short = pd.read_csv(
        short_file,
        low_memory=False,
    )

    season = pd.read_csv(
        season_file,
        low_memory=False,
    )

    intraday = pd.read_csv(
        intraday_file,
        low_memory=False,
    )

    transition = (
        pd.read_csv(
            transition_file,
            low_memory=False,
        )
        if transition_available
        else None
    )

    for frame in [
        lt,
        short,
        season,
        intraday,
    ] + (
        [transition]
        if transition is not None
        else []
    ):
        frame[
            "participant_id"
        ] = (
            frame[
                "participant_id"
            ]
            .astype(str)
        )

    profile = (
        lt.copy()
    )

    profile = merge_one(
        profile,
        short,
        "short",
    )

    profile = merge_one(
        profile,
        season,
        "season",
    )

    profile = merge_one(
        profile,
        intraday,
        "intraday",
    )

    profile = merge_one(
        profile,
        transition,
        "transition",
    )

    active_groups = [
        "long_term",
        "short_tendency",
        "seasonal_summary",
        "intraday_summary",
    ]

    if transition_available:
        active_groups.append(
            "transition"
        )

    model_features = []

    for group in (
        active_groups
    ):
        model_features.extend(
            GROUPS[
                group
            ]
        )

    existing_features = [
        f
        for f in (
            model_features
        )
        if f in (
            profile.columns
        )
    ]

    profile[
        "profile_nonmissing_count"
    ] = (
        profile[
            existing_features
        ]
        .notna()
        .sum(axis=1)
    )

    profile[
        "profile_feature_count"
    ] = len(
        existing_features
    )

    # High-coverage flag: at least 80% of the model-facing subject-level
    # historical profile is represented.
    profile[
        "profile_ready_flag"
    ] = (
        profile[
            "profile_nonmissing_count"
        ]
        >= np.ceil(
            0.80
            * max(
                1,
                len(
                    existing_features
                ),
            )
        )
    ).astype(int)

    # Percentiles are interpretation-only. They are not part of default model
    # inputs and are kept in a separate file to avoid accidental duplication.
    pct = profile[
        [
            "participant_id",
        ]
    ].copy()

    for f in (
        existing_features
    ):
        pct[
            f"{f}_percentile"
        ] = empirical_percentile(
            profile[
                f
            ]
        )

    out_dir = ensure_dir(
        Path(
            args.out_root
        )
        / str(args.year)
    )

    profile_file = (
        out_dir
        / f"participant_strategy_profile_{args.year}.csv"
    )

    pct_file = (
        out_dir
        / f"participant_strategy_profile_percentiles_{args.year}.csv"
    )

    dictionary_file = (
        out_dir
        / f"strategy_profile_feature_dictionary_{args.year}.csv"
    )

    manifest_file = (
        out_dir
        / f"strategy_profile_manifest_{args.year}.json"
    )

    dictionary = (
        feature_dictionary(
            transition_available
        )
    )

    profile.to_csv(
        profile_file,
        index=False,
        encoding="utf-8-sig",
    )

    pct.to_csv(
        pct_file,
        index=False,
        encoding="utf-8-sig",
    )

    dictionary.to_csv(
        dictionary_file,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "reference_year": int(
            args.year
        ),
        "subject_profile_file": str(
            profile_file
        ),
        "subject_profile_feature_count": int(
            len(
                existing_features
            )
        ),
        "subject_profile_groups": (
            active_groups
        ),
        "transition_available": bool(
            transition_available
        ),
        "model_facing_subject_features": (
            existing_features
        ),
        "month_context_lookup_file": str(
            month_lookup_file
        ),
        "slot_context_lookup_file": str(
            slot_lookup_file
        ),
        "future_usage": {
            "subject_profile": (
                "freeze from historical reference period"
            ),
            "month_context": (
                "lookup by target calendar month"
            ),
            "slot_context": (
                "lookup by target local slot"
            ),
            "target_year_recent_bids_required": False,
        },
    }

    manifest_file.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "Final historical subject strategy profile complete"
    )
    print("=" * 72)
    print(
        f"Reference year: {args.year}"
    )
    print(
        f"Participants: {len(profile):,}"
    )
    print(
        f"Subject-level model features: "
        f"{len(existing_features)}"
    )
    print(
        f"Transition included: "
        f"{transition_available}"
    )
    print(
        f"Ready participants: "
        f"{int(profile['profile_ready_flag'].sum()):,}"
    )
    print(
        f"Profile: {profile_file}"
    )
    print(
        f"Month lookup: {month_lookup_file}"
    )
    print(
        f"Slot lookup: {slot_lookup_file}"
    )
    print(
        f"Manifest: {manifest_file}"
    )


if __name__ == "__main__":
    main()
