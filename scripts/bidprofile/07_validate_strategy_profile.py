#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_validate_strategy_profile.py

Validation for the new historical multi-timescale bidprofile architecture.

This validator no longer evaluates KMeans clusters inside bidprofile.
Template clustering belongs to bidtemplate.

Validation targets:
1. LT split-half stability.
2. Subject-profile feature coverage by group.
3. Pairwise redundancy among model-facing subject-level features.
4. Historical short-term tendency coverage.
5. Seasonal participant x month lookup coverage.
6. Intraday participant x slot lookup coverage.
7. Optional template-transition coverage.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    LT_FEATURES,
    SHORT_TENDENCY_FEATURES,
    SEASONAL_SUMMARY_FEATURES,
    INTRADAY_SUMMARY_FEATURES,
    TRANSITION_FEATURES,
    ensure_dir,
)


LT_DAILY_MAP = {
    "lt_bid_level": "daily_bid_level",
    "lt_adjustment_magnitude": "daily_adjustment_magnitude",
    "lt_quantity_hhi": "daily_quantity_hhi",
    "lt_effective_segment_count": "daily_effective_segment_count",
    "lt_flat_curve_rate": "daily_flat_curve_rate",
    "lt_tail_uplift_ratio": "daily_tail_uplift_ratio",
    "lt_curve_bend_ratio": "daily_curve_bend_ratio",
}


def lt_split_half(
    daily,
):
    rows = []

    for lt, col in (
        LT_DAILY_MAP.items()
    ):
        h1 = (
            daily[
                daily[
                    "local_date"
                ].dt.month
                <= 6
            ]
            .groupby(
                "participant_id"
            )[col]
            .median()
        )

        h2 = (
            daily[
                daily[
                    "local_date"
                ].dt.month
                >= 7
            ]
            .groupby(
                "participant_id"
            )[col]
            .median()
        )

        z = pd.concat(
            [
                h1,
                h2,
            ],
            axis=1,
            keys=[
                "h1",
                "h2",
            ],
        ).dropna()

        rho = (
            z[
                "h1"
            ]
            .corr(
                z[
                    "h2"
                ],
                method="spearman",
            )
            if len(
                z
            ) >= 10
            else np.nan
        )

        rows.append(
            {
                "feature": lt,
                "participants": int(
                    len(
                        z
                    )
                ),
                "spearman_h1_h2": rho,
            }
        )

    return pd.DataFrame(
        rows
    )


def coverage_by_group(
    profile,
    transition_available,
):
    groups = {
        "long_term": LT_FEATURES,
        "short_tendency": SHORT_TENDENCY_FEATURES,
        "seasonal_summary": SEASONAL_SUMMARY_FEATURES,
        "intraday_summary": INTRADAY_SUMMARY_FEATURES,
    }

    if transition_available:
        groups[
            "transition"
        ] = TRANSITION_FEATURES

    rows = []

    for group, features in (
        groups.items()
    ):
        existing = [
            f
            for f in features
            if f in (
                profile.columns
            )
        ]

        if not existing:
            rows.append(
                {
                    "feature_group": group,
                    "feature_count": 0,
                    "participants": len(
                        profile
                    ),
                    "median_nonmissing_share": np.nan,
                    "full_group_complete_share": 0.0,
                }
            )
            continue

        per_feature = (
            profile[
                existing
            ]
            .notna()
            .mean()
        )

        full_complete = (
            profile[
                existing
            ]
            .notna()
            .all(axis=1)
            .mean()
        )

        rows.append(
            {
                "feature_group": group,
                "feature_count": len(
                    existing
                ),
                "participants": len(
                    profile
                ),
                "median_nonmissing_share": float(
                    per_feature.median()
                ),
                "min_nonmissing_share": float(
                    per_feature.min()
                ),
                "full_group_complete_share": float(
                    full_complete
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def pairwise_redundancy(
    profile,
    features,
):
    rows = []

    existing = [
        f
        for f in features
        if f in (
            profile.columns
        )
    ]

    for a, b in combinations(
        existing,
        2,
    ):
        z = (
            profile[
                [
                    a,
                    b,
                ]
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .dropna()
        )

        rho = (
            z[
                a
            ]
            .corr(
                z[
                    b
                ],
                method="spearman",
            )
            if len(
                z
            ) >= 10
            else np.nan
        )

        rows.append(
            {
                "feature_1": a,
                "feature_2": b,
                "n": len(
                    z
                ),
                "spearman": rho,
                "abs_spearman": (
                    abs(
                        rho
                    )
                    if pd.notna(
                        rho
                    )
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def month_lookup_coverage(
    month_profile,
):
    rows = []

    for pid, g in (
        month_profile.groupby(
            "participant_id",
            sort=False,
        )
    ):
        available = (
            pd.to_numeric(
                g[
                    "month_bid_level"
                ],
                errors="coerce",
            )
            .notna()
            .sum()
        )

        rows.append(
            {
                "participant_id": pid,
                "available_months": int(
                    available
                ),
                "full_12_month_flag": int(
                    available
                    == 12
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def slot_lookup_coverage(
    slot_profile,
):
    rows = []

    for pid, g in (
        slot_profile.groupby(
            "participant_id",
            sort=False,
        )
    ):
        available = (
            pd.to_numeric(
                g[
                    "slot_bid_level_mean"
                ],
                errors="coerce",
            )
            .notna()
            .sum()
        )

        rows.append(
            {
                "participant_id": pid,
                "available_slots": int(
                    available
                ),
                "full_24_hour_flag": int(
                    available
                    >= 24
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
        "--results-root",
        default=(
            "results/"
            "bidprofile_validation"
        ),
    )

    args = p.parse_args()

    root = Path(
        args.root
    )

    daily_file = (
        root
        / "daily"
        / str(args.year)
        / f"daily_strategy_core_{args.year}.csv"
    )

    profile_dir = (
        root
        / "strategy_profile"
        / str(args.year)
    )

    profile_file = (
        profile_dir
        / f"participant_strategy_profile_{args.year}.csv"
    )

    manifest_file = (
        profile_dir
        / f"strategy_profile_manifest_{args.year}.json"
    )

    context_dir = (
        root
        / "context_profile"
        / str(args.year)
    )

    month_file = (
        context_dir
        / f"participant_month_strategy_profile_{args.year}.csv"
    )

    slot_file = (
        context_dir
        / f"participant_slot_strategy_profile_{args.year}.csv"
    )

    for f in (
        daily_file,
        profile_file,
        manifest_file,
        month_file,
        slot_file,
    ):
        if not f.exists():
            raise FileNotFoundError(
                f
            )

    daily = pd.read_csv(
        daily_file,
        low_memory=False,
    )

    profile = pd.read_csv(
        profile_file,
        low_memory=False,
    )

    month_profile = pd.read_csv(
        month_file,
        low_memory=False,
    )

    slot_profile = pd.read_csv(
        slot_file,
        low_memory=False,
    )

    manifest = json.loads(
        manifest_file.read_text(
            encoding="utf-8"
        )
    )

    daily[
        "local_date"
    ] = (
        pd.to_datetime(
            daily[
                "local_date"
            ]
        )
        .dt.normalize()
    )

    transition_available = bool(
        manifest.get(
            "transition_available",
            False,
        )
    )

    model_features = [
        f
        for f in (
            manifest.get(
                "model_facing_subject_features",
                [],
            )
        )
        if f in (
            profile.columns
        )
    ]

    stability = (
        lt_split_half(
            daily
        )
    )

    coverage = (
        coverage_by_group(
            profile,
            transition_available,
        )
    )

    redundancy = (
        pairwise_redundancy(
            profile,
            model_features,
        )
    )

    month_coverage = (
        month_lookup_coverage(
            month_profile
        )
    )

    slot_coverage = (
        slot_lookup_coverage(
            slot_profile
        )
    )

    out_dir = ensure_dir(
        Path(
            args.results_root
        )
        / str(args.year)
    )

    stability.to_csv(
        out_dir
        / "lt_split_half_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage.to_csv(
        out_dir
        / "profile_group_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    redundancy.to_csv(
        out_dir
        / "profile_feature_redundancy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    month_coverage.to_csv(
        out_dir
        / "month_lookup_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    slot_coverage.to_csv(
        out_dir
        / "slot_lookup_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    high_redundancy = (
        redundancy[
            redundancy[
                "abs_spearman"
            ] >= 0.90
        ]
    )

    lines = [
        (
            "Historical multi-timescale bidprofile validation "
            f"- {args.year}"
        ),
        "=" * 84,
        "",
        "Architecture:",
        "  subject profile = LT + historical short-term tendency",
        "                  + seasonal summary + intraday summary",
        "                  + optional template-transition summary",
        "  month/slot context = historical lookup by future target calendar",
        "  target-year recent bid curves required = NO",
        "",
        f"Participants: {len(profile):,}",
        (
            "Subject-level model features: "
            f"{len(model_features)}"
        ),
        (
            "Transition included: "
            f"{transition_available}"
        ),
        (
            "Ready participants: "
            f"{int(pd.to_numeric(profile['profile_ready_flag'], errors='coerce').fillna(0).sum()):,}"
            if (
                "profile_ready_flag"
                in profile.columns
            )
            else "Ready participants: unavailable"
        ),
        (
            "Median LT H1-H2 Spearman: "
            f"{stability['spearman_h1_h2'].median():.4f}"
        ),
        (
            "Subject-profile pairs with |Spearman| >= 0.90: "
            f"{len(high_redundancy)}"
        ),
        (
            "Median available months: "
            f"{month_coverage['available_months'].median():.1f}"
        ),
        (
            "Median available local slots: "
            f"{slot_coverage['available_slots'].median():.1f}"
        ),
        "",
        "Feature-group coverage:",
    ]

    for _, r in (
        coverage.iterrows()
    ):
        lines.append(
            f"  {r['feature_group']}: "
            f"features={int(r['feature_count'])}, "
            f"median_nonmissing="
            f"{r['median_nonmissing_share']:.2%}, "
            f"complete_group="
            f"{r['full_group_complete_share']:.2%}"
        )

    lines += [
        "",
        "Important:",
        "  KMeans/template clustering is intentionally NOT validated here.",
        "  Template discovery and structural-family validation belong to bidtemplate.",
        "  This validator only checks the historical strategy-profile layer.",
    ]

    summary_file = (
        out_dir
        / "summary.txt"
    )

    summary_file.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "\n".join(
            lines
        )
    )

    print()
    print(
        f"Done: {out_dir}"
    )


if __name__ == "__main__":
    main()
