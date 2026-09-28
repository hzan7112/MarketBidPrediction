#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_build_strategy_transition_profile.py

Historical template assignments -> participant-level strategy-transition profile.

This script is intentionally downstream of bidtemplate. It is template-ID
agnostic: the template library may contain T00/T01/.../FLAT today or the newer
F0/F1/... structural families later. Only the assignment column `template_id`
is required.

The script does NOT make bidprofile depend on target-year recent bids. It
summarizes template usage and transition behavior over the complete historical
reference period.

Outputs
-------
strategy_transition_daily_<year>.csv
strategy_transition_profile_<year>.csv

Final subject-level transition features:
- transition_dominant_template_share
- transition_active_template_count
- transition_template_usage_entropy
- transition_daily_dominant_switch_rate
- transition_pair_entropy

Auxiliary daily diversity is also retained as
`transition_daily_template_entropy_median`.

Auxiliary:
- transition_daily_dominant_inertia = 1 - switch_rate
- transition_dominant_template_id
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    TRANSITION_FEATURES,
    ensure_dir,
    normalized_entropy,
)


def build_daily_template_table(
    assignment_files,
    chunksize,
):
    counts = defaultdict(
        Counter
    )

    for file_no, file in enumerate(
        assignment_files,
        start=1,
    ):
        print(
            f"[05] assignment file "
            f"{file_no}/{len(assignment_files)}: "
            f"{file.name}",
            flush=True,
        )

        header = pd.read_csv(
            file,
            nrows=0,
        )

        required = {
            "participant_id",
            "local_date",
            "template_id",
        }

        missing = (
            required
            - set(
                header.columns
            )
        )

        if missing:
            raise KeyError(
                f"{file.name} missing columns: "
                f"{sorted(missing)}"
            )

        for chunk in pd.read_csv(
            file,
            usecols=[
                "participant_id",
                "local_date",
                "template_id",
            ],
            chunksize=chunksize,
            low_memory=False,
        ):
            chunk[
                "participant_id"
            ] = (
                chunk[
                    "participant_id"
                ]
                .astype(str)
            )

            chunk[
                "local_date"
            ] = (
                pd.to_datetime(
                    chunk[
                        "local_date"
                    ],
                    errors="coerce",
                )
                .dt.normalize()
            )

            chunk[
                "template_id"
            ] = (
                chunk[
                    "template_id"
                ]
                .astype(str)
            )

            chunk = chunk.dropna(
                subset=[
                    "local_date",
                ]
            )

            grouped = (
                chunk.groupby(
                    [
                        "participant_id",
                        "local_date",
                        "template_id",
                    ],
                    sort=False,
                )
                .size()
            )

            for (
                pid,
                date,
                tid,
            ), n in (
                grouped.items()
            ):
                counts[
                    (
                        str(pid),
                        pd.Timestamp(
                            date
                        ),
                    )
                ][
                    str(tid)
                ] += int(
                    n
                )

    participant_usage = defaultdict(Counter)
    rows = []

    for (
        pid,
        date,
    ), counter in (
        counts.items()
    ):
        total = int(
            sum(
                counter.values()
            )
        )

        if total <= 0:
            continue

        dominant_id, dominant_n = (
            counter.most_common(
                1
            )[
                0
            ]
        )

        participant_usage[
            str(pid)
        ].update(
            counter
        )

        rows.append(
            {
                "participant_id": pid,
                "local_date": date,
                "template_observation_count": total,
                "daily_active_template_count": int(
                    len(
                        counter
                    )
                ),
                "daily_dominant_template_id": dominant_id,
                "daily_dominant_template_share": float(
                    dominant_n
                    / total
                ),
                "daily_template_entropy": normalized_entropy(
                    list(
                        counter.values()
                    )
                ),
            }
        )

    if not rows:
        return (
            pd.DataFrame(),
            participant_usage,
        )

    daily = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "participant_id",
                "local_date",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        daily,
        participant_usage,
    )


def build_participant_transition_profile(
    daily,
    participant_usage,
):
    rows = []

    for pid, g in daily.groupby(
        "participant_id",
        sort=False,
    ):
        g = (
            g.sort_values(
                "local_date"
            )
            .reset_index(
                drop=True
            )
        )

        # Exact interval-level historical template usage, accumulated while
        # reading assignment files. This avoids approximating annual template
        # preference from daily dominant templates.
        usage = participant_usage.get(
            str(pid),
            Counter(),
        )

        total_usage = float(
            sum(
                usage.values()
            )
        )

        if total_usage > 0:
            dominant_id = max(
                usage,
                key=usage.get,
            )

            dominant_share = float(
                usage[
                    dominant_id
                ]
                / total_usage
            )
        else:
            dominant_id = ""
            dominant_share = np.nan

        active_count = int(
            len(
                usage
            )
        )

        usage_entropy = (
            normalized_entropy(
                list(
                    usage.values()
                )
            )
            if usage
            else np.nan
        )

        # Only adjacent calendar-day dominant states define a transition.
        transition_total = 0
        transition_switch = 0
        transition_pairs = Counter()

        dates = pd.to_datetime(
            g[
                "local_date"
            ]
        )

        templates = (
            g[
                "daily_dominant_template_id"
            ]
            .astype(str)
            .to_numpy()
        )

        for j in range(
            1,
            len(
                g
            ),
        ):
            if (
                dates.iloc[
                    j
                ]
                - dates.iloc[
                    j - 1
                ]
            ).days != 1:
                continue

            transition_total += 1

            pair = (
                templates[
                    j - 1
                ],
                templates[
                    j
                ],
            )

            transition_pairs[
                pair
            ] += 1

            if (
                templates[
                    j
                ]
                != templates[
                    j - 1
                ]
            ):
                transition_switch += 1

        switch_rate = (
            float(
                transition_switch
                / transition_total
            )
            if transition_total
            else np.nan
        )

        entropy_series = (
            pd.to_numeric(
                g[
                    "daily_template_entropy"
                ],
                errors="coerce",
            )
            .dropna()
        )

        rows.append(
            {
                "participant_id": str(
                    pid
                ),
                "transition_active_days": int(
                    len(
                        g
                    )
                ),
                "transition_observations": int(
                    pd.to_numeric(
                        g[
                            "template_observation_count"
                        ],
                        errors="coerce",
                    )
                    .fillna(
                        0
                    )
                    .sum()
                ),
                "transition_dominant_template_id": dominant_id,
                "transition_dominant_template_share": dominant_share,
                "transition_active_template_count": active_count,
                "transition_template_usage_entropy": usage_entropy,
                "transition_daily_dominant_switch_rate": switch_rate,
                "transition_daily_dominant_inertia": (
                    1.0
                    - switch_rate
                    if np.isfinite(
                        switch_rate
                    )
                    else np.nan
                ),
                "transition_pair_entropy": (
                    normalized_entropy(
                        list(
                            transition_pairs.values()
                        )
                    )
                    if transition_pairs
                    else np.nan
                ),
                "transition_daily_template_entropy_median": (
                    float(
                        entropy_series.median()
                    )
                    if len(
                        entropy_series
                    )
                    else np.nan
                ),
                "transition_pair_count": int(
                    transition_total
                ),
            }
        )

    out = pd.DataFrame(
        rows
    )

    out[
        "transition_nonmissing_count"
    ] = (
        out[
            TRANSITION_FEATURES
        ]
        .notna()
        .sum(axis=1)
    )

    out[
        "transition_ready_flag"
    ] = (
        out[
            "transition_nonmissing_count"
        ] >= 4
    ).astype(int)

    return out


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    p.add_argument(
        "--assignment-root",
        default=(
            "data/processed/"
            "bidtemplate"
        ),
    )

    p.add_argument(
        "--out-root",
        default=(
            "data/processed/"
            "final_clean/transition_profile"
        ),
    )

    p.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
    )

    args = p.parse_args()

    assignment_dir = (
        Path(
            args.assignment_root
        )
        / str(
            args.year
        )
        / "template_library"
        / "assignments"
    )

    files = sorted(
        assignment_dir.glob(
            "*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No template assignment CSV files under "
            f"{assignment_dir}. "
            "Run bidtemplate template discovery/assignment first."
        )

    out_dir = ensure_dir(
        Path(
            args.out_root
        )
        / str(
            args.year
        )
    )

    daily_file = (
        out_dir
        / f"strategy_transition_daily_{args.year}.csv"
    )

    profile_file = (
        out_dir
        / f"strategy_transition_profile_{args.year}.csv"
    )

    (
        daily,
        participant_usage,
    ) = (
        build_daily_template_table(
            assignment_files=files,
            chunksize=args.chunksize,
        )
    )

    if daily.empty:
        raise RuntimeError(
            "No valid template assignment rows found."
        )

    profile = (
        build_participant_transition_profile(
            daily,
            participant_usage,
        )
    )

    daily.to_csv(
        daily_file,
        index=False,
        encoding="utf-8-sig",
    )

    profile.to_csv(
        profile_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print(
        "Historical strategy-transition profile complete"
    )
    print("=" * 72)
    print(
        f"Template assignment files: "
        f"{len(files)}"
    )
    print(
        f"Participant-days: "
        f"{len(daily):,}"
    )
    print(
        f"Participants: "
        f"{len(profile):,}"
    )
    print(
        f"Transition-ready: "
        f"{int(profile['transition_ready_flag'].sum()):,}"
    )
    print(
        f"Daily audit: {daily_file}"
    )
    print(
        f"Participant transition profile: "
        f"{profile_file}"
    )


if __name__ == "__main__":
    main()
