#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_build_seasonal_intraday_strategy_profile.py

Build historical seasonal and intraday strategy components.

Outputs
-------
1. participant_month_strategy_profile_<year>.csv
   participant x month historical conditional profile.
   This table is intended to be looked up by the FUTURE TARGET MONTH.

2. participant_slot_strategy_profile_<year>.csv
   participant x local time slot historical conditional profile.
   This table is intended to be looked up by the FUTURE TARGET SLOT.

3. seasonal_strategy_summary_<year>.csv
   participant-level low-dimensional seasonal summary.

4. intraday_strategy_summary_<year>.csv
   participant-level low-dimensional intraday summary.

Important
---------
All tables are constructed from the historical reference year. Future
prediction does not require target-year recent bid curves. The target calendar
(month/slot) only selects which historical conditional profile row to use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    DAILY_SHAPE_COLS,
    LT_SHAPE_COLS,
    SEASONAL_SUMMARY_FEATURES,
    INTRADAY_SUMMARY_FEATURES,
    ensure_dir,
    robust_scale,
    rmse,
)


MONTH_SCALAR_MAP = {
    "bid_level": (
        "daily_bid_level",
        "lt_bid_level",
    ),
    "adjustment_magnitude": (
        "daily_adjustment_magnitude",
        "lt_adjustment_magnitude",
    ),
    "quantity_hhi": (
        "daily_quantity_hhi",
        "lt_quantity_hhi",
    ),
    "effective_segment_count": (
        "daily_effective_segment_count",
        "lt_effective_segment_count",
    ),
    "flat_curve_rate": (
        "daily_flat_curve_rate",
        "lt_flat_curve_rate",
    ),
    "tail_uplift_ratio": (
        "daily_tail_uplift_ratio",
        "lt_tail_uplift_ratio",
    ),
    "curve_bend_ratio": (
        "daily_curve_bend_ratio",
        "lt_curve_bend_ratio",
    ),
}


SLOT_SCALAR_MAP = {
    "bid_level": (
        "slot_bid_level_mean",
        "lt_bid_level",
    ),
    "adjustment_magnitude": (
        "slot_adjustment_magnitude_mean",
        "lt_adjustment_magnitude",
    ),
    "quantity_hhi": (
        "slot_quantity_hhi_mean",
        "lt_quantity_hhi",
    ),
    "effective_segment_count": (
        "slot_effective_segment_count_mean",
        "lt_effective_segment_count",
    ),
    "flat_curve_rate": (
        "slot_flat_curve_flag_mean",
        "lt_flat_curve_rate",
    ),
    "tail_uplift_ratio": (
        "slot_tail_uplift_ratio_mean",
        "lt_tail_uplift_ratio",
    ),
    "curve_bend_ratio": (
        "slot_curve_bend_ratio_mean",
        "lt_curve_bend_ratio",
    ),
}


def build_month_profiles(
    daily,
    lt,
    min_month_days,
):
    lt_lookup = (
        lt.set_index(
            "participant_id"
        )
    )

    rows = []

    for pid, g in daily.groupby(
        "participant_id",
        sort=False,
    ):
        if (
            pid
            not in lt_lookup.index
        ):
            continue

        annual = (
            lt_lookup.loc[
                pid
            ]
        )

        annual_shape = np.asarray(
            [
                pd.to_numeric(
                    pd.Series(
                        [
                            annual.get(
                                c,
                                np.nan,
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
                for c in (
                    LT_SHAPE_COLS
                )
            ],
            dtype=float,
        )

        for month in range(
            1,
            13,
        ):
            gm = g[
                g[
                    "local_date"
                ].dt.month.eq(
                    month
                )
            ]

            row = {
                "participant_id": pid,
                "month": int(
                    month
                ),
                "month_active_days": int(
                    gm[
                        "local_date"
                    ].nunique()
                ),
            }

            sufficient = (
                row[
                    "month_active_days"
                ]
                >= min_month_days
            )

            for stem, (
                daily_col,
                lt_col,
            ) in (
                MONTH_SCALAR_MAP.items()
            ):
                vals = (
                    pd.to_numeric(
                        gm[
                            daily_col
                        ],
                        errors="coerce",
                    )
                    .dropna()
                )

                month_value = (
                    float(
                        vals.median()
                    )
                    if (
                        sufficient
                        and len(
                            vals
                        )
                    )
                    else np.nan
                )

                annual_value = (
                    pd.to_numeric(
                        pd.Series(
                            [
                                annual.get(
                                    lt_col,
                                    np.nan,
                                )
                            ]
                        ),
                        errors="coerce",
                    )
                    .iloc[
                        0
                    ]
                )

                row[
                    f"month_{stem}"
                ] = (
                    month_value
                )

                row[
                    f"season_{stem}_delta"
                ] = (
                    month_value
                    - annual_value
                    if (
                        np.isfinite(
                            month_value
                        )
                        and np.isfinite(
                            annual_value
                        )
                    )
                    else np.nan
                )

            shape = (
                gm[
                    DAILY_SHAPE_COLS
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
            )

            complete = (
                shape.notna()
                .all(axis=1)
            )

            if (
                sufficient
                and complete.sum()
                >= 3
                and np.isfinite(
                    annual_shape
                ).all()
            ):
                month_proto = (
                    shape.loc[
                        complete
                    ]
                    .median(
                        axis=0
                    )
                    .to_numpy(
                        float
                    )
                )

                row[
                    "season_shape_shift"
                ] = rmse(
                    month_proto,
                    annual_shape,
                )

            else:
                row[
                    "season_shape_shift"
                ] = np.nan

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def build_seasonal_summary(
    month_profile,
):
    rows = []

    for pid, g in (
        month_profile.groupby(
            "participant_id",
            sort=False,
        )
    ):
        def value_range(
            col,
        ):
            x = (
                pd.to_numeric(
                    g[
                        col
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            return (
                float(
                    x.max()
                    - x.min()
                )
                if len(
                    x
                ) >= 2
                else np.nan
            )

        shape_shift = (
            pd.to_numeric(
                g[
                    "season_shape_shift"
                ],
                errors="coerce",
            )
            .dropna()
        )

        rows.append(
            {
                "participant_id": pid,
                "season_months_available": int(
                    (
                        pd.to_numeric(
                            g[
                                "month_bid_level"
                            ],
                            errors="coerce",
                        )
                        .notna()
                    )
                    .sum()
                ),
                "season_bid_level_range": value_range(
                    "month_bid_level"
                ),
                "season_adjustment_magnitude_range": value_range(
                    "month_adjustment_magnitude"
                ),
                "season_quantity_hhi_range": value_range(
                    "month_quantity_hhi"
                ),
                "season_effective_segment_count_range": value_range(
                    "month_effective_segment_count"
                ),
                "season_tail_uplift_ratio_range": value_range(
                    "month_tail_uplift_ratio"
                ),
                "season_shape_shift_median": (
                    float(
                        shape_shift.median()
                    )
                    if len(
                        shape_shift
                    )
                    else np.nan
                ),
            }
        )

    out = pd.DataFrame(
        rows
    )

    out[
        "season_nonmissing_count"
    ] = (
        out[
            SEASONAL_SUMMARY_FEATURES
        ]
        .notna()
        .sum(axis=1)
    )

    out[
        "season_ready_flag"
    ] = (
        out[
            "season_nonmissing_count"
        ] >= 4
    ).astype(int)

    return out


def build_slot_profiles(
    slot_core,
    lt,
):
    out = slot_core.merge(
        lt[
            [
                "participant_id",
                "lt_bid_level",
                "lt_adjustment_magnitude",
                "lt_quantity_hhi",
                "lt_effective_segment_count",
                "lt_flat_curve_rate",
                "lt_tail_uplift_ratio",
                "lt_curve_bend_ratio",
            ]
        ],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )

    for stem, (
        slot_col,
        lt_col,
    ) in (
        SLOT_SCALAR_MAP.items()
    ):
        out[
            f"intraday_{stem}_delta"
        ] = (
            pd.to_numeric(
                out[
                    slot_col
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                out[
                    lt_col
                ],
                errors="coerce",
            )
        )

    return out


def build_intraday_summary(
    slot_profile,
    daily,
):
    switch_by_pid = (
        daily.groupby(
            "participant_id"
        )[
            "daily_curve_switch_rate"
        ]
        .median()
        if (
            "daily_curve_switch_rate"
            in daily.columns
        )
        else pd.Series(
            dtype=float
        )
    )

    rows = []

    for pid, g in (
        slot_profile.groupby(
            "participant_id",
            sort=False,
        )
    ):
        def value_range(
            col,
        ):
            x = (
                pd.to_numeric(
                    g[
                        col
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            return (
                float(
                    x.max()
                    - x.min()
                )
                if len(
                    x
                ) >= 2
                else np.nan
            )

        bid_slot = (
            pd.to_numeric(
                g[
                    "slot_bid_level_mean"
                ],
                errors="coerce",
            )
            .dropna()
        )

        rows.append(
            {
                "participant_id": pid,
                "intraday_slots_available": int(
                    bid_slot.shape[
                        0
                    ]
                ),
                "intraday_bid_level_range": value_range(
                    "slot_bid_level_mean"
                ),
                "intraday_bid_level_robust_scale": robust_scale(
                    bid_slot
                ),
                "intraday_adjustment_magnitude_range": value_range(
                    "slot_adjustment_magnitude_mean"
                ),
                "intraday_quantity_hhi_range": value_range(
                    "slot_quantity_hhi_mean"
                ),
                "intraday_switch_rate": (
                    float(
                        switch_by_pid.get(
                            pid,
                            np.nan,
                        )
                    )
                    if np.isfinite(
                        switch_by_pid.get(
                            pid,
                            np.nan,
                        )
                    )
                    else np.nan
                ),
            }
        )

    out = pd.DataFrame(
        rows
    )

    out[
        "intraday_nonmissing_count"
    ] = (
        out[
            INTRADAY_SUMMARY_FEATURES
        ]
        .notna()
        .sum(axis=1)
    )

    out[
        "intraday_ready_flag"
    ] = (
        out[
            "intraday_nonmissing_count"
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
            "final_clean/context_profile"
        ),
    )

    p.add_argument(
        "--min-month-days",
        type=int,
        default=10,
    )

    args = p.parse_args()

    daily_file = (
        Path(args.root)
        / "daily"
        / str(args.year)
        / f"daily_strategy_core_{args.year}.csv"
    )

    slot_file = (
        Path(args.root)
        / "daily"
        / str(args.year)
        / f"intraday_slot_core_{args.year}.csv"
    )

    lt_file = (
        Path(args.root)
        / "long_term"
        / str(args.year)
        / f"long_term_strategy_profile_{args.year}.csv"
    )

    for f in (
        daily_file,
        slot_file,
        lt_file,
    ):
        if not f.exists():
            raise FileNotFoundError(
                f
            )

    daily = pd.read_csv(
        daily_file,
        low_memory=False,
    )

    slot_core = pd.read_csv(
        slot_file,
        low_memory=False,
    )

    lt = pd.read_csv(
        lt_file,
        low_memory=False,
    )

    for frame in (
        daily,
        slot_core,
        lt,
    ):
        frame[
            "participant_id"
        ] = (
            frame[
                "participant_id"
            ]
            .astype(str)
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

    out_dir = ensure_dir(
        Path(args.out_root)
        / str(args.year)
    )

    month_profile = (
        build_month_profiles(
            daily=daily,
            lt=lt,
            min_month_days=args.min_month_days,
        )
    )

    seasonal_summary = (
        build_seasonal_summary(
            month_profile
        )
    )

    slot_profile = (
        build_slot_profiles(
            slot_core=slot_core,
            lt=lt,
        )
    )

    intraday_summary = (
        build_intraday_summary(
            slot_profile=slot_profile,
            daily=daily,
        )
    )

    month_file = (
        out_dir
        / f"participant_month_strategy_profile_{args.year}.csv"
    )

    season_file = (
        out_dir
        / f"seasonal_strategy_summary_{args.year}.csv"
    )

    slot_out_file = (
        out_dir
        / f"participant_slot_strategy_profile_{args.year}.csv"
    )

    intraday_file = (
        out_dir
        / f"intraday_strategy_summary_{args.year}.csv"
    )

    month_profile.to_csv(
        month_file,
        index=False,
        encoding="utf-8-sig",
    )

    seasonal_summary.to_csv(
        season_file,
        index=False,
        encoding="utf-8-sig",
    )

    slot_profile.to_csv(
        slot_out_file,
        index=False,
        encoding="utf-8-sig",
    )

    intraday_summary.to_csv(
        intraday_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print(
        "Seasonal / intraday historical profile complete"
    )
    print("=" * 72)
    print(
        f"Month profile rows: "
        f"{len(month_profile):,}"
    )
    print(
        f"Slot profile rows: "
        f"{len(slot_profile):,}"
    )
    print(
        f"Season-ready participants: "
        f"{int(seasonal_summary['season_ready_flag'].sum()):,}"
    )
    print(
        f"Intraday-ready participants: "
        f"{int(intraday_summary['intraday_ready_flag'].sum()):,}"
    )
    print(
        f"Month lookup: {month_file}"
    )
    print(
        f"Slot lookup:  {slot_out_file}"
    )
    print(
        f"Season summary: {season_file}"
    )
    print(
        f"Intraday summary: {intraday_file}"
    )


if __name__ == "__main__":
    main()
