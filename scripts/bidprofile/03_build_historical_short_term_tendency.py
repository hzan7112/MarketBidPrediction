#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_build_historical_short_term_tendency.py

Historical daily strategy core -> participant-level short-term strategy tendency.

Purpose
-------
The new technical route does NOT require future target-year recent bid curves
to construct the subject strategy profile. Therefore this script uses only the
complete historical reference period and summarizes how each participant
typically behaves when it deviates from its own earlier history.

For each historical day d:
    recent window = d-7 ... d-1
    baseline window = d-67 ... d-8

Eight scalar historical states are computed causally inside the reference year.
The day-level states are used only as historical evidence; the final model-facing
output is participant-level tendency statistics.

Final subject-level features:
- short_bid_level_abs_z_median
- short_bid_level_abs_z_p90
- short_bid_level_high_state_rate
- short_adjustment_magnitude_abs_z_median
- short_structure_abs_z_median
- short_shape_shift_median
- short_break_rate
- short_ready_day_share
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    DAILY_SCALARS,
    DAILY_SHAPE_COLS,
    STRUCTURAL_STEMS,
    DEFAULT_SCALE_FLOORS,
    DEFAULT_Z_CLIP,
    SHORT_TENDENCY_FEATURES,
    ensure_dir,
)


def rolling_mad(series, window):
    return series.rolling(
        window=window,
        min_periods=1,
    ).apply(
        lambda x: (
            np.median(
                np.abs(
                    x - np.median(x)
                )
            )
            if len(x)
            else np.nan
        ),
        raw=True,
    )


def build_scalar_state(
    s,
    min_recent,
    min_long,
    scale_floor,
    z_clip,
):
    recent_source = s.shift(1)
    long_source = s.shift(8)

    recent_count = (
        recent_source
        .rolling(
            7,
            min_periods=1,
        )
        .count()
    )

    long_count = (
        long_source
        .rolling(
            60,
            min_periods=1,
        )
        .count()
    )

    recent = (
        recent_source
        .rolling(
            7,
            min_periods=1,
        )
        .median()
    )

    baseline = (
        long_source
        .rolling(
            60,
            min_periods=1,
        )
        .median()
    )

    scale = (
        1.4826
        * rolling_mad(
            long_source,
            60,
        )
    )

    eligible = (
        (recent_count >= min_recent)
        & (long_count >= min_long)
        & baseline.notna()
    )

    shift = (
        recent
        - baseline
    )

    z = pd.Series(
        np.nan,
        index=s.index,
        dtype=float,
    )

    break_flag = pd.Series(
        0,
        index=s.index,
        dtype=int,
    )

    positive_scale = (
        eligible
        & scale.notna()
        & (
            scale
            > float(scale_floor)
        )
    )

    z.loc[
        positive_scale
    ] = (
        shift.loc[
            positive_scale
        ]
        / scale.loc[
            positive_scale
        ]
    ).clip(
        -float(z_clip),
        float(z_clip),
    )

    practical_zero = (
        eligible
        & scale.notna()
        & (
            scale
            <= float(scale_floor)
        )
    )

    same = (
        practical_zero
        & (
            shift.abs()
            <= float(scale_floor)
        )
    )

    changed = (
        practical_zero
        & (
            shift.abs()
            > float(scale_floor)
        )
    )

    z.loc[
        same
    ] = 0.0

    break_flag.loc[
        changed
    ] = 1

    represented = (
        z.notna()
        | break_flag.eq(1)
    ).astype(int)

    return {
        "z": z,
        "break": break_flag,
        "represented": represented,
        "recent_count": recent_count,
        "long_count": long_count,
    }


def build_participant_history(
    pid,
    g,
    min_recent,
    min_long,
    min_recent_shape,
    min_long_shape,
    z_clip,
):
    g = (
        g.copy()
        .sort_values(
            "local_date"
        )
        .drop_duplicates(
            subset=["local_date"],
            keep="last",
        )
    )

    original_dates = pd.Index(
        g["local_date"],
        name="local_date",
    )

    full_dates = pd.date_range(
        original_dates.min(),
        original_dates.max(),
        freq="D",
        name="local_date",
    )

    cal = (
        g.set_index(
            "local_date"
        )
        .reindex(
            full_dates
        )
    )

    active = (
        cal.index.isin(
            original_dates
        )
    )

    daily_state = pd.DataFrame(
        index=cal.index
    )

    daily_state[
        "participant_id"
    ] = str(pid)

    represented_cols = []
    break_cols = []

    for stem, col in (
        DAILY_SCALARS.items()
    ):
        s = pd.to_numeric(
            cal[col],
            errors="coerce",
        )

        state = (
            build_scalar_state(
                s=s,
                min_recent=min_recent,
                min_long=min_long,
                scale_floor=DEFAULT_SCALE_FLOORS[
                    stem
                ],
                z_clip=z_clip,
            )
        )

        daily_state[
            f"hist_st_{stem}_z"
        ] = state[
            "z"
        ]

        daily_state[
            f"hist_break_{stem}"
        ] = state[
            "break"
        ]

        rcol = (
            f"_represented_{stem}"
        )

        daily_state[
            rcol
        ] = state[
            "represented"
        ]

        represented_cols.append(
            rcol
        )

        break_cols.append(
            f"hist_break_{stem}"
        )

    # Shape shift: recent 7-day prototype vs earlier 60-day prototype.
    shape = (
        cal[
            DAILY_SHAPE_COLS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    shape_valid = (
        shape.notna()
        .all(axis=1)
    )

    shape_clean = (
        shape.where(
            np.repeat(
                shape_valid
                .to_numpy()[:, None],
                len(
                    DAILY_SHAPE_COLS
                ),
                axis=1,
            )
        )
    )

    recent_shape_count = (
        shape_valid.astype(float)
        .shift(1)
        .rolling(
            7,
            min_periods=1,
        )
        .sum()
    )

    long_shape_count = (
        shape_valid.astype(float)
        .shift(8)
        .rolling(
            60,
            min_periods=1,
        )
        .sum()
    )

    recent_proto = (
        shape_clean.shift(1)
        .rolling(
            7,
            min_periods=1,
        )
        .median()
    )

    long_proto = (
        shape_clean.shift(8)
        .rolling(
            60,
            min_periods=1,
        )
        .median()
    )

    shape_shift = np.sqrt(
        (
            (
                recent_proto
                - long_proto
            ) ** 2
        )
        .mean(axis=1)
    )

    shape_eligible = (
        (
            recent_shape_count
            >= min_recent_shape
        )
        & (
            long_shape_count
            >= min_long_shape
        )
        & recent_proto.notna()
        .all(axis=1)
        & long_proto.notna()
        .all(axis=1)
    )

    daily_state[
        "hist_st_shape_shift"
    ] = (
        shape_shift.where(
            shape_eligible,
            np.nan,
        )
    )

    daily_state[
        "hist_break_any"
    ] = (
        daily_state[
            break_cols
        ]
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )

    daily_state[
        "hist_scalar_represented_count"
    ] = (
        daily_state[
            represented_cols
        ]
        .sum(axis=1)
        .astype(int)
    )

    daily_state[
        "hist_st_ready_flag"
    ] = (
        daily_state[
            "hist_scalar_represented_count"
        ] >= 6
    ).astype(int)

    daily_state = (
        daily_state.loc[
            active
        ]
        .drop(
            columns=represented_cols
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # Participant-level historical short-term tendency
    # ------------------------------------------------------------
    bid_z = pd.to_numeric(
        daily_state[
            "hist_st_bid_level_z"
        ],
        errors="coerce",
    ).dropna()

    adj_z = pd.to_numeric(
        daily_state[
            "hist_st_adjustment_magnitude_z"
        ],
        errors="coerce",
    ).dropna()

    structure_cols = [
        f"hist_st_{stem}_z"
        for stem in (
            STRUCTURAL_STEMS
        )
    ]

    structure_abs = (
        daily_state[
            structure_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .abs()
        .mean(
            axis=1,
            skipna=True,
        )
    )

    shape_series = pd.to_numeric(
        daily_state[
            "hist_st_shape_shift"
        ],
        errors="coerce",
    ).dropna()

    ready = pd.to_numeric(
        daily_state[
            "hist_st_ready_flag"
        ],
        errors="coerce",
    ).fillna(0)

    break_any = pd.to_numeric(
        daily_state[
            "hist_break_any"
        ],
        errors="coerce",
    ).fillna(0)

    summary = {
        "participant_id": str(pid),
        "historical_days": int(
            len(daily_state)
        ),
        "short_bid_level_abs_z_median": (
            float(
                bid_z.abs()
                .median()
            )
            if len(bid_z)
            else np.nan
        ),
        "short_bid_level_abs_z_p90": (
            float(
                bid_z.abs()
                .quantile(0.90)
            )
            if len(bid_z)
            else np.nan
        ),
        # Historical frequency of a meaningfully high recent price state.
        "short_bid_level_high_state_rate": (
            float(
                (bid_z > 1.0)
                .mean()
            )
            if len(bid_z)
            else np.nan
        ),
        "short_adjustment_magnitude_abs_z_median": (
            float(
                adj_z.abs()
                .median()
            )
            if len(adj_z)
            else np.nan
        ),
        "short_structure_abs_z_median": (
            float(
                structure_abs
                .dropna()
                .median()
            )
            if structure_abs.notna()
            .any()
            else np.nan
        ),
        "short_shape_shift_median": (
            float(
                shape_series.median()
            )
            if len(shape_series)
            else np.nan
        ),
        "short_break_rate": (
            float(
                break_any.mean()
            )
            if len(break_any)
            else np.nan
        ),
        "short_ready_day_share": (
            float(
                ready.mean()
            )
            if len(ready)
            else np.nan
        ),
    }

    return (
        daily_state,
        summary,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    p.add_argument(
        "--daily-root",
        default=(
            "data/processed/"
            "final_clean/daily"
        ),
    )

    p.add_argument(
        "--out-root",
        default=(
            "data/processed/"
            "final_clean/short_term_history"
        ),
    )

    p.add_argument(
        "--min-recent",
        type=int,
        default=3,
    )

    p.add_argument(
        "--min-long",
        type=int,
        default=20,
    )

    p.add_argument(
        "--min-recent-shape",
        type=int,
        default=2,
    )

    p.add_argument(
        "--min-long-shape",
        type=int,
        default=10,
    )

    p.add_argument(
        "--z-clip",
        type=float,
        default=DEFAULT_Z_CLIP,
    )

    args = p.parse_args()

    src = (
        Path(args.daily_root)
        / str(args.year)
        / f"daily_strategy_core_{args.year}.csv"
    )

    if not src.exists():
        raise FileNotFoundError(
            src
        )

    out_dir = ensure_dir(
        Path(args.out_root)
        / str(args.year)
    )

    daily_state_file = (
        out_dir
        / f"historical_short_term_state_{args.year}.csv"
    )

    summary_file = (
        out_dir
        / f"historical_short_term_tendency_{args.year}.csv"
    )

    df = pd.read_csv(
        src,
        low_memory=False,
    )

    df[
        "participant_id"
    ] = (
        df[
            "participant_id"
        ]
        .astype(str)
    )

    df[
        "local_date"
    ] = (
        pd.to_datetime(
            df[
                "local_date"
            ]
        )
        .dt.normalize()
    )

    daily_parts = []
    summary_rows = []

    groups = list(
        df.groupby(
            "participant_id",
            sort=False,
        )
    )

    for idx, (
        pid,
        g,
    ) in enumerate(
        groups,
        start=1,
    ):
        daily_state, summary = (
            build_participant_history(
                pid=pid,
                g=g,
                min_recent=args.min_recent,
                min_long=args.min_long,
                min_recent_shape=args.min_recent_shape,
                min_long_shape=args.min_long_shape,
                z_clip=args.z_clip,
            )
        )

        daily_parts.append(
            daily_state
        )

        summary_rows.append(
            summary
        )

        if (
            idx == 1
            or idx % 50 == 0
            or idx == len(groups)
        ):
            print(
                f"[03] participants "
                f"{idx:,}/{len(groups):,}",
                flush=True,
            )

    daily_out = pd.concat(
        daily_parts,
        ignore_index=True,
    )

    summary_out = pd.DataFrame(
        summary_rows
    )

    summary_out[
        "short_nonmissing_count"
    ] = (
        summary_out[
            SHORT_TENDENCY_FEATURES
        ]
        .notna()
        .sum(axis=1)
    )

    summary_out[
        "short_tendency_ready_flag"
    ] = (
        summary_out[
            "short_nonmissing_count"
        ] >= 6
    ).astype(int)

    daily_out.to_csv(
        daily_state_file,
        index=False,
        encoding="utf-8-sig",
    )

    summary_out.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print(
        "Historical short-term tendency build complete"
    )
    print("=" * 72)
    print(
        "This output is a historical participant trait; "
        "it is not an online target-year lag feature."
    )
    print(
        f"Participant-days: "
        f"{len(daily_out):,}"
    )
    print(
        f"Participants: "
        f"{len(summary_out):,}"
    )
    print(
        f"Ready participants: "
        f"{int(summary_out['short_tendency_ready_flag'].sum()):,}"
    )
    print(
        f"Daily audit: {daily_state_file}"
    )
    print(
        f"Participant summary: {summary_file}"
    )


if __name__ == "__main__":
    main()
