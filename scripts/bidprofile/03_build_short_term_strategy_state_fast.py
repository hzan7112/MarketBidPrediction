#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_build_short_term_strategy_state.py

Fast calendar-window implementation.

Logic is unchanged:
- recent window: [d-7, d-1]
- long window:   [d-67, d-8]
- scalar minimum days: recent >= 3, long >= 20
- shape minimum days: recent >= 2, long >= 10
- robust scale = 1.4826 * MAD
- if long MAD == 0 and recent shift != 0 -> Strategy Break
- if long MAD == 0 and recent shift == 0 -> z = 0

The speedup comes from:
1. reindexing each participant onto a daily calendar;
2. using vectorized rolling windows instead of rescanning the participant
   history for every participant-day.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    DAILY_SCALARS,
    DAILY_SHAPE_COLS,
    ensure_dir,
)

MIN_RECENT = 3
MIN_LONG = 20
MIN_RECENT_SHAPE = 2
MIN_LONG_SHAPE = 10
EPS = 1e-12


def rolling_mad(series: pd.Series, window: int) -> pd.Series:
    """Exact rolling MAD using raw NumPy arrays."""
    return series.rolling(
        window=window,
        min_periods=1,
    ).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
        if len(x) else np.nan,
        raw=True,
    )


def process_participant(pid, g):
    g = g.copy()
    g["local_date"] = pd.to_datetime(g["local_date"]).dt.normalize()
    g = g.sort_values("local_date").drop_duplicates(
        subset=["local_date"],
        keep="last",
    )

    if g.empty:
        return pd.DataFrame()

    original_dates = pd.Index(g["local_date"], name="local_date")

    full_dates = pd.date_range(
        original_dates.min(),
        original_dates.max(),
        freq="D",
        name="local_date",
    )

    cal = (
        g.set_index("local_date")
        .reindex(full_dates)
    )

    active_flag = cal.index.isin(original_dates)

    out = pd.DataFrame(index=cal.index)
    out["participant_id"] = pid

    # Number of actual active days in the calendar windows.
    active = pd.Series(
        active_flag.astype(float),
        index=cal.index,
    )
    out["recent_active_days"] = (
        active.shift(1)
        .rolling(7, min_periods=1)
        .sum()
        .fillna(0)
        .astype(int)
    )
    out["long_active_days"] = (
        active.shift(8)
        .rolling(60, min_periods=1)
        .sum()
        .fillna(0)
        .astype(int)
    )

    represented_cols = []
    break_cols = []

    # ------------------------------------------------------------
    # 8 scalar short-term states
    # ------------------------------------------------------------
    for stem, col in DAILY_SCALARS.items():
        s = pd.to_numeric(
            cal[col],
            errors="coerce",
        )

        # Previous 7 calendar days: d-7 ... d-1
        recent_source = s.shift(1)
        recent_count = recent_source.rolling(
            7,
            min_periods=1,
        ).count()
        recent_median = recent_source.rolling(
            7,
            min_periods=1,
        ).median()

        # Earlier 60 calendar days: d-67 ... d-8
        long_source = s.shift(8)
        long_count = long_source.rolling(
            60,
            min_periods=1,
        ).count()
        long_median = long_source.rolling(
            60,
            min_periods=1,
        ).median()

        long_mad = rolling_mad(
            long_source,
            60,
        )
        long_scale = 1.4826 * long_mad

        eligible = (
            (recent_count >= MIN_RECENT)
            & (long_count >= MIN_LONG)
        )

        shift = recent_median - long_median

        z = pd.Series(
            np.nan,
            index=cal.index,
            dtype=float,
        )
        br = pd.Series(
            0,
            index=cal.index,
            dtype=int,
        )

        scale_positive = eligible & (long_scale > EPS)
        z.loc[scale_positive] = (
            shift.loc[scale_positive]
            / long_scale.loc[scale_positive]
        )

        zero_scale = eligible & (long_scale <= EPS)
        same = zero_scale & (shift.abs() <= EPS)
        changed = zero_scale & (shift.abs() > EPS)

        z.loc[same] = 0.0
        br.loc[changed] = 1

        zcol = f"st_{stem}_z"
        bcol = f"break_{stem}"

        out[zcol] = z
        out[bcol] = br

        represented = (
            z.notna() | br.eq(1)
        ).astype(int)

        rcol = f"_represented_{stem}"
        out[rcol] = represented
        represented_cols.append(rcol)
        break_cols.append(bcol)

    # ------------------------------------------------------------
    # Shape shift
    # ------------------------------------------------------------
    shape = cal[DAILY_SHAPE_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Only a day with all 21 coordinates is a valid shape day.
    shape_valid = shape.notna().all(axis=1)
    shape_clean = shape.where(
        np.repeat(
            shape_valid.to_numpy()[:, None],
            len(DAILY_SHAPE_COLS),
            axis=1,
        )
    )

    recent_shape_source = shape_clean.shift(1)
    long_shape_source = shape_clean.shift(8)

    recent_shape_count = (
        shape_valid.astype(float)
        .shift(1)
        .rolling(7, min_periods=1)
        .sum()
    )
    long_shape_count = (
        shape_valid.astype(float)
        .shift(8)
        .rolling(60, min_periods=1)
        .sum()
    )

    recent_proto = recent_shape_source.rolling(
        7,
        min_periods=1,
    ).median()
    long_proto = long_shape_source.rolling(
        60,
        min_periods=1,
    ).median()

    shape_shift = np.sqrt(
        (
            (recent_proto - long_proto) ** 2
        ).mean(axis=1)
    )

    shape_eligible = (
        (recent_shape_count >= MIN_RECENT_SHAPE)
        & (long_shape_count >= MIN_LONG_SHAPE)
    )

    out["st_shape_shift"] = shape_shift.where(
        shape_eligible,
        np.nan,
    )

    # ------------------------------------------------------------
    # Break summary / readiness
    # ------------------------------------------------------------
    out["strategy_break_count"] = out[
        break_cols
    ].sum(axis=1).astype(int)

    out["strategy_break_any"] = (
        out["strategy_break_count"] > 0
    ).astype(int)

    def build_break_types(row):
        names = []
        for stem in DAILY_SCALARS:
            if row[f"break_{stem}"] == 1:
                names.append(stem)
        return "|".join(names)

    # Only called once per final active day, not over all internal windows.
    final = out.loc[active_flag].copy()
    final["strategy_break_types"] = final.apply(
        build_break_types,
        axis=1,
    )

    final["st_scalar_represented_count"] = final[
        represented_cols
    ].sum(axis=1).astype(int)

    final["st_ready_flag"] = (
        final["st_scalar_represented_count"] >= 6
    ).astype(int)

    final = final.drop(
        columns=represented_cols
    )

    final = final.reset_index()
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--daily-root",
        default="data/processed/final_clean/daily",
    )
    p.add_argument(
        "--out-root",
        default="data/processed/final_clean/short_term",
    )
    args = p.parse_args()

    src = (
        Path(args.daily_root)
        / str(args.year)
        / f"daily_strategy_core_{args.year}.csv"
    )
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = ensure_dir(
        Path(args.out_root) / str(args.year)
    )
    out_file = (
        out_dir
        / f"short_term_strategy_state_{args.year}.csv"
    )

    df = pd.read_csv(
        src,
        low_memory=False,
    )
    df["local_date"] = pd.to_datetime(
        df["local_date"]
    ).dt.normalize()

    groups = list(
        df.groupby(
            "participant_id",
            sort=False,
        )
    )

    results = []

    total = len(groups)

    for idx, (pid, g) in enumerate(
        groups,
        start=1,
    ):
        result = process_participant(
            pid,
            g,
        )
        if not result.empty:
            results.append(result)

        if (
            idx == 1
            or idx % 50 == 0
            or idx == total
        ):
            print(
                f"[03] participants "
                f"{idx:,}/{total:,}",
                flush=True,
            )

    result = pd.concat(
        results,
        ignore_index=True,
    )

    # Keep a clean and stable column order.
    ordered = [
        "participant_id",
        "local_date",
        "recent_active_days",
        "long_active_days",
    ]

    for stem in DAILY_SCALARS:
        ordered += [
            f"st_{stem}_z",
            f"break_{stem}",
        ]

    ordered += [
        "st_shape_shift",
        "strategy_break_count",
        "strategy_break_any",
        "strategy_break_types",
        "st_scalar_represented_count",
        "st_ready_flag",
    ]

    result = result[ordered]

    result.to_csv(
        out_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Participant-days: {len(result):,}"
    )
    print(
        f"ST ready: "
        f"{result['st_ready_flag'].sum():,}"
    )
    print(
        f"Break days: "
        f"{result['strategy_break_any'].sum():,}"
    )
    print(
        f"Done: {out_file}"
    )


if __name__ == "__main__":
    main()
