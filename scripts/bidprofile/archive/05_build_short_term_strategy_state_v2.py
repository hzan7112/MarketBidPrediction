#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_build_short_term_strategy_state.py
Version: 2026-09-17-v2

Purpose
-------
Build a universal, causal short-term bidding-strategy state from the daily
strategy summary.

IMPORTANT: no future leakage
----------------------------
For state date d:

Recent window:
    [d - SHORT_DAYS, d - 1]

Long-term reference window:
    [d - SHORT_DAYS - LONG_DAYS, d - SHORT_DAYS - 1]

The two windows are disjoint and contain only dates strictly before d.

Default:
    SHORT_DAYS = 7
    LONG_DAYS  = 60
    MIN_SHORT  = 3 valid days
    MIN_LONG   = 20 valid days

This script intentionally does NOT use the full-year static profile generated
by 04 as the baseline for historical dates, because doing so would leak future
information into prediction-time short-term states.

Run:
    python scripts/05_build_short_term_strategy_state.py --year 2025

Input:
    data/processed/daily_strategy_summary/2025/
        daily_strategy_summary_2025.csv

Output:
    data/processed/short_term_strategy_state/2025/
        short_term_strategy_state_2025.csv
    results/05_short_term_state/summary_2025.txt

State construction
------------------
For each daily strategy feature f:

recent_f(d)   = median of f in recent window
baseline_f(d) = median of f in prior long window
scale_f(d)    = 1.4826 * MAD of f in prior long window

raw_shift_f(d) = recent_f - baseline_f

robust short-term deviation:
    z_f = raw_shift_f / scale_f

When scale_f == 0:
- if recent_f == baseline_f, z_f = 0;
- if recent_f != baseline_f, z_f is left NaN and
  zero_scale_change_flag = 1.

This avoids generating arbitrarily huge z-scores for participants whose
historical strategy was perfectly constant.

Curve-shape state
-----------------
For normalized daily shape vectors V_d:

recent shape prototype   = component-wise median over recent window
baseline shape prototype = component-wise median over prior long window

st_shape_prototype_shift
    = RMSE(recent prototype, baseline prototype)

This directly measures recent structural movement in the bid curve while
remaining independent of absolute price level.

The output is a state available BEFORE the start of `state_date`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2026-09-17-v2"
ROBUST_SCALE = 1.4826
EPS = 1e-12

DEFAULT_SHORT_DAYS = 7
DEFAULT_LONG_DAYS = 60
DEFAULT_MIN_SHORT = 3
DEFAULT_MIN_LONG = 20

N_SHAPE_GRID = 21
DAILY_SHAPE_COLS = [f"daily_shape_v{i:02d}" for i in range(N_SHAPE_GRID)]


# History-identifiable daily strategy variables retained for the short-term state.
FEATURES = {
    # Price / adjustment
    "daily_bid_level": "bid_level",
    "daily_self_adjustment_bias": "adjustment_bias",
    "daily_self_adjustment_magnitude": "adjustment_magnitude",
    "daily_adjacent_level_change_magnitude": "adjacent_level_change",

    # Quantity allocation
    "daily_quantity_hhi": "quantity_hhi",

    # Curve organization
    "daily_effective_segment_count": "effective_segment_count",
    "daily_flat_curve_rate": "flat_curve_rate",
    "daily_tail_uplift_ratio": "tail_uplift_ratio",
    "daily_curve_bend_ratio": "curve_bend_ratio",

    # Shape/strategy adjustment
    "daily_same_slot_shape_change": "same_slot_shape_change",
    "daily_adjacent_shape_change": "adjacent_shape_change",
    "daily_curve_mode_switch_rate": "curve_mode_switch_rate",
    "daily_flat_curve_switch_rate": "flat_curve_switch_rate",
}


def robust_stats(values, min_count):
    """Return count, median, robust scale for finite values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < min_count:
        return n, np.nan, np.nan

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = ROBUST_SCALE * mad
    return n, med, scale


def median_with_count(values, min_count):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < min_count:
        return n, np.nan
    return n, float(np.median(x))


def shape_prototype(shape_matrix, min_days):
    """
    Component-wise median prototype.
    Only rows with all shape components finite are used.
    """
    x = np.asarray(shape_matrix, dtype=float)

    if x.ndim != 2 or x.shape[1] != N_SHAPE_GRID:
        return 0, np.full(N_SHAPE_GRID, np.nan)

    row_ok = np.isfinite(x).all(axis=1)
    x = x[row_ok]
    n = len(x)

    if n < min_days:
        return n, np.full(N_SHAPE_GRID, np.nan)

    return n, np.median(x, axis=0)


def build_participant_states(
    g,
    short_days,
    long_days,
    min_short,
    min_long,
    min_short_shape,
    min_long_shape,
):
    """
    Build causal states for one participant x market_product.
    """
    g = g.sort_values("local_date").copy()
    g["local_date"] = pd.to_datetime(g["local_date"], errors="coerce")
    g = g[g["local_date"].notna()].copy()

    if g.empty:
        return pd.DataFrame()

    dates = g["local_date"].to_numpy(dtype="datetime64[D]")
    date_int = dates.astype("int64")
    n = len(g)

    feature_arrays = {
        src: pd.to_numeric(g[src], errors="coerce").to_numpy(float)
        for src in FEATURES
    }
    shape_array = g[DAILY_SHAPE_COLS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)

    rows = []

    for t in range(n):
        d = date_int[t]

        recent_start_day = d - short_days
        recent_end_day = d - 1

        long_start_day = d - short_days - long_days
        long_end_day = d - short_days - 1

        # searchsorted works because local_date is sorted.
        r0 = np.searchsorted(date_int, recent_start_day, side="left")
        r1 = np.searchsorted(date_int, recent_end_day, side="right")

        l0 = np.searchsorted(date_int, long_start_day, side="left")
        l1 = np.searchsorted(date_int, long_end_day, side="right")

        out = {
            "participant_id": g["participant_id"].iloc[t],
            "market_product": g["market_product"].iloc[t],
            "state_date": pd.Timestamp(g["local_date"].iloc[t]).date(),

            "short_window_days": short_days,
            "long_window_days": long_days,

            "recent_calendar_start": (
                pd.Timestamp("1970-01-01")
                + pd.Timedelta(days=int(recent_start_day))
            ).date(),
            "recent_calendar_end": (
                pd.Timestamp("1970-01-01")
                + pd.Timedelta(days=int(recent_end_day))
            ).date(),
            "long_calendar_start": (
                pd.Timestamp("1970-01-01")
                + pd.Timedelta(days=int(long_start_day))
            ).date(),
            "long_calendar_end": (
                pd.Timestamp("1970-01-01")
                + pd.Timedelta(days=int(long_end_day))
            ).date(),

            # generic activity-day counts, independent of a particular feature
            "recent_active_days": int(max(0, r1 - r0)),
            "long_active_days": int(max(0, l1 - l0)),
        }

        ready_feature_count = 0
        zero_scale_changed_count = 0

        for src, stem in FEATURES.items():
            arr = feature_arrays[src]

            short_count, recent = median_with_count(
                arr[r0:r1],
                min_short,
            )
            long_count, baseline, scale = robust_stats(
                arr[l0:l1],
                min_long,
            )

            raw_shift = (
                recent - baseline
                if np.isfinite(recent) and np.isfinite(baseline)
                else np.nan
            )

            z = np.nan
            zero_scale_change_flag = 0

            if np.isfinite(raw_shift) and np.isfinite(scale):
                if scale > EPS:
                    z = raw_shift / scale
                    ready_feature_count += 1
                else:
                    if abs(raw_shift) <= EPS:
                        z = 0.0
                        ready_feature_count += 1
                    else:
                        zero_scale_change_flag = 1
                        zero_scale_changed_count += 1

            out[f"{stem}_recent_days"] = int(short_count)
            out[f"{stem}_long_days"] = int(long_count)
            out[f"{stem}_recent"] = recent
            out[f"{stem}_baseline"] = baseline
            out[f"{stem}_baseline_scale"] = scale
            out[f"st_{stem}_raw_shift"] = raw_shift
            out[f"st_{stem}_z"] = z
            out[f"{stem}_zero_scale_change_flag"] = (
                zero_scale_change_flag
            )

        # Recent-vs-long-term curve shape prototype shift.
        recent_shape_days, recent_shape = shape_prototype(
            shape_array[r0:r1],
            min_short_shape,
        )
        long_shape_days, long_shape = shape_prototype(
            shape_array[l0:l1],
            min_long_shape,
        )

        shape_shift = np.nan
        if (
            np.isfinite(recent_shape).all()
            and np.isfinite(long_shape).all()
        ):
            diff = recent_shape - long_shape
            shape_shift = float(np.sqrt(np.mean(diff * diff)))

        out["recent_shape_days"] = int(recent_shape_days)
        out["long_shape_days"] = int(long_shape_days)
        out["st_shape_prototype_shift"] = shape_shift

        # The recent shape prototype itself can be useful to downstream models.
        for j in range(N_SHAPE_GRID):
            out[f"st_shape_v{j:02d}"] = (
                float(recent_shape[j])
                if np.isfinite(recent_shape[j])
                else np.nan
            )

        out["state_ready_feature_count"] = ready_feature_count
        out["zero_scale_changed_feature_count"] = (
            zero_scale_changed_count
        )

        # Overall readiness:
        # at least half of scalar state dimensions are valid.
        out["short_term_state_ready_flag"] = int(
            ready_feature_count >= int(np.ceil(len(FEATURES) / 2))
        )

        rows.append(out)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--daily-root",
        default="data/processed/daily_strategy_summary",
    )
    p.add_argument(
        "--output-root",
        default="data/processed/short_term_strategy_state",
    )
    p.add_argument(
        "--results-root",
        default="results/05_short_term_state",
    )

    p.add_argument(
        "--short-days",
        type=int,
        default=DEFAULT_SHORT_DAYS,
    )
    p.add_argument(
        "--long-days",
        type=int,
        default=DEFAULT_LONG_DAYS,
    )
    p.add_argument(
        "--min-short",
        type=int,
        default=DEFAULT_MIN_SHORT,
    )
    p.add_argument(
        "--min-long",
        type=int,
        default=DEFAULT_MIN_LONG,
    )
    p.add_argument(
        "--min-short-shape",
        type=int,
        default=2,
    )
    p.add_argument(
        "--min-long-shape",
        type=int,
        default=10,
    )

    args = p.parse_args()

    if args.short_days < 1 or args.long_days < 1:
        raise ValueError("short-days and long-days must be >= 1.")
    if not (1 <= args.min_short <= args.short_days):
        raise ValueError("min-short must be within [1, short-days].")
    if not (1 <= args.min_long <= args.long_days):
        raise ValueError("min-long must be within [1, long-days].")

    daily_file = (
        Path(args.daily_root)
        / str(args.year)
        / f"daily_strategy_summary_{args.year}.csv"
    )
    if not daily_file.exists():
        raise FileNotFoundError(daily_file)

    out_dir = Path(args.output_root) / str(args.year)
    results_dir = Path(args.results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(daily_file, nrows=0)
    required = (
        ["participant_id", "market_product", "local_date"]
        + list(FEATURES.keys())
        + DAILY_SHAPE_COLS
    )
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise ValueError(
            f"Daily strategy summary is missing required fields: {missing}"
        )

    print(f"[read] {daily_file}")
    daily = pd.read_csv(
        daily_file,
        usecols=required,
        low_memory=False,
    )

    daily["local_date"] = pd.to_datetime(
        daily["local_date"], errors="coerce"
    )

    duplicate_days = int(
        daily.duplicated(
            ["participant_id", "market_product", "local_date"]
        ).sum()
    )
    if duplicate_days:
        raise ValueError(
            f"Daily input contains {duplicate_days} duplicate participant-days."
        )

    parts = []
    groups = daily.groupby(
        ["participant_id", "market_product"],
        sort=False,
    )

    total_groups = groups.ngroups
    for idx, ((pid, product), g) in enumerate(groups, start=1):
        part = build_participant_states(
            g,
            args.short_days,
            args.long_days,
            args.min_short,
            args.min_long,
            args.min_short_shape,
            args.min_long_shape,
        )
        parts.append(part)

        if idx == 1 or idx % 100 == 0 or idx == total_groups:
            print(
                f"[state] {idx}/{total_groups} participants "
                f"({pid}, {product})"
            )

    states = pd.concat(parts, ignore_index=True)

    output_file = (
        out_dir / f"short_term_strategy_state_{args.year}.csv"
    )
    states = states.sort_values(
        ["participant_id", "market_product", "state_date"]
    )
    states.to_csv(output_file, index=False)

    ready = states["short_term_state_ready_flag"].eq(1)

    scalar_z_cols = [f"st_{stem}_z" for stem in FEATURES.values()]
    scalar_raw_cols = [
        f"st_{stem}_raw_shift" for stem in FEATURES.values()
    ]

    lines = [
        f"Universal short-term strategy state summary - {args.year}",
        f"Version: {VERSION}",
        "=" * 72,
        f"Participant-day states: {len(states):,}",
        f"Participants: {states['participant_id'].nunique():,}",
        f"Market products: {states['market_product'].nunique():,}",
        f"Short recent window: {args.short_days} calendar days",
        f"Long reference window: {args.long_days} calendar days",
        f"Minimum recent valid days per scalar feature: {args.min_short}",
        f"Minimum long valid days per scalar feature: {args.min_long}",
        f"Ready short-term states: {ready.sum():,} "
        f"({ready.mean()*100:.2f}%)",
        f"States with recent-vs-long shape shift available: "
        f"{states['st_shape_prototype_shift'].notna().sum():,}",
        f"States containing at least one zero-scale strategy break: "
        f"{states['zero_scale_changed_feature_count'].gt(0).sum():,}",
        "",
        "Causal window definition for state date d:",
        f"  recent: [d-{args.short_days}, d-1]",
        f"  baseline: "
        f"[d-{args.short_days + args.long_days}, "
        f"d-{args.short_days + 1}]",
        "",
        "Scalar short-term state z-features:",
    ]
    lines += [f"  {c}" for c in scalar_z_cols]
    lines += [
        "",
        "Raw shifts are also retained for interpretability:",
    ]
    lines += [f"  {c}" for c in scalar_raw_cols]
    lines += [
        "",
        "Curve-shape state:",
        "  st_shape_prototype_shift",
        "  st_shape_v00 ... st_shape_v20",
        "",
        "No future date, same-day target information, market outcome,",
        "or full-year long-term profile is used to construct historical states.",
        f"Output: {output_file.resolve()}",
    ]

    summary_file = results_dir / f"summary_{args.year}.txt"
    summary_file.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
