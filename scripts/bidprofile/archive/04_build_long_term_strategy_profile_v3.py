#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_build_long_term_strategy_profile_v3.py
Version: 2026-09-17-v3

Robust long-term strategy profile construction.

v3 removes all assumptions about how source files align with local calendar
days. All interval rows are first partitioned by `local_date`. Therefore one
participant-day is aggregated exactly once even when the same local date is
split across any number of source files.

Workflow
--------
1. Read each interval-atom file and matching temporal-behavior file.
2. Merge by interval_index + source_file.
3. Partition merged interval rows into temporary local-date buckets.
4. For each local date:
   - combine rows from all source files;
   - remove exact duplicate physical intervals using
     participant_id + market_product + timestamp_utc;
   - compute participant-day statistics once.
5. Build long-term participant profiles from equal-weight daily summaries.

Run:
    python scripts/04_build_long_term_strategy_profile_v3.py --year 2025
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2026-09-17-v3"
ROBUST_SCALE = 1.4826
N_SHAPE_GRID = 21
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(N_SHAPE_GRID)]

ATOM_USECOLS = [
    "interval_index",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "market_product",
    "source_file",
    "curve_mode",
    "bid_level",
    "effective_segment_count",
    "quantity_hhi",
    "tail_uplift_ratio",
    "curve_bend_ratio",
    "flat_curve_flag",
    "shape_defined_flag",
] + SHAPE_COLS

TEMP_USECOLS = [
    "interval_index",
    "source_file",
    "baseline_ready_flag",
    "self_bid_level_residual",
    "same_slot_level_change",
    "same_slot_shape_distance",
    "adjacent_interval_flag",
    "adjacent_bid_level_change",
    "adjacent_shape_distance",
    "curve_mode_switch_flag",
    "flat_curve_switch_flag",
]


def robust_scale(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    med = x.median()
    return ROBUST_SCALE * (x - med).abs().median()


def discover_pairs(atom_dir, temp_dir, year):
    atom_files = sorted(atom_dir.glob(f"interval_strategy_atoms_{year}_*.csv"))
    if not atom_files:
        raise FileNotFoundError(f"No atom files in {atom_dir.resolve()}")

    pairs = []
    for af in atom_files:
        suffix = af.stem.replace("interval_strategy_atoms_", "")
        tf = temp_dir / f"temporal_strategy_behavior_{suffix}.csv"
        if not tf.exists():
            raise FileNotFoundError(tf)
        pairs.append((af, tf))
    return pairs


def load_pair(atom_file, temp_file):
    atoms = pd.read_csv(
        atom_file,
        usecols=ATOM_USECOLS,
        low_memory=False,
    )
    temp = pd.read_csv(
        temp_file,
        usecols=TEMP_USECOLS,
        low_memory=False,
    )

    df = atoms.merge(
        temp,
        on=["interval_index", "source_file"],
        how="left",
        validate="one_to_one",
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], errors="coerce", utc=True
    )
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="coerce"
    )
    df["local_date"] = df["timestamp_local"].dt.strftime("%Y-%m-%d")

    df["_abs_self_residual"] = pd.to_numeric(
        df["self_bid_level_residual"], errors="coerce"
    ).abs()
    df["_abs_adjacent_level_change"] = pd.to_numeric(
        df["adjacent_bid_level_change"], errors="coerce"
    ).abs()

    return df


def append_date_bucket(df, bucket_file):
    bucket_file.parent.mkdir(parents=True, exist_ok=True)
    exists = bucket_file.exists()
    df.to_csv(
        bucket_file,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
    )


def stage_by_local_date(pairs, stage_dir):
    """
    Partition all merged interval rows into local-date files.
    This is source-file-boundary agnostic.
    """
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    date_sources = defaultdict(set)
    source_coverage = []
    total_rows = 0

    for atom_file, temp_file in pairs:
        print(f"[stage] {atom_file.name}")
        df = load_pair(atom_file, temp_file)
        total_rows += len(df)

        valid_dates = sorted(df["local_date"].dropna().unique().tolist())
        source_coverage.append({
            "source_file": atom_file.name,
            "rows": len(df),
            "local_date_min": valid_dates[0] if valid_dates else None,
            "local_date_max": valid_dates[-1] if valid_dates else None,
            "distinct_local_dates": len(valid_dates),
        })

        for local_date, part in df.groupby("local_date", sort=False):
            if pd.isna(local_date):
                continue
            date_sources[str(local_date)].add(atom_file.name)
            bucket = stage_dir / f"{local_date}.csv"
            append_date_bucket(part, bucket)

        print(
            f"  rows={len(df):,}, "
            f"local_dates={len(valid_dates)}, "
            f"range={valid_dates[0] if valid_dates else 'NA'}"
            f"..{valid_dates[-1] if valid_dates else 'NA'}"
        )

    return (
        total_rows,
        date_sources,
        pd.DataFrame(source_coverage),
    )


def aggregate_one_local_date(df):
    keys = ["participant_id", "market_product", "local_date"]

    g = df.groupby(keys, sort=False, dropna=False)

    daily = g.agg(
        intervals_observed=("interval_index", "size"),
        valid_bid_intervals=("bid_level", "count"),
        baseline_ready_intervals=("baseline_ready_flag", "sum"),
        shape_defined_intervals=("shape_defined_flag", "sum"),

        daily_bid_level=("bid_level", "median"),
        daily_effective_segment_count=("effective_segment_count", "median"),
        daily_quantity_hhi=("quantity_hhi", "median"),
        daily_tail_uplift_ratio=("tail_uplift_ratio", "median"),
        daily_curve_bend_ratio=("curve_bend_ratio", "median"),
        daily_flat_curve_rate=("flat_curve_flag", "mean"),

        daily_self_adjustment_bias=("self_bid_level_residual", "median"),
        daily_self_adjustment_magnitude=("_abs_self_residual", "median"),
        daily_adjacent_level_change_magnitude=(
            "_abs_adjacent_level_change", "median"
        ),

        daily_same_slot_shape_change=("same_slot_shape_distance", "median"),
        daily_adjacent_shape_change=("adjacent_shape_distance", "median"),

        daily_curve_mode_switch_rate=("curve_mode_switch_flag", "mean"),
        daily_flat_curve_switch_rate=("flat_curve_switch_flag", "mean"),
    ).reset_index()

    adj_p90 = (
        g["_abs_self_residual"]
        .quantile(0.90)
        .rename("daily_self_adjustment_p90")
        .reset_index()
    )
    shape_p90 = (
        g["adjacent_shape_distance"]
        .quantile(0.90)
        .rename("daily_adjacent_shape_change_p90")
        .reset_index()
    )

    daily = daily.merge(adj_p90, on=keys, how="left")
    daily = daily.merge(shape_p90, on=keys, how="left")

    shape_df = df.loc[
        df["shape_defined_flag"].fillna(0).eq(1),
        keys + SHAPE_COLS,
    ]

    if len(shape_df):
        daily_shape = (
            shape_df.groupby(keys, sort=False)[SHAPE_COLS]
            .median()
            .reset_index()
            .rename(columns={c: f"daily_{c}" for c in SHAPE_COLS})
        )
        daily = daily.merge(daily_shape, on=keys, how="left")
    else:
        for c in SHAPE_COLS:
            daily[f"daily_{c}"] = np.nan

    denom = daily["intervals_observed"].replace(0, np.nan)
    daily["daily_valid_bid_rate"] = daily["valid_bid_intervals"] / denom
    daily["daily_baseline_ready_rate"] = (
        daily["baseline_ready_intervals"] / denom
    )
    daily["daily_shape_defined_rate"] = (
        daily["shape_defined_intervals"] / denom
    )

    return daily


def build_daily_from_stage(stage_dir):
    daily_parts = []
    duplicate_intervals_removed = 0
    bucket_stats = []

    bucket_files = sorted(stage_dir.glob("*.csv"))
    if not bucket_files:
        raise ValueError("No local-date staging files were produced.")

    for i, bucket_file in enumerate(bucket_files, start=1):
        df = pd.read_csv(bucket_file, low_memory=False)

        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"], errors="coerce", utc=True
        )

        before = len(df)
        dedup_key = [
            "participant_id",
            "market_product",
            "timestamp_utc",
        ]

        df = (
            df.sort_values(
                dedup_key + ["source_file", "interval_index"]
            )
            .drop_duplicates(subset=dedup_key, keep="last")
            .reset_index(drop=True)
        )

        removed = before - len(df)
        duplicate_intervals_removed += removed

        daily = aggregate_one_local_date(df)
        daily_parts.append(daily)

        bucket_stats.append({
            "local_date": bucket_file.stem,
            "raw_interval_rows": before,
            "deduplicated_interval_rows": len(df),
            "duplicate_intervals_removed": removed,
            "participant_days": len(daily),
        })

        if i == 1 or i % 30 == 0 or i == len(bucket_files):
            print(
                f"[daily] {i}/{len(bucket_files)} "
                f"{bucket_file.stem}: intervals={len(df):,}, "
                f"participant-days={len(daily):,}"
            )

    daily = pd.concat(daily_parts, ignore_index=True)

    dup_days = int(
        daily.duplicated(
            ["participant_id", "market_product", "local_date"]
        ).sum()
    )
    if dup_days:
        raise ValueError(
            f"Internal error: {dup_days} duplicate participant-days remain "
            "after local-date partitioning."
        )

    return (
        daily,
        duplicate_intervals_removed,
        pd.DataFrame(bucket_stats),
    )


def build_long_term_profile(daily):
    keys = ["participant_id", "market_product"]
    g = daily.groupby(keys, sort=False)

    profile = g.agg(
        lt_active_days=("local_date", "nunique"),
        lt_total_intervals=("intervals_observed", "sum"),
        lt_valid_bid_intervals=("valid_bid_intervals", "sum"),

        lt_bid_level=("daily_bid_level", "median"),
        lt_quantity_hhi=("daily_quantity_hhi", "median"),
        lt_effective_segment_count=("daily_effective_segment_count", "median"),
        lt_flat_curve_rate=("daily_flat_curve_rate", "median"),
        lt_tail_uplift_ratio=("daily_tail_uplift_ratio", "median"),
        lt_curve_bend_ratio=("daily_curve_bend_ratio", "median"),
        lt_shape_defined_rate=("daily_shape_defined_rate", "median"),

        lt_self_adjustment_bias=("daily_self_adjustment_bias", "median"),
        lt_self_adjustment_magnitude=(
            "daily_self_adjustment_magnitude", "median"
        ),
        lt_self_adjustment_p90=("daily_self_adjustment_p90", "median"),

        lt_adjacent_level_change_magnitude=(
            "daily_adjacent_level_change_magnitude", "median"
        ),
        lt_same_slot_shape_change=("daily_same_slot_shape_change", "median"),
        lt_adjacent_shape_change=("daily_adjacent_shape_change", "median"),
        lt_adjacent_shape_change_p90=(
            "daily_adjacent_shape_change_p90", "median"
        ),

        lt_curve_mode_switch_rate=("daily_curve_mode_switch_rate", "median"),
        lt_flat_curve_switch_rate=("daily_flat_curve_switch_rate", "median"),

        lt_valid_bid_rate=("daily_valid_bid_rate", "median"),
        lt_baseline_ready_rate=("daily_baseline_ready_rate", "median"),
    ).reset_index()

    level_scale = (
        g["daily_bid_level"]
        .apply(robust_scale)
        .rename("lt_bid_level_scale")
        .reset_index()
    )
    profile = profile.merge(level_scale, on=keys, how="left")

    hhi_scale = (
        g["daily_quantity_hhi"]
        .apply(robust_scale)
        .rename("lt_quantity_hhi_scale")
        .reset_index()
    )
    profile = profile.merge(hhi_scale, on=keys, how="left")

    daily_shape_cols = [f"daily_shape_v{i:02d}" for i in range(N_SHAPE_GRID)]
    shape_proto = (
        g[daily_shape_cols]
        .median()
        .reset_index()
        .rename(
            columns={
                f"daily_shape_v{i:02d}": f"lt_shape_v{i:02d}"
                for i in range(N_SHAPE_GRID)
            }
        )
    )
    profile = profile.merge(shape_proto, on=keys, how="left")

    return profile


def add_daily_shape_deviation(daily, profile):
    proto_cols = [f"lt_shape_v{i:02d}" for i in range(N_SHAPE_GRID)]
    daily_cols = [f"daily_shape_v{i:02d}" for i in range(N_SHAPE_GRID)]

    x = daily.merge(
        profile[["participant_id", "market_product"] + proto_cols],
        on=["participant_id", "market_product"],
        how="left",
        validate="many_to_one",
    )

    a = x[daily_cols].to_numpy(float)
    b = x[proto_cols].to_numpy(float)

    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    dist = np.full(len(x), np.nan)

    if valid.any():
        delta = a[valid] - b[valid]
        dist[valid] = np.sqrt(np.mean(delta * delta, axis=1))

    x["daily_shape_deviation"] = dist
    return x.drop(columns=proto_cols)


def build_strategy_persistence(daily):
    rows = []

    for (pid, product), g in daily.groupby(
        ["participant_id", "market_product"],
        sort=False,
    ):
        g = g.sort_values("local_date").copy()

        dates = pd.to_datetime(g["local_date"], errors="coerce")
        x = pd.to_numeric(
            g["daily_self_adjustment_bias"], errors="coerce"
        )

        prev_x = x.shift(1)
        prev_date = dates.shift(1)
        consecutive = (dates - prev_date).dt.days.eq(1)

        valid = x.notna() & prev_x.notna() & consecutive
        n_pairs = int(valid.sum())

        corr = np.nan
        if n_pairs >= 5:
            xv = x[valid].to_numpy(float)
            pv = prev_x[valid].to_numpy(float)
            if np.std(xv) > 0 and np.std(pv) > 0:
                corr = float(np.corrcoef(xv, pv)[0, 1])

        rows.append({
            "participant_id": pid,
            "market_product": product,
            "lt_strategy_persistence": corr,
            "lt_persistence_pair_count": n_pairs,
        })

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--atom-root",
        default="data/processed/interval_strategy_atoms",
    )
    p.add_argument(
        "--temporal-root",
        default="data/processed/temporal_strategy_behavior",
    )
    p.add_argument(
        "--daily-root",
        default="data/processed/daily_strategy_summary",
    )
    p.add_argument(
        "--profile-root",
        default="data/processed/long_term_strategy_profile",
    )
    p.add_argument(
        "--results-root",
        default="results/04_long_term_profile",
    )
    p.add_argument(
        "--stage-root",
        default="data/processed/_tmp_daily_partition",
    )
    p.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep temporary local-date staging files for diagnostics.",
    )
    args = p.parse_args()

    atom_dir = Path(args.atom_root) / str(args.year)
    temp_dir = Path(args.temporal_root) / str(args.year)
    daily_dir = Path(args.daily_root) / str(args.year)
    profile_dir = Path(args.profile_root) / str(args.year)
    results_dir = Path(args.results_root)
    stage_dir = Path(args.stage_root) / str(args.year)

    daily_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(atom_dir, temp_dir, args.year)

    (
        staged_rows,
        date_sources,
        source_coverage,
    ) = stage_by_local_date(pairs, stage_dir)

    (
        daily,
        duplicate_intervals_removed,
        bucket_stats,
    ) = build_daily_from_stage(stage_dir)

    source_coverage.to_csv(
        results_dir / f"source_local_date_coverage_{args.year}.csv",
        index=False,
    )
    bucket_stats.to_csv(
        results_dir / f"local_date_bucket_summary_{args.year}.csv",
        index=False,
    )

    multi_source_dates = {
        d: sorted(sources)
        for d, sources in date_sources.items()
        if len(sources) > 1
    }
    pd.DataFrame([
        {
            "local_date": d,
            "source_count": len(sources),
            "sources": " | ".join(sources),
        }
        for d, sources in sorted(multi_source_dates.items())
    ]).to_csv(
        results_dir / f"multi_source_local_dates_{args.year}.csv",
        index=False,
    )

    profile = build_long_term_profile(daily)
    daily = add_daily_shape_deviation(daily, profile)

    keys = ["participant_id", "market_product"]

    dev = (
        daily.groupby(keys, sort=False)["daily_shape_deviation"]
        .agg(
            lt_shape_day_deviation_median="median",
            lt_shape_day_deviation_p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    profile = profile.merge(dev, on=keys, how="left")

    persistence = build_strategy_persistence(daily)
    profile = profile.merge(persistence, on=keys, how="left")

    daily = daily.sort_values(
        ["participant_id", "market_product", "local_date"]
    )
    daily_file = daily_dir / f"daily_strategy_summary_{args.year}.csv"
    daily.to_csv(daily_file, index=False)

    shape_proto_cols = [f"lt_shape_v{i:02d}" for i in range(N_SHAPE_GRID)]

    ordered = [
        "participant_id",
        "market_product",

        "lt_bid_level",
        "lt_bid_level_scale",
        "lt_self_adjustment_bias",
        "lt_self_adjustment_magnitude",
        "lt_self_adjustment_p90",
        "lt_strategy_persistence",

        "lt_quantity_hhi",
        "lt_quantity_hhi_scale",

        "lt_effective_segment_count",
        "lt_flat_curve_rate",
        "lt_tail_uplift_ratio",
        "lt_curve_bend_ratio",
        "lt_shape_defined_rate",
        "lt_shape_day_deviation_median",
        "lt_shape_day_deviation_p90",
    ] + shape_proto_cols + [
        "lt_adjacent_level_change_magnitude",
        "lt_same_slot_shape_change",
        "lt_adjacent_shape_change",
        "lt_adjacent_shape_change_p90",
        "lt_curve_mode_switch_rate",
        "lt_flat_curve_switch_rate",

        "lt_active_days",
        "lt_total_intervals",
        "lt_valid_bid_intervals",
        "lt_valid_bid_rate",
        "lt_baseline_ready_rate",
        "lt_persistence_pair_count",
    ]

    profile = profile[ordered].sort_values(
        ["market_product", "participant_id"]
    )

    profile_file = (
        profile_dir / f"long_term_strategy_profile_{args.year}.csv"
    )
    profile.to_csv(profile_file, index=False)

    duplicate_days = int(
        daily.duplicated(
            ["participant_id", "market_product", "local_date"]
        ).sum()
    )

    lines = [
        f"Universal long-term strategy profile summary - {args.year}",
        f"Version: {VERSION}",
        "=" * 72,
        f"Source files processed: {len(pairs)}",
        f"Interval rows staged: {staged_rows:,}",
        f"Distinct local dates: {daily['local_date'].nunique():,}",
        f"Local dates represented by >1 source file: {len(multi_source_dates):,}",
        f"Exact duplicate physical intervals removed: "
        f"{duplicate_intervals_removed:,}",
        f"Participant-days: {len(daily):,}",
        f"Duplicate participant-days after local-date aggregation: "
        f"{duplicate_days:,}",
        f"Participants: {profile['participant_id'].nunique():,}",
        f"Market products: {profile['market_product'].nunique():,}",
        f"Median active days per participant: "
        f"{profile['lt_active_days'].median():.1f}",
        f"Median valid-bid rate: "
        f"{profile['lt_valid_bid_rate'].median():.4f}",
        f"Median baseline-ready rate: "
        f"{profile['lt_baseline_ready_rate'].median():.4f}",
        f"Profiles with strategy persistence available: "
        f"{profile['lt_strategy_persistence'].notna().sum():,}",
        f"Profiles with long-term shape prototype available: "
        f"{profile['lt_shape_v00'].notna().sum():,}",
        "",
        "Boundary handling:",
        "  source-file date boundaries are ignored;",
        "  all interval rows are partitioned by local_date first;",
        "  every participant-day is aggregated exactly once.",
        "",
        "Long-term history-identifiable profile dimensions:",
        "  A. price/level baseline and adjustment tendency",
        "  B. quantity allocation",
        "  C. curve organization",
        "  D. adjustment/switching behavior",
        "",
        f"Daily summary: {daily_file.resolve()}",
        f"Long-term profile: {profile_file.resolve()}",
    ]

    summary_file = results_dir / f"summary_{args.year}.txt"
    summary_file.write_text("\n".join(lines), encoding="utf-8")

    if not args.keep_stage:
        shutil.rmtree(stage_dir, ignore_errors=True)

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
