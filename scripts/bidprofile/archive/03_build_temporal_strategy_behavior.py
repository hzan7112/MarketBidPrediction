#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_build_temporal_strategy_behavior.py
Version: 2026-09-17-v1

Purpose
-------
Build universal temporal bidding-behavior features from interval strategy atoms.

This module is market-agnostic:
- no PJM field names;
- no assumption of thermal / wind / PV / hydro / storage;
- no assumption that a trading day has exactly 24 intervals;
- no future information is used.

Core idea
---------
For each participant and market product, compare the current bid with:
1) its own prior bids at the same local market slot;
2) its immediately preceding market interval.

"Same local slot" is represented by seconds since local midnight, so the same
logic works for hourly, 30-min, 15-min, 5-min, etc. markets.

The historical price baseline is:
    median(previous W valid bid_level observations at the same local slot)

Only prior observations are used.

Run:
    python scripts/03_build_temporal_strategy_behavior.py --year 2025

Input:
    data/processed/interval_strategy_atoms/2025/
        interval_strategy_atoms_2025_*.csv

Output:
    data/processed/temporal_strategy_behavior/2025/
        temporal_strategy_behavior_2025_*.csv
    results/03_temporal_behavior/summary_2025.txt

Output features
---------------
same_slot_history_count
self_bid_level_baseline
self_bid_level_residual
same_slot_level_change
same_slot_shape_distance

prev_time_gap_seconds
nominal_interval_seconds
adjacent_interval_flag
adjacent_bid_level_change
adjacent_shape_distance
adjacent_quantity_hhi_change
adjacent_effective_segment_count_change
adjacent_tail_uplift_ratio_change
adjacent_curve_bend_ratio_change
curve_mode_switch_flag
flat_curve_switch_flag

Notes
-----
- No arbitrary "active/inactive" threshold is imposed here.
- No composite activity index is created.
- Long-term/short-term aggregation is deferred to later modules.
- self_bid_level_residual is kept in original price units here.
  Cross-participant normalization belongs to the profile layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


VERSION = "2026-09-17-v1"
EPS = 1e-12
DEFAULT_HISTORY_WINDOW = 30
DEFAULT_MIN_HISTORY = 5
N_SHAPE_GRID = 21


META_COLS = [
    "interval_index",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "market_product",
    "curve_mode",
    "has_offer",
    "source_market",
    "source_file",
]

ATOM_COLS = [
    "bid_level",
    "effective_segment_count",
    "quantity_hhi",
    "tail_uplift_ratio",
    "curve_bend_ratio",
    "flat_curve_flag",
    "shape_defined_flag",
]

SHAPE_COLS = [f"shape_v{i:02d}" for i in range(N_SHAPE_GRID)]


def infer_nominal_interval_seconds(timestamp_utc: pd.Series) -> float:
    """Infer the dominant market interval spacing from unique UTC timestamps."""
    ts = pd.to_datetime(timestamp_utc, errors="coerce", utc=True).dropna()
    if ts.empty:
        return np.nan

    vals = np.sort(ts.drop_duplicates().astype("int64").to_numpy())
    if len(vals) < 2:
        return np.nan

    diffs = np.diff(vals) / 1e9
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan

    # Round to the nearest second before mode calculation.
    rounded = np.rint(diffs).astype(np.int64)
    vc = pd.Series(rounded).value_counts()
    return float(vc.index[0])


def local_slot_seconds(timestamp_local: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp_local, errors="coerce")
    return (
        ts.dt.hour * 3600
        + ts.dt.minute * 60
        + ts.dt.second
    ).astype("Int64")


def chronological_files(input_dir: Path, year: int):
    files = list(input_dir.glob(f"interval_strategy_atoms_{year}_*.csv"))
    if not files:
        return []

    keyed = []
    for f in files:
        head = pd.read_csv(
            f,
            usecols=["timestamp_utc"],
            nrows=2000,
        )
        ts = pd.to_datetime(head["timestamp_utc"], errors="coerce", utc=True)
        first = ts.min()
        keyed.append((pd.Timestamp.max.tz_localize("UTC") if pd.isna(first) else first, f))

    return [f for _, f in sorted(keyed, key=lambda x: x[0])]


def compute_same_slot_features(
    df: pd.DataFrame,
    level_history_tail: pd.DataFrame | None,
    shape_history_tail: pd.DataFrame | None,
    history_window: int,
    min_history: int,
):
    """
    Same-slot history uses participant_id + market_product + local_slot_seconds.
    Returns features and updated history tails.
    """
    n = len(df)
    row_id = np.arange(n, dtype=np.int64)

    out = pd.DataFrame(index=np.arange(n))
    out["same_slot_history_count"] = 0
    out["self_bid_level_baseline"] = np.nan
    out["self_bid_level_residual"] = np.nan
    out["same_slot_level_change"] = np.nan
    out["same_slot_shape_distance"] = np.nan
    out["baseline_ready_flag"] = 0

    slot_keys = [
        "participant_id",
        "market_product",
        "local_slot_seconds",
    ]

    # ---------- Price-level same-slot history ----------
    curr_level = df.loc[
        df["bid_level"].notna(),
        slot_keys + ["timestamp_utc", "bid_level"]
    ].copy()
    curr_level["_row_id"] = row_id[df["bid_level"].notna().to_numpy()]
    curr_level["_is_current"] = 1

    if level_history_tail is not None and len(level_history_tail):
        hist = level_history_tail.copy()
        hist["_row_id"] = -1
        hist["_is_current"] = 0
        combo = pd.concat([hist, curr_level], ignore_index=True)
    else:
        combo = curr_level.copy()

    if len(combo):
        combo = combo.sort_values(
            slot_keys + ["timestamp_utc", "_is_current"]
        ).reset_index(drop=True)

        group_arrays = [combo[c] for c in slot_keys]

        prev_level = combo.groupby(
            slot_keys, sort=False
        )["bid_level"].shift(1)

        shifted = prev_level

        rolling_median = (
            shifted
            .groupby(group_arrays, sort=False)
            .rolling(history_window, min_periods=1)
            .median()
            .reset_index(level=list(range(len(slot_keys))), drop=True)
        )

        rolling_count = (
            shifted
            .groupby(group_arrays, sort=False)
            .rolling(history_window, min_periods=1)
            .count()
            .reset_index(level=list(range(len(slot_keys))), drop=True)
        )

        combo["_baseline"] = rolling_median
        combo["_history_count"] = rolling_count.fillna(0).astype(np.int32)
        combo["_prev_level"] = prev_level

        cur = combo[combo["_is_current"].eq(1)].copy()
        rid = cur["_row_id"].to_numpy(np.int64)

        out.loc[rid, "same_slot_history_count"] = (
            cur["_history_count"].to_numpy(np.int32)
        )
        out.loc[rid, "self_bid_level_baseline"] = (
            cur["_baseline"].to_numpy(float)
        )
        out.loc[rid, "same_slot_level_change"] = (
            cur["bid_level"].to_numpy(float)
            - cur["_prev_level"].to_numpy(float)
        )

        ready = cur["_history_count"].to_numpy(np.int32) >= min_history
        residual = (
            cur["bid_level"].to_numpy(float)
            - cur["_baseline"].to_numpy(float)
        )
        residual[~ready] = np.nan

        out.loc[rid, "self_bid_level_residual"] = residual
        out.loc[rid, "baseline_ready_flag"] = ready.astype(np.int8)

        # Keep only prior/current valid levels required for next file.
        level_history_tail = (
            combo[
                slot_keys + ["timestamp_utc", "bid_level"]
            ]
            .groupby(slot_keys, sort=False, group_keys=False)
            .tail(history_window)
            .reset_index(drop=True)
        )
    else:
        level_history_tail = pd.DataFrame(
            columns=slot_keys + ["timestamp_utc", "bid_level"]
        )

    # ---------- Same-slot shape change ----------
    shape_ok = df["shape_defined_flag"].fillna(0).eq(1)
    curr_shape = df.loc[
        shape_ok,
        slot_keys + ["timestamp_utc"] + SHAPE_COLS
    ].copy()
    curr_shape["_row_id"] = row_id[shape_ok.to_numpy()]
    curr_shape["_is_current"] = 1

    if shape_history_tail is not None and len(shape_history_tail):
        hist_shape = shape_history_tail.copy()
        hist_shape["_row_id"] = -1
        hist_shape["_is_current"] = 0
        shape_combo = pd.concat(
            [hist_shape, curr_shape],
            ignore_index=True,
        )
    else:
        shape_combo = curr_shape.copy()

    if len(shape_combo):
        shape_combo = shape_combo.sort_values(
            slot_keys + ["timestamp_utc", "_is_current"]
        ).reset_index(drop=True)

        prev_shape = (
            shape_combo.groupby(slot_keys, sort=False)[SHAPE_COLS]
            .shift(1)
        )

        current_mat = shape_combo[SHAPE_COLS].to_numpy(float)
        prev_mat = prev_shape.to_numpy(float)
        valid_pair = (
            np.isfinite(current_mat).all(axis=1)
            & np.isfinite(prev_mat).all(axis=1)
        )

        dist = np.full(len(shape_combo), np.nan)
        if valid_pair.any():
            diff = current_mat[valid_pair] - prev_mat[valid_pair]
            dist[valid_pair] = np.sqrt(np.mean(diff * diff, axis=1))

        shape_combo["_shape_prev_same_slot_distance"] = dist

        cur = shape_combo[shape_combo["_is_current"].eq(1)]
        rid = cur["_row_id"].to_numpy(np.int64)
        out.loc[rid, "same_slot_shape_distance"] = (
            cur["_shape_prev_same_slot_distance"].to_numpy(float)
        )

        # Only the most recent same-slot shape is needed for next file.
        shape_history_tail = (
            shape_combo[
                slot_keys + ["timestamp_utc"] + SHAPE_COLS
            ]
            .groupby(slot_keys, sort=False, group_keys=False)
            .tail(1)
            .reset_index(drop=True)
        )
    else:
        shape_history_tail = pd.DataFrame(
            columns=slot_keys + ["timestamp_utc"] + SHAPE_COLS
        )

    return out, level_history_tail, shape_history_tail


def compute_adjacent_features(
    df: pd.DataFrame,
    prev_interval_tail: pd.DataFrame | None,
    nominal_seconds: float,
):
    """
    Compare current row with the immediately preceding participant interval.
    Changes are reported only when timestamps are adjacent according to the
    inferred nominal market interval.
    """
    n = len(df)
    row_id = np.arange(n, dtype=np.int64)

    pair_keys = ["participant_id", "market_product"]

    needed = (
        pair_keys
        + [
            "timestamp_utc",
            "bid_level",
            "effective_segment_count",
            "quantity_hhi",
            "tail_uplift_ratio",
            "curve_bend_ratio",
            "flat_curve_flag",
            "curve_mode",
            "shape_defined_flag",
        ]
        + SHAPE_COLS
    )

    cur = df[needed].copy()
    cur["_row_id"] = row_id
    cur["_is_current"] = 1

    if prev_interval_tail is not None and len(prev_interval_tail):
        hist = prev_interval_tail.copy()
        hist["_row_id"] = -1
        hist["_is_current"] = 0
        combo = pd.concat([hist, cur], ignore_index=True)
    else:
        combo = cur.copy()

    combo = combo.sort_values(
        pair_keys + ["timestamp_utc", "_is_current"]
    ).reset_index(drop=True)

    g = combo.groupby(pair_keys, sort=False)

    prev_ts = g["timestamp_utc"].shift(1)
    prev_level = g["bid_level"].shift(1)
    prev_seg = g["effective_segment_count"].shift(1)
    prev_hhi = g["quantity_hhi"].shift(1)
    prev_tail = g["tail_uplift_ratio"].shift(1)
    prev_bend = g["curve_bend_ratio"].shift(1)
    prev_flat = g["flat_curve_flag"].shift(1)
    prev_mode = g["curve_mode"].shift(1)
    prev_shape_defined = g["shape_defined_flag"].shift(1)
    prev_shape = g[SHAPE_COLS].shift(1)

    ts = pd.to_datetime(combo["timestamp_utc"], errors="coerce", utc=True)
    prev_ts = pd.to_datetime(prev_ts, errors="coerce", utc=True)
    gap = (ts - prev_ts).dt.total_seconds()

    if np.isfinite(nominal_seconds):
        adjacent = (
            gap.notna()
            & np.isclose(
                gap.to_numpy(float),
                nominal_seconds,
                atol=1.0,
                rtol=0.0,
            )
        )
    else:
        adjacent = np.zeros(len(combo), dtype=bool)

    out_combo = pd.DataFrame(index=combo.index)
    out_combo["prev_time_gap_seconds"] = gap
    out_combo["adjacent_interval_flag"] = adjacent.astype(np.int8)

    def adjacent_diff(current_col, previous):
        a = pd.to_numeric(combo[current_col], errors="coerce").to_numpy(float)
        b = pd.to_numeric(previous, errors="coerce").to_numpy(float)
        v = a - b
        v[~adjacent] = np.nan
        return v

    out_combo["adjacent_bid_level_change"] = adjacent_diff(
        "bid_level", prev_level
    )
    out_combo["adjacent_quantity_hhi_change"] = adjacent_diff(
        "quantity_hhi", prev_hhi
    )
    out_combo["adjacent_effective_segment_count_change"] = adjacent_diff(
        "effective_segment_count", prev_seg
    )
    out_combo["adjacent_tail_uplift_ratio_change"] = adjacent_diff(
        "tail_uplift_ratio", prev_tail
    )
    out_combo["adjacent_curve_bend_ratio_change"] = adjacent_diff(
        "curve_bend_ratio", prev_bend
    )

    # Structural switches only when adjacent and both states are observed.
    current_mode = combo["curve_mode"].astype("string")
    pmode = prev_mode.astype("string")
    mode_valid = current_mode.notna() & pmode.notna()
    mode_switch = (
        adjacent
        & mode_valid.to_numpy()
        & current_mode.ne(pmode).to_numpy()
    )
    out_combo["curve_mode_switch_flag"] = np.where(
        adjacent & mode_valid.to_numpy(),
        mode_switch.astype(np.int8),
        np.nan,
    )

    current_flat = pd.to_numeric(
        combo["flat_curve_flag"], errors="coerce"
    )
    pflat = pd.to_numeric(prev_flat, errors="coerce")
    flat_valid = current_flat.notna() & pflat.notna()
    flat_switch = (
        adjacent
        & flat_valid.to_numpy()
        & current_flat.ne(pflat).to_numpy()
    )
    out_combo["flat_curve_switch_flag"] = np.where(
        adjacent & flat_valid.to_numpy(),
        flat_switch.astype(np.int8),
        np.nan,
    )

    # Adjacent shape distance.
    current_shape = combo[SHAPE_COLS].to_numpy(float)
    previous_shape = prev_shape.to_numpy(float)
    shape_pair = (
        adjacent
        & combo["shape_defined_flag"].fillna(0).eq(1).to_numpy()
        & pd.to_numeric(
            prev_shape_defined, errors="coerce"
        ).fillna(0).eq(1).to_numpy()
        & np.isfinite(current_shape).all(axis=1)
        & np.isfinite(previous_shape).all(axis=1)
    )

    shape_dist = np.full(len(combo), np.nan)
    if shape_pair.any():
        d = current_shape[shape_pair] - previous_shape[shape_pair]
        shape_dist[shape_pair] = np.sqrt(np.mean(d * d, axis=1))
    out_combo["adjacent_shape_distance"] = shape_dist

    # Extract current rows back to original file order.
    current_mask = combo["_is_current"].eq(1)
    cur_combo = combo.loc[current_mask, ["_row_id"]].copy()
    cur_out = out_combo.loc[current_mask].copy()
    cur_out["_row_id"] = cur_combo["_row_id"].to_numpy(np.int64)
    cur_out = cur_out.sort_values("_row_id").set_index("_row_id")
    cur_out = cur_out.reindex(np.arange(n)).reset_index(drop=True)
    cur_out["nominal_interval_seconds"] = nominal_seconds

    # Keep last row per participant/product for the next source file.
    prev_interval_tail = (
        combo[needed]
        .groupby(pair_keys, sort=False, group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return cur_out, prev_interval_tail


def process_file(
    input_file: Path,
    output_file: Path,
    level_history_tail,
    shape_history_tail,
    prev_interval_tail,
    history_window: int,
    min_history: int,
):
    print(f"[read] {input_file}")

    header = pd.read_csv(input_file, nrows=0)
    missing = [
        c for c in META_COLS + ATOM_COLS + SHAPE_COLS
        if c not in header.columns
    ]
    if missing:
        raise ValueError(
            f"{input_file.name}: missing required interval atoms: {missing}"
        )

    usecols = META_COLS + ATOM_COLS + SHAPE_COLS
    dtype = {c: "float32" for c in SHAPE_COLS}
    dtype.update({
        "bid_level": "float64",
        "effective_segment_count": "float32",
        "quantity_hhi": "float32",
        "tail_uplift_ratio": "float32",
        "curve_bend_ratio": "float32",
        "flat_curve_flag": "float32",
        "shape_defined_flag": "float32",
    })

    df = pd.read_csv(
        input_file,
        usecols=usecols,
        dtype=dtype,
        low_memory=False,
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], errors="coerce", utc=True
    )
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="coerce"
    )
    df["local_slot_seconds"] = local_slot_seconds(df["timestamp_local"])

    # Chronological order is required for strictly backward-looking features.
    df["_original_order"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(
        ["timestamp_utc", "participant_id", "market_product"]
    ).reset_index(drop=True)

    nominal_seconds = infer_nominal_interval_seconds(df["timestamp_utc"])

    same_slot, level_history_tail, shape_history_tail = (
        compute_same_slot_features(
            df,
            level_history_tail,
            shape_history_tail,
            history_window,
            min_history,
        )
    )

    adjacent, prev_interval_tail = compute_adjacent_features(
        df,
        prev_interval_tail,
        nominal_seconds,
    )

    key_cols = [
        "interval_index",
        "participant_id",
        "timestamp_utc",
        "timestamp_local",
        "market_product",
        "source_market",
        "source_file",
        "local_slot_seconds",
    ]

    out = pd.concat(
        [
            df[key_cols].reset_index(drop=True),
            same_slot.reset_index(drop=True),
            adjacent.reset_index(drop=True),
        ],
        axis=1,
    )

    # Restore source-file row order for easier joins with interval atoms.
    out["_original_order"] = df["_original_order"].to_numpy()
    out = (
        out.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)

    valid_level = df["bid_level"].notna()
    return (
        {
            "file": input_file.name,
            "rows": len(df),
            "valid_bid_level_rows": int(valid_level.sum()),
            "baseline_ready_rows": int(
                out["baseline_ready_flag"].fillna(0).sum()
            ),
            "same_slot_residual_nonmissing": int(
                out["self_bid_level_residual"].notna().sum()
            ),
            "same_slot_shape_distance_nonmissing": int(
                out["same_slot_shape_distance"].notna().sum()
            ),
            "adjacent_pairs": int(
                out["adjacent_interval_flag"].fillna(0).sum()
            ),
            "adjacent_shape_distance_nonmissing": int(
                out["adjacent_shape_distance"].notna().sum()
            ),
            "nominal_interval_seconds": nominal_seconds,
        },
        level_history_tail,
        shape_history_tail,
        prev_interval_tail,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--input-root",
        default="data/processed/interval_strategy_atoms",
    )
    p.add_argument(
        "--output-root",
        default="data/processed/temporal_strategy_behavior",
    )
    p.add_argument(
        "--results-root",
        default="results/03_temporal_behavior",
    )
    p.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="Number of prior valid same-slot observations used for the rolling median.",
    )
    p.add_argument(
        "--min-history",
        type=int,
        default=DEFAULT_MIN_HISTORY,
        help="Minimum prior same-slot observations before residual is considered reliable.",
    )
    args = p.parse_args()

    if args.history_window < 1:
        raise ValueError("--history-window must be >= 1.")
    if args.min_history < 1 or args.min_history > args.history_window:
        raise ValueError("--min-history must be within [1, history-window].")

    input_dir = Path(args.input_root) / str(args.year)
    output_dir = Path(args.output_root) / str(args.year)
    results_dir = Path(args.results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    files = chronological_files(input_dir, args.year)
    if not files:
        raise FileNotFoundError(
            f"No interval strategy atom files in {input_dir.resolve()}"
        )

    level_history_tail = None
    shape_history_tail = None
    prev_interval_tail = None
    summaries = []

    for f in files:
        suffix = f.stem.replace("interval_strategy_atoms_", "")
        output_file = (
            output_dir / f"temporal_strategy_behavior_{suffix}.csv"
        )

        (
            summary,
            level_history_tail,
            shape_history_tail,
            prev_interval_tail,
        ) = process_file(
            f,
            output_file,
            level_history_tail,
            shape_history_tail,
            prev_interval_tail,
            args.history_window,
            args.min_history,
        )
        summaries.append(summary)

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(
        results_dir / f"file_summary_{args.year}.csv",
        index=False,
    )

    total = int(sdf["rows"].sum())
    valid = int(sdf["valid_bid_level_rows"].sum())

    lines = [
        f"Universal temporal strategy behavior summary - {args.year}",
        f"Version: {VERSION}",
        "=" * 72,
        f"Files processed: {len(sdf)}",
        f"Rows processed: {total:,}",
        f"Rows with valid bid level: {valid:,}",
        f"History window W: {args.history_window}",
        f"Minimum prior same-slot history: {args.min_history}",
        f"Baseline-ready rows: {sdf['baseline_ready_rows'].sum():,}",
        f"Self bid-level residual available: "
        f"{sdf['same_slot_residual_nonmissing'].sum():,}",
        f"Same-slot shape distance available: "
        f"{sdf['same_slot_shape_distance_nonmissing'].sum():,}",
        f"Adjacent market-interval pairs: {sdf['adjacent_pairs'].sum():,}",
        f"Adjacent shape distance available: "
        f"{sdf['adjacent_shape_distance_nonmissing'].sum():,}",
        "",
        "Temporal behavior fields:",
        "  same_slot_history_count",
        "  self_bid_level_baseline",
        "  self_bid_level_residual",
        "  same_slot_level_change",
        "  same_slot_shape_distance",
        "  prev_time_gap_seconds",
        "  nominal_interval_seconds",
        "  adjacent_interval_flag",
        "  adjacent_bid_level_change",
        "  adjacent_shape_distance",
        "  adjacent_quantity_hhi_change",
        "  adjacent_effective_segment_count_change",
        "  adjacent_tail_uplift_ratio_change",
        "  adjacent_curve_bend_ratio_change",
        "  curve_mode_switch_flag",
        "  flat_curve_switch_flag",
        "",
        "No composite activity score or arbitrary strategy threshold is used.",
        f"Output directory: {output_dir.resolve()}",
    ]

    (results_dir / f"summary_{args.year}.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
