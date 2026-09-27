#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_standardize_pjm_bids_v2.py
Version: 2026-09-17-v2

PJM adapter -> unified market-agnostic bid representation.

The standardized representation is split into two relational tables so
downstream strategy code never depends on PJM-specific fields such as
mw1..mw20 / bid1..bid20.

1) Interval table: one row per participant x market interval x product
2) Segment table : zero or more ordered price-quantity breakpoints per interval

Run:
    python scripts/01_standardize_pjm_bids_v2.py --year 2025

Input:
    data/raw/energy_market_offers/2025/energy_market_offers_2025_*.csv

Output:
    data/standardized/intervals/2025/standardized_intervals_2025_*.csv
    data/standardized/bid_segments/2025/standardized_bid_segments_2025_*.csv
    results/01_standardize/summary_2025.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "2026-09-17-v2"
N_SEGMENTS = 20
DEFAULT_CHUNK_ROWS = 100_000


def normalize_curve_mode(series):
    if pd.api.types.is_bool_dtype(series):
        arr = series.fillna(False).to_numpy(bool)
    else:
        arr = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y", "t"])
            .to_numpy(bool)
        )
    return np.where(arr, "sloped", "block")


def validate_schema(columns, filename):
    cmap = {c.lower(): c for c in columns}
    required = [
        "unit_code",
        "bid_datetime_beginning_utc",
        "bid_datetime_beginning_ept",
        "bid_slope_flag",
    ]
    missing = [c for c in required if c not in cmap]
    for i in range(1, N_SEGMENTS + 1):
        if f"mw{i}" not in cmap:
            missing.append(f"mw{i}")
        if f"bid{i}" not in cmap:
            missing.append(f"bid{i}")
    if missing:
        raise ValueError(f"{filename}: missing required columns: {missing}")


def append_csv(df, path, first_write):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )


def standardize_file(input_file, interval_file, segment_file, chunk_rows):
    print(f"[source] {input_file}")

    for p in [interval_file, segment_file]:
        if p.exists():
            p.unlink()

    reader = pd.read_csv(input_file, low_memory=False, chunksize=chunk_rows)

    first_intervals = True
    first_segments = True
    source_row_offset = 0
    next_interval_index = 0

    summary = {
        "file": input_file.name,
        "raw_rows": 0,
        "standardized_intervals": 0,
        "offer_intervals": 0,
        "no_offer_intervals": 0,
        "standardized_segments": 0,
        "rows_dropped_bad_key": 0,
        "duplicate_quantity_points_removed": 0,
    }

    for chunk_no, df in enumerate(reader, start=1):
        df.columns = [c.strip() for c in df.columns]
        if chunk_no == 1:
            validate_schema(df.columns, input_file.name)

        cmap = {c.lower(): c for c in df.columns}
        unit_col = cmap["unit_code"]
        utc_col = cmap["bid_datetime_beginning_utc"]
        local_col = cmap["bid_datetime_beginning_ept"]
        slope_col = cmap["bid_slope_flag"]

        n_raw = len(df)
        summary["raw_rows"] += n_raw

        source_row_index = source_row_offset + np.arange(n_raw, dtype=np.int64)
        source_row_offset += n_raw

        participant = df[unit_col].astype("string")
        timestamp_utc = pd.to_datetime(df[utc_col], errors="coerce", utc=True)
        timestamp_local = pd.to_datetime(df[local_col], errors="coerce")

        key_ok = (
            participant.notna()
            & participant.str.len().gt(0)
            & timestamp_utc.notna()
            & timestamp_local.notna()
        ).to_numpy()

        summary["rows_dropped_bad_key"] += int((~key_ok).sum())
        if not key_ok.any():
            continue

        df = df.loc[key_ok].reset_index(drop=True)
        participant = participant.loc[key_ok].reset_index(drop=True)
        timestamp_utc = timestamp_utc.loc[key_ok].reset_index(drop=True)
        timestamp_local = timestamp_local.loc[key_ok].reset_index(drop=True)
        source_row_index = source_row_index[key_ok]

        n = len(df)
        interval_index = np.arange(
            next_interval_index,
            next_interval_index + n,
            dtype=np.int64,
        )
        next_interval_index += n

        mw = np.column_stack([
            pd.to_numeric(df[cmap[f"mw{i}"]], errors="coerce").to_numpy(float)
            for i in range(1, N_SEGMENTS + 1)
        ])
        bid = np.column_stack([
            pd.to_numeric(df[cmap[f"bid{i}"]], errors="coerce").to_numpy(float)
            for i in range(1, N_SEGMENTS + 1)
        ])

        valid = np.isfinite(mw) & np.isfinite(bid)
        valid_count = valid.sum(axis=1).astype(np.int16)
        has_offer = valid_count > 0

        intervals = pd.DataFrame({
            "interval_index": interval_index,
            "participant_id": participant,
            "timestamp_utc": timestamp_utc,
            "timestamp_local": timestamp_local,
            "market_product": "ENERGY",
            "curve_mode": normalize_curve_mode(df[slope_col]),
            "has_offer": has_offer.astype(np.int8),
            "raw_valid_point_count": valid_count,
            "source_market": "PJM",
            "source_file": input_file.name,
            "source_row_index": source_row_index,
        })

        append_csv(intervals, interval_file, first_intervals)
        first_intervals = False

        summary["standardized_intervals"] += n
        summary["offer_intervals"] += int(has_offer.sum())
        summary["no_offer_intervals"] += int((~has_offer).sum())

        r, c = np.nonzero(valid)
        if len(r):
            segments = pd.DataFrame({
                "interval_index": interval_index[r],
                "segment_id": (c + 1).astype(np.int16),
                "quantity": mw[r, c],
                "price": bid[r, c],
            })

            before = len(segments)
            segments = (
                segments
                .sort_values(
                    ["interval_index", "quantity", "price", "segment_id"]
                )
                .drop_duplicates(
                    subset=["interval_index", "quantity"],
                    keep="last",
                )
                .sort_values(["interval_index", "quantity", "price"])
                .reset_index(drop=True)
            )
            summary["duplicate_quantity_points_removed"] += before - len(segments)

            segments["segment_id"] = (
                segments.groupby("interval_index", sort=False).cumcount() + 1
            ).astype(np.int16)

            append_csv(segments, segment_file, first_segments)
            first_segments = False
            summary["standardized_segments"] += len(segments)

        print(
            f"  chunk {chunk_no}: rows={n:,}, "
            f"offers={int(has_offer.sum()):,}, "
            f"valid_slots={int(valid.sum()):,}"
        )

    if first_segments:
        empty = pd.DataFrame(
            columns=["interval_index", "segment_id", "quantity", "price"]
        )
        append_csv(empty, segment_file, True)

    print(
        f"[done] intervals={summary['standardized_intervals']:,}, "
        f"segments={summary['standardized_segments']:,}"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--input-root", default="data/raw/energy_market_offers")
    p.add_argument("--interval-root", default="data/standardized/intervals")
    p.add_argument("--segment-root", default="data/standardized/bid_segments")
    p.add_argument("--results-root", default="results/01_standardize")
    p.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = p.parse_args()

    input_dir = Path(args.input_root) / str(args.year)
    interval_dir = Path(args.interval_root) / str(args.year)
    segment_dir = Path(args.segment_root) / str(args.year)
    results_dir = Path(args.results_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(f"energy_market_offers_{args.year}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No PJM CSV found in {input_dir.resolve()}")

    summaries = []
    for src in files:
        suffix = src.stem.replace("energy_market_offers_", "")
        interval_file = interval_dir / f"standardized_intervals_{suffix}.csv"
        segment_file = segment_dir / f"standardized_bid_segments_{suffix}.csv"
        summaries.append(
            standardize_file(
                src,
                interval_file,
                segment_file,
                args.chunk_rows,
            )
        )

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(results_dir / f"file_summary_{args.year}.csv", index=False)

    lines = [
        f"Unified PJM bid standardization summary - {args.year}",
        f"Version: {VERSION}",
        "=" * 72,
        f"Files processed: {len(sdf)}",
        f"Raw source rows: {sdf['raw_rows'].sum():,}",
        f"Standardized intervals: {sdf['standardized_intervals'].sum():,}",
        f"Intervals with offer: {sdf['offer_intervals'].sum():,}",
        f"Intervals without offer: {sdf['no_offer_intervals'].sum():,}",
        f"Standardized bid segments: {sdf['standardized_segments'].sum():,}",
        f"Rows dropped for bad participant/time key: "
        f"{sdf['rows_dropped_bad_key'].sum():,}",
        f"Duplicate quantity breakpoints removed: "
        f"{sdf['duplicate_quantity_points_removed'].sum():,}",
        "",
        "Canonical interval table:",
        "  interval_index, participant_id, timestamp_utc, timestamp_local,",
        "  market_product, curve_mode, has_offer, raw_valid_point_count,",
        "  source_market, source_file, source_row_index",
        "",
        "Canonical segment table:",
        "  interval_index, segment_id, quantity, price",
        "",
        "Physical and market-state variables are intentionally stored separately.",
        f"Interval output: {interval_dir.resolve()}",
        f"Segment output:  {segment_dir.resolve()}",
    ]

    (results_dir / f"summary_{args.year}.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
