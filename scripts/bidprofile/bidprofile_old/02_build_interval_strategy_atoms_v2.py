#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_build_interval_strategy_atoms_v2.py
Version: 2026-09-17-v2

Build universal single-interval bidding-strategy atoms from canonical
interval + segment tables.

This module is market-agnostic and contains no PJM-specific field names.

Run:
    python scripts/02_build_interval_strategy_atoms_v2.py --year 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "2026-09-17-v2"
EPS = 1e-9
PRICE_MERGE_TOL = 1e-6
N_SHAPE_GRID = 21
DEFAULT_BATCH_INTERVALS = 200_000


def evaluate_at(x, p, valid, sloped, z):
    n, kmax = x.shape
    out = np.full(n, np.nan, dtype=np.float64)
    counts = valid.sum(axis=1)
    has = counts > 0
    if not has.any():
        return out

    first = valid.argmax(axis=1)
    last = kmax - 1 - valid[:, ::-1].argmax(axis=1)
    rows = np.arange(n)

    if z <= EPS:
        out[has] = p[rows[has], first[has]]
        return out

    if z >= 1.0 - EPS:
        out[has] = p[rows[has], last[has]]
        return out

    block = has & ~sloped
    if block.any():
        le = valid & (x <= z + EPS)
        idx = le.sum(axis=1) - 1
        rr = np.where(block)[0]
        out[rr] = p[rr, np.clip(idx[rr], 0, kmax - 1)]

    lin = has & sloped
    if lin.any():
        ge = valid & (x >= z - EPS)
        hi = ge.argmax(axis=1)
        rr = np.where(lin)[0]
        h = hi[rr]

        at_first = h == first[rr]
        if at_first.any():
            r0 = rr[at_first]
            out[r0] = p[r0, first[r0]]

        r1 = rr[~at_first]
        if len(r1):
            h1 = hi[r1]
            l1 = h1 - 1
            x0 = x[r1, l1]
            x1 = x[r1, h1]
            p0 = p[r1, l1]
            p1 = p[r1, h1]
            frac = np.divide(
                z - x0,
                x1 - x0,
                out=np.zeros(len(r1), dtype=np.float64),
                where=np.abs(x1 - x0) > EPS,
            )
            out[r1] = p0 + frac * (p1 - p0)

    return out


def compute_batch(meta, seg, max_segments):
    n = len(meta)
    q = np.full((n, max_segments), np.nan, dtype=np.float32)
    p = np.full((n, max_segments), np.nan, dtype=np.float32)

    start_index = int(meta["interval_index"].iloc[0])
    end_index = int(meta["interval_index"].iloc[-1]) + 1

    sidx = seg["interval_index"].to_numpy(np.int64)
    lo = np.searchsorted(sidx, start_index, side="left")
    hi = np.searchsorted(sidx, end_index, side="left")
    sub = seg.iloc[lo:hi]

    if len(sub):
        r = sub["interval_index"].to_numpy(np.int64) - start_index
        c = sub["segment_id"].to_numpy(np.int64) - 1
        if c.max(initial=-1) >= max_segments:
            raise ValueError("segment_id exceeds allocated max_segments.")

        q[r, c] = pd.to_numeric(
            sub["quantity"], errors="coerce"
        ).to_numpy(np.float32)
        p[r, c] = pd.to_numeric(
            sub["price"], errors="coerce"
        ).to_numpy(np.float32)

    valid = np.isfinite(q) & np.isfinite(p)
    counts = valid.sum(axis=1)
    has = counts > 0

    rows = np.arange(n)
    first = valid.argmax(axis=1)
    last = max_segments - 1 - valid[:, ::-1].argmax(axis=1)

    q_first = np.full(n, np.nan)
    q_last = np.full(n, np.nan)
    p_first = np.full(n, np.nan)
    p_last = np.full(n, np.nan)

    q_first[has] = q[rows[has], first[has]]
    q_last[has] = q[rows[has], last[has]]
    p_first[has] = p[rows[has], first[has]]
    p_last[has] = p[rows[has], last[has]]

    q_span = q_last - q_first
    p_range = p_last - p_first

    positive_span = has & (counts >= 2) & (q_span > EPS)
    flat = has & (np.abs(p_range) <= PRICE_MERGE_TOL)

    mode = meta["curve_mode"].astype(str).str.lower().to_numpy()
    bad_mode = ~np.isin(mode, ["block", "sloped"])
    if bad_mode.any():
        raise ValueError(
            f"Unsupported curve_mode: {sorted(set(mode[bad_mode].tolist()))}"
        )
    sloped = mode == "sloped"

    x = np.full_like(q, np.nan, dtype=np.float32)
    if positive_span.any():
        x[positive_span] = (
            q[positive_span] - q_first[positive_span, None]
        ) / q_span[positive_span, None]
        x[~valid] = np.nan

    bid_level = np.full(n, np.nan)
    bid_level[has & ~positive_span] = p_first[has & ~positive_span]

    pair_valid = valid[:, :-1] & valid[:, 1:]
    dx = x[:, 1:] - x[:, :-1]

    lin = positive_span & sloped
    if lin.any():
        area = dx * (p[:, 1:] + p[:, :-1]) / 2.0
        area[~pair_valid] = 0.0
        bid_level[lin] = np.nansum(area[lin], axis=1)

    block = positive_span & ~sloped
    if block.any():
        area = dx * p[:, :-1]
        area[~pair_valid] = 0.0
        bid_level[block] = np.nansum(area[block], axis=1)

    eff_count = np.zeros(n, dtype=np.int16)
    eff_count[has] = 1
    if max_segments > 1:
        price_change = (
            pair_valid
            & (np.abs(p[:, 1:] - p[:, :-1]) > PRICE_MERGE_TOL)
        )
        eff_count += price_change.sum(axis=1).astype(np.int16)

    quantity_hhi = np.full(n, np.nan)
    if positive_span.any():
        dq = np.where(pair_valid, q[:, 1:] - q[:, :-1], 0.0)
        dq = np.where(dq > 0, dq, 0.0)
        shares = np.divide(
            dq,
            q_span[:, None],
            out=np.zeros_like(dq, dtype=np.float32),
            where=q_span[:, None] > EPS,
        )
        quantity_hhi[positive_span] = np.sum(
            shares[positive_span] ** 2,
            axis=1,
        )

    p02 = evaluate_at(x, p, valid, sloped, 0.2)
    p08 = evaluate_at(x, p, valid, sloped, 0.8)

    tail_ratio = np.full(n, np.nan)
    bend_ratio = np.full(n, np.nan)
    nonflat_shape = positive_span & ~flat

    tail_ratio[positive_span & flat] = 0.0
    bend_ratio[positive_span & flat] = 0.0

    if nonflat_shape.any():
        denom = np.abs(p_range) + EPS
        tail_ratio[nonflat_shape] = (
            p_last[nonflat_shape] - p08[nonflat_shape]
        ) / denom[nonflat_shape]
        bend_ratio[nonflat_shape] = (
            (p_last[nonflat_shape] - p08[nonflat_shape])
            - (p02[nonflat_shape] - p_first[nonflat_shape])
        ) / denom[nonflat_shape]

    shape_defined = nonflat_shape
    grid = np.linspace(0.0, 1.0, N_SHAPE_GRID)
    shape = np.full((n, N_SHAPE_GRID), np.nan, dtype=np.float32)

    if shape_defined.any():
        for j, z in enumerate(grid):
            pz = evaluate_at(x, p, valid, sloped, float(z))
            vals = np.divide(
                pz - p_first,
                p_range,
                out=np.full(n, np.nan),
                where=np.abs(p_range) > EPS,
            )
            shape[shape_defined, j] = vals[shape_defined].astype(np.float32)

    out = meta.copy()
    out["bid_level"] = bid_level
    out["bid_floor"] = p_first
    out["bid_ceiling"] = p_last
    out["price_range"] = p_range
    out["quantity_span"] = q_span
    out["raw_segment_count"] = counts.astype(np.int16)
    out["effective_segment_count"] = eff_count
    out["quantity_hhi"] = quantity_hhi
    out["tail_uplift_ratio"] = tail_ratio
    out["curve_bend_ratio"] = bend_ratio
    out["flat_curve_flag"] = flat.astype(np.int8)
    out["shape_defined_flag"] = shape_defined.astype(np.int8)

    for j in range(N_SHAPE_GRID):
        out[f"shape_v{j:02d}"] = shape[:, j]

    return out


def process_pair(interval_file, segment_file, output_file, batch_intervals):
    print(f"[intervals] {interval_file}")
    meta = pd.read_csv(interval_file, low_memory=False)
    seg = pd.read_csv(segment_file, low_memory=False)

    required_meta = [
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
    required_seg = ["interval_index", "segment_id", "quantity", "price"]

    mm = [c for c in required_meta if c not in meta.columns]
    ms = [c for c in required_seg if c not in seg.columns]
    if mm or ms:
        raise ValueError(f"Canonical schema missing: interval={mm}, segment={ms}")

    meta = meta.sort_values("interval_index").reset_index(drop=True)
    seg = seg.sort_values(["interval_index", "segment_id"]).reset_index(drop=True)

    idx = meta["interval_index"].to_numpy(np.int64)
    if len(idx) and not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
        raise ValueError("interval_index must be contiguous within one source file.")

    max_segments = (
        int(pd.to_numeric(seg["segment_id"], errors="coerce").max())
        if len(seg) else 1
    )
    max_segments = max(1, max_segments)

    if output_file.exists():
        output_file.unlink()

    first_write = True
    summary = {
        "file": interval_file.name,
        "intervals": len(meta),
        "offer_intervals": 0,
        "shape_defined": 0,
        "flat_curves": 0,
        "single_or_zero_span": 0,
        "block_curves": 0,
        "sloped_curves": 0,
    }

    for start in range(0, len(meta), batch_intervals):
        end = min(start + batch_intervals, len(meta))
        atoms = compute_batch(meta.iloc[start:end].copy(), seg, max_segments)

        atoms.to_csv(
            output_file,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

        has = atoms["raw_segment_count"] > 0
        summary["offer_intervals"] += int(has.sum())
        summary["shape_defined"] += int(atoms["shape_defined_flag"].sum())
        summary["flat_curves"] += int(atoms["flat_curve_flag"].sum())
        summary["single_or_zero_span"] += int(
            (
                (atoms["raw_segment_count"] <= 1)
                | (pd.to_numeric(atoms["quantity_span"], errors="coerce") <= EPS)
            ).sum()
        )
        summary["block_curves"] += int(
            ((atoms["curve_mode"] == "block") & has).sum()
        )
        summary["sloped_curves"] += int(
            ((atoms["curve_mode"] == "sloped") & has).sum()
        )

        print(f"  batch {start:,}:{end:,}")

    print(f"[done] {output_file}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--interval-root", default="data/standardized/intervals")
    p.add_argument("--segment-root", default="data/standardized/bid_segments")
    p.add_argument("--output-root", default="data/processed/interval_strategy_atoms")
    p.add_argument("--results-root", default="results/02_interval_atoms")
    p.add_argument("--batch-intervals", type=int, default=DEFAULT_BATCH_INTERVALS)
    args = p.parse_args()

    interval_dir = Path(args.interval_root) / str(args.year)
    segment_dir = Path(args.segment_root) / str(args.year)
    output_dir = Path(args.output_root) / str(args.year)
    results_dir = Path(args.results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    interval_files = sorted(
        interval_dir.glob(f"standardized_intervals_{args.year}_*.csv")
    )
    if not interval_files:
        raise FileNotFoundError(
            f"No standardized interval files in {interval_dir.resolve()}"
        )

    summaries = []
    for interval_file in interval_files:
        suffix = interval_file.stem.replace("standardized_intervals_", "")
        segment_file = segment_dir / f"standardized_bid_segments_{suffix}.csv"
        if not segment_file.exists():
            raise FileNotFoundError(segment_file)

        output_file = output_dir / f"interval_strategy_atoms_{suffix}.csv"
        summaries.append(
            process_pair(
                interval_file,
                segment_file,
                output_file,
                args.batch_intervals,
            )
        )

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(results_dir / f"file_summary_{args.year}.csv", index=False)

    total = int(sdf["intervals"].sum())
    offers = int(sdf["offer_intervals"].sum())
    pct_offer = lambda c: (
        100.0 * float(sdf[c].sum()) / offers if offers else np.nan
    )

    lines = [
        f"Universal interval strategy atoms summary - {args.year}",
        f"Version: {VERSION}",
        "=" * 72,
        f"Files processed: {len(sdf)}",
        f"All participant-intervals: {total:,}",
        f"Intervals with valid offer: {offers:,}",
        f"Shape-defined offer intervals: {sdf['shape_defined'].sum():,} "
        f"({pct_offer('shape_defined'):.2f}% of valid offers)",
        f"Flat offer curves: {sdf['flat_curves'].sum():,} "
        f"({pct_offer('flat_curves'):.2f}% of valid offers)",
        f"Single/zero-span offers: {sdf['single_or_zero_span'].sum():,} "
        f"({pct_offer('single_or_zero_span'):.2f}% of valid offers)",
        f"Block offers: {sdf['block_curves'].sum():,} "
        f"({pct_offer('block_curves'):.2f}% of valid offers)",
        f"Sloped offers: {sdf['sloped_curves'].sum():,} "
        f"({pct_offer('sloped_curves'):.2f}% of valid offers)",
        "",
        "Core universal atoms:",
        "  bid_level",
        "  bid_floor",
        "  bid_ceiling",
        "  price_range",
        "  quantity_span",
        "  raw_segment_count",
        "  effective_segment_count",
        "  quantity_hhi",
        "  tail_uplift_ratio",
        "  curve_bend_ratio",
        "  flat_curve_flag",
        "  shape_v00 ... shape_v20",
        "",
        "Market/history-dependent strategy features are intentionally deferred.",
        f"Output directory: {output_dir.resolve()}",
    ]

    (results_dir / f"summary_{args.year}.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
