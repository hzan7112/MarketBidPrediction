#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_build_long_term_strategy_profile.py

Daily core -> final compact long-term strategy profile.

Exactly nine continuous LT features:
1. lt_bid_level
2. lt_adjustment_magnitude
3. lt_strategy_persistence
4. lt_quantity_hhi
5. lt_effective_segment_count
6. lt_flat_curve_rate
7. lt_tail_uplift_ratio
8. lt_curve_bend_ratio
9. lt_shape_variability
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import DAILY_SHAPE_COLS, LT_SHAPE_COLS, ensure_dir, rmse

def persistence(g):
    g = g.sort_values("local_date")
    x = pd.to_numeric(g["daily_adjustment_bias"], errors="coerce")
    d = pd.to_datetime(g["local_date"])

    vals_a, vals_b = [], []
    for j in range(1, len(g)):
        if (d.iloc[j] - d.iloc[j - 1]).days != 1:
            continue
        a, b = x.iloc[j - 1], x.iloc[j]
        if np.isfinite(a) and np.isfinite(b):
            vals_a.append(a)
            vals_b.append(b)

    if len(vals_a) < 5:
        return np.nan

    a = np.asarray(vals_a, float)
    b = np.asarray(vals_b, float)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--daily-root",
        default="data/processed/final_clean/daily",
    )
    p.add_argument(
        "--out-root",
        default="data/processed/final_clean/long_term",
    )
    args = p.parse_args()

    src = (
        Path(args.daily_root) / str(args.year)
        / f"daily_strategy_core_{args.year}.csv"
    )
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = ensure_dir(Path(args.out_root) / str(args.year))
    out_file = out_dir / f"long_term_strategy_profile_{args.year}.csv"

    df = pd.read_csv(src, low_memory=False)
    df["local_date"] = pd.to_datetime(df["local_date"])

    rows = []
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("local_date").copy()

        row = {
            "participant_id": pid,
            "active_days": int(g["local_date"].nunique()),
            "lt_bid_level": pd.to_numeric(g["daily_bid_level"], errors="coerce").median(),
            "lt_adjustment_magnitude": pd.to_numeric(
                g["daily_adjustment_magnitude"], errors="coerce"
            ).median(),
            "lt_strategy_persistence": persistence(g),
            "lt_quantity_hhi": pd.to_numeric(
                g["daily_quantity_hhi"], errors="coerce"
            ).median(),
            "lt_effective_segment_count": pd.to_numeric(
                g["daily_effective_segment_count"], errors="coerce"
            ).median(),
            "lt_flat_curve_rate": pd.to_numeric(
                g["daily_flat_curve_rate"], errors="coerce"
            ).median(),
            "lt_tail_uplift_ratio": pd.to_numeric(
                g["daily_tail_uplift_ratio"], errors="coerce"
            ).median(),
            "lt_curve_bend_ratio": pd.to_numeric(
                g["daily_curve_bend_ratio"], errors="coerce"
            ).median(),
        }

        shape = g[DAILY_SHAPE_COLS].apply(pd.to_numeric, errors="coerce")
        complete = shape.notna().all(axis=1)
        shape_valid = shape[complete]

        if len(shape_valid):
            proto = shape_valid.median(axis=0).to_numpy(float)
            dev = [
                rmse(v, proto)
                for v in shape_valid.to_numpy(float)
            ]
            row["lt_shape_variability"] = float(np.median(dev))
            row["shape_defined_days"] = int(len(shape_valid))
            for c, v in zip(LT_SHAPE_COLS, proto):
                row[c] = float(v)
        else:
            row["lt_shape_variability"] = np.nan
            row["shape_defined_days"] = 0
            for c in LT_SHAPE_COLS:
                row[c] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    core = [
        "lt_bid_level",
        "lt_adjustment_magnitude",
        "lt_strategy_persistence",
        "lt_quantity_hhi",
        "lt_effective_segment_count",
        "lt_flat_curve_rate",
        "lt_tail_uplift_ratio",
        "lt_curve_bend_ratio",
        "lt_shape_variability",
    ]
    out["lt_nonmissing_count"] = out[core].notna().sum(axis=1)
    out["lt_ready_flag"] = (out["lt_nonmissing_count"] >= 7).astype(int)

    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Participants: {len(out):,}")
    print(f"LT ready: {out['lt_ready_flag'].sum():,}")
    print(f"Done: {out_file}")

if __name__ == "__main__":
    main()
