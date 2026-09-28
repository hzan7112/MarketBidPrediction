#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_build_long_term_strategy_profile.py

Historical daily strategy core -> full-reference-period 9-D LT profile.

The LT profile is a participant-level slow variable built from the complete
historical reference period. It is not recomputed from target-year recent bids.

Formal LT dimensions:
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

from strategy_core import (
    DAILY_SHAPE_COLS,
    LT_FEATURES,
    LT_SHAPE_COLS,
    ensure_dir,
    rmse,
)


def persistence(g):
    g = g.sort_values("local_date")

    x = pd.to_numeric(
        g["daily_adjustment_bias"],
        errors="coerce",
    )

    d = pd.to_datetime(
        g["local_date"]
    )

    a_vals = []
    b_vals = []

    for j in range(1, len(g)):
        if (
            d.iloc[j]
            - d.iloc[j - 1]
        ).days != 1:
            continue

        a = x.iloc[j - 1]
        b = x.iloc[j]

        if np.isfinite(a) and np.isfinite(b):
            a_vals.append(a)
            b_vals.append(b)

    if len(a_vals) < 5:
        return np.nan

    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)

    if (
        np.std(a) <= 1e-12
        or np.std(b) <= 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def build_one_participant(pid, g):
    g = (
        g.sort_values("local_date")
        .copy()
    )

    row = {
        "participant_id": str(pid),
        "active_days": int(
            g["local_date"].nunique()
        ),
        "lt_bid_level": pd.to_numeric(
            g["daily_bid_level"],
            errors="coerce",
        ).median(),
        "lt_adjustment_magnitude": pd.to_numeric(
            g["daily_adjustment_magnitude"],
            errors="coerce",
        ).median(),
        "lt_strategy_persistence": persistence(g),
        "lt_quantity_hhi": pd.to_numeric(
            g["daily_quantity_hhi"],
            errors="coerce",
        ).median(),
        "lt_effective_segment_count": pd.to_numeric(
            g["daily_effective_segment_count"],
            errors="coerce",
        ).median(),
        "lt_flat_curve_rate": pd.to_numeric(
            g["daily_flat_curve_rate"],
            errors="coerce",
        ).median(),
        "lt_tail_uplift_ratio": pd.to_numeric(
            g["daily_tail_uplift_ratio"],
            errors="coerce",
        ).median(),
        "lt_curve_bend_ratio": pd.to_numeric(
            g["daily_curve_bend_ratio"],
            errors="coerce",
        ).median(),
    }

    shape = (
        g[DAILY_SHAPE_COLS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    complete = (
        shape.notna()
        .all(axis=1)
    )

    shape_valid = shape.loc[
        complete
    ]

    if len(shape_valid):
        prototype = (
            shape_valid
            .median(axis=0)
            .to_numpy(float)
        )

        deviation = np.asarray(
            [
                rmse(v, prototype)
                for v in shape_valid.to_numpy(float)
            ],
            dtype=float,
        )

        row[
            "lt_shape_variability"
        ] = float(
            np.nanmedian(deviation)
        )

        row[
            "shape_defined_days"
        ] = int(
            len(shape_valid)
        )

        for c, v in zip(
            LT_SHAPE_COLS,
            prototype,
        ):
            row[c] = float(v)

    else:
        row[
            "lt_shape_variability"
        ] = np.nan

        row[
            "shape_defined_days"
        ] = 0

        for c in (
            LT_SHAPE_COLS
        ):
            row[c] = np.nan

    row[
        "lt_nonmissing_count"
    ] = int(
        sum(
            pd.notna(
                row[f]
            )
            for f in (
                LT_FEATURES
            )
        )
    )

    row[
        "lt_ready_flag"
    ] = int(
        row[
            "lt_nonmissing_count"
        ] >= 7
    )

    return row


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
            "final_clean/long_term"
        ),
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

    out_file = (
        out_dir
        / f"long_term_strategy_profile_{args.year}.csv"
    )

    df = pd.read_csv(
        src,
        low_memory=False,
    )

    df[
        "participant_id"
    ] = (
        df["participant_id"]
        .astype(str)
    )

    df[
        "local_date"
    ] = (
        pd.to_datetime(
            df["local_date"]
        )
        .dt.normalize()
    )

    rows = []

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
        rows.append(
            build_one_participant(
                pid,
                g,
            )
        )

        if (
            idx == 1
            or idx % 100 == 0
            or idx == len(groups)
        ):
            print(
                f"[02] participants "
                f"{idx:,}/{len(groups):,}",
                flush=True,
            )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        out_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print(
        "Long-term historical strategy profile complete"
    )
    print("=" * 72)
    print(
        f"Reference year: {args.year}"
    )
    print(
        f"Participants: {len(out):,}"
    )
    print(
        f"LT ready: "
        f"{int(out['lt_ready_flag'].sum()):,}"
    )
    print(
        f"Done: {out_file}"
    )


if __name__ == "__main__":
    main()
