#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_build_daily_strategy_core.py

Raw PJM Energy Market Generation Offers -> compact daily strategy table.

Only the quantities required by the final strategy profile are calculated:
- price level
- self-history adjustment bias / magnitude
- quantity HHI
- effective segment count
- flat-curve rate
- tail uplift
- curve bend
- 21-point normalized shape

No candidate-feature zoo, no adjacent-switch variables, no redundant QC fields.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    SHAPE_COLS, DAILY_SHAPE_COLS, ensure_dir,
    first_existing, parse_bool
)

EPS = 1e-12
PRICE_EPS = 1e-9
GRID = np.linspace(0.0, 1.0, 21)

def clean_points(q, p):
    ok = np.isfinite(q) & np.isfinite(p)
    q = np.asarray(q, dtype=float)[ok]
    p = np.asarray(p, dtype=float)[ok]
    if len(q) == 0:
        return q, p

    order = np.argsort(q, kind="mergesort")
    q, p = q[order], p[order]

    uq, up = [], []
    j = 0
    while j < len(q):
        k = j + 1
        while k < len(q) and abs(q[k] - q[j]) <= EPS:
            k += 1
        uq.append(q[j])
        up.append(np.nanmax(p[j:k]))
        j = k

    return np.asarray(uq, float), np.asarray(up, float)

def step_eval(xp, fp, x):
    idx = np.searchsorted(xp, x, side="right") - 1
    idx = np.clip(idx, 0, len(fp) - 1)
    y = fp[idx]
    if len(fp):
        y[np.isclose(x, 1.0)] = fp[-1]
    return y

def interval_features(q, p, sloped):
    q, p = clean_points(q, p)
    out = {
        "bid_level": np.nan,
        "quantity_hhi": np.nan,
        "effective_segment_count": np.nan,
        "flat_curve_flag": np.nan,
        "tail_uplift_ratio": np.nan,
        "curve_bend_ratio": np.nan,
    }
    out.update({c: np.nan for c in SHAPE_COLS})

    if len(q) == 0:
        return out

    if len(q) == 1 or q[-1] - q[0] <= EPS:
        out["bid_level"] = float(p[0])
        out["quantity_hhi"] = 1.0
        out["effective_segment_count"] = 1.0
        out["flat_curve_flag"] = 1.0
        out["tail_uplift_ratio"] = 0.0
        out["curve_bend_ratio"] = 0.0
        return out

    span = q[-1] - q[0]
    x = (q - q[0]) / span
    dx = np.diff(x)

    if sloped:
        level = np.sum(0.5 * (p[:-1] + p[1:]) * dx)
    else:
        level = np.sum(p[:-1] * dx)

    out["bid_level"] = float(level)
    out["quantity_hhi"] = float(np.sum(dx ** 2))
    out["effective_segment_count"] = float(
        1 + np.sum(np.abs(np.diff(p)) > PRICE_EPS)
    )

    prange = p[-1] - p[0]
    flat = abs(prange) <= PRICE_EPS
    out["flat_curve_flag"] = float(flat)

    if flat:
        out["tail_uplift_ratio"] = 0.0
        out["curve_bend_ratio"] = 0.0
        return out

    if sloped:
        pg = np.interp(GRID, x, p)
    else:
        pg = step_eval(x, p, GRID)

    p0 = float(pg[0])
    p02 = float(pg[4])
    p08 = float(pg[16])
    p1 = float(pg[-1])

    denom = abs(p1 - p0) + EPS
    out["tail_uplift_ratio"] = (p1 - p08) / denom
    out["curve_bend_ratio"] = ((p1 - p08) - (p02 - p0)) / denom

    shape = (pg - p0) / (p1 - p0)
    for c, v in zip(SHAPE_COLS, shape):
        out[c] = float(v)

    return out

def _canon(name):
    """Normalize CSV header names without changing the original column object."""
    s = str(name).strip().lower()
    for ch in (" ", "-", "/", ".", "(", ")"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _pick(columns, candidates, required=True):
    lookup = {_canon(c): c for c in columns}
    for name in candidates:
        key = _canon(name)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(
            f"Cannot find any of {candidates}. "
            f"Available columns: {list(columns)}"
        )
    return None


def detect_columns(columns):
    # PJM Data Miner energy_market_offers official fields:
    # bid_datetime_beginning_utc / bid_datetime_beginning_ept
    unit_col = _pick(
        columns,
        [
            "unit_code",
            "unit",
            "resource_name",
            "resource",
            "generator_id",
        ],
    )

    utc_col = _pick(
        columns,
        [
            "bid_datetime_beginning_utc",
            "datetime_beginning_utc",
            "timestamp_utc",
            "datetime_utc",
            "utc_timestamp",
        ],
        required=False,
    )

    local_col = _pick(
        columns,
        [
            "bid_datetime_beginning_ept",
            "datetime_beginning_ept",
            "timestamp_ept",
            "timestamp_local",
            "datetime_beginning_est",
            "local_timestamp",
        ],
        required=False,
    )

    if utc_col is None and local_col is None:
        raise KeyError(
            "Cannot find PJM bid time column. Expected "
            "'bid_datetime_beginning_utc' or "
            "'bid_datetime_beginning_ept'. "
            f"Available columns: {list(columns)}"
        )

    slope_col = _pick(
        columns,
        [
            "bid_slope_flag",
            "usebidslope",
            "use_bid_slope",
            "slope_flag",
            "bid_slope",
        ],
        required=False,
    )

    lookup = {_canon(c): c for c in columns}
    mw_cols, bid_cols = [], []

    # Support both the current 20-point files and older 10-point PJM files.
    for k in range(1, 21):
        mw = lookup.get(f"mw{k}")
        bid = lookup.get(f"bid{k}")
        if mw is not None and bid is not None:
            mw_cols.append(mw)
            bid_cols.append(bid)

    if not mw_cols:
        raise KeyError(
            "Cannot find paired MW/BID columns. Expected MW1..MW10/20 "
            f"and BID1..BID10/20. Available columns: {list(columns)}"
        )

    return unit_col, utc_col, local_col, slope_col, mw_cols, bid_cols

def _detect_datetime_format(series):
    """
    Detect the common PJM timestamp format once from the first non-null sample.

    Typical Data Miner exports are one of:
      01/01/2025 12:00:00 AM
      01/01/2025 00:00:00
      2025-01-01 00:00:00
      2025-01-01T00:00:00
    """
    s = series.dropna().astype(str)
    if s.empty:
        return None

    sample = s.iloc[0].strip()

    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[APap][Mm]", sample):
        return "%m/%d/%Y %I:%M:%S %p"

    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}", sample):
        return "%m/%d/%Y %H:%M:%S"

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}:\d{2}", sample):
        return "%Y-%m-%d %H:%M:%S"

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{2}:\d{2}", sample):
        return "%Y-%m-%dT%H:%M:%S"

    return None


def _parse_datetime_fast(series, utc=False):
    fmt = _detect_datetime_format(series)

    if fmt is not None:
        return pd.to_datetime(
            series,
            format=fmt,
            errors="coerce",
            utc=utc,
        )

    # Pandas >= 2.0: explicit mixed mode avoids the repeated warning and
    # is still preferable to implicit dateutil fallback.
    try:
        return pd.to_datetime(
            series,
            format="mixed",
            errors="coerce",
            utc=utc,
        )
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas.
        return pd.to_datetime(
            series,
            errors="coerce",
            utc=utc,
        )


def add_time_columns(df, utc_col, local_col):
    if utc_col is not None:
        utc = _parse_datetime_fast(df[utc_col], utc=True)
    else:
        local_tmp = _parse_datetime_fast(df[local_col], utc=False)
        utc = local_tmp.dt.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC")

    if local_col is not None:
        local = _parse_datetime_fast(df[local_col], utc=False)

        # PJM EPT values are normally timezone-naive local timestamps.
        # If the source happens to be timezone-aware, normalize to New York.
        try:
            if local.dt.tz is not None:
                local = (
                    local.dt.tz_convert("America/New_York")
                    .dt.tz_localize(None)
                )
        except AttributeError:
            pass
    else:
        local = (
            utc.dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

    df = df.copy()
    df["_timestamp_utc"] = utc
    df["_timestamp_local"] = local
    df["_local_date"] = local.dt.normalize()
    df["_local_slot"] = (
        local.dt.hour * 3600
        + local.dt.minute * 60
        + local.dt.second
    )
    return df

def aggregate_daily(frame):
    if frame.empty:
        return frame

    agg = {
        "bid_level": "median",
        "adjustment_bias": "median",
        "adjustment_magnitude": "median",
        "quantity_hhi": "median",
        "effective_segment_count": "median",
        "flat_curve_flag": "mean",
        "tail_uplift_ratio": "median",
        "curve_bend_ratio": "median",
    }
    agg.update({c: "median" for c in SHAPE_COLS})

    d = (
        frame.groupby(["participant_id", "local_date"], sort=False)
        .agg(agg)
        .reset_index()
    )
    d = d.rename(columns={
        "bid_level": "daily_bid_level",
        "adjustment_bias": "daily_adjustment_bias",
        "adjustment_magnitude": "daily_adjustment_magnitude",
        "quantity_hhi": "daily_quantity_hhi",
        "effective_segment_count": "daily_effective_segment_count",
        "flat_curve_flag": "daily_flat_curve_rate",
        "tail_uplift_ratio": "daily_tail_uplift_ratio",
        "curve_bend_ratio": "daily_curve_bend_ratio",
        **{a: b for a, b in zip(SHAPE_COLS, DAILY_SHAPE_COLS)},
    })
    return d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--raw-root",
        default="data/raw/energy_market_offers",
    )
    p.add_argument(
        "--out-root",
        default="data/processed/final_clean/daily",
    )
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--same-slot-window", type=int, default=30)
    p.add_argument("--min-same-slot-history", type=int, default=5)
    args = p.parse_args()

    raw_dir = Path(args.raw_root) / str(args.year)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {raw_dir}")

    out_dir = ensure_dir(Path(args.out_root) / str(args.year))
    out_file = out_dir / f"daily_strategy_core_{args.year}.csv"
    if out_file.exists():
        out_file.unlink()

    history = defaultdict(lambda: deque(maxlen=args.same_slot_window))
    carry = pd.DataFrame()
    wrote_header = False

    for file in files:
        print(f"[file] {file.name}")
        header = pd.read_csv(file, nrows=0)
        unit_col, utc_col, local_col, slope_col, mw_cols, bid_cols = detect_columns(header.columns)

        print(
            "  [columns] "
            f"unit={unit_col}, utc={utc_col}, local={local_col}, "
            f"slope={slope_col}, curve_points={len(mw_cols)}"
        )

        usecols = [unit_col] + mw_cols + bid_cols
        for c in [utc_col, local_col, slope_col]:
            if c is not None and c not in usecols:
                usecols.append(c)

        for chunk_no, chunk in enumerate(
            pd.read_csv(
                file,
                usecols=usecols,
                chunksize=args.chunksize,
                low_memory=False,
            ),
            start=1,
        ):
            print(
                f"  [chunk {chunk_no:03d}] "
                f"raw_rows={len(chunk):,}",
                flush=True,
            )

            chunk = add_time_columns(chunk, utc_col, local_col)
            chunk = chunk.dropna(
                subset=["_timestamp_local", "_local_date", unit_col]
            )
            chunk = chunk.sort_values("_timestamp_utc", kind="mergesort")

            rows = []
            qmat = chunk[mw_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            pmat = chunk[bid_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            units = chunk[unit_col].astype(str).to_numpy()
            slots = chunk["_local_slot"].to_numpy()
            dates = chunk["_local_date"].to_numpy()

            if slope_col is not None:
                slopes = chunk[slope_col].map(parse_bool).to_numpy()
            else:
                slopes = np.zeros(len(chunk), dtype=bool)

            for j in range(len(chunk)):
                feat = interval_features(qmat[j], pmat[j], bool(slopes[j]))
                level = feat["bid_level"]
                key = (units[j], int(slots[j]))
                h = history[key]

                if np.isfinite(level) and len(h) >= args.min_same_slot_history:
                    baseline = float(np.median(np.asarray(h, dtype=float)))
                    residual = level - baseline
                else:
                    residual = np.nan

                if np.isfinite(level):
                    h.append(float(level))

                row = {
                    "participant_id": units[j],
                    "local_date": pd.Timestamp(dates[j]),
                    "bid_level": level,
                    "adjustment_bias": residual,
                    "adjustment_magnitude": abs(residual) if np.isfinite(residual) else np.nan,
                    "quantity_hhi": feat["quantity_hhi"],
                    "effective_segment_count": feat["effective_segment_count"],
                    "flat_curve_flag": feat["flat_curve_flag"],
                    "tail_uplift_ratio": feat["tail_uplift_ratio"],
                    "curve_bend_ratio": feat["curve_bend_ratio"],
                }
                for c in SHAPE_COLS:
                    row[c] = feat[c]
                rows.append(row)

            cur = pd.DataFrame(rows)
            if not carry.empty:
                cur = pd.concat([carry, cur], ignore_index=True)

            if cur.empty:
                continue

            max_date = cur["local_date"].max()
            done = cur[cur["local_date"] < max_date]
            carry = cur[cur["local_date"] == max_date].copy()

            if not done.empty:
                daily = aggregate_daily(done)
                daily.to_csv(
                    out_file,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                    encoding="utf-8-sig",
                )
                wrote_header = True

            print(
                f"  [chunk {chunk_no:03d}] processed, "
                f"carry_rows={len(carry):,}",
                flush=True,
            )

    if not carry.empty:
        daily = aggregate_daily(carry)
        daily.to_csv(
            out_file,
            mode="a",
            header=not wrote_header,
            index=False,
            encoding="utf-8-sig",
        )

    print(f"Done: {out_file}")

if __name__ == "__main__":
    main()
