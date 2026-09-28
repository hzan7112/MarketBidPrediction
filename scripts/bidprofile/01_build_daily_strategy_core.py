#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_build_daily_strategy_core_v4.py

PJM raw Energy Market Offers -> daily strategy core.

Changes relative to the previous final-clean 01:
1. Explicit curve preprocessing contract:
   - MW and BID must both be finite;
   - optional MW sentinel values are removed;
   - zero/negative BID values are preserved;
   - valid points are re-packed, sorted by MW, duplicate MW keeps max BID;
   - <2 distinct valid MW points or zero MW span => invalid interval;
   - missing intervals are never filled/interpolated.
2. Flat curve means all valid prices are equal within tolerance, not merely
   equal endpoint prices.
3. Adds historical intraday behavior atoms:
   - daily_curve_type_count
   - daily_curve_switch_rate
   - daily_intraday_price_range
4. Simultaneously accumulates participant x local-slot historical means.
   These slot profiles are later used to construct target-slot historical
   context without using target-year recent bids.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_core import (
    SHAPE_COLS,
    DAILY_SHAPE_COLS,
    ensure_dir,
    parse_bool,
)

EPS = 1e-12
PRICE_EPS = 1e-9
GRID = np.linspace(0.0, 1.0, 21)


def clean_points(q, p, mw_sentinels=(), mw_sentinel_tol=1e-9):
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    ok = np.isfinite(q) & np.isfinite(p)

    for sentinel in mw_sentinels:
        if np.isfinite(sentinel):
            ok &= ~np.isclose(
                q,
                float(sentinel),
                rtol=0.0,
                atol=float(mw_sentinel_tol),
            )

    q = q[ok]
    p = p[ok]
    raw_valid_point_count = int(len(q))

    if raw_valid_point_count == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            0,
        )

    order = np.argsort(q, kind="mergesort")
    q = q[order]
    p = p[order]

    uq, up = [], []
    j = 0

    while j < len(q):
        k = j + 1
        while k < len(q) and abs(q[k] - q[j]) <= EPS:
            k += 1

        uq.append(q[j])
        up.append(np.max(p[j:k]))
        j = k

    return (
        np.asarray(uq, dtype=float),
        np.asarray(up, dtype=float),
        raw_valid_point_count,
    )


def curve_key(q, p, sloped):
    """Deterministic exact cleaned-curve key used only within daily switching."""
    h = hashlib.blake2b(digest_size=12)
    h.update(b"S" if sloped else b"B")
    h.update(np.asarray(q, dtype="<f8").tobytes())
    h.update(np.asarray(p, dtype="<f8").tobytes())
    return h.hexdigest()


def step_eval(xp, fp, x):
    idx = np.searchsorted(xp, x, side="right") - 1
    idx = np.clip(idx, 0, len(fp) - 1)
    y = fp[idx].astype(float, copy=True)
    if len(fp):
        y[np.isclose(x, 1.0)] = fp[-1]
    return y


def interval_features(
    q,
    p,
    sloped,
    mw_sentinels=(),
    mw_sentinel_tol=1e-9,
):
    q, p, raw_valid_point_count = clean_points(
        q,
        p,
        mw_sentinels=mw_sentinels,
        mw_sentinel_tol=mw_sentinel_tol,
    )

    out = {
        "valid_curve_flag": 0,
        "invalid_reason": "",
        "raw_valid_point_count": raw_valid_point_count,
        "clean_point_count": int(len(q)),
        "shape_defined_flag": 0,
        "curve_key": "",
        "p_min": np.nan,
        "p_max": np.nan,
        "bid_level": np.nan,
        "quantity_hhi": np.nan,
        "effective_segment_count": np.nan,
        "flat_curve_flag": np.nan,
        "tail_uplift_ratio": np.nan,
        "curve_bend_ratio": np.nan,
    }
    out.update({c: np.nan for c in SHAPE_COLS})

    if len(q) == 0:
        out["invalid_reason"] = "no_valid_points"
        return out

    if len(q) < 2:
        out["invalid_reason"] = "single_clean_point"
        return out

    span = float(q[-1] - q[0])
    if not np.isfinite(span) or span <= EPS:
        out["invalid_reason"] = "zero_quantity_span"
        return out

    out["valid_curve_flag"] = 1
    out["curve_key"] = curve_key(q, p, sloped)
    out["p_min"] = float(np.min(p))
    out["p_max"] = float(np.max(p))

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

    p_range = float(np.max(p) - np.min(p))
    flat = p_range <= PRICE_EPS
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
    endpoint_span = p1 - p0

    # Still a valid curve, but endpoint-normalized shape is undefined.
    if abs(endpoint_span) <= PRICE_EPS:
        return out

    denom = abs(endpoint_span) + EPS

    out["tail_uplift_ratio"] = (p1 - p08) / denom
    out["curve_bend_ratio"] = (
        (p1 - p08) - (p02 - p0)
    ) / denom

    shape = (pg - p0) / endpoint_span
    if not np.all(np.isfinite(shape)):
        return out

    out["shape_defined_flag"] = 1

    for c, v in zip(SHAPE_COLS, shape):
        out[c] = float(v)

    return out


def _canon(name):
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
            "'bid_datetime_beginning_ept'."
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

    for k in range(1, 21):
        mw = lookup.get(f"mw{k}")
        bid = lookup.get(f"bid{k}")
        if mw is not None and bid is not None:
            mw_cols.append(mw)
            bid_cols.append(bid)

    if not mw_cols:
        raise KeyError("Cannot find paired MW/BID columns.")

    return unit_col, utc_col, local_col, slope_col, mw_cols, bid_cols


def _detect_datetime_format(series):
    s = series.dropna().astype(str)
    if s.empty:
        return None

    sample = s.iloc[0].strip()

    if re.fullmatch(
        r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[APap][Mm]",
        sample,
    ):
        return "%m/%d/%Y %I:%M:%S %p"

    if re.fullmatch(
        r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}",
        sample,
    ):
        return "%m/%d/%Y %H:%M:%S"

    if re.fullmatch(
        r"\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}:\d{2}",
        sample,
    ):
        return "%Y-%m-%d %H:%M:%S"

    if re.fullmatch(
        r"\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{2}:\d{2}",
        sample,
    ):
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

    try:
        return pd.to_datetime(
            series,
            format="mixed",
            errors="coerce",
            utc=utc,
        )
    except (TypeError, ValueError):
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
        utc = (
            local_tmp.dt.tz_localize(
                "America/New_York",
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
            .dt.tz_convert("UTC")
        )

    if local_col is not None:
        local = _parse_datetime_fast(df[local_col], utc=False)
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


def _intraday_summary(g):
    g = g.sort_values(
        ["timestamp_utc", "local_slot_seconds"],
        kind="mergesort",
    )

    keys = g["curve_key"].astype(str).to_numpy()
    n = len(keys)

    curve_type_count = int(pd.Series(keys).nunique())

    if n >= 2:
        switch_count = int(np.sum(keys[1:] != keys[:-1]))
        switch_rate = float(switch_count / (n - 1))
    else:
        switch_count = 0
        switch_rate = np.nan

    # Price range is defined across distinct curve types, not duplicated hours.
    unique_curves = (
        g.drop_duplicates(subset=["curve_key"], keep="first")
    )

    if len(unique_curves) <= 1:
        intraday_price_range = 0.0
    else:
        rmax = (
            pd.to_numeric(unique_curves["p_max"], errors="coerce").max()
            - pd.to_numeric(unique_curves["p_max"], errors="coerce").min()
        )
        rmin = (
            pd.to_numeric(unique_curves["p_min"], errors="coerce").max()
            - pd.to_numeric(unique_curves["p_min"], errors="coerce").min()
        )
        rw = (
            pd.to_numeric(unique_curves["bid_level"], errors="coerce").max()
            - pd.to_numeric(unique_curves["bid_level"], errors="coerce").min()
        )
        intraday_price_range = float(np.nanmean([rmax, rmin, rw]))

    return pd.Series(
        {
            "daily_curve_type_count": curve_type_count,
            "daily_curve_switch_count": switch_count,
            "daily_curve_switch_rate": switch_rate,
            "daily_intraday_price_range": intraday_price_range,
        }
    )


def aggregate_daily(frame):
    if frame.empty:
        return pd.DataFrame()

    agg = {
        "valid_interval_count": "sum",
        "shape_defined_flag": "sum",
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

    base = (
        frame.groupby(
            ["participant_id", "local_date"],
            sort=False,
        )
        .agg(agg)
        .reset_index()
    )

    intraday_cols = [
        "timestamp_utc",
        "local_slot_seconds",
        "curve_key",
        "p_min",
        "p_max",
        "bid_level",
    ]

    intra = (
        frame.groupby(
            ["participant_id", "local_date"],
            sort=False,
            group_keys=False,
        )[intraday_cols]
        .apply(_intraday_summary)
        .reset_index()
    )

    d = base.merge(
        intra,
        on=["participant_id", "local_date"],
        how="left",
        validate="one_to_one",
    )

    d = d.rename(
        columns={
            "valid_interval_count": "daily_valid_interval_count",
            "shape_defined_flag": "daily_shape_defined_interval_count",
            "bid_level": "daily_bid_level",
            "adjustment_bias": "daily_adjustment_bias",
            "adjustment_magnitude": "daily_adjustment_magnitude",
            "quantity_hhi": "daily_quantity_hhi",
            "effective_segment_count": "daily_effective_segment_count",
            "flat_curve_flag": "daily_flat_curve_rate",
            "tail_uplift_ratio": "daily_tail_uplift_ratio",
            "curve_bend_ratio": "daily_curve_bend_ratio",
            **{
                a: b
                for a, b in zip(SHAPE_COLS, DAILY_SHAPE_COLS)
            },
        }
    )

    return d




SLOT_FEATURES = [
    "bid_level",
    "adjustment_magnitude",
    "quantity_hhi",
    "effective_segment_count",
    "flat_curve_flag",
    "tail_uplift_ratio",
    "curve_bend_ratio",
]


def update_slot_accumulator(acc, participant_id, local_slot_seconds, row):
    key = (str(participant_id), int(local_slot_seconds))

    if key not in acc:
        acc[key] = {
            "observation_count": 0,
            **{f"{c}_sum": 0.0 for c in SLOT_FEATURES},
            **{f"{c}_count": 0 for c in SLOT_FEATURES},
        }

    rec = acc[key]
    rec["observation_count"] += 1

    for c in SLOT_FEATURES:
        v = row.get(c, np.nan)
        if np.isfinite(v):
            rec[f"{c}_sum"] += float(v)
            rec[f"{c}_count"] += 1


def slot_accumulator_to_frame(acc):
    rows = []

    for (pid, slot), rec in acc.items():
        row = {
            "participant_id": pid,
            "local_slot_seconds": int(slot),
            "local_hour": int(slot // 3600),
            "slot_observation_count": int(rec["observation_count"]),
        }

        for c in SLOT_FEATURES:
            n = rec[f"{c}_count"]
            row[f"slot_{c}_mean"] = (
                rec[f"{c}_sum"] / n
                if n > 0
                else np.nan
            )
            row[f"slot_{c}_count"] = int(n)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["participant_id", "local_slot_seconds"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


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
    p.add_argument(
        "--mw-sentinel",
        type=float,
        action="append",
        default=[],
        help=(
            "MW missing-value sentinel. Repeat for multiple values. "
            "No sentinel is assumed when omitted."
        ),
    )
    p.add_argument(
        "--mw-sentinel-tol",
        type=float,
        default=1e-9,
    )
    args = p.parse_args()

    raw_dir = Path(args.raw_root) / str(args.year)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {raw_dir}")

    out_dir = ensure_dir(Path(args.out_root) / str(args.year))
    out_file = out_dir / f"daily_strategy_core_{args.year}.csv"
    slot_file = out_dir / f"intraday_slot_core_{args.year}.csv"

    if out_file.exists():
        out_file.unlink()
    if slot_file.exists():
        slot_file.unlink()

    print(
        "[preprocess] "
        f"MW sentinels={args.mw_sentinel if args.mw_sentinel else 'none'}"
    )
    print(
        "[preprocess] BID=0 and BID<0 are retained; "
        "invalid intervals are not filled/interpolated."
    )

    history = defaultdict(
        lambda: deque(maxlen=args.same_slot_window)
    )

    carry = pd.DataFrame()
    wrote_header = False

    # Exact streaming sums/counts for participant x local-slot historical
    # profiles. The key count is small relative to raw interval rows.
    slot_accumulator = {}

    total_raw = 0
    total_valid = 0
    invalid_reasons = Counter()

    for file in files:
        print(f"[file] {file.name}")

        header = pd.read_csv(file, nrows=0)
        (
            unit_col,
            utc_col,
            local_col,
            slope_col,
            mw_cols,
            bid_cols,
        ) = detect_columns(header.columns)

        print(
            "  [columns] "
            f"unit={unit_col}, utc={utc_col}, local={local_col}, "
            f"slope={slope_col}, curve_points={len(mw_cols)}"
        )

        usecols = [unit_col] + mw_cols + bid_cols
        for c in [utc_col, local_col, slope_col]:
            if c is not None and c not in usecols:
                usecols.append(c)

        reader = pd.read_csv(
            file,
            usecols=usecols,
            chunksize=args.chunksize,
            low_memory=False,
        )

        for chunk_no, chunk in enumerate(reader, start=1):
            total_raw += len(chunk)

            print(
                f"  [chunk {chunk_no:03d}] raw_rows={len(chunk):,}",
                flush=True,
            )

            chunk = add_time_columns(chunk, utc_col, local_col)

            before = len(chunk)
            chunk = chunk.dropna(
                subset=["_timestamp_local", "_local_date", unit_col]
            )
            invalid_time = before - len(chunk)
            invalid_reasons["invalid_time_or_unit"] += invalid_time

            chunk = chunk.sort_values(
                "_timestamp_utc",
                kind="mergesort",
            )

            qmat = (
                chunk[mw_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(float)
            )
            pmat = (
                chunk[bid_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(float)
            )
            units = chunk[unit_col].astype(str).to_numpy()
            slots = chunk["_local_slot"].to_numpy()
            dates = chunk["_local_date"].to_numpy()
            utc_values = chunk["_timestamp_utc"].to_numpy()

            if slope_col is not None:
                slopes = chunk[slope_col].map(parse_bool).to_numpy()
            else:
                slopes = np.zeros(len(chunk), dtype=bool)

            rows = []
            chunk_invalid = Counter()

            for j in range(len(chunk)):
                feat = interval_features(
                    qmat[j],
                    pmat[j],
                    bool(slopes[j]),
                    mw_sentinels=args.mw_sentinel,
                    mw_sentinel_tol=args.mw_sentinel_tol,
                )

                if feat["valid_curve_flag"] != 1:
                    reason = feat["invalid_reason"] or "invalid_curve"
                    invalid_reasons[reason] += 1
                    chunk_invalid[reason] += 1
                    continue

                total_valid += 1

                level = feat["bid_level"]
                key = (units[j], int(slots[j]))
                h = history[key]

                if (
                    np.isfinite(level)
                    and len(h) >= args.min_same_slot_history
                ):
                    baseline = float(
                        np.median(np.asarray(h, dtype=float))
                    )
                    residual = level - baseline
                else:
                    residual = np.nan

                if np.isfinite(level):
                    h.append(float(level))

                row = {
                    "participant_id": units[j],
                    "local_date": pd.Timestamp(dates[j]),
                    "timestamp_utc": utc_values[j],
                    "local_slot_seconds": int(slots[j]),
                    "valid_interval_count": 1,
                    "shape_defined_flag": int(feat["shape_defined_flag"]),
                    "curve_key": feat["curve_key"],
                    "p_min": feat["p_min"],
                    "p_max": feat["p_max"],
                    "bid_level": level,
                    "adjustment_bias": residual,
                    "adjustment_magnitude": (
                        abs(residual)
                        if np.isfinite(residual)
                        else np.nan
                    ),
                    "quantity_hhi": feat["quantity_hhi"],
                    "effective_segment_count": feat[
                        "effective_segment_count"
                    ],
                    "flat_curve_flag": feat["flat_curve_flag"],
                    "tail_uplift_ratio": feat["tail_uplift_ratio"],
                    "curve_bend_ratio": feat["curve_bend_ratio"],
                }

                for c in SHAPE_COLS:
                    row[c] = feat[c]

                rows.append(row)

                update_slot_accumulator(
                    slot_accumulator,
                    participant_id=units[j],
                    local_slot_seconds=int(slots[j]),
                    row=row,
                )

            cur = pd.DataFrame(rows)

            if not carry.empty:
                cur = pd.concat([carry, cur], ignore_index=True)

            if cur.empty:
                print(
                    f"  [chunk {chunk_no:03d}] valid=0, "
                    f"invalid={sum(chunk_invalid.values()):,}",
                    flush=True,
                )
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

            reason_text = (
                ", ".join(
                    f"{k}={v:,}"
                    for k, v in sorted(chunk_invalid.items())
                )
                if chunk_invalid
                else "none"
            )
            print(
                f"  [chunk {chunk_no:03d}] "
                f"kept={len(rows):,}, invalid[{reason_text}], "
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

    slot_df = slot_accumulator_to_frame(
        slot_accumulator
    )

    if not slot_df.empty:
        slot_df.to_csv(
            slot_file,
            index=False,
            encoding="utf-8-sig",
        )

    print()
    print("=" * 72)
    print("Historical strategy atoms build complete")
    print("=" * 72)
    print(f"Raw rows:        {total_raw:,}")
    print(f"Valid intervals: {total_valid:,}")
    print(
        f"Invalid rows:    "
        f"{sum(invalid_reasons.values()):,}"
    )

    if invalid_reasons:
        print("[invalid reasons]")
        for reason, count in invalid_reasons.most_common():
            print(f"  {reason}: {count:,}")

    print(f"Daily core:       {out_file}")
    print(f"Intraday slots:   {slot_file}")


if __name__ == "__main__":
    main()
