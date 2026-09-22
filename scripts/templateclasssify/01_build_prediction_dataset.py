#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_build_prediction_dataset.py

Stage 3 / bidprediction
Build a leakage-controlled interval-level supervision dataset for PJM bid-curve
prediction from the finalized bidprofile + bidtemplate pipelines.

Core design
-----------
Target grain:
    participant_id x historical bid interval

Features are built only from information that would have been available before
PJM's day-ahead bid deadline. By default the simulated prediction cutoff is:
    11:00 America/New_York on the day before the target operating day.

Feature groups
~~~~~~~~~~~~~~
A. Rolling strategy profile (NO full-year LT leakage)
   - 9 rolling LT features, calculated from daily_strategy_core using dates < d
   - 9 ST features from short_term_strategy_state; the finalized ST builder
     already uses [d-7,d-1] vs [d-67,d-8]
   - 8 Break flags from the same short-term table

B. Participant own-state proxies
   - lagged same-slot template / mode / structure / scale labels
   - lagged same-slot PJM offer operating-parameter proxies, when present:
       no_load_cost, cold/inter/hot_start_cost, max_daily_starts, min_runtime,
       max/min/avg_ecomax, max/min/avg_ecomin
   - only PREVIOUS historical offers are exposed as features; current-row offer
     parameters are never written as predictors.

C. Market environment
   The script auto-detects official PJM Data Miner CSV folders when available:
       load_frcstd_hist
       frcstd_gen_outages
       day_gen_capacity
       hrl_load_metered
       da_hrl_lmps
       gen_by_fuel
       reserve_market_results
   Market adapters apply availability-aware rules. Missing feeds do not abort the
   build; their absence is recorded in the manifest.

Targets
~~~~~~~
Directly inherited from bidtemplate/05_build_curve_parameter_labels.py:
    y_template_id
    y_curve_mode
    y_effective_segment_count
    y_breakpoint_count
    y_breakpoint_x_json
    y_q_anchor_mw
    y_q_span_mw
    y_p_anchor
    y_p_span

Default inputs
--------------
    data/processed/final_clean/daily/<year>/daily_strategy_core_<year>.csv
    data/processed/final_clean/short_term/<year>/short_term_strategy_state_<year>.csv
    data/processed/bidtemplate/<year>/parameter_labels/curve_parameter_labels_*.csv
    data/raw/energy_market_offers/<year>/*.csv
    data/raw/<PJM-feed>/<year>/*.csv      (optional market feeds)

Default outputs
---------------
    data/processed/bidprediction/<year>/
        rolling_strategy_profile_<year>.csv
        market_feature_table_<year>.csv
        dataset_parts/prediction_dataset_<source>.csv
        prediction_dataset_manifest_<year>.csv
        prediction_feature_schema_<year>.csv

Important leakage rule
----------------------
The current target offer is NEVER used as an input feature. Historical offer
labels and raw operating-parameter fields are added to history only after the
current sample's feature row has been constructed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Finalized feature definitions from bidprofile
# -----------------------------------------------------------------------------

LT_FEATURES = [
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

ST_FEATURES = [
    "st_bid_level_z",
    "st_adjustment_bias_z",
    "st_adjustment_magnitude_z",
    "st_quantity_hhi_z",
    "st_effective_segment_count_z",
    "st_flat_curve_rate_z",
    "st_tail_uplift_ratio_z",
    "st_curve_bend_ratio_z",
    "st_shape_shift",
]

BREAK_FEATURES = [
    "break_bid_level",
    "break_adjustment_bias",
    "break_adjustment_magnitude",
    "break_quantity_hhi",
    "break_effective_segment_count",
    "break_flat_curve_rate",
    "break_tail_uplift_ratio",
    "break_curve_bend_ratio",
]

DAILY_TO_LT = {
    "daily_bid_level": "lt_bid_level",
    "daily_adjustment_magnitude": "lt_adjustment_magnitude",
    "daily_quantity_hhi": "lt_quantity_hhi",
    "daily_effective_segment_count": "lt_effective_segment_count",
    "daily_flat_curve_rate": "lt_flat_curve_rate",
    "daily_tail_uplift_ratio": "lt_tail_uplift_ratio",
    "daily_curve_bend_ratio": "lt_curve_bend_ratio",
}

DAILY_SHAPE_COLS = [f"daily_shape_v{i:02d}" for i in range(21)]

# Raw PJM unit-state proxy fields officially present in energy_market_offers.
UNIT_STATE_RAW_FIELDS = [
    "no_load_cost",
    "cold_start_cost",
    "inter_start_cost",
    "hot_start_cost",
    "max_daily_starts",
    "min_runtime",
    "max_ecomax",
    "min_ecomax",
    "avg_ecomax",
    "max_ecomin",
    "min_ecomin",
    "avg_ecomin",
]

TARGET_RENAME = {
    "template_id": "y_template_id",
    "curve_mode": "y_curve_mode",
    "effective_segment_count": "y_effective_segment_count",
    "breakpoint_count": "y_breakpoint_count",
    "breakpoint_x_json": "y_breakpoint_x_json",
    "q_anchor_mw": "y_q_anchor_mw",
    "q_span_mw": "y_q_span_mw",
    "p_anchor": "y_p_anchor",
    "p_span": "y_p_span",
}

EPS = 1e-12
NY_TZ = "America/New_York"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canon(name) -> str:
    s = str(name).strip().lower()
    for ch in (" ", "-", "/", ".", "(", ")"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def pick(columns: Iterable[str], candidates: Iterable[str], required=True):
    lookup = {canon(c): c for c in columns}
    for c in candidates:
        key = canon(c)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Cannot find any of {list(candidates)} in {list(columns)}")
    return None


def norm_pid(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def norm_local_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def to_local_naive(series: pd.Series) -> pd.Series:
    x = pd.to_datetime(series, errors="coerce")
    try:
        if x.dt.tz is not None:
            return x.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    except AttributeError:
        pass
    return x


def local_date_to_cutoff_utc(local_date: pd.Series, cutoff_hour: int) -> pd.Series:
    """Convert each operating local_date d to (d-1 at cutoff_hour EPT) in UTC."""
    d = pd.to_datetime(local_date, errors="coerce").dt.normalize()
    naive = d - pd.Timedelta(days=1) + pd.to_timedelta(cutoff_hour, unit="h")
    aware = naive.dt.tz_localize(NY_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return aware.dt.tz_convert("UTC")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def entropy_norm(values: Iterable[str]) -> float:
    vals = [str(v) for v in values if pd.notna(v) and str(v) != ""]
    if not vals:
        return np.nan
    counts = np.asarray(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p))
    k = len(counts)
    return float(h / np.log(k)) if k > 1 else 0.0


def dominant_value(values: Iterable[str]):
    vals = [str(v) for v in values if pd.notna(v) and str(v) != ""]
    if not vals:
        return pd.NA, np.nan
    c = Counter(vals)
    # deterministic tie-break by lexical order
    best = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best[0], best[1] / len(vals)


def rolling_corr_from_pairs(x: pd.Series, y: pd.Series, window: int, min_pairs=5) -> pd.Series:
    """Fast rolling Pearson correlation using pairwise valid observations."""
    valid = x.notna() & y.notna()
    xv = x.where(valid)
    yv = y.where(valid)

    n = valid.astype(float).rolling(window, min_periods=1).sum()
    sx = xv.fillna(0.0).rolling(window, min_periods=1).sum()
    sy = yv.fillna(0.0).rolling(window, min_periods=1).sum()
    sxx = (xv.fillna(0.0) ** 2).rolling(window, min_periods=1).sum()
    syy = (yv.fillna(0.0) ** 2).rolling(window, min_periods=1).sum()
    sxy = (xv.fillna(0.0) * yv.fillna(0.0)).rolling(window, min_periods=1).sum()

    cov_num = sxy - sx * sy / n.replace(0, np.nan)
    # Round-off can make a theoretically non-negative centered sum of
    # squares slightly negative (for example -1e-14). Clip before sqrt so
    # long rolling runs do not emit RuntimeWarning noise.
    vx = (sxx - sx * sx / n.replace(0, np.nan)).clip(lower=0.0)
    vy = (syy - sy * sy / n.replace(0, np.nan)).clip(lower=0.0)
    denom = np.sqrt(vx * vy)
    corr = cov_num / denom
    corr[(n < min_pairs) | (denom <= EPS)] = np.nan
    return corr


# -----------------------------------------------------------------------------
# Rolling leakage-free LT profile
# -----------------------------------------------------------------------------

def rolling_shape_variability(shape: pd.DataFrame, lookback_days: int) -> pd.Series:
    """Exact rolling analogue of lt_shape_variability using only prior days."""
    arr = shape.to_numpy(float)
    valid = np.all(np.isfinite(arr), axis=1)
    out = np.full(len(arr), np.nan, dtype=float)

    for j in range(len(arr)):
        lo = max(0, j - lookback_days)
        idx = np.flatnonzero(valid[lo:j]) + lo
        if len(idx) == 0:
            continue
        h = arr[idx]
        proto = np.median(h, axis=0)
        dev = np.sqrt(np.mean((h - proto) ** 2, axis=1))
        out[j] = float(np.median(dev))

    return pd.Series(out, index=shape.index)


def build_rolling_lt_table(
    daily: pd.DataFrame,
    lookback_days: int,
    min_history_days: int,
) -> pd.DataFrame:
    required = [
        "participant_id", "local_date", "daily_adjustment_bias",
        *DAILY_TO_LT.keys(), *DAILY_SHAPE_COLS,
    ]
    missing = [c for c in required if c not in daily.columns]
    if missing:
        raise KeyError(f"daily strategy core missing columns: {missing}")

    daily = daily[required].copy()
    daily["participant_id"] = norm_pid(daily["participant_id"])
    daily["local_date"] = norm_local_date(daily["local_date"])
    daily = daily.dropna(subset=["participant_id", "local_date"])

    results = []
    groups = list(daily.groupby("participant_id", sort=False))
    total = len(groups)

    for idx, (pid, g) in enumerate(groups, start=1):
        g = (
            g.sort_values("local_date")
            .drop_duplicates("local_date", keep="last")
            .copy()
        )
        if g.empty:
            continue

        original_dates = pd.DatetimeIndex(g["local_date"])
        full_dates = pd.date_range(original_dates.min(), original_dates.max(), freq="D")
        cal = g.set_index("local_date").reindex(full_dates)
        active = pd.Series(cal.index.isin(original_dates).astype(float), index=cal.index)

        out = pd.DataFrame(index=cal.index)
        out["participant_id"] = pid
        out["local_date"] = cal.index
        out["rolling_lt_history_days"] = (
            active.shift(1).rolling(lookback_days, min_periods=1).sum().fillna(0).astype(int)
        )

        for daily_col, lt_col in DAILY_TO_LT.items():
            s = safe_numeric(cal[daily_col])
            out[lt_col] = s.shift(1).rolling(lookback_days, min_periods=1).median()

        # Persistence: correlation of consecutive-day adjustment biases inside
        # the historical window, with all pairs ending before target day d.
        bias = safe_numeric(cal["daily_adjustment_bias"])
        prev = bias.shift(1)
        pair_ok = bias.notna() & prev.notna()
        pair_x = prev.where(pair_ok).shift(1)
        pair_y = bias.where(pair_ok).shift(1)
        out["lt_strategy_persistence"] = rolling_corr_from_pairs(
            pair_x, pair_y, lookback_days, min_pairs=5
        )

        shape = cal[DAILY_SHAPE_COLS].apply(pd.to_numeric, errors="coerce")
        out["lt_shape_variability"] = rolling_shape_variability(shape, lookback_days)

        out["rolling_lt_nonmissing_count"] = out[LT_FEATURES].notna().sum(axis=1)
        out["rolling_lt_ready_flag"] = (
            (out["rolling_lt_history_days"] >= min_history_days)
            & (out["rolling_lt_nonmissing_count"] >= 7)
        ).astype(int)

        out = out.loc[cal.index.isin(original_dates)].reset_index(drop=True)
        results.append(out)

        if idx == 1 or idx % 100 == 0 or idx == total:
            print(f"[rolling LT] participants {idx:,}/{total:,}", flush=True)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def build_profile_table(
    daily_file: Path,
    st_file: Path,
    lookback_days: int,
    min_history_days: int,
) -> pd.DataFrame:
    print(f"[profile] daily: {daily_file}")
    print(f"[profile] ST:    {st_file}")

    daily = pd.read_csv(daily_file, low_memory=False)
    rolling_lt = build_rolling_lt_table(daily, lookback_days, min_history_days)

    st_required = [
        "participant_id", "local_date", "st_ready_flag",
        *ST_FEATURES, *BREAK_FEATURES,
    ]
    st_header = pd.read_csv(st_file, nrows=0).columns.tolist()
    missing = [c for c in st_required if c not in st_header]
    if missing:
        raise KeyError(f"short-term state missing columns: {missing}")

    st = pd.read_csv(st_file, usecols=st_required, low_memory=False)
    st["participant_id"] = norm_pid(st["participant_id"])
    st["local_date"] = norm_local_date(st["local_date"])
    st = st.drop_duplicates(["participant_id", "local_date"], keep="last")

    out = rolling_lt.merge(
        st,
        on=["participant_id", "local_date"],
        how="left",
        validate="one_to_one",
    )
    out["st_ready_flag"] = safe_numeric(out["st_ready_flag"]).fillna(0).astype(int)
    out["profile_ready_flag"] = (
        out["rolling_lt_ready_flag"].eq(1) & out["st_ready_flag"].eq(1)
    ).astype(int)
    return out.sort_values(["participant_id", "local_date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# PJM raw market data adapters
# -----------------------------------------------------------------------------

def feed_files(raw_root: Path, feed: str, year: int) -> list[Path]:
    candidates = [
        raw_root / feed / str(year),
        raw_root / feed,
        raw_root / "pjm" / feed / str(year),
        raw_root / "pjm" / feed,
    ]
    for p in candidates:
        if p.exists():
            files = sorted(p.glob("*.csv"))
            if files:
                return files
    return []


def read_csv_files(files: list[Path], usecols=None) -> pd.DataFrame:
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, usecols=usecols, low_memory=False))
        except ValueError:
            # Column spelling/version can vary. Fall back to full read and adapt.
            parts.append(pd.read_csv(f, low_memory=False))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_hour_grid(year: int, cutoff_hour: int) -> pd.DataFrame:
    # UTC grid avoids duplicated local clock hours at DST fallback.
    start_local = pd.Timestamp(year=year, month=1, day=1, tz=NY_TZ)
    end_local = pd.Timestamp(year=year + 1, month=1, day=1, tz=NY_TZ)
    utc = pd.date_range(start_local.tz_convert("UTC"), end_local.tz_convert("UTC"), freq="h", inclusive="left")
    local = utc.tz_convert(NY_TZ)

    grid = pd.DataFrame({
        "target_timestamp_utc": utc,
        "target_timestamp_local": local.tz_localize(None),
    })
    grid["local_date"] = grid["target_timestamp_local"].dt.normalize()
    grid["local_slot_seconds"] = (
        grid["target_timestamp_local"].dt.hour * 3600
        + grid["target_timestamp_local"].dt.minute * 60
        + grid["target_timestamp_local"].dt.second
    )
    grid["prediction_cutoff_utc"] = local_date_to_cutoff_utc(grid["local_date"], cutoff_hour)
    return grid


def adapt_load_forecast(raw_root: Path, year: int, grid: pd.DataFrame, area: str) -> pd.DataFrame:
    files = feed_files(raw_root, "load_frcstd_hist", year)
    if not files:
        return pd.DataFrame()

    df = read_csv_files(files)
    cols = df.columns
    eval_col = pick(cols, ["evaluated_at_utc", "evaluated_at_datetime_utc"], required=False)
    target_col = pick(cols, ["forecast_hour_beginning_utc", "forecast_datetime_beginning_utc"], required=False)
    area_col = pick(cols, ["forecast_area"], required=False)
    value_col = pick(cols, ["forecast_load_mw"], required=False)
    if None in (eval_col, target_col, value_col):
        print("[market] load_frcstd_hist found but required columns are missing; skipped")
        return pd.DataFrame()

    d = df[[c for c in [eval_col, target_col, area_col, value_col] if c is not None]].copy()
    d["_eval"] = to_utc(d[eval_col])
    d["_target"] = to_utc(d[target_col])
    d["_value"] = safe_numeric(d[value_col])
    if area_col is not None:
        a = d[area_col].astype(str).str.upper().str.strip()
        wanted = area.upper().strip()
        exact = a.eq(wanted)
        if not exact.any() and wanted == "RTO":
            exact = a.isin(["RTO", "RTO_COMBINED", "PJM", "PJM_RTO"])
        d = d.loc[exact]

    d = d.dropna(subset=["_eval", "_target", "_value"])
    if d.empty:
        return pd.DataFrame()

    # Join each forecast candidate to target-hour cutoff and keep latest issuance <= cutoff.
    key = grid[["target_timestamp_utc", "prediction_cutoff_utc"]].copy()
    x = d.merge(key, left_on="_target", right_on="target_timestamp_utc", how="inner")
    x = x[x["_eval"] <= x["prediction_cutoff_utc"]]
    if x.empty:
        return pd.DataFrame()
    x = x.sort_values(["target_timestamp_utc", "_eval"]).drop_duplicates(
        "target_timestamp_utc", keep="last"
    )
    x["mkt_load_forecast_mw"] = x["_value"]
    x["mkt_load_forecast_age_h"] = (
        (x["prediction_cutoff_utc"] - x["_eval"]).dt.total_seconds() / 3600.0
    )
    return x[["target_timestamp_utc", "mkt_load_forecast_mw", "mkt_load_forecast_age_h"]]


def adapt_forecast_outages(raw_root: Path, year: int, grid: pd.DataFrame) -> pd.DataFrame:
    files = feed_files(raw_root, "frcstd_gen_outages", year)
    if not files:
        return pd.DataFrame()
    df = read_csv_files(files)
    exec_col = pick(df.columns, ["forecast_execution_date_ept"], required=False)
    date_col = pick(df.columns, ["forecast_date"], required=False)
    val_cols = {
        "mkt_forecast_outage_rto_mw": pick(df.columns, ["forecast_gen_outage_mw_rto"], required=False),
        "mkt_forecast_outage_west_mw": pick(df.columns, ["forecast_gen_outage_mw_west"], required=False),
        "mkt_forecast_outage_other_mw": pick(df.columns, ["forecast_gen_outage_mw_other"], required=False),
    }
    if exec_col is None or date_col is None or all(v is None for v in val_cols.values()):
        print("[market] frcstd_gen_outages found but required columns are missing; skipped")
        return pd.DataFrame()

    d = pd.DataFrame({
        "_exec_date": pd.to_datetime(df[exec_col], errors="coerce").dt.normalize(),
        "local_date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
    })
    for out_col, src in val_cols.items():
        d[out_col] = safe_numeric(df[src]) if src is not None else np.nan
    d = d.dropna(subset=["_exec_date", "local_date"])

    # Only use execution dates <= d-2 because feed exposes a date, not an exact
    # issuance time. This is deliberately conservative relative to d-1 11:00.
    d["_safe_cut_date"] = d["local_date"] - pd.Timedelta(days=2)
    d = d[d["_exec_date"] <= d["_safe_cut_date"]]
    d = d.sort_values(["local_date", "_exec_date"]).drop_duplicates("local_date", keep="last")
    return grid[["target_timestamp_utc", "local_date"]].merge(
        d.drop(columns=["_exec_date", "_safe_cut_date"]), on="local_date", how="left"
    ).drop(columns=["local_date"])


def adapt_previous_day_capacity(raw_root: Path, year: int, grid: pd.DataFrame) -> pd.DataFrame:
    files = feed_files(raw_root, "day_gen_capacity", year)
    if not files:
        return pd.DataFrame()
    df = read_csv_files(files)
    time_col = pick(df.columns, ["bid_datetime_beginning_utc", "bid_datetime_beginning_ept"], required=False)
    if time_col is None:
        return pd.DataFrame()

    ts = pd.to_datetime(df[time_col], errors="coerce", utc=("utc" in canon(time_col)))
    if getattr(ts.dt, "tz", None) is not None:
        local_date = ts.dt.tz_convert(NY_TZ).dt.tz_localize(None).dt.normalize()
    else:
        local_date = ts.dt.normalize()

    d = pd.DataFrame({"local_date": local_date})
    mapping = {
        "mkt_prev_day_eco_max_mw": pick(df.columns, ["eco_max"], required=False),
        "mkt_prev_day_emerg_max_mw": pick(df.columns, ["emerg_max"], required=False),
        "mkt_prev_day_total_committed_mw": pick(df.columns, ["total_committed"], required=False),
    }
    for out_col, src in mapping.items():
        d[out_col] = safe_numeric(df[src]) if src is not None else np.nan

    agg = d.groupby("local_date", as_index=False).median(numeric_only=True)
    # Feature for target day d is capacity record from d-1.
    agg["local_date"] = agg["local_date"] + pd.Timedelta(days=1)
    return grid[["target_timestamp_utc", "local_date"]].merge(
        agg, on="local_date", how="left"
    ).drop(columns=["local_date"])


def _build_cutoff_history_features(
    grid: pd.DataFrame,
    ts: pd.Series,
    values: pd.DataFrame,
    prefix: str,
    trailing_hours: int = 24,
) -> pd.DataFrame:
    """For each target-day cutoff, summarize observed market values <= cutoff."""
    hist = values.copy()
    hist["_ts"] = to_utc(ts)
    hist = hist.dropna(subset=["_ts"]).sort_values("_ts")
    if hist.empty:
        return pd.DataFrame()

    value_cols = [c for c in hist.columns if c != "_ts"]
    cut = grid[["local_date", "prediction_cutoff_utc"]].drop_duplicates("local_date").sort_values("prediction_cutoff_utc")

    rows = []
    hist_ts = hist["_ts"].to_numpy(dtype="datetime64[ns]")
    for r in cut.itertuples(index=False):
        cutoff = pd.Timestamp(r.prediction_cutoff_utc)
        hi = np.searchsorted(hist_ts, cutoff.to_datetime64(), side="right")
        if hi <= 0:
            row = {"local_date": r.local_date}
            for c in value_cols:
                row[f"{prefix}_{c}_last"] = np.nan
                row[f"{prefix}_{c}_mean24h"] = np.nan
                row[f"{prefix}_{c}_std24h"] = np.nan
            rows.append(row)
            continue

        lo_time = cutoff - pd.Timedelta(hours=trailing_hours)
        lo = np.searchsorted(hist_ts, lo_time.to_datetime64(), side="left")
        w = hist.iloc[lo:hi]
        last = hist.iloc[hi - 1]
        row = {"local_date": r.local_date}
        for c in value_cols:
            s = safe_numeric(w[c])
            row[f"{prefix}_{c}_last"] = pd.to_numeric(pd.Series([last[c]]), errors="coerce").iloc[0]
            row[f"{prefix}_{c}_mean24h"] = float(s.mean()) if s.notna().any() else np.nan
            row[f"{prefix}_{c}_std24h"] = float(s.std(ddof=0)) if s.notna().sum() >= 2 else np.nan
        rows.append(row)

    daily = pd.DataFrame(rows)
    return grid[["target_timestamp_utc", "local_date"]].merge(daily, on="local_date", how="left").drop(columns=["local_date"])


def adapt_metered_load(raw_root: Path, year: int, grid: pd.DataFrame) -> pd.DataFrame:
    files = feed_files(raw_root, "hrl_load_metered", year)
    if not files:
        return pd.DataFrame()
    df = read_csv_files(files)
    time_col = pick(df.columns, ["datetime_beginning_utc"], required=False)
    mw_col = pick(df.columns, ["mw"], required=False)
    if time_col is None or mw_col is None:
        return pd.DataFrame()

    d = pd.DataFrame({"ts": to_utc(df[time_col]), "mw": safe_numeric(df[mw_col])})
    # Aggregate all published areas per timestamp. If PJM exports overlapping
    # area hierarchies this is only a market-pressure proxy, not official RTO load.
    d = d.groupby("ts", as_index=False)["mw"].sum(min_count=1)
    return _build_cutoff_history_features(grid, d["ts"], d[["mw"]], "mkt_load")


def adapt_da_lmp(raw_root: Path, year: int, grid: pd.DataFrame) -> pd.DataFrame:
    files = feed_files(raw_root, "da_hrl_lmps", year)
    if not files:
        return pd.DataFrame()
    df = read_csv_files(files)
    time_col = pick(df.columns, ["datetime_beginning_utc"], required=False)
    price_col = pick(df.columns, ["system_energy_price_da", "total_lmp_da"], required=False)
    if time_col is None or price_col is None:
        return pd.DataFrame()

    d = pd.DataFrame({"ts": to_utc(df[time_col]), "price": safe_numeric(df[price_col])})
    # System energy price is common system-wide; median also tolerates duplicate pnodes.
    d = d.groupby("ts", as_index=False)["price"].median()
    return _build_cutoff_history_features(grid, d["ts"], d[["price"]], "mkt_da_price")


def adapt_gen_by_fuel(raw_root: Path, year: int, grid: pd.DataFrame) -> pd.DataFrame:
    files = feed_files(raw_root, "gen_by_fuel", year)
    if not files:
        return pd.DataFrame()
    df = read_csv_files(files)
    time_col = pick(df.columns, ["datetime_beginning_utc"], required=False)
    fuel_col = pick(df.columns, ["fuel_type"], required=False)
    mw_col = pick(df.columns, ["mw"], required=False)
    renew_col = pick(df.columns, ["is_renewable"], required=False)
    if None in (time_col, fuel_col, mw_col):
        return pd.DataFrame()

    d = pd.DataFrame({
        "ts": to_utc(df[time_col]),
        "fuel": df[fuel_col].astype(str).str.lower().str.strip(),
        "mw": safe_numeric(df[mw_col]),
    })
    if renew_col is not None:
        renew = df[renew_col].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
    else:
        renew = d["fuel"].str.contains("wind|solar|hydro|renew", regex=True, na=False)
    d["renew_mw"] = d["mw"].where(renew, 0.0)

    def classify(f):
        if "coal" in f:
            return "coal"
        if "gas" in f:
            return "gas"
        if "nuclear" in f:
            return "nuclear"
        if "wind" in f:
            return "wind"
        if "solar" in f:
            return "solar"
        return "other"

    d["class"] = d["fuel"].map(classify)
    piv = d.pivot_table(index="ts", columns="class", values="mw", aggfunc="sum", fill_value=0).reset_index()
    total = d.groupby("ts", as_index=False)["mw"].sum().rename(columns={"mw": "total_gen_mw"})
    renew = d.groupby("ts", as_index=False)["renew_mw"].sum()
    x = total.merge(renew, on="ts", how="left").merge(piv, on="ts", how="left")
    x["renew_share"] = x["renew_mw"] / x["total_gen_mw"].replace(0, np.nan)
    keep = [c for c in ["total_gen_mw", "renew_share", "coal", "gas", "nuclear", "wind", "solar"] if c in x.columns]
    return _build_cutoff_history_features(grid, x["ts"], x[keep], "mkt_genmix")


def adapt_reserve(raw_root: Path, year: int, grid: pd.DataFrame) -> pd.DataFrame:
    files = feed_files(raw_root, "reserve_market_results", year)
    if not files:
        return pd.DataFrame()
    df = read_csv_files(files)
    time_col = pick(df.columns, ["datetime_beginning_utc"], required=False)
    locale_col = pick(df.columns, ["locale", "area"], required=False)
    service_col = pick(df.columns, ["service", "reserve_type"], required=False)
    mcp_col = pick(df.columns, ["mcp", "market_clearing_price"], required=False)
    qty_col = pick(df.columns, ["total_mw", "reserve_quantity"], required=False)
    if time_col is None or service_col is None or (mcp_col is None and qty_col is None):
        return pd.DataFrame()

    d = df.copy()
    if locale_col is not None:
        loc = d[locale_col].astype(str).str.upper().str.strip()
        mask = loc.isin(["PJM_RTO", "RTO", "PJM"])
        if mask.any():
            d = d.loc[mask].copy()
    d["_ts"] = to_utc(d[time_col])
    d["_service"] = d[service_col].astype(str).str.upper().str.strip()

    parts = []
    for val_name, src in [("mcp", mcp_col), ("qty", qty_col)]:
        if src is None:
            continue
        temp = d.pivot_table(index="_ts", columns="_service", values=src, aggfunc="median")
        temp.columns = [f"{val_name}_{canon(c)}" for c in temp.columns]
        parts.append(temp)
    if not parts:
        return pd.DataFrame()
    x = pd.concat(parts, axis=1).reset_index().rename(columns={"_ts": "ts"})
    return _build_cutoff_history_features(grid, x["ts"], x.drop(columns=["ts"]), "mkt_reserve")


def build_market_feature_table(raw_root: Path, year: int, cutoff_hour: int, load_forecast_area: str):
    grid = build_hour_grid(year, cutoff_hour)
    adapters = [
        ("load_frcstd_hist", lambda: adapt_load_forecast(raw_root, year, grid, load_forecast_area)),
        ("frcstd_gen_outages", lambda: adapt_forecast_outages(raw_root, year, grid)),
        ("day_gen_capacity", lambda: adapt_previous_day_capacity(raw_root, year, grid)),
        ("hrl_load_metered", lambda: adapt_metered_load(raw_root, year, grid)),
        ("da_hrl_lmps", lambda: adapt_da_lmp(raw_root, year, grid)),
        ("gen_by_fuel", lambda: adapt_gen_by_fuel(raw_root, year, grid)),
        ("reserve_market_results", lambda: adapt_reserve(raw_root, year, grid)),
    ]

    out = grid.copy()
    feed_status = []
    for name, fn in adapters:
        files = feed_files(raw_root, name, year)
        if not files:
            print(f"[market] {name}: missing", flush=True)
            feed_status.append((name, 0, 0))
            continue
        print(f"[market] {name}: {len(files)} file(s)", flush=True)
        try:
            f = fn()
        except Exception as e:
            print(f"[market] {name}: adapter failed: {e}", flush=True)
            feed_status.append((name, len(files), -1))
            continue
        if f.empty:
            feed_status.append((name, len(files), 0))
            continue
        before = set(out.columns)
        out = out.merge(f, on="target_timestamp_utc", how="left", validate="one_to_one")
        added = len(set(out.columns) - before)
        feed_status.append((name, len(files), added))

    market_cols = [c for c in out.columns if c.startswith("mkt_")]
    if market_cols:
        out["market_nonmissing_count"] = out[market_cols].notna().sum(axis=1)
        out["market_ready_flag"] = (out["market_nonmissing_count"] > 0).astype(int)
    else:
        out["market_nonmissing_count"] = 0
        out["market_ready_flag"] = 0

    status = pd.DataFrame(feed_status, columns=["feed", "file_count", "feature_count"])
    return out, status


# -----------------------------------------------------------------------------
# Raw offer current-state extraction, used only to create LAGGED predictors
# -----------------------------------------------------------------------------

def raw_offer_state_columns(raw_file: Path):
    header = pd.read_csv(raw_file, nrows=0)
    lookup = {canon(c): c for c in header.columns}
    mapping = {}
    for f in UNIT_STATE_RAW_FIELDS:
        if f in lookup:
            mapping[f] = lookup[f]
    return mapping


def extract_current_raw_state(label_df: pd.DataFrame, raw_file: Path, chunksize: int) -> pd.DataFrame:
    """Return raw state fields aligned to label_df rows by source_row_index."""
    mapping = raw_offer_state_columns(raw_file)
    out = pd.DataFrame(index=np.arange(len(label_df)))
    for f in UNIT_STATE_RAW_FIELDS:
        out[f] = np.nan
    if not mapping:
        return out

    indices = pd.to_numeric(label_df["source_row_index"], errors="raise").astype(np.int64).to_numpy()
    order = np.argsort(indices)
    sorted_idx = indices[order]
    result = np.full((len(label_df), len(UNIT_STATE_RAW_FIELDS)), np.nan, dtype=float)

    usecols = list(dict.fromkeys(mapping.values()))
    offset = 0
    pos = 0
    for chunk in pd.read_csv(raw_file, usecols=usecols, chunksize=chunksize, low_memory=False):
        n = len(chunk)
        hi = np.searchsorted(sorted_idx, offset + n, side="left")
        if hi > pos:
            local_rows = sorted_idx[pos:hi] - offset
            selected = chunk.iloc[local_rows].reset_index(drop=True)
            orig_positions = order[pos:hi]
            for j, f in enumerate(UNIT_STATE_RAW_FIELDS):
                src = mapping.get(f)
                if src is not None:
                    result[orig_positions, j] = pd.to_numeric(selected[src], errors="coerce").to_numpy(float)
            pos = hi
        offset += n
        if pos >= len(sorted_idx):
            break

    if pos != len(sorted_idx):
        raise RuntimeError(
            f"Raw-state row alignment failed for {raw_file.name}: labels={len(sorted_idx)}, matched={pos}"
        )
    return pd.DataFrame(result, columns=UNIT_STATE_RAW_FIELDS)


# -----------------------------------------------------------------------------
# Historical participant state from previous SAME-SLOT offers
# -----------------------------------------------------------------------------

HISTORY_NUMERIC_LABELS = [
    "effective_segment_count",
    "breakpoint_count",
    "q_anchor_mw",
    "q_span_mw",
    "p_anchor",
    "p_span",
]


def _rolling_by_calendar_days(
    source: pd.DataFrame,
    value_cols: list[str],
    days: int,
    agg: str,
) -> pd.DataFrame:
    """Calendar-window rolling stats within participant + local-slot groups."""
    if source.empty or not value_cols:
        return pd.DataFrame(index=source.index)

    group_cols = ["participant_id", "local_slot_seconds"]
    tmp = source[group_cols + ["local_date", *value_cols]].copy().set_index("local_date")
    roller = tmp.groupby(group_cols, sort=False)[value_cols].rolling(
        f"{int(days)}D", closed="left", min_periods=1
    )
    if agg == "median":
        rolled = roller.median()
    elif agg == "sum":
        rolled = roller.sum()
    else:
        raise ValueError(f"Unsupported rolling aggregation: {agg}")

    vals = rolled.reset_index(drop=True)
    if len(vals) != len(source):
        raise RuntimeError("Rolling-history alignment length mismatch.")
    vals.index = source.index
    return vals


def _strict_previous_day_snapshot(source: pd.DataFrame) -> pd.DataFrame:
    """One previous-date snapshot for lag-1 fields, never same-day rows."""
    group_cols = ["participant_id", "local_slot_seconds"]
    lag_cols = [
        "template_id", "curve_mode",
        *HISTORY_NUMERIC_LABELS,
        *UNIT_STATE_RAW_FIELDS,
    ]

    daily_last = (
        source.sort_values(
            group_cols + ["local_date", "timestamp_utc", "_history_order"],
            kind="mergesort",
        )
        .groupby(group_cols + ["local_date"], sort=False, as_index=False)
        .tail(1)
        .sort_values(group_cols + ["local_date"], kind="mergesort")
        .copy()
    )

    g = daily_last.groupby(group_cols, sort=False)
    daily_last["_prev_local_date"] = g["local_date"].shift(1)
    for c in lag_cols:
        daily_last[f"_prev_{c}"] = g[c].shift(1)

    keep = group_cols + ["local_date", "_prev_local_date"] + [
        f"_prev_{c}" for c in lag_cols
    ]
    return daily_last[keep]


def add_historical_state_features(
    labels: pd.DataFrame,
    current_raw_state: pd.DataFrame,
    histories: dict,
    history_max_days: int,
) -> pd.DataFrame:
    """Vectorized previous-same-slot history features with strict date leakage control."""
    t0 = time.perf_counter()
    x = labels.copy().reset_index(drop=True)
    raw = current_raw_state.reset_index(drop=True)

    x["participant_id"] = norm_pid(x["participant_id"])
    x["local_date"] = norm_local_date(x["local_date"])
    x["local_slot_seconds"] = safe_numeric(x["local_slot_seconds"]).fillna(-1).astype(int)
    x["timestamp_utc"] = to_utc(x["timestamp_utc"])
    x["_orig_order_tmp"] = np.arange(len(x), dtype=np.int64)

    raw["_orig_order_tmp"] = np.arange(len(raw), dtype=np.int64)
    current = x.merge(raw, on="_orig_order_tmp", how="left", validate="one_to_one")
    current["_is_current"] = 1
    current["_history_order"] = np.arange(len(current), dtype=np.int64)

    hist_cols = [
        "participant_id", "local_slot_seconds", "local_date", "timestamp_utc",
        "template_id", "curve_mode",
        *HISTORY_NUMERIC_LABELS,
        *UNIT_STATE_RAW_FIELDS,
    ]

    tail = histories.get("__tail_df__")
    if not isinstance(tail, pd.DataFrame) or tail.empty:
        tail = pd.DataFrame(columns=hist_cols)
    else:
        tail = tail[hist_cols].copy()

    min_date = current["local_date"].min()
    max_date = current["local_date"].max()
    if pd.notna(min_date) and not tail.empty:
        tail = tail[
            tail["local_date"] >= min_date - pd.Timedelta(days=history_max_days)
        ].copy()

    tail["_is_current"] = 0
    tail["_orig_order_tmp"] = -1
    tail["_history_order"] = np.arange(-len(tail), 0, dtype=np.int64)

    current_hist = current[hist_cols + ["_is_current", "_orig_order_tmp", "_history_order"]]
    if tail.empty:
        combined = current_hist.copy().reset_index(drop=True)
    else:
        combined = pd.concat([tail, current_hist], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["participant_id", "local_slot_seconds", "local_date", "timestamp_utc", "_history_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    if pd.notna(min_date):
        roll_source = combined[
            combined["local_date"] >= min_date - pd.Timedelta(days=30)
        ].copy()
    else:
        roll_source = combined.copy()
    roll_source = roll_source.reset_index(drop=True)

    numeric_cols = [*HISTORY_NUMERIC_LABELS, *UNIT_STATE_RAW_FIELDS]
    for c in numeric_cols:
        roll_source[c] = safe_numeric(roll_source[c])

    print(
        f"  [history] current={len(current):,}, retained_tail={len(tail):,}, "
        f"rolling_rows={len(roll_source):,}", flush=True
    )

    r7 = _rolling_by_calendar_days(roll_source, numeric_cols, 7, "median")
    r30 = _rolling_by_calendar_days(roll_source, numeric_cols, 30, "median")

    feat = roll_source[["_is_current", "_orig_order_tmp"]].copy()
    for c in HISTORY_NUMERIC_LABELS:
        feat[f"hist7_median_{c}"] = r7[c].to_numpy()
        feat[f"hist30_median_{c}"] = r30[c].to_numpy()
    for c in UNIT_STATE_RAW_FIELDS:
        feat[f"unit7_median_{c}"] = r7[c].to_numpy()
        feat[f"unit30_median_{c}"] = r30[c].to_numpy()

    tids = roll_source["template_id"].astype("string")
    template_ids = sorted([str(v) for v in tids.dropna().unique()])
    if template_ids:
        dummy = pd.get_dummies(tids, dtype=np.int16).reindex(columns=template_ids, fill_value=0)
        cat_source = pd.concat(
            [
                roll_source[["participant_id", "local_slot_seconds", "local_date"]].reset_index(drop=True),
                dummy.reset_index(drop=True),
            ], axis=1,
        )
        counts = _rolling_by_calendar_days(cat_source, template_ids, 30, "sum")
        cm = counts.to_numpy(float)
        total = np.nansum(cm, axis=1)
        valid_total = total > 0
        safe_cm = np.where(np.isnan(cm), 0.0, cm)
        dom_idx = np.argmax(safe_cm, axis=1)
        dom = np.asarray(template_ids, dtype=object)[dom_idx]
        feat["hist30_dominant_template_id"] = pd.Series(dom, index=feat.index).where(valid_total, pd.NA)
        mx = np.max(safe_cm, axis=1)
        feat["hist30_dominant_template_share"] = np.divide(
            mx, total, out=np.full(len(total), np.nan), where=valid_total
        )
        pmat = np.divide(
            safe_cm, total[:, None], out=np.zeros_like(safe_cm), where=valid_total[:, None]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.sum(np.where(pmat > 0, pmat * np.log(pmat), 0.0), axis=1)
        k = np.sum(safe_cm > 0, axis=1)
        entropy = np.full(len(total), np.nan, dtype=float)
        entropy[valid_total & (k <= 1)] = 0.0
        mk = valid_total & (k > 1)
        entropy[mk] = h[mk] / np.log(k[mk])
        feat["hist30_template_entropy"] = entropy
        feat["hist30_observation_count"] = total.astype(np.int32)
    else:
        feat["hist30_dominant_template_id"] = pd.NA
        feat["hist30_dominant_template_share"] = np.nan
        feat["hist30_template_entropy"] = np.nan
        feat["hist30_observation_count"] = 0

    lag_map = _strict_previous_day_snapshot(combined)
    lag_current = current[[
        "_orig_order_tmp", "participant_id", "local_slot_seconds", "local_date"
    ]].merge(
        lag_map,
        on=["participant_id", "local_slot_seconds", "local_date"],
        how="left",
        validate="many_to_one",
    )

    lag_current["hist_prev_available_flag"] = lag_current["_prev_local_date"].notna().astype(int)
    lag_current["hist_days_since_prev_same_slot"] = (
        lag_current["local_date"] - lag_current["_prev_local_date"]
    ).dt.days.astype(float)
    lag_current["hist_lag1_template_id"] = lag_current["_prev_template_id"]
    lag_current["hist_lag1_curve_mode"] = lag_current["_prev_curve_mode"]
    for c in HISTORY_NUMERIC_LABELS:
        lag_current[f"hist_lag1_{c}"] = safe_numeric(lag_current[f"_prev_{c}"])
    for c in UNIT_STATE_RAW_FIELDS:
        lag_current[f"unit_lag1_{c}"] = safe_numeric(lag_current[f"_prev_{c}"])

    lag_keep = [
        "_orig_order_tmp", "hist_prev_available_flag", "hist_days_since_prev_same_slot",
        "hist_lag1_template_id", "hist_lag1_curve_mode",
        *[f"hist_lag1_{c}" for c in HISTORY_NUMERIC_LABELS],
        *[f"unit_lag1_{c}" for c in UNIT_STATE_RAW_FIELDS],
    ]
    lag_current = lag_current[lag_keep]

    roll_current = feat[feat["_is_current"].eq(1)].drop(columns=["_is_current"])
    features = lag_current.merge(
        roll_current, on="_orig_order_tmp", how="left", validate="one_to_one"
    )
    out = x.merge(features, on="_orig_order_tmp", how="left", validate="one_to_one")
    out = out.sort_values("_orig_order_tmp").drop(columns=["_orig_order_tmp"]).reset_index(drop=True)

    new_tail = combined[hist_cols].copy()
    if pd.notna(max_date):
        new_tail = new_tail[
            new_tail["local_date"] >= max_date - pd.Timedelta(days=history_max_days)
        ].copy()
    histories["__tail_df__"] = new_tail.reset_index(drop=True)

    print(
        f"  [history] vectorized complete in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


# -----------------------------------------------------------------------------
# Dataset part assembly
# -----------------------------------------------------------------------------

def find_raw_file(raw_offer_dir: Path, source_file: str) -> Path:
    p = raw_offer_dir / source_file
    if p.exists():
        return p
    matches = list(raw_offer_dir.glob(source_file))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Raw energy_market_offers source not found: {p}")


def label_file_source(label_df: pd.DataFrame) -> str:
    vals = label_df["source_file"].dropna().astype(str).unique()
    if len(vals) != 1:
        raise ValueError(f"Expected one source_file per label part, got {vals.tolist()}")
    return vals[0]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out["timestamp_local"], errors="coerce")
    out["cal_hour"] = t.dt.hour
    out["cal_minute"] = t.dt.minute
    out["cal_dayofweek"] = t.dt.dayofweek
    out["cal_month"] = t.dt.month
    out["cal_dayofyear"] = t.dt.dayofyear
    out["cal_is_weekend"] = t.dt.dayofweek.isin([5, 6]).astype(int)
    out["cal_hour_sin"] = np.sin(2 * np.pi * out["cal_hour"] / 24.0)
    out["cal_hour_cos"] = np.cos(2 * np.pi * out["cal_hour"] / 24.0)
    out["cal_doy_sin"] = np.sin(2 * np.pi * out["cal_dayofyear"] / 365.25)
    out["cal_doy_cos"] = np.cos(2 * np.pi * out["cal_dayofyear"] / 365.25)
    return out


def assemble_part(
    label_file: Path,
    raw_offer_dir: Path,
    profile: pd.DataFrame,
    market: pd.DataFrame,
    histories: dict,
    chunksize: int,
    history_max_days: int,
    cutoff_hour: int,
) -> pd.DataFrame:
    t_part = time.perf_counter()
    print("  [1/5] loading parameter labels ...", flush=True)
    labels = pd.read_csv(label_file, low_memory=False)
    print(f"        label rows={len(labels):,}", flush=True)
    required = [
        "sample_id", "participant_id", "timestamp_utc", "timestamp_local",
        "local_date", "local_slot_seconds", "source_file", "source_row_index",
        *TARGET_RENAME.keys(),
    ]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise KeyError(f"{label_file.name} missing columns: {missing}")

    source_file = label_file_source(labels)
    raw_file = find_raw_file(raw_offer_dir, source_file)
    print(f"  [2/5] aligning laggable raw unit-state fields from {raw_file.name} ...", flush=True)
    raw_state = extract_current_raw_state(labels, raw_file, chunksize)
    available_state = [c for c in UNIT_STATE_RAW_FIELDS if raw_state[c].notna().any()]
    print(f"        available unit-state fields={len(available_state)}", flush=True)
    print("  [3/5] building previous-same-slot historical features ...", flush=True)
    labels = add_historical_state_features(
        labels, raw_state, histories, history_max_days
    )

    labels["participant_id"] = norm_pid(labels["participant_id"])
    labels["local_date"] = norm_local_date(labels["local_date"])
    labels["timestamp_utc"] = to_utc(labels["timestamp_utc"])
    labels["timestamp_local"] = to_local_naive(labels["timestamp_local"])
    labels["prediction_cutoff_utc"] = local_date_to_cutoff_utc(labels["local_date"], cutoff_hour)

    # Merge leakage-free rolling profile.
    print("  [4/5] joining rolling profile and market table ...", flush=True)
    part = labels.merge(
        profile,
        on=["participant_id", "local_date"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_profile"),
    )

    # Market table is hourly; use UTC target timestamp.
    market_cols = [c for c in market.columns if c not in {
        "target_timestamp_local", "local_date", "local_slot_seconds", "prediction_cutoff_utc"
    }]
    m = market[market_cols].copy()
    part = part.merge(
        m,
        left_on="timestamp_utc",
        right_on="target_timestamp_utc",
        how="left",
        validate="many_to_one",
    )
    if "target_timestamp_utc" in part.columns:
        part = part.drop(columns=["target_timestamp_utc"])

    part = add_calendar_features(part)

    # Readiness / leakage-audit columns.
    if "profile_ready_flag" not in part:
        part["profile_ready_flag"] = 0
    if "market_ready_flag" not in part:
        part["market_ready_flag"] = 0
    part["unit_state_ready_flag"] = part["hist_prev_available_flag"].fillna(0).astype(int)
    part["prediction_ready_flag"] = (
        part["profile_ready_flag"].fillna(0).astype(int).eq(1)
        & part["market_ready_flag"].fillna(0).astype(int).eq(1)
    ).astype(int)

    # Rename targets and selected QA columns. Current raw state is intentionally absent.
    part = part.rename(columns=TARGET_RENAME)

    keep_meta = [
        "sample_id", "participant_id", "timestamp_utc", "timestamp_local",
        "local_date", "local_slot_seconds", "prediction_cutoff_utc",
        "source_file", "source_row_index", "source_market", "market_product",
        "template_family", "template_cluster",
        "template_shape_mae", "template_shape_rmse",
        "template_price_mae", "template_price_rmse",
    ]
    keep_meta = [c for c in keep_meta if c in part.columns]

    feature_cols = [
        *LT_FEATURES,
        *ST_FEATURES,
        *BREAK_FEATURES,
        "rolling_lt_history_days", "rolling_lt_nonmissing_count",
        "rolling_lt_ready_flag", "st_ready_flag", "profile_ready_flag",
        *[c for c in part.columns if c.startswith("hist_")],
        *[c for c in part.columns if c.startswith("unit_")],
        *[c for c in part.columns if c.startswith("mkt_")],
        "market_nonmissing_count", "market_ready_flag", "unit_state_ready_flag",
        "cal_hour", "cal_minute", "cal_dayofweek", "cal_month", "cal_dayofyear",
        "cal_is_weekend", "cal_hour_sin", "cal_hour_cos", "cal_doy_sin", "cal_doy_cos",
    ]
    # Stable unique order.
    seen = set()
    feature_cols = [c for c in feature_cols if c in part.columns and not (c in seen or seen.add(c))]

    target_cols = [v for v in TARGET_RENAME.values() if v in part.columns]
    tail_cols = ["prediction_ready_flag"]

    result = part[keep_meta + feature_cols + target_cols + tail_cols]
    print(
        f"  [5/5] assembled {len(result):,} rows in {time.perf_counter() - t_part:.1f}s",
        flush=True,
    )
    return result


# -----------------------------------------------------------------------------
# Schema / manifest
# -----------------------------------------------------------------------------

def build_schema(columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c in TARGET_RENAME.values():
            group, role = "target", "target"
        elif c in ["sample_id", "participant_id", "timestamp_utc", "timestamp_local", "local_date",
                   "local_slot_seconds", "prediction_cutoff_utc", "source_file", "source_row_index",
                   "source_market", "market_product", "template_family", "template_cluster",
                   "template_shape_mae", "template_shape_rmse", "template_price_mae", "template_price_rmse"]:
            group, role = "metadata", "metadata"
        elif c in LT_FEATURES or c.startswith("rolling_lt_"):
            group, role = "profile_LT", "feature"
        elif c in ST_FEATURES:
            group, role = "profile_ST", "feature"
        elif c in BREAK_FEATURES:
            group, role = "profile_Break", "feature"
        elif c.startswith("hist_"):
            group, role = "participant_history", "feature"
        elif c.startswith("unit_"):
            group, role = "unit_state_proxy", "feature"
        elif c.startswith("mkt_") or c in ["market_nonmissing_count", "market_ready_flag"]:
            group, role = "market_environment", "feature"
        elif c.startswith("cal_"):
            group, role = "calendar", "feature"
        elif c.endswith("_ready_flag"):
            group, role = "readiness", "feature"
        else:
            group, role = "other", "metadata"
        rows.append({"column": c, "role": role, "feature_group": group})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--daily-root", default="data/processed/final_clean/daily")
    p.add_argument("--short-term-root", default="data/processed/final_clean/short_term")
    p.add_argument("--bidtemplate-root", default="data/processed/bidtemplate")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--out-root", default="data/processed/bidprediction")
    p.add_argument("--lt-lookback-days", type=int, default=180)
    p.add_argument("--lt-min-history-days", type=int, default=30)
    p.add_argument("--history-max-days", type=int, default=90)
    p.add_argument("--cutoff-hour-ept", type=int, default=11)
    p.add_argument("--load-forecast-area", default="RTO")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--max-label-files", type=int, default=None)
    p.add_argument(
        "--reuse-intermediate",
        action="store_true",
        help=(
            "Reuse existing rolling_strategy_profile and market_feature_table "
            "when present. Useful when only dataset assembly code changed."
        ),
    )
    p.add_argument(
        "--require-market",
        action="store_true",
        help="Fail if no PJM market feature feed is available.",
    )
    args = p.parse_args()

    year = args.year
    daily_file = Path(args.daily_root) / str(year) / f"daily_strategy_core_{year}.csv"
    st_file = Path(args.short_term_root) / str(year) / f"short_term_strategy_state_{year}.csv"
    label_dir = Path(args.bidtemplate_root) / str(year) / "parameter_labels"
    raw_offer_dir = Path(args.raw_root) / "energy_market_offers" / str(year)

    for f in [daily_file, st_file]:
        if not f.exists():
            raise FileNotFoundError(f)
    if not label_dir.exists():
        raise FileNotFoundError(label_dir)
    if not raw_offer_dir.exists():
        raise FileNotFoundError(raw_offer_dir)

    label_files = sorted(label_dir.glob("curve_parameter_labels_*.csv"))
    if args.max_label_files is not None:
        label_files = label_files[:args.max_label_files]
    if not label_files:
        raise FileNotFoundError(f"No curve_parameter_labels_*.csv under {label_dir}")

    out_dir = ensure_dir(Path(args.out_root) / str(year))
    part_dir = ensure_dir(out_dir / "dataset_parts")

    print("=" * 80)
    print("Build leakage-controlled bid-prediction dataset")
    print("=" * 80)
    print(f"Year:                  {year}")
    print(f"Prediction cutoff:     D-1 {args.cutoff_hour_ept:02d}:00 EPT")
    print(f"Rolling LT lookback:   {args.lt_lookback_days} days")
    print(f"Rolling LT min hist:   {args.lt_min_history_days} active days")
    print(f"Label files:           {len(label_files)}")

    # 1) Rolling profile.
    profile_file = out_dir / f"rolling_strategy_profile_{year}.csv"
    if args.reuse_intermediate and profile_file.exists():
        print(f"[profile] reusing: {profile_file}", flush=True)
        profile = pd.read_csv(profile_file, low_memory=False)
        profile["participant_id"] = norm_pid(profile["participant_id"])
        profile["local_date"] = norm_local_date(profile["local_date"])
    else:
        profile = build_profile_table(
            daily_file,
            st_file,
            lookback_days=args.lt_lookback_days,
            min_history_days=args.lt_min_history_days,
        )
        profile.to_csv(profile_file, index=False, encoding="utf-8-sig", float_format="%.10g")
    print(f"[profile] rows={len(profile):,}, ready={profile['profile_ready_flag'].sum():,}")
    print(f"[profile] saved: {profile_file}")

    # 2) Market environment.
    market_file = out_dir / f"market_feature_table_{year}.csv"
    feed_status_file = out_dir / f"market_feed_status_{year}.csv"
    if args.reuse_intermediate and market_file.exists():
        print(f"[market] reusing: {market_file}", flush=True)
        market = pd.read_csv(market_file, low_memory=False)
        market["target_timestamp_utc"] = to_utc(market["target_timestamp_utc"])
        if "prediction_cutoff_utc" in market.columns:
            market["prediction_cutoff_utc"] = to_utc(market["prediction_cutoff_utc"])
        if "target_timestamp_local" in market.columns:
            market["target_timestamp_local"] = to_local_naive(market["target_timestamp_local"])
        if "local_date" in market.columns:
            market["local_date"] = norm_local_date(market["local_date"])
        if feed_status_file.exists():
            feed_status = pd.read_csv(feed_status_file, low_memory=False)
        else:
            feed_status = pd.DataFrame(columns=["feed", "file_count", "feature_count"])
    else:
        market, feed_status = build_market_feature_table(
            Path(args.raw_root), year, args.cutoff_hour_ept, args.load_forecast_area
        )
        market.to_csv(market_file, index=False, encoding="utf-8-sig", float_format="%.10g")
        feed_status.to_csv(feed_status_file, index=False, encoding="utf-8-sig")

    market_feature_cols = [c for c in market.columns if c.startswith("mkt_")]
    if args.require_market and not market_feature_cols:
        raise RuntimeError(
            "--require-market was set, but none of the supported PJM market feeds were found."
        )
    print(f"[market] features={len(market_feature_cols)}, ready_hours={market['market_ready_flag'].sum():,}")
    print(f"[market] saved: {market_file}")

    # 3) Interval-level dataset parts. Histories persist across monthly/source files.
    histories = {}
    manifest_rows = []
    schema = None
    total_rows = total_profile_ready = total_market_ready = total_prediction_ready = 0

    for i, label_file in enumerate(label_files, start=1):
        print(f"\n[dataset {i}/{len(label_files)}] {label_file.name}", flush=True)
        part = assemble_part(
            label_file=label_file,
            raw_offer_dir=raw_offer_dir,
            profile=profile,
            market=market,
            histories=histories,
            chunksize=args.chunksize,
            history_max_days=args.history_max_days,
            cutoff_hour=args.cutoff_hour_ept,
        )

        source_stem = label_file.stem.replace("curve_parameter_labels_", "")
        out_file = part_dir / f"prediction_dataset_{source_stem}.csv"
        part.to_csv(out_file, index=False, encoding="utf-8-sig", float_format="%.10g")

        if schema is None:
            schema = build_schema(part.columns.tolist())

        profile_ready = int(safe_numeric(part.get("profile_ready_flag", 0)).fillna(0).sum())
        market_ready = int(safe_numeric(part.get("market_ready_flag", 0)).fillna(0).sum())
        pred_ready = int(safe_numeric(part["prediction_ready_flag"]).fillna(0).sum())
        unit_ready = int(safe_numeric(part.get("unit_state_ready_flag", 0)).fillna(0).sum())

        manifest_rows.append({
            "label_file": label_file.name,
            "output_file": str(out_file),
            "rows": len(part),
            "profile_ready_rows": profile_ready,
            "market_ready_rows": market_ready,
            "unit_state_ready_rows": unit_ready,
            "prediction_ready_rows": pred_ready,
            "prediction_ready_share": pred_ready / len(part) if len(part) else np.nan,
        })

        total_rows += len(part)
        total_profile_ready += profile_ready
        total_market_ready += market_ready
        total_prediction_ready += pred_ready

        print(
            f"  rows={len(part):,}, profile_ready={profile_ready:,}, "
            f"market_ready={market_ready:,}, unit_state_ready={unit_ready:,}, "
            f"prediction_ready={pred_ready:,}",
            flush=True,
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_file = out_dir / f"prediction_dataset_manifest_{year}.csv"
    manifest.to_csv(manifest_file, index=False, encoding="utf-8-sig")

    schema_file = out_dir / f"prediction_feature_schema_{year}.csv"
    if schema is not None:
        schema.to_csv(schema_file, index=False, encoding="utf-8-sig")

    # Compact configuration / audit record.
    config = {
        "year": year,
        "prediction_cutoff_rule": f"D-1 {args.cutoff_hour_ept:02d}:00 America/New_York",
        "rolling_lt_lookback_days": args.lt_lookback_days,
        "rolling_lt_min_history_days": args.lt_min_history_days,
        "participant_history_max_days": args.history_max_days,
        "load_forecast_area": args.load_forecast_area,
        "profile_definition": "rolling 9 LT + finalized 9 ST + finalized 8 Break",
        "target_definition": list(TARGET_RENAME.values()),
        "leakage_controls": [
            "rolling LT uses local dates strictly before target local_date",
            "ST/Break builder uses prior-day windows",
            "participant historical features are extracted before current target is appended",
            "current raw PJM unit-state fields are never exposed as predictors",
            "load forecast uses latest evaluation <= simulated bid cutoff",
            "outage forecast uses conservative execution_date <= target_date-2",
            "actual market histories are summarized only up to simulated bid cutoff",
            "current target-day day_gen_capacity is not used; previous day only",
        ],
        "rows": total_rows,
        "profile_ready_rows": total_profile_ready,
        "market_ready_rows": total_market_ready,
        "prediction_ready_rows": total_prediction_ready,
        "prediction_ready_share": total_prediction_ready / total_rows if total_rows else None,
    }
    config_file = out_dir / f"prediction_dataset_build_config_{year}.json"
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print("Prediction dataset build complete")
    print("=" * 80)
    print(f"Rows:                  {total_rows:,}")
    print(f"Profile-ready:         {total_profile_ready:,}")
    print(f"Market-ready:          {total_market_ready:,}")
    print(f"Prediction-ready:      {total_prediction_ready:,}")
    if total_rows:
        print(f"Prediction-ready share:{total_prediction_ready / total_rows:.2%}")
    print(f"Rolling profile:       {profile_file}")
    print(f"Market table:          {market_file}")
    print(f"Market feed status:    {feed_status_file}")
    print(f"Dataset parts:         {part_dir}")
    print(f"Manifest:              {manifest_file}")
    print(f"Feature schema:        {schema_file}")
    print(f"Build config:          {config_file}")


if __name__ == "__main__":
    main()
