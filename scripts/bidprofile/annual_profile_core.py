# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL_FEATURES = [
    "annual_bid_level",
    "annual_adjustment_magnitude",
    "annual_strategy_persistence",
    "annual_quantity_hhi",
    "annual_effective_segment_count",
    "annual_flat_curve_rate",
    "annual_tail_uplift_ratio",
    "annual_curve_bend_ratio",
    "annual_shape_variability",
]

DAILY_SOURCE = {
    "annual_bid_level": "daily_bid_level",
    "annual_adjustment_magnitude": "daily_adjustment_magnitude",
    "annual_quantity_hhi": "daily_quantity_hhi",
    "annual_effective_segment_count": "daily_effective_segment_count",
    "annual_flat_curve_rate": "daily_flat_curve_rate",
    "annual_tail_uplift_ratio": "daily_tail_uplift_ratio",
    "annual_curve_bend_ratio": "daily_curve_bend_ratio",
}

DAILY_SHAPE_COLS = [f"daily_shape_v{i:02d}" for i in range(21)]
ANNUAL_SHAPE_COLS = [f"annual_shape_v{i:02d}" for i in range(21)]

FEATURE_CN = {
    "annual_bid_level": "报价水平",
    "annual_adjustment_magnitude": "调整幅度",
    "annual_strategy_persistence": "策略持续性",
    "annual_quantity_hhi": "容量集中度",
    "annual_effective_segment_count": "有效段数",
    "annual_flat_curve_rate": "平价偏好",
    "annual_tail_uplift_ratio": "尾部抬价",
    "annual_curve_bend_ratio": "曲线弯折",
    "annual_shape_variability": "形态波动",
}

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))

def robust_lag1_persistence(g, min_pairs=5, eps=1e-12):
    if "daily_adjustment_bias" not in g.columns:
        return np.nan, 0, 0

    x = _num(g["daily_adjustment_bias"])
    d = pd.to_datetime(g["local_date"]).dt.normalize()

    a_vals, b_vals = [], []
    for j in range(1, len(g)):
        if (d.iloc[j] - d.iloc[j - 1]).days != 1:
            continue
        a, b = x.iloc[j - 1], x.iloc[j]
        if np.isfinite(a) and np.isfinite(b):
            a_vals.append(float(a))
            b_vals.append(float(b))

    n = len(a_vals)
    if n < min_pairs:
        return np.nan, n, 0

    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    sa = float(np.std(a))
    sb = float(np.std(b))

    if sa <= eps and sb <= eps:
        same = np.all(np.abs(a - b) <= eps)
        return (1.0 if same else 0.0), n, 1
    if sa <= eps or sb <= eps:
        return 0.0, n, 1

    return float(np.corrcoef(a, b)[0, 1]), n, 0

def build_group_profile(g, min_persistence_pairs=5, min_shape_days=2):
    if g.empty:
        raise ValueError("Cannot build profile from empty group.")

    g = g.copy()
    g["local_date"] = pd.to_datetime(g["local_date"]).dt.normalize()
    g = g.sort_values("local_date").drop_duplicates("local_date", keep="last").reset_index(drop=True)

    dates = g["local_date"]
    row = {
        "source_start_date": dates.min(),
        "source_end_date": dates.max(),
        "active_days": int(dates.nunique()),
        "active_months": int(dates.dt.to_period("M").nunique()),
        "calendar_span_days": int((dates.max() - dates.min()).days + 1),
    }

    for out_col, src_col in DAILY_SOURCE.items():
        s = _num(g[src_col]) if src_col in g.columns else pd.Series(dtype=float)
        row[f"{out_col}_valid_days"] = int(s.notna().sum())
        if s.notna().sum() == 0:
            row[out_col] = np.nan
        elif out_col in {"annual_flat_curve_rate", "annual_curve_bend_ratio"}:
            # flat_curve_rate is a frequency; curve_bend_ratio is a signed
            # asymmetry tendency. Annual means retain both interpretations.
            row[out_col] = float(s.mean())
        else:
            row[out_col] = float(s.median())

    p, n_pairs, const_flag = robust_lag1_persistence(
        g, min_pairs=min_persistence_pairs
    )
    row["annual_strategy_persistence"] = p
    row["annual_strategy_persistence_valid_pairs"] = int(n_pairs)
    row["annual_strategy_persistence_constant_fallback_flag"] = int(const_flag)

    if all(c in g.columns for c in DAILY_SHAPE_COLS):
        shape = g[DAILY_SHAPE_COLS].apply(pd.to_numeric, errors="coerce")
        valid_shape = shape.loc[shape.notna().all(axis=1)]
    else:
        valid_shape = pd.DataFrame(columns=DAILY_SHAPE_COLS)

    row["annual_shape_defined_days"] = int(len(valid_shape))
    row["annual_shape_defined_rate"] = float(len(valid_shape) / len(g)) if len(g) else np.nan

    row["annual_shape_variability_flat_fallback_flag"] = 0

    if len(valid_shape) >= min_shape_days:
        proto = valid_shape.median(axis=0).to_numpy(float)
        dev = [rmse(v, proto) for v in valid_shape.to_numpy(float)]
        row["annual_shape_variability"] = float(np.median(dev))
        for c, v in zip(ANNUAL_SHAPE_COLS, proto):
            row[c] = float(v)
    else:
        flat_rate = row.get("annual_flat_curve_rate", np.nan)
        if np.isfinite(flat_rate) and flat_rate >= 1.0 - 1e-12:
            # A participant that is flat throughout the source year has no
            # normalized non-flat shape to estimate, but its shape is
            # structurally invariant. Record zero variability explicitly.
            row["annual_shape_variability"] = 0.0
            row["annual_shape_variability_flat_fallback_flag"] = 1
            for c in ANNUAL_SHAPE_COLS:
                row[c] = 0.0
        else:
            row["annual_shape_variability"] = np.nan
            for c in ANNUAL_SHAPE_COLS:
                row[c] = np.nan

    return row

def empirical_percentile(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True) * 100.0
