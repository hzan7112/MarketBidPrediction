# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

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

PA_FEATURES = [
    "lt_bid_level",
    "lt_adjustment_magnitude",
    "lt_strategy_persistence",
]
QS_FEATURES = [
    "lt_quantity_hhi",
    "lt_effective_segment_count",
    "lt_flat_curve_rate",
]
SHAPE_FEATURES = [
    "lt_tail_uplift_ratio",
    "lt_curve_bend_ratio",
    "lt_shape_variability",
]

DAILY_SCALARS = {
    "bid_level": "daily_bid_level",
    "adjustment_bias": "daily_adjustment_bias",
    "adjustment_magnitude": "daily_adjustment_magnitude",
    "quantity_hhi": "daily_quantity_hhi",
    "effective_segment_count": "daily_effective_segment_count",
    "flat_curve_rate": "daily_flat_curve_rate",
    "tail_uplift_ratio": "daily_tail_uplift_ratio",
    "curve_bend_ratio": "daily_curve_bend_ratio",
}

SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
DAILY_SHAPE_COLS = [f"daily_shape_v{i:02d}" for i in range(21)]
LT_SHAPE_COLS = [f"lt_shape_v{i:02d}" for i in range(21)]

def robust_scale(x):
    s = pd.Series(x, dtype=float).dropna()
    if s.empty:
        return np.nan
    med = s.median()
    return 1.4826 * np.median(np.abs(s.to_numpy() - med))

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def first_existing(columns, candidates, required=True):
    lower = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Cannot find any of columns: {candidates}")
    return None

def parse_bool(v):
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "sloped", "slope"}

def empirical_percentile(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True) * 100.0
