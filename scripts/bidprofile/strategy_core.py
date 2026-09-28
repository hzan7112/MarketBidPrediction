# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Historical subject-profile architecture
# ---------------------------------------------------------------------
# The final subject strategy profile is built ONLY from a complete historical
# reference period. Future prediction should not require target-year bid lags.
#
# Subject-level profile:
#   LT + historical short-term tendency + seasonal summary
#   + intraday summary + optional template-transition summary
#
# In addition, two historical context lookup tables are kept:
#   participant x month
#   participant x local slot
# They allow the future prediction pipeline to attach the subject's historical
# behavior for the target month / target time slot without using future bids.

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

SHORT_TENDENCY_FEATURES = [
    "short_bid_level_abs_z_median",
    "short_bid_level_abs_z_p90",
    "short_bid_level_high_state_rate",
    "short_adjustment_magnitude_abs_z_median",
    "short_structure_abs_z_median",
    "short_shape_shift_median",
    "short_break_rate",
    "short_ready_day_share",
]

SEASONAL_SUMMARY_FEATURES = [
    "season_bid_level_range",
    "season_adjustment_magnitude_range",
    "season_quantity_hhi_range",
    "season_effective_segment_count_range",
    "season_tail_uplift_ratio_range",
    "season_shape_shift_median",
]

INTRADAY_SUMMARY_FEATURES = [
    "intraday_bid_level_range",
    "intraday_bid_level_robust_scale",
    "intraday_adjustment_magnitude_range",
    "intraday_quantity_hhi_range",
    "intraday_switch_rate",
]

TRANSITION_FEATURES = [
    "transition_dominant_template_share",
    "transition_active_template_count",
    "transition_template_usage_entropy",
    "transition_daily_dominant_switch_rate",
    "transition_pair_entropy",
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

STRUCTURAL_STEMS = [
    "quantity_hhi",
    "effective_segment_count",
    "flat_curve_rate",
    "tail_uplift_ratio",
    "curve_bend_ratio",
]

SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
DAILY_SHAPE_COLS = [f"daily_shape_v{i:02d}" for i in range(21)]
LT_SHAPE_COLS = [f"lt_shape_v{i:02d}" for i in range(21)]

DEFAULT_SCALE_FLOORS = {
    "bid_level": 1e-3,
    "adjustment_bias": 1e-3,
    "adjustment_magnitude": 1e-3,
    "quantity_hhi": 1e-6,
    "effective_segment_count": 1e-6,
    "flat_curve_rate": 1e-6,
    "tail_uplift_ratio": 1e-6,
    "curve_bend_ratio": 1e-6,
}

DEFAULT_Z_CLIP = 10.0


def robust_scale(x):
    s = pd.Series(x, dtype=float).dropna()
    if s.empty:
        return np.nan
    med = float(s.median())
    return float(1.4826 * np.median(np.abs(s.to_numpy(float) - med)))


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def normalized_entropy(counts):
    x = np.asarray(counts, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) <= 1:
        return 0.0
    p = x / x.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p)))


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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
