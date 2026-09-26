#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04a3_build_stable_theta_continuous_dataset.py

Build the final stable-theta residual + continuous-gate dataset.

Stable theta
------------
Price:
    theta_p_base       = P(u=0)
    theta_slope        = [P(u=0.70)-P(u=0)] / 0.70
    theta_tail_uplift  = P(u=1)-P(u=0.70)-0.30*theta_slope

Quantity:
    theta_q_base_mw
    theta_q_span_mw
    theta_q1..theta_q5

Here u is the template coordinate. q1..q5 map the physical quantity coordinate x
to five equal template-coordinate segments. Therefore price theta and quantity
theta are separated but remain consistent with final template reconstruction.

Historical state:
    same participant + same local slot, strictly before current date:
        lag1 stable theta
        last-7-active-observation mean/median/std
        last-30-active-observation mean/median/std
        short trend = lag1 - hist7_mean
        long trend  = hist7_mean - hist30_mean
        days since lag1 and history counts

These are all low-dimensional theta histories. Historical 21-point curves are
never exposed to the predictor.

No historical 21-point curve is exposed to the model.

Additional residual-learning context
------------------------------------
For Z_base / Z_tr / market / unit-state numerical features, this script also
builds previous-same-slot changes:
    delta_ctx_<feature> = current_<feature> - previous_same_slot_<feature>


Run:
python scripts/bidprediction/04a3_build_stable_theta_continuous_dataset.py --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


GRID = np.linspace(0.0, 1.0, 21)

STABLE_THETA = [
    "stable_theta_p_base",
    "stable_theta_slope",
    "stable_theta_tail_uplift",
    "stable_theta_q_base_mw",
    "stable_theta_q_span_mw",
    "stable_theta_q1",
    "stable_theta_q2",
    "stable_theta_q3",
    "stable_theta_q4",
    "stable_theta_q5",
]

HIST_STABLE_THETA = [
    "hist_lag1_" + c
    for c in STABLE_THETA
]


HIST_STAT_PREFIXES = [
    "hist7_mean_",
    "hist7_median_",
    "hist7_std_",
    "hist30_mean_",
    "hist30_median_",
    "hist30_std_",
    "hist_trend_short_",
    "hist_trend_long_",
]

HIST_STAT_THETA = [
    prefix + c
    for prefix in HIST_STAT_PREFIXES
    for c in STABLE_THETA
]

HIST_META_FEATURES = [
    "hist_theta_days_since_lag1",
    "hist_theta_count7",
    "hist_theta_count30",
]

HIST_ALL_FEATURES = (
    HIST_STABLE_THETA
    + HIST_STAT_THETA
    + HIST_META_FEATURES
)


CONTEXT_GROUPS = {
    "profile_LT",
    "profile_ST",
    "profile_Break",
    "transition_strategy_profile",
    "market_environment",
    "unit_state_proxy",
}

CONTEXT_EXCLUDE = {
    "rolling_lt_ready_flag",
    "st_ready_flag",
    "profile_ready_flag",
    "market_ready_flag",
    "unit_state_ready_flag",
    "prediction_ready_flag",
    "parameter_ready_flag",
    "theta_ready_flag",
    "market_nonmissing_count",
    "rolling_lt_nonmissing_count",
    "hist_prev_available_flag",
    "tr_ready_flag",
}


def context_columns_from_schema(schema: pd.DataFrame) -> list[str]:
    s = schema[
        schema["role"].astype(str).str.lower().eq("feature")
    ].copy()

    cols = []
    for row in s.itertuples(index=False):
        c = str(row.column)
        g = str(row.feature_group)
        if g in CONTEXT_GROUPS and c not in CONTEXT_EXCLUDE:
            cols.append(c)

    return uniq(cols)

SOURCE_REQUIRED = [
    "sample_id",
    "participant_id",
    "local_date",
    "local_slot_seconds",
    "theta_ready_flag",
    "y_template_id",
    "p_anchor",
    "p_span",
    "theta_q_base_mw",
    "theta_q_span_mw",
    "theta_q1",
    "theta_q2",
    "theta_q3",
    "theta_q4",
    "theta_q5",
]


def norm(s):
    return s.astype("string").str.strip()


def num(s):
    return pd.to_numeric(s, errors="coerce")


def uniq(xs):
    return list(dict.fromkeys(xs))


def month_key(path: Path):
    m = re.search(r"(\d{4})[_-](\d{2})", path.stem)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def shape_cols(header):
    exact = [f"shape_v{i:02d}" for i in range(21)]
    if all(c in header for c in exact):
        return exact
    return None


def discover_curve_samples(year_dir: Path):
    root = year_dir / "curve_samples"
    if not root.exists():
        raise FileNotFoundError(root)

    out = {}
    for p in sorted(root.rglob("*.csv")):
        try:
            h = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue
        if "sample_id" not in h:
            continue
        if shape_cols(h) is None:
            continue
        mk = month_key(p)
        if mk:
            out.setdefault(mk, []).append(p)

    if not out:
        raise FileNotFoundError(
            f"No curve_samples with sample_id + shape_v00..20 under {root}"
        )
    return out


def load_month_shapes(files):
    blocks = []
    for p in files:
        h = pd.read_csv(p, nrows=0).columns.tolist()
        sc = shape_cols(h)
        use = ["sample_id", "template_family", *sc]
        d = pd.read_csv(p, usecols=use, low_memory=False)
        d["sample_id"] = norm(d["sample_id"])
        d = d.rename(
            columns={c: f"shape_v{i:02d}" for i, c in enumerate(sc)}
        )
        blocks.append(d)

    out = pd.concat(blocks, ignore_index=True)
    if out["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in Stage2 curve samples.")
    return out


def compute_stable_theta(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()

    sc = [f"shape_v{i:02d}" for i in range(21)]
    shape = (
        out[sc]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    flat = (
        out["template_family"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("flat")
        .to_numpy()
    )
    if flat.any():
        shape[flat, :] = 0.0

    pa = num(out["p_anchor"]).to_numpy(float)
    ps = num(out["p_span"]).to_numpy(float)
    P = pa[:, None] + ps[:, None] * shape

    Q = (
        out[[f"theta_q{i}" for i in range(1, 6)]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    Q = np.maximum(Q, 1e-8)
    Q /= Q.sum(axis=1, keepdims=True)

    # u=0.70 lies halfway inside template segment [0.60,0.80].
    # Physical x at that point depends on q1..q5.
    x70 = Q[:, 0] + Q[:, 1] + Q[:, 2] + 0.5 * Q[:, 3]
    pos = np.clip(x70 * 20.0, 0.0, 20.0)
    lo = np.floor(pos).astype(np.int16)
    hi = np.minimum(lo + 1, 20)
    f = pos - lo
    row = np.arange(len(out))
    p70 = P[row, lo] + f * (P[row, hi] - P[row, lo])

    p0 = P[:, 0]
    p1 = P[:, -1]

    slope = (p70 - p0) / 0.70
    tail = p1 - p70 - 0.30 * slope

    out["stable_theta_p_base"] = p0
    out["stable_theta_slope"] = slope
    out["stable_theta_tail_uplift"] = tail
    out["stable_theta_q_base_mw"] = num(out["theta_q_base_mw"])
    out["stable_theta_q_span_mw"] = num(out["theta_q_span_mw"])

    for i in range(5):
        out[f"stable_theta_q{i+1}"] = Q[:, i]

    return out


def read_split(path: Path):
    s = pd.read_csv(path).set_index("split")
    tr_end = pd.Timestamp(s.loc["train", "last_date"]).normalize()
    val_start = pd.Timestamp(s.loc["val", "first_date"]).normalize()
    val_end = pd.Timestamp(s.loc["val", "last_date"]).normalize()
    test_start = pd.Timestamp(s.loc["test", "first_date"]).normalize()
    return tr_end, val_start, val_end, test_start


def split_masks(date_series, tr_end, val_start, val_end, test_start):
    d = pd.to_datetime(date_series, errors="coerce").dt.normalize()
    return {
        "train": d <= tr_end,
        "val": (d >= val_start) & (d <= val_end),
        "test": d >= test_start,
    }


def priority_for_sample_id(sample_id, seed):
    base = pd.util.hash_pandas_object(
        sample_id.astype("string"),
        index=False,
    ).to_numpy(np.uint64)
    return base ^ np.uint64(seed * 0x9E3779B1)


class StableMonthCache:
    def __init__(
        self,
        source_by_month,
        sample_files,
        context_cols,
        max_months=5,
        history_month_lookback=4,
    ):
        self.source_by_month = source_by_month
        self.sample_files = sample_files
        self.context_cols = list(context_cols)
        self.max_months = max_months
        self.history_month_lookback = int(history_month_lookback)

        self.cache = {}
        self.order = []

        # History summary is much smaller than the raw monthly table and is
        # reused by all chunks of the same month.
        self.history_cache = {}
        self.history_order = []

    def get(self, mk):
        if mk in self.cache:
            if mk in self.order:
                self.order.remove(mk)
            self.order.append(mk)
            return self.cache[mk]

        src = self.source_by_month.get(mk)
        sf = self.sample_files.get(mk)

        if src is None or sf is None:
            return pd.DataFrame()

        header = pd.read_csv(src, nrows=0).columns.tolist()
        required = uniq(SOURCE_REQUIRED + self.context_cols)
        missing = [c for c in required if c not in header]
        if missing:
            raise KeyError(f"{src.name} missing columns: {missing}")

        d = pd.read_csv(
            src,
            usecols=required,
            low_memory=False,
        )

        for c in self.context_cols:
            d[c] = num(d[c]).astype("float32")

        d["sample_id"] = norm(d["sample_id"])
        d["participant_id"] = norm(d["participant_id"])
        d["local_date"] = pd.to_datetime(
            d["local_date"], errors="coerce"
        ).dt.normalize()
        d["local_slot_seconds"] = (
            num(d["local_slot_seconds"]).fillna(-1).astype(int)
        )

        shapes = load_month_shapes(sf)
        d = d.merge(
            shapes,
            on="sample_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )

        joined = d["template_family"].notna()
        if not joined.all():
            raise ValueError(
                f"{int((~joined).sum()):,} rows in {src.name} "
                "failed Stage2 shape join."
            )

        d = compute_stable_theta(d)

        keep = [
            "sample_id",
            "participant_id",
            "local_date",
            "local_slot_seconds",
            *STABLE_THETA,
            *self.context_cols,
            *[f"shape_v{i:02d}" for i in range(21)],
        ]
        d = d[keep].copy()

        self.cache[mk] = d
        self.order.append(mk)

        while len(self.order) > self.max_months:
            old = self.order.pop(0)
            self.cache.pop(old, None)

        return d

    @staticmethod
    def _daily_last(d: pd.DataFrame) -> pd.DataFrame:
        """
        One deterministic historical observation per
        participant + date + local slot.
        """
        if d.empty:
            return d

        x = d.copy()

        x["__source_row_order"] = pd.to_numeric(
            x["sample_id"]
            .astype("string")
            .str.rsplit(":", n=1)
            .str[-1],
            errors="coerce",
        )

        # Stable fallback if sample_id does not end with a numeric row index.
        x["__sample_lex"] = x["sample_id"].astype("string")

        x = (
            x.sort_values(
                [
                    "participant_id",
                    "local_slot_seconds",
                    "local_date",
                    "__source_row_order",
                    "__sample_lex",
                ],
                kind="mergesort",
            )
            .drop_duplicates(
                [
                    "participant_id",
                    "local_date",
                    "local_slot_seconds",
                ],
                keep="last",
            )
            .drop(
                columns=[
                    "__source_row_order",
                    "__sample_lex",
                ]
            )
            .reset_index(drop=True)
        )

        return x

    def _months_for_history(self, mk):
        current = pd.Period(str(mk), freq="M")
        available = sorted(
            [
                m
                for m in self.source_by_month
                if m in self.sample_files
            ]
        )

        selected = []

        for m in available:
            p = pd.Period(str(m), freq="M")
            delta = current.ordinal - p.ordinal

            if 0 <= delta <= self.history_month_lookback:
                selected.append(m)

        return selected

    @staticmethod
    def _rolling_stat(lag_theta, group_arrays, window, stat):
        gb = lag_theta.groupby(
            group_arrays,
            sort=False,
        )

        roll = gb.rolling(
            window=window,
            min_periods=2,
        )

        if stat == "mean":
            z = roll.mean()
        elif stat == "median":
            z = roll.median()
        elif stat == "std":
            z = roll.std(ddof=0)
        else:
            raise ValueError(stat)

        return z.reset_index(
            level=[0, 1],
            drop=True,
        ).sort_index()

    def history_features(self, mk):
        """
        Build features for rows in month `mk` using strictly earlier
        observations from the same participant + local slot.

        Windows are the last 7 / 30 ACTIVE same-slot observations. This avoids
        inventing missing calendar-day bids and is robust to inactive days.
        """
        if mk in self.history_cache:
            if mk in self.history_order:
                self.history_order.remove(mk)
            self.history_order.append(mk)
            return self.history_cache[mk]

        months = self._months_for_history(mk)

        blocks = []

        for m in months:
            raw = self.get(m)
            if raw.empty:
                continue

            keep = [
                "sample_id",
                "participant_id",
                "local_date",
                "local_slot_seconds",
                *STABLE_THETA,
                *self.context_cols,
            ]

            blocks.append(
                self._daily_last(
                    raw[keep]
                )
            )

        if not blocks:
            return pd.DataFrame()

        daily = pd.concat(
            blocks,
            ignore_index=True,
        )

        daily = (
            daily.sort_values(
                [
                    "participant_id",
                    "local_slot_seconds",
                    "local_date",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        keys = [
            daily["participant_id"],
            daily["local_slot_seconds"],
        ]

        grouped = daily.groupby(
            [
                "participant_id",
                "local_slot_seconds",
            ],
            sort=False,
        )

        lag_theta = grouped[
            STABLE_THETA
        ].shift(1)

        lag_date = grouped[
            "local_date"
        ].shift(1)

        # -----------------------------------------------------------------
        # Build ALL historical feature columns in dictionaries first, then
        # concatenate once. This avoids pandas DataFrame fragmentation caused
        # by repeated ``history[col] = ...`` insertions.
        # -----------------------------------------------------------------
        base_history = daily[
            [
                "participant_id",
                "local_date",
                "local_slot_seconds",
            ]
        ].copy()

        feature_data = {}

        for c in STABLE_THETA:
            feature_data[
                "hist_lag1_" + c
            ] = lag_theta[c].to_numpy(float)

        # Previous same-slot context is used only to construct delta_ctx.
        lag_ctx_cols = []

        if self.context_cols:
            lag_ctx = grouped[
                self.context_cols
            ].shift(1)

            for c in self.context_cols:
                hc = "hist_lag1_ctx_" + c
                feature_data[hc] = (
                    num(lag_ctx[c])
                    .to_numpy(dtype=np.float32)
                )
                lag_ctx_cols.append(hc)

        roll7_mean = self._rolling_stat(
            lag_theta, keys, 7, "mean"
        )
        roll7_median = self._rolling_stat(
            lag_theta, keys, 7, "median"
        )
        roll7_std = self._rolling_stat(
            lag_theta, keys, 7, "std"
        )

        roll30_mean = self._rolling_stat(
            lag_theta, keys, 30, "mean"
        )
        roll30_median = self._rolling_stat(
            lag_theta, keys, 30, "median"
        )
        roll30_std = self._rolling_stat(
            lag_theta, keys, 30, "std"
        )

        for c in STABLE_THETA:
            lag_arr = feature_data[
                "hist_lag1_" + c
            ]

            h7_mean = roll7_mean[
                c
            ].to_numpy(float)

            h7_median = roll7_median[
                c
            ].to_numpy(float)

            h7_std = roll7_std[
                c
            ].to_numpy(float)

            h30_mean = roll30_mean[
                c
            ].to_numpy(float)

            h30_median = roll30_median[
                c
            ].to_numpy(float)

            h30_std = roll30_std[
                c
            ].to_numpy(float)

            feature_data[
                "hist7_mean_" + c
            ] = h7_mean

            feature_data[
                "hist7_median_" + c
            ] = h7_median

            feature_data[
                "hist7_std_" + c
            ] = h7_std

            feature_data[
                "hist30_mean_" + c
            ] = h30_mean

            feature_data[
                "hist30_median_" + c
            ] = h30_median

            feature_data[
                "hist30_std_" + c
            ] = h30_std

            feature_data[
                "hist_trend_short_" + c
            ] = (
                lag_arr
                - h7_mean
            )

            feature_data[
                "hist_trend_long_" + c
            ] = (
                h7_mean
                - h30_mean
            )

        feature_data[
            "hist_theta_days_since_lag1"
        ] = (
            (
                daily["local_date"]
                - pd.to_datetime(
                    lag_date,
                    errors="coerce",
                )
            )
            .dt.days
            .to_numpy(
                dtype=np.float32,
                na_value=np.nan,
            )
        )

        count_source = lag_theta[
            STABLE_THETA[0]
        ]

        count7 = (
            count_source.groupby(
                keys,
                sort=False,
            )
            .rolling(
                window=7,
                min_periods=1,
            )
            .count()
            .reset_index(
                level=[0, 1],
                drop=True,
            )
            .sort_index()
        )

        count30 = (
            count_source.groupby(
                keys,
                sort=False,
            )
            .rolling(
                window=30,
                min_periods=1,
            )
            .count()
            .reset_index(
                level=[0, 1],
                drop=True,
            )
            .sort_index()
        )

        feature_data[
            "hist_theta_count7"
        ] = count7.to_numpy(
            dtype=np.float32
        )

        feature_data[
            "hist_theta_count30"
        ] = count30.to_numpy(
            dtype=np.float32
        )

        history_features = pd.DataFrame(
            feature_data,
            index=base_history.index,
        )

        history = pd.concat(
            [
                base_history,
                history_features,
            ],
            axis=1,
            copy=False,
        ).copy()

        current_month = (
            history["local_date"]
            .dt.strftime("%Y-%m")
            .eq(str(mk))
        )

        history = (
            history.loc[current_month]
            .reset_index(drop=True)
        )

        self.history_cache[mk] = history
        self.history_order.append(mk)

        while len(self.history_order) > 2:
            old = self.history_order.pop(0)
            self.history_cache.pop(old, None)

        return history



def attach_current_and_history(
    chunk: pd.DataFrame,
    current_lookup: pd.DataFrame,
    cache: StableMonthCache,
    context_cols: list[str],
    month: str,
):
    out = chunk.copy().reset_index(drop=True)

    out["sample_id"] = norm(out["sample_id"])
    out["participant_id"] = norm(out["participant_id"])

    out["local_date"] = pd.to_datetime(
        out["local_date"],
        errors="coerce",
    ).dt.normalize()

    out["local_slot_seconds"] = (
        num(out["local_slot_seconds"])
        .fillna(-1)
        .astype(int)
    )

    # Current target theta + curve target for evaluation only.
    shape_cols_now = [
        f"shape_v{i:02d}"
        for i in range(21)
    ]

    current = current_lookup[
        [
            "sample_id",
            *STABLE_THETA,
            *shape_cols_now,
        ]
    ].copy()

    out = out.merge(
        current,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if out[STABLE_THETA].isna().any(axis=1).any():
        bad = int(
            out[STABLE_THETA]
            .isna()
            .any(axis=1)
            .sum()
        )
        raise ValueError(
            f"{bad:,} rows missing current stable theta."
        )

    hist = cache.history_features(month)

    if hist.empty:
        for c in HIST_ALL_FEATURES:
            out[c] = np.nan

        for c in context_cols:
            out["delta_ctx_" + c] = np.nan

        return out

    hist_ctx_cols = [
        "hist_lag1_ctx_" + c
        for c in context_cols
    ]

    out = out.merge(
        hist,
        on=[
            "participant_id",
            "local_date",
            "local_slot_seconds",
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    for c in context_cols:
        hc = "hist_lag1_ctx_" + c

        out["delta_ctx_" + c] = (
            num(out[c])
            - num(out[hc])
        ).astype("float32")

    out = out.drop(
        columns=[
            c
            for c in hist_ctx_cols
            if c in out.columns
        ]
    )

    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--root", default="data/processed/bidprediction")
    ap.add_argument("--bidtemplate-root", default="data/processed/bidtemplate")
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--train-sample-rows", type=int, default=300_000)
    ap.add_argument(
        "--history-month-lookback",
        type=int,
        default=4,
        help="Months loaded to construct last-30 active same-slot theta history.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    source_dir = base / "template_adjustment_dataset"
    source_parts = sorted(
        (source_dir / "dataset_parts").glob(
            "template_adjustment_dataset_*.csv"
        )
    )
    if not source_parts:
        raise FileNotFoundError(
            "Run 04a2_build_template_adjustment_dataset.py first."
        )

    schema_file = (
        source_dir
        / f"template_adjustment_schema_{args.year}.csv"
    )
    split_file = base / "validation" / "temporal_split_summary.csv"

    schema = pd.read_csv(schema_file)
    context_cols = context_columns_from_schema(schema)
    delta_context_cols = [
        "delta_ctx_" + c
        for c in context_cols
    ]

    feature_cols = (
        schema.loc[
            schema["role"].astype(str).str.lower().eq("feature"),
            "column",
        ]
        .astype(str)
        .tolist()
    )

    base_cols = uniq(
        [
            "sample_id",
            "participant_id",
            "local_date",
            "local_slot_seconds",
            "theta_ready_flag",
            "y_template_id",
            "p_anchor",
            "p_span",
            "q_anchor_mw",
            "q_span_mw",
            *[f"theta_q{i}" for i in range(1, 6)],
            *feature_cols,
        ]
    )

    source_by_month = {
        month_key(p): p
        for p in source_parts
        if month_key(p)
    }

    bidtemplate_year = Path(args.bidtemplate_root) / str(args.year)
    sample_files = discover_curve_samples(bidtemplate_year)
    cache = StableMonthCache(
        source_by_month,
        sample_files,
        context_cols=context_cols,
        max_months=max(args.history_month_lookback + 1, 5),
        history_month_lookback=args.history_month_lookback,
    )

    tr_end, val_start, val_end, test_start = read_split(split_file)

    out = base / "frozen_stable_theta_continuous_dataset"
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} exists; use --overwrite to rebuild."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Augmented schema.
    schema_aug = schema.copy()
    for c in STABLE_THETA:
        schema_aug = pd.concat(
            [
                schema_aug,
                pd.DataFrame(
                    [{
                        "column": c,
                        "role": "target",
                        "feature_group": "stable_theta_target",
                        "source": "04a3_stable_theta",
                        "leakage_use": "target_only",
                    }]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    for c in HIST_STABLE_THETA:
        schema_aug = pd.concat(
            [
                schema_aug,
                pd.DataFrame(
                    [{
                        "column": c,
                        "role": "feature",
                        "feature_group": "stable_theta_history",
                        "source": "previous_same_slot",
                        "leakage_use": "feature_pre_cutoff",
                    }]
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    for c in HIST_STAT_THETA:
        schema_aug = pd.concat(
            [
                schema_aug,
                pd.DataFrame(
                    [{
                        "column": c,
                        "role": "feature",
                        "feature_group": "stable_theta_history_stats",
                        "source": "strictly_past_same_slot_theta",
                        "leakage_use": "feature_pre_cutoff",
                    }]
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    for c in HIST_META_FEATURES:
        schema_aug = pd.concat(
            [
                schema_aug,
                pd.DataFrame(
                    [{
                        "column": c,
                        "role": "feature",
                        "feature_group": "stable_theta_history_stats",
                        "source": "strictly_past_same_slot_theta",
                        "leakage_use": "feature_pre_cutoff",
                    }]
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    for c in delta_context_cols:
        schema_aug = pd.concat(
            [
                schema_aug,
                pd.DataFrame(
                    [{
                        "column": c,
                        "role": "feature",
                        "feature_group": "context_delta",
                        "source": "current_minus_previous_same_slot",
                        "leakage_use": "feature_pre_cutoff",
                    }]
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    schema_aug.to_csv(
        out / "feature_schema.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shutil.copy2(split_file, out / "temporal_split_summary.csv")

    written = {"train": [], "val": [], "test": []}
    counts = {k: 0 for k in written}
    history_ready = {k: 0 for k in ["train", "val", "test"]}
    reservoir = None

    print("=" * 80)
    print("Build stable-theta continuous-gate dataset")
    print("=" * 80)
    print(f"Year:        {args.year}")
    print(f"Train <=     {tr_end.date()}")
    print(f"Validation:  {val_start.date()} .. {val_end.date()}")
    print(f"Test >=      {test_start.date()}")
    print(f"Context delta features: {len(delta_context_cols)}")
    print(f"Theta history-stat features: {len(HIST_STAT_THETA) + len(HIST_META_FEATURES)}")
    print()

    for part_no, src in enumerate(source_parts, 1):
        mk = month_key(src)
        print(
            f"[source {part_no}/{len(source_parts)}] {src.name}",
            flush=True,
        )

        current_lookup = cache.get(mk)
        if current_lookup.empty:
            raise ValueError(f"Stable lookup is empty for {mk}")

        for chunk_no, chunk in enumerate(
            pd.read_csv(
                src,
                usecols=base_cols,
                chunksize=args.chunksize,
                low_memory=False,
            ),
            1,
        ):
            ready = num(chunk["theta_ready_flag"]).fillna(0).eq(1)
            chunk = chunk.loc[ready].copy()
            if chunk.empty:
                continue

            chunk = attach_current_and_history(
                chunk,
                current_lookup,
                cache,
                context_cols,
                month=mk,
            )

            masks = split_masks(
                chunk["local_date"],
                tr_end,
                val_start,
                val_end,
                test_start,
            )

            for split, mask in masks.items():
                d = chunk.loc[mask].copy()
                if d.empty:
                    continue

                counts[split] += len(d)
                hready = d[HIST_STABLE_THETA].notna().all(axis=1)
                history_ready[split] += int(hready.sum())

                part_dir = out / f"{split}_parts"
                part_dir.mkdir(parents=True, exist_ok=True)
                path = (
                    part_dir
                    / f"{split}_{src.stem}_{chunk_no:04d}.pkl"
                )
                d.to_pickle(path, protocol=5)
                written[split].append(str(path.relative_to(out)))

                if split == "train":
                    x = d.loc[hready].copy()
                    if not x.empty:
                        x["__priority"] = priority_for_sample_id(
                            x["sample_id"],
                            args.seed,
                        )
                        if reservoir is None:
                            reservoir = x
                        else:
                            reservoir = pd.concat(
                                [reservoir, x],
                                ignore_index=True,
                            )
                        if len(reservoir) > args.train_sample_rows:
                            reservoir = (
                                reservoir.nsmallest(
                                    args.train_sample_rows,
                                    "__priority",
                                    keep="first",
                                )
                                .reset_index(drop=True)
                            )

            del chunk
            gc.collect()

    if reservoir is None or reservoir.empty:
        raise ValueError("No history-ready training rows.")

    reservoir = (
        reservoir.nsmallest(
            min(args.train_sample_rows, len(reservoir)),
            "__priority",
            keep="first",
        )
        .drop(columns="__priority")
        .reset_index(drop=True)
    )

    train_sample_file = out / f"train_sample_{len(reservoir)}.pkl"
    reservoir.to_pickle(train_sample_file, protocol=5)

    manifest = {
        "version": "stable-theta-continuous-gate-v1",
        "year": args.year,
        "train_end": str(tr_end.date()),
        "validation_start": str(val_start.date()),
        "validation_end": str(val_end.date()),
        "test_start": str(test_start.date()),
        "stable_theta": STABLE_THETA,
        "history_theta": HIST_STABLE_THETA,
        "history_theta_stats": HIST_STAT_THETA,
        "history_meta_features": HIST_META_FEATURES,
        "history_window_definition": "last 7/30 active same-slot observations, strictly before current date",
        "context_columns": context_cols,
        "delta_context_columns": delta_context_cols,
        "train_rows": counts["train"],
        "val_rows": counts["val"],
        "test_rows": counts["test"],
        "history_ready_train_rows": history_ready["train"],
        "history_ready_val_rows": history_ready["val"],
        "history_ready_test_rows": history_ready["test"],
        "train_sample_rows": len(reservoir),
        "train_sample_file": train_sample_file.name,
        "parts": written,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            f"Stable-theta continuous-gate dataset - {args.year}",
            "=" * 80,
            "",
            f"Train rows: {counts['train']:,}",
            f"Validation rows: {counts['val']:,}",
            f"Test rows: {counts['test']:,}",
            "",
            (
                f"History-ready train: {history_ready['train']:,} "
                f"({history_ready['train']/max(counts['train'],1):.2%})"
            ),
            (
                f"History-ready val: {history_ready['val']:,} "
                f"({history_ready['val']/max(counts['val'],1):.2%})"
            ),
            (
                f"History-ready test: {history_ready['test']:,} "
                f"({history_ready['test']/max(counts['test'],1):.2%})"
            ),
            "",
            f"Fixed history-ready train sample: {len(reservoir):,}",
        ]
    )
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
