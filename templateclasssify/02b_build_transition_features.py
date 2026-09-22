#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02b_build_transition_features.py

Build a leakage-free transition_strategy_profile for Stage 3.

Theory
------
Base strategy profile:
    Z_base = 9 LT + 9 ST + 8 Break

This script adds:
    Z_transition = transition_strategy_profile

Later the prediction-side strategy state can be tested as:
    Z = Z_base
    Z = [Z_base, Z_transition]

M, U and H remain separate:
    M = market environment
    U = unit physical / operational state
    H = direct bid-history state such as hist_lag1_template_id

Leakage rule
------------
For target day D, every transition feature uses ONLY template history from
participant-days strictly before D. The target day's template labels are never
used to construct the target day's features.

Input
-----
data/processed/bidprediction/<year>/dataset_parts/prediction_dataset_*.csv
Required columns:
    participant_id, local_date, y_template_id

Output
------
data/processed/bidprediction/<year>/
    transition_daily_template_state_<year>.csv
    transition_strategy_profile_<year>.csv
    transition_strategy_feature_schema_<year>.csv
    transition_strategy_profile_manifest_<year>.csv
    transition_strategy_profile_summary_<year>.txt

The output grain of transition_strategy_profile is participant_id x local_date,
so it can later be joined to every interval row by these two keys.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TEMPLATE_ORDER = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
TEMPLATE_TO_INT = {t: i for i, t in enumerate(TEMPLATE_ORDER)}
N_TEMPLATE = len(TEMPLATE_ORDER)
LOG_N_TEMPLATE = math.log(N_TEMPLATE)

TRANSITION_FEATURES = [
    "tr_hist_active_days",
    "tr_switch_rate_7d",
    "tr_switch_rate_30d",
    "tr_switch_rate_90d",
    "tr_switch_count_30d",
    "tr_switch_count_90d",
    "tr_ever_switched",
    "tr_dwell_active_days",
    "tr_days_since_last_switch",
    "tr_dominant_share_7d",
    "tr_dominant_share_30d",
    "tr_template_entropy_7d",
    "tr_template_entropy_30d",
    "tr_unique_template_count_7d",
    "tr_unique_template_count_30d",
    "tr_lag1_daily_dominant_share",
    "tr_lag1_daily_template_entropy",
    "tr_lag1_daily_unique_template_count",
    "tr_daily_purity_mean_7d",
    "tr_daily_purity_mean_30d",
    "tr_daily_entropy_mean_7d",
    "tr_daily_entropy_mean_30d",
    "tr_transition_dest_concentration_90d",
    "tr_origin_switch_rate_90d",
    "tr_origin_dest_concentration_90d",
    "tr_reversion_rate_90d",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalized_entropy(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return np.nan
    p = counts[counts > 0].astype(float) / total
    return float(-np.sum(p * np.log(p)) / LOG_N_TEMPLATE)


def build_daily_template_state(files: list[Path], chunksize: int) -> pd.DataFrame:
    required = ["participant_id", "local_date", "y_template_id"]
    pieces = []

    for i, file in enumerate(files, 1):
        header = pd.read_csv(file, nrows=0).columns.tolist()
        missing = [c for c in required if c not in header]
        if missing:
            raise KeyError(f"{file.name} missing columns: {missing}")

        print(f"[daily-state {i}/{len(files)}] {file.name}", flush=True)

        for chunk in pd.read_csv(
            file,
            usecols=required,
            chunksize=chunksize,
            low_memory=False,
        ):
            chunk["participant_id"] = chunk["participant_id"].astype("string").str.strip()
            chunk["local_date"] = pd.to_datetime(chunk["local_date"], errors="coerce").dt.normalize()
            chunk["y_template_id"] = chunk["y_template_id"].astype("string").str.strip()

            valid = (
                chunk["participant_id"].notna()
                & chunk["local_date"].notna()
                & chunk["y_template_id"].isin(TEMPLATE_ORDER)
            )
            chunk = chunk.loc[valid, required]
            if chunk.empty:
                continue

            agg = (
                chunk.groupby(
                    ["participant_id", "local_date", "y_template_id"],
                    observed=True,
                    sort=False,
                )
                .size()
                .rename("interval_count")
                .reset_index()
            )
            pieces.append(agg)

    if not pieces:
        raise ValueError("No valid template rows found.")

    counts = pd.concat(pieces, ignore_index=True)
    counts = (
        counts.groupby(
            ["participant_id", "local_date", "y_template_id"],
            observed=True,
            sort=False,
            as_index=False,
        )["interval_count"]
        .sum()
    )
    counts["template_code"] = counts["y_template_id"].map(TEMPLATE_TO_INT).astype(np.int16)

    totals = (
        counts.groupby(["participant_id", "local_date"], observed=True, as_index=False)
        .agg(
            daily_interval_count=("interval_count", "sum"),
            daily_unique_template_count=("y_template_id", "nunique"),
        )
    )

    ranked = counts.sort_values(
        ["participant_id", "local_date", "interval_count", "template_code"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    dominant = (
        ranked.drop_duplicates(["participant_id", "local_date"], keep="first")
        [["participant_id", "local_date", "y_template_id", "template_code", "interval_count"]]
        .rename(
            columns={
                "y_template_id": "daily_dominant_template_id",
                "template_code": "daily_dominant_template_code",
                "interval_count": "daily_dominant_template_count",
            }
        )
    )

    counts["p"] = counts["interval_count"] / counts.groupby(
        ["participant_id", "local_date"], observed=True
    )["interval_count"].transform("sum")
    counts["entropy_term"] = np.where(
        counts["p"] > 0,
        -counts["p"] * np.log(counts["p"]),
        0.0,
    )
    entropy = (
        counts.groupby(["participant_id", "local_date"], observed=True, as_index=False)["entropy_term"]
        .sum()
        .rename(columns={"entropy_term": "daily_template_entropy"})
    )
    entropy["daily_template_entropy"] /= LOG_N_TEMPLATE

    daily = (
        totals.merge(dominant, on=["participant_id", "local_date"], how="left", validate="one_to_one")
        .merge(entropy, on=["participant_id", "local_date"], how="left", validate="one_to_one")
    )
    daily["daily_dominant_template_share"] = (
        daily["daily_dominant_template_count"] / daily["daily_interval_count"]
    )

    return daily[
        [
            "participant_id",
            "local_date",
            "daily_dominant_template_id",
            "daily_dominant_template_code",
            "daily_dominant_template_share",
            "daily_template_entropy",
            "daily_unique_template_count",
            "daily_interval_count",
        ]
    ].sort_values(["participant_id", "local_date"], kind="mergesort").reset_index(drop=True)


def build_participant_profile(g: pd.DataFrame, min_history_days: int) -> pd.DataFrame:
    g = g.sort_values("local_date", kind="mergesort").reset_index(drop=True)
    n = len(g)

    dates = pd.to_datetime(g["local_date"]).to_numpy(dtype="datetime64[D]")
    day_ord = dates.astype(np.int64)
    codes = g["daily_dominant_template_code"].to_numpy(np.int16)
    purity = pd.to_numeric(g["daily_dominant_template_share"], errors="coerce").to_numpy(float)
    dent = pd.to_numeric(g["daily_template_entropy"], errors="coerce").to_numpy(float)
    duniq = pd.to_numeric(g["daily_unique_template_count"], errors="coerce").to_numpy(float)

    switch = np.zeros(n, dtype=np.int8)
    if n >= 2:
        switch[1:] = (codes[1:] != codes[:-1]).astype(np.int8)

    reversion = np.zeros(n, dtype=np.int8)
    if n >= 3:
        reversion[2:] = ((codes[2:] != codes[1:-1]) & (codes[2:] == codes[:-2])).astype(np.int8)

    runlen = np.ones(n, dtype=np.int32)
    for j in range(1, n):
        if codes[j] == codes[j - 1]:
            runlen[j] = runlen[j - 1] + 1

    tprefix = np.zeros((n + 1, N_TEMPLATE), dtype=np.int32)
    dprefix = np.zeros((n + 1, N_TEMPLATE), dtype=np.int32)
    for j in range(n):
        tprefix[j + 1] = tprefix[j]
        tprefix[j + 1, codes[j]] += 1
        dprefix[j + 1] = dprefix[j]
        if switch[j]:
            dprefix[j + 1, codes[j]] += 1

    sprefix = np.r_[0, np.cumsum(switch, dtype=np.int32)]
    rprefix = np.r_[0, np.cumsum(reversion, dtype=np.int32)]

    pprefix = np.r_[0.0, np.cumsum(np.nan_to_num(purity, nan=0.0))]
    pvalid = np.r_[0, np.cumsum(np.isfinite(purity).astype(np.int32))]
    eprefix = np.r_[0.0, np.cumsum(np.nan_to_num(dent, nan=0.0))]
    evalid = np.r_[0, np.cumsum(np.isfinite(dent).astype(np.int32))]

    features = {c: np.full(n, np.nan, dtype=np.float32) for c in TRANSITION_FEATURES}

    last_switch_idx = -1

    for i in range(n):
        features["tr_hist_active_days"][i] = i
        if i == 0:
            continue

        if switch[i - 1]:
            last_switch_idx = i - 1

        features["tr_lag1_daily_dominant_share"][i] = purity[i - 1]
        features["tr_lag1_daily_template_entropy"][i] = dent[i - 1]
        features["tr_lag1_daily_unique_template_count"][i] = duniq[i - 1]
        features["tr_dwell_active_days"][i] = runlen[i - 1]
        features["tr_ever_switched"][i] = 1.0 if sprefix[i] > 0 else 0.0

        if last_switch_idx >= 0:
            features["tr_days_since_last_switch"][i] = float(day_ord[i] - day_ord[last_switch_idx])

        left = {
            w: int(np.searchsorted(day_ord, day_ord[i] - w, side="left"))
            for w in (7, 30, 90)
        }

        for w in (7, 30, 90):
            l = left[w]
            event_l = max(l, 1)
            opportunities = max(0, i - event_l)
            switch_count = int(sprefix[i] - sprefix[event_l])
            if opportunities > 0:
                features[f"tr_switch_rate_{w}d"][i] = switch_count / opportunities
            if w in (30, 90):
                features[f"tr_switch_count_{w}d"][i] = switch_count

        for w in (7, 30):
            l = left[w]
            counts = tprefix[i] - tprefix[l]
            total = int(counts.sum())
            if total > 0:
                features[f"tr_dominant_share_{w}d"][i] = counts.max() / total
                features[f"tr_template_entropy_{w}d"][i] = normalized_entropy(counts)
                features[f"tr_unique_template_count_{w}d"][i] = int(np.sum(counts > 0))

            pn = int(pvalid[i] - pvalid[l])
            if pn > 0:
                features[f"tr_daily_purity_mean_{w}d"][i] = (pprefix[i] - pprefix[l]) / pn

            en = int(evalid[i] - evalid[l])
            if en > 0:
                features[f"tr_daily_entropy_mean_{w}d"][i] = (eprefix[i] - eprefix[l]) / en

        l90 = left[90]
        event_l90 = max(l90, 1)

        dest_counts = dprefix[i] - dprefix[event_l90]
        n_switch = int(dest_counts.sum())
        if n_switch > 0:
            features["tr_transition_dest_concentration_90d"][i] = dest_counts.max() / n_switch
            n_rev = int(rprefix[i] - rprefix[event_l90])
            features["tr_reversion_rate_90d"][i] = n_rev / n_switch

        origin = int(codes[i - 1])
        if i > event_l90:
            event_idx = np.arange(event_l90, i, dtype=np.int32)
            origin_mask = codes[event_idx - 1] == origin
            n_opp = int(origin_mask.sum())
            if n_opp > 0:
                origin_idx = event_idx[origin_mask]
                sw_mask = switch[origin_idx] == 1
                n_sw = int(sw_mask.sum())
                features["tr_origin_switch_rate_90d"][i] = n_sw / n_opp
                if n_sw > 0:
                    dc = np.bincount(codes[origin_idx[sw_mask]], minlength=N_TEMPLATE)
                    features["tr_origin_dest_concentration_90d"][i] = dc.max() / n_sw

    out = pd.DataFrame(
        {
            "participant_id": g["participant_id"].astype("string").to_numpy(),
            "local_date": pd.to_datetime(g["local_date"]).to_numpy(),
        }
    )
    for c in TRANSITION_FEATURES:
        out[c] = features[c]

    out["tr_ready_flag"] = (out["tr_hist_active_days"] >= min_history_days).astype(np.int8)
    return out


def build_transition_profile(daily: pd.DataFrame, min_history_days: int) -> pd.DataFrame:
    groups = list(daily.groupby("participant_id", observed=True, sort=False))
    frames = []
    total = len(groups)

    for i, (_, g) in enumerate(groups, 1):
        if i == 1 or i % 50 == 0 or i == total:
            print(f"[transition-profile] participant {i}/{total}", flush=True)
        frames.append(build_participant_profile(g, min_history_days))

    return pd.concat(frames, ignore_index=True).sort_values(
        ["participant_id", "local_date"], kind="mergesort"
    ).reset_index(drop=True)


def feature_schema() -> pd.DataFrame:
    desc = {
        "tr_hist_active_days": "Number of participant active days strictly before target day.",
        "tr_switch_rate_7d": "Historical dominant-template switch rate in prior 7 calendar days.",
        "tr_switch_rate_30d": "Historical dominant-template switch rate in prior 30 calendar days.",
        "tr_switch_rate_90d": "Historical dominant-template switch rate in prior 90 calendar days.",
        "tr_switch_count_30d": "Number of historical dominant-template switch events in prior 30 days.",
        "tr_switch_count_90d": "Number of historical dominant-template switch events in prior 90 days.",
        "tr_ever_switched": "1 if any historical dominant-template switch occurred before target day.",
        "tr_dwell_active_days": "Consecutive prior active days ending at D-1 with same dominant template.",
        "tr_days_since_last_switch": "Calendar days from target day to most recent historical switch event.",
        "tr_dominant_share_7d": "Maximum historical dominant-template share in prior 7 days.",
        "tr_dominant_share_30d": "Maximum historical dominant-template share in prior 30 days.",
        "tr_template_entropy_7d": "Normalized entropy of historical daily dominant templates in prior 7 days.",
        "tr_template_entropy_30d": "Normalized entropy of historical daily dominant templates in prior 30 days.",
        "tr_unique_template_count_7d": "Distinct historical daily dominant templates in prior 7 days.",
        "tr_unique_template_count_30d": "Distinct historical daily dominant templates in prior 30 days.",
        "tr_lag1_daily_dominant_share": "Previous active day's within-day dominant-template share.",
        "tr_lag1_daily_template_entropy": "Previous active day's within-day normalized template entropy.",
        "tr_lag1_daily_unique_template_count": "Previous active day's number of distinct templates.",
        "tr_daily_purity_mean_7d": "Mean within-day dominant-template share over prior 7 days.",
        "tr_daily_purity_mean_30d": "Mean within-day dominant-template share over prior 30 days.",
        "tr_daily_entropy_mean_7d": "Mean within-day template entropy over prior 7 days.",
        "tr_daily_entropy_mean_30d": "Mean within-day template entropy over prior 30 days.",
        "tr_transition_dest_concentration_90d": "Maximum destination share among switch events in prior 90 days.",
        "tr_origin_switch_rate_90d": "Prior 90-day switch rate conditional on current historical origin template.",
        "tr_origin_dest_concentration_90d": "Maximum destination share among prior 90-day switches from current origin.",
        "tr_reversion_rate_90d": "Share of prior 90-day switch events that are A->B->A reversions.",
    }

    rows = [
        {
            "column": c,
            "role": "feature",
            "feature_group": "transition_strategy_profile",
            "leakage_rule": "target day D uses only template history from dates < D",
            "description": desc[c],
        }
        for c in TRANSITION_FEATURES
    ]
    rows.append(
        {
            "column": "tr_ready_flag",
            "role": "readiness_flag",
            "feature_group": "transition_strategy_profile",
            "leakage_rule": "derived only from tr_hist_active_days",
            "description": "1 when prior active-day history reaches configured minimum.",
        }
    )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--root", default="data/processed/bidprediction")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--min-history-days", type=int, default=7)
    args = parser.parse_args()

    year_dir = ensure_dir(Path(args.root) / str(args.year))
    dataset_dir = year_dir / "dataset_parts"
    files = sorted(dataset_dir.glob("prediction_dataset_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction dataset files under {dataset_dir}")

    print("=" * 80)
    print("Build leakage-free transition_strategy_profile")
    print("=" * 80)
    print(f"Year:                 {args.year}")
    print(f"Dataset files:        {len(files)}")
    print(f"Minimum history days: {args.min_history_days}")
    print()

    daily = build_daily_template_state(files, args.chunksize)
    daily_file = year_dir / f"transition_daily_template_state_{args.year}.csv"
    daily.to_csv(daily_file, index=False, encoding="utf-8-sig", float_format="%.8g")

    profile = build_transition_profile(daily, args.min_history_days)
    profile_file = year_dir / f"transition_strategy_profile_{args.year}.csv"
    profile.to_csv(profile_file, index=False, encoding="utf-8-sig", float_format="%.8g")

    schema = feature_schema()
    schema_file = year_dir / f"transition_strategy_feature_schema_{args.year}.csv"
    schema.to_csv(schema_file, index=False, encoding="utf-8-sig")

    ordered = daily.sort_values(["participant_id", "local_date"], kind="mergesort").copy()
    ordered["prev_template"] = ordered.groupby("participant_id", observed=True)["daily_dominant_template_id"].shift(1)
    valid_opp = ordered["prev_template"].notna()
    is_switch = valid_opp & ordered["daily_dominant_template_id"].ne(ordered["prev_template"])
    switch_events = int(is_switch.sum())
    switch_opportunities = int(valid_opp.sum())

    ready_rows = int(profile["tr_ready_flag"].sum())
    ready_share = ready_rows / len(profile) if len(profile) else np.nan

    manifest = pd.DataFrame(
        [
            {
                "year": args.year,
                "participants": int(daily["participant_id"].nunique()),
                "participant_days": len(daily),
                "transition_profile_rows": len(profile),
                "transition_ready_rows": ready_rows,
                "transition_ready_share": ready_share,
                "historical_switch_events": switch_events,
                "historical_switch_opportunities": switch_opportunities,
                "daily_switch_rate": switch_events / switch_opportunities if switch_opportunities else np.nan,
                "median_daily_dominant_template_share": float(daily["daily_dominant_template_share"].median()),
                "median_daily_template_entropy": float(daily["daily_template_entropy"].median()),
                "transition_feature_count": len(TRANSITION_FEATURES),
                "min_history_days": args.min_history_days,
                "daily_state_file": str(daily_file),
                "transition_profile_file": str(profile_file),
                "feature_schema_file": str(schema_file),
            }
        ]
    )
    manifest_file = year_dir / f"transition_strategy_profile_manifest_{args.year}.csv"
    manifest.to_csv(manifest_file, index=False, encoding="utf-8-sig")

    missing = profile[TRANSITION_FEATURES].isna().mean().sort_values(ascending=False)

    lines = [
        f"Transition strategy profile - {args.year}",
        "=" * 80,
        "",
        f"Participants:                    {daily['participant_id'].nunique():,}",
        f"Participant-days:                {len(daily):,}",
        f"Transition-profile rows:         {len(profile):,}",
        f"Transition-ready rows:           {ready_rows:,} ({ready_share:.2%})",
        f"Transition feature count:        {len(TRANSITION_FEATURES)}",
        (
            f"Historical daily switch rate:    {switch_events:,}/{switch_opportunities:,} "
            f"({switch_events / switch_opportunities:.2%})"
            if switch_opportunities
            else "Historical daily switch rate:    n/a"
        ),
        f"Median daily dominant share:     {daily['daily_dominant_template_share'].median():.4f}",
        f"Median daily template entropy:   {daily['daily_template_entropy'].median():.6f}",
        "",
        "Strict leakage rule:",
        "  Every target day D uses only template history from dates < D.",
        "",
        "Most-missing transition features:",
    ]
    for c, rate in missing.head(15).items():
        lines.append(f"  {c:<42} {rate:>8.2%}")

    lines += [
        "",
        f"Daily state:        {daily_file}",
        f"Transition profile: {profile_file}",
        f"Feature schema:     {schema_file}",
        f"Manifest:           {manifest_file}",
    ]

    summary = "\n".join(lines)
    summary_file = year_dir / f"transition_strategy_profile_summary_{args.year}.txt"
    summary_file.write_text(summary, encoding="utf-8")

    print()
    print(summary)


if __name__ == "__main__":
    main()
