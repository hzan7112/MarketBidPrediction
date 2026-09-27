#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from annual_profile_core import (
    ANNUAL_FEATURES,
    ANNUAL_SHAPE_COLS,
    build_group_profile,
    empirical_percentile,
)

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def feature_manifest():
    return pd.DataFrame([
        ["annual_bid_level", "报价水平", "median(daily_bid_level)", "主体在源年度通常处于什么报价水平"],
        ["annual_adjustment_magnitude", "调整幅度", "median(daily_adjustment_magnitude)", "相对自身同时槽历史基线，主体通常调整多大"],
        ["annual_strategy_persistence", "策略持续性", "lag1 corr(daily_adjustment_bias)", "调整方向与幅度是否在连续日期间持续"],
        ["annual_quantity_hhi", "容量集中度", "median(daily_quantity_hhi)", "容量长期集中还是分散"],
        ["annual_effective_segment_count", "有效段数", "median(daily_effective_segment_count)", "长期报价结构简单还是精细分段"],
        ["annual_flat_curve_rate", "平价偏好", "mean(daily_flat_curve_rate)", "源年度使用平价结构的日均比例"],
        ["annual_tail_uplift_ratio", "尾部抬价", "median(daily_tail_uplift_ratio)", "高容量尾部长期承载价格增长的程度"],
        ["annual_curve_bend_ratio", "曲线弯折", "mean(daily_curve_bend_ratio)", "价格增长长期更偏头部还是尾部"],
        ["annual_shape_variability", "形态波动", "median RMSE(daily shape, annual prototype)", "归一化曲线形状跨日变化程度"],
    ], columns=["feature", "name_cn", "aggregation", "meaning"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--daily-root", default="data/processed/final_clean/daily")
    p.add_argument("--out-root", default="data/processed/bidprofile_annual")
    p.add_argument("--min-active-days", type=int, default=30)
    p.add_argument("--min-core-features", type=int, default=7)
    p.add_argument("--min-persistence-pairs", type=int, default=5)
    p.add_argument("--min-shape-days", type=int, default=2)
    args = p.parse_args()

    src = Path(args.daily_root) / str(args.year) / f"daily_strategy_core_{args.year}.csv"
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = ensure_dir(Path(args.out_root) / str(args.year))

    df = pd.read_csv(src, low_memory=False)
    df["local_date"] = pd.to_datetime(df["local_date"]).dt.normalize()

    years = sorted(df["local_date"].dropna().dt.year.unique().tolist())
    if years != [args.year]:
        raise ValueError(f"Expected source year {args.year}, found years={years}")

    annual_rows, monthly_rows = [], []
    groups = list(df.groupby("participant_id", sort=False))
    total = len(groups)

    for idx, (pid, g) in enumerate(groups, start=1):
        r = build_group_profile(
            g,
            min_persistence_pairs=args.min_persistence_pairs,
            min_shape_days=args.min_shape_days,
        )
        r["participant_id"] = pid
        r["profile_year"] = args.year
        annual_rows.append(r)

        gm = g.copy()
        gm["profile_month"] = gm["local_date"].dt.to_period("M").astype(str)
        for month, m in gm.groupby("profile_month", sort=True):
            mr = build_group_profile(
                m,
                min_persistence_pairs=3,
                min_shape_days=2,
            )
            mr["participant_id"] = pid
            mr["profile_year"] = args.year
            mr["profile_month"] = month
            monthly_rows.append(mr)

        if idx == 1 or idx % 100 == 0 or idx == total:
            print(f"[02 annual] participants {idx:,}/{total:,}", flush=True)

    profile = pd.DataFrame(annual_rows)
    monthly = pd.DataFrame(monthly_rows)

    profile["annual_nonmissing_count"] = profile[ANNUAL_FEATURES].notna().sum(axis=1)
    profile["annual_complete_9d_flag"] = (
        profile["annual_nonmissing_count"].eq(len(ANNUAL_FEATURES))
    ).astype(int)
    profile["annual_profile_ready_flag"] = (
        (profile["active_days"] >= args.min_active_days)
        & (profile["annual_nonmissing_count"] >= args.min_core_features)
    ).astype(int)

    for f in ANNUAL_FEATURES:
        profile[f"{f}_percentile"] = empirical_percentile(profile[f])

    profile["profile_scope"] = "full_source_year_static"
    profile["same_year_use"] = "retrospective_explanatory_only"
    profile["future_year_use"] = "freeze_then_use_without_future_bids"

    front = [
        "participant_id", "profile_year", "source_start_date", "source_end_date",
        "active_days", "active_months", "calendar_span_days"
    ]
    qc = [c for c in profile.columns if (
        c.endswith("_valid_days")
        or c.endswith("_valid_pairs")
        or c.endswith("_fallback_flag")
        or c in ["annual_shape_defined_days", "annual_shape_defined_rate"]
    )]
    ordered = (
        front
        + ANNUAL_FEATURES
        + ANNUAL_SHAPE_COLS
        + qc
        + ["annual_nonmissing_count", "annual_complete_9d_flag", "annual_profile_ready_flag"]
        + [f"{f}_percentile" for f in ANNUAL_FEATURES]
        + ["profile_scope", "same_year_use", "future_year_use"]
    )
    ordered = list(dict.fromkeys(c for c in ordered if c in profile.columns))
    profile = profile[ordered]

    profile_csv = out_dir / f"annual_strategy_profile_{args.year}.csv"
    monthly_csv = out_dir / f"annual_strategy_profile_monthly_{args.year}.csv"
    manifest_csv = out_dir / f"annual_strategy_profile_feature_manifest_{args.year}.csv"
    manifest_json = out_dir / f"annual_strategy_profile_feature_manifest_{args.year}.json"

    profile.to_csv(profile_csv, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_csv, index=False, encoding="utf-8-sig")
    feature_manifest().to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    manifest_json.write_text(json.dumps({
        "profile_year": args.year,
        "model_input_features": ANNUAL_FEATURES,
        "profile_definition": "full source-year frozen static profile",
        "curve_bend_aggregation": "mean of daily signed bend ratios",
        "shape_variability_all_flat_rule": "all-flat source-year participant -> 0 with fallback flag",
        "same_year_2025_use": "retrospective explanatory-power test only",
        "forbidden_future_inputs": [
            "target-year lag1 bid",
            "target-year recent bid statistics",
            "target-year previous template",
            "target-year previous latent",
            "target-year ST/Break requiring realized bids",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    parquet_file = out_dir / f"annual_strategy_profile_{args.year}.parquet"
    try:
        profile.to_parquet(parquet_file, index=False)
        parquet_msg = str(parquet_file)
    except Exception as e:
        parquet_msg = f"not written ({type(e).__name__}: {e})"

    print()
    print(f"Participants: {len(profile):,}")
    print(f"Ready: {int(profile['annual_profile_ready_flag'].sum()):,} "
          f"({profile['annual_profile_ready_flag'].mean()*100:.2f}%)")
    print(f"Complete 9D: {int(profile['annual_complete_9d_flag'].sum()):,} "
          f"({profile['annual_complete_9d_flag'].mean()*100:.2f}%)")
    print("Feature coverage:")
    for f in ANNUAL_FEATURES:
        n = int(profile[f].notna().sum())
        print(f"  {f:<36s} {n:>5,}/{len(profile):,} ({n/len(profile)*100:6.2f}%)")

    print()
    print(f"Done: {profile_csv}")
    print(f"Monthly diagnostics: {monthly_csv}")
    print(f"Manifest: {manifest_csv}")
    print(f"Parquet: {parquet_msg}")

if __name__ == "__main__":
    main()
