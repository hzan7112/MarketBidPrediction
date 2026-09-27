#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05d_build_final_strategy_profile.py
Version: 2026-09-17-v1

Freeze the history-derived strategy image after feature reduction.

Formal output:
    9 long-term core features
    9 short-term core states
    strategy-break event table

The full 04/05 outputs are preserved as audit/intermediate data. This script
creates the slim tables that downstream visualization/model code should use.

Run:
    python scripts/05d_build_final_strategy_profile.py --year 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


VERSION = "2026-09-17-v1"

LT_CORE = [
    "lt_bid_level",
    "lt_self_adjustment_magnitude",
    "lt_strategy_persistence",
    "lt_quantity_hhi",
    "lt_effective_segment_count",
    "lt_flat_curve_rate",
    "lt_tail_uplift_ratio",
    "lt_curve_bend_ratio",
    "lt_shape_day_deviation_median",
]

ST_CORE = [
    "st_bid_level_z",
    "st_adjustment_bias_z",
    "st_adjustment_magnitude_z",
    "st_quantity_hhi_z",
    "st_effective_segment_count_z",
    "st_flat_curve_rate_z",
    "st_tail_uplift_ratio_z",
    "st_curve_bend_ratio_z",
    "st_shape_prototype_shift",
]

CORE_BREAK_STEMS = [
    "bid_level",
    "adjustment_bias",
    "adjustment_magnitude",
    "quantity_hhi",
    "effective_segment_count",
    "flat_curve_rate",
    "tail_uplift_ratio",
    "curve_bend_ratio",
]

FEATURE_DICTIONARY = [
    ("LT", "price_position", "lt_bid_level", "长期典型报价水平",
     "主体长期典型报价价格水平；当前阶段为自身历史定位，后续再补市场相对定位。"),
    ("LT", "adjustment", "lt_self_adjustment_magnitude", "长期调整幅度",
     "主体相对自身同类时段历史基准的典型调整幅度。"),
    ("LT", "adjustment", "lt_strategy_persistence", "策略持续性",
     "主体进入偏高/偏低策略状态后跨日延续的倾向。"),
    ("LT", "quantity_allocation", "lt_quantity_hhi", "容量配置集中度",
     "报价容量是否长期集中在少数报价段。"),
    ("LT", "curve_organization", "lt_effective_segment_count", "有效报价段数",
     "主体长期采用简单还是精细分段的报价结构。"),
    ("LT", "curve_organization", "lt_flat_curve_rate", "平价曲线偏好",
     "主体长期使用平价/近似单一价格曲线的比例。"),
    ("LT", "tail_strategy", "lt_tail_uplift_ratio", "尾部抬价集中度",
     "总价格变化中集中在高容量尾部的比例。"),
    ("LT", "tail_strategy", "lt_curve_bend_ratio", "曲线弯折特征",
     "曲线尾段相对头段的价格增长结构。"),
    ("LT", "shape_stability", "lt_shape_day_deviation_median", "长期形态稳定性",
     "日级典型曲线相对主体长期典型形态的常规偏离程度。"),

    ("ST", "price_position", "st_bid_level_z", "近期报价水平偏移",
     "最近窗口报价水平相对更早长期基准的稳健标准化偏移。"),
    ("ST", "adjustment", "st_adjustment_bias_z", "近期调整方向",
     "最近主体整体向更高或更低报价方向偏移的程度。"),
    ("ST", "adjustment", "st_adjustment_magnitude_z", "近期调整强度变化",
     "最近调整幅度相对长期习惯的变化。"),
    ("ST", "quantity_allocation", "st_quantity_hhi_z", "近期容量配置变化",
     "最近容量集中/分散程度相对长期习惯的变化。"),
    ("ST", "curve_organization", "st_effective_segment_count_z", "近期分段复杂度变化",
     "最近有效报价段数相对长期习惯的变化。"),
    ("ST", "curve_organization", "st_flat_curve_rate_z", "近期平价结构变化",
     "最近采用平价曲线的倾向相对长期习惯的变化。"),
    ("ST", "tail_strategy", "st_tail_uplift_ratio_z", "近期尾部抬价变化",
     "最近尾部抬价集中度相对长期习惯的变化。"),
    ("ST", "tail_strategy", "st_curve_bend_ratio_z", "近期弯折结构变化",
     "最近曲线前后段增长结构相对长期习惯的变化。"),
    ("ST", "shape_stability", "st_shape_prototype_shift", "近期形态迁移",
     "最近典型曲线与更早长期典型曲线之间的形态距离。"),
]


def check_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--lt-root",
        default="data/processed/long_term_strategy_profile",
    )
    p.add_argument(
        "--st-root",
        default="data/processed/short_term_strategy_state",
    )
    p.add_argument(
        "--output-root",
        default="data/processed/final_strategy_profile",
    )
    p.add_argument(
        "--results-root",
        default="results/05d_final_profile",
    )
    args = p.parse_args()

    lt_file = (
        Path(args.lt_root) / str(args.year)
        / f"long_term_strategy_profile_{args.year}.csv"
    )
    st_file = (
        Path(args.st_root) / str(args.year)
        / f"short_term_strategy_state_{args.year}.csv"
    )

    if not lt_file.exists():
        raise FileNotFoundError(lt_file)
    if not st_file.exists():
        raise FileNotFoundError(st_file)

    out_dir = Path(args.output_root) / str(args.year)
    results_dir = Path(args.results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[read] {lt_file}")
    lt = pd.read_csv(lt_file, low_memory=False)
    print(f"[read] {st_file}")
    st = pd.read_csv(st_file, low_memory=False)

    check_columns(
        lt,
        ["participant_id", "market_product"] + LT_CORE,
        "long-term profile",
    )
    check_columns(
        st,
        [
            "participant_id", "market_product", "state_date",
            "recent_active_days", "long_active_days",
        ] + ST_CORE,
        "short-term state",
    )

    break_cols = [
        f"{stem}_zero_scale_change_flag" for stem in CORE_BREAK_STEMS
    ]
    check_columns(st, break_cols, "short-term state break flags")

    # -------- Final long-term core profile --------
    lt_out = lt[
        ["participant_id", "market_product"] + LT_CORE
    ].copy()

    lt_out["lt_core_nonmissing_count"] = (
        lt_out[LT_CORE].notna().sum(axis=1).astype(np.int16)
    )
    lt_out["lt_core_coverage"] = (
        lt_out["lt_core_nonmissing_count"] / len(LT_CORE)
    )
    # Shape and persistence are optional for some legitimate strategy types.
    lt_out["lt_core_ready_flag"] = (
        lt_out["lt_core_nonmissing_count"] >= 7
    ).astype(np.int8)

    lt_output = out_dir / f"final_long_term_profile_{args.year}.csv"
    lt_out.to_csv(lt_output, index=False)

    # -------- Strategy break events --------
    breaks = st[
        ["participant_id", "market_product", "state_date"] + break_cols
    ].copy()

    for c in break_cols:
        breaks[c] = pd.to_numeric(
            breaks[c], errors="coerce"
        ).fillna(0).astype(np.int8)

    breaks["strategy_break_count"] = (
        breaks[break_cols].sum(axis=1).astype(np.int16)
    )
    breaks["strategy_break_any_flag"] = (
        breaks["strategy_break_count"] > 0
    ).astype(np.int8)

    def break_types(row):
        names = []
        for stem, col in zip(CORE_BREAK_STEMS, break_cols):
            if row[col] == 1:
                names.append(stem)
        return "|".join(names)

    breaks["strategy_break_types"] = breaks.apply(
        break_types, axis=1
    )

    break_output = out_dir / f"strategy_break_events_{args.year}.csv"
    breaks.to_csv(break_output, index=False)

    # -------- Final short-term core state --------
    st_out = st[
        [
            "participant_id",
            "market_product",
            "state_date",
            "recent_active_days",
            "long_active_days",
        ] + ST_CORE
    ].copy()

    # For the 8 scalar z-features, a zero-scale structural break still means
    # the state dimension is represented by the separate break-event table.
    scalar_core = ST_CORE[:-1]
    represented = np.zeros((len(st_out), len(scalar_core)), dtype=bool)

    for j, feature in enumerate(scalar_core):
        stem = feature[len("st_"):-len("_z")]
        bcol = f"{stem}_zero_scale_change_flag"
        represented[:, j] = (
            st_out[feature].notna().to_numpy()
            | st[bcol].fillna(0).eq(1).to_numpy()
        )

    st_out["st_core_scalar_represented_count"] = (
        represented.sum(axis=1).astype(np.int16)
    )
    st_out["st_shape_state_available_flag"] = (
        st_out["st_shape_prototype_shift"].notna().astype(np.int8)
    )
    st_out["st_core_ready_flag"] = (
        st_out["st_core_scalar_represented_count"] >= 6
    ).astype(np.int8)

    st_output = out_dir / f"final_short_term_state_{args.year}.csv"
    st_out.to_csv(st_output, index=False)

    # -------- Feature dictionary --------
    fd = pd.DataFrame(
        FEATURE_DICTIONARY,
        columns=[
            "time_scale",
            "dimension",
            "feature",
            "chinese_name",
            "interpretation",
        ],
    )
    fd_output = out_dir / f"final_feature_dictionary_{args.year}.csv"
    fd.to_csv(fd_output, index=False, encoding="utf-8-sig")

    # -------- Summary --------
    lt_ready = lt_out["lt_core_ready_flag"].eq(1)
    st_ready = st_out["st_core_ready_flag"].eq(1)
    brk = breaks["strategy_break_any_flag"].eq(1)

    lines = [
        f"Final history-derived strategy profile summary - {args.year}",
        f"Version: {VERSION}",
        "=" * 72,
        f"Participants: {lt_out['participant_id'].nunique():,}",
        f"Participant-day short-term states: {len(st_out):,}",
        "",
        "Final formal representation:",
        f"  Long-term core features: {len(LT_CORE)}",
        f"  Short-term core states: {len(ST_CORE)}",
        f"  Core break dimensions: {len(CORE_BREAK_STEMS)}",
        "",
        f"Long-term core-ready participants: {lt_ready.sum():,} "
        f"({lt_ready.mean()*100:.2f}%)",
        f"Short-term core-ready states: {st_ready.sum():,} "
        f"({st_ready.mean()*100:.2f}%)",
        f"Participant-days with >=1 strategy break: {brk.sum():,} "
        f"({brk.mean()*100:.2f}%)",
        "",
        "Final LT core:",
    ] + [f"  {x}" for x in LT_CORE] + [
        "",
        "Final ST core:",
    ] + [f"  {x}" for x in ST_CORE] + [
        "",
        "Removed from formal core but preserved upstream:",
        "  lt_self_adjustment_bias",
        "  lt_self_adjustment_p90",
        "  lt_bid_level_scale (normalization helper)",
        "  lt_shape_defined_rate (QC/applicability)",
        "  curve_mode_switch_rate",
        "  flat_curve_switch_rate (rare-event auxiliary)",
        "  st_adjacent_level_change_z",
        "  st_same_slot_shape_change_z",
        "  st_adjacent_shape_change_z (auxiliary)",
        "",
        f"Final LT profile: {lt_output.resolve()}",
        f"Final ST state: {st_output.resolve()}",
        f"Break events: {break_output.resolve()}",
        f"Feature dictionary: {fd_output.resolve()}",
    ]

    summary_file = results_dir / f"summary_{args.year}.txt"
    summary_file.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
