#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05e_validate_final_strategy_profile.py
Version: 2026-09-17-v1

Re-validate only the frozen final 9+9+Break representation.

Fixes from the earlier exploratory validator:
- validation is restricted to final core features;
- win/tie/loss are mutually exclusive;
- zero-within-scale discrete/structural features are not assigned absurd
  infinite separation ratios;
- no arbitrary overall score is created.

Run:
    python scripts/05e_validate_final_strategy_profile.py --year 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


VERSION = "2026-09-17-v1"
ROBUST_SCALE = 1.4826
EPS = 1e-12

LT_DAILY_MAP = {
    "lt_bid_level": "daily_bid_level",
    "lt_self_adjustment_magnitude": "daily_self_adjustment_magnitude",
    "lt_quantity_hhi": "daily_quantity_hhi",
    "lt_effective_segment_count": "daily_effective_segment_count",
    "lt_flat_curve_rate": "daily_flat_curve_rate",
    "lt_tail_uplift_ratio": "daily_tail_uplift_ratio",
    "lt_curve_bend_ratio": "daily_curve_bend_ratio",
    "lt_shape_day_deviation_median": "daily_shape_deviation",
}

ST_TARGET_MAP = {
    "st_bid_level_z": ("bid_level", "daily_bid_level"),
    "st_adjustment_bias_z": ("adjustment_bias", "daily_self_adjustment_bias"),
    "st_adjustment_magnitude_z": (
        "adjustment_magnitude", "daily_self_adjustment_magnitude"
    ),
    "st_quantity_hhi_z": ("quantity_hhi", "daily_quantity_hhi"),
    "st_effective_segment_count_z": (
        "effective_segment_count", "daily_effective_segment_count"
    ),
    "st_flat_curve_rate_z": ("flat_curve_rate", "daily_flat_curve_rate"),
    "st_tail_uplift_ratio_z": (
        "tail_uplift_ratio", "daily_tail_uplift_ratio"
    ),
    "st_curve_bend_ratio_z": (
        "curve_bend_ratio", "daily_curve_bend_ratio"
    ),
}


def rscale(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    med = x.median()
    return float(ROBUST_SCALE * (x - med).abs().median())


def spearman(a, b, min_n=20):
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < min_n or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan, n
    return float(x[ok].corr(y[ok], method="spearman")), n


def split_half(daily, min_days):
    d = daily.copy()
    d["local_date"] = pd.to_datetime(d["local_date"], errors="coerce")
    d["half"] = np.where(d["local_date"].dt.month <= 6, "H1", "H2")
    keys = ["participant_id", "market_product"]

    rows = []
    for lt_feature, daily_feature in LT_DAILY_MAP.items():
        if daily_feature not in d.columns:
            continue
        a = (
            d.groupby(keys + ["half"], sort=False)[daily_feature]
            .agg(["median", "count"])
            .reset_index()
        )
        h1 = a[
            (a["half"] == "H1") & (a["count"] >= min_days)
        ][keys + ["median"]].rename(columns={"median": "h1"})
        h2 = a[
            (a["half"] == "H2") & (a["count"] >= min_days)
        ][keys + ["median"]].rename(columns={"median": "h2"})
        m = h1.merge(h2, on=keys, how="inner")
        rho, n = spearman(m["h1"], m["h2"])
        rows.append({
            "lt_feature": lt_feature,
            "daily_feature": daily_feature,
            "participants_compared": n,
            "spearman_h1_h2": rho,
            "median_abs_half_shift": (
                (m["h2"] - m["h1"]).abs().median()
                if len(m) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def between_within(daily, min_days):
    keys = ["participant_id", "market_product"]
    rows = []

    for lt_feature, daily_feature in LT_DAILY_MAP.items():
        if daily_feature not in daily.columns:
            continue

        items = []
        for key, g in daily.groupby(keys, sort=False):
            x = pd.to_numeric(g[daily_feature], errors="coerce").dropna()
            if len(x) < min_days:
                continue
            items.append({
                "participant_id": key[0],
                "market_product": key[1],
                "median": x.median(),
                "within_scale": rscale(x),
            })

        p = pd.DataFrame(items)
        if len(p):
            between = rscale(p["median"])
            within = pd.to_numeric(
                p["within_scale"], errors="coerce"
            ).median()
        else:
            between = np.nan
            within = np.nan

        ratio = np.nan
        stable_separation = 0
        if np.isfinite(between) and np.isfinite(within):
            if within > EPS:
                ratio = between / within
            elif between > EPS:
                stable_separation = 1

        if stable_separation:
            interpretation = "stable_cross_participant_difference"
        elif np.isfinite(ratio) and ratio > 1:
            interpretation = "between_gt_within"
        elif np.isfinite(ratio):
            interpretation = "within_ge_between"
        elif (
            np.isfinite(between) and between <= EPS
            and np.isfinite(within) and within <= EPS
        ):
            interpretation = "mostly_degenerate_or_discrete"
        else:
            interpretation = "not_comparable"

        rows.append({
            "lt_feature": lt_feature,
            "daily_feature": daily_feature,
            "participants_used": len(p),
            "between_participant_robust_scale": between,
            "median_within_participant_robust_scale": within,
            "between_within_ratio": ratio,
            "stable_separation_flag": stable_separation,
            "interpretation": interpretation,
        })

    return pd.DataFrame(rows)


def redundancy(final_lt):
    core = [
        c for c in LT_DAILY_MAP
        if c in final_lt.columns
    ] + (
        ["lt_strategy_persistence"]
        if "lt_strategy_persistence" in final_lt.columns else []
    )

    x = final_lt[core].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman", min_periods=20)

    pairs = []
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            v = corr.iloc[i, j]
            if np.isfinite(v):
                pairs.append({
                    "feature_1": core[i],
                    "feature_2": core[j],
                    "spearman": float(v),
                    "abs_spearman": float(abs(v)),
                })
    pairs = pd.DataFrame(pairs)
    if len(pairs):
        pairs = pairs.sort_values(
            "abs_spearman", ascending=False
        ).reset_index(drop=True)
    return corr, pairs


def causal_signal(daily, original_st):
    d = daily.copy()
    d["local_date"] = pd.to_datetime(
        d["local_date"], errors="coerce"
    ).dt.date
    s = original_st.copy()
    s["state_date"] = pd.to_datetime(
        s["state_date"], errors="coerce"
    ).dt.date

    rows = []
    for st_feature, (stem, target_col) in ST_TARGET_MAP.items():
        recent_col = f"{stem}_recent"
        baseline_col = f"{stem}_baseline"
        raw_col = f"st_{stem}_raw_shift"

        needed = [
            recent_col, baseline_col, raw_col, "state_date",
            "participant_id", "market_product",
        ]
        if any(c not in s.columns for c in needed):
            continue

        target = d[
            ["participant_id", "market_product", "local_date", target_col]
        ].copy()

        m = s[needed].merge(
            target,
            left_on=["participant_id", "market_product", "state_date"],
            right_on=["participant_id", "market_product", "local_date"],
            how="inner",
        )

        y = pd.to_numeric(m[target_col], errors="coerce")
        recent = pd.to_numeric(m[recent_col], errors="coerce")
        baseline = pd.to_numeric(m[baseline_col], errors="coerce")
        shift = pd.to_numeric(m[raw_col], errors="coerce")

        ok = y.notna() & recent.notna() & baseline.notna()
        y, recent, baseline, shift = (
            y[ok], recent[ok], baseline[ok], shift[ok]
        )

        e_recent = (y - recent).abs().to_numpy(float)
        e_long = (y - baseline).abs().to_numpy(float)

        tie = np.isclose(e_recent, e_long, rtol=1e-9, atol=1e-12)
        win = (e_recent < e_long) & ~tie
        loss = (e_recent > e_long) & ~tie

        long_mae = float(np.mean(e_long)) if len(e_long) else np.nan
        recent_mae = float(np.mean(e_recent)) if len(e_recent) else np.nan
        improve = (
            100 * (long_mae - recent_mae) / long_mae
            if np.isfinite(long_mae) and long_mae > EPS else np.nan
        )
        rho, n = spearman(shift, y - baseline)

        rows.append({
            "st_feature": st_feature,
            "target_feature": target_col,
            "samples": len(y),
            "long_baseline_mae": long_mae,
            "recent_window_mae": recent_mae,
            "mae_improvement_pct": improve,
            "win_rate": float(win.mean()) if len(win) else np.nan,
            "tie_rate": float(tie.mean()) if len(tie) else np.nan,
            "loss_rate": float(loss.mean()) if len(loss) else np.nan,
            "spearman_recent_shift_vs_current_deviation": rho,
            "correlation_samples": n,
        })

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--daily-root",
        default="data/processed/daily_strategy_summary",
    )
    p.add_argument(
        "--final-root",
        default="data/processed/final_strategy_profile",
    )
    p.add_argument(
        "--original-st-root",
        default="data/processed/short_term_strategy_state",
    )
    p.add_argument(
        "--results-root",
        default="results/05e_final_profile_validation",
    )
    p.add_argument("--min-half-days", type=int, default=20)
    p.add_argument("--min-profile-days", type=int, default=30)
    args = p.parse_args()

    daily_file = (
        Path(args.daily_root) / str(args.year)
        / f"daily_strategy_summary_{args.year}.csv"
    )
    final_lt_file = (
        Path(args.final_root) / str(args.year)
        / f"final_long_term_profile_{args.year}.csv"
    )
    original_st_file = (
        Path(args.original_st_root) / str(args.year)
        / f"short_term_strategy_state_{args.year}.csv"
    )

    for f in [daily_file, final_lt_file, original_st_file]:
        if not f.exists():
            raise FileNotFoundError(f)

    out_dir = Path(args.results_root) / str(args.year)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(daily_file, low_memory=False)
    final_lt = pd.read_csv(final_lt_file, low_memory=False)
    original_st = pd.read_csv(original_st_file, low_memory=False)

    print("[validate] split-half stability")
    stable = split_half(daily, args.min_half_days)
    stable.to_csv(
        out_dir / f"final_split_half_stability_{args.year}.csv",
        index=False,
    )

    print("[validate] between/within")
    sep = between_within(daily, args.min_profile_days)
    sep.to_csv(
        out_dir / f"final_between_within_{args.year}.csv",
        index=False,
    )

    print("[validate] redundancy")
    corr, pairs = redundancy(final_lt)
    corr.to_csv(
        out_dir / f"final_long_term_correlation_{args.year}.csv"
    )
    pairs.to_csv(
        out_dir / f"final_long_term_correlation_pairs_{args.year}.csv",
        index=False,
    )

    print("[validate] causal short-term signal")
    causal = causal_signal(daily, original_st)
    causal.to_csv(
        out_dir / f"final_short_term_causal_signal_{args.year}.csv",
        index=False,
    )

    rhos = pd.to_numeric(
        stable["spearman_h1_h2"], errors="coerce"
    ).dropna()
    imps = pd.to_numeric(
        causal["mae_improvement_pct"], errors="coerce"
    ).dropna()
    high_corr = (
        pairs[pairs["abs_spearman"] >= 0.90]
        if len(pairs) else pairs
    )

    lines = [
        f"Final strategy-profile validation - {args.year}",
        f"Version: {VERSION}",
        "=" * 72,
        f"Participants: {final_lt['participant_id'].nunique():,}",
        "",
        "1. Long-term split-half stability",
        f"Validated final LT traits: {len(rhos)}",
        f"Median H1-H2 Spearman: "
        f"{rhos.median():.4f}" if len(rhos)
        else "Median H1-H2 Spearman: NA",
        f"Positive H1-H2 traits: "
        f"{int((rhos > 0).sum())}/{len(rhos)}" if len(rhos)
        else "Positive H1-H2 traits: NA",
        "",
        "2. Final-core redundancy",
        f"Pairs with |Spearman| >= 0.90: {len(high_corr)}",
        "",
        "3. Causal short-term signal",
        f"Validated final scalar ST states: {len(imps)}",
        f"Median recent-window MAE improvement: "
        f"{imps.median():.2f}%" if len(imps)
        else "Median recent-window MAE improvement: NA",
        f"Positive MAE improvement: "
        f"{int((imps > 0).sum())}/{len(imps)}" if len(imps)
        else "Positive MAE improvement: NA",
        "",
        "Win/tie/loss are mutually exclusive in this version.",
        "Discrete or zero-within-scale traits are flagged instead of producing",
        "artificially huge separation ratios.",
        "",
        f"Results directory: {out_dir.resolve()}",
    ]

    summary = out_dir / f"summary_{args.year}.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
