#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05c_validate_strategy_profile_stage.py
Version: 2026-09-17-v1

Stage validation for the 2025 bidding-strategy profile.

This is NOT a new profile-construction step and does not require external
market data. It evaluates whether the history-derived profile is:

1) stable across time,
2) discriminative across participants,
3) non-redundant,
4) useful as a causal short-term state.

Run:
    python scripts/05c_validate_strategy_profile_stage.py --year 2025

Inputs:
    data/processed/daily_strategy_summary/2025/daily_strategy_summary_2025.csv
    data/processed/long_term_strategy_profile/2025/long_term_strategy_profile_2025.csv
    data/processed/short_term_strategy_state/2025/short_term_strategy_state_2025.csv

Outputs:
    results/profile_stage_validation/2025/
        summary_2025.txt
        split_half_stability_2025.csv
        between_within_separation_2025.csv
        long_term_feature_correlation_2025.csv
        highly_correlated_feature_pairs_2025.csv
        causal_short_term_signal_2025.csv
        short_term_z_distribution_2025.csv
        representative_participants_2025.csv
        figures/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VERSION = "2026-09-17-v1"
ROBUST_SCALE = 1.4826
EPS = 1e-12


DAILY_TO_STEM = {
    # Price / adjustment
    "daily_bid_level": "bid_level",
    "daily_self_adjustment_bias": "adjustment_bias",
    "daily_self_adjustment_magnitude": "adjustment_magnitude",
    "daily_adjacent_level_change_magnitude": "adjacent_level_change",

    # Quantity allocation
    "daily_quantity_hhi": "quantity_hhi",

    # Curve organization
    "daily_effective_segment_count": "effective_segment_count",
    "daily_flat_curve_rate": "flat_curve_rate",
    "daily_tail_uplift_ratio": "tail_uplift_ratio",
    "daily_curve_bend_ratio": "curve_bend_ratio",

    # Adjustment / switching
    "daily_same_slot_shape_change": "same_slot_shape_change",
    "daily_adjacent_shape_change": "adjacent_shape_change",
    "daily_curve_mode_switch_rate": "curve_mode_switch_rate",
    "daily_flat_curve_switch_rate": "flat_curve_switch_rate",
}


LT_SCALAR_FEATURES = [
    "lt_bid_level",
    "lt_bid_level_scale",
    "lt_self_adjustment_bias",
    "lt_self_adjustment_magnitude",
    "lt_self_adjustment_p90",
    "lt_strategy_persistence",

    "lt_quantity_hhi",
    "lt_quantity_hhi_scale",

    "lt_effective_segment_count",
    "lt_flat_curve_rate",
    "lt_tail_uplift_ratio",
    "lt_curve_bend_ratio",
    "lt_shape_defined_rate",
    "lt_shape_day_deviation_median",
    "lt_shape_day_deviation_p90",

    "lt_adjacent_level_change_magnitude",
    "lt_same_slot_shape_change",
    "lt_adjacent_shape_change",
    "lt_adjacent_shape_change_p90",
    "lt_curve_mode_switch_rate",
    "lt_flat_curve_switch_rate",
]


def robust_scale(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    med = s.median()
    return float(ROBUST_SCALE * (s - med).abs().median())


def safe_spearman(a, b, min_n=20):
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if int(ok.sum()) < min_n:
        return np.nan, int(ok.sum())
    xv = x[ok]
    yv = y[ok]
    if xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan, int(ok.sum())
    return float(xv.corr(yv, method="spearman")), int(ok.sum())


def split_half_stability(daily, min_valid_days):
    """
    Compare participant rankings in Jan-Jun vs Jul-Dec.

    A stable long-term trait should preserve participant ordering reasonably
    well across the two halves, even if the absolute market level changes.
    """
    d = daily.copy()
    d["local_date"] = pd.to_datetime(d["local_date"], errors="coerce")
    d["half"] = np.where(d["local_date"].dt.month <= 6, "H1", "H2")

    keys = ["participant_id", "market_product"]
    rows = []

    for feature in DAILY_TO_STEM:
        tmp = d[keys + ["half", feature]].copy()
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")

        agg = (
            tmp.groupby(keys + ["half"], sort=False)[feature]
            .agg(["median", "count"])
            .reset_index()
        )

        h1 = agg[
            (agg["half"] == "H1")
            & (agg["count"] >= min_valid_days)
        ][keys + ["median"]].rename(columns={"median": "h1"})

        h2 = agg[
            (agg["half"] == "H2")
            & (agg["count"] >= min_valid_days)
        ][keys + ["median"]].rename(columns={"median": "h2"})

        m = h1.merge(h2, on=keys, how="inner")

        rho, n = safe_spearman(m["h1"], m["h2"], min_n=20)

        overall_scale = robust_scale(
            pd.concat([m["h1"], m["h2"]], ignore_index=True)
        )
        median_abs_half_shift = (
            (m["h2"] - m["h1"]).abs().median()
            if len(m) else np.nan
        )
        normalized_half_shift = (
            median_abs_half_shift / overall_scale
            if np.isfinite(overall_scale) and overall_scale > EPS
            else np.nan
        )

        rows.append({
            "feature": feature,
            "participants_compared": n,
            "spearman_h1_h2": rho,
            "median_abs_half_shift": median_abs_half_shift,
            "cross_participant_robust_scale": overall_scale,
            "normalized_half_shift": normalized_half_shift,
        })

    return pd.DataFrame(rows)


def between_within_separation(daily, min_valid_days):
    """
    Robust separation ratio:
        between-participant robust scale of participant medians
        -------------------------------------------------------
        median within-participant daily robust scale

    >1 means cross-participant heterogeneity is larger than the typical
    day-to-day variation within one participant. This is descriptive, not a
    hard pass/fail threshold.
    """
    keys = ["participant_id", "market_product"]
    rows = []

    for feature in DAILY_TO_STEM:
        tmp = daily[keys + [feature]].copy()
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")

        per_participant = []
        for key, g in tmp.groupby(keys, sort=False):
            x = g[feature].dropna()
            if len(x) < min_valid_days:
                continue
            per_participant.append({
                "participant_id": key[0],
                "market_product": key[1],
                "median": float(x.median()),
                "within_scale": robust_scale(x),
                "n_days": len(x),
            })

        p = pd.DataFrame(per_participant)

        if len(p):
            between_scale = robust_scale(p["median"])
            typical_within = pd.to_numeric(
                p["within_scale"], errors="coerce"
            ).median()

            ratio = (
                between_scale / typical_within
                if np.isfinite(between_scale)
                and np.isfinite(typical_within)
                and typical_within > EPS
                else np.nan
            )
        else:
            between_scale = np.nan
            typical_within = np.nan
            ratio = np.nan

        rows.append({
            "feature": feature,
            "participants_used": len(p),
            "between_participant_robust_scale": between_scale,
            "median_within_participant_robust_scale": typical_within,
            "between_within_separation_ratio": ratio,
        })

    return pd.DataFrame(rows)


def correlation_validation(profile):
    cols = [
        c for c in LT_SCALAR_FEATURES
        if c in profile.columns
        and pd.to_numeric(profile[c], errors="coerce").notna().sum() >= 20
    ]

    x = profile[cols].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman", min_periods=20)

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if np.isfinite(val):
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "spearman": float(val),
                    "abs_spearman": float(abs(val)),
                })

    pair_df = pd.DataFrame(pairs)
    if len(pair_df):
        pair_df = pair_df.sort_values(
            "abs_spearman", ascending=False
        ).reset_index(drop=True)

    high = pair_df[
        pair_df["abs_spearman"] >= 0.90
    ].copy() if len(pair_df) else pair_df.copy()

    return corr, pair_df, high


def causal_short_term_signal(daily, states):
    """
    Does the recent 7-day strategy contain useful information about today's
    strategy compared with the older 60-day baseline?

    No model is fitted. For state date d:
      target = actual daily strategy feature on d
      long predictor = prior long-window median
      recent predictor = prior recent-window median

    If recent MAE < long MAE, the short-term state carries incremental temporal
    information beyond the older baseline.
    """
    d = daily.copy()
    d["local_date"] = pd.to_datetime(
        d["local_date"], errors="coerce"
    ).dt.date

    s = states.copy()
    s["state_date"] = pd.to_datetime(
        s["state_date"], errors="coerce"
    ).dt.date

    keys_left = ["participant_id", "market_product", "state_date"]
    keys_right = ["participant_id", "market_product", "local_date"]

    rows = []

    for target_col, stem in DAILY_TO_STEM.items():
        need_state = [
            "participant_id",
            "market_product",
            "state_date",
            f"{stem}_recent",
            f"{stem}_baseline",
            f"st_{stem}_raw_shift",
        ]

        if any(c not in s.columns for c in need_state):
            continue

        target = d[
            ["participant_id", "market_product", "local_date", target_col]
        ].copy()

        m = s[need_state].merge(
            target,
            left_on=keys_left,
            right_on=keys_right,
            how="inner",
        )

        y = pd.to_numeric(m[target_col], errors="coerce")
        recent = pd.to_numeric(m[f"{stem}_recent"], errors="coerce")
        baseline = pd.to_numeric(
            m[f"{stem}_baseline"], errors="coerce"
        )
        raw_shift = pd.to_numeric(
            m[f"st_{stem}_raw_shift"], errors="coerce"
        )

        ok = y.notna() & recent.notna() & baseline.notna()
        m = m.loc[ok].copy()
        y = y[ok]
        recent = recent[ok]
        baseline = baseline[ok]
        raw_shift = raw_shift[ok]

        if len(m) == 0:
            continue

        e_long = (y - baseline).abs()
        e_recent = (y - recent).abs()

        long_mae = float(e_long.mean())
        recent_mae = float(e_recent.mean())

        improvement = (
            100.0 * (long_mae - recent_mae) / long_mae
            if long_mae > EPS else np.nan
        )
        win_rate = float((e_recent < e_long).mean())
        tie_rate = float(np.isclose(e_recent, e_long).mean())

        current_deviation = y - baseline
        rho, n_corr = safe_spearman(
            raw_shift,
            current_deviation,
            min_n=20,
        )

        rows.append({
            "feature": target_col,
            "samples": len(m),
            "long_baseline_mae": long_mae,
            "recent_window_mae": recent_mae,
            "recent_vs_long_mae_improvement_pct": improvement,
            "recent_win_rate": win_rate,
            "tie_rate": tie_rate,
            "spearman_recent_shift_vs_current_deviation": rho,
            "correlation_samples": n_corr,
        })

    return pd.DataFrame(rows)


def short_term_distribution(states):
    rows = []

    z_cols = [
        c for c in states.columns
        if c.startswith("st_") and c.endswith("_z")
    ]

    for c in z_cols:
        x = pd.to_numeric(states[c], errors="coerce").dropna()
        if len(x) == 0:
            continue

        rows.append({
            "feature": c,
            "nonmissing": len(x),
            "median": x.median(),
            "p01": x.quantile(0.01),
            "p10": x.quantile(0.10),
            "p90": x.quantile(0.90),
            "p99": x.quantile(0.99),
            "abs_ge_1_rate": (x.abs() >= 1).mean(),
            "abs_ge_2_rate": (x.abs() >= 2).mean(),
            "zero_rate": np.isclose(x, 0.0).mean(),
        })

    return pd.DataFrame(rows)


def representative_participants(profile):
    """
    Select interpretable examples for manual inspection.
    These are NOT clusters or permanent participant labels.
    """
    p = profile.copy()
    p = p[
        (pd.to_numeric(p["lt_active_days"], errors="coerce") >= 300)
        & (pd.to_numeric(p["lt_valid_bid_rate"], errors="coerce") >= 0.90)
    ].copy()

    examples = []

    def add_example(label, sort_col, ascending):
        q = p[p[sort_col].notna()].sort_values(
            sort_col, ascending=ascending
        )
        if len(q):
            r = q.iloc[0]
            examples.append({
                "case": label,
                "participant_id": r["participant_id"],
                "market_product": r["market_product"],
                "selection_feature": sort_col,
                "selection_value": r[sort_col],
            })

    add_example(
        "stable_price_behavior",
        "lt_self_adjustment_magnitude",
        True,
    )
    add_example(
        "high_price_adjustment",
        "lt_self_adjustment_magnitude",
        False,
    )
    add_example(
        "high_shape_adjustment",
        "lt_adjacent_shape_change",
        False,
    )
    add_example(
        "high_curve_switching",
        "lt_curve_mode_switch_rate",
        False,
    )

    return pd.DataFrame(examples)


def save_figures(
    validation_dir,
    stability,
    separation,
    causal,
    zstats,
    representatives,
    daily,
):
    fig_dir = validation_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Split-half stability
    x = stability.dropna(subset=["spearman_h1_h2"]).sort_values(
        "spearman_h1_h2"
    )
    if len(x):
        plt.figure(figsize=(9, 7))
        plt.barh(x["feature"], x["spearman_h1_h2"])
        plt.xlabel("Spearman correlation: H1 vs H2 participant ranking")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(fig_dir / "01_split_half_stability.png", dpi=180)
        plt.close()

    # 2) Between/within separation
    x = separation.dropna(
        subset=["between_within_separation_ratio"]
    ).sort_values("between_within_separation_ratio")
    if len(x):
        plt.figure(figsize=(9, 7))
        plt.barh(
            x["feature"],
            x["between_within_separation_ratio"],
        )
        plt.xlabel("Robust between-participant / within-participant scale")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(
            fig_dir / "02_between_within_separation.png",
            dpi=180,
        )
        plt.close()

    # 3) Short-term causal improvement
    x = causal.dropna(
        subset=["recent_vs_long_mae_improvement_pct"]
    ).sort_values("recent_vs_long_mae_improvement_pct")
    if len(x):
        plt.figure(figsize=(9, 7))
        plt.barh(
            x["feature"],
            x["recent_vs_long_mae_improvement_pct"],
        )
        plt.xlabel("Recent 7-day MAE improvement over older baseline (%)")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(
            fig_dir / "03_short_term_incremental_signal.png",
            dpi=180,
        )
        plt.close()

    # 4) Short-term z central ranges
    if len(zstats):
        x = zstats.sort_values("median")
        pos = np.arange(len(x))
        lower = x["median"] - x["p10"]
        upper = x["p90"] - x["median"]

        plt.figure(figsize=(9, 7))
        plt.errorbar(
            x["median"],
            pos,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=3,
        )
        plt.yticks(pos, x["feature"])
        plt.xlabel("Short-term robust z: median and P10-P90")
        plt.tight_layout()
        plt.savefig(
            fig_dir / "04_short_term_z_distribution.png",
            dpi=180,
        )
        plt.close()

    # 5+) Representative participant time series
    if len(representatives):
        d = daily.copy()
        d["local_date"] = pd.to_datetime(
            d["local_date"], errors="coerce"
        )

        for _, r in representatives.iterrows():
            g = d[
                (d["participant_id"] == r["participant_id"])
                & (d["market_product"] == r["market_product"])
            ].sort_values("local_date")

            if len(g) == 0:
                continue

            plt.figure(figsize=(10, 4))
            plt.plot(
                g["local_date"],
                pd.to_numeric(
                    g["daily_bid_level"], errors="coerce"
                ),
            )
            plt.xlabel("Date")
            plt.ylabel("Daily bid level")
            plt.title(
                f"{r['case']} | participant={r['participant_id']}"
            )
            plt.tight_layout()

            safe_case = str(r["case"]).replace(" ", "_")
            plt.savefig(
                fig_dir / f"case_{safe_case}.png",
                dpi=180,
            )
            plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--daily-root",
        default="data/processed/daily_strategy_summary",
    )
    p.add_argument(
        "--profile-root",
        default="data/processed/long_term_strategy_profile",
    )
    p.add_argument(
        "--state-root",
        default="data/processed/short_term_strategy_state",
    )
    p.add_argument(
        "--results-root",
        default="results/profile_stage_validation",
    )
    p.add_argument(
        "--min-half-valid-days",
        type=int,
        default=20,
    )
    p.add_argument(
        "--min-profile-valid-days",
        type=int,
        default=30,
    )
    args = p.parse_args()

    daily_file = (
        Path(args.daily_root)
        / str(args.year)
        / f"daily_strategy_summary_{args.year}.csv"
    )
    profile_file = (
        Path(args.profile_root)
        / str(args.year)
        / f"long_term_strategy_profile_{args.year}.csv"
    )
    state_file = (
        Path(args.state_root)
        / str(args.year)
        / f"short_term_strategy_state_{args.year}.csv"
    )

    for f in [daily_file, profile_file, state_file]:
        if not f.exists():
            raise FileNotFoundError(f)

    out_dir = Path(args.results_root) / str(args.year)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[read] {daily_file}")
    daily = pd.read_csv(daily_file, low_memory=False)

    print(f"[read] {profile_file}")
    profile = pd.read_csv(profile_file, low_memory=False)

    print(f"[read] {state_file}")
    states = pd.read_csv(state_file, low_memory=False)

    # 1. Temporal stability
    print("[validate] split-half stability")
    stability = split_half_stability(
        daily,
        args.min_half_valid_days,
    )
    stability.to_csv(
        out_dir / f"split_half_stability_{args.year}.csv",
        index=False,
    )

    # 2. Between vs within participant separation
    print("[validate] between/within separation")
    separation = between_within_separation(
        daily,
        args.min_profile_valid_days,
    )
    separation.to_csv(
        out_dir / f"between_within_separation_{args.year}.csv",
        index=False,
    )

    # 3. Redundancy
    print("[validate] long-term feature correlation")
    corr, pair_df, high_corr = correlation_validation(profile)
    corr.to_csv(
        out_dir / f"long_term_feature_correlation_{args.year}.csv"
    )
    high_corr.to_csv(
        out_dir / f"highly_correlated_feature_pairs_{args.year}.csv",
        index=False,
    )
    pair_df.head(30).to_csv(
        out_dir / f"top_correlated_feature_pairs_{args.year}.csv",
        index=False,
    )

    # 4. Causal short-term incremental signal
    print("[validate] causal short-term signal")
    causal = causal_short_term_signal(daily, states)
    causal.to_csv(
        out_dir / f"causal_short_term_signal_{args.year}.csv",
        index=False,
    )

    # 5. Short-term state distribution
    print("[validate] short-term state distribution")
    zstats = short_term_distribution(states)
    zstats.to_csv(
        out_dir / f"short_term_z_distribution_{args.year}.csv",
        index=False,
    )

    # 6. Representative cases
    reps = representative_participants(profile)
    reps.to_csv(
        out_dir / f"representative_participants_{args.year}.csv",
        index=False,
    )

    save_figures(
        out_dir,
        stability,
        separation,
        causal,
        zstats,
        reps,
        daily,
    )

    # Compact summary statistics.
    stable_rhos = pd.to_numeric(
        stability["spearman_h1_h2"], errors="coerce"
    ).dropna()

    sep_ratios = pd.to_numeric(
        separation["between_within_separation_ratio"],
        errors="coerce",
    ).dropna()

    improvements = pd.to_numeric(
        causal["recent_vs_long_mae_improvement_pct"],
        errors="coerce",
    ).dropna()

    high_corr_count = len(high_corr)

    lines = [
        f"PJM strategy-profile stage validation - {args.year}",
        f"Version: {VERSION}",
        "=" * 72,
        f"Participants: {profile['participant_id'].nunique():,}",
        f"Participant-days: {len(daily):,}",
        f"Short-term state rows: {len(states):,}",
        "",
        "1. Split-half temporal stability",
        f"Validated features: {len(stable_rhos)}",
        f"Median H1-H2 Spearman: "
        f"{stable_rhos.median():.4f}" if len(stable_rhos) else
        "Median H1-H2 Spearman: NA",
        f"Features with positive H1-H2 Spearman: "
        f"{int((stable_rhos > 0).sum())}/{len(stable_rhos)}"
        if len(stable_rhos) else
        "Features with positive H1-H2 Spearman: NA",
        "",
        "2. Between-participant vs within-participant separation",
        f"Validated features: {len(sep_ratios)}",
        f"Median robust separation ratio: "
        f"{sep_ratios.median():.4f}" if len(sep_ratios) else
        "Median robust separation ratio: NA",
        f"Features with separation ratio > 1: "
        f"{int((sep_ratios > 1).sum())}/{len(sep_ratios)}"
        if len(sep_ratios) else
        "Features with separation ratio > 1: NA",
        "",
        "3. Long-term feature redundancy",
        f"Feature pairs with |Spearman| >= 0.90: {high_corr_count}",
        "",
        "4. Causal short-term incremental signal",
        f"Validated target features: {len(improvements)}",
        f"Median recent-window MAE improvement over older baseline: "
        f"{improvements.median():.2f}%"
        if len(improvements) else
        "Median recent-window MAE improvement over older baseline: NA",
        f"Features with positive MAE improvement: "
        f"{int((improvements > 0).sum())}/{len(improvements)}"
        if len(improvements) else
        "Features with positive MAE improvement: NA",
        "",
        "Interpretation rule:",
        "  This report does not use one arbitrary pass/fail score.",
        "  A credible profile should show participant-level temporal stability,",
        "  meaningful between-participant heterogeneity, limited redundancy,",
        "  and incremental causal information in the short-term state.",
        "",
        f"Results directory: {out_dir.resolve()}",
    ]

    summary_file = out_dir / f"summary_{args.year}.txt"
    summary_file.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
