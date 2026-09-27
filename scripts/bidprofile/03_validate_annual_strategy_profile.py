#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from annual_profile_core import (
    ANNUAL_FEATURES,
    FEATURE_CN,
    build_group_profile,
)

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def spearman(a, b):
    z = pd.concat([
        pd.to_numeric(a, errors="coerce").rename("a"),
        pd.to_numeric(b, errors="coerce").rename("b"),
    ], axis=1).dropna()
    if len(z) < 10:
        return np.nan, len(z)
    return float(z["a"].corr(z["b"], method="spearman")), len(z)

def rebuild_profiles(df, label):
    rows = []
    for pid, g in df.groupby("participant_id", sort=False):
        r = build_group_profile(g, min_persistence_pairs=5, min_shape_days=2)
        r["participant_id"] = pid
        r["period"] = label
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--daily-root", default="data/processed/final_clean/daily")
    p.add_argument("--annual-root", default="data/processed/bidprofile_annual")
    p.add_argument("--out-root", default="results/bidprofile_annual_validation")
    args = p.parse_args()

    daily_file = Path(args.daily_root) / str(args.year) / f"daily_strategy_core_{args.year}.csv"
    annual_file = Path(args.annual_root) / str(args.year) / f"annual_strategy_profile_{args.year}.csv"
    monthly_file = Path(args.annual_root) / str(args.year) / f"annual_strategy_profile_monthly_{args.year}.csv"

    for f in [daily_file, annual_file, monthly_file]:
        if not f.exists():
            raise FileNotFoundError(f)

    out_dir = ensure_dir(Path(args.out_root) / str(args.year))

    daily = pd.read_csv(daily_file, low_memory=False)
    annual = pd.read_csv(annual_file, low_memory=False)
    monthly = pd.read_csv(monthly_file, low_memory=False)
    daily["local_date"] = pd.to_datetime(daily["local_date"]).dt.normalize()

    # 1) Coverage
    coverage_rows = []
    n_all = len(annual)
    for f in ANNUAL_FEATURES:
        n = int(pd.to_numeric(annual[f], errors="coerce").notna().sum())
        coverage_rows.append({
            "feature": f,
            "name_cn": FEATURE_CN[f],
            "participants": n,
            "coverage_share": n / n_all if n_all else np.nan,
            "missing_share": 1 - n / n_all if n_all else np.nan,
        })
    coverage = pd.DataFrame(coverage_rows)

    # 1b) Population dispersion audit.
    # This catches annual aggregation that collapses most participants to the
    # same value even when rank stability looks acceptable.
    dispersion_rows = []
    for f in ANNUAL_FEATURES:
        x = pd.to_numeric(annual[f], errors="coerce").dropna()
        if len(x):
            q05, q25, q50, q75, q95 = x.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
            iqr = float(q75 - q25)
            central90 = float(q95 - q05)
            unique_rounded = int(x.round(8).nunique())
            top_share = float(x.round(8).value_counts(normalize=True).iloc[0])
            relative_iqr = (
                iqr / central90
                if central90 > 1e-12
                else np.nan
            )
        else:
            q05 = q25 = q50 = q75 = q95 = np.nan
            iqr = central90 = relative_iqr = np.nan
            unique_rounded = 0
            top_share = np.nan

        dispersion_rows.append({
            "feature": f,
            "name_cn": FEATURE_CN[f],
            "n": len(x),
            "p05": q05,
            "p25": q25,
            "median": q50,
            "p75": q75,
            "p95": q95,
            "iqr": iqr,
            "p95_minus_p05": central90,
            "relative_iqr_to_p90_range": relative_iqr,
            "unique_values_rounded_8dp": unique_rounded,
            "largest_exact_value_share_rounded_8dp": top_share,
            "central_iqr_collapse_flag": int(
                np.isfinite(iqr) and iqr <= 1e-10 and central90 > 1e-10
            ),
        })
    dispersion = pd.DataFrame(dispersion_rows)

    # 2) H1-H2 stability for all 9 dimensions
    print("[03 annual] rebuilding H1 profiles ...", flush=True)
    h1 = daily[daily["local_date"].dt.month <= 6].copy()
    p1 = rebuild_profiles(h1, "H1").set_index("participant_id")

    print("[03 annual] rebuilding H2 profiles ...", flush=True)
    h2 = daily[daily["local_date"].dt.month >= 7].copy()
    p2 = rebuild_profiles(h2, "H2").set_index("participant_id")

    stability_rows = []
    for f in ANNUAL_FEATURES:
        idx = p1.index.intersection(p2.index)
        rho, n = spearman(p1.loc[idx, f], p2.loc[idx, f])
        stability_rows.append({
            "feature": f,
            "name_cn": FEATURE_CN[f],
            "participants_compared": n,
            "spearman_h1_h2": rho,
        })
    stability = pd.DataFrame(stability_rows)

    # 3) Monthly vs annual rank stability
    annual_idx = annual.set_index("participant_id")
    monthly_rank_rows = []
    for month, gm in monthly.groupby("profile_month", sort=True):
        gm = gm.set_index("participant_id")
        idx = gm.index.intersection(annual_idx.index)
        for f in ANNUAL_FEATURES:
            rho, n = spearman(gm.loc[idx, f], annual_idx.loc[idx, f])
            monthly_rank_rows.append({
                "profile_month": month,
                "feature": f,
                "name_cn": FEATURE_CN[f],
                "participants_compared": n,
                "spearman_month_vs_annual": rho,
            })
    monthly_rank = pd.DataFrame(monthly_rank_rows)

    month_summary = (
        monthly_rank.groupby(["feature", "name_cn"], as_index=False)
        .agg(
            months_available=("spearman_month_vs_annual", "count"),
            median_month_vs_annual_spearman=("spearman_month_vs_annual", "median"),
            min_month_vs_annual_spearman=("spearman_month_vs_annual", "min"),
            max_month_vs_annual_spearman=("spearman_month_vs_annual", "max"),
        )
    )

    # 4) Redundancy
    redundancy_rows = []
    for a, b in combinations(ANNUAL_FEATURES, 2):
        z = annual[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        rho = z[a].corr(z[b], method="spearman") if len(z) >= 10 else np.nan
        redundancy_rows.append({
            "feature_1": a,
            "feature_2": b,
            "n": len(z),
            "spearman": rho,
            "abs_spearman": abs(rho) if pd.notna(rho) else np.nan,
            "high_redundancy_ge_0_90_flag": int(pd.notna(rho) and abs(rho) >= 0.90),
        })
    redundancy = pd.DataFrame(redundancy_rows)

    # 5) Within-participant monthly deviation normalized by annual population IQR
    seasonality_rows = []
    for f in ANNUAL_FEATURES:
        av = pd.to_numeric(annual[f], errors="coerce").dropna()
        if len(av) >= 10:
            q25, q75 = av.quantile([0.25, 0.75])
            pop_iqr = float(q75 - q25)
        else:
            pop_iqr = np.nan

        per_pid = []
        for pid, gm in monthly.groupby("participant_id", sort=False):
            if pid not in annual_idx.index:
                continue
            a = pd.to_numeric(pd.Series([annual_idx.loc[pid, f]]), errors="coerce").iloc[0]
            m = pd.to_numeric(gm[f], errors="coerce").dropna()
            if not np.isfinite(a) or len(m) < 2:
                continue
            per_pid.append(float(np.median(np.abs(m.to_numpy(float) - a))))

        med_dev = float(np.median(per_pid)) if per_pid else np.nan
        norm_dev = (
            med_dev / pop_iqr
            if np.isfinite(pop_iqr) and pop_iqr > 1e-12
            else np.nan
        )
        seasonality_rows.append({
            "feature": f,
            "name_cn": FEATURE_CN[f],
            "participants_with_2plus_months": len(per_pid),
            "population_annual_iqr": pop_iqr,
            "median_participant_month_abs_deviation": med_dev,
            "normalized_month_deviation_over_population_iqr": norm_dev,
        })
    seasonality = pd.DataFrame(seasonality_rows)

    review = (
        coverage
        .merge(dispersion, on=["feature", "name_cn"], how="left")
        .merge(stability, on=["feature", "name_cn"], how="left")
        .merge(month_summary, on=["feature", "name_cn"], how="left")
        .merge(seasonality, on=["feature", "name_cn"], how="left")
    )

    # Audit flags only; no automatic deletion.
    review["coverage_lt_70pct_flag"] = (review["coverage_share"] < 0.70).astype(int)
    review["h1h2_lt_0_70_flag"] = (
        (review["spearman_h1_h2"] < 0.70) | review["spearman_h1_h2"].isna()
    ).astype(int)
    review["monthly_rank_lt_0_70_flag"] = (
        (review["median_month_vs_annual_spearman"] < 0.70)
        | review["median_month_vs_annual_spearman"].isna()
    ).astype(int)
    review["seasonality_high_flag"] = (
        review["normalized_month_deviation_over_population_iqr"] > 0.50
    ).fillna(False).astype(int)
    review["central_iqr_collapse_flag"] = (
        review["central_iqr_collapse_flag"].fillna(0).astype(int)
    )

    high_red = redundancy[redundancy["high_redundancy_ge_0_90_flag"].eq(1)].copy()

    coverage.to_csv(out_dir / "annual_feature_coverage.csv", index=False, encoding="utf-8-sig")
    dispersion.to_csv(out_dir / "annual_feature_dispersion.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(out_dir / "annual_split_half_stability.csv", index=False, encoding="utf-8-sig")
    monthly_rank.to_csv(out_dir / "annual_monthly_rank_stability.csv", index=False, encoding="utf-8-sig")
    month_summary.to_csv(out_dir / "annual_monthly_rank_stability_summary.csv", index=False, encoding="utf-8-sig")
    redundancy.to_csv(out_dir / "annual_feature_redundancy.csv", index=False, encoding="utf-8-sig")
    seasonality.to_csv(out_dir / "annual_feature_seasonality.csv", index=False, encoding="utf-8-sig")
    review.to_csv(out_dir / "annual_feature_review.csv", index=False, encoding="utf-8-sig")

    lines = [
        f"Annual static strategy profile validation - {args.year}",
        "=" * 78,
        "",
        f"Participants: {len(annual):,}",
        f"Ready: {int(annual['annual_profile_ready_flag'].sum()):,} "
        f"({annual['annual_profile_ready_flag'].mean()*100:.2f}%)",
        f"Complete 9D: {int(annual['annual_complete_9d_flag'].sum()):,} "
        f"({annual['annual_complete_9d_flag'].mean()*100:.2f}%)",
        f"Median H1-H2 Spearman: {stability['spearman_h1_h2'].median():.4f}",
        f"Median month-vs-annual Spearman: "
        f"{month_summary['median_month_vs_annual_spearman'].median():.4f}",
        f"High-redundancy pairs |rho|>=0.90: {len(high_red)}",
        "",
        "Feature review:",
    ]

    for _, r in review.iterrows():
        flags = []
        if int(r["coverage_lt_70pct_flag"]):
            flags.append("low coverage")
        if int(r["h1h2_lt_0_70_flag"]):
            flags.append("weak H1-H2")
        if int(r["monthly_rank_lt_0_70_flag"]):
            flags.append("season-sensitive rank")
        if int(r["seasonality_high_flag"]):
            flags.append("large monthly shift")
        if int(r["central_iqr_collapse_flag"]):
            flags.append("central IQR collapse")
        flag_text = ", ".join(flags) if flags else "no automatic warning"

        lines.append(
            f"  {r['feature']}: coverage={r['coverage_share']*100:.2f}%, "
            f"H1H2={r['spearman_h1_h2']:.4f}, "
            f"month-rank={r['median_month_vs_annual_spearman']:.4f}, "
            f"norm-month-dev={r['normalized_month_deviation_over_population_iqr']:.4f}, "
            f"IQR={r['iqr']:.6g}; "
            f"{flag_text}"
        )

    if len(high_red):
        lines += ["", "High-redundancy pairs:"]
        for _, r in high_red.sort_values("abs_spearman", ascending=False).iterrows():
            lines.append(
                f"  {r['feature_1']} <-> {r['feature_2']}: "
                f"rho={r['spearman']:.4f}, n={int(r['n'])}"
            )

    lines += [
        "",
        "Important:",
        "  Flags are audit aids only; they do not automatically delete a feature.",
        "  A 2025 full-year profile used on 2025 is retrospective/explanatory only.",
    ]

    summary = out_dir / "annual_profile_validation_summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"Done: {out_dir}")

if __name__ == "__main__":
    main()
