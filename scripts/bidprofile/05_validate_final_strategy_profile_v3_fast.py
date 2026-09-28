#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_validate_final_strategy_profile.py

Final validation of the independent 9-D strategy profile.

Core principles
---------------
1. The profile itself is the 9 independent LT dimensions.
2. KMeans is only auxiliary discovery.
3. Every clustering space is evaluated on COMPLETE CASES of that space.
4. No missing-value imputation is allowed for cluster discovery/validation.
5. K=2..6 are compared side-by-side.
6. Small clusters are flagged, not automatically rejected, because a small
   cluster may represent a real minority strategy.

Outputs
-------
A. LT split-half stability
B. LT feature redundancy
C. ST incremental signal
D. Independent-state coverage
E. Candidate-K validation for every clustering space
F. Per-feature Kruskal epsilon^2 for every candidate K
G. Final selected-K review summary
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "5")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "5")

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
)
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
)

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import RobustScaler

from strategy_core import (
    LT_FEATURES,
    PA_FEATURES,
    QS_FEATURES,
    SHAPE_FEATURES,
    DAILY_SCALARS,
    ensure_dir,
)

SPACES = {
    "full_9d": LT_FEATURES,
    "price_adjustment_3d": PA_FEATURES,
    "quantity_structure_3d": QS_FEATURES,
    "curve_shape_3d": SHAPE_FEATURES,
}

K_CANDIDATES = range(2, 7)

LT_DAILY_MAP = {
    "lt_bid_level": "daily_bid_level",
    "lt_adjustment_magnitude": "daily_adjustment_magnitude",
    "lt_quantity_hhi": "daily_quantity_hhi",
    "lt_effective_segment_count": "daily_effective_segment_count",
    "lt_flat_curve_rate": "daily_flat_curve_rate",
    "lt_tail_uplift_ratio": "daily_tail_uplift_ratio",
    "lt_curve_bend_ratio": "daily_curve_bend_ratio",
}


def split_half(daily):
    daily = daily.copy()
    daily["local_date"] = pd.to_datetime(daily["local_date"])

    rows = []
    for lt, col in LT_DAILY_MAP.items():
        h1 = (
            daily[daily["local_date"].dt.month <= 6]
            .groupby("participant_id")[col]
            .median()
        )
        h2 = (
            daily[daily["local_date"].dt.month >= 7]
            .groupby("participant_id")[col]
            .median()
        )

        z = pd.concat([h1, h2], axis=1, keys=["h1", "h2"]).dropna()

        rho = (
            z["h1"].corr(z["h2"], method="spearman")
            if len(z) >= 10
            else np.nan
        )

        rows.append(
            {
                "feature": lt,
                "participants": len(z),
                "spearman_h1_h2": rho,
            }
        )

    return pd.DataFrame(rows)


def feature_redundancy(profile):
    rows = []

    for a, b in combinations(LT_FEATURES, 2):
        z = (
            profile[[a, b]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )

        rho = (
            z[a].corr(z[b], method="spearman")
            if len(z) >= 10
            else np.nan
        )

        rows.append(
            {
                "feature_1": a,
                "feature_2": b,
                "n": len(z),
                "spearman": rho,
                "abs_spearman": (
                    abs(rho) if pd.notna(rho) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def st_increment(daily):
    """
    Vectorized calendar-window validation.

    recent = d-7 ... d-1
    long   = d-67 ... d-8

    This is the same window definition as the ST state builder, but avoids
    rescanning the participant history for every participant-day.
    """
    daily = daily.copy()
    daily["local_date"] = pd.to_datetime(
        daily["local_date"]
    ).dt.normalize()

    err_long = {
        stem: []
        for stem in DAILY_SCALARS
    }
    err_recent = {
        stem: []
        for stem in DAILY_SCALARS
    }

    groups = daily.groupby(
        "participant_id",
        sort=False,
    )

    total = daily["participant_id"].nunique()
    done = 0

    for _, g in groups:
        done += 1

        g = (
            g.sort_values("local_date")
            .drop_duplicates(
                subset=["local_date"],
                keep="last",
            )
        )

        if g.empty:
            continue

        original_dates = pd.Index(
            g["local_date"]
        )

        full_dates = pd.date_range(
            original_dates.min(),
            original_dates.max(),
            freq="D",
        )

        cal = (
            g.set_index("local_date")
            .reindex(full_dates)
        )

        active = cal.index.isin(
            original_dates
        )

        for stem, col in DAILY_SCALARS.items():
            s = pd.to_numeric(
                cal[col],
                errors="coerce",
            )

            recent_source = s.shift(1)
            long_source = s.shift(8)

            recent_count = (
                recent_source
                .rolling(
                    7,
                    min_periods=1,
                )
                .count()
            )
            long_count = (
                long_source
                .rolling(
                    60,
                    min_periods=1,
                )
                .count()
            )

            recent_med = (
                recent_source
                .rolling(
                    7,
                    min_periods=1,
                )
                .median()
            )
            long_med = (
                long_source
                .rolling(
                    60,
                    min_periods=1,
                )
                .median()
            )

            valid = (
                active
                & s.notna().to_numpy()
                & (recent_count.to_numpy() >= 3)
                & (long_count.to_numpy() >= 20)
            )

            if not valid.any():
                continue

            y = s.to_numpy()[valid]
            r = recent_med.to_numpy()[valid]
            l = long_med.to_numpy()[valid]

            err_recent[stem].append(
                np.abs(y - r)
            )
            err_long[stem].append(
                np.abs(y - l)
            )

        if (
            done == 1
            or done % 100 == 0
            or done == total
        ):
            print(
                f"[05] ST validation participants "
                f"{done:,}/{total:,}",
                flush=True,
            )

    rows = []

    for stem in DAILY_SCALARS:
        er = (
            np.concatenate(
                err_recent[stem]
            )
            if err_recent[stem]
            else np.array([])
        )
        el = (
            np.concatenate(
                err_long[stem]
            )
            if err_long[stem]
            else np.array([])
        )

        if len(el):
            ml = float(
                np.mean(el)
            )
            mr = float(
                np.mean(er)
            )
            imp = (
                (ml - mr)
                / ml
                * 100.0
                if ml > 0
                else np.nan
            )
        else:
            ml = mr = imp = np.nan

        rows.append(
            {
                "feature": stem,
                "n": len(el),
                "long_mae": ml,
                "recent_mae": mr,
                "improvement_pct": imp,
            }
        )

    return pd.DataFrame(rows)


def state_coverage(profile):
    rows = []

    for f in LT_FEATURES:
        col = f"{f}_state"

        if col not in profile.columns:
            continue

        s = profile[col]
        valid = int(s.notna().sum())

        for state in ["低", "中", "高"]:
            n = int((s == state).sum())
            rows.append(
                {
                    "feature": f,
                    "state": state,
                    "participant_count": n,
                    "share_of_valid": (
                        n / valid if valid else np.nan
                    ),
                    "share_of_all": (
                        n / len(profile) if len(profile) else np.nan
                    ),
                }
            )

        missing = int(s.isna().sum())
        rows.append(
            {
                "feature": f,
                "state": "缺失",
                "participant_count": missing,
                "share_of_valid": np.nan,
                "share_of_all": (
                    missing / len(profile)
                    if len(profile) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def complete_case_scaled(profile, features):
    """
    Complete-case only. No imputation.

    Winsorization and RobustScaler match the clustering pipeline.
    """
    raw = (
        profile[features]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .copy()
    )

    clipped = raw.copy()

    for f in features:
        lo = clipped[f].quantile(0.01)
        hi = clipped[f].quantile(0.99)
        clipped[f] = clipped[f].clip(lo, hi)

    x = RobustScaler(
        quantile_range=(25, 75)
    ).fit_transform(clipped)

    return raw, x


def candidate_cluster_validation(profile):
    summary_rows = []
    feature_rows = []

    for space_name, features in SPACES.items():
        print()
        print(
            f"[05] cluster review: {space_name}",
            flush=True,
        )

        raw, x = complete_case_scaled(profile, features)
        n = len(raw)

        print(
            f"     complete cases = {n:,}",
            flush=True,
        )

        for k in K_CANDIDATES:
            if n <= k:
                continue

            print(
                f"     evaluating K={k} ...",
                flush=True,
            )

            model = KMeans(
                n_clusters=k,
                n_init=100,
                random_state=42,
            )
            labels = model.fit_predict(x)

            sil = float(
                silhouette_score(x, labels)
            )
            ch = float(
                calinski_harabasz_score(x, labels)
            )
            db = float(
                davies_bouldin_score(x, labels)
            )

            aris = []

            for seed in range(10):
                alt = KMeans(
                    n_clusters=k,
                    n_init=20,
                    random_state=seed,
                ).fit_predict(x)

                aris.append(
                    adjusted_rand_score(
                        labels,
                        alt,
                    )
                )

            med_ari = float(np.median(aris))

            counts = np.bincount(
                labels,
                minlength=k,
            )
            min_count = int(counts.min())
            max_count = int(counts.max())

            min_share = float(
                min_count / n
            )
            max_share = float(
                max_count / n
            )

            # Balance diagnostics only; NOT automatic rejection criteria.
            small_lt_5pct = int(
                min_share < 0.05
            )
            tiny_lt_2pct = int(
                min_share < 0.02
            )

            summary_rows.append(
                {
                    "space": space_name,
                    "complete_n": n,
                    "k": k,
                    "silhouette": sil,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "median_ari": med_ari,
                    "min_cluster_count": min_count,
                    "min_cluster_share": min_share,
                    "max_cluster_count": max_count,
                    "max_cluster_share": max_share,
                    "small_cluster_lt_5pct_flag": small_lt_5pct,
                    "tiny_cluster_lt_2pct_flag": tiny_lt_2pct,
                }
            )

            temp = raw.copy()
            temp["_cluster"] = labels

            for f in features:
                groups = []

                for c in range(k):
                    vals = (
                        pd.to_numeric(
                            temp.loc[
                                temp["_cluster"].eq(c),
                                f,
                            ],
                            errors="coerce",
                        )
                        .dropna()
                        .to_numpy()
                    )

                    if len(vals):
                        groups.append(vals)

                total_n = sum(
                    len(g)
                    for g in groups
                )

                if len(groups) >= 2:
                    H, p = kruskal(*groups)

                    eps2 = max(
                        0.0,
                        (
                            H
                            - len(groups)
                            + 1
                        )
                        / (
                            total_n
                            - len(groups)
                        ),
                    )
                else:
                    H = p = eps2 = np.nan

                feature_rows.append(
                    {
                        "space": space_name,
                        "k": k,
                        "feature": f,
                        "kruskal_H": H,
                        "kruskal_p": p,
                        "epsilon_squared": eps2,
                    }
                )

            print(
                f"       sil={sil:.4f}, "
                f"CH={ch:.1f}, "
                f"DB={db:.4f}, "
                f"ARI={med_ari:.4f}, "
                f"min={min_count} "
                f"({min_share*100:.2f}%)",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    features = pd.DataFrame(feature_rows)

    return summary, features


def build_selected_review(
    candidate_summary,
    discovery,
    feature_sep,
):
    rows = []

    for _, d in discovery.iterrows():
        space = d["space"]

        if not int(d["accepted"]):
            rows.append(
                {
                    "space": space,
                    "selected_k": np.nan,
                    "status": "not_accepted",
                }
            )
            continue

        k = int(d["selected_k"])

        r = candidate_summary[
            candidate_summary["space"].eq(space)
            & candidate_summary["k"].eq(k)
        ]

        if r.empty:
            continue

        r = r.iloc[0]

        f = feature_sep[
            feature_sep["space"].eq(space)
            & feature_sep["k"].eq(k)
        ]

        rows.append(
            {
                "space": space,
                "selected_k": k,
                "status": "accepted",
                "complete_n": int(r["complete_n"]),
                "silhouette": r["silhouette"],
                "calinski_harabasz": r["calinski_harabasz"],
                "davies_bouldin": r["davies_bouldin"],
                "median_ari": r["median_ari"],
                "min_cluster_count": int(r["min_cluster_count"]),
                "min_cluster_share": r["min_cluster_share"],
                "max_cluster_share": r["max_cluster_share"],
                "median_feature_epsilon_squared": (
                    f["epsilon_squared"].median()
                    if len(f) else np.nan
                ),
                "max_feature_epsilon_squared": (
                    f["epsilon_squared"].max()
                    if len(f) else np.nan
                ),
                "small_cluster_lt_5pct_flag": int(
                    r["small_cluster_lt_5pct_flag"]
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--year",
        type=int,
        default=2025,
    )
    p.add_argument(
        "--root",
        default="data/processed/final_clean",
    )
    p.add_argument(
        "--results-root",
        default="results/final_clean_validation",
    )
    args = p.parse_args()

    daily_file = (
        Path(args.root)
        / "daily"
        / str(args.year)
        / f"daily_strategy_core_{args.year}.csv"
    )

    profile_file = (
        Path(args.root)
        / "strategy_profile"
        / str(args.year)
        / f"participant_strategy_profile_{args.year}.csv"
    )

    discovery_file = (
        Path(args.root)
        / "strategy_profile"
        / str(args.year)
        / f"cluster_discovery_summary_{args.year}.csv"
    )

    daily = pd.read_csv(
        daily_file,
        low_memory=False,
    )
    profile = pd.read_csv(
        profile_file,
        low_memory=False,
    )
    discovery = pd.read_csv(
        discovery_file,
        low_memory=False,
    )

    out_dir = ensure_dir(
        Path(args.results_root)
        / str(args.year)
    )

    print("[05] validating LT stability ...", flush=True)
    stability = split_half(daily)

    print("[05] validating LT redundancy ...", flush=True)
    redundancy = feature_redundancy(profile)

    print("[05] validating ST incremental signal ...", flush=True)
    st = st_increment(daily)

    print("[05] validating independent-state coverage ...", flush=True)
    coverage = state_coverage(profile)

    candidate_summary, feature_sep = (
        candidate_cluster_validation(
            profile
        )
    )

    selected_review = build_selected_review(
        candidate_summary,
        discovery,
        feature_sep,
    )

    stability.to_csv(
        out_dir / "lt_split_half_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    redundancy.to_csv(
        out_dir / "lt_feature_redundancy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    st.to_csv(
        out_dir / "st_incremental_signal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage.to_csv(
        out_dir / "independent_state_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_summary.to_csv(
        out_dir / "cluster_candidate_k_review.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_sep.to_csv(
        out_dir / "cluster_candidate_feature_separation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_review.to_csv(
        out_dir / "selected_cluster_review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    high_redundancy = redundancy[
        redundancy["abs_spearman"] >= 0.90
    ]

    lines = [
        f"Final strategy-profile validation - {args.year}",
        "=" * 76,
        "",
        "Profile definition:",
        "  9 independent long-term strategy dimensions.",
        "  KMeans is auxiliary natural-pattern discovery only.",
        "  Cluster validation uses complete cases only; no imputation.",
        "",
        f"Participants: {len(profile):,}",
        (
            "Median LT split-half Spearman: "
            f"{stability['spearman_h1_h2'].median():.4f}"
        ),
        (
            "LT pairs with |Spearman| >= 0.90: "
            f"{len(high_redundancy)}"
        ),
        (
            "Median ST MAE improvement: "
            f"{st['improvement_pct'].median():.2f}%"
        ),
        "",
        "Selected cluster solutions from 04:",
    ]

    for _, r in selected_review.iterrows():
        if r["status"] != "accepted":
            lines.append(
                f"  {r['space']}: not accepted"
            )
            continue

        warning = (
            " [small cluster <5%]"
            if int(r["small_cluster_lt_5pct_flag"])
            else ""
        )

        lines.append(
            f"  {r['space']}: "
            f"K={int(r['selected_k'])}, "
            f"N={int(r['complete_n'])}, "
            f"sil={r['silhouette']:.4f}, "
            f"CH={r['calinski_harabasz']:.1f}, "
            f"DB={r['davies_bouldin']:.4f}, "
            f"ARI={r['median_ari']:.4f}, "
            f"min_cluster={int(r['min_cluster_count'])} "
            f"({r['min_cluster_share']*100:.2f}%), "
            f"median_eps2="
            f"{r['median_feature_epsilon_squared']:.4f}"
            f"{warning}"
        )

    lines += [
        "",
        "Important:",
        "  A small cluster is flagged but not automatically rejected.",
        "  Compare neighboring K values before deciding whether it is",
        "  a genuine minority strategy or over-segmentation.",
    ]

    (
        out_dir / "summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))
    print()
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
