#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_build_strategy_profile.py

Final principle
---------------
The strategy profile itself is NOT a cluster label.

Each participant keeps nine independent long-term strategy dimensions:
1. bid level
2. adjustment magnitude
3. strategy persistence
4. quantity HHI
5. effective segment count
6. flat-curve preference
7. tail uplift
8. curve bend
9. shape variability

For interpretation, each dimension additionally receives:
- empirical percentile in the eligible participant population
- an independent low / middle / high state based on empirical terciles

KMeans is only an auxiliary discovery tool:
- full 9-D profile
- price/adjustment 3-D subspace
- quantity/structure 3-D subspace
- shape 3-D subspace

For every space, K=2..6 is searched.
A cluster solution is retained only when it passes BOTH:
- silhouette >= 0.40
- median ARI across random seeds >= 0.80

No artificial cross-product combination is constructed.
No 3x4x2 archetype system is constructed.
"""
from __future__ import annotations

import os
import warnings

# Windows + MKL: limit thread fan-out and silence known non-fatal KMeans/joblib warnings.
# These must be set before importing NumPy / scikit-learn.
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
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import RobustScaler

from strategy_core import (
    LT_FEATURES,
    PA_FEATURES,
    QS_FEATURES,
    SHAPE_FEATURES,
    ensure_dir,
)

SPACES = {
    "full_9d": LT_FEATURES,
    "price_adjustment_3d": PA_FEATURES,
    "quantity_structure_3d": QS_FEATURES,
    "curve_shape_3d": SHAPE_FEATURES,
}

K_CANDIDATES = range(2, 7)
MIN_SILHOUETTE = 0.40
MIN_MEDIAN_ARI = 0.80


def empirical_percentile(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.rank(method="average", pct=True) * 100.0


def independent_state_from_percentile(p):
    """
    Independent descriptive state only.
    These states are NOT clusters and are NOT combined into archetypes.
    """
    out = pd.Series(pd.NA, index=p.index, dtype="object")
    out[p <= 33.333333] = "低"
    out[(p > 33.333333) & (p < 66.666667)] = "中"
    out[p >= 66.666667] = "高"
    return out


def prep_space(train, all_df, features):
    """
    Natural clustering must use only participants with complete observations
    in the current feature space. Missing dimensions are never median-imputed
    for cluster discovery.
    """
    train_x = (
        train[features]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .copy()
    )

    valid_all = (
        all_df[features]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )
    all_x = (
        all_df.loc[valid_all, features]
        .apply(pd.to_numeric, errors="coerce")
        .copy()
    )

    bounds = {}
    for f in features:
        lo = train_x[f].quantile(0.01)
        hi = train_x[f].quantile(0.99)
        bounds[f] = (lo, hi)
        train_x[f] = train_x[f].clip(lo, hi)
        all_x[f] = all_x[f].clip(lo, hi)

    scaler = RobustScaler(quantile_range=(25, 75))
    x_train = scaler.fit_transform(train_x)
    x_all = scaler.transform(all_x)

    return (
        train_x.index,
        valid_all,
        x_train,
        x_all,
    )


def evaluate_kmeans(x, k):
    ref = KMeans(
        n_clusters=k,
        n_init=50,
        random_state=42,
    ).fit_predict(x)

    sil = silhouette_score(x, ref)

    aris = []
    for seed in range(10):
        lab = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=seed,
        ).fit_predict(x)
        aris.append(adjusted_rand_score(ref, lab))

    return {
        "k": k,
        "silhouette": float(sil),
        "median_ari": float(np.median(aris)),
        "labels": ref,
    }


def discover_space(train, all_df, features):
    train_index, valid_all, x_train, x_all = prep_space(
        train,
        all_df,
        features,
    )

    candidates = []
    for k in K_CANDIDATES:
        if len(train) <= k:
            continue

        print(
            f"    evaluating K={k} ...",
            flush=True,
        )
        result = evaluate_kmeans(x_train, k)
        candidates.append(result)

        print(
            f"      silhouette={result['silhouette']:.4f}, "
            f"median_ARI={result['median_ari']:.4f}",
            flush=True,
        )

    metrics = pd.DataFrame(
        [
            {
                "k": c["k"],
                "silhouette": c["silhouette"],
                "median_ari": c["median_ari"],
            }
            for c in candidates
        ]
    )

    if metrics.empty:
        return None, metrics

    eligible = metrics[
        (metrics["silhouette"] >= MIN_SILHOUETTE)
        & (metrics["median_ari"] >= MIN_MEDIAN_ARI)
    ].copy()

    if eligible.empty:
        return None, metrics

    # Among genuinely stable solutions, choose the strongest geometric separation.
    best_row = eligible.sort_values(
        ["silhouette", "median_ari"],
        ascending=False,
    ).iloc[0]

    best_k = int(best_row["k"])

    model = KMeans(
        n_clusters=best_k,
        n_init=100,
        random_state=42,
    )
    train_labels = model.fit_predict(x_train)
    all_labels = model.predict(x_all)

    counts = np.bincount(train_labels, minlength=best_k)
    min_cluster_count = int(counts.min())
    min_cluster_share = float(counts.min() / counts.sum())

    return {
        "k": best_k,
        "silhouette": float(best_row["silhouette"]),
        "median_ari": float(best_row["median_ari"]),
        "train_index": train_index,
        "valid_all": valid_all,
        "train_labels": train_labels,
        "all_labels": all_labels,
        "cluster_counts": counts,
        "min_cluster_count": min_cluster_count,
        "min_cluster_share": min_cluster_share,
    }, metrics


def cluster_percentile_profile(train, features, labels):
    temp = train.copy()
    temp["_cluster"] = labels

    pct = pd.DataFrame(index=temp.index)
    for f in features:
        pct[f] = empirical_percentile(temp[f])
    pct["_cluster"] = labels

    counts = (
        temp["_cluster"]
        .value_counts()
        .rename("participant_count")
        .rename_axis("cluster")
        .reset_index()
    )

    prof = (
        pct.groupby("_cluster")[features]
        .median()
        .reset_index()
        .rename(columns={"_cluster": "cluster"})
        .merge(counts, on="cluster", how="left")
        .sort_values("cluster")
    )
    return prof


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--lt-root",
        default="data/processed/final_clean/long_term",
    )
    p.add_argument(
        "--out-root",
        default="data/processed/final_clean/strategy_profile",
    )
    args = p.parse_args()

    src = (
        Path(args.lt_root)
        / str(args.year)
        / f"long_term_strategy_profile_{args.year}.csv"
    )
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = ensure_dir(Path(args.out_root) / str(args.year))
    df = pd.read_csv(src, low_memory=False)

    # ------------------------------------------------------------
    # 1. Profile body = nine independent dimensions
    # ------------------------------------------------------------
    profile = df.copy()

    for f in LT_FEATURES:
        pct_col = f"{f}_percentile"
        state_col = f"{f}_state"

        profile[pct_col] = empirical_percentile(profile[f])
        profile[state_col] = independent_state_from_percentile(
            profile[pct_col]
        )

    # Compact human-readable fingerprint, but NOT a combined cluster.
    state_cols = [f"{f}_state" for f in LT_FEATURES]
    profile["independent_state_fingerprint"] = profile[state_cols].apply(
        lambda r: " | ".join(
            [
                f"{f.replace('lt_', '')}={r[f'{f}_state']}"
                for f in LT_FEATURES
                if pd.notna(r[f"{f}_state"])
            ]
        ),
        axis=1,
    )

    # ------------------------------------------------------------
    # 2. KMeans = auxiliary natural-pattern discovery only
    # ------------------------------------------------------------
    train = profile[profile["lt_ready_flag"].eq(1)].copy()

    if len(train) < 50:
        raise ValueError("Too few LT-ready participants for clustering discovery.")

    discovery_rows = []

    for space_name, features in SPACES.items():
        print()
        print(
            f"[04] space={space_name}, "
            f"features={len(features)}",
            flush=True,
        )

        result, metrics = discover_space(train, profile, features)

        metrics.to_csv(
            out_dir / f"{space_name}_k_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )

        cluster_col = f"{space_name}_cluster"
        accepted_col = f"{space_name}_cluster_accepted"

        complete_train_n = int(
            train[features]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .all(axis=1)
            .sum()
        )
        complete_all_mask = (
            profile[features]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .all(axis=1)
        )
        complete_all_n = int(complete_all_mask.sum())

        if result is None:
            print(
                f"  -> rejected: complete_n={complete_train_n}, no K passed "
                f"silhouette>={MIN_SILHOUETTE:.2f} and "
                f"median_ARI>={MIN_MEDIAN_ARI:.2f}",
                flush=True,
            )
            profile[cluster_col] = pd.NA
            profile[accepted_col] = 0

            discovery_rows.append(
                {
                    "space": space_name,
                    "features": ",".join(features),
                    "complete_discovery_n": complete_train_n,
                    "complete_all_n": complete_all_n,
                    "accepted": 0,
                    "selected_k": np.nan,
                    "silhouette": np.nan,
                    "median_ari": np.nan,
                    "min_cluster_count": np.nan,
                    "min_cluster_share": np.nan,
                }
            )
            continue

        print(
            f"  -> accepted K={result['k']}, "
            f"complete_n={complete_train_n}, "
            f"silhouette={result['silhouette']:.4f}, "
            f"median_ARI={result['median_ari']:.4f}, "
            f"min_cluster={result['min_cluster_count']} "
            f"({result['min_cluster_share']*100:.2f}%)",
            flush=True,
        )

        # Only participants with all features observed in this subspace get
        # a cluster label. Missing-feature participants remain unclassified.
        profile[cluster_col] = pd.Series(
            pd.NA,
            index=profile.index,
            dtype="Int64",
        )
        profile.loc[result["valid_all"], cluster_col] = result[
            "all_labels"
        ].astype(int)
        profile[accepted_col] = result["valid_all"].astype(int)

        train_complete = train.loc[result["train_index"]].copy()
        prof = cluster_percentile_profile(
            train_complete,
            features,
            result["train_labels"],
        )
        prof.to_csv(
            out_dir / f"{space_name}_cluster_percentile_profile.csv",
            index=False,
            encoding="utf-8-sig",
        )

        discovery_rows.append(
            {
                "space": space_name,
                "features": ",".join(features),
                "complete_discovery_n": complete_train_n,
                "complete_all_n": complete_all_n,
                "accepted": 1,
                "selected_k": result["k"],
                "silhouette": result["silhouette"],
                "median_ari": result["median_ari"],
                "min_cluster_count": result["min_cluster_count"],
                "min_cluster_share": result["min_cluster_share"],
            }
        )

    discovery = pd.DataFrame(discovery_rows)

    profile_file = (
        out_dir / f"participant_strategy_profile_{args.year}.csv"
    )
    discovery_file = (
        out_dir / f"cluster_discovery_summary_{args.year}.csv"
    )

    profile.to_csv(
        profile_file,
        index=False,
        encoding="utf-8-sig",
    )
    discovery.to_csv(
        discovery_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Participants: {len(profile):,}")
    print(f"LT-ready discovery cohort: {len(train):,}")
    print()
    print("[Natural cluster discovery]")
    print(discovery.to_string(index=False))
    print()
    print("Important: cluster labels are auxiliary only.")
    print("The strategy profile itself remains the 9 independent dimensions.")
    print(f"Done: {profile_file}")


if __name__ == "__main__":
    main()
