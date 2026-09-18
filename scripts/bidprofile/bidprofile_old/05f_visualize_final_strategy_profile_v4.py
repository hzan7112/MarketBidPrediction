#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05f_visualize_final_strategy_profile_v4.py
Version: 2026-09-18-v4

Purpose
-------
Validate the independent discriminative power of different bidding-strategy
feature groups instead of relying on one full 9-D clustering result.

Ablation sets
-------------
A. full_9d
   all 9 final LT strategy features

B. no_shape_variability_8d
   all final LT features except lt_shape_day_deviation_median

C. price_adjustment_3d
   lt_bid_level
   lt_self_adjustment_magnitude
   lt_strategy_persistence

D. quantity_structure_3d
   lt_quantity_hhi
   lt_effective_segment_count
   lt_flat_curve_rate

E. curve_shape_3d
   lt_tail_uplift_ratio
   lt_curve_bend_ratio
   lt_shape_day_deviation_median

For every feature set:
- select K from a candidate range by full-space silhouette;
- report Silhouette / CH / DB;
- report KMeans initialization stability via median ARI;
- assign participant clusters;
- compute per-feature Kruskal-Wallis epsilon-squared;
- compute pairwise Cliff's delta;
- generate observed-value cluster distributions for the features in that set.

Cross-ablation outputs:
- ablation_clustering_summary_2025.csv
- 01_ablation_clustering_comparison.png
- 02_ablation_feature_separation_heatmap.png

Run
---
python scripts\\05f_visualize_final_strategy_profile_v4.py --year 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import RobustScaler


VERSION = "2026-09-18-v4"

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

FEATURE_SETS = {
    "full_9d": [
        "lt_bid_level",
        "lt_self_adjustment_magnitude",
        "lt_strategy_persistence",
        "lt_quantity_hhi",
        "lt_effective_segment_count",
        "lt_flat_curve_rate",
        "lt_tail_uplift_ratio",
        "lt_curve_bend_ratio",
        "lt_shape_day_deviation_median",
    ],
    "no_shape_variability_8d": [
        "lt_bid_level",
        "lt_self_adjustment_magnitude",
        "lt_strategy_persistence",
        "lt_quantity_hhi",
        "lt_effective_segment_count",
        "lt_flat_curve_rate",
        "lt_tail_uplift_ratio",
        "lt_curve_bend_ratio",
    ],
    "price_adjustment_3d": [
        "lt_bid_level",
        "lt_self_adjustment_magnitude",
        "lt_strategy_persistence",
    ],
    "quantity_structure_3d": [
        "lt_quantity_hhi",
        "lt_effective_segment_count",
        "lt_flat_curve_rate",
    ],
    "curve_shape_3d": [
        "lt_tail_uplift_ratio",
        "lt_curve_bend_ratio",
        "lt_shape_day_deviation_median",
    ],
}

SET_DISPLAY_CN = {
    "full_9d": "全部9维",
    "no_shape_variability_8d": "去除形态波动后8维",
    "price_adjustment_3d": "价格/调整3维",
    "quantity_structure_3d": "数量/结构3维",
    "curve_shape_3d": "曲线形态3维",
}

SET_DISPLAY_EN = {
    "full_9d": "Full 9D",
    "no_shape_variability_8d": "8D w/o shape variability",
    "price_adjustment_3d": "Price / adjustment 3D",
    "quantity_structure_3d": "Quantity / structure 3D",
    "curve_shape_3d": "Curve-shape 3D",
}

DISPLAY_CN = {
    "lt_bid_level": "长期报价水平",
    "lt_self_adjustment_magnitude": "长期调整幅度",
    "lt_strategy_persistence": "策略持续性",
    "lt_quantity_hhi": "容量配置集中度",
    "lt_effective_segment_count": "有效报价段数",
    "lt_flat_curve_rate": "平价曲线偏好",
    "lt_tail_uplift_ratio": "尾部抬价倾向",
    "lt_curve_bend_ratio": "曲线弯折结构",
    "lt_shape_day_deviation_median": "曲线形态波动",
}

DISPLAY_EN = {
    "lt_bid_level": "Bid level",
    "lt_self_adjustment_magnitude": "Adjustment magnitude",
    "lt_strategy_persistence": "Strategy persistence",
    "lt_quantity_hhi": "Quantity concentration",
    "lt_effective_segment_count": "Effective segments",
    "lt_flat_curve_rate": "Flat-curve preference",
    "lt_tail_uplift_ratio": "Tail uplift",
    "lt_curve_bend_ratio": "Curve bend",
    "lt_shape_day_deviation_median": "Shape variability",
}


def setup_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def winsorize_frame(df, features, q_low=0.01, q_high=0.99):
    out = df[features].apply(pd.to_numeric, errors="coerce").copy()
    bounds = []

    for f in features:
        lo = out[f].quantile(q_low)
        hi = out[f].quantile(q_high)

        if pd.notna(lo) and pd.notna(hi) and hi >= lo:
            out[f] = out[f].clip(lo, hi)

        bounds.append({
            "feature": f,
            "q_low": lo,
            "q_high": hi,
        })

    return out, pd.DataFrame(bounds)


def prepare_matrix(cohort, features):
    """
    Imputation is ONLY for clustering distance.
    Raw observed values are used for all separation plots/statistics.
    """
    x_clip, bounds = winsorize_frame(cohort, features)

    missing = (
        x_clip.isna().mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x_clip)

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    x_scaled = scaler.fit_transform(x_imp)

    return x_scaled, bounds, missing


def evaluate_k(x, k_min, k_max, random_state):
    rows = []

    for k in range(k_min, k_max + 1):
        labels = KMeans(
            n_clusters=k,
            n_init=80,
            random_state=random_state,
        ).fit_predict(x)

        rows.append({
            "k": k,
            "silhouette": silhouette_score(x, labels),
            "calinski_harabasz": calinski_harabasz_score(x, labels),
            "davies_bouldin": davies_bouldin_score(x, labels),
        })

    df = pd.DataFrame(rows)

    best_k = int(
        df.loc[df["silhouette"].idxmax(), "k"]
    )

    return df, best_k


def fit_clusters(x, k, random_state):
    model = KMeans(
        n_clusters=k,
        n_init=150,
        random_state=random_state,
    )
    labels = model.fit_predict(x)
    return model, labels


def stability_ari(x, k, reference_labels, n_runs=30):
    vals = []

    for seed in range(n_runs):
        labels = KMeans(
            n_clusters=k,
            n_init=30,
            random_state=seed,
        ).fit_predict(x)

        vals.append(
            adjusted_rand_score(
                reference_labels,
                labels,
            )
        )

    return float(np.median(vals)), pd.DataFrame({
        "seed": range(n_runs),
        "ari_vs_reference": vals,
    })


def feature_discrimination(cohort, features):
    """
    Kruskal-Wallis epsilon-squared:
        eps2 = (H - K + 1)/(N - K)
    """
    clusters = sorted(cohort["cluster"].unique())
    rows = []

    for f in features:
        groups = []
        total_n = 0

        for c in clusters:
            x = pd.to_numeric(
                cohort.loc[cohort["cluster"] == c, f],
                errors="coerce",
            ).dropna().to_numpy()

            if len(x):
                groups.append(x)
                total_n += len(x)

        if len(groups) < 2:
            H = p = eps2 = np.nan
        else:
            H, p = kruskal(*groups)
            K = len(groups)
            eps2 = (
                (H - K + 1) / (total_n - K)
                if total_n > K else np.nan
            )
            if np.isfinite(eps2):
                eps2 = max(0.0, float(eps2))

        rows.append({
            "feature": f,
            "valid_n": total_n,
            "kruskal_H": H,
            "kruskal_p": p,
            "epsilon_squared": eps2,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("epsilon_squared", ascending=False)
        .reset_index(drop=True)
    )


def pairwise_cliffs_delta(cohort, features):
    clusters = sorted(cohort["cluster"].unique())
    rows = []

    for f in features:
        for a, b in combinations(clusters, 2):
            xa = pd.to_numeric(
                cohort.loc[cohort["cluster"] == a, f],
                errors="coerce",
            ).dropna().to_numpy()

            xb = pd.to_numeric(
                cohort.loc[cohort["cluster"] == b, f],
                errors="coerce",
            ).dropna().to_numpy()

            if len(xa) == 0 or len(xb) == 0:
                delta = np.nan
            else:
                U = mannwhitneyu(
                    xa,
                    xb,
                    alternative="two-sided",
                    method="auto",
                ).statistic

                delta = 2.0 * U / (len(xa) * len(xb)) - 1.0

            rows.append({
                "feature": f,
                "cluster_a": int(a),
                "cluster_b": int(b),
                "pair": f"C{a} vs C{b}",
                "n_a": len(xa),
                "n_b": len(xb),
                "cliffs_delta": delta,
                "abs_cliffs_delta": (
                    abs(delta) if np.isfinite(delta) else np.nan
                ),
            })

    return pd.DataFrame(rows)


def observed_cluster_percentiles(cohort, features):
    out = cohort[
        ["participant_id", "market_product", "cluster"]
    ].copy()

    for f in features:
        x = pd.to_numeric(cohort[f], errors="coerce")
        out[f] = x.rank(
            method="average",
            pct=True,
        ) * 100

    rows = []

    for c, g in out.groupby("cluster", sort=True):
        row = {
            "cluster": int(c),
            "participant_count": len(g),
        }

        for f in features:
            row[f] = g[f].median()

        rows.append(row)

    return pd.DataFrame(rows)


def plot_ablation_summary(summary, fig_dir, set_labels):
    d = summary.copy()

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    x = np.arange(len(d))
    width = 0.36

    ax.bar(
        x - width / 2,
        d["silhouette"],
        width=width,
        label="Silhouette",
    )
    ax.bar(
        x + width / 2,
        d["median_ari"],
        width=width,
        label="Median ARI",
    )

    ax.set_xticks(
        x,
        [set_labels[s] for s in d["feature_set"]],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(
        "不同报价策略特征集合的聚类能力对比"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.18)

    for i, r in d.iterrows():
        ax.text(
            i,
            1.01,
            f"K={int(r['best_k'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(
        fig_dir / "01_ablation_clustering_comparison.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ablation_feature_heatmap(all_sep, fig_dir, labels_map, set_labels):
    """
    Rows = feature sets
    Cols = all 9 LT features
    Cell = epsilon^2 if feature belongs to that set, NaN otherwise
    """
    mat = np.full(
        (len(FEATURE_SETS), len(LT_CORE)),
        np.nan,
    )

    set_names = list(FEATURE_SETS.keys())

    for i, set_name in enumerate(set_names):
        sub = all_sep[
            all_sep["feature_set"] == set_name
        ].set_index("feature")

        for j, f in enumerate(LT_CORE):
            if f in sub.index:
                mat[i, j] = sub.loc[
                    f,
                    "epsilon_squared",
                ]

    fig, ax = plt.subplots(figsize=(12.8, 5.4))

    masked = np.ma.masked_invalid(mat)
    vmax = np.nanmax(mat)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = ax.imshow(
        masked,
        aspect="auto",
        vmin=0,
        vmax=vmax,
    )

    ax.set_xticks(
        np.arange(len(LT_CORE)),
        [labels_map[f] for f in LT_CORE],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(set_names)),
        [set_labels[s] for s in set_names],
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(
        "Kruskal-Wallis ε²：该特征对该组聚类的总体区分强度"
    )

    ax.set_title(
        "消融后，各报价策略特征在对应聚类中的区分能力"
    )

    fig.tight_layout()
    fig.savefig(
        fig_dir / "02_ablation_feature_separation_heatmap.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_feature_distributions(
    cohort,
    features,
    set_name,
    out_dir,
    labels_map,
    set_labels,
):
    n = len(features)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.0 * cols, 4.0 * rows),
        constrained_layout=True,
    )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes = axes.ravel()
    clusters = sorted(cohort["cluster"].unique())

    for ax, f in zip(axes, features):
        all_x = pd.to_numeric(
            cohort[f],
            errors="coerce",
        )

        lo = all_x.quantile(0.01)
        hi = all_x.quantile(0.99)

        data = []
        positions = []

        for pos, c in enumerate(clusters, start=1):
            x = pd.to_numeric(
                cohort.loc[
                    cohort["cluster"] == c,
                    f,
                ],
                errors="coerce",
            ).dropna()

            if pd.notna(lo) and pd.notna(hi):
                x = x.clip(lo, hi)

            if len(x):
                data.append(x.to_numpy())
                positions.append(pos)

        if data:
            vp = ax.violinplot(
                data,
                positions=positions,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.82,
            )

            for body in vp["bodies"]:
                body.set_alpha(0.35)

            for pos, x in zip(positions, data):
                q1, med, q3 = np.quantile(
                    x,
                    [0.25, 0.5, 0.75],
                )
                ax.vlines(
                    pos,
                    q1,
                    q3,
                    linewidth=3,
                )
                ax.scatter(
                    pos,
                    med,
                    s=28,
                    zorder=3,
                )

        ax.set_title(labels_map[f])
        ax.set_xticks(
            range(1, len(clusters) + 1),
            [f"C{c}" for c in clusters],
        )
        ax.grid(axis="y", alpha=0.18)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{set_labels[set_name]}：不同策略类型在各特征上的真实分布"
    )

    fig.savefig(
        out_dir / f"{set_name}_feature_distributions.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_percentile_heatmap(
    pct,
    features,
    set_name,
    out_dir,
    labels_map,
    set_labels,
):
    d = pct.sort_values("cluster")
    mat = d[features].to_numpy(float)

    fig, ax = plt.subplots(
        figsize=(max(7.5, 1.25 * len(features)), 4.2)
    )

    im = ax.imshow(
        mat,
        aspect="auto",
        vmin=0,
        vmax=100,
    )

    ax.set_xticks(
        np.arange(len(features)),
        [labels_map[f] for f in features],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(d)),
        [
            f"C{int(c)} (n={int(n)})"
            for c, n in zip(
                d["cluster"],
                d["participant_count"],
            )
        ],
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"P{mat[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Cluster 内主体在全部 cohort 中的中位经验百分位")

    ax.set_title(
        f"{set_labels[set_name]}：策略类型画像指纹"
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / f"{set_name}_cluster_percentile_heatmap.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_effect_heatmap(
    effect,
    features,
    set_name,
    out_dir,
    labels_map,
    set_labels,
):
    pairs = sorted(effect["pair"].unique())

    if len(pairs) == 0:
        return

    pivot = (
        effect.pivot(
            index="feature",
            columns="pair",
            values="abs_cliffs_delta",
        )
        .reindex(features)
        .reindex(columns=pairs)
    )

    mat = pivot.to_numpy(float)

    fig, ax = plt.subplots(
        figsize=(max(7.5, 1.1 * len(pairs) + 4), 4.8)
    )

    im = ax.imshow(
        mat,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(len(pairs)),
        pairs,
        rotation=35,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(features)),
        [labels_map[f] for f in features],
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("|Cliff's delta|")

    ax.set_title(
        f"{set_labels[set_name]}：各 Cluster 两两特征区分度"
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / f"{set_name}_pairwise_cliffs_delta.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_one_feature_set(
    base_cohort,
    set_name,
    features,
    out_root,
    k_min,
    k_max,
    random_state,
    labels_map,
    set_labels,
):
    set_dir = out_root / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    x_scaled, bounds, missing = prepare_matrix(
        base_cohort,
        features,
    )

    bounds.to_csv(
        set_dir / "winsorization_bounds.csv",
        index=False,
    )
    missing.to_csv(
        set_dir / "missingness.csv",
        index=False,
    )

    max_k = min(
        k_max,
        len(base_cohort) - 1,
    )

    metrics, best_k = evaluate_k(
        x_scaled,
        k_min,
        max_k,
        random_state,
    )
    metrics.to_csv(
        set_dir / "cluster_selection_metrics.csv",
        index=False,
    )

    model, labels = fit_clusters(
        x_scaled,
        best_k,
        random_state,
    )

    cohort = base_cohort.copy()
    cohort["cluster"] = labels

    labels_df = cohort[
        ["participant_id", "market_product", "cluster"]
    ]
    labels_df.to_csv(
        set_dir / "participant_cluster_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    median_ari, stability_df = stability_ari(
        x_scaled,
        best_k,
        labels,
        n_runs=30,
    )
    stability_df.to_csv(
        set_dir / "cluster_stability.csv",
        index=False,
    )

    sep = feature_discrimination(
        cohort,
        features,
    )
    sep["feature_set"] = set_name
    sep.to_csv(
        set_dir / "feature_discrimination_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    effect = pairwise_cliffs_delta(
        cohort,
        features,
    )
    effect["feature_set"] = set_name
    effect.to_csv(
        set_dir / "pairwise_cliffs_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pct = observed_cluster_percentiles(
        cohort,
        features,
    )
    pct.to_csv(
        set_dir / "cluster_percentile_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw = (
        cohort.groupby("cluster")[features]
        .median()
        .reset_index()
    )
    counts = (
        cohort.groupby("cluster")
        .size()
        .rename("participant_count")
        .reset_index()
    )
    raw = raw.merge(counts, on="cluster")
    raw.to_csv(
        set_dir / "cluster_raw_medians.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_feature_distributions(
        cohort,
        features,
        set_name,
        set_dir,
        labels_map,
        set_labels,
    )

    plot_percentile_heatmap(
        pct,
        features,
        set_name,
        set_dir,
        labels_map,
        set_labels,
    )

    plot_effect_heatmap(
        effect,
        features,
        set_name,
        set_dir,
        labels_map,
        set_labels,
    )

    best_row = metrics.loc[
        metrics["k"] == best_k
    ].iloc[0]

    cluster_sizes = (
        cohort.groupby("cluster")
        .size()
        .sort_index()
    )

    summary = {
        "feature_set": set_name,
        "feature_count": len(features),
        "participants": len(cohort),
        "best_k": best_k,
        "silhouette": best_row["silhouette"],
        "calinski_harabasz": best_row["calinski_harabasz"],
        "davies_bouldin": best_row["davies_bouldin"],
        "median_ari": median_ari,
        "largest_cluster_share": cluster_sizes.max() / len(cohort),
        "smallest_cluster_share": cluster_sizes.min() / len(cohort),
        "max_feature_epsilon_squared": sep["epsilon_squared"].max(),
        "median_feature_epsilon_squared": sep["epsilon_squared"].median(),
        "top_feature": (
            sep.iloc[0]["feature"] if len(sep) else None
        ),
    }

    return summary, sep, effect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--final-root",
        default="data/processed/final_strategy_profile",
    )
    parser.add_argument(
        "--results-root",
        default="results/05f_strategy_ablation_validation",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    use_cn = setup_font()
    labels_map = DISPLAY_CN if use_cn else DISPLAY_EN
    set_labels = SET_DISPLAY_CN if use_cn else SET_DISPLAY_EN

    final_file = (
        Path(args.final_root)
        / str(args.year)
        / f"final_long_term_profile_{args.year}.csv"
    )

    if not final_file.exists():
        raise FileNotFoundError(final_file)

    out_root = (
        Path(args.results_root)
        / str(args.year)
    )
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    profile = pd.read_csv(
        final_file,
        low_memory=False,
    )

    require_columns(
        profile,
        [
            "participant_id",
            "market_product",
            "lt_core_ready_flag",
        ] + LT_CORE,
        "final LT profile",
    )

    # Same cohort for all ablations, so metric comparisons are fair.
    base_cohort = profile[
        profile["lt_core_ready_flag"].eq(1)
    ].copy().reset_index(drop=True)

    if len(base_cohort) < 50:
        raise ValueError(
            f"Only {len(base_cohort)} LT-core-ready participants."
        )

    all_summaries = []
    all_sep = []
    all_effect = []

    for set_name, features in FEATURE_SETS.items():
        print(
            f"[run] {set_name}: "
            f"{len(features)} features"
        )

        summary, sep, effect = run_one_feature_set(
            base_cohort,
            set_name,
            features,
            out_root,
            args.k_min,
            args.k_max,
            args.random_state,
            labels_map,
            set_labels,
        )

        all_summaries.append(summary)
        all_sep.append(sep)
        all_effect.append(effect)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(
        out_root / f"ablation_clustering_summary_{args.year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sep_df = pd.concat(
        all_sep,
        ignore_index=True,
    )
    sep_df.to_csv(
        out_root / f"ablation_feature_discrimination_{args.year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    effect_df = pd.concat(
        all_effect,
        ignore_index=True,
    )
    effect_df.to_csv(
        out_root / f"ablation_pairwise_cliffs_delta_{args.year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_ablation_summary(
        summary_df,
        fig_dir,
        set_labels,
    )

    plot_ablation_feature_heatmap(
        sep_df,
        fig_dir,
        labels_map,
        set_labels,
    )

    # Compact text summary.
    lines = [
        f"Strategy-profile ablation validation - {args.year}",
        f"Version: {VERSION}",
        "=" * 78,
        f"Common clustering cohort: {len(base_cohort):,}",
        "",
    ]

    for _, r in summary_df.iterrows():
        lines += [
            f"[{r['feature_set']}]",
            f"  features = {int(r['feature_count'])}",
            f"  best K = {int(r['best_k'])}",
            f"  silhouette = {r['silhouette']:.4f}",
            f"  CH = {r['calinski_harabasz']:.2f}",
            f"  DB = {r['davies_bouldin']:.4f}",
            f"  median ARI = {r['median_ari']:.4f}",
            f"  largest cluster share = {r['largest_cluster_share']*100:.1f}%",
            f"  smallest cluster share = {r['smallest_cluster_share']*100:.1f}%",
            f"  top separating feature = {r['top_feature']}",
            f"  max epsilon^2 = {r['max_feature_epsilon_squared']:.4f}",
            f"  median epsilon^2 = {r['median_feature_epsilon_squared']:.4f}",
            "",
        ]

    lines += [
        "Main comparison outputs:",
        f"  {(fig_dir / '01_ablation_clustering_comparison.png').resolve()}",
        f"  {(fig_dir / '02_ablation_feature_separation_heatmap.png').resolve()}",
        "",
        "Each feature-set directory contains:",
        "  cluster_selection_metrics.csv",
        "  participant_cluster_labels.csv",
        "  cluster_stability.csv",
        "  feature_discrimination_statistics.csv",
        "  pairwise_cliffs_delta.csv",
        "  cluster_percentile_profile.csv",
        "  cluster_raw_medians.csv",
        "  <feature_set>_feature_distributions.png",
        "  <feature_set>_cluster_percentile_heatmap.png",
        "  <feature_set>_pairwise_cliffs_delta.png",
    ]

    (out_root / f"summary_{args.year}.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(f"Done: {out_root.resolve()}")


if __name__ == "__main__":
    main()
