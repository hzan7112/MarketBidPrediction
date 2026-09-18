#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_visualize_final_strategy_profile.py

Visualize the final profile as nine independent strategy dimensions.

Primary figures:
1. 9-D independent feature percentile distribution
2. participant-by-feature percentile heatmap
3. population state shares for every independent feature
4. accepted natural KMeans spaces only
5. cluster centroid percentile heatmaps for accepted spaces

No artificial combination archetypes are plotted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

CN = {
    "lt_bid_level": "报价水平",
    "lt_adjustment_magnitude": "调整幅度",
    "lt_strategy_persistence": "持续性",
    "lt_quantity_hhi": "容量集中度",
    "lt_effective_segment_count": "有效段数",
    "lt_flat_curve_rate": "平价偏好",
    "lt_tail_uplift_ratio": "尾部抬价",
    "lt_curve_bend_ratio": "曲线弯折",
    "lt_shape_variability": "形态波动",
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
            return


def plot_feature_percentile_distributions(profile, out_file):
    data = []
    labels = []

    for f in LT_FEATURES:
        x = pd.to_numeric(
            profile[f"{f}_percentile"],
            errors="coerce",
        ).dropna()

        if len(x):
            data.append(x.to_numpy())
            labels.append(CN[f])

    fig, ax = plt.subplots(figsize=(13, 6))
    vp = ax.violinplot(
        data,
        positions=np.arange(1, len(data) + 1),
        showextrema=False,
        widths=0.85,
    )

    for body in vp["bodies"]:
        body.set_alpha(0.35)

    for i, x in enumerate(data, 1):
        q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
        ax.vlines(i, q1, q3, linewidth=3)
        ax.scatter(i, med, s=30, zorder=3)

    ax.set_xticks(
        np.arange(1, len(labels) + 1),
        labels,
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("经验百分位")
    ax.set_ylim(0, 100)
    ax.set_title("9 个独立长期策略维度的主体分布")
    ax.grid(axis="y", alpha=0.18)

    fig.tight_layout()
    fig.savefig(out_file, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_participant_heatmap(profile, out_file):
    pcols = [f"{f}_percentile" for f in LT_FEATURES]
    mat = profile[pcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    # Sort participants by the first principal ordering proxy:
    # lexicographic order of available percentiles, not a new composite score.
    fill = np.where(np.isfinite(mat), mat, -1)
    order = np.lexsort(tuple(fill[:, j] for j in reversed(range(fill.shape[1]))))
    mat = mat[order]

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=100,
    )

    ax.set_xticks(
        np.arange(len(LT_FEATURES)),
        [CN[f] for f in LT_FEATURES],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("主体（按独立特征排序）")
    ax.set_title("主体 9 维独立策略画像百分位热图")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("经验百分位")

    fig.tight_layout()
    fig.savefig(out_file, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_state_shares(profile, out_file):
    lows = []
    mids = []
    highs = []

    for f in LT_FEATURES:
        s = profile[f"{f}_state"]
        valid = s.notna().sum()
        if valid == 0:
            lows.append(0)
            mids.append(0)
            highs.append(0)
            continue

        lows.append((s == "低").sum() / valid * 100)
        mids.append((s == "中").sum() / valid * 100)
        highs.append((s == "高").sum() / valid * 100)

    y = np.arange(len(LT_FEATURES))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(y, lows, label="低")
    ax.barh(y, mids, left=lows, label="中")
    left2 = np.array(lows) + np.array(mids)
    ax.barh(y, highs, left=left2, label="高")

    ax.set_yticks(y, [CN[f] for f in LT_FEATURES])
    ax.set_xlabel("有效主体占比 (%)")
    ax.set_xlim(0, 100)
    ax.set_title("每个策略特征的独立状态分布")
    ax.legend()
    ax.grid(axis="x", alpha=0.18)

    fig.tight_layout()
    fig.savefig(out_file, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_summary(discovery, out_file):
    d = discovery.copy()
    d["status"] = np.where(d["accepted"].eq(1), "保留", "不保留")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(d))
    vals = d["silhouette"].fillna(0).to_numpy()

    ax.bar(x, vals)
    ax.set_xticks(x, d["space"], rotation=20, ha="right")
    ax.set_ylabel("Silhouette")
    ax.set_title("KMeans 仅作为自然组合发现：哪些空间真正可聚类")
    ax.axhline(0.40, linestyle="--", linewidth=1)

    for i, r in d.iterrows():
        txt = (
            f"{r['status']}\n"
            + (
                f"K={int(r['selected_k'])}\nARI={r['median_ari']:.2f}"
                if r["accepted"] == 1
                else "未通过门槛"
            )
        )
        ax.text(i, vals[i] + 0.02, txt, ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(0.65, vals.max() + 0.18))
    ax.grid(axis="y", alpha=0.18)

    fig.tight_layout()
    fig.savefig(out_file, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_accepted_cluster_heatmaps(profile, discovery, out_dir):
    for _, r in discovery[discovery["accepted"].eq(1)].iterrows():
        space = r["space"]
        features = SPACES[space]
        ccol = f"{space}_cluster"

        if ccol not in profile.columns:
            continue

        x = profile.dropna(subset=[ccol]).copy()
        if x.empty:
            continue

        rows = []
        for c, g in x.groupby(ccol):
            row = {"cluster": int(c), "n": len(g)}
            for f in features:
                row[f] = pd.to_numeric(
                    g[f"{f}_percentile"],
                    errors="coerce",
                ).median()
            rows.append(row)

        d = pd.DataFrame(rows).sort_values("cluster")
        mat = d[features].to_numpy(float)

        fig, ax = plt.subplots(
            figsize=(max(7, 1.5 * len(features)), max(4, 0.7 * len(d) + 2))
        )
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=100)

        ax.set_xticks(
            np.arange(len(features)),
            [CN[f] for f in features],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(
            np.arange(len(d)),
            [f"C{int(c)} (n={int(n)})" for c, n in zip(d["cluster"], d["n"])],
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j, i, f"{mat[i, j]:.0f}",
                        ha="center", va="center", fontsize=9
                    )

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("中位经验百分位")
        ax.set_title(f"{space}: 数据真实支持的 KMeans 组合")

        fig.tight_layout()
        fig.savefig(
            out_dir / f"05_{space}_accepted_cluster_heatmap.png",
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--profile-root",
        default="data/processed/final_clean/strategy_profile",
    )
    p.add_argument(
        "--results-root",
        default="results/final_clean_visualization",
    )
    args = p.parse_args()

    setup_font()

    profile_file = (
        Path(args.profile_root)
        / str(args.year)
        / f"participant_strategy_profile_{args.year}.csv"
    )
    discovery_file = (
        Path(args.profile_root)
        / str(args.year)
        / f"cluster_discovery_summary_{args.year}.csv"
    )

    if not profile_file.exists():
        raise FileNotFoundError(profile_file)
    if not discovery_file.exists():
        raise FileNotFoundError(discovery_file)

    profile = pd.read_csv(profile_file, low_memory=False)
    discovery = pd.read_csv(discovery_file, low_memory=False)

    out_dir = ensure_dir(Path(args.results_root) / str(args.year))

    plot_feature_percentile_distributions(
        profile,
        out_dir / "01_independent_feature_percentile_distributions.png",
    )
    plot_participant_heatmap(
        profile,
        out_dir / "02_participant_9d_profile_heatmap.png",
    )
    plot_state_shares(
        profile,
        out_dir / "03_independent_feature_state_shares.png",
    )
    plot_cluster_summary(
        discovery,
        out_dir / "04_kmeans_natural_cluster_discovery.png",
    )
    plot_accepted_cluster_heatmaps(
        profile,
        discovery,
        out_dir,
    )

    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
