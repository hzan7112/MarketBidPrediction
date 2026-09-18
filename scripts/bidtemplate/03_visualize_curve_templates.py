#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_visualize_curve_templates.py

Mode-aware visualization of the final bid-curve template library.
Run 04_validate_template_representatives.py first so that each shape template
has a real block representative and a real sloped representative.

Outputs
-------
data/processed/bidtemplate/<year>/template_library/figures/
    template_share_all.png
    template_share_shape_only.png
    template_gallery.png
    top_templates_overlay.png
    template_visual_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_pct(x):
    return "NA" if pd.isna(x) else f"{x * 100:.2f}%"


def load_inputs(base_dir: Path):
    lib_file = base_dir / "curve_template_library.csv"
    summary_file = base_dir / "curve_template_cluster_summary.csv"
    validation_dir = base_dir / "validation"
    mode_file = validation_dir / "template_mode_composition.csv"
    rep_file = validation_dir / "template_representative_samples.csv"

    for f in [lib_file, summary_file, mode_file, rep_file]:
        if not f.exists():
            raise FileNotFoundError(
                f"Missing file: {f}\n"
                "Run 02 first, then run 04 before running this visualization."
            )

    lib = pd.read_csv(lib_file)
    summary = pd.read_csv(summary_file)
    mode_df = pd.read_csv(mode_file)
    rep_df = pd.read_csv(rep_file)

    if "representative_mode" not in rep_df.columns:
        raise ValueError(
            "template_representative_samples.csv is from the old 04 version. "
            "Rerun the updated 04_validate_template_representatives.py first."
        )

    df = lib.merge(
        summary[[
            "template_id",
            "mean_shape_mae",
            "mean_shape_rmse",
            "mean_price_mae",
            "mean_price_rmse",
        ]],
        on="template_id",
        how="left",
    ).merge(
        mode_df[[
            "template_id",
            "block_share",
            "sloped_share",
        ]],
        on="template_id",
        how="left",
    )

    return df, rep_df


def plot_template_share_all(df, out_file):
    plot_df = df.sort_values("share_of_all", ascending=True).copy()
    labels = plot_df["template_id"].tolist()
    shares = plot_df["share_of_all"].to_numpy(float)
    counts = plot_df["sample_count"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.48 * len(plot_df) + 1.5)))
    bars = ax.barh(labels, shares)
    ax.set_title("Curve template share of all valid bid curves")
    ax.set_xlabel("Share of all valid curves")
    ax.set_ylabel("Template")
    ax.set_xlim(0, max(shares) * 1.18)
    ax.grid(axis="x", alpha=0.25)

    for bar, share, count in zip(bars, shares, counts):
        ax.text(
            bar.get_width() + max(shares) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{share*100:.2f}%   n={count:,}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def plot_template_share_shape_only(df, out_file):
    plot_df = df[df["template_family"] == "shape"].copy()
    plot_df = plot_df.sort_values("share_of_shape", ascending=True)
    labels = plot_df["template_id"].tolist()
    shares = plot_df["share_of_shape"].to_numpy(float)
    counts = plot_df["sample_count"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.48 * len(plot_df) + 1.5)))
    bars = ax.barh(labels, shares)
    ax.set_title("Shape-template share within non-flat curves")
    ax.set_xlabel("Share of shape family")
    ax.set_ylabel("Template")
    ax.set_xlim(0, max(shares) * 1.18)
    ax.grid(axis="x", alpha=0.25)

    for bar, share, count in zip(bars, shares, counts):
        ax.text(
            bar.get_width() + max(shares) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{share*100:.2f}%   n={count:,}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def plot_template_gallery(df, rep_df, out_file, ncols=4):
    plot_df = df.copy()
    plot_df["sort_key"] = plot_df["template_id"].map(
        lambda x: 999 if x == "FLAT" else int(str(x).replace("T", ""))
    )
    plot_df = plot_df.sort_values("sort_key").drop(columns=["sort_key"])

    n = len(plot_df)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.9 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, plot_df.iterrows()):
        tid = row["template_id"]
        if row["template_family"] == "flat":
            ax.text(0.5, 0.60, "FLAT", ha="center", va="center", fontsize=18)
            ax.text(0.5, 0.43, "all prices equal", ha="center", va="center", fontsize=11)
            ax.text(
                0.5,
                0.25,
                f"share={format_pct(row['share_of_all'])}\n"
                f"n={int(row['sample_count']):,}",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("FLAT template family", fontsize=10)
            continue

        center = row[SHAPE_COLS].to_numpy(float)
        ax.plot(GRID, center, linestyle="--", linewidth=2.2, label="centroid")

        g = rep_df[rep_df["template_id"] == tid]
        block = g[g["representative_mode"] == "block"]
        sloped = g[g["representative_mode"] == "sloped"]

        if not block.empty:
            br = block.iloc[0]
            y = br[SHAPE_COLS].to_numpy(float)
            ax.step(
                GRID,
                y,
                where="post",
                linewidth=1.8,
                label=f"real block (MAE={br['representative_shape_mae']:.3f})",
            )
        if not sloped.empty:
            sr = sloped.iloc[0]
            y = sr[SHAPE_COLS].to_numpy(float)
            ax.plot(
                GRID,
                y,
                linewidth=1.8,
                label=f"real sloped (MAE={sr['representative_shape_mae']:.3f})",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.set_title(
            f"{tid}  all={format_pct(row['share_of_all'])}\n"
            f"block={format_pct(row['block_share'])}  "
            f"sloped={format_pct(row['sloped_share'])}",
            fontsize=10,
        )
        ax.set_xlabel("Normalized quantity x")
        ax.set_ylabel("Normalized price shape")
        ax.legend(fontsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Bid-curve templates: centroid and real block/sloped representatives",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def plot_top_templates_overlay(df, out_file, top_n=8):
    shape_df = (
        df[df["template_family"] == "shape"]
        .sort_values("share_of_all", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for _, row in shape_df.iterrows():
        ax.plot(
            GRID,
            row[SHAPE_COLS].to_numpy(float),
            marker="o",
            markersize=3,
            label=f"{row['template_id']} ({row['share_of_all']*100:.2f}%)",
        )

    ax.set_title(f"Top {top_n} template centroids by share")
    ax.set_xlabel("Normalized quantity x")
    ax.set_ylabel("Normalized price shape")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def save_visual_summary(df, out_file):
    cols = [
        "template_id",
        "template_family",
        "sample_count",
        "share_of_all",
        "share_of_shape",
        "block_share",
        "sloped_share",
        "mean_shape_mae",
        "mean_price_mae",
    ]
    df.sort_values("share_of_all", ascending=False)[cols].to_csv(
        out_file,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--in-root", default="data/processed/bidtemplate")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--gallery-cols", type=int, default=4)
    args = parser.parse_args()

    base_dir = Path(args.in_root) / str(args.year) / "template_library"
    fig_dir = ensure_dir(base_dir / "figures")
    df, rep_df = load_inputs(base_dir)

    share_all = fig_dir / "template_share_all.png"
    share_shape = fig_dir / "template_share_shape_only.png"
    gallery = fig_dir / "template_gallery.png"
    overlay = fig_dir / "top_templates_overlay.png"
    summary = fig_dir / "template_visual_summary.csv"

    plot_template_share_all(df, share_all)
    plot_template_share_shape_only(df, share_shape)
    plot_template_gallery(df, rep_df, gallery, ncols=args.gallery_cols)
    plot_top_templates_overlay(df, overlay, top_n=args.top_n)
    save_visual_summary(df, summary)

    print("=" * 72)
    print("Mode-aware template visualization complete")
    print("=" * 72)
    print(f"Output figures: {fig_dir}")
    print(f" - {share_all.name}")
    print(f" - {share_shape.name}")
    print(f" - {gallery.name}")
    print(f" - {overlay.name}")
    print(f" - {summary.name}")


if __name__ == "__main__":
    main()
