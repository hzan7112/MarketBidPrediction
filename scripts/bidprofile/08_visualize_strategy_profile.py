#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_visualize_strategy_profile.py

Compact visual diagnostics for the new historical multi-timescale bidprofile.

Figures:
1. subject-profile feature-group coverage
2. population median historical monthly bid-level delta
3. population median historical intraday bid-level delta
4. optional distribution of daily dominant-template switch rate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy_core import (
    LT_FEATURES,
    SHORT_TENDENCY_FEATURES,
    SEASONAL_SUMMARY_FEATURES,
    INTRADAY_SUMMARY_FEATURES,
    TRANSITION_FEATURES,
    ensure_dir,
)


def plot_group_coverage(
    profile,
    transition_available,
    out_file,
):
    groups = {
        "LT": LT_FEATURES,
        "Short tendency": SHORT_TENDENCY_FEATURES,
        "Seasonal": SEASONAL_SUMMARY_FEATURES,
        "Intraday": INTRADAY_SUMMARY_FEATURES,
    }

    if transition_available:
        groups[
            "Transition"
        ] = TRANSITION_FEATURES

    labels = []
    values = []

    for label, features in (
        groups.items()
    ):
        existing = [
            f
            for f in features
            if f in (
                profile.columns
            )
        ]

        if not existing:
            continue

        coverage = (
            profile[
                existing
            ]
            .notna()
            .mean()
            .median()
        )

        labels.append(
            label
        )

        values.append(
            100.0
            * float(
                coverage
            )
        )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    x = np.arange(
        len(
            labels
        )
    )

    ax.bar(
        x,
        values,
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Median non-missing share (%)"
    )

    ax.set_title(
        "Historical subject-profile feature-group coverage"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_month_delta(
    month_profile,
    out_file,
):
    if (
        "season_bid_level_delta"
        not in month_profile.columns
    ):
        return

    s = (
        month_profile
        .groupby(
            "month"
        )[
            "season_bid_level_delta"
        ]
        .median()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.plot(
        s.index,
        s.values,
        marker="o",
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        range(
            1,
            13,
        )
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Median historical bid-level delta"
    )

    ax.set_title(
        "Historical seasonal bid-level pattern"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_slot_delta(
    slot_profile,
    out_file,
):
    if (
        "intraday_bid_level_delta"
        not in slot_profile.columns
    ):
        return

    temp = (
        slot_profile.copy()
    )

    if (
        "local_hour"
        not in temp.columns
    ):
        temp[
            "local_hour"
        ] = (
            pd.to_numeric(
                temp[
                    "local_slot_seconds"
                ],
                errors="coerce",
            )
            // 3600
        )

    s = (
        temp.groupby(
            "local_hour"
        )[
            "intraday_bid_level_delta"
        ]
        .median()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.plot(
        s.index,
        s.values,
        marker="o",
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        range(
            0,
            24,
        )
    )

    ax.set_xlabel(
        "Local hour"
    )

    ax.set_ylabel(
        "Median historical bid-level delta"
    )

    ax.set_title(
        "Historical intraday bid-level pattern"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_transition(
    profile,
    out_file,
):
    col = (
        "transition_daily_dominant_switch_rate"
    )

    if (
        col
        not in profile.columns
    ):
        return

    x = (
        pd.to_numeric(
            profile[
                col
            ],
            errors="coerce",
        )
        .dropna()
    )

    if x.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    ax.hist(
        x,
        bins=30,
    )

    ax.set_xlabel(
        "Daily dominant-template switch rate"
    )

    ax.set_ylabel(
        "Participants"
    )

    ax.set_title(
        "Historical strategy-transition tendency"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    p.add_argument(
        "--root",
        default=(
            "data/processed/"
            "final_clean"
        ),
    )

    p.add_argument(
        "--results-root",
        default=(
            "results/"
            "bidprofile_visualization"
        ),
    )

    args = p.parse_args()

    root = Path(
        args.root
    )

    profile_dir = (
        root
        / "strategy_profile"
        / str(
            args.year
        )
    )

    context_dir = (
        root
        / "context_profile"
        / str(
            args.year
        )
    )

    profile_file = (
        profile_dir
        / f"participant_strategy_profile_{args.year}.csv"
    )

    manifest_file = (
        profile_dir
        / f"strategy_profile_manifest_{args.year}.json"
    )

    month_file = (
        context_dir
        / f"participant_month_strategy_profile_{args.year}.csv"
    )

    slot_file = (
        context_dir
        / f"participant_slot_strategy_profile_{args.year}.csv"
    )

    for f in (
        profile_file,
        manifest_file,
        month_file,
        slot_file,
    ):
        if not f.exists():
            raise FileNotFoundError(
                f
            )

    profile = pd.read_csv(
        profile_file,
        low_memory=False,
    )

    month_profile = pd.read_csv(
        month_file,
        low_memory=False,
    )

    slot_profile = pd.read_csv(
        slot_file,
        low_memory=False,
    )

    manifest = json.loads(
        manifest_file.read_text(
            encoding="utf-8"
        )
    )

    out_dir = ensure_dir(
        Path(
            args.results_root
        )
        / str(
            args.year
        )
    )

    plot_group_coverage(
        profile,
        bool(
            manifest.get(
                "transition_available",
                False,
            )
        ),
        out_dir
        / "01_profile_group_coverage.png",
    )

    plot_month_delta(
        month_profile,
        out_dir
        / "02_historical_month_bid_level_delta.png",
    )

    plot_slot_delta(
        slot_profile,
        out_dir
        / "03_historical_intraday_bid_level_delta.png",
    )

    plot_transition(
        profile,
        out_dir
        / "04_template_transition_switch_rate.png",
    )

    print(
        f"Done: {out_dir}"
    )


if __name__ == "__main__":
    main()
