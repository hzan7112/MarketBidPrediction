#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_build_profile_template_relation_dataset.py

Build datasets for validating the relation between the Stage-1 strategy profile
(9 LT + 9 ST + 8 Break) and the Stage-2 bid-curve templates.

Important
---------
This script does not use any auxiliary cluster labels from bidprofile.
Only the 26 interpretable profile/state features are joined to the template data.

Outputs
-------
data/processed/bidtemplate/<year>/profile_template_relation/
    profile_template_daily_<year>.csv
    profile_template_participant_<year>.csv
    profile_template_relation_manifest_<year>.csv

Daily table:
    one row per participant-day
    9 LT + 9 ST + 8 Break
    template counts/shares + dominant template

Participant table:
    one row per participant
    9 LT
    annual template counts/shares + dominant template
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LT_FEATURES = [
    "lt_bid_level",
    "lt_adjustment_magnitude",
    "lt_strategy_persistence",
    "lt_quantity_hhi",
    "lt_effective_segment_count",
    "lt_flat_curve_rate",
    "lt_tail_uplift_ratio",
    "lt_curve_bend_ratio",
    "lt_shape_variability",
]

ST_FEATURES = [
    "st_bid_level_z",
    "st_adjustment_bias_z",
    "st_adjustment_magnitude_z",
    "st_quantity_hhi_z",
    "st_effective_segment_count_z",
    "st_flat_curve_rate_z",
    "st_tail_uplift_ratio_z",
    "st_curve_bend_ratio_z",
    "st_shape_shift",
]

BREAK_FEATURES = [
    "break_bid_level",
    "break_adjustment_bias",
    "break_adjustment_magnitude",
    "break_quantity_hhi",
    "break_effective_segment_count",
    "break_flat_curve_rate",
    "break_tail_uplift_ratio",
    "break_curve_bend_ratio",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_pid(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def load_template_ids(library_file: Path) -> list[str]:
    if not library_file.exists():
        raise FileNotFoundError(library_file)
    lib = pd.read_csv(library_file, usecols=["template_id"])
    ids = lib["template_id"].astype(str).tolist()
    if not ids:
        raise ValueError("Template library is empty.")
    return ids


def aggregate_assignments(files: list[Path], chunksize: int) -> pd.DataFrame:
    pieces = []
    total_rows = 0

    for i, file in enumerate(files, start=1):
        print(f"[assignment {i}/{len(files)}] {file.name}", flush=True)
        file_rows = 0
        file_parts = []

        for chunk in pd.read_csv(
            file,
            usecols=["participant_id", "local_date", "template_id"],
            chunksize=chunksize,
            low_memory=False,
        ):
            chunk["participant_id"] = norm_pid(chunk["participant_id"])
            chunk["local_date"] = norm_date(chunk["local_date"])
            chunk["template_id"] = chunk["template_id"].astype("string").str.strip()

            valid = (
                chunk["participant_id"].notna()
                & chunk["local_date"].notna()
                & chunk["template_id"].notna()
            )
            chunk = chunk.loc[valid]
            file_rows += len(chunk)

            if len(chunk):
                g = (
                    chunk.groupby(
                        ["participant_id", "local_date", "template_id"],
                        observed=True,
                        sort=False,
                    )
                    .size()
                    .rename("template_count")
                    .reset_index()
                )
                file_parts.append(g)

        if file_parts:
            fg = pd.concat(file_parts, ignore_index=True)
            fg = (
                fg.groupby(
                    ["participant_id", "local_date", "template_id"],
                    observed=True,
                    sort=False,
                )["template_count"]
                .sum()
                .reset_index()
            )
            pieces.append(fg)

        total_rows += file_rows
        print(f"  valid assignment rows={file_rows:,}", flush=True)

    if not pieces:
        raise ValueError("No valid template assignments found.")

    counts = pd.concat(pieces, ignore_index=True)
    counts = (
        counts.groupby(
            ["participant_id", "local_date", "template_id"],
            observed=True,
            sort=False,
        )["template_count"]
        .sum()
        .reset_index()
    )

    print(f"Total assignment rows aggregated: {total_rows:,}")
    print(f"Participant-day-template cells:   {len(counts):,}")
    return counts


def counts_to_wide(
    counts: pd.DataFrame,
    template_ids: list[str],
    group_cols: list[str],
) -> pd.DataFrame:
    agg = (
        counts.groupby(group_cols + ["template_id"], observed=True)["template_count"]
        .sum()
        .reset_index()
    )

    wide = agg.pivot_table(
        index=group_cols,
        columns="template_id",
        values="template_count",
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )

    for tid in template_ids:
        if tid not in wide.columns:
            wide[tid] = 0

    wide = wide[template_ids].reset_index()

    count_cols = []
    share_cols = []
    for tid in template_ids:
        c = f"template_count_{tid}"
        s = f"template_share_{tid}"
        wide = wide.rename(columns={tid: c})
        count_cols.append(c)
        share_cols.append(s)

    wide["template_observation_count"] = wide[count_cols].sum(axis=1).astype(int)

    denom = wide["template_observation_count"].replace(0, np.nan)
    for tid, c, s in zip(template_ids, count_cols, share_cols):
        wide[s] = wide[c] / denom

    count_mat = wide[count_cols].to_numpy(float)
    share_mat = wide[share_cols].to_numpy(float)

    max_idx = np.argmax(count_mat, axis=1)
    wide["dominant_template_id"] = np.asarray(template_ids, dtype=object)[max_idx]
    wide["dominant_template_count"] = count_mat[np.arange(len(wide)), max_idx].astype(int)
    wide["dominant_template_share"] = (
        wide["dominant_template_count"] / wide["template_observation_count"]
    )
    wide["distinct_template_count"] = (count_mat > 0).sum(axis=1).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(share_mat > 0, np.log(share_mat), 0.0)
        entropy = -np.sum(np.where(share_mat > 0, share_mat * logp, 0.0), axis=1)
    if len(template_ids) > 1:
        entropy = entropy / np.log(len(template_ids))
    wide["template_entropy_norm"] = entropy

    return wide


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--assignment-root",
        default="data/processed/bidtemplate",
    )
    p.add_argument(
        "--strategy-profile-root",
        default="data/processed/final_clean/strategy_profile",
    )
    p.add_argument(
        "--short-term-root",
        default="data/processed/final_clean/short_term",
    )
    p.add_argument(
        "--out-root",
        default="data/processed/bidtemplate",
    )
    p.add_argument("--chunksize", type=int, default=500_000)
    args = p.parse_args()

    year = args.year

    template_dir = (
        Path(args.assignment_root) / str(year) / "template_library"
    )
    assignment_dir = template_dir / "assignments"
    assignment_files = sorted(assignment_dir.glob("template_assignments_*.csv"))
    if not assignment_files:
        raise FileNotFoundError(f"No template assignments under {assignment_dir}")

    template_ids = load_template_ids(template_dir / "curve_template_library.csv")
    print(f"Template IDs ({len(template_ids)}): {template_ids}")

    lt_file = (
        Path(args.strategy_profile_root)
        / str(year)
        / f"participant_strategy_profile_{year}.csv"
    )
    st_file = (
        Path(args.short_term_root)
        / str(year)
        / f"short_term_strategy_state_{year}.csv"
    )
    if not lt_file.exists():
        raise FileNotFoundError(lt_file)
    if not st_file.exists():
        raise FileNotFoundError(st_file)

    counts = aggregate_assignments(assignment_files, args.chunksize)

    # Daily template distribution.
    daily = counts_to_wide(
        counts,
        template_ids,
        group_cols=["participant_id", "local_date"],
    )

    # Participant-level annual template preference.
    participant_counts = (
        counts.groupby(["participant_id", "template_id"], observed=True)["template_count"]
        .sum()
        .reset_index()
    )
    participant = counts_to_wide(
        participant_counts,
        template_ids,
        group_cols=["participant_id"],
    )
    active_days = (
        daily.groupby("participant_id", observed=True)["local_date"]
        .nunique()
        .rename("template_active_days")
        .reset_index()
    )
    participant = participant.merge(active_days, on="participant_id", how="left")

    # Load exactly the 9 LT features from the final profile body.
    lt_header = pd.read_csv(lt_file, nrows=0).columns.tolist()
    lt_required = ["participant_id", "lt_ready_flag"] + LT_FEATURES
    missing_lt = [c for c in lt_required if c not in lt_header]
    if missing_lt:
        raise KeyError(f"LT profile missing columns: {missing_lt}")

    lt = pd.read_csv(lt_file, usecols=lt_required, low_memory=False)
    lt["participant_id"] = norm_pid(lt["participant_id"])
    lt = lt.drop_duplicates("participant_id", keep="last")

    # Load exactly the 9 ST + 8 Break features.
    st_header = pd.read_csv(st_file, nrows=0).columns.tolist()
    st_required = ["participant_id", "local_date", "st_ready_flag"] + ST_FEATURES + BREAK_FEATURES
    missing_st = [c for c in st_required if c not in st_header]
    if missing_st:
        raise KeyError(f"ST state missing columns: {missing_st}")

    st = pd.read_csv(st_file, usecols=st_required, low_memory=False)
    st["participant_id"] = norm_pid(st["participant_id"])
    st["local_date"] = norm_date(st["local_date"])
    st = st.drop_duplicates(["participant_id", "local_date"], keep="last")

    # Merge.
    daily = daily.merge(lt, on="participant_id", how="left", validate="many_to_one")
    daily = daily.merge(
        st,
        on=["participant_id", "local_date"],
        how="left",
        validate="one_to_one",
    )

    participant = participant.merge(
        lt,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    daily["lt_ready_flag"] = pd.to_numeric(daily["lt_ready_flag"], errors="coerce").fillna(0).astype(int)
    daily["st_ready_flag"] = pd.to_numeric(daily["st_ready_flag"], errors="coerce").fillna(0).astype(int)
    daily["profile_ready_flag"] = (
        daily["lt_ready_flag"].eq(1) & daily["st_ready_flag"].eq(1)
    ).astype(int)

    participant["lt_ready_flag"] = pd.to_numeric(
        participant["lt_ready_flag"], errors="coerce"
    ).fillna(0).astype(int)

    out_dir = ensure_dir(
        Path(args.out_root) / str(year) / "profile_template_relation"
    )
    daily_file = out_dir / f"profile_template_daily_{year}.csv"
    participant_file = out_dir / f"profile_template_participant_{year}.csv"
    manifest_file = out_dir / f"profile_template_relation_manifest_{year}.csv"

    daily = daily.sort_values(["participant_id", "local_date"])
    participant = participant.sort_values("participant_id")

    daily.to_csv(daily_file, index=False, encoding="utf-8-sig", float_format="%.10g")
    participant.to_csv(participant_file, index=False, encoding="utf-8-sig", float_format="%.10g")

    manifest = pd.DataFrame(
        [
            {
                "year": year,
                "assignment_files": len(assignment_files),
                "template_count": len(template_ids),
                "participant_days": len(daily),
                "participants": len(participant),
                "profile_ready_days": int(daily["profile_ready_flag"].sum()),
                "lt_ready_participants": int(participant["lt_ready_flag"].sum()),
                "median_daily_dominant_share": float(daily["dominant_template_share"].median()),
                "median_daily_template_entropy": float(daily["template_entropy_norm"].median()),
                "daily_output": str(daily_file),
                "participant_output": str(participant_file),
            }
        ]
    )
    manifest.to_csv(manifest_file, index=False, encoding="utf-8-sig")

    print()
    print("=" * 72)
    print("Profile-template relation dataset complete")
    print("=" * 72)
    print(f"Participants:              {len(participant):,}")
    print(f"LT-ready participants:     {participant['lt_ready_flag'].sum():,}")
    print(f"Participant-days:          {len(daily):,}")
    print(f"Profile-ready days:        {daily['profile_ready_flag'].sum():,}")
    print(f"Median dominant share:     {daily['dominant_template_share'].median():.3f}")
    print(f"Median normalized entropy: {daily['template_entropy_norm'].median():.3f}")
    print(f"Daily:       {daily_file}")
    print(f"Participant: {participant_file}")
    print(f"Manifest:    {manifest_file}")


if __name__ == "__main__":
    main()
