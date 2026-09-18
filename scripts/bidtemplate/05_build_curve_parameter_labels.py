#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_build_curve_parameter_labels.py

Stage 2 -> Stage 3 bridge:
build interpretable supervision labels for complete bid-curve generation.

The current template library is kept unchanged:
    T00 ... T(K-1) + FLAT

For every template-assigned historical bid, this script adds structural labels:
    curve_mode              : flat / block / sloped
    effective_segment_count : number of effective curve segments
    breakpoint_count        : number of internal/step breakpoints
    breakpoint_x_json       : normalized breakpoint positions in [0, 1]

and retains scale labels:
    q_anchor_mw, q_span_mw, p_anchor, p_span

Definitions
-----------
FLAT:
    effective_segment_count = 1
    breakpoint_x_json = []

BLOCK:
    Consecutive equal-price blocks are merged.
    effective_segment_count = number of remaining price runs.
    breakpoint_x_json contains the normalized quantity at which each new
    price level starts. A terminal jump at x=1 is retained because it is part
    of the same step convention used by 01_build_curve_samples.py.

SLOPED:
    Each pair of adjacent distinct-MW clean points defines one linear segment.
    effective_segment_count = clean_point_count - 1
    breakpoint_x_json contains interior knot positions x[1:-1].

Inputs
------
data/raw/energy_market_offers/<year>/*.csv

data/processed/bidtemplate/<year>/template_library/assignments/
    template_assignments_curve_samples_<source_file_stem>.csv

Outputs
-------
data/processed/bidtemplate/<year>/parameter_labels/
    curve_parameter_labels_<source_file_stem>.csv

data/processed/bidtemplate/<year>/
    curve_parameter_labels_manifest_<year>.csv
    curve_parameter_structure_summary_<year>.csv

The output is intentionally compact. Variable-length breakpoint targets are
stored as compact JSON arrays. Padding/fixed-width targets should be created
later when a specific prediction model is chosen.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
PRICE_EPS = 1e-9


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canon(name):
    s = str(name).strip().lower()
    for ch in (" ", "-", "/", ".", "(", ")"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _pick(columns, candidates, required=True):
    lookup = {_canon(c): c for c in columns}
    for name in candidates:
        key = _canon(name)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(
            f"Cannot find any of {candidates}. Available columns: {list(columns)}"
        )
    return None


def parse_bool(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        return bool(v)
    return str(v).strip().lower() in {
        "1", "true", "t", "yes", "y", "sloped", "slope"
    }


def detect_curve_columns(columns):
    slope_col = _pick(
        columns,
        ["bid_slope_flag", "usebidslope", "use_bid_slope", "slope_flag", "bid_slope"],
        required=False,
    )

    lookup = {_canon(c): c for c in columns}
    mw_cols, bid_cols = [], []
    for k in range(1, 21):
        mw = lookup.get(f"mw{k}")
        bid = lookup.get(f"bid{k}")
        if mw is not None and bid is not None:
            mw_cols.append(mw)
            bid_cols.append(bid)

    if not mw_cols:
        raise KeyError("Cannot find paired MW/BID columns (MW1...MW10/20, BID1...BID10/20).")

    return slope_col, mw_cols, bid_cols


# ---------------------------------------------------------------------
# Same point cleaning rule as 01_build_curve_samples.py
# ---------------------------------------------------------------------

def clean_points(q, p):
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    ok = np.isfinite(q) & np.isfinite(p)
    q, p = q[ok], p[ok]
    raw_valid_count = len(q)

    if raw_valid_count == 0:
        return np.empty(0), np.empty(0), 0

    order = np.argsort(q, kind="mergesort")
    q, p = q[order], p[order]

    uq, up = [], []
    j = 0
    while j < len(q):
        k = j + 1
        while k < len(q) and abs(q[k] - q[j]) <= EPS:
            k += 1
        uq.append(q[j])
        up.append(np.nanmax(p[j:k]))
        j = k

    return np.asarray(uq, float), np.asarray(up, float), raw_valid_count


def compact_json(values) -> str:
    return json.dumps([round(float(v), 10) for v in values], separators=(",", ":"))


def extract_structure(q_raw, p_raw, source_mode: str, template_id: str):
    q, p, raw_valid_count = clean_points(q_raw, p_raw)
    clean_n = len(q)

    if clean_n == 0:
        raise ValueError("Assigned sample unexpectedly has no valid MW/BID points.")

    q0, q1 = float(q[0]), float(q[-1])
    q_span = q1 - q0
    if clean_n >= 2 and q_span > EPS:
        x = (q - q0) / q_span
    else:
        x = np.zeros(clean_n, dtype=float)

    # FLAT is treated as its own family regardless of the raw PJM slope flag.
    if str(template_id).upper() == "FLAT":
        return {
            "curve_mode": "flat",
            "source_curve_mode": source_mode,
            "raw_valid_point_count": raw_valid_count,
            "clean_point_count": clean_n,
            "price_change_count": 0,
            "effective_segment_count": 1,
            "breakpoint_count": 0,
            "breakpoint_x_json": "[]",
        }

    if clean_n < 2 or q_span <= EPS:
        raise ValueError("Assigned non-flat sample has fewer than two distinct-MW clean points.")

    if source_mode == "block":
        # Merge consecutive equal-price runs. The first run starts at x=0;
        # every later run start is a block breakpoint.
        run_starts = [0]
        for i in range(1, clean_n):
            if abs(float(p[i]) - float(p[i - 1])) > PRICE_EPS:
                run_starts.append(i)

        breakpoints = x[np.asarray(run_starts[1:], dtype=int)] if len(run_starts) > 1 else []
        price_change_count = len(run_starts) - 1
        segment_count = len(run_starts)

    elif source_mode == "sloped":
        # Every adjacent pair of clean knots defines a linear segment.
        breakpoints = x[1:-1]
        price_change_count = int(np.sum(np.abs(np.diff(p)) > PRICE_EPS))
        segment_count = clean_n - 1

    else:
        raise ValueError(f"Unsupported source curve mode: {source_mode}")

    return {
        "curve_mode": source_mode,
        "source_curve_mode": source_mode,
        "raw_valid_point_count": raw_valid_count,
        "clean_point_count": clean_n,
        "price_change_count": price_change_count,
        "effective_segment_count": int(segment_count),
        "breakpoint_count": int(len(breakpoints)),
        "breakpoint_x_json": compact_json(breakpoints),
    }


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--raw-root", default="data/raw/energy_market_offers")
    parser.add_argument("--processed-root", default="data/processed/bidtemplate")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    year_dir = Path(args.processed_root) / str(args.year)
    assignment_dir = year_dir / "template_library" / "assignments"
    out_dir = ensure_dir(year_dir / "parameter_labels")
    raw_dir = Path(args.raw_root) / str(args.year)

    assignment_files = sorted(assignment_dir.glob("template_assignments_*.csv"))
    if args.max_files is not None:
        assignment_files = assignment_files[:args.max_files]
    if not assignment_files:
        raise FileNotFoundError(f"No assignment CSV files under {assignment_dir}")

    assignment_cols = [
        "sample_id", "participant_id", "timestamp_utc", "timestamp_local",
        "local_date", "local_slot_seconds", "source_market", "market_product",
        "source_file", "source_row_index", "curve_mode", "template_family",
        "q_anchor_mw", "q_span_mw", "p_anchor", "p_span",
        "template_id", "template_cluster", "shape_mae", "shape_rmse",
        "price_mae", "price_rmse",
    ]

    manifest_rows = []
    structure_frames = []
    total_rows = total_flat = total_block = total_sloped = 0

    print("=" * 72)
    print("Build curve-parameter supervision labels")
    print("=" * 72)
    print(f"Assignment files: {len(assignment_files)}")

    for file_no, assignment_file in enumerate(assignment_files, start=1):
        print(f"\n[file {file_no}/{len(assignment_files)}] {assignment_file.name}", flush=True)

        header = pd.read_csv(assignment_file, nrows=0)
        missing = [c for c in assignment_cols if c not in header.columns]
        if missing:
            raise KeyError(f"{assignment_file.name} missing required columns: {missing}")

        assign = pd.read_csv(
            assignment_file,
            usecols=assignment_cols,
            low_memory=False,
        )
        if assign.empty:
            continue

        source_files = assign["source_file"].dropna().astype(str).unique()
        if len(source_files) != 1:
            raise ValueError(
                f"Expected exactly one source_file in {assignment_file.name}, got {source_files.tolist()}"
            )

        raw_file = raw_dir / source_files[0]
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw source file not found: {raw_file}")

        assign["source_row_index"] = pd.to_numeric(
            assign["source_row_index"], errors="raise"
        ).astype(np.int64)
        assign = assign.sort_values("source_row_index").reset_index(drop=True)
        if assign["source_row_index"].duplicated().any():
            raise ValueError(f"Duplicate source_row_index in {assignment_file.name}")

        aidx = assign["source_row_index"].to_numpy(np.int64)

        raw_header = pd.read_csv(raw_file, nrows=0)
        slope_col, mw_cols, bid_cols = detect_curve_columns(raw_header.columns)
        raw_usecols = mw_cols + bid_cols + ([slope_col] if slope_col is not None else [])

        out_file = out_dir / f"curve_parameter_labels_{raw_file.stem}.csv"
        if out_file.exists():
            out_file.unlink()
        wrote_header = False

        offset = 0
        matched = 0
        file_counts = {"flat": 0, "block": 0, "sloped": 0}
        segment_records = []

        for chunk in pd.read_csv(
            raw_file,
            usecols=raw_usecols,
            chunksize=args.chunksize,
            low_memory=False,
        ):
            n = len(chunk)
            lo = np.searchsorted(aidx, offset, side="left")
            hi = np.searchsorted(aidx, offset + n, side="left")

            if hi > lo:
                sub_assign = assign.iloc[lo:hi].copy().reset_index(drop=True)
                local_idx = aidx[lo:hi] - offset
                raw_sel = chunk.iloc[local_idx].reset_index(drop=True)

                qmat = raw_sel[mw_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                pmat = raw_sel[bid_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

                if slope_col is None:
                    raw_modes = np.full(len(raw_sel), "block", dtype=object)
                else:
                    raw_modes = np.where(
                        raw_sel[slope_col].map(parse_bool).to_numpy(bool),
                        "sloped",
                        "block",
                    )

                rows = []
                for j in range(len(sub_assign)):
                    a = sub_assign.iloc[j]
                    assigned_mode = str(a["curve_mode"]).strip().lower()
                    source_mode = str(raw_modes[j])

                    if assigned_mode not in {"block", "sloped"}:
                        raise ValueError(
                            f"Unexpected assignment curve_mode={assigned_mode} for sample {a['sample_id']}"
                        )
                    if assigned_mode != source_mode:
                        raise ValueError(
                            f"Curve-mode mismatch for {a['sample_id']}: assignment={assigned_mode}, raw={source_mode}"
                        )

                    struct = extract_structure(
                        qmat[j], pmat[j], source_mode, str(a["template_id"])
                    )

                    row = {
                        "sample_id": a["sample_id"],
                        "participant_id": a["participant_id"],
                        "timestamp_utc": a["timestamp_utc"],
                        "timestamp_local": a["timestamp_local"],
                        "local_date": a["local_date"],
                        "local_slot_seconds": a["local_slot_seconds"],
                        "source_market": a["source_market"],
                        "market_product": a["market_product"],
                        "source_file": a["source_file"],
                        "source_row_index": int(a["source_row_index"]),
                        "template_id": a["template_id"],
                        "template_family": a["template_family"],
                        "template_cluster": a["template_cluster"],
                        **struct,
                        "q_anchor_mw": a["q_anchor_mw"],
                        "q_span_mw": a["q_span_mw"],
                        "p_anchor": a["p_anchor"],
                        "p_span": a["p_span"],
                        "template_shape_mae": a["shape_mae"],
                        "template_shape_rmse": a["shape_rmse"],
                        "template_price_mae": a["price_mae"],
                        "template_price_rmse": a["price_rmse"],
                    }
                    rows.append(row)
                    file_counts[struct["curve_mode"]] += 1
                    segment_records.append(
                        (struct["curve_mode"], struct["effective_segment_count"])
                    )

                frame = pd.DataFrame(rows)
                frame.to_csv(
                    out_file,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                    encoding="utf-8-sig",
                    float_format="%.10g",
                )
                wrote_header = True
                matched += len(frame)

            offset += n

        if matched != len(assign):
            raise RuntimeError(
                f"Row alignment failed for {raw_file.name}: assignments={len(assign):,}, matched={matched:,}"
            )

        seg_df = pd.DataFrame(segment_records, columns=["curve_mode", "effective_segment_count"])
        if not seg_df.empty:
            structure_frames.append(seg_df)

        block_seg_mean = (
            seg_df.loc[seg_df["curve_mode"].eq("block"), "effective_segment_count"].mean()
            if not seg_df.empty else np.nan
        )
        sloped_seg_mean = (
            seg_df.loc[seg_df["curve_mode"].eq("sloped"), "effective_segment_count"].mean()
            if not seg_df.empty else np.nan
        )

        manifest_rows.append({
            "source_file": raw_file.name,
            "assignment_rows": len(assign),
            "written_rows": matched,
            "flat_rows": file_counts["flat"],
            "block_rows": file_counts["block"],
            "sloped_rows": file_counts["sloped"],
            "mean_block_segment_count": block_seg_mean,
            "mean_sloped_segment_count": sloped_seg_mean,
            "output_file": str(out_file),
        })

        total_rows += matched
        total_flat += file_counts["flat"]
        total_block += file_counts["block"]
        total_sloped += file_counts["sloped"]

        print(
            f"  written={matched:,}, flat={file_counts['flat']:,}, "
            f"block={file_counts['block']:,}, sloped={file_counts['sloped']:,}",
            flush=True,
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_file = year_dir / f"curve_parameter_labels_manifest_{args.year}.csv"
    manifest.to_csv(manifest_file, index=False, encoding="utf-8-sig")

    if structure_frames:
        all_struct = pd.concat(structure_frames, ignore_index=True)
        summary = (
            all_struct.groupby(["curve_mode", "effective_segment_count"], dropna=False)
            .size()
            .rename("sample_count")
            .reset_index()
        )
        summary["share_within_mode"] = summary.groupby("curve_mode")["sample_count"].transform(
            lambda s: s / s.sum()
        )
        summary["share_of_all"] = summary["sample_count"] / summary["sample_count"].sum()
    else:
        summary = pd.DataFrame(
            columns=[
                "curve_mode", "effective_segment_count", "sample_count",
                "share_within_mode", "share_of_all"
            ]
        )

    summary_file = year_dir / f"curve_parameter_structure_summary_{args.year}.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print()
    print("=" * 72)
    print("Curve-parameter label build complete")
    print("=" * 72)
    print(f"All labels:   {total_rows:,}")
    print(f"FLAT:         {total_flat:,} ({total_flat/total_rows:.2%})")
    print(f"BLOCK:        {total_block:,} ({total_block/total_rows:.2%})")
    print(f"SLOPED:       {total_sloped:,} ({total_sloped/total_rows:.2%})")
    print(f"Manifest:     {manifest_file}")
    print(f"Structure:    {summary_file}")
    print(f"Label files:  {out_dir}")


if __name__ == "__main__":
    main()
