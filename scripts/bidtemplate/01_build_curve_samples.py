#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_build_curve_samples.py

Stage 2: Bid-template library
PJM raw Energy Market Offers -> market-independent curve samples.

IMPORTANT
---------
This script does NOT use bid-profile features.

The 9 LT + 9 ST + 8 Break profile describes:
    "what kind of participant is this / how is it changing?"

This script describes:
    "what does the actual bid curve at this interval look like?"

The two data streams will only be joined later when predicting template choice.

For every historical interval, the curve is converted to:

    quantity:
        x = (q - q0) / (q1 - q0), x in [0, 1]

    non-flat price shape:
        shape(x) = [P(x) - P(0)] / [P(1) - P(0)]

sampled on:
        x = 0, 0.05, ..., 1.00

The information intentionally removed by normalization is retained separately:
    q_anchor_mw = q0
    q_span_mw   = q1 - q0
    p_anchor    = P(0)
    p_span      = P(1) - P(0)

Therefore a template T_k(x) can later reconstruct a curve as:

    q(x) = q_anchor_mw + q_span_mw * x
    P(x) = p_anchor + p_span * T_k(x)

Flat curves are NOT forced through the normalization formula.
They form an explicit "flat" family and are separated before shape clustering.

Outputs
-------
data/processed/bidtemplate/<year>/curve_samples/
    curve_samples_<source_file_stem>.csv

data/processed/bidtemplate/<year>/
    curve_samples_manifest_<year>.csv

The next step should cluster only rows with:
    template_family == "shape"
    shape_cluster_eligible_flag == 1

Rows with:
    template_family == "flat"
are an explicit flat template family and do not enter KMeans/K-Medoids.

Invalid/degenerate rows are counted in the manifest.
They are omitted from the sample files by default.
Use --keep-invalid if an audit table is needed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
PRICE_EPS = 1e-9
GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_bool(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        return bool(v)

    s = str(v).strip().lower()
    return s in {
        "1", "true", "t", "yes", "y",
        "sloped", "slope",
    }


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
            f"Cannot find any of {candidates}. "
            f"Available columns: {list(columns)}"
        )
    return None


# ---------------------------------------------------------------------
# PJM column adapter
# ---------------------------------------------------------------------

def detect_columns(columns):
    unit_col = _pick(
        columns,
        [
            "unit_code",
            "unit",
            "resource_name",
            "resource",
            "generator_id",
        ],
    )

    utc_col = _pick(
        columns,
        [
            "bid_datetime_beginning_utc",
            "datetime_beginning_utc",
            "timestamp_utc",
            "datetime_utc",
            "utc_timestamp",
        ],
        required=False,
    )

    local_col = _pick(
        columns,
        [
            "bid_datetime_beginning_ept",
            "datetime_beginning_ept",
            "timestamp_ept",
            "timestamp_local",
            "datetime_beginning_est",
            "local_timestamp",
        ],
        required=False,
    )

    if utc_col is None and local_col is None:
        raise KeyError(
            "Cannot find PJM bid timestamp column. "
            "Expected bid_datetime_beginning_utc and/or "
            "bid_datetime_beginning_ept."
        )

    slope_col = _pick(
        columns,
        [
            "bid_slope_flag",
            "usebidslope",
            "use_bid_slope",
            "slope_flag",
            "bid_slope",
        ],
        required=False,
    )

    lookup = {_canon(c): c for c in columns}

    mw_cols = []
    bid_cols = []

    # Support both 10-point and 20-point exports.
    for k in range(1, 21):
        mw = lookup.get(f"mw{k}")
        bid = lookup.get(f"bid{k}")

        if mw is not None and bid is not None:
            mw_cols.append(mw)
            bid_cols.append(bid)

    if not mw_cols:
        raise KeyError(
            "Cannot find paired MW/BID columns. "
            "Expected MW1...MW10/20 and BID1...BID10/20."
        )

    return (
        unit_col,
        utc_col,
        local_col,
        slope_col,
        mw_cols,
        bid_cols,
    )


# ---------------------------------------------------------------------
# Fast timestamp parsing
# ---------------------------------------------------------------------

def _detect_datetime_format(series):
    s = series.dropna().astype(str)

    if s.empty:
        return None

    sample = s.iloc[0].strip()

    patterns = [
        (
            r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[APap][Mm]",
            "%m/%d/%Y %I:%M:%S %p",
        ),
        (
            r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}",
            "%m/%d/%Y %H:%M:%S",
        ),
        (
            r"\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}:\d{2}",
            "%Y-%m-%d %H:%M:%S",
        ),
        (
            r"\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{2}:\d{2}",
            "%Y-%m-%dT%H:%M:%S",
        ),
    ]

    for pattern, fmt in patterns:
        if re.fullmatch(pattern, sample):
            return fmt

    return None


def parse_datetime_fast(series, utc=False):
    fmt = _detect_datetime_format(series)

    if fmt is not None:
        return pd.to_datetime(
            series,
            format=fmt,
            errors="coerce",
            utc=utc,
        )

    try:
        return pd.to_datetime(
            series,
            format="mixed",
            errors="coerce",
            utc=utc,
        )
    except (TypeError, ValueError):
        return pd.to_datetime(
            series,
            errors="coerce",
            utc=utc,
        )


def add_time_columns(df, utc_col, local_col):
    if utc_col is not None:
        utc = parse_datetime_fast(
            df[utc_col],
            utc=True,
        )
    else:
        local_tmp = parse_datetime_fast(
            df[local_col],
            utc=False,
        )

        utc = (
            local_tmp.dt.tz_localize(
                "America/New_York",
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
            .dt.tz_convert("UTC")
        )

    if local_col is not None:
        local = parse_datetime_fast(
            df[local_col],
            utc=False,
        )

        try:
            if local.dt.tz is not None:
                local = (
                    local.dt
                    .tz_convert("America/New_York")
                    .dt.tz_localize(None)
                )
        except AttributeError:
            pass
    else:
        local = (
            utc.dt
            .tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

    out = df.copy()

    out["_timestamp_utc"] = utc
    out["_timestamp_local"] = local
    out["_local_date"] = local.dt.normalize()
    out["_local_slot_seconds"] = (
        local.dt.hour * 3600
        + local.dt.minute * 60
        + local.dt.second
    )

    return out


# ---------------------------------------------------------------------
# Curve processing
# ---------------------------------------------------------------------

def clean_points(q, p):
    """
    Keep finite (q,p) pairs, sort by quantity, and collapse duplicate
    quantities by retaining the maximum price at that quantity.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    ok = np.isfinite(q) & np.isfinite(p)
    q = q[ok]
    p = p[ok]

    raw_valid_count = len(q)

    if raw_valid_count == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            0,
        )

    order = np.argsort(
        q,
        kind="mergesort",
    )
    q = q[order]
    p = p[order]

    uq = []
    up = []

    j = 0

    while j < len(q):
        k = j + 1

        while (
            k < len(q)
            and abs(q[k] - q[j]) <= EPS
        ):
            k += 1

        uq.append(q[j])
        up.append(
            np.nanmax(p[j:k])
        )

        j = k

    return (
        np.asarray(uq, dtype=float),
        np.asarray(up, dtype=float),
        raw_valid_count,
    )


def step_eval(xp, fp, x):
    """
    PJM block/step interpretation:
    price at a breakpoint is held until the next breakpoint.
    """
    idx = (
        np.searchsorted(
            xp,
            x,
            side="right",
        )
        - 1
    )

    idx = np.clip(
        idx,
        0,
        len(fp) - 1,
    )

    y = fp[idx].astype(float, copy=True)

    if len(fp):
        y[np.isclose(x, 1.0)] = fp[-1]

    return y


def build_curve_sample(q_raw, p_raw, sloped):
    """
    Convert one historical offer into a template-library sample.

    Returns a dictionary with:
    - reconstruction parameters;
    - QC flags;
    - 21-D normalized shape for non-flat eligible curves.
    """
    q, p, raw_valid_count = clean_points(
        q_raw,
        p_raw,
    )

    out = {
        "raw_valid_point_count": raw_valid_count,
        "clean_point_count": len(q),

        "q_anchor_mw": np.nan,
        "q_max_mw": np.nan,
        "q_span_mw": np.nan,

        "p_anchor": np.nan,
        "p_end": np.nan,
        "p_span": np.nan,
        "p_min": np.nan,
        "p_max": np.nan,
        "p_range": np.nan,

        "flat_curve_flag": 0,
        "monotone_nondecreasing_flag": np.nan,
        "price_decrease_count": np.nan,

        "template_family": "invalid",
        "template_eligible_flag": 0,
        "shape_cluster_eligible_flag": 0,
        "invalid_reason": "",
    }

    out.update(
        {c: np.nan for c in SHAPE_COLS}
    )

    if len(q) == 0:
        out["invalid_reason"] = "no_valid_points"
        return out

    out["q_anchor_mw"] = float(q[0])
    out["q_max_mw"] = float(q[-1])
    out["q_span_mw"] = float(q[-1] - q[0])

    out["p_anchor"] = float(p[0])
    out["p_end"] = float(p[-1])
    out["p_span"] = float(p[-1] - p[0])

    out["p_min"] = float(np.min(p))
    out["p_max"] = float(np.max(p))
    out["p_range"] = float(
        np.max(p) - np.min(p)
    )

    price_diff = np.diff(p)

    out["price_decrease_count"] = int(
        np.sum(
            price_diff < -PRICE_EPS
        )
    )

    out["monotone_nondecreasing_flag"] = int(
        out["price_decrease_count"] == 0
    )

    if len(q) < 2:
        out["invalid_reason"] = "single_clean_point"
        return out

    q_span = q[-1] - q[0]

    if q_span <= EPS:
        out["invalid_reason"] = "zero_quantity_span"
        return out

    # A truly flat curve means ALL prices are effectively equal.
    flat = (
        out["p_range"]
        <= PRICE_EPS
    )

    out["flat_curve_flag"] = int(flat)

    if flat:
        out["template_family"] = "flat"
        out["template_eligible_flag"] = 1
        out["shape_cluster_eligible_flag"] = 0
        return out

    # Current shape normalization requires different endpoint prices.
    # A non-flat curve with equal endpoints is treated as an anomalous/
    # non-monotone shape and is not forced into the shape-template space.
    p_span = p[-1] - p[0]

    if abs(p_span) <= PRICE_EPS:
        out["invalid_reason"] = (
            "nonflat_but_zero_endpoint_price_span"
        )
        return out

    x = (
        (q - q[0])
        / q_span
    )

    if sloped:
        pg = np.interp(
            GRID,
            x,
            p,
        )
    else:
        pg = step_eval(
            x,
            p,
            GRID,
        )

    p0 = float(pg[0])
    p1 = float(pg[-1])

    endpoint_span = p1 - p0

    if abs(endpoint_span) <= PRICE_EPS:
        out["invalid_reason"] = (
            "evaluated_zero_endpoint_price_span"
        )
        return out

    shape = (
        pg - p0
    ) / endpoint_span

    if not np.all(
        np.isfinite(shape)
    ):
        out["invalid_reason"] = "nonfinite_shape"
        return out

    out["template_family"] = "shape"
    out["template_eligible_flag"] = 1
    out["shape_cluster_eligible_flag"] = 1

    for c, v in zip(
        SHAPE_COLS,
        shape,
    ):
        out[c] = float(v)

    return out


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--raw-root",
        default="data/raw/energy_market_offers",
    )

    parser.add_argument(
        "--out-root",
        default="data/processed/bidtemplate",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help=(
            "Also write invalid/degenerate curves. "
            "Default: only template-eligible curves are written."
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional test mode: process only the first N source files.",
    )

    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional test mode: stop after N raw rows in each file.",
    )

    args = parser.parse_args()

    raw_dir = (
        Path(args.raw_root)
        / str(args.year)
    )

    files = sorted(
        raw_dir.glob("*.csv")
    )

    if args.max_files is not None:
        files = files[
            :args.max_files
        ]

    if not files:
        raise FileNotFoundError(
            f"No CSV files under {raw_dir}"
        )

    year_dir = ensure_dir(
        Path(args.out_root)
        / str(args.year)
    )

    sample_dir = ensure_dir(
        year_dir
        / "curve_samples"
    )

    manifest_rows = []

    total_raw = 0
    total_written = 0
    total_flat = 0
    total_shape = 0
    total_invalid = 0

    for file_no, file in enumerate(
        files,
        start=1,
    ):
        print()
        print(
            f"[file {file_no}/{len(files)}] "
            f"{file.name}",
            flush=True,
        )

        header = pd.read_csv(
            file,
            nrows=0,
        )

        (
            unit_col,
            utc_col,
            local_col,
            slope_col,
            mw_cols,
            bid_cols,
        ) = detect_columns(
            header.columns
        )

        print(
            "  [columns] "
            f"unit={unit_col}, "
            f"utc={utc_col}, "
            f"local={local_col}, "
            f"slope={slope_col}, "
            f"curve_points={len(mw_cols)}",
            flush=True,
        )

        usecols = (
            [unit_col]
            + mw_cols
            + bid_cols
        )

        for c in [
            utc_col,
            local_col,
            slope_col,
        ]:
            if (
                c is not None
                and c not in usecols
            ):
                usecols.append(c)

        out_file = (
            sample_dir
            / f"curve_samples_{file.stem}.csv"
        )

        if out_file.exists():
            out_file.unlink()

        wrote_header = False
        source_row_offset = 0

        file_raw = 0
        file_written = 0
        file_flat = 0
        file_shape = 0
        file_invalid = 0
        file_nondecreasing = 0

        reader = pd.read_csv(
            file,
            usecols=usecols,
            chunksize=args.chunksize,
            low_memory=False,
        )

        for chunk_no, chunk in enumerate(
            reader,
            start=1,
        ):
            if (
                args.max_rows_per_file
                is not None
            ):
                remaining = (
                    args.max_rows_per_file
                    - file_raw
                )

                if remaining <= 0:
                    break

                if len(chunk) > remaining:
                    chunk = chunk.iloc[
                        :remaining
                    ].copy()

            raw_n = len(chunk)

            if raw_n == 0:
                break

            print(
                f"  [chunk {chunk_no:03d}] "
                f"raw_rows={raw_n:,}",
                flush=True,
            )

            chunk = add_time_columns(
                chunk,
                utc_col,
                local_col,
            )

            qmat = (
                chunk[mw_cols]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .to_numpy(float)
            )

            pmat = (
                chunk[bid_cols]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .to_numpy(float)
            )

            units = (
                chunk[unit_col]
                .astype("string")
                .to_numpy()
            )

            if slope_col is not None:
                slopes = (
                    chunk[slope_col]
                    .map(parse_bool)
                    .to_numpy(bool)
                )
            else:
                slopes = np.zeros(
                    len(chunk),
                    dtype=bool,
                )

            utc_values = (
                chunk["_timestamp_utc"]
                .to_numpy()
            )

            local_values = (
                chunk["_timestamp_local"]
                .to_numpy()
            )

            date_values = (
                chunk["_local_date"]
                .to_numpy()
            )

            slot_values = (
                chunk["_local_slot_seconds"]
                .to_numpy()
            )

            rows = []

            chunk_flat = 0
            chunk_shape = 0
            chunk_invalid = 0
            chunk_nondecreasing = 0

            for j in range(
                len(chunk)
            ):
                feat = build_curve_sample(
                    qmat[j],
                    pmat[j],
                    bool(slopes[j]),
                )

                eligible = int(
                    feat[
                        "template_eligible_flag"
                    ]
                )

                if (
                    not eligible
                    and not args.keep_invalid
                ):
                    chunk_invalid += 1
                    continue

                source_row_index = (
                    source_row_offset
                    + j
                )

                row = {
                    "sample_id": (
                        f"{file.stem}:"
                        f"{source_row_index}"
                    ),
                    "participant_id": (
                        str(units[j])
                        if pd.notna(units[j])
                        else ""
                    ),
                    "timestamp_utc": (
                        utc_values[j]
                    ),
                    "timestamp_local": (
                        local_values[j]
                    ),
                    "local_date": (
                        date_values[j]
                    ),
                    "local_slot_seconds": (
                        slot_values[j]
                    ),
                    "source_market": "PJM",
                    "market_product": "ENERGY",
                    "source_file": file.name,
                    "source_row_index": (
                        source_row_index
                    ),
                    "curve_mode": (
                        "sloped"
                        if slopes[j]
                        else "block"
                    ),
                }

                row.update(feat)
                rows.append(row)

                if (
                    feat[
                        "template_family"
                    ]
                    == "flat"
                ):
                    chunk_flat += 1

                elif (
                    feat[
                        "template_family"
                    ]
                    == "shape"
                ):
                    chunk_shape += 1

                else:
                    chunk_invalid += 1

                if (
                    feat[
                        "monotone_nondecreasing_flag"
                    ]
                    == 1
                ):
                    chunk_nondecreasing += 1

            source_row_offset += raw_n

            # Invalid rows omitted by default still need to be counted.
            if not args.keep_invalid:
                kept_n = (
                    chunk_flat
                    + chunk_shape
                )

                omitted_invalid = (
                    raw_n
                    - kept_n
                )

                chunk_invalid = (
                    omitted_invalid
                )

            frame = pd.DataFrame(
                rows
            )

            if not frame.empty:
                frame.to_csv(
                    out_file,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                    encoding="utf-8-sig",
                    float_format="%.10g",
                )

                wrote_header = True

            file_raw += raw_n
            file_written += len(frame)
            file_flat += chunk_flat
            file_shape += chunk_shape
            file_invalid += chunk_invalid
            file_nondecreasing += (
                chunk_nondecreasing
            )

            print(
                f"    written={len(frame):,}, "
                f"shape={chunk_shape:,}, "
                f"flat={chunk_flat:,}, "
                f"invalid={chunk_invalid:,}",
                flush=True,
            )

            if (
                args.max_rows_per_file
                is not None
                and file_raw
                >= args.max_rows_per_file
            ):
                break

        valid_for_monotone = (
            file_flat
            + file_shape
        )

        manifest_rows.append(
            {
                "source_file": file.name,
                "raw_rows": file_raw,
                "written_rows": file_written,
                "shape_family_rows": file_shape,
                "flat_family_rows": file_flat,
                "invalid_rows": file_invalid,
                "shape_share_of_written": (
                    file_shape
                    / file_written
                    if file_written
                    else np.nan
                ),
                "flat_share_of_written": (
                    file_flat
                    / file_written
                    if file_written
                    else np.nan
                ),
                "nondecreasing_share_of_eligible": (
                    file_nondecreasing
                    / valid_for_monotone
                    if valid_for_monotone
                    else np.nan
                ),
                "output_file": str(
                    out_file
                ),
            }
        )

        total_raw += file_raw
        total_written += file_written
        total_flat += file_flat
        total_shape += file_shape
        total_invalid += file_invalid

        print(
            "  [file done] "
            f"raw={file_raw:,}, "
            f"written={file_written:,}, "
            f"shape={file_shape:,}, "
            f"flat={file_flat:,}, "
            f"invalid={file_invalid:,}",
            flush=True,
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_file = (
        year_dir
        / f"curve_samples_manifest_{args.year}.csv"
    )

    manifest.to_csv(
        manifest_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print(
        "Bid-template curve sample build complete"
    )
    print("=" * 72)
    print(
        f"Raw rows:      {total_raw:,}"
    )
    print(
        f"Written rows:  {total_written:,}"
    )
    print(
        f"Shape family:  {total_shape:,}"
    )
    print(
        f"Flat family:   {total_flat:,}"
    )
    print(
        f"Invalid rows:  {total_invalid:,}"
    )

    if total_written:
        print(
            "Shape share:   "
            f"{total_shape/total_written:.2%}"
        )
        print(
            "Flat share:    "
            f"{total_flat/total_written:.2%}"
        )

    print(
        f"Manifest: {manifest_file}"
    )
    print(
        f"Samples:  {sample_dir}"
    )


if __name__ == "__main__":
    main()
