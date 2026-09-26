#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04a_build_curve_parameter_dataset.py

Stage 3 / bidprediction
Build the leakage-controlled dataset for curve-parameter regression.

Targets
-------
q_anchor_mw
q_span_mw
p_anchor
p_span

The Stage2 curve-template label table is joined to the Stage3 prediction
dataset. The existing leakage-controlled features are preserved, and the
26 transition-strategy features (Z_tr) are added by participant_id/local_date.

Important separation
--------------------
Features:
    Z_base + Z_tr + M + U + H

Condition / label metadata:
    y_template_id
    y_curve_mode
    y_effective_segment_count

Regression targets:
    q_anchor_mw
    q_span_mw
    p_anchor
    p_span

No realized curve parameter is added to the feature role.

Default output
--------------
data/processed/bidprediction/<year>/curve_parameter_dataset/

    dataset_parts/
        curve_parameter_dataset_*.csv

    curve_parameter_dataset_schema_<year>.csv
    curve_parameter_dataset_manifest_<year>.csv
    curve_parameter_target_summary_<year>.csv
    curve_parameter_template_summary_<year>.csv
    curve_parameter_build_config_<year>.json
    summary.txt

Readiness
---------
parameter_ready_flag = 1 only when:
    prediction_ready_flag == 1
    AND a Stage2 parameter label is matched
    AND all 4 regression targets are finite
    AND q_span_mw > 0
    AND p_span >= 0
    AND Stage2 template_id == Stage3 y_template_id

The output keeps non-ready rows by default for auditing. 04b should train only
on parameter_ready_flag == 1. Use --ready-only if a compact ready-only dataset
is preferred.

Join strategy
-------------
Preferred:
    sample_id

Fallback:
    participant_id + timestamp_utc

Fallback 2:
    participant_id + local_date + local_slot_seconds

The label table is first partitioned into temporary monthly caches, so the
5M+ label rows never need to be loaded into memory at once.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

TARGET_COLUMNS = [
    "q_anchor_mw",
    "q_span_mw",
    "p_anchor",
    "p_span",
]

LABEL_METADATA = [
    "template_id",
    "template_family",
    "template_cluster",
    "curve_mode",
    "effective_segment_count",
    "breakpoint_count",
    "breakpoint_x_json",
    "template_shape_mae",
    "template_shape_rmse",
    "template_price_mae",
    "template_price_rmse",
    "source_file",
    "source_row_index",
    "source_market",
    "market_product",
]

AUDIT_COLUMNS = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
    "prediction_cutoff_utc",
    "prediction_ready_flag",
    "profile_ready_flag",
    "market_ready_flag",
    "unit_state_ready_flag",
    "y_template_id",
    "y_curve_mode",
    "y_effective_segment_count",
]

EXCLUDE_PREDICTION_FEATURES = {
    "rolling_lt_ready_flag",
    "st_ready_flag",
    "profile_ready_flag",
    "market_ready_flag",
    "unit_state_ready_flag",
    "prediction_ready_flag",
    "market_nonmissing_count",
    "rolling_lt_nonmissing_count",
    "hist_prev_available_flag",
}

QUALITY_COLUMNS = [
    "parameter_label_available_flag",
    "parameter_target_valid_flag",
    "parameter_template_match_flag",
    "parameter_ready_flag",
]

LABEL_PREFIX_RENAME = {
    "template_id": "label_template_id",
    "template_family": "label_template_family",
    "template_cluster": "label_template_cluster",
    "curve_mode": "label_curve_mode",
    "effective_segment_count": "label_effective_segment_count",
    "breakpoint_count": "label_breakpoint_count",
    "breakpoint_x_json": "label_breakpoint_x_json",
    "template_shape_mae": "label_template_shape_mae",
    "template_shape_rmse": "label_template_shape_rmse",
    "template_price_mae": "label_template_price_mae",
    "template_price_rmse": "label_template_price_rmse",
    "source_file": "label_source_file",
    "source_row_index": "label_source_row_index",
    "source_market": "label_source_market",
    "market_product": "label_market_product",
}


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def unique_list(cols):
    out = []
    seen = set()
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def month_from_path(path: Path) -> str:
    m = re.search(r"(\d{4})[_-](\d{2})", path.stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return path.stem


def normalize_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def normalize_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def normalize_utc_key(s: pd.Series) -> pd.Series:
    """
    Stable timestamp join key. Stored as UTC ISO second precision.
    """
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").astype("string")


def infer_month(df: pd.DataFrame) -> pd.Series:
    if "local_date" in df.columns:
        d = pd.to_datetime(df["local_date"], errors="coerce")
    elif "timestamp_local" in df.columns:
        d = pd.to_datetime(df["timestamp_local"], errors="coerce")
    elif "timestamp_utc" in df.columns:
        d = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    else:
        raise KeyError(
            "Label table needs one of local_date/timestamp_local/timestamp_utc "
            "for monthly partitioning."
        )
    return d.dt.strftime("%Y-%m").astype("string")


# =============================================================================
# Input discovery
# =============================================================================

def discover_prediction_parts(part_dir: Path) -> list[Path]:
    files = sorted(part_dir.glob("prediction_dataset_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No prediction dataset parts under {part_dir}"
        )
    return files


def discover_label_sources(
    bidtemplate_root: Path,
    year: int,
    user_path: str | None,
) -> list[Path]:
    """
    Prefer canonical single-file outputs. If --labels points to a directory,
    all matching CSV files in that directory are used.
    """
    if user_path:
        p = Path(user_path)

        if p.is_file():
            return [p]

        if p.is_dir():
            candidates = sorted(
                [
                    x for x in p.rglob("*.csv")
                    if "curve_parameter_labels" in x.name.lower()
                    or "parameter_labels" in x.name.lower()
                ]
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No curve parameter label CSV under {p}"
                )
            return candidates

        raise FileNotFoundError(p)

    base = bidtemplate_root / str(year)

    canonical = [
        base / f"curve_parameter_labels_{year}.csv",
        base / "curve_parameter_labels_offers.csv",
        base / "curve_parameter_labels.csv",
    ]

    for p in canonical:
        if p.exists():
            return [p]

    candidates = sorted(
        [
            x for x in base.rglob("*.csv")
            if (
                "curve_parameter_labels" in x.name.lower()
                or "parameter_labels" in x.name.lower()
            )
            and "manifest" not in x.name.lower()
            and "summary" not in x.name.lower()
        ]
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover Stage2 curve parameter labels under "
            f"{base}. Use --labels <file-or-directory>."
        )

    if len(candidates) == 1:
        return candidates

    # If they look like monthly/part files from the same directory, use all.
    parents = {x.parent.resolve() for x in candidates}
    month_like = all(
        re.search(r"\d{4}[_-]\d{2}", x.stem)
        for x in candidates
    )

    if len(parents) == 1 and month_like:
        return candidates

    # Otherwise avoid silently mixing versions.
    display = "\n".join(f"  - {x}" for x in candidates[:30])
    raise RuntimeError(
        "Multiple possible curve-parameter label tables were found. "
        "Specify the intended source with --labels.\n"
        + display
    )


def load_prediction_schema(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"column", "role", "feature_group"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"{path.name} missing schema columns: {sorted(missing)}"
        )

    return df


def load_transition_schema(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun 02b_build_transition_features.py first."
        )

    df = pd.read_csv(path)

    required = {"column", "role", "feature_group"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"{path.name} missing schema columns: {sorted(missing)}"
        )

    return df


# =============================================================================
# Feature definitions
# =============================================================================

def get_feature_columns(
    prediction_schema: pd.DataFrame,
    transition_schema: pd.DataFrame,
):
    pred = prediction_schema[
        prediction_schema["role"].astype(str).eq("feature")
    ].copy()

    pred_features = [
        c
        for c in pred["column"].astype(str).tolist()
        if c not in EXCLUDE_PREDICTION_FEATURES
    ]

    pred_group = dict(
        zip(
            pred["column"].astype(str),
            pred["feature_group"].astype(str),
        )
    )

    tr = transition_schema[
        transition_schema["role"].astype(str).eq("feature")
    ].copy()

    tr_features = tr["column"].astype(str).tolist()

    if not tr_features:
        raise ValueError(
            "No Z_tr features found in transition strategy schema."
        )

    tr_group = dict(
        zip(
            tr["column"].astype(str),
            tr["feature_group"].astype(str),
        )
    )

    return (
        unique_list(pred_features),
        unique_list(tr_features),
        pred_group,
        tr_group,
    )


def load_transition_profile(
    path: Path,
    tr_features: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun 02b_build_transition_features.py first."
        )

    needed = [
        "participant_id",
        "local_date",
    ] + tr_features

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in needed if c not in header]

    if missing:
        raise KeyError(
            f"{path.name} missing columns: {missing}"
        )

    df = pd.read_csv(
        path,
        usecols=needed,
        low_memory=False,
    )

    df["participant_id"] = normalize_id(df["participant_id"])
    df["local_date"] = normalize_date(df["local_date"])

    if df[["participant_id", "local_date"]].duplicated().any():
        raise ValueError(
            "transition_strategy_profile contains duplicate "
            "participant_id/local_date rows."
        )

    for c in tr_features:
        df[c] = safe_numeric(df[c]).astype("float32")

    return df


# =============================================================================
# Join-key resolution
# =============================================================================

def resolve_join_keys(
    prediction_header: list[str],
    label_header: list[str],
):
    p = set(prediction_header)
    l = set(label_header)

    if "sample_id" in p and "sample_id" in l:
        return ["sample_id"], "sample_id"

    if {
        "participant_id",
        "timestamp_utc",
    }.issubset(p) and {
        "participant_id",
        "timestamp_utc",
    }.issubset(l):
        return [
            "participant_id",
            "timestamp_utc",
        ], "participant_id+timestamp_utc"

    if {
        "participant_id",
        "local_date",
        "local_slot_seconds",
    }.issubset(p) and {
        "participant_id",
        "local_date",
        "local_slot_seconds",
    }.issubset(l):
        return [
            "participant_id",
            "local_date",
            "local_slot_seconds",
        ], "participant_id+local_date+local_slot_seconds"

    raise KeyError(
        "Cannot resolve a safe join key between prediction dataset and "
        "curve parameter labels. Need sample_id, or participant_id+timestamp_utc, "
        "or participant_id+local_date+local_slot_seconds."
    )


def normalize_join_columns(
    df: pd.DataFrame,
    join_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()

    for c in join_cols:
        if c == "sample_id":
            out[c] = normalize_id(out[c])

        elif c == "participant_id":
            out[c] = normalize_id(out[c])

        elif c == "timestamp_utc":
            out[c] = normalize_utc_key(out[c])

        elif c == "local_date":
            out[c] = normalize_date(out[c])

        elif c == "local_slot_seconds":
            out[c] = safe_numeric(out[c]).astype("Int64")

    return out


# =============================================================================
# Label monthly cache
# =============================================================================

def prepare_label_cache(
    label_sources: list[Path],
    cache_dir: Path,
    join_cols: list[str],
    chunksize: int,
    rebuild: bool,
):
    """
    Partition Stage2 parameter labels by local month.

    Returns:
        month -> cache csv path
        source row count
    """
    if rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)

    ensure_dir(cache_dir)

    existing = sorted(
        cache_dir.glob("curve_parameter_labels_*.csv")
    )

    manifest_file = (
        cache_dir
        / "_cache_manifest.json"
    )

    if existing and manifest_file.exists() and not rebuild:
        manifest = json.loads(
            manifest_file.read_text(
                encoding="utf-8"
            )
        )

        if manifest.get("join_cols") == join_cols:
            return (
                {
                    p.stem.replace(
                        "curve_parameter_labels_",
                        "",
                    ): p
                    for p in existing
                },
                int(
                    manifest.get(
                        "source_rows",
                        0,
                    )
                ),
            )

    for p in cache_dir.glob(
        "curve_parameter_labels_*.csv"
    ):
        p.unlink()

    needed = unique_list(
        join_cols
        + [
            "participant_id",
            "timestamp_utc",
            "timestamp_local",
            "local_date",
            "local_slot_seconds",
        ]
        + LABEL_METADATA
        + TARGET_COLUMNS
    )

    written_header = set()
    source_rows = 0
    month_rows = Counter()

    for source_no, source in enumerate(
        label_sources,
        1,
    ):
        header = pd.read_csv(
            source,
            nrows=0,
        ).columns.tolist()

        required = unique_list(
            join_cols
            + TARGET_COLUMNS
            + ["template_id"]
        )

        missing = [
            c for c in required
            if c not in header
        ]

        if missing:
            raise KeyError(
                f"{source.name} missing required label columns: {missing}"
            )

        usecols = [
            c for c in needed
            if c in header
        ]

        print(
            f"[label-cache {source_no}/{len(label_sources)}] "
            f"{source}",
            flush=True,
        )

        for chunk in pd.read_csv(
            source,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ):
            source_rows += len(chunk)

            month = infer_month(
                chunk
            )

            chunk = normalize_join_columns(
                chunk,
                join_cols,
            )

            # Keep only the actual join keys plus label-side fields in the
            # monthly cache.  With sample_id as the join key, Stage2 labels
            # also contain participant_id/local_date/timestamps.  Keeping
            # those duplicate identity columns would make pandas rename the
            # prediction-side fields to *_x/*_y after merge and later remove
            # the canonical local_date column.
            cache_keep = unique_list(
                join_cols
                + [c for c in LABEL_METADATA if c in chunk.columns]
                + [c for c in TARGET_COLUMNS if c in chunk.columns]
            )
            chunk = chunk[cache_keep].copy()

            for m in month.dropna().unique():
                mask = month.eq(m)

                sub = chunk.loc[
                    mask
                ].copy()

                if sub.empty:
                    continue

                # Prefix label-only metadata to avoid collisions after join.
                sub = sub.rename(
                    columns=LABEL_PREFIX_RENAME
                )

                path = (
                    cache_dir
                    / f"curve_parameter_labels_{m}.csv"
                )

                write_header = (
                    str(path)
                    not in written_header
                    and not path.exists()
                )

                sub.to_csv(
                    path,
                    mode="w" if write_header else "a",
                    header=write_header,
                    index=False,
                    encoding="utf-8-sig",
                )

                written_header.add(
                    str(path)
                )

                month_rows[
                    str(m)
                ] += len(sub)

    cache_files = sorted(
        cache_dir.glob(
            "curve_parameter_labels_*.csv"
        )
    )

    if not cache_files:
        raise RuntimeError(
            "No monthly label cache was produced."
        )

    manifest = {
        "join_cols": join_cols,
        "source_rows": source_rows,
        "label_sources": [
            str(x)
            for x in label_sources
        ],
        "month_rows": dict(
            sorted(
                month_rows.items()
            )
        ),
    }

    manifest_file.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        {
            p.stem.replace(
                "curve_parameter_labels_",
                "",
            ): p
            for p in cache_files
        },
        source_rows,
    )


class MonthlyLabelLoader:
    def __init__(
        self,
        cache_files: dict[str, Path],
        join_cols: list[str],
        max_cached_months: int = 2,
    ):
        self.cache_files = cache_files
        self.join_cols = join_cols
        self.max_cached_months = max_cached_months
        self.cache: dict[str, pd.DataFrame] = {}
        self.order: list[str] = []

    def get(
        self,
        month: str,
    ) -> pd.DataFrame:
        if month in self.cache:
            return self.cache[month]

        path = self.cache_files.get(month)

        if path is None:
            return pd.DataFrame(
                columns=self.join_cols
            )

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        df = normalize_join_columns(
            df,
            self.join_cols,
        )

        dup = df.duplicated(
            self.join_cols,
            keep=False,
        )

        if dup.any():
            examples = (
                df.loc[
                    dup,
                    self.join_cols,
                ]
                .head(10)
                .to_dict("records")
            )

            raise ValueError(
                f"Stage2 parameter labels contain duplicate join keys "
                f"in month {month}. Examples: {examples}"
            )

        self.cache[
            month
        ] = df

        self.order.append(
            month
        )

        while len(
            self.order
        ) > self.max_cached_months:
            old = self.order.pop(0)
            self.cache.pop(
                old,
                None,
            )

        return df


# =============================================================================
# Dataset processing
# =============================================================================

def prediction_month_series(
    df: pd.DataFrame,
) -> pd.Series:
    if "local_date" in df.columns:
        d = pd.to_datetime(
            df["local_date"],
            errors="coerce",
        )
    elif "timestamp_local" in df.columns:
        d = pd.to_datetime(
            df["timestamp_local"],
            errors="coerce",
        )
    elif "timestamp_utc" in df.columns:
        d = pd.to_datetime(
            df["timestamp_utc"],
            errors="coerce",
            utc=True,
        )
    else:
        raise KeyError(
            "Prediction dataset needs local_date/timestamp_local/timestamp_utc."
        )

    return d.dt.strftime(
        "%Y-%m"
    ).astype("string")


def build_output_schema(
    output_columns: list[str],
    prediction_schema: pd.DataFrame,
    transition_schema: pd.DataFrame,
    join_cols: list[str],
):
    pred_role = dict(
        zip(
            prediction_schema[
                "column"
            ].astype(str),
            prediction_schema[
                "role"
            ].astype(str),
        )
    )

    pred_group = dict(
        zip(
            prediction_schema[
                "column"
            ].astype(str),
            prediction_schema[
                "feature_group"
            ].astype(str),
        )
    )

    tr_role = dict(
        zip(
            transition_schema[
                "column"
            ].astype(str),
            transition_schema[
                "role"
            ].astype(str),
        )
    )

    tr_group = dict(
        zip(
            transition_schema[
                "column"
            ].astype(str),
            transition_schema[
                "feature_group"
            ].astype(str),
        )
    )

    rows = []

    for c in output_columns:
        if c in TARGET_COLUMNS:
            role = "target"
            group = "curve_parameter_target"
            source = "bidtemplate_curve_parameter_labels"
            leakage_use = "target_only"

        elif c in QUALITY_COLUMNS:
            role = "quality_flag"
            group = "curve_parameter_readiness"
            source = "04a_derived"
            leakage_use = "audit_only"

        elif c.startswith("label_"):
            role = "label_metadata"
            group = "curve_parameter_label_metadata"
            source = "bidtemplate_curve_parameter_labels"
            leakage_use = "evaluation_only"

        elif c in {
            "y_template_id",
            "y_curve_mode",
            "y_effective_segment_count",
        }:
            role = "label_metadata"
            group = "curve_template_label"
            source = "prediction_dataset"
            leakage_use = (
                "training_condition_or_evaluation; "
                "replace with predicted template in full-pipeline inference"
                if c == "y_template_id"
                else "evaluation_only"
            )

        elif c in tr_role:
            role = "feature"
            group = (
                tr_group.get(c)
                or "transition_strategy_profile"
            )
            source = "transition_strategy_profile"
            leakage_use = "feature"

        elif c in pred_role:
            original_role = pred_role.get(c, "")
            original_group = pred_group.get(c, "")

            if original_role == "feature":
                role = "feature"
                group = original_group
                leakage_use = "feature"
            else:
                role = original_role or "metadata"
                group = original_group or "prediction_dataset_metadata"
                leakage_use = "audit_only"

            source = "prediction_dataset"

        elif c in join_cols:
            role = "identifier"
            group = "join_key"
            source = "prediction_dataset"
            leakage_use = "audit_only"

        else:
            role = "metadata"
            group = "audit_metadata"
            source = "prediction_dataset_or_derived"
            leakage_use = "audit_only"

        rows.append(
            {
                "column": c,
                "role": role,
                "feature_group": group,
                "source": source,
                "leakage_use": leakage_use,
            }
        )

    return pd.DataFrame(
        rows
    )


def process_prediction_parts(
    prediction_parts: list[Path],
    output_part_dir: Path,
    pred_features: list[str],
    tr_features: list[str],
    transition_profile: pd.DataFrame,
    label_loader: MonthlyLabelLoader,
    join_cols: list[str],
    chunksize: int,
    ready_only: bool,
):
    ensure_dir(
        output_part_dir
    )

    totals = Counter()
    month_stats = defaultdict(Counter)
    target_sums = defaultdict(float)
    target_sq_sums = defaultdict(float)
    target_counts = Counter()
    target_mins = {
        c: np.inf
        for c in TARGET_COLUMNS
    }
    target_maxs = {
        c: -np.inf
        for c in TARGET_COLUMNS
    }
    template_stats = defaultdict(Counter)

    output_columns_seen = None
    manifest_rows = []

    for part_no, part in enumerate(
        prediction_parts,
        1,
    ):
        header = pd.read_csv(
            part,
            nrows=0,
        ).columns.tolist()

        required = unique_list(
            [
                "participant_id",
                "local_date",
                "prediction_ready_flag",
                "y_template_id",
            ]
            + join_cols
        )

        missing = [
            c for c in required
            if c not in header
        ]

        if missing:
            raise KeyError(
                f"{part.name} missing required prediction columns: {missing}"
            )

        audit_cols = [
            c
            for c in AUDIT_COLUMNS
            if c in header
        ]

        usecols = unique_list(
            join_cols
            + audit_cols
            + [
                c
                for c in pred_features
                if c in header
            ]
        )

        out_name = part.name.replace(
            "prediction_dataset_",
            "curve_parameter_dataset_",
            1,
        )

        out_file = (
            output_part_dir
            / out_name
        )

        if out_file.exists():
            out_file.unlink()

        write_header = True
        part_stats = Counter()

        print(
            f"[dataset {part_no}/{len(prediction_parts)}] "
            f"{part.name} -> {out_name}",
            flush=True,
        )

        for chunk in pd.read_csv(
            part,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ):
            original_rows = len(
                chunk
            )

            totals[
                "prediction_rows"
            ] += original_rows

            part_stats[
                "prediction_rows"
            ] += original_rows

            chunk = normalize_join_columns(
                chunk,
                join_cols,
            )

            chunk["participant_id"] = normalize_id(
                chunk["participant_id"]
            )

            chunk["local_date"] = normalize_date(
                chunk["local_date"]
            )

            pred_ready = safe_numeric(
                chunk["prediction_ready_flag"]
            ).fillna(0).eq(1)

            totals[
                "prediction_ready_rows"
            ] += int(
                pred_ready.sum()
            )

            part_stats[
                "prediction_ready_rows"
            ] += int(
                pred_ready.sum()
            )

            # Add Z_tr using only historical dates < target day.
            chunk = chunk.merge(
                transition_profile,
                on=[
                    "participant_id",
                    "local_date",
                ],
                how="left",
                validate="many_to_one",
                sort=False,
            )

            months = prediction_month_series(
                chunk
            )

            joined_blocks = []

            for month in months.dropna().unique():
                mask = months.eq(
                    month
                )

                sub = chunk.loc[
                    mask
                ].copy()

                labels = label_loader.get(
                    str(month)
                )

                if labels.empty:
                    joined = sub.copy()

                    # Create expected label columns as missing.
                    for c in TARGET_COLUMNS:
                        joined[c] = np.nan

                    for original, renamed in LABEL_PREFIX_RENAME.items():
                        joined[renamed] = pd.NA

                else:
                    # Old caches created by an interrupted earlier run may
                    # still contain duplicate prediction-side identity/time
                    # columns (e.g. local_date) when sample_id is the join key.
                    # Restrict the label frame here as well so rerunning does
                    # not require rebuilding the cache.
                    label_join_cols = unique_list(
                        join_cols
                        + [
                            c for c in labels.columns
                            if c.startswith("label_")
                        ]
                        + [
                            c for c in TARGET_COLUMNS
                            if c in labels.columns
                        ]
                    )
                    labels_for_join = labels[label_join_cols].copy()

                    joined = sub.merge(
                        labels_for_join,
                        on=join_cols,
                        how="left",
                        validate="many_to_one",
                        sort=False,
                    )

                joined_blocks.append(
                    joined
                )

            # Rows with invalid/unparseable month still need to remain auditable.
            missing_month = months.isna()

            if missing_month.any():
                sub = chunk.loc[
                    missing_month
                ].copy()

                for c in TARGET_COLUMNS:
                    sub[c] = np.nan

                for original, renamed in LABEL_PREFIX_RENAME.items():
                    sub[renamed] = pd.NA

                joined_blocks.append(
                    sub
                )

            joined = pd.concat(
                joined_blocks,
                axis=0,
            ).sort_index()

            # -----------------------------------------------------------------
            # Label/readiness checks.
            # -----------------------------------------------------------------

            available = pd.Series(
                True,
                index=joined.index,
            )

            for c in TARGET_COLUMNS:
                if c not in joined.columns:
                    joined[c] = np.nan
                available &= joined[c].notna()

            if "label_template_id" not in joined.columns:
                joined[
                    "label_template_id"
                ] = pd.NA

            label_available = joined[
                "label_template_id"
            ].notna()

            for c in TARGET_COLUMNS:
                label_available &= joined[c].notna()

            target_num = {}

            for c in TARGET_COLUMNS:
                target_num[c] = safe_numeric(
                    joined[c]
                )
                joined[c] = target_num[c]

            finite = pd.Series(
                True,
                index=joined.index,
            )

            for c in TARGET_COLUMNS:
                finite &= np.isfinite(
                    target_num[c].to_numpy(
                        dtype=float,
                        na_value=np.nan,
                    )
                )

            target_valid = (
                finite
                & target_num["q_span_mw"].gt(0)
                & target_num["p_span"].ge(0)
            )

            pred_template = (
                joined["y_template_id"]
                .astype("string")
                .str.strip()
            )

            label_template = (
                joined["label_template_id"]
                .astype("string")
                .str.strip()
            )

            template_match = (
                pred_template.notna()
                & label_template.notna()
                & pred_template.eq(
                    label_template
                )
            )

            pred_ready_joined = safe_numeric(
                joined["prediction_ready_flag"]
            ).fillna(0).eq(1)

            parameter_ready = (
                pred_ready_joined
                & label_available
                & target_valid
                & template_match
            )

            joined[
                "parameter_label_available_flag"
            ] = label_available.astype(
                "int8"
            )

            joined[
                "parameter_target_valid_flag"
            ] = target_valid.astype(
                "int8"
            )

            joined[
                "parameter_template_match_flag"
            ] = template_match.astype(
                "int8"
            )

            joined[
                "parameter_ready_flag"
            ] = parameter_ready.astype(
                "int8"
            )

            totals[
                "label_available_rows"
            ] += int(
                label_available.sum()
            )

            totals[
                "target_valid_rows"
            ] += int(
                target_valid.sum()
            )

            totals[
                "template_match_rows"
            ] += int(
                template_match.sum()
            )

            totals[
                "parameter_ready_rows"
            ] += int(
                parameter_ready.sum()
            )

            totals[
                "prediction_ready_missing_label"
            ] += int(
                (
                    pred_ready_joined
                    & ~label_available
                ).sum()
            )

            totals[
                "prediction_ready_template_mismatch"
            ] += int(
                (
                    pred_ready_joined
                    & label_available
                    & ~template_match
                ).sum()
            )

            totals[
                "prediction_ready_invalid_target"
            ] += int(
                (
                    pred_ready_joined
                    & label_available
                    & template_match
                    & ~target_valid
                ).sum()
            )

            part_stats[
                "label_available_rows"
            ] += int(
                label_available.sum()
            )

            part_stats[
                "parameter_ready_rows"
            ] += int(
                parameter_ready.sum()
            )

            # -----------------------------------------------------------------
            # Summaries only on parameter-ready rows.
            # -----------------------------------------------------------------

            ready = joined.loc[
                parameter_ready
            ]

            if not ready.empty:
                ready_month = (
                    ready["local_date"]
                    .dt.strftime(
                        "%Y-%m"
                    )
                )

                for m, n in ready_month.value_counts().items():
                    month_stats[
                        str(m)
                    ][
                        "parameter_ready_rows"
                    ] += int(
                        n
                    )

                for c in TARGET_COLUMNS:
                    x = safe_numeric(
                        ready[c]
                    ).dropna().astype(
                        float
                    )

                    if len(x):
                        target_counts[
                            c
                        ] += len(
                            x
                        )
                        target_sums[
                            c
                        ] += float(
                            x.sum()
                        )
                        target_sq_sums[
                            c
                        ] += float(
                            np.square(
                                x.to_numpy()
                            ).sum()
                        )
                        target_mins[
                            c
                        ] = min(
                            target_mins[c],
                            float(
                                x.min()
                            ),
                        )
                        target_maxs[
                            c
                        ] = max(
                            target_maxs[c],
                            float(
                                x.max()
                            ),
                        )

                for template_id, g in ready.groupby(
                    "y_template_id",
                    dropna=False,
                ):
                    key = str(
                        template_id
                    )

                    template_stats[
                        key
                    ][
                        "rows"
                    ] += len(
                        g
                    )

                    for c in TARGET_COLUMNS:
                        x = safe_numeric(
                            g[c]
                        ).dropna()

                        if len(x):
                            template_stats[
                                key
                            ][
                                f"{c}_sum"
                            ] += float(
                                x.sum()
                            )

            # -----------------------------------------------------------------
            # Final output column order.
            # -----------------------------------------------------------------

            front = [
                c
                for c in [
                    "sample_id",
                    "participant_id",
                    "timestamp_utc",
                    "timestamp_local",
                    "local_date",
                    "local_slot_seconds",
                    "prediction_cutoff_utc",
                    "prediction_ready_flag",
                    "parameter_label_available_flag",
                    "parameter_target_valid_flag",
                    "parameter_template_match_flag",
                    "parameter_ready_flag",
                    "y_template_id",
                    "y_curve_mode",
                    "y_effective_segment_count",
                    "label_template_id",
                    "label_template_family",
                    "label_template_cluster",
                    "label_curve_mode",
                    "label_effective_segment_count",
                    "label_breakpoint_count",
                    "label_breakpoint_x_json",
                ]
                if c in joined.columns
            ]

            feature_cols_out = [
                c
                for c in pred_features + tr_features
                if c in joined.columns
                and c not in front
                and c not in TARGET_COLUMNS
            ]

            evaluation_cols = [
                c
                for c in [
                    "label_template_shape_mae",
                    "label_template_shape_rmse",
                    "label_template_price_mae",
                    "label_template_price_rmse",
                    "label_source_file",
                    "label_source_row_index",
                    "label_source_market",
                    "label_market_product",
                ]
                if c in joined.columns
            ]

            final_cols = unique_list(
                front
                + feature_cols_out
                + TARGET_COLUMNS
                + evaluation_cols
            )

            joined = joined[
                final_cols
            ]

            if ready_only:
                joined = joined.loc[
                    parameter_ready
                ]

            if output_columns_seen is None:
                output_columns_seen = final_cols

            elif output_columns_seen != final_cols:
                raise RuntimeError(
                    "Output column order changed between chunks. "
                    "Check inconsistent prediction-part schemas."
                )

            if not joined.empty:
                joined.to_csv(
                    out_file,
                    mode="w" if write_header else "a",
                    header=write_header,
                    index=False,
                    encoding="utf-8-sig",
                )

                write_header = False

                totals[
                    "written_rows"
                ] += len(
                    joined
                )

                part_stats[
                    "written_rows"
                ] += len(
                    joined
                )

        manifest_rows.append(
            {
                "input_part": str(
                    part
                ),
                "output_part": str(
                    out_file
                ),
                "prediction_rows": part_stats[
                    "prediction_rows"
                ],
                "prediction_ready_rows": part_stats[
                    "prediction_ready_rows"
                ],
                "label_available_rows": part_stats[
                    "label_available_rows"
                ],
                "parameter_ready_rows": part_stats[
                    "parameter_ready_rows"
                ],
                "written_rows": part_stats[
                    "written_rows"
                ],
            }
        )

    target_summary_rows = []

    for c in TARGET_COLUMNS:
        n = target_counts[
            c
        ]

        mean = (
            target_sums[
                c
            ] / n
            if n
            else np.nan
        )

        variance = (
            target_sq_sums[
                c
            ] / n
            - mean * mean
            if n
            else np.nan
        )

        std = (
            float(
                np.sqrt(
                    max(
                        variance,
                        0.0,
                    )
                )
            )
            if n
            else np.nan
        )

        target_summary_rows.append(
            {
                "target": c,
                "ready_nonmissing_rows": n,
                "mean": mean,
                "std": std,
                "min": (
                    target_mins[
                        c
                    ]
                    if np.isfinite(
                        target_mins[
                            c
                        ]
                    )
                    else np.nan
                ),
                "max": (
                    target_maxs[
                        c
                    ]
                    if np.isfinite(
                        target_maxs[
                            c
                        ]
                    )
                    else np.nan
                ),
            }
        )

    template_summary_rows = []

    for template_id in sorted(
        template_stats
    ):
        stat = template_stats[
            template_id
        ]

        n = stat[
            "rows"
        ]

        row = {
            "template_id": template_id,
            "parameter_ready_rows": n,
            "share_of_parameter_ready": (
                n
                / totals[
                    "parameter_ready_rows"
                ]
                if totals[
                    "parameter_ready_rows"
                ]
                else np.nan
            ),
        }

        for c in TARGET_COLUMNS:
            row[
                f"{c}_mean"
            ] = (
                stat[
                    f"{c}_sum"
                ] / n
                if n
                else np.nan
            )

        template_summary_rows.append(
            row
        )

    return {
        "totals": totals,
        "manifest": pd.DataFrame(
            manifest_rows
        ),
        "target_summary": pd.DataFrame(
            target_summary_rows
        ),
        "template_summary": pd.DataFrame(
            template_summary_rows
        ),
        "output_columns": output_columns_seen or [],
        "month_stats": month_stats,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--prediction-root",
        default="data/processed/bidprediction",
    )

    parser.add_argument(
        "--bidtemplate-root",
        default="data/processed/bidtemplate",
    )

    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Optional Stage2 curve-parameter label CSV or directory. "
            "If omitted, auto-discovery is used."
        ),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--label-chunksize",
        type=int,
        default=250_000,
    )

    parser.add_argument(
        "--ready-only",
        action="store_true",
        help=(
            "Write only parameter_ready_flag==1 rows. "
            "Default keeps all rows for audit."
        ),
    )

    parser.add_argument(
        "--rebuild-label-cache",
        action="store_true",
    )

    parser.add_argument(
        "--keep-label-cache",
        action="store_true",
        help=(
            "Keep temporary monthly Stage2 label cache after build."
        ),
    )

    args = parser.parse_args()

    pred_base = (
        Path(
            args.prediction_root
        )
        / str(
            args.year
        )
    )

    bidtemplate_base = Path(
        args.bidtemplate_root
    )

    output_dir = ensure_dir(
        pred_base
        / "curve_parameter_dataset"
    )

    output_part_dir = ensure_dir(
        output_dir
        / "dataset_parts"
    )

    label_cache_dir = (
        output_dir
        / "_label_cache"
    )

    prediction_parts = discover_prediction_parts(
        pred_base
        / "dataset_parts"
    )

    prediction_schema_file = (
        pred_base
        / f"prediction_feature_schema_{args.year}.csv"
    )

    transition_schema_file = (
        pred_base
        / f"transition_strategy_feature_schema_{args.year}.csv"
    )

    transition_profile_file = (
        pred_base
        / f"transition_strategy_profile_{args.year}.csv"
    )

    prediction_schema = load_prediction_schema(
        prediction_schema_file
    )

    transition_schema = load_transition_schema(
        transition_schema_file
    )

    (
        pred_features,
        tr_features,
        pred_group,
        tr_group,
    ) = get_feature_columns(
        prediction_schema,
        transition_schema,
    )

    transition_profile = load_transition_profile(
        transition_profile_file,
        tr_features,
    )

    label_sources = discover_label_sources(
        bidtemplate_root=bidtemplate_base,
        year=args.year,
        user_path=args.labels,
    )

    prediction_header = pd.read_csv(
        prediction_parts[0],
        nrows=0,
    ).columns.tolist()

    label_header = pd.read_csv(
        label_sources[0],
        nrows=0,
    ).columns.tolist()

    join_cols, join_method = resolve_join_keys(
        prediction_header,
        label_header,
    )

    print("=" * 80)
    print("Build curve parameter prediction dataset")
    print("=" * 80)
    print(f"Year:                 {args.year}")
    print(f"Prediction parts:     {len(prediction_parts)}")
    print(f"Prediction features:  {len(pred_features)}")
    print(f"Z_tr features:        {len(tr_features)}")
    print(f"Join method:          {join_method}")
    print(f"Join columns:         {join_cols}")
    print("Stage2 label sources:")
    for p in label_sources:
        print(f"  {p}")
    print(f"Output:               {output_dir}")
    print()

    (
        cache_files,
        label_source_rows,
    ) = prepare_label_cache(
        label_sources=label_sources,
        cache_dir=label_cache_dir,
        join_cols=join_cols,
        chunksize=args.label_chunksize,
        rebuild=args.rebuild_label_cache,
    )

    print()
    print(
        f"Monthly label caches: {len(cache_files)}; "
        f"Stage2 source rows: {label_source_rows:,}"
    )
    print()

    loader = MonthlyLabelLoader(
        cache_files=cache_files,
        join_cols=join_cols,
        max_cached_months=2,
    )

    result = process_prediction_parts(
        prediction_parts=prediction_parts,
        output_part_dir=output_part_dir,
        pred_features=pred_features,
        tr_features=tr_features,
        transition_profile=transition_profile,
        label_loader=loader,
        join_cols=join_cols,
        chunksize=args.chunksize,
        ready_only=args.ready_only,
    )

    totals = result[
        "totals"
    ]

    # -------------------------------------------------------------------------
    # Output metadata.
    # -------------------------------------------------------------------------

    schema = build_output_schema(
        output_columns=result[
            "output_columns"
        ],
        prediction_schema=prediction_schema,
        transition_schema=transition_schema,
        join_cols=join_cols,
    )

    schema.to_csv(
        output_dir
        / f"curve_parameter_dataset_schema_{args.year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result[
        "manifest"
    ].to_csv(
        output_dir
        / f"curve_parameter_dataset_manifest_{args.year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result[
        "target_summary"
    ].to_csv(
        output_dir
        / f"curve_parameter_target_summary_{args.year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result[
        "template_summary"
    ].to_csv(
        output_dir
        / f"curve_parameter_template_summary_{args.year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_config = {
        "year": args.year,
        "join_method": join_method,
        "join_columns": join_cols,
        "prediction_parts": [
            str(x)
            for x in prediction_parts
        ],
        "stage2_label_sources": [
            str(x)
            for x in label_sources
        ],
        "stage2_label_source_rows": int(
            label_source_rows
        ),
        "prediction_feature_count": len(
            pred_features
        ),
        "transition_feature_count": len(
            tr_features
        ),
        "regression_targets": TARGET_COLUMNS,
        "ready_only_output": bool(
            args.ready_only
        ),
        "parameter_ready_definition": {
            "prediction_ready_flag": 1,
            "stage2_parameter_label_available": True,
            "all_four_targets_finite": True,
            "q_span_mw": "> 0",
            "p_span": ">= 0",
            "stage2_template_equals_y_template_id": True,
        },
        "leakage_note": (
            "Curve parameters are targets only. y_template_id is label/conditioning "
            "metadata, not an ordinary feature. Full-pipeline evaluation must replace "
            "the true template condition with the locked template predictor."
        ),
    }

    (
        output_dir
        / f"curve_parameter_build_config_{args.year}.json"
    ).write_text(
        json.dumps(
            build_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pred_rows = totals[
        "prediction_rows"
    ]

    pred_ready = totals[
        "prediction_ready_rows"
    ]

    parameter_ready = totals[
        "parameter_ready_rows"
    ]

    summary_lines = [
        f"Curve parameter prediction dataset - {args.year}",
        "=" * 80,
        "",
        f"Prediction rows:                 {pred_rows:,}",
        f"Prediction-ready rows:           {pred_ready:,}",
        f"Stage2 label source rows:         {label_source_rows:,}",
        f"Label-available rows:             {totals['label_available_rows']:,}",
        f"Target-valid rows:                {totals['target_valid_rows']:,}",
        f"Template-match rows:              {totals['template_match_rows']:,}",
        f"Parameter-ready rows:             {parameter_ready:,}",
        (
            f"Parameter-ready / all:            "
            f"{parameter_ready / pred_rows:.2%}"
            if pred_rows
            else "Parameter-ready / all:            n/a"
        ),
        (
            f"Parameter-ready / prediction-ready:"
            f" {parameter_ready / pred_ready:.2%}"
            if pred_ready
            else "Parameter-ready / prediction-ready: n/a"
        ),
        "",
        "Prediction-ready failures:",
        (
            f"  Missing Stage2 label:           "
            f"{totals['prediction_ready_missing_label']:,}"
        ),
        (
            f"  Template mismatch:              "
            f"{totals['prediction_ready_template_mismatch']:,}"
        ),
        (
            f"  Invalid parameter target:       "
            f"{totals['prediction_ready_invalid_target']:,}"
        ),
        "",
        f"Join method: {join_method}",
        f"Join columns: {', '.join(join_cols)}",
        "",
        f"Prediction feature count: {len(pred_features)}",
        f"Transition Z_tr count:    {len(tr_features)}",
        f"Regression targets:       {', '.join(TARGET_COLUMNS)}",
        "",
        "Leakage rule:",
        "  q_anchor_mw/q_span_mw/p_anchor/p_span are TARGETS only.",
        "  y_template_id is template-label/conditioning metadata, not an ordinary feature.",
        "  Full-pipeline evaluation must use the locked predicted template instead of",
        "  the true y_template_id as the template condition.",
        "",
        f"Written rows: {totals['written_rows']:,}",
        f"Ready-only output: {args.ready_only}",
    ]

    summary = "\n".join(
        summary_lines
    )

    (
        output_dir
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    if not args.keep_label_cache:
        shutil.rmtree(
            label_cache_dir,
            ignore_errors=True,
        )

    print()
    print(summary)
    print()
    print(
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
