#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04a3_freeze_modeling_dataset.py

Freeze the modeling data ONCE after 04a2.

After this script finishes, 04b / 04c / 04d no longer:
- rescan the 5.1M-row monthly CSV dataset,
- recompute train/val/test dates,
- resample the 300k training set,
- rejoin Stage2 curve samples.

Frozen outputs:
data/processed/bidprediction/<year>/frozen_modeling_dataset/
    feature_schema.csv
    manifest.json
    train_sample_300000.pkl
    train_parts/*.pkl
    val_parts/*.pkl
    test_parts/*.pkl
    test_curve_parts/*.pkl

The split is inherited exactly from:
data/processed/bidprediction/<year>/validation/temporal_split_summary.csv

Run:
python scripts/bidprediction/04a3_freeze_modeling_dataset.py --year 2025
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RAW_THETA = [
    "theta_p_base",
    "theta_alpha",
    "theta_beta",
    "theta_q_base_mw",
    "theta_q_span_mw",
    "theta_q1",
    "theta_q2",
    "theta_q3",
    "theta_q4",
    "theta_q5",
]

BASE_COLUMNS = [
    "sample_id",
    "participant_id",
    "local_date",
    "theta_ready_flag",
    "y_template_id",
    "q_anchor_mw",
    "q_span_mw",
    "p_anchor",
    "p_span",
    *RAW_THETA,
]


def norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def uniq(xs):
    return list(dict.fromkeys(xs))


def month_key(path: Path):
    m = re.search(r"(\d{4})[_-](\d{2})", path.stem)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def shape_cols(header):
    exact = [f"shape_v{i:02d}" for i in range(21)]
    if all(c in header for c in exact):
        return exact

    groups = {}
    for c in header:
        m = re.match(r"^(.*?)(?:_)?v(\d{2})$", str(c), flags=re.I)
        if m and 0 <= int(m.group(2)) <= 20:
            groups.setdefault(m.group(1), {})[int(m.group(2))] = c

    for g in groups.values():
        if len(g) == 21 and all(i in g for i in range(21)):
            return [g[i] for i in range(21)]

    return None


def discover_curve_samples(year_dir: Path):
    root = year_dir / "curve_samples"
    if not root.exists():
        raise FileNotFoundError(root)

    out = {}
    for p in sorted(root.rglob("*.csv")):
        try:
            h = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue

        if "sample_id" not in h:
            continue

        sc = shape_cols(h)
        if sc is None:
            continue

        mk = month_key(p)
        if mk:
            out.setdefault(mk, []).append(p)

    if not out:
        raise FileNotFoundError(
            f"No Stage2 curve samples with sample_id + shape_v00..20 under {root}"
        )

    return out


def load_month_shapes(files):
    blocks = []

    for p in files:
        h = pd.read_csv(p, nrows=0).columns.tolist()
        sc = shape_cols(h)

        d = pd.read_csv(
            p,
            usecols=["sample_id"] + sc,
            low_memory=False,
        )

        d["sample_id"] = norm(d["sample_id"])

        d = d.rename(
            columns={
                c: f"shape_v{i:02d}"
                for i, c in enumerate(sc)
            }
        )

        blocks.append(d)

    out = pd.concat(blocks, ignore_index=True)

    if out["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in Stage2 curve samples.")

    return out


def read_split(path: Path):
    s = pd.read_csv(path).set_index("split")

    tr_end = pd.Timestamp(s.loc["train", "last_date"]).normalize()
    val_start = pd.Timestamp(s.loc["val", "first_date"]).normalize()
    val_end = pd.Timestamp(s.loc["val", "last_date"]).normalize()
    test_start = pd.Timestamp(s.loc["test", "first_date"]).normalize()

    return tr_end, val_start, val_end, test_start


def split_name(date_series, tr_end, val_start, val_end, test_start):
    d = pd.to_datetime(date_series, errors="coerce").dt.normalize()

    train = d <= tr_end
    val = (d >= val_start) & (d <= val_end)
    test = d >= test_start

    return train, val, test


def write_chunk(
    df: pd.DataFrame,
    out_dir: Path,
    split: str,
    source_stem: str,
    chunk_no: int,
):
    if df.empty:
        return None

    d = out_dir / f"{split}_parts"
    d.mkdir(parents=True, exist_ok=True)

    path = d / f"{split}_{source_stem}_{chunk_no:04d}.pkl"
    df.to_pickle(path, protocol=5)
    return path


def priority_for_sample_id(sample_id: pd.Series, seed: int):
    base = pd.util.hash_pandas_object(
        sample_id.astype("string"),
        index=False,
    ).to_numpy(np.uint64)

    # deterministic seed mixing
    mix = np.uint64(seed * 0x9E3779B1)
    return base ^ mix


def update_reservoir(
    reservoir: pd.DataFrame | None,
    new_rows: pd.DataFrame,
    max_rows: int,
    seed: int,
):
    if new_rows.empty:
        return reservoir

    x = new_rows.copy()
    x["__freeze_priority"] = priority_for_sample_id(
        x["sample_id"],
        seed,
    )

    if reservoir is None:
        merged = x
    else:
        merged = pd.concat(
            [reservoir, x],
            ignore_index=True,
        )

    if len(merged) > max_rows:
        merged = merged.nsmallest(
            max_rows,
            "__freeze_priority",
            keep="first",
        ).reset_index(drop=True)

    return merged


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--bidtemplate-root",
        default="data/processed/bidtemplate",
    )
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--train-sample-rows", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    source_dir = base / "template_adjustment_dataset"
    source_parts = sorted(
        (source_dir / "dataset_parts").glob(
            "template_adjustment_dataset_*.csv"
        )
    )

    if not source_parts:
        raise FileNotFoundError(
            "Run 04a2_build_template_adjustment_dataset.py first."
        )

    schema_file = (
        source_dir
        / f"template_adjustment_schema_{args.year}.csv"
    )

    if not schema_file.exists():
        raise FileNotFoundError(schema_file)

    split_file = (
        base
        / "validation"
        / "temporal_split_summary.csv"
    )

    if not split_file.exists():
        raise FileNotFoundError(split_file)

    schema = pd.read_csv(schema_file)

    role = schema["role"].astype(str).str.lower()
    feature_cols = schema.loc[
        role.eq("feature"),
        "column",
    ].astype(str).tolist()

    keep_cols = uniq(BASE_COLUMNS + feature_cols)

    first_header = pd.read_csv(
        source_parts[0],
        nrows=0,
    ).columns.tolist()

    missing = [c for c in keep_cols if c not in first_header]

    if missing:
        raise KeyError(
            f"Source dataset missing required frozen columns: {missing}"
        )

    tr_end, val_start, val_end, test_start = read_split(
        split_file
    )

    out = base / "frozen_modeling_dataset"

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} already exists. "
                "Use --overwrite only if you intentionally want to rebuild the frozen dataset."
            )
        shutil.rmtree(out)

    out.mkdir(parents=True)

    # Copy schema and split summary for immutable experiment provenance.
    shutil.copy2(
        schema_file,
        out / "feature_schema.csv",
    )
    shutil.copy2(
        split_file,
        out / "temporal_split_summary.csv",
    )

    bidtemplate_year = (
        Path(args.bidtemplate_root)
        / str(args.year)
    )

    sample_files = discover_curve_samples(
        bidtemplate_year
    )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
        "test_curve": 0,
    }

    reservoir = None
    written_files = {
        "train": [],
        "val": [],
        "test": [],
        "test_curve": [],
    }

    print("=" * 80)
    print("Freeze modeling dataset")
    print("=" * 80)
    print(f"Year:                 {args.year}")
    print(f"Train <=              {tr_end.date()}")
    print(f"Validation:           {val_start.date()} .. {val_end.date()}")
    print(f"Test >=               {test_start.date()}")
    print(f"Frozen features:      {len(feature_cols)}")
    print(f"Train sample target:  {args.train_sample_rows:,}")
    print()

    for part_no, src in enumerate(source_parts, 1):
        mk = month_key(src)

        print(
            f"[source {part_no}/{len(source_parts)}] {src.name}",
            flush=True,
        )

        month_shapes = None

        if mk in sample_files:
            # Load once per source month. It is only used if this month contains test rows.
            month_shapes = load_month_shapes(
                sample_files[mk]
            )

        for chunk_no, chunk in enumerate(
            pd.read_csv(
                src,
                usecols=keep_cols,
                chunksize=args.chunksize,
                low_memory=False,
            ),
            1,
        ):
            chunk["sample_id"] = norm(chunk["sample_id"])
            chunk["participant_id"] = norm(chunk["participant_id"])
            chunk["local_date"] = pd.to_datetime(
                chunk["local_date"],
                errors="coerce",
            ).dt.normalize()

            ready = (
                num(chunk["theta_ready_flag"])
                .fillna(0)
                .eq(1)
            )

            chunk = chunk.loc[ready].copy()

            if chunk.empty:
                continue

            m_train, m_val, m_test = split_name(
                chunk["local_date"],
                tr_end,
                val_start,
                val_end,
                test_start,
            )

            split_masks = {
                "train": m_train,
                "val": m_val,
                "test": m_test,
            }

            for split, mask in split_masks.items():
                d = chunk.loc[mask].copy()

                if d.empty:
                    continue

                counts[split] += len(d)

                path = write_chunk(
                    d,
                    out,
                    split,
                    src.stem,
                    chunk_no,
                )

                written_files[split].append(
                    str(path.relative_to(out))
                )

                if split == "train":
                    reservoir = update_reservoir(
                        reservoir,
                        d,
                        args.train_sample_rows,
                        args.seed,
                    )

                if split == "test":
                    if month_shapes is None:
                        raise FileNotFoundError(
                            f"No Stage2 curve samples discovered for test month {mk}"
                        )

                    curve = d.merge(
                        month_shapes,
                        on="sample_id",
                        how="left",
                        validate="one_to_one",
                        sort=False,
                    )

                    sc = [f"shape_v{i:02d}" for i in range(21)]

                    shape_ok = curve[sc].notna().all(axis=1)

                    if not shape_ok.all():
                        bad = int((~shape_ok).sum())
                        raise ValueError(
                            f"{bad} test rows in {src.name} have no Stage2 shape join."
                        )

                    counts["test_curve"] += len(curve)

                    curve_dir = out / "test_curve_parts"
                    curve_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    curve_path = (
                        curve_dir
                        / f"test_curve_{src.stem}_{chunk_no:04d}.pkl"
                    )

                    curve.to_pickle(
                        curve_path,
                        protocol=5,
                    )

                    written_files["test_curve"].append(
                        str(curve_path.relative_to(out))
                    )

            del chunk
            gc.collect()

        del month_shapes
        gc.collect()

    if reservoir is None or reservoir.empty:
        raise ValueError("Frozen training reservoir is empty.")

    reservoir = (
        reservoir
        .sort_values("__freeze_priority")
        .head(args.train_sample_rows)
        .drop(columns="__freeze_priority")
        .reset_index(drop=True)
    )

    if len(reservoir) != args.train_sample_rows:
        raise ValueError(
            f"Expected {args.train_sample_rows:,} frozen train-sample rows, "
            f"got {len(reservoir):,}."
        )

    train_sample_file = (
        out
        / f"train_sample_{args.train_sample_rows}.pkl"
    )

    reservoir.to_pickle(
        train_sample_file,
        protocol=5,
    )

    if counts["test"] != counts["test_curve"]:
        raise ValueError(
            "Frozen test/test_curve row-count mismatch: "
            f"{counts['test']:,} vs {counts['test_curve']:,}"
        )

    frozen_schema_rows = []

    for c in keep_cols:
        frozen_schema_rows.append(
            {
                "column": c,
                "source": (
                    "feature"
                    if c in feature_cols
                    else "metadata_or_target"
                ),
            }
        )

    for i in range(21):
        frozen_schema_rows.append(
            {
                "column": f"shape_v{i:02d}",
                "source": "stage2_curve_shape_test_only",
            }
        )

    pd.DataFrame(
        frozen_schema_rows
    ).to_csv(
        out / "frozen_columns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "version": "frozen-modeling-dataset-v1",
        "year": args.year,
        "source_dataset": str(source_dir),
        "source_schema": str(schema_file),
        "source_split": str(split_file),
        "train_end": str(tr_end.date()),
        "validation_start": str(val_start.date()),
        "validation_end": str(val_end.date()),
        "test_start": str(test_start.date()),
        "feature_count": len(feature_cols),
        "frozen_base_column_count": len(keep_cols),
        "train_rows": counts["train"],
        "val_rows": counts["val"],
        "test_rows": counts["test"],
        "test_curve_rows": counts["test_curve"],
        "train_sample_rows": len(reservoir),
        "train_sample_file": train_sample_file.name,
        "seed": args.seed,
        "chunksize": args.chunksize,
        "parts": written_files,
    }

    (
        out / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            f"Frozen modeling dataset - {args.year}",
            "=" * 80,
            "",
            f"Train <= {tr_end.date()}",
            f"Validation = {val_start.date()} .. {val_end.date()}",
            f"Test >= {test_start.date()}",
            "",
            f"Train rows: {counts['train']:,}",
            f"Validation rows: {counts['val']:,}",
            f"Test rows: {counts['test']:,}",
            f"Test-curve rows: {counts['test_curve']:,}",
            f"Fixed train sample: {len(reservoir):,}",
            f"Frozen feature columns: {len(feature_cols)}",
            "",
            "04b/04c/04d should read ONLY this frozen dataset.",
        ]
    )

    (
        out / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
