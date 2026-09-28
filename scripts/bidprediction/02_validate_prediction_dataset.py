#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_validate_prediction_dataset.py

Stage 3 / bidprediction
Audit the leakage-controlled prediction dataset produced by
01_build_prediction_dataset.py before any forecasting model is trained.

This script is streaming/chunked and does NOT concatenate the 5M+ interval
rows into memory.

Checks
------
1. Row/readiness coverage by month.
2. Feature missingness on all rows and prediction-ready rows.
3. Historical same-slot state availability.
4. Target distributions: template, mode, segment count.
5. Leakage sanity:
   prediction_cutoff_utc < target timestamp
   cutoff local date == target local_date - 1 day
6. Strict chronological train/validation/test split.
7. Template coverage in every split.

Default split
-------------
train : first 70% of distinct prediction-ready dates
val   : next 15%
test  : final 15%

No random row split is used.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_schema(schema_file: Path) -> pd.DataFrame:
    if not schema_file.exists():
        raise FileNotFoundError(schema_file)
    schema = pd.read_csv(schema_file)
    required = {"column", "role", "feature_group"}
    missing = required - set(schema.columns)
    if missing:
        raise KeyError(f"Schema missing columns: {sorted(missing)}")
    return schema


def discover_parts(part_dir: Path) -> list[Path]:
    files = sorted(part_dir.glob("prediction_dataset_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction dataset parts under {part_dir}")
    return files


def month_from_file(file: Path) -> str:
    m = re.search(r"(\d{4})[_-](\d{2})", file.stem)
    return f"{m.group(1)}-{m.group(2)}" if m else file.stem


def choose_split_boundaries(date_counts: Counter, train_frac: float, val_frac: float):
    dates = sorted(date_counts)
    if len(dates) < 3:
        raise ValueError("Too few prediction-ready dates for chronological split.")

    n = len(dates)
    train_end_idx = max(0, min(n - 3, int(np.floor(n * train_frac)) - 1))
    val_end_idx = max(
        train_end_idx + 1,
        min(n - 2, int(np.floor(n * (train_frac + val_frac))) - 1),
    )

    train_end = pd.Timestamp(dates[train_end_idx])
    val_end = pd.Timestamp(dates[val_end_idx])
    test_start = pd.Timestamp(dates[val_end_idx + 1])
    return train_end, val_end, test_start


def assign_split(dates: pd.Series, train_end: pd.Timestamp, val_end: pd.Timestamp):
    d = pd.to_datetime(dates, errors="coerce").dt.normalize()
    out = pd.Series(pd.NA, index=d.index, dtype="object")
    out.loc[d <= train_end] = "train"
    out.loc[(d > train_end) & (d <= val_end)] = "val"
    out.loc[d > val_end] = "test"
    return out


def pass1(files, feature_cols, chunksize):
    total_rows = total_ready = 0
    total_profile_ready = total_market_ready = total_unit_ready = 0

    month_stats = defaultdict(Counter)
    missing_all = Counter()
    missing_ready = Counter()
    nonmissing_ready = Counter()

    template_counts = Counter()
    mode_counts = Counter()
    segment_counts = Counter()
    date_ready_counts = Counter()

    participant_ready_rows = Counter()
    participant_ready_dates = defaultdict(set)
    unit_hist_counts = Counter()
    leakage = Counter()

    needed = set(
        [
            "participant_id",
            "timestamp_utc",
            "timestamp_local",
            "local_date",
            "prediction_cutoff_utc",
            "profile_ready_flag",
            "market_ready_flag",
            "unit_state_ready_flag",
            "prediction_ready_flag",
            "hist_prev_available_flag",
            "y_template_id",
            "y_curve_mode",
            "y_effective_segment_count",
        ] + feature_cols
    )

    for i, file in enumerate(files, 1):
        month = month_from_file(file)
        header = pd.read_csv(file, nrows=0).columns.tolist()
        usecols = [c for c in header if c in needed]

        required = [
            "participant_id",
            "local_date",
            "prediction_ready_flag",
            "y_template_id",
            "y_curve_mode",
            "y_effective_segment_count",
        ]
        missing = [c for c in required if c not in header]
        if missing:
            raise KeyError(f"{file.name} missing columns: {missing}")

        print(f"[pass1 {i}/{len(files)}] {file.name}", flush=True)

        for chunk in pd.read_csv(
            file, usecols=usecols, chunksize=chunksize, low_memory=False
        ):
            n = len(chunk)
            total_rows += n
            month_stats[month]["rows"] += n

            pred_ready = safe_numeric(
                chunk.get("prediction_ready_flag", pd.Series(0, index=chunk.index))
            ).fillna(0).eq(1)
            profile_ready = safe_numeric(
                chunk.get("profile_ready_flag", pd.Series(0, index=chunk.index))
            ).fillna(0).eq(1)
            market_ready = safe_numeric(
                chunk.get("market_ready_flag", pd.Series(0, index=chunk.index))
            ).fillna(0).eq(1)
            unit_ready = safe_numeric(
                chunk.get("unit_state_ready_flag", pd.Series(0, index=chunk.index))
            ).fillna(0).eq(1)

            total_ready += int(pred_ready.sum())
            total_profile_ready += int(profile_ready.sum())
            total_market_ready += int(market_ready.sum())
            total_unit_ready += int(unit_ready.sum())

            month_stats[month]["profile_ready"] += int(profile_ready.sum())
            month_stats[month]["market_ready"] += int(market_ready.sum())
            month_stats[month]["unit_state_ready"] += int(unit_ready.sum())
            month_stats[month]["prediction_ready"] += int(pred_ready.sum())

            for c in feature_cols:
                if c not in chunk.columns:
                    missing_all[c] += n
                    missing_ready[c] += int(pred_ready.sum())
                    continue

                na = chunk[c].isna()
                missing_all[c] += int(na.sum())
                if pred_ready.any():
                    na_r = na.loc[pred_ready]
                    missing_ready[c] += int(na_r.sum())
                    nonmissing_ready[c] += int((~na_r).sum())

            ready = chunk.loc[pred_ready].copy()
            if ready.empty:
                continue

            template_counts.update(
                ready["y_template_id"].astype("string").fillna("<NA>").astype(str)
            )
            mode_counts.update(
                ready["y_curve_mode"].astype("string").fillna("<NA>").astype(str)
            )

            seg = safe_numeric(ready["y_effective_segment_count"])
            segment_counts.update(
                "<NA>" if pd.isna(v) else str(int(v)) for v in seg
            )

            dates = pd.to_datetime(ready["local_date"], errors="coerce").dt.normalize()
            pids = ready["participant_id"].astype("string").str.strip()
            valid = dates.notna() & pids.notna()

            for dd, pp in zip(dates[valid], pids[valid]):
                dkey = dd.date().isoformat()
                pkey = str(pp)
                date_ready_counts[dkey] += 1
                participant_ready_rows[pkey] += 1
                participant_ready_dates[pkey].add(dkey)

            if "hist_prev_available_flag" in ready.columns:
                h = safe_numeric(ready["hist_prev_available_flag"]).fillna(0).astype(int)
                unit_hist_counts["available"] += int(h.eq(1).sum())
                unit_hist_counts["missing"] += int(h.eq(0).sum())

            if "timestamp_utc" in ready.columns and "prediction_cutoff_utc" in ready.columns:
                ts = pd.to_datetime(ready["timestamp_utc"], errors="coerce", utc=True)
                cutoff = pd.to_datetime(
                    ready["prediction_cutoff_utc"], errors="coerce", utc=True
                )
                ok = ts.notna() & cutoff.notna()
                leakage["timestamp_pairs"] += int(ok.sum())
                leakage["cutoff_ge_target"] += int((ok & (cutoff >= ts)).sum())

            if "timestamp_local" in ready.columns and "prediction_cutoff_utc" in ready.columns:
                local_ts = pd.to_datetime(ready["timestamp_local"], errors="coerce")
                cutoff_utc = pd.to_datetime(
                    ready["prediction_cutoff_utc"], errors="coerce", utc=True
                )
                cutoff_local = (
                    cutoff_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
                )
                ok = local_ts.notna() & cutoff_local.notna()
                expected = local_ts.dt.normalize() - pd.Timedelta(days=1)
                leakage["cutoff_date_pairs"] += int(ok.sum())
                leakage["cutoff_wrong_date"] += int(
                    (ok & (cutoff_local.dt.normalize() != expected)).sum()
                )

    return {
        "total_rows": total_rows,
        "total_ready": total_ready,
        "total_profile_ready": total_profile_ready,
        "total_market_ready": total_market_ready,
        "total_unit_ready": total_unit_ready,
        "month_stats": month_stats,
        "missing_all": missing_all,
        "missing_ready": missing_ready,
        "nonmissing_ready": nonmissing_ready,
        "template_counts": template_counts,
        "mode_counts": mode_counts,
        "segment_counts": segment_counts,
        "date_ready_counts": date_ready_counts,
        "participant_ready_rows": participant_ready_rows,
        "participant_ready_dates": participant_ready_dates,
        "unit_hist_counts": unit_hist_counts,
        "leakage": leakage,
    }


def pass2(files, chunksize, train_end, val_end):
    split_rows = Counter()
    split_participants = defaultdict(set)
    split_dates = defaultdict(set)
    template_split = Counter()

    for i, file in enumerate(files, 1):
        print(f"[pass2 {i}/{len(files)}] {file.name}", flush=True)
        header = pd.read_csv(file, nrows=0).columns.tolist()
        usecols = [
            c for c in [
                "participant_id",
                "local_date",
                "prediction_ready_flag",
                "y_template_id",
            ] if c in header
        ]

        for chunk in pd.read_csv(
            file, usecols=usecols, chunksize=chunksize, low_memory=False
        ):
            ready = safe_numeric(chunk["prediction_ready_flag"]).fillna(0).eq(1)
            d = chunk.loc[ready].copy()
            if d.empty:
                continue

            d["split"] = assign_split(d["local_date"], train_end, val_end)
            d = d[d["split"].notna()]

            for split, g in d.groupby("split", observed=True):
                split = str(split)
                split_rows[split] += len(g)
                split_participants[split].update(
                    g["participant_id"].astype("string").dropna().astype(str)
                )
                split_dates[split].update(
                    pd.to_datetime(g["local_date"], errors="coerce")
                    .dropna().dt.date.astype(str)
                )
                vc = (
                    g["y_template_id"]
                    .astype("string")
                    .fillna("<NA>")
                    .astype(str)
                    .value_counts()
                )
                for tid, n in vc.items():
                    template_split[(split, str(tid))] += int(n)

    return split_rows, split_participants, split_dates, template_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--root", default="data/processed/bidprediction")
    p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    args = p.parse_args()

    if args.train_frac <= 0 or args.val_frac <= 0:
        raise ValueError("train-frac and val-frac must be positive.")
    if args.train_frac + args.val_frac >= 1:
        raise ValueError("train-frac + val-frac must be < 1.")

    base = Path(args.root) / str(args.year)
    files = discover_parts(base / "dataset_parts")
    schema = read_schema(base / f"prediction_feature_schema_{args.year}.csv")

    feature_cols = (
        schema.loc[schema["role"].eq("feature"), "column"].astype(str).tolist()
    )
    model_feature_cols = [
        c for c in feature_cols
        if not c.endswith("_ready_flag")
        and c != "market_nonmissing_count"
    ]

    out_dir = ensure_dir(base / "validation")

    print("=" * 80)
    print("Validate bid-prediction dataset")
    print("=" * 80)
    print(f"Year:               {args.year}")
    print(f"Dataset parts:      {len(files)}")
    print(f"Model features:     {len(model_feature_cols)}")

    s = pass1(files, model_feature_cols, args.chunksize)

    total_rows = s["total_rows"]
    total_ready = s["total_ready"]

    audit = pd.DataFrame([
        {"metric": "rows", "value": total_rows},
        {"metric": "profile_ready_rows", "value": s["total_profile_ready"]},
        {"metric": "market_ready_rows", "value": s["total_market_ready"]},
        {"metric": "unit_state_ready_rows", "value": s["total_unit_ready"]},
        {"metric": "prediction_ready_rows", "value": total_ready},
        {"metric": "prediction_ready_share",
         "value": total_ready / total_rows if total_rows else np.nan},
        {"metric": "ready_participants", "value": len(s["participant_ready_rows"])},
        {"metric": "ready_dates", "value": len(s["date_ready_counts"])},
        {"metric": "template_classes", "value": len(s["template_counts"])},
    ])
    audit.to_csv(
        out_dir / "dataset_audit_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    monthly_rows = []
    for month in sorted(s["month_stats"]):
        x = s["month_stats"][month]
        n = x["rows"]
        monthly_rows.append({
            "month": month,
            "rows": n,
            "profile_ready_rows": x["profile_ready"],
            "market_ready_rows": x["market_ready"],
            "unit_state_ready_rows": x["unit_state_ready"],
            "prediction_ready_rows": x["prediction_ready"],
            "prediction_ready_share": x["prediction_ready"] / n if n else np.nan,
            "unit_state_ready_share": x["unit_state_ready"] / n if n else np.nan,
        })
    pd.DataFrame(monthly_rows).to_csv(
        out_dir / "monthly_ready_coverage.csv",
        index=False, encoding="utf-8-sig"
    )

    miss_rows = []
    for c in model_feature_cols:
        miss_rows.append({
            "feature": c,
            "missing_all_rows": s["missing_all"][c],
            "missing_all_share": (
                s["missing_all"][c] / total_rows if total_rows else np.nan
            ),
            "missing_ready_rows": s["missing_ready"][c],
            "missing_ready_share": (
                s["missing_ready"][c] / total_ready if total_ready else np.nan
            ),
            "nonmissing_ready_rows": s["nonmissing_ready"][c],
        })
    missingness = pd.DataFrame(miss_rows).sort_values(
        ["missing_ready_share", "feature"], ascending=[False, True]
    )
    missingness.to_csv(
        out_dir / "feature_missingness.csv",
        index=False, encoding="utf-8-sig"
    )

    template_df = pd.DataFrame([
        {"template_id": k, "sample_count": v}
        for k, v in s["template_counts"].items()
    ]).sort_values("sample_count", ascending=False)
    if len(template_df):
        template_df["share"] = (
            template_df["sample_count"] / template_df["sample_count"].sum()
        )
    template_df.to_csv(
        out_dir / "template_distribution.csv",
        index=False, encoding="utf-8-sig"
    )

    mode_df = pd.DataFrame([
        {"curve_mode": k, "sample_count": v}
        for k, v in s["mode_counts"].items()
    ]).sort_values("sample_count", ascending=False)
    if len(mode_df):
        mode_df["share"] = mode_df["sample_count"] / mode_df["sample_count"].sum()
    mode_df.to_csv(
        out_dir / "curve_mode_distribution.csv",
        index=False, encoding="utf-8-sig"
    )

    seg_df = pd.DataFrame([
        {"effective_segment_count": k, "sample_count": v}
        for k, v in s["segment_counts"].items()
    ])
    if len(seg_df):
        seg_df["_sort"] = pd.to_numeric(
            seg_df["effective_segment_count"], errors="coerce"
        )
        seg_df = seg_df.sort_values(
            ["_sort", "effective_segment_count"], na_position="last"
        ).drop(columns="_sort")
        seg_df["share"] = seg_df["sample_count"] / seg_df["sample_count"].sum()
    seg_df.to_csv(
        out_dir / "segment_count_distribution.csv",
        index=False, encoding="utf-8-sig"
    )

    hist_avail = s["unit_hist_counts"]["available"]
    hist_missing = s["unit_hist_counts"]["missing"]
    pd.DataFrame([
        {
            "metric": "hist_prev_available_rows",
            "value": hist_avail,
            "share_of_ready": hist_avail / total_ready if total_ready else np.nan,
        },
        {
            "metric": "hist_prev_missing_rows",
            "value": hist_missing,
            "share_of_ready": hist_missing / total_ready if total_ready else np.nan,
        },
    ]).to_csv(
        out_dir / "unit_history_coverage.csv",
        index=False, encoding="utf-8-sig"
    )

    leakage_df = pd.DataFrame([
        {
            "check": "prediction_cutoff_before_target_timestamp",
            "pairs_checked": s["leakage"]["timestamp_pairs"],
            "violations": s["leakage"]["cutoff_ge_target"],
        },
        {
            "check": "cutoff_local_date_equals_target_D_minus_1",
            "pairs_checked": s["leakage"]["cutoff_date_pairs"],
            "violations": s["leakage"]["cutoff_wrong_date"],
        },
    ])
    leakage_df["violation_share"] = leakage_df.apply(
        lambda r: (
            r["violations"] / r["pairs_checked"]
            if r["pairs_checked"] else np.nan
        ),
        axis=1,
    )
    leakage_df.to_csv(
        out_dir / "leakage_audit.csv",
        index=False, encoding="utf-8-sig"
    )

    train_end, val_end, test_start = choose_split_boundaries(
        s["date_ready_counts"], args.train_frac, args.val_frac
    )
    split_rows, split_participants, split_dates, template_split = pass2(
        files, args.chunksize, train_end, val_end
    )

    split_summary = []
    for split in ["train", "val", "test"]:
        dates = sorted(split_dates[split])
        split_summary.append({
            "split": split,
            "rows": split_rows[split],
            "participants": len(split_participants[split]),
            "distinct_dates": len(dates),
            "first_date": dates[0] if dates else "",
            "last_date": dates[-1] if dates else "",
            "share_of_ready": (
                split_rows[split] / total_ready if total_ready else np.nan
            ),
        })
    pd.DataFrame(split_summary).to_csv(
        out_dir / "temporal_split_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    template_split_rows = []
    all_templates = sorted(s["template_counts"])
    for split in ["train", "val", "test"]:
        nsplit = split_rows[split]
        for tid in all_templates:
            n = template_split[(split, tid)]
            template_split_rows.append({
                "split": split,
                "template_id": tid,
                "sample_count": n,
                "share_within_split": n / nsplit if nsplit else np.nan,
            })
    template_split_df = pd.DataFrame(template_split_rows)
    template_split_df.to_csv(
        out_dir / "template_split_distribution.csv",
        index=False, encoding="utf-8-sig"
    )

    participant_rows = []
    for pid, nrows in s["participant_ready_rows"].items():
        dates = sorted(s["participant_ready_dates"][pid])
        participant_rows.append({
            "participant_id": pid,
            "ready_rows": nrows,
            "ready_days": len(dates),
            "first_ready_date": dates[0] if dates else "",
            "last_ready_date": dates[-1] if dates else "",
        })
    pd.DataFrame(participant_rows).sort_values(
        "ready_rows", ascending=False
    ).to_csv(
        out_dir / "participant_split_coverage.csv",
        index=False, encoding="utf-8-sig"
    )

    top_missing = missingness.head(12)
    min_template_split = (
        template_split_df.groupby("template_id")["sample_count"]
        .min()
        .sort_values()
    )
    violations = int(leakage_df["violations"].sum())

    lines = [
        f"Prediction dataset validation - {args.year}",
        "=" * 80,
        "",
        f"Rows:                   {total_rows:,}",
        f"Prediction-ready:       {total_ready:,} "
        f"({total_ready/total_rows:.2%})",
        f"Ready participants:     {len(s['participant_ready_rows']):,}",
        f"Ready dates:            {len(s['date_ready_counts']):,}",
        f"Template classes:       {len(s['template_counts'])}",
        (
            f"Unit-history available: {hist_avail:,} "
            f"({hist_avail/total_ready:.2%})"
            if total_ready else
            "Unit-history available: NA"
        ),
        "",
        "Temporal split:",
        f"  train <= {train_end.date()}",
        f"  val   <= {val_end.date()}",
        f"  test  >= {test_start.date()}",
        "",
        f"Leakage-audit violations: {violations}",
        "",
        "Most-missing model features on prediction-ready rows:",
    ]

    for r in top_missing.itertuples():
        lines.append(f"  {r.feature}: {r.missing_ready_share:.2%}")

    lines += [
        "",
        "Minimum sample count for each template across train/val/test:",
    ]
    for tid, n in min_template_split.items():
        lines.append(f"  {tid}: {int(n):,}")

    summary_file = out_dir / "summary.txt"
    summary_file.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
