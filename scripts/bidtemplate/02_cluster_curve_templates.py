#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_cluster_curve_templates.py

Stage 2: Bid-template library
01 curve samples -> compact curve-template library + full historical assignments.

Design
------
1. Cluster ONLY eligible non-flat shape curves:
       template_family == "shape"
       shape_cluster_eligible_flag == 1

2. Flat curves remain an explicit independent family:
       template_id = "FLAT"

3. The 21-D normalized shapes are clustered on a representative random sample.
   Candidate K values are compared using held-out reconstruction error,
   silhouette score, cluster balance, and an elbow score.

4. The selected centers are then used to assign ALL historical rows.
   The output assignments therefore provide direct supervision for the later
   template-choice prediction stage.

Inputs
------
data/processed/bidtemplate/<year>/curve_samples/*.csv

Outputs
-------
data/processed/bidtemplate/<year>/template_library/
    candidate_k_metrics.csv
    curve_template_library.csv
    curve_template_cluster_summary.csv
    curve_template_model.joblib
    assignments/
        template_assignments_<source_file_stem>.csv

Notes
-----
- Template IDs are ordered by normalized curve area, from low area to high area,
  so IDs are deterministic and interpretable across reruns with the same data.
- For shape curves:
      price_MAE = shape_MAE * abs(p_span)
      price_RMSE = shape_RMSE * abs(p_span)
  because q_anchor/q_span are retained exactly by Stage 1.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
RNG_SEED = 42


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_k_list(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or min(values) < 2:
        raise ValueError("Candidate K values must contain integers >= 2.")
    return values


def shape_filter(df: pd.DataFrame) -> np.ndarray:
    fam = df["template_family"].astype(str).eq("shape").to_numpy()
    eligible = pd.to_numeric(
        df["shape_cluster_eligible_flag"], errors="coerce"
    ).fillna(0).to_numpy() == 1
    return fam & eligible


def read_total_shape_count(year_dir: Path) -> int | None:
    manifest = year_dir / f"curve_samples_manifest_{year_dir.name}.csv"
    if not manifest.exists():
        return None
    df = pd.read_csv(manifest)
    if "shape_family_rows" not in df.columns:
        return None
    return int(pd.to_numeric(df["shape_family_rows"], errors="coerce").fillna(0).sum())


def collect_shape_sample(
    files: list[Path],
    target_size: int,
    chunksize: int,
    seed: int,
    total_shape_hint: int | None,
) -> np.ndarray:
    """Random row-frequency sample of the shape family without loading all rows."""
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    collected = 0

    if total_shape_hint and total_shape_hint > 0:
        p = min(1.0, 1.15 * target_size / total_shape_hint)
    else:
        p = None

    usecols = ["template_family", "shape_cluster_eligible_flag", *SHAPE_COLS]

    for file_no, file in enumerate(files, start=1):
        print(f"[sample {file_no}/{len(files)}] {file.name}", flush=True)

        for chunk in pd.read_csv(
            file,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ):
            mask = shape_filter(chunk)
            if not mask.any():
                continue

            X = chunk.loc[mask, SHAPE_COLS].to_numpy(np.float32)
            finite = np.all(np.isfinite(X), axis=1)
            X = X[finite]
            if len(X) == 0:
                continue

            if p is None:
                # Fallback when no manifest is available: retain a bounded random
                # subset from each chunk, then downsample globally at the end.
                n = min(len(X), max(1000, target_size // max(1, len(files) * 4)))
            else:
                n = int(rng.binomial(len(X), p))
                n = min(n, len(X))

            if n > 0:
                idx = rng.choice(len(X), size=n, replace=False)
                blocks.append(X[idx])
                collected += n

        print(f"  collected_so_far={collected:,}", flush=True)

    if not blocks:
        raise RuntimeError("No eligible shape rows found.")

    X = np.vstack(blocks)

    if len(X) > target_size:
        idx = rng.choice(len(X), size=target_size, replace=False)
        X = X[idx]

    if len(X) < min(10_000, target_size):
        print(
            f"[warning] only {len(X):,} shape samples collected; "
            "model selection may be less stable.",
            flush=True,
        )

    return X.astype(np.float32, copy=False)


def eval_model(model: MiniBatchKMeans, X: np.ndarray) -> tuple[np.ndarray, dict]:
    labels = model.predict(X)
    centers = model.cluster_centers_[labels]
    diff = X - centers

    row_mae = np.mean(np.abs(diff), axis=1)
    row_rmse = np.sqrt(np.mean(diff * diff, axis=1))

    counts = np.bincount(labels, minlength=model.n_clusters)
    shares = counts / len(labels)

    return labels, {
        "shape_mae": float(np.mean(row_mae)),
        "shape_rmse": float(np.mean(row_rmse)),
        "min_cluster_share": float(np.min(shares)),
        "max_cluster_share": float(np.max(shares)),
    }


def compute_elbow_scores(k_values: list[int], errors: list[float]) -> np.ndarray:
    """Distance to the line joining the first and last (K, error) points."""
    x = np.asarray(k_values, dtype=float)
    y = np.asarray(errors, dtype=float)

    if len(x) <= 2 or np.allclose(y, y[0]):
        return np.zeros(len(x), dtype=float)

    xn = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)

    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    line = p2 - p1
    denom = np.linalg.norm(line)

    scores = []
    for xx, yy in zip(xn, yn):
        p = np.array([xx, yy])
        v = p - p1
        d = abs(line[0] * v[1] - line[1] * v[0]) / max(denom, 1e-12)
        scores.append(float(d))

    return np.asarray(scores)


def select_k(metrics: pd.DataFrame, min_cluster_share: float) -> int:
    valid = metrics[metrics["min_cluster_share"] >= min_cluster_share].copy()
    if valid.empty:
        valid = metrics.copy()

    # Primary rule: elbow among non-fragmented solutions.
    best_elbow = valid["elbow_score"].max()
    candidates = valid[np.isclose(valid["elbow_score"], best_elbow)]

    # Tie-breaker: higher silhouette, then smaller K.
    candidates = candidates.sort_values(
        ["silhouette", "k"], ascending=[False, True]
    )
    return int(candidates.iloc[0]["k"])


def reorder_centers_by_area(centers: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    area = np.trapz(centers, GRID, axis=1)
    order = np.argsort(area, kind="mergesort")
    ordered = centers[order]
    old_to_new = {int(old): int(new) for new, old in enumerate(order)}
    return ordered, old_to_new


def map_labels(labels: np.ndarray, old_to_new: dict[int, int]) -> np.ndarray:
    lut = np.empty(len(old_to_new), dtype=np.int32)
    for old, new in old_to_new.items():
        lut[old] = new
    return lut[labels]


def template_name(cluster_id: int) -> str:
    return f"T{cluster_id:02d}"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--in-root", default="data/processed/bidtemplate"
    )
    parser.add_argument(
        "--sample-size", type=int, default=300_000,
        help="Representative shape sample used for K selection and final center fitting.",
    )
    parser.add_argument(
        "--candidate-k", default="4,6,8,10,12,16,20"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Optional fixed K. If omitted, K is selected automatically.",
    )
    parser.add_argument(
        "--chunksize", type=int, default=100_000
    )
    parser.add_argument(
        "--batch-size", type=int, default=8192
    )
    parser.add_argument(
        "--silhouette-sample", type=int, default=5000
    )
    parser.add_argument(
        "--min-cluster-share", type=float, default=0.005,
        help="Preferred minimum share for each shape cluster during automatic K selection.",
    )
    parser.add_argument(
        "--seed", type=int, default=RNG_SEED
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Optional test mode: use only the first N curve-sample files.",
    )
    args = parser.parse_args()

    year_dir = Path(args.in_root) / str(args.year)
    sample_dir = year_dir / "curve_samples"
    files = sorted(sample_dir.glob("curve_samples_*.csv"))

    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(f"No curve sample CSV files under {sample_dir}")

    out_dir = ensure_dir(year_dir / "template_library")
    assignment_dir = ensure_dir(out_dir / "assignments")

    k_values = parse_k_list(args.candidate_k)
    if args.k is not None and args.k < 2:
        raise ValueError("--k must be >= 2")

    total_shape_hint = None if args.max_files else read_total_shape_count(year_dir)

    print("=" * 72)
    print("Stage 2 curve-template clustering")
    print("=" * 72)
    print(f"Files:              {len(files)}")
    print(f"Target sample size: {args.sample_size:,}")
    print(f"Candidate K:        {k_values}")
    print(f"Fixed K:            {args.k if args.k is not None else 'auto'}")
    if total_shape_hint is not None:
        print(f"Shape rows hint:    {total_shape_hint:,}")

    X = collect_shape_sample(
        files=files,
        target_size=args.sample_size,
        chunksize=args.chunksize,
        seed=args.seed,
        total_shape_hint=total_shape_hint,
    )

    print()
    print(f"Representative sample: {len(X):,} x {X.shape[1]}")

    X_train, X_val = train_test_split(
        X,
        test_size=0.20,
        random_state=args.seed,
        shuffle=True,
    )

    metric_rows = []

    print()
    print("[candidate K evaluation]")

    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=args.seed,
            batch_size=args.batch_size,
            n_init=10,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        model.fit(X_train)

        labels, ev = eval_model(model, X_val)

        sil_n = min(args.silhouette_sample, len(X_val))
        if sil_n >= max(100, 2 * k):
            sil = float(
                silhouette_score(
                    X_val,
                    labels,
                    sample_size=sil_n,
                    random_state=args.seed,
                )
            )
        else:
            sil = np.nan

        metric_rows.append(
            {
                "k": k,
                "val_shape_mae": ev["shape_mae"],
                "val_shape_rmse": ev["shape_rmse"],
                "silhouette": sil,
                "min_cluster_share": ev["min_cluster_share"],
                "max_cluster_share": ev["max_cluster_share"],
            }
        )

        print(
            f"  K={k:2d}  "
            f"MAE={ev['shape_mae']:.6f}  "
            f"RMSE={ev['shape_rmse']:.6f}  "
            f"sil={sil:.4f}  "
            f"min_share={ev['min_cluster_share']:.3%}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows).sort_values("k").reset_index(drop=True)
    metrics["elbow_score"] = compute_elbow_scores(
        metrics["k"].tolist(), metrics["val_shape_rmse"].tolist()
    )

    selected_k = (
        int(args.k)
        if args.k is not None
        else select_k(metrics, args.min_cluster_share)
    )
    metrics["selected_flag"] = (metrics["k"] == selected_k).astype(int)

    metric_file = out_dir / "candidate_k_metrics.csv"
    metrics.to_csv(metric_file, index=False, encoding="utf-8-sig")

    print()
    print(f"Selected K = {selected_k}")
    print(f"Candidate metrics: {metric_file}")

    # Final centers: fit selected K on the full representative sample.
    final_model = MiniBatchKMeans(
        n_clusters=selected_k,
        random_state=args.seed,
        batch_size=args.batch_size,
        n_init=20,
        max_iter=300,
        reassignment_ratio=0.01,
    )
    final_model.fit(X)

    ordered_centers, old_to_new = reorder_centers_by_area(
        final_model.cluster_centers_.astype(np.float64)
    )

    model_file = out_dir / "curve_template_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "old_to_new": old_to_new,
            "ordered_centers": ordered_centers,
            "shape_cols": SHAPE_COLS,
            "grid": GRID,
            "selected_k": selected_k,
            "seed": args.seed,
        },
        model_file,
    )

    # Streaming full-data assignment and aggregation.
    cluster_count = np.zeros(selected_k, dtype=np.int64)
    cluster_shape_mae_sum = np.zeros(selected_k, dtype=np.float64)
    cluster_shape_rmse_sum = np.zeros(selected_k, dtype=np.float64)
    cluster_price_mae_sum = np.zeros(selected_k, dtype=np.float64)
    cluster_price_rmse_sum = np.zeros(selected_k, dtype=np.float64)
    cluster_shape_mae_max = np.zeros(selected_k, dtype=np.float64)
    cluster_shape_rmse_max = np.zeros(selected_k, dtype=np.float64)

    total_shape = 0
    total_flat = 0
    total_written = 0

    base_cols = [
        "sample_id",
        "participant_id",
        "timestamp_utc",
        "timestamp_local",
        "local_date",
        "local_slot_seconds",
        "source_market",
        "market_product",
        "source_file",
        "source_row_index",
        "curve_mode",
        "template_family",
        "shape_cluster_eligible_flag",
        "q_anchor_mw",
        "q_span_mw",
        "p_anchor",
        "p_span",
    ]

    print()
    print("[full historical assignment]")

    for file_no, file in enumerate(files, start=1):
        print(f"[assign {file_no}/{len(files)}] {file.name}", flush=True)

        header = pd.read_csv(file, nrows=0)
        missing = [c for c in base_cols if c not in header.columns]
        if missing:
            raise KeyError(f"{file.name} missing required columns: {missing}")

        usecols = base_cols + SHAPE_COLS
        out_file = assignment_dir / f"template_assignments_{file.stem}.csv"
        if out_file.exists():
            out_file.unlink()
        wrote_header = False

        file_shape = 0
        file_flat = 0
        file_written = 0

        for chunk in pd.read_csv(
            file,
            usecols=usecols,
            chunksize=args.chunksize,
            low_memory=False,
        ):
            fam = chunk["template_family"].astype(str)
            shape_mask = (
                fam.eq("shape")
                & pd.to_numeric(
                    chunk["shape_cluster_eligible_flag"], errors="coerce"
                ).fillna(0).eq(1)
            ).to_numpy()
            flat_mask = fam.eq("flat").to_numpy()

            out_parts = []

            if shape_mask.any():
                src = chunk.loc[shape_mask, base_cols].copy()
                Xc = chunk.loc[shape_mask, SHAPE_COLS].to_numpy(np.float32)
                finite = np.all(np.isfinite(Xc), axis=1)

                if not finite.all():
                    src = src.loc[finite].copy()
                    Xc = Xc[finite]

                if len(Xc):
                    old_labels = final_model.predict(Xc)
                    labels = map_labels(old_labels, old_to_new)
                    centers = ordered_centers[labels]
                    diff = Xc - centers
                    shape_mae = np.mean(np.abs(diff), axis=1)
                    shape_rmse = np.sqrt(np.mean(diff * diff, axis=1))
                    p_span_abs = np.abs(
                        pd.to_numeric(src["p_span"], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(float)
                    )
                    price_mae = shape_mae * p_span_abs
                    price_rmse = shape_rmse * p_span_abs

                    src["template_id"] = [template_name(int(x)) for x in labels]
                    src["template_cluster"] = labels.astype(int)
                    src["shape_mae"] = shape_mae
                    src["shape_rmse"] = shape_rmse
                    src["price_mae"] = price_mae
                    src["price_rmse"] = price_rmse
                    out_parts.append(src)

                    for cid in range(selected_k):
                        m = labels == cid
                        if not m.any():
                            continue
                        n = int(m.sum())
                        cluster_count[cid] += n
                        cluster_shape_mae_sum[cid] += float(shape_mae[m].sum())
                        cluster_shape_rmse_sum[cid] += float(shape_rmse[m].sum())
                        cluster_price_mae_sum[cid] += float(price_mae[m].sum())
                        cluster_price_rmse_sum[cid] += float(price_rmse[m].sum())
                        cluster_shape_mae_max[cid] = max(
                            cluster_shape_mae_max[cid], float(shape_mae[m].max())
                        )
                        cluster_shape_rmse_max[cid] = max(
                            cluster_shape_rmse_max[cid], float(shape_rmse[m].max())
                        )

                    file_shape += len(src)

            if flat_mask.any():
                flat = chunk.loc[flat_mask, base_cols].copy()
                flat["template_id"] = "FLAT"
                flat["template_cluster"] = -1
                flat["shape_mae"] = np.nan
                flat["shape_rmse"] = np.nan
                flat["price_mae"] = 0.0
                flat["price_rmse"] = 0.0
                out_parts.append(flat)
                file_flat += len(flat)

            if out_parts:
                out = pd.concat(out_parts, ignore_index=True)
                out.to_csv(
                    out_file,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                    encoding="utf-8-sig",
                    float_format="%.10g",
                )
                wrote_header = True
                file_written += len(out)

        total_shape += file_shape
        total_flat += file_flat
        total_written += file_written

        print(
            f"  shape={file_shape:,}, flat={file_flat:,}, written={file_written:,}",
            flush=True,
        )

    # Cluster summary from all assigned shape rows.
    summary_rows = []
    for cid in range(selected_k):
        n = int(cluster_count[cid])
        summary_rows.append(
            {
                "template_id": template_name(cid),
                "template_family": "shape",
                "sample_count": n,
                "share_of_shape": n / total_shape if total_shape else np.nan,
                "share_of_all": n / total_written if total_written else np.nan,
                "mean_shape_mae": cluster_shape_mae_sum[cid] / n if n else np.nan,
                "mean_shape_rmse": cluster_shape_rmse_sum[cid] / n if n else np.nan,
                "mean_price_mae": cluster_price_mae_sum[cid] / n if n else np.nan,
                "mean_price_rmse": cluster_price_rmse_sum[cid] / n if n else np.nan,
                "max_shape_mae": cluster_shape_mae_max[cid] if n else np.nan,
                "max_shape_rmse": cluster_shape_rmse_max[cid] if n else np.nan,
            }
        )

    summary_rows.append(
        {
            "template_id": "FLAT",
            "template_family": "flat",
            "sample_count": total_flat,
            "share_of_shape": np.nan,
            "share_of_all": total_flat / total_written if total_written else np.nan,
            "mean_shape_mae": np.nan,
            "mean_shape_rmse": np.nan,
            "mean_price_mae": 0.0,
            "mean_price_rmse": 0.0,
            "max_shape_mae": np.nan,
            "max_shape_rmse": np.nan,
        }
    )

    summary = pd.DataFrame(summary_rows)
    summary_file = out_dir / "curve_template_cluster_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    # Final template library: ordered shape centers + explicit FLAT family.
    library_rows = []
    summary_lookup = summary.set_index("template_id")

    for cid, center in enumerate(ordered_centers):
        tid = template_name(cid)
        row = {
            "template_id": tid,
            "template_family": "shape",
            "template_cluster": cid,
            "sample_count": int(summary_lookup.loc[tid, "sample_count"]),
            "share_of_shape": float(summary_lookup.loc[tid, "share_of_shape"]),
            "share_of_all": float(summary_lookup.loc[tid, "share_of_all"]),
            "shape_area": float(np.trapz(center, GRID)),
            "shape_midpoint": float(center[10]),
        }
        row.update({c: float(v) for c, v in zip(SHAPE_COLS, center)})
        library_rows.append(row)

    flat_row = {
        "template_id": "FLAT",
        "template_family": "flat",
        "template_cluster": -1,
        "sample_count": int(total_flat),
        "share_of_shape": np.nan,
        "share_of_all": total_flat / total_written if total_written else np.nan,
        "shape_area": np.nan,
        "shape_midpoint": np.nan,
    }
    flat_row.update({c: np.nan for c in SHAPE_COLS})
    library_rows.append(flat_row)

    library = pd.DataFrame(library_rows)
    library_file = out_dir / "curve_template_library.csv"
    library.to_csv(library_file, index=False, encoding="utf-8-sig")

    print()
    print("=" * 72)
    print("Curve-template clustering complete")
    print("=" * 72)
    print(f"Selected shape K:  {selected_k}")
    print(f"Shape rows:        {total_shape:,}")
    print(f"Flat rows:         {total_flat:,}")
    print(f"All assignments:   {total_written:,}")
    print(f"Template classes:  {selected_k + 1} ({selected_k} shape + FLAT)")
    print(f"K metrics:         {metric_file}")
    print(f"Template library:  {library_file}")
    print(f"Cluster summary:   {summary_file}")
    print(f"Model:             {model_file}")
    print(f"Assignments:       {assignment_dir}")


if __name__ == "__main__":
    main()
