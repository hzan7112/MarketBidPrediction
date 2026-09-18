#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_validate_template_representatives.py

Validate the real PJM offer modes inside each template and find TWO real
historical representatives for every shape template:
    1) nearest real block curve to the KMeans centroid;
    2) nearest real sloped curve to the KMeans centroid.

This script does NOT re-cluster data. It reuses the model produced by
02_cluster_curve_templates.py.

Outputs
-------
data/processed/bidtemplate/<year>/template_library/validation/
    template_mode_composition.csv
    template_representative_samples.csv
    template_mode_composition.png
    centroid_vs_mode_representatives.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID = np.linspace(0.0, 1.0, 21)
SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
MODES = ("block", "sloped")

META_COLS = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def template_name(cid: int) -> str:
    return f"T{cid:02d}"


def map_labels(old_labels: np.ndarray, old_to_new) -> np.ndarray:
    if isinstance(old_to_new, dict):
        return np.asarray([old_to_new[int(x)] for x in old_labels], dtype=int)
    mapping = np.asarray(old_to_new, dtype=int)
    return mapping[np.asarray(old_labels, dtype=int)]


def load_model(model_file: Path):
    payload = joblib.load(model_file)
    required = ["model", "old_to_new", "ordered_centers", "selected_k"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Model file missing keys: {missing}")

    model = payload["model"]
    old_to_new = payload["old_to_new"]
    centers = np.asarray(payload["ordered_centers"], dtype=np.float32)
    selected_k = int(payload["selected_k"])

    if centers.shape != (selected_k, 21):
        raise ValueError(
            f"Unexpected ordered_centers shape {centers.shape}; "
            f"expected ({selected_k}, 21)."
        )
    return model, old_to_new, centers, selected_k


def scan_all_samples(files, model, old_to_new, centers, selected_k, chunksize):
    # columns: block, sloped, other
    mode_counts = np.zeros((selected_k, 3), dtype=np.int64)
    flat_mode_counts = np.zeros(3, dtype=np.int64)

    # nearest representative separately for block/sloped
    best_mae = {
        mode: np.full(selected_k, np.inf, dtype=float)
        for mode in MODES
    }
    best_rmse = {
        mode: np.full(selected_k, np.inf, dtype=float)
        for mode in MODES
    }
    best_rows = {
        mode: [None] * selected_k
        for mode in MODES
    }

    total_shape = 0
    total_flat = 0
    usecols = META_COLS + SHAPE_COLS

    for file_no, file in enumerate(files, start=1):
        print(f"[scan {file_no}/{len(files)}] {file.name}", flush=True)

        header = pd.read_csv(file, nrows=0)
        missing = [c for c in usecols if c not in header.columns]
        if missing:
            raise KeyError(f"{file.name} missing required columns: {missing}")

        file_shape = 0
        file_flat = 0

        for chunk in pd.read_csv(
            file,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ):
            family = chunk["template_family"].astype(str)
            mode_series = chunk["curve_mode"].astype(str).str.lower()

            shape_mask = (
                family.eq("shape")
                & pd.to_numeric(
                    chunk["shape_cluster_eligible_flag"], errors="coerce"
                ).fillna(0).eq(1)
            ).to_numpy()
            flat_mask = family.eq("flat").to_numpy()

            if shape_mask.any():
                src = chunk.loc[shape_mask].copy()
                X = src[SHAPE_COLS].to_numpy(np.float32)
                finite = np.all(np.isfinite(X), axis=1)
                if not finite.all():
                    src = src.loc[finite].copy()
                    X = X[finite]

                if len(X):
                    labels = map_labels(model.predict(X), old_to_new)
                    C = centers[labels]
                    diff = X - C
                    mae = np.mean(np.abs(diff), axis=1)
                    rmse = np.sqrt(np.mean(diff * diff, axis=1))
                    modes = src["curve_mode"].astype(str).str.lower().to_numpy()

                    for cid in range(selected_k):
                        cluster_mask = labels == cid
                        if not cluster_mask.any():
                            continue

                        cm = modes[cluster_mask]
                        mode_counts[cid, 0] += int(np.sum(cm == "block"))
                        mode_counts[cid, 1] += int(np.sum(cm == "sloped"))
                        mode_counts[cid, 2] += int(
                            np.sum(~np.isin(cm, ["block", "sloped"]))
                        )

                        for wanted_mode in MODES:
                            m = cluster_mask & (modes == wanted_mode)
                            if not m.any():
                                continue

                            idx = np.flatnonzero(m)
                            j = int(idx[np.argmin(mae[m])])
                            if float(mae[j]) >= best_mae[wanted_mode][cid]:
                                continue

                            best_mae[wanted_mode][cid] = float(mae[j])
                            best_rmse[wanted_mode][cid] = float(rmse[j])
                            row = src.iloc[j]
                            best_rows[wanted_mode][cid] = {
                                **{c: row[c] for c in META_COLS},
                                "template_id": template_name(cid),
                                "template_cluster": cid,
                                "representative_mode": wanted_mode,
                                "representative_shape_mae": float(mae[j]),
                                "representative_shape_rmse": float(rmse[j]),
                                **{c: float(row[c]) for c in SHAPE_COLS},
                            }

                    n = len(X)
                    file_shape += n
                    total_shape += n

            if flat_mask.any():
                fm = mode_series.loc[flat_mask].to_numpy()
                flat_mode_counts[0] += int(np.sum(fm == "block"))
                flat_mode_counts[1] += int(np.sum(fm == "sloped"))
                flat_mode_counts[2] += int(
                    np.sum(~np.isin(fm, ["block", "sloped"]))
                )
                n = int(flat_mask.sum())
                file_flat += n
                total_flat += n

        print(f"  shape={file_shape:,}, flat={file_flat:,}", flush=True)

    return (
        mode_counts,
        flat_mode_counts,
        best_rows,
        best_mae,
        best_rmse,
        total_shape,
        total_flat,
    )


def build_mode_table(mode_counts, flat_mode_counts):
    rows = []
    for cid in range(len(mode_counts)):
        block, sloped, other = [int(x) for x in mode_counts[cid]]
        total = block + sloped + other
        rows.append({
            "template_id": template_name(cid),
            "template_family": "shape",
            "sample_count": total,
            "block_count": block,
            "sloped_count": sloped,
            "other_mode_count": other,
            "block_share": block / total if total else np.nan,
            "sloped_share": sloped / total if total else np.nan,
            "other_mode_share": other / total if total else np.nan,
        })

    block, sloped, other = [int(x) for x in flat_mode_counts]
    total = block + sloped + other
    rows.append({
        "template_id": "FLAT",
        "template_family": "flat",
        "sample_count": total,
        "block_count": block,
        "sloped_count": sloped,
        "other_mode_count": other,
        "block_share": block / total if total else np.nan,
        "sloped_share": sloped / total if total else np.nan,
        "other_mode_share": other / total if total else np.nan,
    })
    return pd.DataFrame(rows)


def build_representative_table(best_rows, centers):
    rows = []
    for cid in range(len(centers)):
        for mode in MODES:
            row = best_rows[mode][cid]
            if row is None:
                continue

            out = dict(row)
            for i, v in enumerate(centers[cid]):
                out[f"centroid_v{i:02d}"] = float(v)

            q0 = pd.to_numeric(pd.Series([out["q_anchor_mw"]]), errors="coerce").iloc[0]
            qspan = pd.to_numeric(pd.Series([out["q_span_mw"]]), errors="coerce").iloc[0]
            p0 = pd.to_numeric(pd.Series([out["p_anchor"]]), errors="coerce").iloc[0]
            pspan = pd.to_numeric(pd.Series([out["p_span"]]), errors="coerce").iloc[0]
            rep_shape = np.asarray([out[c] for c in SHAPE_COLS], dtype=float)

            if np.isfinite(q0) and np.isfinite(qspan):
                for i, v in enumerate(q0 + qspan * GRID):
                    out[f"representative_q_v{i:02d}"] = float(v)
            if np.isfinite(p0) and np.isfinite(pspan):
                for i, v in enumerate(p0 + pspan * rep_shape):
                    out[f"representative_p_v{i:02d}"] = float(v)

            rows.append(out)

    return pd.DataFrame(rows)


def plot_mode_composition(mode_df: pd.DataFrame, out_file: Path):
    labels = mode_df["template_id"].tolist()
    block = mode_df["block_share"].to_numpy(float) * 100
    sloped = mode_df["sloped_share"].to_numpy(float) * 100
    other = mode_df["other_mode_share"].fillna(0).to_numpy(float) * 100

    x = np.arange(len(mode_df))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x, block, label="block")
    ax.bar(x, sloped, bottom=block, label="sloped")
    if np.any(other > 0):
        ax.bar(x, other, bottom=block + sloped, label="other")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share within template (%)")
    ax.set_xlabel("Template")
    ax.set_title("Block vs sloped composition inside each template")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def plot_centroid_vs_modes(rep_df, mode_df, out_file: Path, ncols: int = 4):
    if rep_df.empty:
        return

    tids = sorted(
        rep_df["template_id"].unique(),
        key=lambda x: int(str(x).replace("T", "")),
    )
    n = len(tids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.9 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()
    centroid_cols = [f"centroid_v{i:02d}" for i in range(21)]
    mode_lookup = mode_df.set_index("template_id")

    for ax, tid in zip(axes, tids):
        g = rep_df[rep_df["template_id"] == tid]
        center = g.iloc[0][centroid_cols].to_numpy(float)
        ax.plot(GRID, center, linestyle="--", linewidth=2.2, label="centroid")

        for mode in MODES:
            gm = g[g["representative_mode"] == mode]
            if gm.empty:
                continue
            row = gm.iloc[0]
            y = row[SHAPE_COLS].to_numpy(float)
            label = f"real {mode}  MAE={row['representative_shape_mae']:.4f}"
            if mode == "block":
                ax.step(GRID, y, where="post", linewidth=1.8, label=label)
            else:
                ax.plot(GRID, y, linewidth=1.8, label=label)
            ax.scatter(GRID, y, s=8)

        comp = mode_lookup.loc[tid]
        ax.set_title(
            f"{tid}  block={comp['block_share']:.1%}  "
            f"sloped={comp['sloped_share']:.1%}",
            fontsize=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Normalized quantity x")
        ax.set_ylabel("Normalized price shape")
        ax.legend(fontsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "KMeans centroid vs nearest real block and sloped bid curves",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--root", default="data/processed/bidtemplate")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--gallery-cols", type=int, default=4)
    args = parser.parse_args()

    year_dir = Path(args.root) / str(args.year)
    sample_dir = year_dir / "curve_samples"
    template_dir = year_dir / "template_library"
    model_file = template_dir / "curve_template_model.joblib"
    out_dir = ensure_dir(template_dir / "validation")

    files = sorted(sample_dir.glob("curve_samples_*.csv"))
    if not files:
        raise FileNotFoundError(f"No curve sample CSVs under {sample_dir}")
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model: {model_file}")

    model, old_to_new, centers, selected_k = load_model(model_file)

    print("=" * 72)
    print("Template mode-aware validation")
    print("=" * 72)
    print(f"Files:       {len(files)}")
    print(f"Selected K:  {selected_k}")
    print(f"Model:       {model_file}")
    print()

    (
        mode_counts,
        flat_mode_counts,
        best_rows,
        best_mae,
        best_rmse,
        total_shape,
        total_flat,
    ) = scan_all_samples(
        files,
        model,
        old_to_new,
        centers,
        selected_k,
        args.chunksize,
    )

    mode_df = build_mode_table(mode_counts, flat_mode_counts)
    rep_df = build_representative_table(best_rows, centers)

    mode_file = out_dir / "template_mode_composition.csv"
    rep_file = out_dir / "template_representative_samples.csv"
    mode_fig = out_dir / "template_mode_composition.png"
    rep_fig = out_dir / "centroid_vs_mode_representatives.png"

    mode_df.to_csv(mode_file, index=False, encoding="utf-8-sig")
    rep_df.to_csv(rep_file, index=False, encoding="utf-8-sig", float_format="%.10g")
    plot_mode_composition(mode_df, mode_fig)
    plot_centroid_vs_modes(rep_df, mode_df, rep_fig, ncols=args.gallery_cols)

    print()
    print("[mode composition]")
    show = mode_df[[
        "template_id", "sample_count", "block_share", "sloped_share"
    ]].copy()
    show["block_share"] = show["block_share"].map(lambda x: f"{x:.2%}")
    show["sloped_share"] = show["sloped_share"].map(lambda x: f"{x:.2%}")
    print(show.to_string(index=False))

    print()
    print("[real representatives by mode]")
    print(
        rep_df[[
            "template_id",
            "representative_mode",
            "sample_id",
            "participant_id",
            "representative_shape_mae",
        ]].to_string(index=False)
    )

    print()
    print("=" * 72)
    print("Template validation complete")
    print("=" * 72)
    print(f"Shape rows:       {total_shape:,}")
    print(f"Flat rows:        {total_flat:,}")
    print(f"Mode table:       {mode_file}")
    print(f"Representatives:  {rep_file}")
    print(f"Mode figure:      {mode_fig}")
    print(f"Representative figure: {rep_fig}")


if __name__ == "__main__":
    main()
