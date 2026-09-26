#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07a_filter_macro_b_curves.py

Macro-B oracle screening experiment:
retain only regular staircase / mildly increasing bid curves.

A modeling row is retained only when BOTH:
    current curve      is Macro-B
    previous raw curve is Macro-B

This intentionally removes cross-family transitions so the experiment answers:
"Does the existing 06 curve-change regression become useful after reducing
curve-family heterogeneity?"

Macro-B is a bid-behavior family, NOT a fuel/resource type.

Threshold policy
----------------
Hard generic rules:
- every price point > 0
- non-flat
- approximately nondecreasing
- >= 2 meaningful upward steps

TRAIN-only learned upper thresholds:
- max single-step share: Q90
- normalized roughness: Q90
- tail-uplift share: Q95

Run:
python scripts/bidprediction/07a_filter_macro_b_curves.py --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SHAPE_COLS = [f"shape_v{i:02d}" for i in range(21)]
PREV_PRICE_COLS = [f"_curvevec_p{i:02d}_lag1" for i in range(21)]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def files_from_manifest(manifest, split):
    return [
        x["file"] if isinstance(x, dict) else x
        for x in manifest["parts"][split]
    ]


def current_price(d):
    shape = d[SHAPE_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(np.float64)

    pa = num(d["p_anchor"]).to_numpy(np.float64)
    ps = num(d["p_span"]).to_numpy(np.float64)

    zero = np.abs(ps) <= 1e-12
    if zero.any():
        shape[zero] = np.nan_to_num(
            shape[zero],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return pa[:, None] + ps[:, None] * shape


def previous_price(d):
    return np.column_stack(
        [
            num(d[c]).to_numpy(np.float64)
            for c in PREV_PRICE_COLS
        ]
    )


def curve_features(price):
    price = np.asarray(price, dtype=np.float64)
    finite = np.isfinite(price).all(axis=1)

    pmin = np.nanmin(price, axis=1)
    pmax = np.nanmax(price, axis=1)
    span = pmax - pmin
    scale = np.maximum(span, 1e-8)

    diff = np.diff(price, axis=1)
    tol = np.maximum(
        1e-8,
        1e-4 * scale,
    )

    positive = np.maximum(diff, 0.0)

    second = np.diff(
        price,
        n=2,
        axis=1,
    )

    return pd.DataFrame(
        {
            "finite": finite,
            "min_price": pmin,
            "span": span,
            "monotone_violation_share": (
                diff < -tol[:, None]
            ).mean(axis=1),
            "max_step_share": (
                positive.max(axis=1)
                / scale
            ),
            "significant_step_count": (
                positive
                > (
                    0.02
                    * scale[:, None]
                )
            ).sum(axis=1),
            "roughness": (
                np.mean(
                    np.abs(second),
                    axis=1,
                )
                / scale
            ),
            "tail_share": (
                price[:, 20]
                - price[:, 14]
            ) / scale,
        }
    )


def base_mask(f, cfg):
    return (
        f["finite"].astype(bool)
        & (
            f["min_price"]
            > cfg["min_price"]
        )
        & (
            f["span"]
            > 1e-8
        )
        & (
            f["monotone_violation_share"]
            <= cfg["max_monotone_violation"]
        )
        & (
            f["significant_step_count"]
            >= cfg["min_significant_steps"]
        )
    )


def macro_b_mask(f, cfg):
    b = base_mask(f, cfg)

    return (
        b
        & (
            f["max_step_share"]
            <= cfg["max_step_share_max"]
        )
        & (
            f["roughness"]
            <= cfg["roughness_max"]
        )
        & (
            f["tail_share"]
            <= cfg["tail_share_max"]
        )
    )


def fit_thresholds(source, manifest, args):
    base_cfg = {
        "min_price": float(args.min_price),
        "max_monotone_violation": float(
            args.max_monotone_violation
        ),
        "min_significant_steps": int(
            args.min_significant_steps
        ),
    }

    blocks = []

    train_files = files_from_manifest(
        manifest,
        "train",
    )

    for i, rel in enumerate(train_files, 1):
        p = source / rel
        print(
            f"[fit {i}/{len(train_files)}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(p)
        f = curve_features(
            current_price(d)
        )

        m = base_mask(
            f,
            base_cfg,
        )

        if m.any():
            blocks.append(
                f.loc[
                    m,
                    [
                        "max_step_share",
                        "roughness",
                        "tail_share",
                    ],
                ].copy()
            )

        del d, f
        gc.collect()

    if not blocks:
        raise ValueError(
            "No TRAIN rows satisfy Macro-B base rules."
        )

    x = pd.concat(
        blocks,
        ignore_index=True,
    )

    if len(x) > args.max_fit_rows:
        x = x.sample(
            n=args.max_fit_rows,
            random_state=args.seed,
        ).reset_index(drop=True)

    cfg = {
        **base_cfg,
        "shape_quantile": float(
            args.shape_quantile
        ),
        "tail_quantile": float(
            args.tail_quantile
        ),
        "max_step_share_max": float(
            x["max_step_share"].quantile(
                args.shape_quantile
            )
        ),
        "roughness_max": float(
            x["roughness"].quantile(
                args.shape_quantile
            )
        ),
        "tail_share_max": float(
            x["tail_share"].quantile(
                args.tail_quantile
            )
        ),
        "fit_candidate_rows": int(len(x)),
        "threshold_source": "TRAIN only",
    }

    return cfg


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--source-dir",
        default="curve_change_latent_dataset",
    )
    ap.add_argument(
        "--min-price",
        type=float,
        default=0.0,
    )
    ap.add_argument(
        "--max-monotone-violation",
        type=float,
        default=0.05,
    )
    ap.add_argument(
        "--min-significant-steps",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--shape-quantile",
        type=float,
        default=0.90,
    )
    ap.add_argument(
        "--tail-quantile",
        type=float,
        default=0.95,
    )
    ap.add_argument(
        "--max-fit-rows",
        type=int,
        default=500_000,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    source = base / args.source_dir

    manifest = json.loads(
        (
            source
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    out_dir = (
        base
        / "macro_b_filtered_dataset"
    )

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out_dir} exists. Use --overwrite."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"07a Macro-B filtering - {args.year}"
    )
    print("=" * 80)

    cfg = fit_thresholds(
        source,
        manifest,
        args,
    )

    (
        out_dir
        / "macro_b_thresholds.json"
    ).write_text(
        json.dumps(
            cfg,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            cfg,
            ensure_ascii=False,
            indent=2,
        )
    )

    out_manifest = {
        "year": args.year,
        "source_dataset": str(source),
        "macro_family": "Macro-B",
        "definition": (
            "positive, non-flat, approximately nondecreasing, "
            "multi-step, non-extreme staircase/mild-increase"
        ),
        "thresholds": cfg,
        "source_model_features": manifest.get(
            "source_model_features",
            manifest.get(
                "model_features",
                [],
            ),
        ),
        "parts": {},
    }

    stats = []
    template_parts = []

    for split in ["train", "val", "test"]:
        src_files = files_from_manifest(
            manifest,
            split,
        )

        split_dir = (
            out_dir
            / "parts"
            / split
        )
        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        written = []

        n_total = 0
        n_current = 0
        n_prev = 0
        n_both = 0

        for i, rel in enumerate(src_files, 1):
            p = source / rel

            print(
                f"[{split} {i}/{len(src_files)}] {p.name}",
                flush=True,
            )

            d = pd.read_pickle(p)

            cf = curve_features(
                current_price(d)
            )
            pf = curve_features(
                previous_price(d)
            )

            cm = macro_b_mask(
                cf,
                cfg,
            ).to_numpy(bool)

            pm = macro_b_mask(
                pf,
                cfg,
            ).to_numpy(bool)

            both = cm & pm

            n_total += len(d)
            n_current += int(cm.sum())
            n_prev += int(pm.sum())
            n_both += int(both.sum())

            if "y_template_id" in d.columns:
                t = pd.DataFrame(
                    {
                        "split": split,
                        "template_id": (
                            d["y_template_id"]
                            .astype("string")
                        ),
                        "rows": 1,
                        "current_macro_b": cm.astype(
                            np.int8
                        ),
                        "stable_macro_b": both.astype(
                            np.int8
                        ),
                    }
                )

                template_parts.append(
                    t.groupby(
                        [
                            "split",
                            "template_id",
                        ],
                        dropna=False,
                        as_index=False,
                    )[
                        [
                            "rows",
                            "current_macro_b",
                            "stable_macro_b",
                        ]
                    ].sum()
                )

            keep = (
                d.loc[both]
                .copy()
                .reset_index(drop=True)
            )

            if keep.empty:
                continue

            name = (
                f"{split}_macro_b_"
                f"{i:04d}.pkl"
            )

            path = split_dir / name
            keep.to_pickle(path)

            written.append(
                {
                    "file": str(
                        path.relative_to(
                            out_dir
                        )
                    ),
                    "rows": int(len(keep)),
                    "source_file": str(rel),
                }
            )

            del d, cf, pf, keep
            gc.collect()

        out_manifest["parts"][split] = written

        stats.append(
            {
                "split": split,
                "source_rows": int(n_total),
                "current_macro_b_rows": int(
                    n_current
                ),
                "current_macro_b_share": (
                    n_current
                    / max(
                        n_total,
                        1,
                    )
                ),
                "previous_macro_b_rows": int(
                    n_prev
                ),
                "previous_macro_b_share": (
                    n_prev
                    / max(
                        n_total,
                        1,
                    )
                ),
                "stable_macro_b_rows": int(
                    n_both
                ),
                "stable_macro_b_share": (
                    n_both
                    / max(
                        n_total,
                        1,
                    )
                ),
            }
        )

    stat_df = pd.DataFrame(stats)

    stat_df.to_csv(
        out_dir
        / "split_filter_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if template_parts:
        tmp = pd.concat(
            template_parts,
            ignore_index=True,
        )

        tmp = (
            tmp.groupby(
                [
                    "split",
                    "template_id",
                ],
                as_index=False,
            )[
                [
                    "rows",
                    "current_macro_b",
                    "stable_macro_b",
                ]
            ].sum()
        )

        tmp[
            "current_macro_b_share"
        ] = (
            tmp["current_macro_b"]
            / tmp["rows"]
        )

        tmp[
            "stable_macro_b_share"
        ] = (
            tmp["stable_macro_b"]
            / tmp["rows"]
        )

        tmp.to_csv(
            out_dir
            / "template_composition.csv",
            index=False,
            encoding="utf-8-sig",
        )

    out_manifest[
        "split_statistics"
    ] = stats

    (
        out_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            out_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            (
                f"07a Macro-B filtering - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            "Frozen TRAIN-only thresholds:",
            json.dumps(
                cfg,
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Filtering statistics:",
            stat_df.to_string(
                index=False
            ),
            "",
            (
                "Retained modeling rows require BOTH current and previous "
                "curves to be Macro-B."
            ),
        ]
    )

    (
        out_dir
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
