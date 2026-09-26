#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10c_train_final_curve_regressor.py

Final lightweight regressors for the full feature-only bid-curve task.

FINAL primary model:
    Random Forest

Audit baselines:
    train mean
    Ridge
    Spline-GAM

No model selection is performed here. Random Forest is frozen as the primary
model by the 04-09 route convergence. Baselines are retained only for final
comparison.

Input
-----
data/processed/bidprediction/<year>/final_absolute_curve_dataset/

Output
------
data/processed/bidprediction/<year>/final_curve_regression_models/
    train_mean_baseline.joblib
    ridge_model.joblib
    spline_gam_models.joblib
    random_forest_model.joblib
    prediction_parts/{val,test}/*.pkl
    val_latent_metrics.csv
    test_latent_metrics.csv
    manifest.json
    summary.txt

Run
---
python scripts/bidprediction/10c_train_final_curve_regressor.py \
    --year 2025 --overwrite
"""

from __future__ import annotations

import argparse
import re
import gc
import json
import math
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


META = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
    "prediction_cutoff_utc",
    "y_template_id",
]

SHAPE = [
    f"shape_v{i:02d}"
    for i in range(21)
]

CURVE = [
    *SHAPE,
    "p_anchor",
    "p_span",
    "q_anchor_mw",
    "q_span_mw",
]

FORBIDDEN = (
    "latent",
    "curvevec",
    "theta",
    "template",
    "shape_v",
    "p_anchor",
    "p_span",
    "q_anchor",
    "q_span",
)


def parts(manifest, split):
    return [
        x["file"]
        if isinstance(x, dict)
        else x
        for x in manifest["parts"][split]
    ]


def nframe(d, cols):
    return d[
        cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )


def leakage_check(features):
    """
    Precise final-input leakage guard.

    Allowed examples:
      lt_shape_variability
      tr_template_entropy_7d
      tr_template_entropy_30d
      tr_unique_template_count_7d
      tr_unique_template_count_30d
      tr_lag1_daily_template_entropy
      tr_lag1_daily_unique_template_count

    These are aggregate strategy-profile / strategy-transition features and
    belong to the frozen feature set.

    Forbidden:
      raw current curve targets (shape_v00..20, p_anchor, p_span,
      q_anchor_mw, q_span_mw);
      raw historical curve vectors (curvevec);
      curve latent/history targets (latent);
      theta targets/history;
      raw/current/lagged template IDs (names ending in template_id);
      explicit template inertia fields.
    """
    exact_forbidden = {
        "p_anchor",
        "p_span",
        "q_anchor_mw",
        "q_span_mw",
        "y_template_id",
        "template_id",
    }

    bad = []

    for f in features:
        low = str(f).lower()

        is_raw_shape_target = bool(
            re.fullmatch(r"shape_v\d{2}", low)
        )

        is_template_id = (
            low.endswith("template_id")
            or low.endswith("_template_id")
        )

        is_direct_history_or_target = any(
            token in low
            for token in (
                "curvevec",
                "latent",
                "theta",
            )
        )

        is_template_inertia = (
            "template_inertia" in low
        )

        if (
            low in exact_forbidden
            or is_raw_shape_target
            or is_template_id
            or is_direct_history_or_target
            or is_template_inertia
        ):
            bad.append(f)

    if bad:
        raise RuntimeError(
            "Forbidden direct bid-history/target information entered "
            "final model_features: "
            + ", ".join(
                bad[:40]
            )
        )


def load_train_sample(
    dataset,
    manifest,
    features,
    targets,
    max_rows,
    seed,
):
    fs = parts(
        manifest,
        "train",
    )

    per = max(
        1,
        int(
            math.ceil(
                max_rows
                / max(
                    len(fs),
                    1,
                )
            )
        ),
    )

    blocks = []

    for i, rel in enumerate(
        fs,
        1,
    ):
        path = (
            dataset
            / rel
        )

        print(
            f"[TRAIN sample {i}/{len(fs)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        if len(d) > per:
            d = d.sample(
                n=per,
                random_state=(
                    seed
                    + 7919 * i
                ),
            )

        blocks.append(
            d[
                [
                    *features,
                    *targets,
                ]
            ].copy()
        )

        del d
        gc.collect()

    if not blocks:
        raise ValueError(
            "No TRAIN rows."
        )

    train = pd.concat(
        blocks,
        ignore_index=True,
    )

    if len(train) > max_rows:
        train = (
            train.sample(
                n=max_rows,
                random_state=seed,
            )
            .reset_index(
                drop=True
            )
        )

    return train


def top_spearman(
    d,
    features,
    target,
    k,
):
    y = pd.to_numeric(
        d[
            target
        ],
        errors="coerce",
    )

    scores = []

    for c in features:
        x = pd.to_numeric(
            d[
                c
            ],
            errors="coerce",
        )

        ok = (
            x.notna()
            & y.notna()
        )

        if (
            ok.sum() < 100
            or x.loc[
                ok
            ].nunique()
            <= 1
        ):
            continue

        r = x.loc[
            ok
        ].corr(
            y.loc[
                ok
            ],
            method="spearman",
        )

        if pd.notna(
            r
        ):
            scores.append(
                (
                    c,
                    abs(
                        float(
                            r
                        )
                    ),
                    float(
                        r
                    ),
                )
            )

    scores.sort(
        key=lambda z: z[
            1
        ],
        reverse=True,
    )

    return scores[
        :k
    ]


def fit_ridge(
    train,
    features,
    targets,
    alpha,
):
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha
                ),
            ),
        ]
    )

    model.fit(
        nframe(
            train,
            features,
        ),
        nframe(
            train,
            targets,
        ).to_numpy(
            np.float64
        ),
    )

    return model


def fit_gam(
    train,
    features,
    targets,
    topk,
    knots,
    degree,
    alpha,
):
    models = {}
    selection = {}

    for i, target in enumerate(
        targets,
        1,
    ):
        print(
            f"[Spline-GAM {i}/{len(targets)}] "
            f"{target}",
            flush=True,
        )

        top = top_spearman(
            train,
            features,
            target,
            topk,
        )

        selected = [
            x[
                0
            ]
            for x in top
        ]

        if not selected:
            selected = features[
                :min(
                    topk,
                    len(
                        features
                    ),
                )
            ]

        model = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=knots,
                        degree=degree,
                        knots="quantile",
                        extrapolation="constant",
                        include_bias=False,
                        sparse_output=True,
                    ),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=alpha,
                        solver="lsqr",
                    ),
                ),
            ]
        )

        model.fit(
            nframe(
                train,
                selected,
            ),
            pd.to_numeric(
                train[
                    target
                ],
                errors="coerce",
            ).to_numpy(
                np.float64
            ),
        )

        models[
            target
        ] = {
            "features": selected,
            "model": model,
        }

        selection[
            target
        ] = {
            "features": selected,
            "top_context": [
                {
                    "feature": c,
                    "abs_spearman": a,
                    "spearman": r,
                }
                for c, a, r in top
            ],
        }

    return (
        models,
        selection,
    )


def predict_gam(
    d,
    models,
    targets,
):
    out = np.empty(
        (
            len(d),
            len(targets),
        ),
        dtype=np.float32,
    )

    for j, target in enumerate(
        targets
    ):
        spec = models[
            target
        ]

        out[
            :,
            j,
        ] = spec[
            "model"
        ].predict(
            nframe(
                d,
                spec[
                    "features"
                ],
            )
        ).astype(
            np.float32
        )

    return out


def latent_metric_row(
    split,
    model,
    true,
    pred,
):
    err = (
        pred
        - true
    )

    return {
        "split": split,
        "model": model,
        "rows": int(
            len(
                true
            )
        ),
        "latent_mae": float(
            np.mean(
                np.abs(
                    err
                )
            )
        ),
        "latent_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        err
                    )
                )
            )
        ),
    }


def predict_split(
    dataset,
    manifest,
    split,
    out,
    features,
    targets,
    mean_z,
    ridge,
    gam,
    rf,
):
    pred_dir = (
        out
        / "prediction_parts"
        / split
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = parts(
        manifest,
        split,
    )

    written = []

    metric_sums = {
        name: {
            "rows": 0,
            "abs": 0.0,
            "sq": 0.0,
            "points": 0,
        }
        for name in [
            "train_mean",
            "ridge",
            "spline_gam",
            "random_forest",
        ]
    }

    for i, rel in enumerate(
        files,
        1,
    ):
        path = (
            dataset
            / rel
        )

        print(
            f"[predict {split} {i}/{len(files)}] "
            f"{path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        X = nframe(
            d,
            features,
        )

        true = nframe(
            d,
            targets,
        ).to_numpy(
            np.float32
        )

        preds = {
            "train_mean": np.repeat(
                mean_z[
                    None,
                    :,
                ],
                len(d),
                axis=0,
            ).astype(
                np.float32
            ),
            "ridge": ridge.predict(
                X
            ).astype(
                np.float32
            ),
            "spline_gam": predict_gam(
                d,
                gam,
                targets,
            ),
            "random_forest": rf.predict(
                X
            ).astype(
                np.float32
            ),
        }

        for name, pred in preds.items():
            e = (
                pred
                - true
            )

            st = metric_sums[
                name
            ]

            st[
                "rows"
            ] += len(
                d
            )

            st[
                "abs"
            ] += float(
                np.abs(
                    e
                ).sum()
            )

            st[
                "sq"
            ] += float(
                np.square(
                    e
                ).sum()
            )

            st[
                "points"
            ] += int(
                e.size
            )

        keep = list(
            dict.fromkeys(
                [
                    *[
                        c
                        for c in META
                        if c in d.columns
                    ],
                    *[
                        c
                        for c in CURVE
                        if c in d.columns
                    ],
                    *[
                        c
                        for c in d.columns
                        if c.startswith(
                            "reference"
                        )
                    ],
                ]
            )
        )

        out_d = d[
            keep
        ].copy()

        for j, target in enumerate(
            targets
        ):
            out_d[
                f"true_{target}"
            ] = true[
                :,
                j,
            ]

            for name, pred in preds.items():
                out_d[
                    f"pred_{name}_{target}"
                ] = pred[
                    :,
                    j,
                ]

        name = (
            f"{split}_final_predictions_"
            f"{i:04d}.pkl"
        )

        out_path = (
            pred_dir
            / name
        )

        out_d.to_pickle(
            out_path,
            protocol=5,
        )

        written.append(
            {
                "file": str(
                    out_path.relative_to(
                        out
                    )
                ),
                "rows": int(
                    len(
                        out_d
                    )
                ),
            }
        )

        del (
            d,
            X,
            true,
            preds,
            out_d,
        )

        gc.collect()

    metrics = []

    for name, st in metric_sums.items():
        metrics.append(
            {
                "split": split,
                "model": name,
                "rows": int(
                    st[
                        "rows"
                    ]
                ),
                "latent_mae": float(
                    st[
                        "abs"
                    ]
                    / max(
                        st[
                            "points"
                        ],
                        1,
                    )
                ),
                "latent_rmse": float(
                    np.sqrt(
                        st[
                            "sq"
                        ]
                        / max(
                            st[
                                "points"
                            ],
                            1,
                        )
                    )
                ),
            }
        )

    return (
        written,
        pd.DataFrame(
            metrics
        ),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--year",
        type=int,
        default=2025,
    )
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--dataset-dir",
        default="final_absolute_curve_dataset",
    )
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=300_000,
    )
    ap.add_argument(
        "--gam-train-rows",
        type=int,
        default=150_000,
    )
    ap.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
    )
    ap.add_argument(
        "--gam-alpha",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--gam-top-context",
        type=int,
        default=24,
    )
    ap.add_argument(
        "--gam-knots",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--gam-degree",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--rf-trees",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--rf-max-depth",
        type=int,
        default=18,
    )
    ap.add_argument(
        "--rf-min-leaf",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--rf-max-features",
        type=float,
        default=0.5,
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = ap.parse_args()

    base = (
        Path(
            args.root
        )
        / str(
            args.year
        )
    )

    dataset = (
        base
        / args.dataset_dir
    )

    manifest = json.loads(
        (
            dataset
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    features = list(
        manifest[
            "model_features"
        ]
    )

    targets = list(
        manifest[
            "latent_columns"
        ]
    )

    leakage_check(
        features
    )

    if len(
        features
    ) != 83:
        raise RuntimeError(
            f"Final route expects 83 model features, got {len(features)}."
        )

    if len(
        targets
    ) != 8:
        raise RuntimeError(
            f"Final route expects 8 absolute latent targets, got {len(targets)}."
        )

    out = (
        base
        / "final_curve_regression_models"
    )

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} exists. Use --overwrite."
            )

        shutil.rmtree(
            out
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"10c Final curve regressor - {args.year}"
    )
    print("=" * 80)
    print(
        f"Model features = {len(features)}"
    )
    print(
        f"Absolute latent targets = {len(targets)}"
    )
    print(
        "Primary model = Random Forest (frozen)"
    )
    print(
        "Ridge / Spline-GAM / train mean = audit baselines only"
    )
    print(
        "Historical-bid leakage check = PASS"
    )
    print()

    train = load_train_sample(
        dataset,
        manifest,
        features,
        targets,
        args.max_train_rows,
        args.seed,
    )

    print(
        f"Shared TRAIN sample = {len(train):,}"
    )

    mean_z = (
        nframe(
            train,
            targets,
        )
        .mean(
            axis=0
        )
        .to_numpy(
            np.float64
        )
    )

    print(
        "[fit] Ridge",
        flush=True,
    )

    ridge = fit_ridge(
        train,
        features,
        targets,
        args.ridge_alpha,
    )

    if len(
        train
    ) > args.gam_train_rows:
        gam_train = (
            train.sample(
                n=args.gam_train_rows,
                random_state=(
                    args.seed
                    + 17
                ),
            )
            .reset_index(
                drop=True
            )
        )
    else:
        gam_train = train

    print(
        "[fit] Spline-GAM",
        flush=True,
    )

    gam, gam_selection = fit_gam(
        gam_train,
        features,
        targets,
        args.gam_top_context,
        args.gam_knots,
        args.gam_degree,
        args.gam_alpha,
    )

    print(
        "[fit] Random Forest",
        flush=True,
    )

    rf = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=args.rf_trees,
                    max_depth=args.rf_max_depth,
                    min_samples_leaf=args.rf_min_leaf,
                    max_features=args.rf_max_features,
                    n_jobs=args.n_jobs,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    rf.fit(
        nframe(
            train,
            features,
        ),
        nframe(
            train,
            targets,
        ).to_numpy(
            np.float32
        ),
    )

    joblib.dump(
        {
            "mean_z": mean_z,
            "targets": targets,
        },
        out
        / "train_mean_baseline.joblib",
        compress=3,
    )

    joblib.dump(
        {
            "model": ridge,
            "features": features,
            "targets": targets,
        },
        out
        / "ridge_model.joblib",
        compress=3,
    )

    joblib.dump(
        {
            "models": gam,
            "targets": targets,
            "selection": gam_selection,
        },
        out
        / "spline_gam_models.joblib",
        compress=3,
    )

    joblib.dump(
        {
            "model": rf,
            "features": features,
            "targets": targets,
            "primary_model": True,
        },
        out
        / "random_forest_model.joblib",
        compress=3,
    )

    (
        out
        / "spline_gam_feature_selection.json"
    ).write_text(
        json.dumps(
            gam_selection,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_manifest = {
        "version": "final-curve-regressors-v1",
        "year": int(
            args.year
        ),
        "source_dataset": str(
            dataset
        ),
        "model_features": features,
        "latent_columns": targets,
        "primary_model": "random_forest",
        "audit_baselines": [
            "train_mean",
            "ridge",
            "spline_gam",
        ],
        "models": [
            "train_mean",
            "ridge",
            "spline_gam",
            "random_forest",
        ],
        "parts": {},
    }

    metric_tables = []

    for split in [
        "val",
        "test",
    ]:
        written, metrics = predict_split(
            dataset,
            manifest,
            split,
            out,
            features,
            targets,
            mean_z,
            ridge,
            gam,
            rf,
        )

        output_manifest[
            "parts"
        ][
            split
        ] = written

        metrics.to_csv(
            out
            / f"{split}_latent_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        metric_tables.append(
            metrics
        )

    (
        out
        / "manifest.json"
    ).write_text(
        json.dumps(
            output_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    all_metrics = pd.concat(
        metric_tables,
        ignore_index=True,
    )

    summary = "\n".join(
        [
            (
                f"10c Final curve regressor - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                f"Model features = "
                f"{len(features)}"
            ),
            (
                f"Absolute latent targets = "
                f"{len(targets)}"
            ),
            (
                f"Shared TRAIN sample = "
                f"{len(train):,}"
            ),
            (
                f"Spline-GAM TRAIN sample = "
                f"{len(gam_train):,}"
            ),
            "Historical-bid leakage check = PASS",
            "",
            "PRIMARY MODEL = random_forest",
            (
                "Ridge / Spline-GAM / train_mean are audit baselines; "
                "10d does not re-select the final model."
            ),
            "",
            "Latent diagnostics:",
            all_metrics.to_string(
                index=False
            ),
        ]
    )

    (
        out
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(
        summary
    )
    print()
    print(
        f"Outputs: {out}"
    )


if __name__ == "__main__":
    main()
