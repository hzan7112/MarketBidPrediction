#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09b_train_prediction_family_classifier.py

Feature-only Prediction-Family classifier for the K=5 absolute families
diagnosed in 09a.

Inputs:
- 83 feature-only predictors from 08b
- TRUE absolute latent target only for creating TRAIN/VAL/TEST family labels
- K=5 TRAIN-fitted clustering model from 09a

Classifier inputs NEVER contain the current/previous bid curve, latent target,
template ID, or theta.

Models:
- Logistic Regression
- Decision Tree
- Random Forest

Selection:
- validation Macro-F1
- tie-break: Balanced Accuracy, then Accuracy

Outputs:
data/processed/bidprediction/<year>/prediction_family_classifier/
"""

from __future__ import annotations

import argparse, gc, json, math, shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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

META = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
]


def parts(m, split):
    return [
        x["file"] if isinstance(x, dict) else x
        for x in m["parts"][split]
    ]


def nframe(d, cols):
    return d[cols].apply(
        pd.to_numeric,
        errors="coerce",
    )


def leakage_check(features):
    bad = [
        f
        for f in features
        if any(
            token in f.lower()
            for token in FORBIDDEN
        )
    ]

    if bad:
        raise RuntimeError(
            "Forbidden bid-history/target information in classifier features: "
            + ", ".join(
                bad[:30]
            )
        )


def family_ids(k):
    return [
        f"F{i:02d}"
        for i in range(k)
    ]


def labels_from_latent(
    d,
    targets,
    cluster_model,
):
    z = nframe(
        d,
        targets,
    ).to_numpy(
        np.float64
    )

    z = np.asarray(
        z,
        dtype=cluster_model.cluster_centers_.dtype,
    )

    y = cluster_model.predict(
        z
    )

    return y.astype(
        np.int16
    )


def load_train_sample(
    dataset,
    manifest,
    features,
    targets,
    cluster_model,
    max_rows,
    seed,
):
    fs = parts(
        manifest,
        "train",
    )

    per_part = max(
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
        path = dataset / rel

        print(
            f"[TRAIN sample {i}/{len(fs)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(
            path
        )

        if d.empty:
            continue

        y = labels_from_latent(
            d,
            targets,
            cluster_model,
        )

        keep = d[
            features
        ].copy()

        keep["_family_int"] = y

        if len(keep) > per_part:
            keep = keep.sample(
                n=per_part,
                random_state=(
                    seed
                    + 7919 * i
                ),
            )

        blocks.append(
            keep
        )

        del d, keep, y
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


def fit_models(
    train,
    features,
    args,
):
    X = nframe(
        train,
        features,
    )

    y = train[
        "_family_int"
    ].to_numpy(
        np.int16
    )

    lr = Pipeline(
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
                "classifier",
                LogisticRegression(
                    C=args.lr_c,
                    max_iter=args.lr_max_iter,
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            ),
        ]
    )

    dt = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=args.dt_max_depth,
                    min_samples_leaf=args.dt_min_leaf,
                    random_state=args.seed,
                ),
            ),
        ]
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
                "classifier",
                RandomForestClassifier(
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

    print("[fit] Logistic Regression", flush=True)
    lr.fit(
        X,
        y,
    )

    print("[fit] Decision Tree", flush=True)
    dt.fit(
        X,
        y,
    )

    print("[fit] Random Forest", flush=True)
    rf.fit(
        X,
        y,
    )

    return {
        "logistic_regression": lr,
        "decision_tree": dt,
        "random_forest": rf,
    }


def metric_row(
    split,
    model_name,
    y_true,
    y_pred,
):
    return {
        "split": split,
        "model": model_name,
        "rows": int(
            len(
                y_true
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def predict_split(
    dataset,
    manifest,
    split,
    out_dir,
    features,
    targets,
    cluster_model,
    models,
    k,
):
    pred_dir = (
        out_dir
        / "prediction_parts"
        / split
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_buffers = {
        name: {
            "true": [],
            "pred": [],
        }
        for name in models
    }

    written = []

    fs = parts(
        manifest,
        split,
    )

    for i, rel in enumerate(
        fs,
        1,
    ):
        path = dataset / rel

        print(
            f"[predict {split} {i}/{len(fs)}] {path.name}",
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

        y_true = labels_from_latent(
            d,
            targets,
            cluster_model,
        )

        keep_cols = [
            c
            for c in META
            if c in d.columns
        ]

        out_d = d[
            keep_cols
        ].copy()

        out_d[
            "true_family_int"
        ] = y_true

        out_d[
            "true_family_id"
        ] = [
            f"F{x:02d}"
            for x in y_true
        ]

        for name, model in models.items():
            pred = model.predict(
                X
            ).astype(
                np.int16
            )

            out_d[
                f"pred_{name}_family_int"
            ] = pred

            out_d[
                f"pred_{name}_family_id"
            ] = [
                f"F{x:02d}"
                for x in pred
            ]

            if hasattr(
                model,
                "predict_proba",
            ):
                proba = model.predict_proba(
                    X
                )

                classes = model.classes_.astype(
                    int
                )

                full = np.zeros(
                    (
                        len(d),
                        k,
                    ),
                    dtype=np.float32,
                )

                for jj, cls in enumerate(
                    classes
                ):
                    full[
                        :,
                        int(cls),
                    ] = proba[
                        :,
                        jj,
                    ]

                for cls in range(
                    k
                ):
                    out_d[
                        f"pred_{name}_prob_F{cls:02d}"
                    ] = full[
                        :,
                        cls,
                    ]

            metric_buffers[
                name
            ][
                "true"
            ].append(
                y_true
            )

            metric_buffers[
                name
            ][
                "pred"
            ].append(
                pred
            )

        out_name = (
            f"{split}_family_predictions_"
            f"{i:04d}.pkl"
        )

        out_path = (
            pred_dir
            / out_name
        )

        out_d.to_pickle(
            out_path
        )

        written.append(
            {
                "file": str(
                    out_path.relative_to(
                        out_dir
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
            y_true,
            out_d,
        )
        gc.collect()

    metrics = []
    confusion_rows = []
    reports = {}

    labels = list(
        range(
            k
        )
    )

    for name, buf in metric_buffers.items():
        yt = np.concatenate(
            buf[
                "true"
            ]
        )

        yp = np.concatenate(
            buf[
                "pred"
            ]
        )

        metrics.append(
            metric_row(
                split,
                name,
                yt,
                yp,
            )
        )

        cm = confusion_matrix(
            yt,
            yp,
            labels=labels,
        )

        for i_true in labels:
            for i_pred in labels:
                confusion_rows.append(
                    {
                        "split": split,
                        "model": name,
                        "true_family": f"F{i_true:02d}",
                        "pred_family": f"F{i_pred:02d}",
                        "rows": int(
                            cm[
                                i_true,
                                i_pred,
                            ]
                        ),
                    }
                )

        reports[
            name
        ] = classification_report(
            yt,
            yp,
            labels=labels,
            target_names=[
                f"F{x:02d}"
                for x in labels
            ],
            output_dict=True,
            zero_division=0,
        )

    return (
        written,
        pd.DataFrame(
            metrics
        ),
        pd.DataFrame(
            confusion_rows
        ),
        reports,
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
        default="macro_b_feature_only_dataset",
    )

    ap.add_argument(
        "--family-dir",
        default="absolute_prediction_family_diagnostics",
    )

    ap.add_argument(
        "--k",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=300_000,
    )

    ap.add_argument(
        "--lr-c",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--lr-max-iter",
        type=int,
        default=800,
    )

    ap.add_argument(
        "--dt-max-depth",
        type=int,
        default=14,
    )

    ap.add_argument(
        "--dt-min-leaf",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--rf-trees",
        type=int,
        default=192,
    )

    ap.add_argument(
        "--rf-max-depth",
        type=int,
        default=18,
    )

    ap.add_argument(
        "--rf-min-leaf",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--rf-max-features",
        default="sqrt",
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

    family_dir = (
        base
        / args.family_dir
    )

    ds_manifest = json.loads(
        (
            dataset
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    features = list(
        ds_manifest[
            "model_features"
        ]
    )

    targets = list(
        ds_manifest[
            "latent_columns"
        ]
    )

    leakage_check(
        features
    )

    family_bundle = joblib.load(
        family_dir
        / "cluster_models.joblib"
    )

    cluster_models = family_bundle[
        "models"
    ]

    if args.k not in cluster_models:
        raise KeyError(
            f"K={args.k} not found in 09a cluster models. "
            f"Available: {sorted(cluster_models)}"
        )

    cluster_model = cluster_models[
        args.k
    ]

    if len(
        targets
    ) != cluster_model.cluster_centers_.shape[
        1
    ]:
        raise ValueError(
            "08b latent dimension and 09a family latent dimension differ."
        )

    out_dir = (
        base
        / "prediction_family_classifier"
    )

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                out_dir
            )

        shutil.rmtree(
            out_dir
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"09b Prediction-Family classifier - {args.year}"
    )
    print("=" * 80)
    print(
        f"K = {args.k}"
    )
    print(
        f"Feature-only inputs = {len(features)}"
    )
    print(
        "Historical-bid leakage check = PASS"
    )
    print()

    train = load_train_sample(
        dataset,
        ds_manifest,
        features,
        targets,
        cluster_model,
        args.max_train_rows,
        args.seed,
    )

    print(
        f"Shared TRAIN sample = {len(train):,}"
    )

    train_dist = (
        train[
            "_family_int"
        ]
        .value_counts(
            normalize=False
        )
        .sort_index()
    )

    models = fit_models(
        train,
        features,
        args,
    )

    model_files = {}

    for name, model in models.items():
        path = (
            out_dir
            / f"{name}_classifier.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "features": features,
                "k": int(
                    args.k
                ),
            },
            path,
            compress=3,
        )

        model_files[
            name
        ] = path.name

    pred_manifest = {
        "year": int(
            args.year
        ),
        "source_dataset": str(
            dataset
        ),
        "family_source": str(
            family_dir
        ),
        "k": int(
            args.k
        ),
        "model_features": features,
        "latent_columns": targets,
        "models": list(
            models
        ),
        "parts": {},
    }

    metric_tables = []
    confusion_tables = []
    report_json = {}

    for split in [
        "val",
        "test",
    ]:
        (
            written,
            metrics,
            confusion,
            reports,
        ) = predict_split(
            dataset,
            ds_manifest,
            split,
            out_dir,
            features,
            targets,
            cluster_model,
            models,
            args.k,
        )

        pred_manifest[
            "parts"
        ][
            split
        ] = written

        metric_tables.append(
            metrics
        )

        confusion_tables.append(
            confusion
        )

        report_json[
            split
        ] = reports

    metrics = pd.concat(
        metric_tables,
        ignore_index=True,
    )

    confusion = pd.concat(
        confusion_tables,
        ignore_index=True,
    )

    metrics.to_csv(
        out_dir
        / "classifier_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion.to_csv(
        out_dir
        / "classifier_confusion_matrix_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        out_dir
        / "classification_reports.json"
    ).write_text(
        json.dumps(
            report_json,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    val = (
        metrics.loc[
            metrics[
                "split"
            ].eq(
                "val"
            )
        ]
        .sort_values(
            [
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    selected = str(
        val.iloc[
            0
        ][
            "model"
        ]
    )

    selection = {
        "selected_model": selected,
        "selection_split": "validation",
        "selection_metric": "Macro-F1",
        "tie_break": [
            "Balanced Accuracy",
            "Accuracy",
        ],
        "model_file": model_files[
            selected
        ],
        "test_used_for_selection": False,
    }

    (
        out_dir
        / "selected_classifier.json"
    ).write_text(
        json.dumps(
            selection,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pred_manifest[
        "selected_classifier"
    ] = selected

    (
        out_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            pred_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dist_rows = []

    for fam in range(
        args.k
    ):
        count = int(
            train_dist.get(
                fam,
                0,
            )
        )

        dist_rows.append(
            {
                "family_id": f"F{fam:02d}",
                "train_sample_rows": count,
                "train_sample_share": float(
                    count
                    / max(
                        len(train),
                        1,
                    )
                ),
            }
        )

    pd.DataFrame(
        dist_rows
    ).to_csv(
        out_dir
        / "train_family_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = "\n".join(
        [
            (
                f"09b Prediction-Family classifier - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            f"K = {args.k}",
            (
                f"Feature-only inputs = "
                f"{len(features)}"
            ),
            (
                f"Shared TRAIN sample = "
                f"{len(train):,}"
            ),
            "Historical-bid leakage check = PASS",
            "",
            "Classifier metrics:",
            metrics.to_string(
                index=False
            ),
            "",
            (
                "Selected classifier from VALIDATION = "
                f"{selected}"
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
    print(
        summary
    )
    print()
    print(
        f"Outputs: {out_dir}"
    )


if __name__ == "__main__":
    main()
