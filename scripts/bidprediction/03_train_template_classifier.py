#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_train_template_classifier.py

Stage 3 / bidprediction
Unified comparison of four lightweight/interpretable template classifiers.

Target
------
y_template_id in:
    T00 ... T11, FLAT

Models
------
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. LightGBM

Baselines
---------
1. Majority class
2. Previous-same-slot template: hist_lag1_template_id

Theoretical input structure
---------------------------
X_{i,t} = [Z_{i,t}, M_t, U_{i,t}]

Z : strategy profile = 9 LT + 9 ST + 8 Break
M : market environment = feature_group == market_environment
U : unit physical/operational-state proxy = feature_group == unit_state_proxy
H : historical bid state = feature_group == participant_history

H is deliberately separated from U and is used only as an enhancement term.

Feature ablation
----------------
Z
Z_M
Z_U
Z_M_U
Z_H
Z_M_U_H

The core theoretical model is Z_M_U. Calendar features are excluded from this
core experiment so M means actual market environment rather than generic time.

Data split
----------
Uses the strict chronological split created by:
    02_validate_prediction_dataset.py

No random train/validation/test split is used.

Fair-comparison rules
---------------------
- Within each feature set, all four algorithms use the SAME sampled training rows.
- Median imputation is fitted on TRAIN only.
- Logistic Regression additionally uses StandardScaler fitted on TRAIN only.
- participant_id is never used as a feature.
- Validation/test sets are evaluated in full by streaming through monthly CSVs.
- The same validation/test rows are used for all algorithms.

Outputs
-------
data/processed/bidprediction/<year>/template_classifier/
    feature_sets.csv
    template_label_mapping.csv
    training_sample_summary.csv
    template_classifier_metrics.csv
    template_classifier_per_class_metrics.csv
    feature_importance_*.csv
    confusion_*.csv
    model_*.joblib / model_*.txt
    preprocess_*.joblib
    summary.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "This script compares four models including LightGBM.\n"
        "Install it first with:\n"
        "    pip install lightgbm"
    ) from e


# =============================================================================
# Fixed Stage-1 profile features
# =============================================================================

LT_FEATURES = [
    "lt_bid_level",
    "lt_adjustment_magnitude",
    "lt_strategy_persistence",
    "lt_quantity_hhi",
    "lt_effective_segment_count",
    "lt_flat_curve_rate",
    "lt_tail_uplift_ratio",
    "lt_curve_bend_ratio",
    "lt_shape_variability",
]

ST_FEATURES = [
    "st_bid_level_z",
    "st_adjustment_bias_z",
    "st_adjustment_magnitude_z",
    "st_quantity_hhi_z",
    "st_effective_segment_count_z",
    "st_flat_curve_rate_z",
    "st_tail_uplift_ratio_z",
    "st_curve_bend_ratio_z",
    "st_shape_shift",
]

BREAK_FEATURES = [
    "break_bid_level",
    "break_adjustment_bias",
    "break_adjustment_magnitude",
    "break_quantity_hhi",
    "break_effective_segment_count",
    "break_flat_curve_rate",
    "break_tail_uplift_ratio",
    "break_curve_bend_ratio",
]

TARGET = "y_template_id"

TEMPLATE_ORDER = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
TEMPLATE_TO_INT = {name: i for i, name in enumerate(TEMPLATE_ORDER)}
INT_TO_TEMPLATE = {i: name for name, i in TEMPLATE_TO_INT.items()}

MODE_TO_INT = {
    "flat": 0,
    "block": 1,
    "sloped": 2,
}

CATEGORICAL_FEATURES = {
    "hist_lag1_template_id",
    "hist30_dominant_template_id",
    "hist_lag1_curve_mode",
}

# These fields are useful for QC, but should not become predictors.
EXCLUDE_FEATURES = {
    "rolling_lt_ready_flag",
    "st_ready_flag",
    "profile_ready_flag",
    "market_ready_flag",
    "unit_state_ready_flag",
    "market_nonmissing_count",
    "rolling_lt_nonmissing_count",
    "hist_prev_available_flag",
}


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_schema(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"column", "role", "feature_group"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Feature schema missing required columns: {sorted(missing)}"
        )

    return df


def discover_parts(path: Path) -> list[Path]:
    files = sorted(path.glob("prediction_dataset_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No prediction dataset parts found under:\n{path}"
        )
    return files


def load_split_dates(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\n"
            "Run scripts/bidprediction/02_validate_prediction_dataset.py first."
        )

    split = pd.read_csv(path).set_index("split")

    for name in ["train", "val", "test"]:
        if name not in split.index:
            raise ValueError(
                f"Split '{name}' is missing in {path}"
            )

    train_end = pd.Timestamp(split.loc["train", "last_date"])
    val_start = pd.Timestamp(split.loc["val", "first_date"])
    val_end = pd.Timestamp(split.loc["val", "last_date"])
    test_start = pd.Timestamp(split.loc["test", "first_date"])

    rows = {
        name: int(split.loc[name, "rows"])
        for name in ["train", "val", "test"]
    }

    return train_end, val_start, val_end, test_start, rows


def split_mask(
    local_date: pd.Series,
    split_name: str,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.Series:
    d = pd.to_datetime(local_date, errors="coerce").dt.normalize()

    if split_name == "train":
        return d <= train_end

    if split_name == "val":
        return (d >= val_start) & (d <= val_end)

    if split_name == "test":
        return d >= test_start

    raise ValueError(split_name)


def build_feature_sets(schema: pd.DataFrame):
    feature_schema = schema[
        schema["role"].astype(str).eq("feature")
    ].copy()

    feature_to_group = dict(
        zip(
            feature_schema["column"].astype(str),
            feature_schema["feature_group"].astype(str),
        )
    )

    available = set(feature_to_group) - EXCLUDE_FEATURES

    # Z: exact finalized 26-dimensional strategy profile.
    z_expected = LT_FEATURES + ST_FEATURES + BREAK_FEATURES
    missing_z = [c for c in z_expected if c not in available]
    if missing_z:
        raise KeyError(
            "Finalized Z profile is incomplete in prediction_feature_schema: "
            f"{missing_z}"
        )
    z = list(z_expected)

    # M: market environment only.
    m = [
        c for c, g in feature_to_group.items()
        if g == "market_environment" and c in available
    ]

    # U: unit physical / operational-state proxy only.
    u = [
        c for c, g in feature_to_group.items()
        if g == "unit_state_proxy" and c in available
    ]

    # H: historical bid behavior. It is NOT part of U.
    h = [
        c for c, g in feature_to_group.items()
        if g == "participant_history" and c in available
    ]

    def unique(cols):
        out = []
        seen = set()
        for c in cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    z, m, u, h = map(unique, [z, m, u, h])

    feature_sets = {
        "Z": z,
        "Z_M": unique(z + m),
        "Z_U": unique(z + u),
        "Z_M_U": unique(z + m + u),
        "Z_H": unique(z + h),
        "Z_M_U_H": unique(z + m + u + h),
    }

    component_map = {
        "Z": z,
        "M": m,
        "U": u,
        "H": h,
    }

    return feature_sets, component_map, feature_to_group


# =============================================================================
# Encoding
# =============================================================================

def encode_categorical_column(s: pd.Series, col: str) -> pd.Series:
    x = s.astype("string").str.strip()

    if col in {
        "hist_lag1_template_id",
        "hist30_dominant_template_id",
    }:
        return x.map(TEMPLATE_TO_INT).astype("float32")

    if col == "hist_lag1_curve_mode":
        return (
            x.str.lower()
            .map(MODE_TO_INT)
            .astype("float32")
        )

    return safe_numeric(s).astype("float32")


def prepare_X(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    out = {}

    for c in features:
        if c in CATEGORICAL_FEATURES:
            out[c] = encode_categorical_column(df[c], c)
        else:
            out[c] = safe_numeric(df[c]).astype("float32")

    return pd.DataFrame(out, index=df.index)


def encode_target(s: pd.Series) -> np.ndarray:
    x = s.astype("string").str.strip()

    unknown = sorted(
        set(x.dropna().astype(str))
        - set(TEMPLATE_TO_INT)
    )
    if unknown:
        raise ValueError(
            f"Unknown template labels found: {unknown}"
        )

    y = x.map(TEMPLATE_TO_INT)

    if y.isna().any():
        raise ValueError(
            "Target contains missing template labels."
        )

    return y.to_numpy(np.int16)


# =============================================================================
# Fair shared training sample
# =============================================================================

def collect_training_sample(
    files: list[Path],
    all_features: list[str],
    train_end: pd.Timestamp,
    max_train_rows: int | None,
    train_rows_hint: int,
    chunksize: int,
    seed: int,
) -> pd.DataFrame:
    """
    Collect one deterministic training sample containing the UNION of all
    selected feature-set columns.

    Every model and every feature ablation is derived from exactly these rows.
    """

    rng = np.random.default_rng(seed)

    if max_train_rows is None or max_train_rows <= 0:
        keep_prob = 1.0
    else:
        # Small margin, final exact trim below.
        keep_prob = min(
            1.0,
            1.08 * max_train_rows / max(train_rows_hint, 1),
        )

    usecols = list(
        dict.fromkeys(
            [
                "local_date",
                "prediction_ready_flag",
                TARGET,
            ]
            + all_features
        )
    )

    blocks = []
    seen_train_ready = 0

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(
            file,
            nrows=0,
        ).columns.tolist()

        missing = [
            c for c in all_features + [TARGET]
            if c not in header
        ]
        if missing:
            raise KeyError(
                f"{file.name} missing required columns:\n{missing}"
            )

        cols = [c for c in usecols if c in header]

        print(
            f"[train-load {file_no}/{len(files)}] {file.name}",
            flush=True,
        )

        for chunk in pd.read_csv(
            file,
            usecols=cols,
            chunksize=chunksize,
            low_memory=False,
        ):
            ready = safe_numeric(
                chunk["prediction_ready_flag"]
            ).fillna(0).eq(1)

            d = pd.to_datetime(
                chunk["local_date"],
                errors="coerce",
            ).dt.normalize()

            mask = ready & (d <= train_end)

            sub = chunk.loc[
                mask,
                [TARGET] + all_features,
            ]

            seen_train_ready += len(sub)

            if sub.empty:
                continue

            if keep_prob < 1.0:
                take = (
                    rng.random(len(sub))
                    < keep_prob
                )
                sub = sub.loc[take]

            if len(sub):
                blocks.append(sub)

    if not blocks:
        raise ValueError(
            "No prediction-ready training rows were collected."
        )

    train = pd.concat(
        blocks,
        ignore_index=True,
    )

    if (
        max_train_rows is not None
        and max_train_rows > 0
        and len(train) > max_train_rows
    ):
        train = (
            train.sample(
                n=max_train_rows,
                random_state=seed,
            )
            .reset_index(drop=True)
        )

    print(
        f"Shared train rows: available={seen_train_ready:,}, "
        f"sampled={len(train):,}",
        flush=True,
    )

    return train


# =============================================================================
# Preprocessing
# =============================================================================

def fit_imputer(
    X: pd.DataFrame,
) -> SimpleImputer:
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    imputer.fit(X)
    return imputer


def transform_numeric_matrix(
    X: pd.DataFrame,
    imputer: SimpleImputer,
) -> np.ndarray:
    return imputer.transform(X).astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# Model fitting
# =============================================================================

def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_iter: int,
    C: float,
):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        n_jobs=None,
        random_state=seed,
    )
    model.fit(Xs, y)

    return {
        "model": model,
        "scaler": scaler,
    }


def fit_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_depth: int,
    min_samples_leaf: int,
):
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)

    return {
        "model": model,
        "scaler": None,
    }


def fit_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
        verbose=0,
    )
    model.fit(X, y)

    return {
        "model": model,
        "scaler": None,
    }


def sqrt_balanced_sample_weights(
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(
        y,
        minlength=len(TEMPLATE_ORDER),
    ).astype(float)

    w = np.ones_like(counts)
    valid = counts > 0

    raw = (
        counts[valid].sum()
        / counts[valid]
    ) ** 0.5

    sample_mean = (
        np.sum(counts[valid] * raw)
        / counts[valid].sum()
    )

    w[valid] = raw / sample_mean

    return w, w[y]


def fit_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int,
    num_boost_round: int,
    num_leaves: int,
    learning_rate: float,
    min_data_in_leaf: int,
):
    class_weights, sample_weights = (
        sqrt_balanced_sample_weights(y)
    )

    train_set = lgb.Dataset(
        X,
        label=y,
        weight=sample_weights,
        feature_name=feature_names,
        free_raw_data=True,
    )

    params = {
        "objective": "multiclass",
        "num_class": len(TEMPLATE_ORDER),
        "metric": "multi_logloss",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": -1,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": 0,
    }

    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
    )

    return {
        "model": model,
        "scaler": None,
        "class_weights": class_weights,
    }


# =============================================================================
# Prediction / metrics
# =============================================================================

def predict_proba_payload(
    algorithm: str,
    payload: dict,
    X: np.ndarray,
) -> np.ndarray:
    model = payload["model"]

    if algorithm == "LogisticRegression":
        X_use = payload["scaler"].transform(X)
        proba = model.predict_proba(X_use)

    elif algorithm in {
        "DecisionTree",
        "RandomForest",
    }:
        proba = model.predict_proba(X)

    elif algorithm == "LightGBM":
        proba = model.predict(X)

    else:
        raise ValueError(algorithm)

    proba = np.asarray(
        proba,
        dtype=np.float64,
    )

    # sklearn models can theoretically omit a class if absent in the
    # sampled train rows. Expand to fixed 13-class order.
    if algorithm != "LightGBM":
        classes = np.asarray(
            model.classes_,
            dtype=int,
        )
        if (
            proba.shape[1] != len(TEMPLATE_ORDER)
            or not np.array_equal(
                classes,
                np.arange(len(TEMPLATE_ORDER)),
            )
        ):
            full = np.zeros(
                (len(X), len(TEMPLATE_ORDER)),
                dtype=np.float64,
            )
            full[:, classes] = proba
            proba = full

    return proba


def update_confusion(
    cm: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    cm += confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(TEMPLATE_ORDER)),
    )
    return cm


def metrics_from_confusion(
    cm: np.ndarray,
) -> dict:
    total = cm.sum()
    diag = np.diag(cm)

    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)

    accuracy = (
        diag.sum() / total
        if total else np.nan
    )

    recall = np.divide(
        diag,
        support,
        out=np.zeros_like(
            diag,
            dtype=float,
        ),
        where=support > 0,
    )

    precision = np.divide(
        diag,
        pred_count,
        out=np.zeros_like(
            diag,
            dtype=float,
        ),
        where=pred_count > 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(
            recall,
            dtype=float,
        ),
        where=(precision + recall) > 0,
    )

    valid = support > 0

    balanced_accuracy = (
        float(recall[valid].mean())
        if valid.any() else np.nan
    )

    macro_f1 = (
        float(f1[valid].mean())
        if valid.any() else np.nan
    )

    weighted_f1 = (
        float(
            np.sum(f1 * support)
            / total
        )
        if total else np.nan
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def per_class_frame(
    cm: np.ndarray,
    feature_set: str,
    algorithm: str,
    split_name: str,
) -> pd.DataFrame:
    m = metrics_from_confusion(cm)

    rows = []

    for i, tid in enumerate(TEMPLATE_ORDER):
        rows.append({
            "feature_set": feature_set,
            "algorithm": algorithm,
            "split": split_name,
            "template_id": tid,
            "support": int(
                m["support"][i]
            ),
            "precision": float(
                m["precision"][i]
            ),
            "recall": float(
                m["recall"][i]
            ),
            "f1": float(
                m["f1"][i]
            ),
        })

    return pd.DataFrame(rows)


def majority_confusion(
    train_counts: np.ndarray,
    reference_cm: np.ndarray,
):
    majority = int(
        np.argmax(train_counts)
    )

    support = reference_cm.sum(axis=1)

    cm = np.zeros_like(reference_cm)

    for true_cls, n in enumerate(support):
        cm[
            true_cls,
            majority,
        ] = n

    return majority, cm


# =============================================================================
# Streaming multi-model evaluation
# =============================================================================

def evaluate_feature_set_streaming(
    files: list[Path],
    feature_set: str,
    features: list[str],
    imputer: SimpleImputer,
    models: dict,
    split_name: str,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    chunksize: int,
):
    """
    One streaming pass evaluates ALL four algorithms for the feature set.
    """

    states = {}

    for algorithm in models:
        states[algorithm] = {
            "cm": np.zeros(
                (
                    len(TEMPLATE_ORDER),
                    len(TEMPLATE_ORDER),
                ),
                dtype=np.int64,
            ),
            "rows": 0,
            "top2_correct": 0,
            "logloss_sum": 0.0,
        }

    lag_cm = np.zeros(
        (
            len(TEMPLATE_ORDER),
            len(TEMPLATE_ORDER),
        ),
        dtype=np.int64,
    )
    lag_rows = 0

    usecols = list(
        dict.fromkeys(
            [
                "local_date",
                "prediction_ready_flag",
                TARGET,
                "hist_lag1_template_id",
            ]
            + features
        )
    )

    eps = 1e-15

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(
            file,
            nrows=0,
        ).columns.tolist()

        missing = [
            c for c in features + [TARGET]
            if c not in header
        ]
        if missing:
            raise KeyError(
                f"{file.name} missing columns:\n{missing}"
            )

        cols = [
            c for c in usecols
            if c in header
        ]

        print(
            f"[eval {feature_set}/{split_name} "
            f"{file_no}/{len(files)}] {file.name}",
            flush=True,
        )

        for chunk in pd.read_csv(
            file,
            usecols=cols,
            chunksize=chunksize,
            low_memory=False,
        ):
            ready = safe_numeric(
                chunk["prediction_ready_flag"]
            ).fillna(0).eq(1)

            sm = split_mask(
                chunk["local_date"],
                split_name,
                train_end,
                val_start,
                val_end,
                test_start,
            )

            d = chunk.loc[
                ready & sm
            ]

            if d.empty:
                continue

            y = encode_target(
                d[TARGET]
            )

            X_df = prepare_X(
                d,
                features,
            )

            X = transform_numeric_matrix(
                X_df,
                imputer,
            )

            for algorithm, payload in models.items():
                proba = predict_proba_payload(
                    algorithm,
                    payload,
                    X,
                )

                pred = np.argmax(
                    proba,
                    axis=1,
                ).astype(np.int16)

                state = states[algorithm]

                state["cm"] = update_confusion(
                    state["cm"],
                    y,
                    pred,
                )

                state["rows"] += len(y)

                ptrue = proba[
                    np.arange(len(y)),
                    y,
                ]

                state["logloss_sum"] += float(
                    -np.log(
                        np.clip(
                            ptrue,
                            eps,
                            1.0,
                        )
                    ).sum()
                )

                top2 = np.argpartition(
                    proba,
                    kth=-2,
                    axis=1,
                )[:, -2:]

                state["top2_correct"] += int(
                    np.any(
                        top2 == y[:, None],
                        axis=1,
                    ).sum()
                )

            if "hist_lag1_template_id" in d.columns:
                lag = (
                    d["hist_lag1_template_id"]
                    .astype("string")
                    .str.strip()
                    .map(TEMPLATE_TO_INT)
                )

                ok = lag.notna().to_numpy()

                if ok.any():
                    lag_true = y[ok]
                    lag_pred = (
                        lag.loc[ok]
                        .to_numpy(np.int16)
                    )

                    lag_cm = update_confusion(
                        lag_cm,
                        lag_true,
                        lag_pred,
                    )

                    lag_rows += int(
                        ok.sum()
                    )

    result = {}

    for algorithm, state in states.items():
        m = metrics_from_confusion(
            state["cm"]
        )

        n = state["rows"]

        m["rows"] = n
        m["log_loss"] = (
            state["logloss_sum"] / n
            if n else np.nan
        )
        m["top2_accuracy"] = (
            state["top2_correct"] / n
            if n else np.nan
        )

        result[algorithm] = {
            "metrics": m,
            "cm": state["cm"],
        }

    lag_m = metrics_from_confusion(
        lag_cm
    )
    lag_m["rows"] = lag_rows

    return result, lag_m, lag_cm


# =============================================================================
# Feature importance
# =============================================================================

def save_feature_importance(
    out_dir: Path,
    feature_set: str,
    algorithm: str,
    features: list[str],
    payload: dict,
):
    model = payload["model"]

    if algorithm == "LogisticRegression":
        # Mean absolute standardized coefficient over 13 classes.
        imp = np.mean(
            np.abs(model.coef_),
            axis=0,
        )

        df = pd.DataFrame({
            "feature": features,
            "importance": imp,
            "importance_type": "mean_abs_standardized_coefficient",
        })

    elif algorithm in {
        "DecisionTree",
        "RandomForest",
    }:
        df = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
            "importance_type": "impurity_importance",
        })

    elif algorithm == "LightGBM":
        gain = model.feature_importance(
            importance_type="gain"
        )
        split = model.feature_importance(
            importance_type="split"
        )

        df = pd.DataFrame({
            "feature": features,
            "importance": gain,
            "split_count": split,
            "importance_type": "gain",
        })

    else:
        return

    total = df["importance"].sum()
    df["importance_share"] = (
        df["importance"] / total
        if total > 0 else 0.0
    )

    df = df.sort_values(
        "importance",
        ascending=False,
    )

    df.to_csv(
        out_dir
        / f"feature_importance_{feature_set}_{algorithm}.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    p.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )

    p.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
    )

    p.add_argument(
        "--max-train-rows",
        type=int,
        default=600_000,
        help=(
            "Shared training sample size used by ALL four algorithms. "
            "Default 600k keeps Logistic/RF engineering cost reasonable. "
            "Use 0 to use all prediction-ready training rows."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--feature-set",
        choices=[
            "Z",
            "Z_M",
            "Z_U",
            "Z_M_U",
            "Z_H",
            "Z_M_U_H",
            "all",
        ],
        default="all",
    )

    # Logistic Regression
    p.add_argument(
        "--logit-max-iter",
        type=int,
        default=300,
    )

    p.add_argument(
        "--logit-c",
        type=float,
        default=1.0,
    )

    # Decision Tree
    p.add_argument(
        "--dt-max-depth",
        type=int,
        default=18,
    )

    p.add_argument(
        "--dt-min-samples-leaf",
        type=int,
        default=100,
    )

    # Random Forest
    p.add_argument(
        "--rf-trees",
        type=int,
        default=120,
    )

    p.add_argument(
        "--rf-max-depth",
        type=int,
        default=20,
    )

    p.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=50,
    )

    # LightGBM
    p.add_argument(
        "--lgb-rounds",
        type=int,
        default=300,
    )

    p.add_argument(
        "--lgb-num-leaves",
        type=int,
        default=63,
    )

    p.add_argument(
        "--lgb-learning-rate",
        type=float,
        default=0.06,
    )

    p.add_argument(
        "--lgb-min-data-in-leaf",
        type=int,
        default=200,
    )

    args = p.parse_args()

    base = (
        Path(args.root)
        / str(args.year)
    )

    files = discover_parts(
        base / "dataset_parts"
    )

    schema = load_schema(
        base
        / f"prediction_feature_schema_{args.year}.csv"
    )

    feature_sets, component_map, feature_to_group = (
        build_feature_sets(schema)
    )

    selected_sets = (
        list(feature_sets)
        if args.feature_set == "all"
        else [args.feature_set]
    )

    split_file = (
        base
        / "validation"
        / "temporal_split_summary.csv"
    )

    (
        train_end,
        val_start,
        val_end,
        test_start,
        split_rows_hint,
    ) = load_split_dates(split_file)

    out_dir = ensure_dir(
        base / "template_classifier"
    )

    # -------------------------------------------------------------------------
    # Save explicit Z / M / U / H component definitions.
    # -------------------------------------------------------------------------

    component_rows = []
    for component, cols in component_map.items():
        for order, c in enumerate(cols):
            component_rows.append({
                "component": component,
                "order": order,
                "feature": c,
                "feature_group": feature_to_group.get(c, ""),
            })

    pd.DataFrame(component_rows).to_csv(
        out_dir / "input_components_Z_M_U_H.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_rows = []

    for set_name, cols in feature_sets.items():
        for order, c in enumerate(cols):
            feature_rows.append({
                "feature_set": set_name,
                "order": order,
                "feature": c,
                "feature_group": feature_to_group.get(c, ""),
            })

    pd.DataFrame(
        feature_rows
    ).to_csv(
        out_dir / "feature_sets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        {
            "template_id": tid,
            "class_index": idx,
        }
        for tid, idx
        in TEMPLATE_TO_INT.items()
    ]).to_csv(
        out_dir / "template_label_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Template classification with explicit Z / M / U / H separation")
    print("=" * 80)
    print(f"Year:              {args.year}")
    print(f"Train end:         {train_end.date()}")
    print(
        f"Validation:        "
        f"{val_start.date()} .. {val_end.date()}"
    )
    print(f"Test start:        {test_start.date()}")
    print(f"Z strategy profile: {len(component_map['Z'])} features")
    print(f"M market state:     {len(component_map['M'])} features")
    print(f"U unit state:       {len(component_map['U'])} features")
    print(f"H bid history:      {len(component_map['H'])} features")
    print(
        f"Feature sets:      "
        f"{', '.join(selected_sets)}"
    )
    print(
        f"Algorithms:        "
        f"LogisticRegression, DecisionTree, "
        f"RandomForest, LightGBM"
    )
    print(
        f"Shared train rows: "
        f"{'ALL' if args.max_train_rows == 0 else f'{args.max_train_rows:,}'}"
    )
    print()

    # -------------------------------------------------------------------------
    # One shared training-row sample across every algorithm and feature set.
    # We load the UNION of selected feature columns once.
    # -------------------------------------------------------------------------

    all_features = []

    seen = set()

    for set_name in selected_sets:
        for c in feature_sets[set_name]:
            if c not in seen:
                all_features.append(c)
                seen.add(c)

    train_df = collect_training_sample(
        files=files,
        all_features=all_features,
        train_end=train_end,
        max_train_rows=(
            None
            if args.max_train_rows == 0
            else args.max_train_rows
        ),
        train_rows_hint=split_rows_hint["train"],
        chunksize=args.chunksize,
        seed=args.seed,
    )

    y_train_shared = encode_target(
        train_df[TARGET]
    )

    train_counts = np.bincount(
        y_train_shared,
        minlength=len(TEMPLATE_ORDER),
    )

    pd.DataFrame([
        {
            "template_id": TEMPLATE_ORDER[i],
            "train_sample_count": int(train_counts[i]),
            "train_sample_share": (
                train_counts[i] / len(train_df)
                if len(train_df) else np.nan
            ),
        }
        for i in range(len(TEMPLATE_ORDER))
    ]).to_csv(
        out_dir / "training_sample_template_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_rows = []
    per_class_frames = []
    training_rows = []
    baseline_rows = []

    algorithms = [
        "LogisticRegression",
        "DecisionTree",
        "RandomForest",
        "LightGBM",
    ]

    # -------------------------------------------------------------------------
    # Feature-set loop
    # -------------------------------------------------------------------------

    for set_no, set_name in enumerate(
        selected_sets,
        1,
    ):
        features = feature_sets[set_name]

        if not features:
            raise ValueError(
                f"Feature set '{set_name}' has no usable features."
            )

        print()
        print("=" * 80)
        print(
            f"[feature set {set_no}/{len(selected_sets)}] "
            f"{set_name}, features={len(features)}"
        )
        print("=" * 80)

        train_subset = train_df[
            [TARGET] + features
        ].copy()

        X_train_df = prepare_X(
            train_subset,
            features,
        )

        imputer = fit_imputer(
            X_train_df
        )

        X_train = transform_numeric_matrix(
            X_train_df,
            imputer,
        )

        y_train = encode_target(
            train_subset[TARGET]
        )

        # Save preprocessing for this feature set.
        preprocess_payload = {
            "features": features,
            "imputer": imputer,
            "template_to_int": TEMPLATE_TO_INT,
            "mode_to_int": MODE_TO_INT,
        }

        joblib.dump(
            preprocess_payload,
            out_dir
            / f"preprocess_{set_name}.joblib",
        )

        models = {}

        # 1. Logistic Regression
        print("[fit] LogisticRegression", flush=True)

        models["LogisticRegression"] = (
            fit_logistic(
                X=X_train,
                y=y_train,
                seed=args.seed,
                max_iter=args.logit_max_iter,
                C=args.logit_c,
            )
        )

        # 2. Decision Tree
        print("[fit] DecisionTree", flush=True)

        models["DecisionTree"] = (
            fit_decision_tree(
                X=X_train,
                y=y_train,
                seed=args.seed,
                max_depth=args.dt_max_depth,
                min_samples_leaf=args.dt_min_samples_leaf,
            )
        )

        # 3. Random Forest
        print("[fit] RandomForest", flush=True)

        models["RandomForest"] = (
            fit_random_forest(
                X=X_train,
                y=y_train,
                seed=args.seed,
                n_estimators=args.rf_trees,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
            )
        )

        # 4. LightGBM
        print("[fit] LightGBM", flush=True)

        models["LightGBM"] = (
            fit_lightgbm(
                X=X_train,
                y=y_train,
                feature_names=features,
                seed=args.seed,
                num_boost_round=args.lgb_rounds,
                num_leaves=args.lgb_num_leaves,
                learning_rate=args.lgb_learning_rate,
                min_data_in_leaf=args.lgb_min_data_in_leaf,
            )
        )

        # Save models + feature importances.
        for algorithm, payload in models.items():
            if algorithm == "LightGBM":
                payload["model"].save_model(
                    str(
                        out_dir
                        / f"model_{set_name}_{algorithm}.txt"
                    )
                )

            else:
                joblib.dump(
                    payload,
                    out_dir
                    / f"model_{set_name}_{algorithm}.joblib",
                )

            save_feature_importance(
                out_dir=out_dir,
                feature_set=set_name,
                algorithm=algorithm,
                features=features,
                payload=payload,
            )

            training_rows.append({
                "feature_set": set_name,
                "algorithm": algorithm,
                "feature_count": len(features),
                "shared_train_sample_rows": len(train_subset),
                "available_train_rows": split_rows_hint["train"],
                "train_sample_share": (
                    len(train_subset)
                    / split_rows_hint["train"]
                    if split_rows_hint["train"] else np.nan
                ),
            })

        # Free raw X matrix before streaming evaluation.
        del X_train_df
        del X_train

        # ---------------------------------------------------------------------
        # Evaluate all four models together in one pass per split.
        # ---------------------------------------------------------------------

        for split_name in [
            "val",
            "test",
        ]:
            print()
            print(
                f"[evaluate] {set_name} / {split_name}",
                flush=True,
            )

            (
                eval_result,
                lag_m,
                lag_cm,
            ) = evaluate_feature_set_streaming(
                files=files,
                feature_set=set_name,
                features=features,
                imputer=imputer,
                models=models,
                split_name=split_name,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                chunksize=args.chunksize,
            )

            # Lag1 is feature-set independent; save once per split.
            if not any(
                r["split"] == split_name
                for r in baseline_rows
            ):
                baseline_rows.append({
                    "baseline": "hist_lag1_template_id",
                    "split": split_name,
                    "rows": lag_m["rows"],
                    "accuracy": lag_m["accuracy"],
                    "balanced_accuracy": lag_m["balanced_accuracy"],
                    "macro_f1": lag_m["macro_f1"],
                    "weighted_f1": lag_m["weighted_f1"],
                })

            # Use first algorithm's confusion only as the support reference
            # for majority baseline, since support is identical.
            reference_cm = (
                eval_result["LogisticRegression"]["cm"]
            )

            majority_class, majority_cm = majority_confusion(
                train_counts,
                reference_cm,
            )

            majority_m = metrics_from_confusion(
                majority_cm
            )

            if not any(
                r["baseline"] == "majority"
                and r["split"] == split_name
                for r in baseline_rows
            ):
                baseline_rows.append({
                    "baseline": "majority",
                    "split": split_name,
                    "rows": int(reference_cm.sum()),
                    "accuracy": majority_m["accuracy"],
                    "balanced_accuracy": majority_m["balanced_accuracy"],
                    "macro_f1": majority_m["macro_f1"],
                    "weighted_f1": majority_m["weighted_f1"],
                    "majority_template": (
                        INT_TO_TEMPLATE[majority_class]
                    ),
                })

            for algorithm in algorithms:
                result = eval_result[algorithm]
                m = result["metrics"]
                cm = result["cm"]

                metric_rows.append({
                    "feature_set": set_name,
                    "algorithm": algorithm,
                    "split": split_name,
                    "feature_count": len(features),
                    "rows": m["rows"],
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"],
                    "weighted_f1": m["weighted_f1"],
                    "top2_accuracy": m["top2_accuracy"],
                    "log_loss": m["log_loss"],
                    "majority_accuracy": majority_m["accuracy"],
                    "lag1_accuracy": lag_m["accuracy"],
                    "lag1_balanced_accuracy": lag_m["balanced_accuracy"],
                    "lag1_macro_f1": lag_m["macro_f1"],
                    "accuracy_gain_over_lag1": (
                        m["accuracy"]
                        - lag_m["accuracy"]
                    ),
                    "balanced_accuracy_gain_over_lag1": (
                        m["balanced_accuracy"]
                        - lag_m["balanced_accuracy"]
                    ),
                    "macro_f1_gain_over_lag1": (
                        m["macro_f1"]
                        - lag_m["macro_f1"]
                    ),
                })

                per_class_frames.append(
                    per_class_frame(
                        cm=cm,
                        feature_set=set_name,
                        algorithm=algorithm,
                        split_name=split_name,
                    )
                )

                pd.DataFrame(
                    cm,
                    index=TEMPLATE_ORDER,
                    columns=TEMPLATE_ORDER,
                ).to_csv(
                    out_dir
                    / f"confusion_{set_name}_{algorithm}_{split_name}.csv",
                    encoding="utf-8-sig",
                )

                print(
                    f"  {algorithm:<20} "
                    f"Acc={m['accuracy']:.4f}  "
                    f"BalAcc={m['balanced_accuracy']:.4f}  "
                    f"MacroF1={m['macro_f1']:.4f}  "
                    f"Top2={m['top2_accuracy']:.4f}  "
                    f"LogLoss={m['log_loss']:.4f}",
                    flush=True,
                )

            print(
                f"  {'Lag1 baseline':<20} "
                f"Acc={lag_m['accuracy']:.4f}  "
                f"BalAcc={lag_m['balanced_accuracy']:.4f}  "
                f"MacroF1={lag_m['macro_f1']:.4f}",
                flush=True,
            )

            print(
                f"  {'Majority baseline':<20} "
                f"Acc={majority_m['accuracy']:.4f}  "
                f"BalAcc={majority_m['balanced_accuracy']:.4f}  "
                f"MacroF1={majority_m['macro_f1']:.4f}",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # Save aggregate results.
    # -------------------------------------------------------------------------

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        out_dir / "template_classifier_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        training_rows
    ).to_csv(
        out_dir / "training_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        baseline_rows
    ).to_csv(
        out_dir / "baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if per_class_frames:
        pd.concat(
            per_class_frames,
            ignore_index=True,
        ).to_csv(
            out_dir / "template_classifier_per_class_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------------------------------------------------------
    # Compact summary.
    # -------------------------------------------------------------------------

    lines = [
        f"Unified template classifier comparison - {args.year}",
        "=" * 80,
        "",
        f"Train <= {train_end.date()}",
        (
            f"Validation = "
            f"{val_start.date()} .. {val_end.date()}"
        ),
        f"Test >= {test_start.date()}",
        f"Shared train sample rows = {len(train_df):,}",
        f"Z features = {len(component_map['Z'])}",
        f"M features = {len(component_map['M'])}",
        f"U features = {len(component_map['U'])}",
        f"H features = {len(component_map['H'])}",
        "",
    ]

    baseline_df = pd.DataFrame(
        baseline_rows
    )

    for split_name in [
        "val",
        "test",
    ]:
        lines.append(
            f"[{split_name}]"
        )

        b = baseline_df[
            baseline_df["split"].eq(
                split_name
            )
        ]

        for r in b.itertuples():
            lines.append(
                f"  Baseline/{r.baseline}: "
                f"Acc={r.accuracy:.4f}, "
                f"BalAcc={r.balanced_accuracy:.4f}, "
                f"MacroF1={r.macro_f1:.4f}"
            )

        for set_name in selected_sets:
            lines.append(
                f"  {set_name}:"
            )

            sub = metrics_df[
                metrics_df["split"].eq(split_name)
                & metrics_df["feature_set"].eq(set_name)
            ]

            for r in sub.itertuples():
                lines.append(
                    f"    {r.algorithm}: "
                    f"Acc={r.accuracy:.4f}, "
                    f"BalAcc={r.balanced_accuracy:.4f}, "
                    f"MacroF1={r.macro_f1:.4f}, "
                    f"Top2={r.top2_accuracy:.4f}, "
                    f"LogLoss={r.log_loss:.4f}, "
                    f"dMacroF1_vs_Lag1="
                    f"{r.macro_f1_gain_over_lag1:+.4f}"
                )

        lines.append("")

    lines += [
        "Interpretation:",
        "  Z       : strategy profile alone.",
        "  Z_M     : incremental value of market environment M.",
        "  Z_U     : incremental value of unit physical/operational state U.",
        "  Z_M_U   : original theoretical model X=[Z,M,U].",
        "  Z_H     : predictive value of historical bid inertia H.",
        "  Z_M_U_H : enhanced model with H added to the theoretical model.",
        "",
        "Important:",
        "  H is deliberately separated from U.",
        "  Calendar features are excluded from the core Z/M/U experiment.",
        "  Compare every learned model against hist_lag1_template_id baseline.",
    ]

    summary = "\n".join(
        lines
    )

    (
        out_dir / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    config = {
        "year": args.year,
        "train_end": str(
            train_end.date()
        ),
        "val_start": str(
            val_start.date()
        ),
        "val_end": str(
            val_end.date()
        ),
        "test_start": str(
            test_start.date()
        ),
        "shared_max_train_rows": args.max_train_rows,
        "seed": args.seed,
        "selected_feature_sets": selected_sets,
        "input_components": {
            "Z": component_map["Z"],
            "M": component_map["M"],
            "U": component_map["U"],
            "H": component_map["H"],
        },
        "algorithms": algorithms,
        "logistic_regression": {
            "max_iter": args.logit_max_iter,
            "C": args.logit_c,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        "decision_tree": {
            "max_depth": args.dt_max_depth,
            "min_samples_leaf": args.dt_min_samples_leaf,
            "class_weight": "balanced",
        },
        "random_forest": {
            "n_estimators": args.rf_trees,
            "max_depth": args.rf_max_depth,
            "min_samples_leaf": args.rf_min_samples_leaf,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
        },
        "lightgbm": {
            "num_boost_round": args.lgb_rounds,
            "num_leaves": args.lgb_num_leaves,
            "learning_rate": args.lgb_learning_rate,
            "min_data_in_leaf": args.lgb_min_data_in_leaf,
            "class_weighting": "sqrt_inverse_frequency",
        },
    }

    (
        out_dir / "training_config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(
        f"Outputs: {out_dir}"
    )


if __name__ == "__main__":
    main()
