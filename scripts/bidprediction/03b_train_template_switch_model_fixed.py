#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03b_train_template_switch_model.py

Stage 3 / bidprediction
Hierarchical template prediction with explicit Z / M / U / H separation.

Theory
------
Original input structure:
    X_{i,t} = [ Z_{i,t}, M_t, U_{i,t} ]

where:
    Z = strategy profile = 9 LT + 9 ST + 8 Break
    M = market environment
    U = unit physical / operational state proxy

Additional enhancement:
    H = historical bid state / bid inertia

H is deliberately separated from U.

Hierarchical prediction
-----------------------
Layer 1:
    predict whether the current template changes relative to
    hist_lag1_template_id

        switch = 0 -> keep lag1 template
        switch = 1 -> call destination classifier

Layer 2:
    on TRUE switched training rows only,
    predict the destination template.

Final prediction:
    if P(switch) < threshold:
        template_hat = hist_lag1_template_id
    else:
        template_hat = destination classifier prediction

The destination prediction is forced to differ from lag1 template.

Feature sets
------------
Z
Z_M
Z_U
Z_M_U       <- original theoretical model X=[Z,M,U]
Z_H
Z_M_U_H     <- enhanced model with historical bid inertia H

Models
------
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. LightGBM

Validation logic
----------------
- Chronological split comes from 02_validate_prediction_dataset.py.
- Threshold is selected ONLY on validation data.
- Test data is never used for threshold tuning.
- Threshold objective: maximize final 13-class Macro-F1.
- All four algorithms use the same sampled rows within each training stage.

Outputs
-------
data/processed/bidprediction/<year>/template_switch_model/
    input_components_Z_M_U_H.csv
    feature_sets.csv
    switch_rate_summary.csv
    switch_training_summary.csv
    switch_detector_metrics.csv
    destination_classifier_metrics.csv
    hierarchical_template_metrics.csv
    threshold_selection.csv
    feature_importance_*.csv
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "This script requires LightGBM.\n"
        "Install it with:\n"
        "    pip install lightgbm"
    ) from e


# =============================================================================
# Fixed Z definition
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
LAG1_TEMPLATE = "hist_lag1_template_id"

TEMPLATE_ORDER = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
TEMPLATE_TO_INT = {t: i for i, t in enumerate(TEMPLATE_ORDER)}
INT_TO_TEMPLATE = {i: t for t, i in TEMPLATE_TO_INT.items()}

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

EXCLUDE_FEATURES = {
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

ALGORITHMS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "LightGBM",
]


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def discover_parts(path: Path) -> list[Path]:
    files = sorted(path.glob("prediction_dataset_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No prediction dataset parts found under:\n{path}"
        )
    return files


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


def load_split_dates(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun 02_validate_prediction_dataset.py first."
        )

    split = pd.read_csv(path).set_index("split")

    for name in ["train", "val", "test"]:
        if name not in split.index:
            raise ValueError(f"Missing split '{name}' in {path}")

    train_end = pd.Timestamp(split.loc["train", "last_date"])
    val_start = pd.Timestamp(split.loc["val", "first_date"])
    val_end = pd.Timestamp(split.loc["val", "last_date"])
    test_start = pd.Timestamp(split.loc["test", "first_date"])

    return train_end, val_start, val_end, test_start


def split_mask(
    local_date: pd.Series,
    split_name: str,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
):
    d = pd.to_datetime(local_date, errors="coerce").dt.normalize()

    if split_name == "train":
        return d <= train_end

    if split_name == "val":
        return (d >= val_start) & (d <= val_end)

    if split_name == "test":
        return d >= test_start

    raise ValueError(split_name)


# =============================================================================
# Feature construction: explicit Z / M / U / H
# =============================================================================

def build_feature_sets(schema: pd.DataFrame):
    fs = schema[
        schema["role"].astype(str).eq("feature")
    ].copy()

    feature_to_group = dict(
        zip(
            fs["column"].astype(str),
            fs["feature_group"].astype(str),
        )
    )

    available = set(feature_to_group) - EXCLUDE_FEATURES

    z_expected = LT_FEATURES + ST_FEATURES + BREAK_FEATURES

    missing_z = [
        c for c in z_expected
        if c not in available
    ]

    if missing_z:
        raise KeyError(
            "Finalized 26-dimensional Z is incomplete. "
            f"Missing: {missing_z}"
        )

    z = list(z_expected)

    m = [
        c
        for c, g in feature_to_group.items()
        if g == "market_environment"
        and c in available
    ]

    u = [
        c
        for c, g in feature_to_group.items()
        if g == "unit_state_proxy"
        and c in available
    ]

    h = [
        c
        for c, g in feature_to_group.items()
        if g == "participant_history"
        and c in available
    ]

    def unique(cols):
        out = []
        seen = set()
        for c in cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    z = unique(z)
    m = unique(m)
    u = unique(u)
    h = unique(h)

    components = {
        "Z": z,
        "M": m,
        "U": u,
        "H": h,
    }

    feature_sets = {
        "Z": z,
        "Z_M": unique(z + m),
        "Z_U": unique(z + u),
        "Z_M_U": unique(z + m + u),
        "Z_H": unique(z + h),
        "Z_M_U_H": unique(z + m + u + h),
    }

    return feature_sets, components, feature_to_group


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


def prepare_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = {}

    for c in features:
        if c in CATEGORICAL_FEATURES:
            out[c] = encode_categorical_column(df[c], c)
        else:
            out[c] = safe_numeric(df[c]).astype("float32")

    return pd.DataFrame(out, index=df.index)


def encode_template(s: pd.Series) -> np.ndarray:
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
            "Template labels contain missing values."
        )

    return y.to_numpy(np.int16)


def encode_lag_template(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .map(TEMPLATE_TO_INT)
    )


# =============================================================================
# Training data collection
# =============================================================================

def collect_train_pool(
    files,
    all_features,
    train_end,
    chunksize,
):
    """
    Collect all prediction-ready training rows having a valid lag1 template.

    With current data this is a few million rows. The pool is used only to
    construct shared switch/destination training samples, then can be released.
    """

    usecols = list(
        dict.fromkeys(
            [
                "local_date",
                "prediction_ready_flag",
                TARGET,
                LAG1_TEMPLATE,
            ]
            + all_features
        )
    )

    blocks = []

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(
            file,
            nrows=0,
        ).columns.tolist()

        missing = [
            c
            for c in all_features + [TARGET, LAG1_TEMPLATE]
            if c not in header
        ]

        if missing:
            raise KeyError(
                f"{file.name} missing required columns:\n{missing}"
            )

        cols = [
            c for c in usecols
            if c in header
        ]

        print(
            f"[train-pool {file_no}/{len(files)}] {file.name}",
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

            lag = encode_lag_template(
                chunk[LAG1_TEMPLATE]
            )

            mask = (
                ready
                & (d <= train_end)
                & lag.notna()
            )

            sub = chunk.loc[
                mask,
                list(dict.fromkeys([TARGET, LAG1_TEMPLATE] + all_features)),
            ].copy()

            if not sub.empty:
                blocks.append(sub)

    if not blocks:
        raise ValueError(
            "No valid prediction-ready training rows found."
        )

    train = pd.concat(
        blocks,
        ignore_index=True,
    )

    y_now = encode_template(
        train[TARGET]
    )

    y_lag = (
        encode_lag_template(
            train[LAG1_TEMPLATE]
        )
        .to_numpy(np.int16)
    )

    train["y_switch"] = (
        y_now != y_lag
    ).astype(np.int8)

    print(
        f"Train pool rows={len(train):,}, "
        f"switch={int(train['y_switch'].sum()):,} "
        f"({train['y_switch'].mean():.2%})",
        flush=True,
    )

    return train


def make_switch_training_sample(
    train_pool,
    max_rows,
    negative_to_positive,
    seed,
):
    """
    Shared binary training sample.

    All four algorithms and all feature-set ablations use these exact rows.
    Switch events are deliberately over-represented so the model does not
    collapse into always predicting no-switch.
    """

    pos = train_pool[
        train_pool["y_switch"].eq(1)
    ]

    neg = train_pool[
        train_pool["y_switch"].eq(0)
    ]

    if len(pos) == 0:
        raise ValueError(
            "No switched training samples found."
        )

    if max_rows <= 0:
        n_pos = len(pos)
        n_neg = len(neg)
    else:
        desired_neg = int(
            round(
                len(pos)
                * negative_to_positive
            )
        )

        if len(pos) + min(
            len(neg),
            desired_neg,
        ) <= max_rows:
            n_pos = len(pos)
            n_neg = min(
                len(neg),
                desired_neg,
            )
        else:
            n_pos = min(
                len(pos),
                int(
                    max_rows
                    / (1.0 + negative_to_positive)
                ),
            )
            n_neg = min(
                len(neg),
                max_rows - n_pos,
            )

    pos_s = (
        pos
        if n_pos >= len(pos)
        else pos.sample(
            n=n_pos,
            random_state=seed,
        )
    )

    neg_s = (
        neg
        if n_neg >= len(neg)
        else neg.sample(
            n=n_neg,
            random_state=seed + 1,
        )
    )

    sample = pd.concat(
        [pos_s, neg_s],
        ignore_index=True,
    )

    sample = sample.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return sample


def make_destination_training_sample(
    train_pool,
    max_rows,
    seed,
):
    """
    Destination classifier is trained ONLY on true switched rows.
    """

    switched = train_pool[
        train_pool["y_switch"].eq(1)
    ].copy()

    if switched.empty:
        raise ValueError(
            "No switched training rows for destination classifier."
        )

    if (
        max_rows > 0
        and len(switched) > max_rows
    ):
        switched = (
            switched.sample(
                n=max_rows,
                random_state=seed,
            )
            .reset_index(drop=True)
        )

    return switched


# =============================================================================
# Evaluation split loading
# =============================================================================

def collect_eval_split(
    files,
    all_features,
    split_name,
    train_end,
    val_start,
    val_end,
    test_start,
    chunksize,
):
    usecols = list(
        dict.fromkeys(
            [
                "local_date",
                "prediction_ready_flag",
                TARGET,
                LAG1_TEMPLATE,
            ]
            + all_features
        )
    )

    blocks = []

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(
            file,
            nrows=0,
        ).columns.tolist()

        cols = [
            c for c in usecols
            if c in header
        ]

        print(
            f"[load {split_name} {file_no}/{len(files)}] {file.name}",
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

            lag = encode_lag_template(
                chunk[LAG1_TEMPLATE]
            )

            mask = (
                ready
                & sm
                & lag.notna()
            )

            sub = chunk.loc[
                mask,
                list(dict.fromkeys([TARGET, LAG1_TEMPLATE] + all_features)),
            ].copy()

            if not sub.empty:
                blocks.append(sub)

    if not blocks:
        raise ValueError(
            f"No rows found for split={split_name}"
        )

    out = pd.concat(
        blocks,
        ignore_index=True,
    )

    y_now = encode_template(
        out[TARGET]
    )

    y_lag = (
        encode_lag_template(
            out[LAG1_TEMPLATE]
        )
        .to_numpy(np.int16)
    )

    out["y_switch"] = (
        y_now != y_lag
    ).astype(np.int8)

    print(
        f"{split_name}: rows={len(out):,}, "
        f"switch={int(out['y_switch'].sum()):,} "
        f"({out['y_switch'].mean():.2%})",
        flush=True,
    )

    return out


# =============================================================================
# Preprocessing
# =============================================================================

def fit_imputer(X: pd.DataFrame):
    imp = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    imp.fit(X)
    return imp


def transform_X(
    X: pd.DataFrame,
    imputer: SimpleImputer,
):
    return imputer.transform(
        X
    ).astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# Model fitting
# =============================================================================

def fit_logistic(
    X,
    y,
    seed,
    max_iter,
    C,
):
    scaler = StandardScaler()

    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
    )

    model.fit(
        Xs,
        y,
    )

    return {
        "model": model,
        "scaler": scaler,
    }


def fit_tree(
    X,
    y,
    seed,
    max_depth,
    min_samples_leaf,
):
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
    )

    model.fit(
        X,
        y,
    )

    return {
        "model": model,
        "scaler": None,
    }


def fit_rf(
    X,
    y,
    seed,
    n_estimators,
    max_depth,
    min_samples_leaf,
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

    model.fit(
        X,
        y,
    )

    return {
        "model": model,
        "scaler": None,
    }


def sqrt_multiclass_sample_weight(
    y,
):
    counts = np.bincount(
        y,
        minlength=len(TEMPLATE_ORDER),
    ).astype(float)

    class_weight = np.ones_like(
        counts
    )

    valid = counts > 0

    raw = np.sqrt(
        counts[valid].sum()
        / counts[valid]
    )

    norm = (
        np.sum(
            counts[valid] * raw
        )
        / counts[valid].sum()
    )

    class_weight[valid] = (
        raw / norm
    )

    return class_weight[y]


def fit_lgb_binary(
    X,
    y,
    feature_names,
    seed,
    rounds,
    leaves,
    learning_rate,
    min_data_in_leaf,
):
    n_pos = int(
        np.sum(y == 1)
    )

    n_neg = int(
        np.sum(y == 0)
    )

    scale_pos_weight = (
        n_neg / n_pos
        if n_pos > 0
        else 1.0
    )

    ds = lgb.Dataset(
        X,
        label=y,
        feature_name=feature_names,
        free_raw_data=True,
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": learning_rate,
        "num_leaves": leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 0,
    }

    model = lgb.train(
        params,
        ds,
        num_boost_round=rounds,
    )

    return {
        "model": model,
        "scaler": None,
    }


def fit_lgb_multiclass(
    X,
    y,
    feature_names,
    seed,
    rounds,
    leaves,
    learning_rate,
    min_data_in_leaf,
):
    sample_weight = (
        sqrt_multiclass_sample_weight(y)
    )

    ds = lgb.Dataset(
        X,
        label=y,
        weight=sample_weight,
        feature_name=feature_names,
        free_raw_data=True,
    )

    params = {
        "objective": "multiclass",
        "num_class": len(TEMPLATE_ORDER),
        "metric": "multi_logloss",
        "learning_rate": learning_rate,
        "num_leaves": leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 0,
    }

    model = lgb.train(
        params,
        ds,
        num_boost_round=rounds,
    )

    return {
        "model": model,
        "scaler": None,
    }


def fit_four_models(
    X,
    y,
    features,
    binary,
    args,
):
    models = {}

    print("    fit LogisticRegression", flush=True)
    models["LogisticRegression"] = fit_logistic(
        X,
        y,
        args.seed,
        args.logit_max_iter,
        args.logit_c,
    )

    print("    fit DecisionTree", flush=True)
    models["DecisionTree"] = fit_tree(
        X,
        y,
        args.seed,
        args.dt_max_depth,
        args.dt_min_samples_leaf,
    )

    print("    fit RandomForest", flush=True)
    models["RandomForest"] = fit_rf(
        X,
        y,
        args.seed,
        args.rf_trees,
        args.rf_max_depth,
        args.rf_min_samples_leaf,
    )

    print("    fit LightGBM", flush=True)

    if binary:
        models["LightGBM"] = fit_lgb_binary(
            X,
            y,
            features,
            args.seed,
            args.lgb_rounds,
            args.lgb_num_leaves,
            args.lgb_learning_rate,
            args.lgb_min_data_in_leaf,
        )
    else:
        models["LightGBM"] = fit_lgb_multiclass(
            X,
            y,
            features,
            args.seed,
            args.lgb_rounds,
            args.lgb_num_leaves,
            args.lgb_learning_rate,
            args.lgb_min_data_in_leaf,
        )

    return models


# =============================================================================
# Prediction
# =============================================================================

def predict_binary_score(
    algorithm,
    payload,
    X,
):
    model = payload["model"]

    if algorithm == "LogisticRegression":
        p = model.predict_proba(
            payload["scaler"].transform(X)
        )[:, 1]

    elif algorithm in {
        "DecisionTree",
        "RandomForest",
    }:
        p = model.predict_proba(X)[:, 1]

    elif algorithm == "LightGBM":
        p = model.predict(X)

    else:
        raise ValueError(algorithm)

    return np.asarray(
        p,
        dtype=np.float64,
    )


def predict_multiclass_proba(
    algorithm,
    payload,
    X,
):
    model = payload["model"]

    if algorithm == "LogisticRegression":
        p = model.predict_proba(
            payload["scaler"].transform(X)
        )

    elif algorithm in {
        "DecisionTree",
        "RandomForest",
    }:
        p = model.predict_proba(X)

    elif algorithm == "LightGBM":
        p = model.predict(X)

    else:
        raise ValueError(algorithm)

    p = np.asarray(
        p,
        dtype=np.float64,
    )

    # Expand sklearn probabilities to fixed 13 classes if needed.
    if algorithm != "LightGBM":
        classes = np.asarray(
            model.classes_,
            dtype=int,
        )

        if (
            p.shape[1] != len(TEMPLATE_ORDER)
            or not np.array_equal(
                classes,
                np.arange(len(TEMPLATE_ORDER)),
            )
        ):
            full = np.zeros(
                (
                    len(X),
                    len(TEMPLATE_ORDER),
                ),
                dtype=np.float64,
            )

            full[:, classes] = p
            p = full

    return p


def force_destination_not_lag1(
    proba,
    lag_class,
):
    p = np.array(
        proba,
        copy=True,
    )

    rows = np.arange(
        len(p)
    )

    p[
        rows,
        lag_class,
    ] = 0.0

    denom = p.sum(
        axis=1,
        keepdims=True,
    )

    bad = (
        denom[:, 0] <= 0
    )

    if bad.any():
        bad_rows = np.where(
            bad
        )[0]

        p[bad_rows, :] = 1.0

        p[
            bad_rows,
            lag_class[bad_rows],
        ] = 0.0

        denom = p.sum(
            axis=1,
            keepdims=True,
        )

    p /= denom

    pred = np.argmax(
        p,
        axis=1,
    ).astype(np.int16)

    return pred, p


# =============================================================================
# Metrics
# =============================================================================

def multiclass_metrics(
    y_true,
    y_pred,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(TEMPLATE_ORDER)
        ),
    )

    total = cm.sum()
    diag = np.diag(cm)

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

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
        predicted,
        out=np.zeros_like(
            diag,
            dtype=float,
        ),
        where=predicted > 0,
    )

    f1 = np.divide(
        2.0
        * precision
        * recall,
        precision + recall,
        out=np.zeros_like(
            recall,
            dtype=float,
        ),
        where=(precision + recall) > 0,
    )

    valid = support > 0

    return {
        "accuracy": (
            float(
                diag.sum()
                / total
            )
            if total
            else np.nan
        ),
        "balanced_accuracy": (
            float(
                recall[valid].mean()
            )
            if valid.any()
            else np.nan
        ),
        "macro_f1": (
            float(
                f1[valid].mean()
            )
            if valid.any()
            else np.nan
        ),
        "weighted_f1": (
            float(
                np.sum(
                    f1 * support
                )
                / total
            )
            if total
            else np.nan
        ),
        "cm": cm,
    }


def binary_metrics(
    y_true,
    score,
    threshold,
):
    pred = (
        score >= threshold
    ).astype(np.int8)

    result = {
        "accuracy": accuracy_score(
            y_true,
            pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            pred,
        ),
        "precision": precision_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "true_switch_rate": float(
            np.mean(y_true)
        ),
        "predicted_switch_rate": float(
            np.mean(pred)
        ),
    }

    if len(
        np.unique(y_true)
    ) == 2:
        result["roc_auc"] = roc_auc_score(
            y_true,
            score,
        )

        result["pr_auc"] = average_precision_score(
            y_true,
            score,
        )
    else:
        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


# =============================================================================
# Threshold tuning
# =============================================================================

def tune_threshold(
    y_template,
    lag_template,
    switch_score,
    destination_pred,
    thresholds,
):
    rows = []
    best_row = None
    best_key = None

    y_switch = (
        y_template != lag_template
    ).astype(np.int8)

    for threshold in thresholds:
        switch_hat = (
            switch_score >= threshold
        )

        final_pred = (
            lag_template.copy()
        )

        final_pred[
            switch_hat
        ] = destination_pred[
            switch_hat
        ]

        mm = multiclass_metrics(
            y_template,
            final_pred,
        )

        bm = binary_metrics(
            y_switch,
            switch_score,
            threshold,
        )

        row = {
            "threshold": float(threshold),
            "hier_accuracy": mm["accuracy"],
            "hier_balanced_accuracy": mm["balanced_accuracy"],
            "hier_macro_f1": mm["macro_f1"],
            "switch_precision": bm["precision"],
            "switch_recall": bm["recall"],
            "switch_f1": bm["f1"],
            "predicted_switch_rate": bm["predicted_switch_rate"],
        }

        rows.append(row)

        key = (
            mm["macro_f1"],
            mm["balanced_accuracy"],
            mm["accuracy"],
            -abs(float(threshold) - 0.5),
        )

        if (
            best_key is None
            or key > best_key
        ):
            best_key = key
            best_row = row

    return pd.DataFrame(
        rows
    ), best_row


# =============================================================================
# Importance
# =============================================================================

def save_importance(
    out_dir,
    stage,
    feature_set,
    algorithm,
    features,
    payload,
):
    model = payload["model"]

    if algorithm == "LogisticRegression":
        coef = np.asarray(
            model.coef_
        )

        if coef.ndim == 1:
            importance = np.abs(
                coef
            )
        else:
            importance = np.mean(
                np.abs(coef),
                axis=0,
            )

        kind = (
            "mean_abs_standardized_coefficient"
        )

    elif algorithm in {
        "DecisionTree",
        "RandomForest",
    }:
        importance = (
            model.feature_importances_
        )
        kind = (
            "impurity_importance"
        )

    elif algorithm == "LightGBM":
        importance = (
            model.feature_importance(
                importance_type="gain"
            )
        )
        kind = "gain"

    else:
        return

    df = pd.DataFrame({
        "feature": features,
        "importance": importance,
        "importance_type": kind,
    })

    total = df["importance"].sum()

    df["importance_share"] = (
        df["importance"] / total
        if total > 0
        else 0.0
    )

    df.sort_values(
        "importance",
        ascending=False,
    ).to_csv(
        out_dir
        / f"feature_importance_{stage}_{feature_set}_{algorithm}.csv",
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

    p.add_argument(
        "--max-switch-train-rows",
        type=int,
        default=500_000,
        help=(
            "Maximum shared rows for binary switch training."
        ),
    )

    p.add_argument(
        "--negative-to-positive",
        type=float,
        default=3.0,
        help=(
            "No-switch : switch ratio in binary training sample."
        ),
    )

    p.add_argument(
        "--max-destination-train-rows",
        type=int,
        default=250_000,
        help=(
            "Maximum TRUE switched rows for destination classifier. "
            "0 means all switched train rows."
        ),
    )

    p.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--threshold-max",
        type=float,
        default=0.95,
    )

    p.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
    )

    # Logistic
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
        default=80,
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
        default=40,
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
        default=120,
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

    (
        feature_sets,
        components,
        feature_to_group,
    ) = build_feature_sets(
        schema
    )

    selected_sets = (
        list(feature_sets)
        if args.feature_set == "all"
        else [args.feature_set]
    )

    (
        train_end,
        val_start,
        val_end,
        test_start,
    ) = load_split_dates(
        base
        / "validation"
        / "temporal_split_summary.csv"
    )

    out_dir = ensure_dir(
        base
        / "template_switch_model"
    )

    # -------------------------------------------------------------------------
    # Save exact theoretical components.
    # -------------------------------------------------------------------------

    component_rows = []

    for component, cols in components.items():
        for order, c in enumerate(cols):
            component_rows.append({
                "component": component,
                "order": order,
                "feature": c,
                "feature_group": feature_to_group.get(
                    c,
                    "",
                ),
            })

    pd.DataFrame(
        component_rows
    ).to_csv(
        out_dir
        / "input_components_Z_M_U_H.csv",
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
                "feature_group": feature_to_group.get(
                    c,
                    "",
                ),
            })

    pd.DataFrame(
        feature_rows
    ).to_csv(
        out_dir
        / "feature_sets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Hierarchical template switch model with Z / M / U / H separation")
    print("=" * 80)
    print(f"Year:            {args.year}")
    print(f"Train <=         {train_end.date()}")
    print(
        f"Validation:      "
        f"{val_start.date()} .. {val_end.date()}"
    )
    print(f"Test >=          {test_start.date()}")
    print()
    print(
        f"Z features:      {len(components['Z'])}"
    )
    print(
        f"M features:      {len(components['M'])}"
    )
    print(
        f"U features:      {len(components['U'])}"
    )
    print(
        f"H features:      {len(components['H'])}"
    )
    print(
        f"Feature sets:    {', '.join(selected_sets)}"
    )
    print(
        f"Algorithms:      {', '.join(ALGORITHMS)}"
    )
    print()

    # -------------------------------------------------------------------------
    # Union of selected features.
    # -------------------------------------------------------------------------

    all_features = []
    seen = set()

    for set_name in selected_sets:
        for c in feature_sets[set_name]:
            if c not in seen:
                all_features.append(c)
                seen.add(c)

    # -------------------------------------------------------------------------
    # Collect train / validation / test once.
    # -------------------------------------------------------------------------

    train_pool = collect_train_pool(
        files=files,
        all_features=all_features,
        train_end=train_end,
        chunksize=args.chunksize,
    )

    val_df = collect_eval_split(
        files=files,
        all_features=all_features,
        split_name="val",
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        chunksize=args.chunksize,
    )

    test_df = collect_eval_split(
        files=files,
        all_features=all_features,
        split_name="test",
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        chunksize=args.chunksize,
    )

    switch_summary = pd.DataFrame([
        {
            "split": "train",
            "rows": len(train_pool),
            "switch_rows": int(
                train_pool["y_switch"].sum()
            ),
            "switch_rate": float(
                train_pool["y_switch"].mean()
            ),
        },
        {
            "split": "val",
            "rows": len(val_df),
            "switch_rows": int(
                val_df["y_switch"].sum()
            ),
            "switch_rate": float(
                val_df["y_switch"].mean()
            ),
        },
        {
            "split": "test",
            "rows": len(test_df),
            "switch_rows": int(
                test_df["y_switch"].sum()
            ),
            "switch_rate": float(
                test_df["y_switch"].mean()
            ),
        },
    ])

    switch_summary.to_csv(
        out_dir
        / "switch_rate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Shared training rows across all feature sets and algorithms.
    switch_train = make_switch_training_sample(
        train_pool=train_pool,
        max_rows=args.max_switch_train_rows,
        negative_to_positive=args.negative_to_positive,
        seed=args.seed,
    )

    destination_train = make_destination_training_sample(
        train_pool=train_pool,
        max_rows=args.max_destination_train_rows,
        seed=args.seed,
    )

    pd.DataFrame([
        {
            "dataset": "switch_detector_train",
            "rows": len(switch_train),
            "switch_rows": int(
                switch_train["y_switch"].sum()
            ),
            "switch_rate": float(
                switch_train["y_switch"].mean()
            ),
        },
        {
            "dataset": "destination_train",
            "rows": len(destination_train),
            "switch_rows": len(destination_train),
            "switch_rate": 1.0,
        },
    ]).to_csv(
        out_dir
        / "switch_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Full train pool no longer needed.
    del train_pool

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max
        + 0.5 * args.threshold_step,
        args.threshold_step,
    )

    switch_metric_rows = []
    destination_metric_rows = []
    hierarchy_metric_rows = []
    threshold_frames = []

    # -------------------------------------------------------------------------
    # Feature-set loop
    # -------------------------------------------------------------------------

    for set_no, set_name in enumerate(
        selected_sets,
        1,
    ):
        features = feature_sets[
            set_name
        ]

        print()
        print("=" * 80)
        print(
            f"[feature set {set_no}/{len(selected_sets)}] "
            f"{set_name}, features={len(features)}"
        )
        print("=" * 80)

        # -------------------------------------------------------------
        # Shared preprocessing for this feature set.
        # Imputer fitted on binary training sample only.
        # -------------------------------------------------------------

        X_switch_df = prepare_X(
            switch_train,
            features,
        )

        imputer = fit_imputer(
            X_switch_df
        )

        X_switch = transform_X(
            X_switch_df,
            imputer,
        )

        y_switch_train = (
            switch_train[
                "y_switch"
            ].to_numpy(
                np.int8
            )
        )

        X_destination = transform_X(
            prepare_X(
                destination_train,
                features,
            ),
            imputer,
        )

        y_destination = encode_template(
            destination_train[
                TARGET
            ]
        )

        X_val = transform_X(
            prepare_X(
                val_df,
                features,
            ),
            imputer,
        )

        y_val_template = encode_template(
            val_df[TARGET]
        )

        lag_val = (
            encode_lag_template(
                val_df[LAG1_TEMPLATE]
            )
            .to_numpy(np.int16)
        )

        y_val_switch = (
            val_df[
                "y_switch"
            ].to_numpy(
                np.int8
            )
        )

        X_test = transform_X(
            prepare_X(
                test_df,
                features,
            ),
            imputer,
        )

        y_test_template = encode_template(
            test_df[TARGET]
        )

        lag_test = (
            encode_lag_template(
                test_df[LAG1_TEMPLATE]
            )
            .to_numpy(np.int16)
        )

        y_test_switch = (
            test_df[
                "y_switch"
            ].to_numpy(
                np.int8
            )
        )

        print("[stage 1] train switch detector", flush=True)

        switch_models = fit_four_models(
            X=X_switch,
            y=y_switch_train,
            features=features,
            binary=True,
            args=args,
        )

        print("[stage 2] train destination classifier", flush=True)

        destination_models = fit_four_models(
            X=X_destination,
            y=y_destination,
            features=features,
            binary=False,
            args=args,
        )

        joblib.dump(
            {
                "features": features,
                "imputer": imputer,
                "template_to_int": TEMPLATE_TO_INT,
                "mode_to_int": MODE_TO_INT,
            },
            out_dir
            / f"preprocess_{set_name}.joblib",
        )

        # -------------------------------------------------------------
        # Algorithm loop
        # -------------------------------------------------------------

        for algorithm in ALGORITHMS:
            print(
                f"[evaluate] {set_name} / {algorithm}",
                flush=True,
            )

            sw_model = (
                switch_models[
                    algorithm
                ]
            )

            dst_model = (
                destination_models[
                    algorithm
                ]
            )

            # Save models.
            for stage_name, payload in [
                (
                    "switch",
                    sw_model,
                ),
                (
                    "destination",
                    dst_model,
                ),
            ]:
                if algorithm == "LightGBM":
                    payload[
                        "model"
                    ].save_model(
                        str(
                            out_dir
                            / f"model_{stage_name}_{set_name}_{algorithm}.txt"
                        )
                    )
                else:
                    joblib.dump(
                        payload,
                        out_dir
                        / f"model_{stage_name}_{set_name}_{algorithm}.joblib",
                    )

                save_importance(
                    out_dir=out_dir,
                    stage=stage_name,
                    feature_set=set_name,
                    algorithm=algorithm,
                    features=features,
                    payload=payload,
                )

            # ---------------------------------------------------------
            # Validation / test scores.
            # ---------------------------------------------------------

            sw_val_score = (
                predict_binary_score(
                    algorithm,
                    sw_model,
                    X_val,
                )
            )

            sw_test_score = (
                predict_binary_score(
                    algorithm,
                    sw_model,
                    X_test,
                )
            )

            dst_val_proba = (
                predict_multiclass_proba(
                    algorithm,
                    dst_model,
                    X_val,
                )
            )

            dst_test_proba = (
                predict_multiclass_proba(
                    algorithm,
                    dst_model,
                    X_test,
                )
            )

            (
                dst_val_pred,
                dst_val_forced_proba,
            ) = force_destination_not_lag1(
                dst_val_proba,
                lag_val,
            )

            (
                dst_test_pred,
                dst_test_forced_proba,
            ) = force_destination_not_lag1(
                dst_test_proba,
                lag_test,
            )

            # ---------------------------------------------------------
            # Destination-only metrics on TRUE switch rows.
            # ---------------------------------------------------------

            for (
                split_name,
                y_template,
                y_switch,
                dst_pred,
                dst_proba,
            ) in [
                (
                    "val",
                    y_val_template,
                    y_val_switch,
                    dst_val_pred,
                    dst_val_forced_proba,
                ),
                (
                    "test",
                    y_test_template,
                    y_test_switch,
                    dst_test_pred,
                    dst_test_forced_proba,
                ),
            ]:
                mask = (
                    y_switch == 1
                )

                yt = y_template[
                    mask
                ]

                yp = dst_pred[
                    mask
                ]

                pp = dst_proba[
                    mask
                ]

                mm = multiclass_metrics(
                    yt,
                    yp,
                )

                if len(yt):
                    eps = 1e-15

                    p_true = pp[
                        np.arange(
                            len(yt)
                        ),
                        yt,
                    ]

                    log_loss = float(
                        -np.log(
                            np.clip(
                                p_true,
                                eps,
                                1.0,
                            )
                        ).mean()
                    )

                    top2 = np.argpartition(
                        pp,
                        kth=-2,
                        axis=1,
                    )[:, -2:]

                    top2_acc = float(
                        np.any(
                            top2
                            == yt[:, None],
                            axis=1,
                        ).mean()
                    )

                else:
                    log_loss = np.nan
                    top2_acc = np.nan

                destination_metric_rows.append({
                    "feature_set": set_name,
                    "algorithm": algorithm,
                    "split": split_name,
                    "rows": int(
                        mask.sum()
                    ),
                    "accuracy": mm[
                        "accuracy"
                    ],
                    "balanced_accuracy": mm[
                        "balanced_accuracy"
                    ],
                    "macro_f1": mm[
                        "macro_f1"
                    ],
                    "weighted_f1": mm[
                        "weighted_f1"
                    ],
                    "top2_accuracy": top2_acc,
                    "log_loss": log_loss,
                })

            # ---------------------------------------------------------
            # Select threshold ONLY on validation.
            # ---------------------------------------------------------

            threshold_grid, best = (
                tune_threshold(
                    y_template=y_val_template,
                    lag_template=lag_val,
                    switch_score=sw_val_score,
                    destination_pred=dst_val_pred,
                    thresholds=thresholds,
                )
            )

            threshold_grid.insert(
                0,
                "feature_set",
                set_name,
            )

            threshold_grid.insert(
                1,
                "algorithm",
                algorithm,
            )

            threshold_frames.append(
                threshold_grid
            )

            best_threshold = float(
                best[
                    "threshold"
                ]
            )

            # ---------------------------------------------------------
            # Binary switch detector metrics.
            # ---------------------------------------------------------

            for (
                split_name,
                y_switch,
                score,
            ) in [
                (
                    "val",
                    y_val_switch,
                    sw_val_score,
                ),
                (
                    "test",
                    y_test_switch,
                    sw_test_score,
                ),
            ]:
                bm = binary_metrics(
                    y_true=y_switch,
                    score=score,
                    threshold=best_threshold,
                )

                switch_metric_rows.append({
                    "feature_set": set_name,
                    "algorithm": algorithm,
                    "split": split_name,
                    "threshold": best_threshold,
                    **bm,
                })

            # ---------------------------------------------------------
            # End-to-end hierarchical evaluation.
            # ---------------------------------------------------------

            for (
                split_name,
                y_template,
                lag_template,
                switch_score,
                destination_pred,
            ) in [
                (
                    "val",
                    y_val_template,
                    lag_val,
                    sw_val_score,
                    dst_val_pred,
                ),
                (
                    "test",
                    y_test_template,
                    lag_test,
                    sw_test_score,
                    dst_test_pred,
                ),
            ]:
                switch_hat = (
                    switch_score
                    >= best_threshold
                )

                final_pred = (
                    lag_template.copy()
                )

                final_pred[
                    switch_hat
                ] = destination_pred[
                    switch_hat
                ]

                hm = multiclass_metrics(
                    y_template,
                    final_pred,
                )

                lag_m = multiclass_metrics(
                    y_template,
                    lag_template,
                )

                hierarchy_metric_rows.append({
                    "feature_set": set_name,
                    "algorithm": algorithm,
                    "split": split_name,
                    "threshold": best_threshold,
                    "rows": len(
                        y_template
                    ),
                    "accuracy": hm[
                        "accuracy"
                    ],
                    "balanced_accuracy": hm[
                        "balanced_accuracy"
                    ],
                    "macro_f1": hm[
                        "macro_f1"
                    ],
                    "weighted_f1": hm[
                        "weighted_f1"
                    ],
                    "lag1_accuracy": lag_m[
                        "accuracy"
                    ],
                    "lag1_balanced_accuracy": lag_m[
                        "balanced_accuracy"
                    ],
                    "lag1_macro_f1": lag_m[
                        "macro_f1"
                    ],
                    "accuracy_gain_over_lag1": (
                        hm[
                            "accuracy"
                        ]
                        - lag_m[
                            "accuracy"
                        ]
                    ),
                    "balanced_accuracy_gain_over_lag1": (
                        hm[
                            "balanced_accuracy"
                        ]
                        - lag_m[
                            "balanced_accuracy"
                        ]
                    ),
                    "macro_f1_gain_over_lag1": (
                        hm[
                            "macro_f1"
                        ]
                        - lag_m[
                            "macro_f1"
                        ]
                    ),
                    "true_switch_rate": float(
                        np.mean(
                            y_template
                            != lag_template
                        )
                    ),
                    "predicted_switch_rate": float(
                        np.mean(
                            switch_hat
                        )
                    ),
                })

                pd.DataFrame(
                    hm["cm"],
                    index=TEMPLATE_ORDER,
                    columns=TEMPLATE_ORDER,
                ).to_csv(
                    out_dir
                    / f"confusion_hierarchical_{set_name}_{algorithm}_{split_name}.csv",
                    encoding="utf-8-sig",
                )

            print(
                f"    selected threshold={best_threshold:.2f}",
                flush=True,
            )

        # Release large matrices before next feature set.
        del X_switch_df
        del X_switch
        del X_destination
        del X_val
        del X_test
        del switch_models
        del destination_models

    # -------------------------------------------------------------------------
    # Save aggregate outputs.
    # -------------------------------------------------------------------------

    switch_df = pd.DataFrame(
        switch_metric_rows
    )

    destination_df = pd.DataFrame(
        destination_metric_rows
    )

    hierarchy_df = pd.DataFrame(
        hierarchy_metric_rows
    )

    switch_df.to_csv(
        out_dir
        / "switch_detector_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    destination_df.to_csv(
        out_dir
        / "destination_classifier_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hierarchy_df.to_csv(
        out_dir
        / "hierarchical_template_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if threshold_frames:
        pd.concat(
            threshold_frames,
            ignore_index=True,
        ).to_csv(
            out_dir
            / "threshold_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------------------

    lines = [
        f"Hierarchical template switch prediction with Z/M/U/H separation - {args.year}",
        "=" * 80,
        "",
        f"Train <= {train_end.date()}",
        (
            f"Validation = "
            f"{val_start.date()} .. {val_end.date()}"
        ),
        f"Test >= {test_start.date()}",
        "",
        f"Z = {len(components['Z'])} strategy-profile features",
        f"M = {len(components['M'])} market-environment features",
        f"U = {len(components['U'])} unit-state proxy features",
        f"H = {len(components['H'])} bid-history features",
        "",
        "Observed switch rates:",
    ]

    for r in switch_summary.itertuples():
        lines.append(
            f"  {r.split}: "
            f"{int(r.switch_rows):,}/{int(r.rows):,} "
            f"({r.switch_rate:.2%})"
        )

    lines.append("")

    for split_name in [
        "val",
        "test",
    ]:
        lines.append(
            f"[{split_name}]"
        )

        sh = hierarchy_df[
            hierarchy_df[
                "split"
            ].eq(
                split_name
            )
        ]

        ss = switch_df[
            switch_df[
                "split"
            ].eq(
                split_name
            )
        ]

        sd = destination_df[
            destination_df[
                "split"
            ].eq(
                split_name
            )
        ]

        for set_name in selected_sets:
            lines.append(
                f"  {set_name}:"
            )

            for algorithm in ALGORITHMS:
                h = sh[
                    sh[
                        "feature_set"
                    ].eq(
                        set_name
                    )
                    & sh[
                        "algorithm"
                    ].eq(
                        algorithm
                    )
                ]

                s = ss[
                    ss[
                        "feature_set"
                    ].eq(
                        set_name
                    )
                    & ss[
                        "algorithm"
                    ].eq(
                        algorithm
                    )
                ]

                d = sd[
                    sd[
                        "feature_set"
                    ].eq(
                        set_name
                    )
                    & sd[
                        "algorithm"
                    ].eq(
                        algorithm
                    )
                ]

                if (
                    h.empty
                    or s.empty
                    or d.empty
                ):
                    continue

                h = h.iloc[0]
                s = s.iloc[0]
                d = d.iloc[0]

                lines.append(
                    f"    {algorithm}: "
                    f"HierAcc={h['accuracy']:.4f}, "
                    f"HierBalAcc={h['balanced_accuracy']:.4f}, "
                    f"HierMacroF1={h['macro_f1']:.4f}, "
                    f"dAcc_vs_Lag1={h['accuracy_gain_over_lag1']:+.4f}, "
                    f"dMacroF1_vs_Lag1={h['macro_f1_gain_over_lag1']:+.4f}; "
                    f"SwitchF1={s['f1']:.4f}, "
                    f"SwitchRecall={s['recall']:.4f}, "
                    f"PR-AUC={s['pr_auc']:.4f}; "
                    f"DestAcc={d['accuracy']:.4f}, "
                    f"DestMacroF1={d['macro_f1']:.4f}; "
                    f"threshold={h['threshold']:.2f}"
                )

            lines.append("")

    lines += [
        "Interpretation:",
        "  Z       : strategy profile only.",
        "  Z_M     : incremental effect of market environment M.",
        "  Z_U     : incremental effect of unit state U.",
        "  Z_M_U   : original theoretical model X=[Z,M,U].",
        "  Z_H     : effect of bid-history inertia H.",
        "  Z_M_U_H : enhanced model with H added.",
        "",
        "Decision rule:",
        "  The hierarchical approach is useful only if held-out test",
        "  dAcc_vs_Lag1 and/or dMacroF1_vs_Lag1 become positive.",
    ]

    summary = "\n".join(
        lines
    )

    (
        out_dir
        / "summary.txt"
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
        "components": components,
        "feature_sets": selected_sets,
        "algorithms": ALGORITHMS,
        "max_switch_train_rows": args.max_switch_train_rows,
        "negative_to_positive": args.negative_to_positive,
        "max_destination_train_rows": args.max_destination_train_rows,
        "threshold_grid": {
            "min": args.threshold_min,
            "max": args.threshold_max,
            "step": args.threshold_step,
            "selection_objective": "validation_end_to_end_macro_f1",
        },
        "seed": args.seed,
    }

    (
        out_dir
        / "training_config.json"
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
