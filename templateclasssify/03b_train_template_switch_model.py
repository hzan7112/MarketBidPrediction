#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03b_train_template_switch_model.py

Hierarchical template prediction after adding transition_strategy_profile.

Layer 1:
    switch / no-switch relative to hist_lag1_template_id

Layer 2:
    destination template classifier trained only on TRUE historical switches

Final:
    no-switch -> keep hist_lag1_template_id
    switch    -> destination classifier

Input decomposition
-------------------
Z_base : finalized 26-dimensional 9 LT + 9 ST + 8 Break
Z_tr   : 26-dimensional transition_strategy_profile from 02b
M      : market_environment
U      : unit_state_proxy
H      : participant_history

Feature ablations
-----------------
Z_base
Z_base_tr

Z_base_M_U
Z_base_tr_M_U

Z_base_H
Z_base_tr_H

Z_base_M_U_H
Z_base_tr_M_U_H

Important threshold rule
------------------------
Two validation thresholds are reported separately:

1. binary_threshold
   maximizes switch-detector F1 on validation.
   Used only to assess switch-detection capability.

2. hierarchy_threshold
   maximizes final 13-class Macro-F1 on validation.
   The candidate set also includes threshold=1.01, which means
   "never trigger switch" and exactly reproduces the lag1 baseline.

This prevents the binary switch metric from being obscured by a deliberately
conservative final hierarchical decision.

Outputs
-------
data/processed/bidprediction/<year>/template_switch_model_transition/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "LightGBM is required. Install with: pip install lightgbm"
    ) from e


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

MODE_TO_INT = {"flat": 0, "block": 1, "sloped": 2}

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
    "tr_ready_flag",
}

ALGORITHMS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "LightGBM",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def unique_list(cols):
    out, seen = [], set()
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def discover_parts(path: Path) -> list[Path]:
    files = sorted(path.glob("prediction_dataset_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction dataset parts under {path}")
    return files


def load_csv_schema(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"column", "role", "feature_group"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} missing schema columns: {sorted(missing)}")
    return df


def load_split_dates(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun 02_validate_prediction_dataset.py first."
        )

    split = pd.read_csv(path).set_index("split")

    for name in ["train", "val", "test"]:
        if name not in split.index:
            raise ValueError(f"Missing temporal split: {name}")

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
):
    d = pd.to_datetime(local_date, errors="coerce").dt.normalize()

    if split_name == "train":
        return d <= train_end
    if split_name == "val":
        return (d >= val_start) & (d <= val_end)
    if split_name == "test":
        return d >= test_start

    raise ValueError(split_name)


def build_feature_sets(
    prediction_schema: pd.DataFrame,
    transition_schema: pd.DataFrame,
):
    ps = prediction_schema[
        prediction_schema["role"].astype(str).eq("feature")
    ].copy()

    feature_to_group = dict(
        zip(
            ps["column"].astype(str),
            ps["feature_group"].astype(str),
        )
    )

    available = set(feature_to_group) - EXCLUDE_FEATURES

    z_base = LT_FEATURES + ST_FEATURES + BREAK_FEATURES
    missing_z = [c for c in z_base if c not in available]
    if missing_z:
        raise KeyError(f"Missing finalized Z_base columns: {missing_z}")

    m = [
        c for c, g in feature_to_group.items()
        if g == "market_environment" and c in available
    ]
    u = [
        c for c, g in feature_to_group.items()
        if g == "unit_state_proxy" and c in available
    ]
    h = [
        c for c, g in feature_to_group.items()
        if g == "participant_history" and c in available
    ]

    ts = transition_schema[
        transition_schema["role"].astype(str).eq("feature")
    ].copy()

    z_tr = ts["column"].astype(str).tolist()
    if not z_tr:
        raise ValueError("No transition_strategy_profile features found.")

    z_base = unique_list(z_base)
    z_tr = unique_list(z_tr)
    m, u, h = map(unique_list, [m, u, h])

    feature_sets = {
        "Z_base": z_base,
        "Z_base_tr": unique_list(z_base + z_tr),

        "Z_base_M_U": unique_list(z_base + m + u),
        "Z_base_tr_M_U": unique_list(z_base + z_tr + m + u),

        "Z_base_H": unique_list(z_base + h),
        "Z_base_tr_H": unique_list(z_base + z_tr + h),

        "Z_base_M_U_H": unique_list(z_base + m + u + h),
        "Z_base_tr_M_U_H": unique_list(z_base + z_tr + m + u + h),
    }

    components = {
        "Z_base": z_base,
        "Z_tr": z_tr,
        "M": m,
        "U": u,
        "H": h,
    }

    all_group = dict(feature_to_group)
    for c in z_tr:
        all_group[c] = "transition_strategy_profile"

    return feature_sets, components, all_group


def load_transition_profile(
    path: Path,
    transition_features: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun 02b_build_transition_features.py first."
        )

    required = ["participant_id", "local_date"] + transition_features
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in required if c not in header]
    if missing:
        raise KeyError(f"{path.name} missing columns: {missing}")

    tr = pd.read_csv(
        path,
        usecols=required,
        low_memory=False,
    )

    tr["participant_id"] = (
        tr["participant_id"].astype("string").str.strip()
    )
    tr["local_date"] = pd.to_datetime(
        tr["local_date"], errors="coerce"
    ).dt.normalize()

    if tr[["participant_id", "local_date"]].duplicated().any():
        raise ValueError(
            "transition_strategy_profile has duplicate participant_id/local_date."
        )

    for c in transition_features:
        tr[c] = safe_numeric(tr[c]).astype("float32")

    return tr


def merge_transition(
    df: pd.DataFrame,
    transition_profile: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    out["participant_id"] = (
        out["participant_id"].astype("string").str.strip()
    )
    out["local_date"] = pd.to_datetime(
        out["local_date"], errors="coerce"
    ).dt.normalize()

    return out.merge(
        transition_profile,
        on=["participant_id", "local_date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )


def encode_categorical_column(s: pd.Series, col: str) -> pd.Series:
    x = s.astype("string").str.strip()

    if col in {
        "hist_lag1_template_id",
        "hist30_dominant_template_id",
    }:
        return x.map(TEMPLATE_TO_INT).astype("float32")

    if col == "hist_lag1_curve_mode":
        return x.str.lower().map(MODE_TO_INT).astype("float32")

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
    unknown = sorted(set(x.dropna().astype(str)) - set(TEMPLATE_TO_INT))
    if unknown:
        raise ValueError(f"Unknown template labels: {unknown}")

    y = x.map(TEMPLATE_TO_INT)
    if y.isna().any():
        raise ValueError("Template labels contain missing values.")
    return y.to_numpy(np.int16)


def encode_lag_template(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .map(TEMPLATE_TO_INT)
    )


def collect_training_samples(
    files: list[Path],
    all_features: list[str],
    transition_features: list[str],
    transition_profile: pd.DataFrame,
    train_end: pd.Timestamp,
    train_rows_hint: int,
    max_switch_rows: int,
    negative_to_positive: float,
    max_destination_rows: int,
    chunksize: int,
    seed: int,
):
    """
    Stream the training period.

    All positive switch rows are retained first.
    A deterministic random candidate sample of no-switch rows is retained.
    Exact final switch-training size/ratio is selected after the pass.
    """

    rng = np.random.default_rng(seed)

    tr_set = set(transition_features)
    dataset_features = [
        c for c in all_features
        if c not in tr_set
    ]

    usecols = unique_list(
        [
            "participant_id",
            "local_date",
            "prediction_ready_flag",
            TARGET,
            LAG1_TEMPLATE,
        ]
        + dataset_features
    )

    if max_switch_rows > 0:
        desired_neg = int(
            max_switch_rows
            * negative_to_positive
            / (1.0 + negative_to_positive)
        )
        neg_keep_prob = min(
            1.0,
            1.35
            * desired_neg
            / max(train_rows_hint, 1),
        )
    else:
        neg_keep_prob = 1.0

    positive_blocks = []
    negative_blocks = []

    total_rows = 0
    total_switch = 0

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(file, nrows=0).columns.tolist()

        missing = [
            c
            for c in dataset_features
            + [TARGET, LAG1_TEMPLATE, "participant_id", "local_date"]
            if c not in header
        ]
        if missing:
            raise KeyError(f"{file.name} missing columns: {missing}")

        cols = [c for c in usecols if c in header]

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
                chunk["local_date"], errors="coerce"
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
                unique_list(
                    [
                        "participant_id",
                        "local_date",
                        TARGET,
                        LAG1_TEMPLATE,
                    ]
                    + dataset_features
                ),
            ].copy()

            if sub.empty:
                continue

            sub = merge_transition(
                sub,
                transition_profile,
            )

            y_now = encode_template(
                sub[TARGET]
            )
            y_lag = (
                encode_lag_template(
                    sub[LAG1_TEMPLATE]
                )
                .to_numpy(np.int16)
            )

            y_switch = (
                y_now != y_lag
            ).astype(np.int8)

            sub["y_switch"] = y_switch

            total_rows += len(sub)
            total_switch += int(y_switch.sum())

            pos = sub.loc[
                y_switch == 1,
                unique_list(
                    [TARGET, LAG1_TEMPLATE, "y_switch"] + all_features
                ),
            ]

            if not pos.empty:
                positive_blocks.append(pos.copy())

            neg = sub.loc[
                y_switch == 0,
                unique_list(
                    [TARGET, LAG1_TEMPLATE, "y_switch"] + all_features
                ),
            ]

            if not neg.empty:
                if neg_keep_prob < 1.0:
                    keep = (
                        rng.random(len(neg))
                        < neg_keep_prob
                    )
                    neg = neg.loc[keep]

                if not neg.empty:
                    negative_blocks.append(
                        neg.copy()
                    )

    if not positive_blocks:
        raise ValueError("No switch rows in training period.")

    pos_all = pd.concat(
        positive_blocks,
        ignore_index=True,
    )

    neg_all = (
        pd.concat(
            negative_blocks,
            ignore_index=True,
        )
        if negative_blocks
        else pos_all.iloc[0:0].copy()
    )

    destination_train = pos_all.copy()

    if (
        max_destination_rows > 0
        and len(destination_train) > max_destination_rows
    ):
        destination_train = (
            destination_train.sample(
                n=max_destination_rows,
                random_state=seed,
            )
            .reset_index(drop=True)
        )

    if max_switch_rows <= 0:
        n_pos = len(pos_all)
        n_neg = len(neg_all)
    else:
        n_pos = min(
            len(pos_all),
            int(
                max_switch_rows
                / (1.0 + negative_to_positive)
            ),
        )
        n_neg = min(
            len(neg_all),
            max_switch_rows - n_pos,
        )

    pos_s = (
        pos_all
        if n_pos >= len(pos_all)
        else pos_all.sample(
            n=n_pos,
            random_state=seed,
        )
    )

    neg_s = (
        neg_all
        if n_neg >= len(neg_all)
        else neg_all.sample(
            n=n_neg,
            random_state=seed + 1,
        )
    )

    switch_train = (
        pd.concat(
            [pos_s, neg_s],
            ignore_index=True,
        )
        .sample(
            frac=1.0,
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    print(
        f"Training rows with lag1: {total_rows:,}",
        flush=True,
    )
    print(
        f"Observed train switches: {total_switch:,} "
        f"({total_switch / total_rows:.2%})",
        flush=True,
    )
    print(
        f"Switch detector sample: {len(switch_train):,}, "
        f"switch={int(switch_train['y_switch'].sum()):,} "
        f"({switch_train['y_switch'].mean():.2%})",
        flush=True,
    )
    print(
        f"Destination train rows: {len(destination_train):,}",
        flush=True,
    )

    return (
        switch_train,
        destination_train,
        {
            "rows": total_rows,
            "switch_rows": total_switch,
            "switch_rate": (
                total_switch / total_rows
                if total_rows else np.nan
            ),
        },
    )


def collect_eval_split(
    files: list[Path],
    all_features: list[str],
    transition_features: list[str],
    transition_profile: pd.DataFrame,
    split_name: str,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    chunksize: int,
) -> pd.DataFrame:
    tr_set = set(transition_features)
    dataset_features = [
        c for c in all_features
        if c not in tr_set
    ]

    usecols = unique_list(
        [
            "participant_id",
            "local_date",
            "prediction_ready_flag",
            TARGET,
            LAG1_TEMPLATE,
        ]
        + dataset_features
    )

    blocks = []

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(file, nrows=0).columns.tolist()
        cols = [c for c in usecols if c in header]

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

            mask = ready & sm & lag.notna()

            sub = chunk.loc[
                mask,
                unique_list(
                    [
                        "participant_id",
                        "local_date",
                        TARGET,
                        LAG1_TEMPLATE,
                    ]
                    + dataset_features
                ),
            ].copy()

            if sub.empty:
                continue

            sub = merge_transition(
                sub,
                transition_profile,
            )

            y_now = encode_template(
                sub[TARGET]
            )
            y_lag = (
                encode_lag_template(
                    sub[LAG1_TEMPLATE]
                )
                .to_numpy(np.int16)
            )

            sub["y_switch"] = (
                y_now != y_lag
            ).astype(np.int8)

            blocks.append(
                sub[
                    unique_list(
                        [TARGET, LAG1_TEMPLATE, "y_switch"]
                        + all_features
                    )
                ]
            )

    if not blocks:
        raise ValueError(f"No rows found for split={split_name}")

    out = pd.concat(
        blocks,
        ignore_index=True,
    )

    print(
        f"{split_name}: rows={len(out):,}, "
        f"switch={int(out['y_switch'].sum()):,} "
        f"({out['y_switch'].mean():.2%})",
        flush=True,
    )

    return out


def fit_imputer(X: pd.DataFrame) -> SimpleImputer:
    imp = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    imp.fit(X)
    return imp


def transform_X(X: pd.DataFrame, imputer: SimpleImputer) -> np.ndarray:
    return imputer.transform(X).astype(
        np.float32,
        copy=False,
    )


def fit_logistic(X, y, seed, max_iter, C):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(Xs, y)

    return {"model": model, "scaler": scaler}


def fit_tree(X, y, seed, max_depth, min_samples_leaf):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)
    return {"model": model, "scaler": None}


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
    )
    model.fit(X, y)
    return {"model": model, "scaler": None}


def sqrt_multiclass_weights(y):
    counts = np.bincount(
        y,
        minlength=len(TEMPLATE_ORDER),
    ).astype(float)

    cls = np.ones_like(counts)
    valid = counts > 0

    raw = np.sqrt(
        counts[valid].sum()
        / counts[valid]
    )
    norm = (
        np.sum(counts[valid] * raw)
        / counts[valid].sum()
    )
    cls[valid] = raw / norm
    return cls[y]


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
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

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
        "scale_pos_weight": (
            n_neg / n_pos
            if n_pos > 0 else 1.0
        ),
        "verbosity": -1,
        "seed": seed,
        "num_threads": 0,
    }

    model = lgb.train(
        params,
        ds,
        num_boost_round=rounds,
    )

    return {"model": model, "scaler": None}


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
    weight = sqrt_multiclass_weights(y)

    ds = lgb.Dataset(
        X,
        label=y,
        weight=weight,
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

    return {"model": model, "scaler": None}


def fit_four_models(
    X,
    y,
    features,
    binary,
    args,
):
    models = {}

    models["LogisticRegression"] = fit_logistic(
        X,
        y,
        args.seed,
        args.logit_max_iter,
        args.logit_c,
    )

    models["DecisionTree"] = fit_tree(
        X,
        y,
        args.seed,
        args.dt_max_depth,
        args.dt_min_samples_leaf,
    )

    models["RandomForest"] = fit_rf(
        X,
        y,
        args.seed,
        args.rf_trees,
        args.rf_max_depth,
        args.rf_min_samples_leaf,
    )

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


def predict_binary_score(algorithm, payload, X):
    model = payload["model"]

    if algorithm == "LogisticRegression":
        p = model.predict_proba(
            payload["scaler"].transform(X)
        )[:, 1]
    elif algorithm in {"DecisionTree", "RandomForest"}:
        p = model.predict_proba(X)[:, 1]
    elif algorithm == "LightGBM":
        p = model.predict(X)
    else:
        raise ValueError(algorithm)

    return np.asarray(p, dtype=np.float64)


def predict_multiclass_proba(algorithm, payload, X):
    model = payload["model"]

    if algorithm == "LogisticRegression":
        p = model.predict_proba(
            payload["scaler"].transform(X)
        )
    elif algorithm in {"DecisionTree", "RandomForest"}:
        p = model.predict_proba(X)
    elif algorithm == "LightGBM":
        p = model.predict(X)
    else:
        raise ValueError(algorithm)

    p = np.asarray(p, dtype=np.float64)

    if algorithm != "LightGBM":
        classes = np.asarray(model.classes_, dtype=int)
        if (
            p.shape[1] != len(TEMPLATE_ORDER)
            or not np.array_equal(
                classes,
                np.arange(len(TEMPLATE_ORDER)),
            )
        ):
            full = np.zeros(
                (len(X), len(TEMPLATE_ORDER)),
                dtype=np.float64,
            )
            full[:, classes] = p
            p = full

    return p


def force_destination_not_lag1(
    proba: np.ndarray,
    lag_class: np.ndarray,
):
    p = np.array(proba, copy=True)

    rows = np.arange(len(p))
    p[rows, lag_class] = 0.0

    denom = p.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= 0

    if bad.any():
        bad_rows = np.where(bad)[0]
        p[bad_rows, :] = 1.0
        p[bad_rows, lag_class[bad_rows]] = 0.0
        denom = p.sum(axis=1, keepdims=True)

    p /= denom

    pred = np.argmax(
        p,
        axis=1,
    ).astype(np.int16)

    return pred, p


def multiclass_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(TEMPLATE_ORDER)),
    )

    total = cm.sum()
    diag = np.diag(cm)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)

    recall = np.divide(
        diag,
        support,
        out=np.zeros_like(diag, dtype=float),
        where=support > 0,
    )
    precision = np.divide(
        diag,
        pred_count,
        out=np.zeros_like(diag, dtype=float),
        where=pred_count > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diag, dtype=float),
        where=(precision + recall) > 0,
    )

    valid = support > 0

    return {
        "accuracy": float(diag.sum() / total) if total else np.nan,
        "balanced_accuracy": float(recall[valid].mean()) if valid.any() else np.nan,
        "macro_f1": float(f1[valid].mean()) if valid.any() else np.nan,
        "weighted_f1": float(np.sum(f1 * support) / total) if total else np.nan,
        "cm": cm,
    }


def binary_metrics(y_true, score, threshold):
    pred = (score >= threshold).astype(np.int8)

    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
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
        "true_switch_rate": float(np.mean(y_true)),
        "predicted_switch_rate": float(np.mean(pred)),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(
            y_true,
            score,
        )
        out["pr_auc"] = average_precision_score(
            y_true,
            score,
        )
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    return out


def select_binary_threshold(
    y_true,
    score,
    thresholds,
):
    best = None

    for threshold in thresholds:
        m = binary_metrics(
            y_true,
            score,
            threshold,
        )

        key = (
            m["f1"],
            m["balanced_accuracy"],
            m["precision"],
        )

        if best is None or key > best[0]:
            best = (
                key,
                float(threshold),
                m,
            )

    return best[1], best[2]


def tune_hierarchy_threshold(
    y_template,
    lag_template,
    switch_score,
    destination_pred,
    thresholds,
):
    rows = []
    best = None

    for threshold in thresholds:
        switch_hat = (
            switch_score >= threshold
        )

        final_pred = lag_template.copy()
        final_pred[switch_hat] = (
            destination_pred[switch_hat]
        )

        m = multiclass_metrics(
            y_template,
            final_pred,
        )

        row = {
            "threshold": float(threshold),
            "hier_accuracy": m["accuracy"],
            "hier_balanced_accuracy": m["balanced_accuracy"],
            "hier_macro_f1": m["macro_f1"],
            "predicted_switch_rate": float(
                switch_hat.mean()
            ),
        }
        rows.append(row)

        key = (
            m["macro_f1"],
            m["balanced_accuracy"],
            m["accuracy"],
        )

        if best is None or key > best[0]:
            best = (key, row)

    return pd.DataFrame(rows), best[1]


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
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            imp = np.abs(coef)
        else:
            imp = np.mean(np.abs(coef), axis=0)
        kind = "mean_abs_standardized_coefficient"

    elif algorithm in {"DecisionTree", "RandomForest"}:
        imp = model.feature_importances_
        kind = "impurity_importance"

    elif algorithm == "LightGBM":
        imp = model.feature_importance(
            importance_type="gain"
        )
        kind = "gain"

    else:
        return

    df = pd.DataFrame(
        {
            "feature": features,
            "importance": imp,
            "importance_type": kind,
        }
    )

    total = df["importance"].sum()
    df["importance_share"] = (
        df["importance"] / total
        if total > 0 else 0.0
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


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--feature-set",
        choices=[
            "Z_base",
            "Z_base_tr",
            "Z_base_M_U",
            "Z_base_tr_M_U",
            "Z_base_H",
            "Z_base_tr_H",
            "Z_base_M_U_H",
            "Z_base_tr_M_U_H",
            "all",
        ],
        default="all",
    )

    p.add_argument(
        "--max-switch-train-rows",
        type=int,
        default=500_000,
    )
    p.add_argument(
        "--negative-to-positive",
        type=float,
        default=3.0,
    )
    p.add_argument(
        "--max-destination-train-rows",
        type=int,
        default=250_000,
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

    p.add_argument("--logit-max-iter", type=int, default=300)
    p.add_argument("--logit-c", type=float, default=1.0)

    p.add_argument("--dt-max-depth", type=int, default=18)
    p.add_argument("--dt-min-samples-leaf", type=int, default=80)

    p.add_argument("--rf-trees", type=int, default=120)
    p.add_argument("--rf-max-depth", type=int, default=20)
    p.add_argument("--rf-min-samples-leaf", type=int, default=40)

    p.add_argument("--lgb-rounds", type=int, default=300)
    p.add_argument("--lgb-num-leaves", type=int, default=63)
    p.add_argument("--lgb-learning-rate", type=float, default=0.06)
    p.add_argument("--lgb-min-data-in-leaf", type=int, default=120)

    args = p.parse_args()

    base = Path(args.root) / str(args.year)
    files = discover_parts(
        base / "dataset_parts"
    )

    prediction_schema = load_csv_schema(
        base
        / f"prediction_feature_schema_{args.year}.csv"
    )

    transition_schema = load_csv_schema(
        base
        / f"transition_strategy_feature_schema_{args.year}.csv"
    )

    (
        feature_sets,
        components,
        feature_to_group,
    ) = build_feature_sets(
        prediction_schema,
        transition_schema,
    )

    transition_features = components["Z_tr"]

    transition_profile = load_transition_profile(
        base
        / f"transition_strategy_profile_{args.year}.csv",
        transition_features,
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
        split_rows_hint,
    ) = load_split_dates(
        base
        / "validation"
        / "temporal_split_summary.csv"
    )

    out_dir = ensure_dir(
        base
        / "template_switch_model_transition"
    )

    component_rows = []
    for component, cols in components.items():
        for order, c in enumerate(cols):
            component_rows.append(
                {
                    "component": component,
                    "order": order,
                    "feature": c,
                    "feature_group": feature_to_group.get(c, ""),
                }
            )

    pd.DataFrame(component_rows).to_csv(
        out_dir
        / "input_components_Zbase_Ztr_M_U_H.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_rows = []
    for set_name, cols in feature_sets.items():
        for order, c in enumerate(cols):
            feature_rows.append(
                {
                    "feature_set": set_name,
                    "order": order,
                    "feature": c,
                    "feature_group": feature_to_group.get(c, ""),
                }
            )

    pd.DataFrame(feature_rows).to_csv(
        out_dir / "feature_sets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Hierarchical template switch model with transition strategy profile")
    print("=" * 80)
    print(f"Year:            {args.year}")
    print(f"Train <=         {train_end.date()}")
    print(
        f"Validation:      "
        f"{val_start.date()} .. {val_end.date()}"
    )
    print(f"Test >=          {test_start.date()}")
    print()
    print(f"Z_base features: {len(components['Z_base'])}")
    print(f"Z_tr features:   {len(components['Z_tr'])}")
    print(f"M features:      {len(components['M'])}")
    print(f"U features:      {len(components['U'])}")
    print(f"H features:      {len(components['H'])}")
    print(f"Feature sets:    {', '.join(selected_sets)}")
    print()

    all_features = unique_list(
        [
            c
            for set_name in selected_sets
            for c in feature_sets[set_name]
        ]
    )

    (
        switch_train,
        destination_train,
        train_switch_summary,
    ) = collect_training_samples(
        files=files,
        all_features=all_features,
        transition_features=transition_features,
        transition_profile=transition_profile,
        train_end=train_end,
        train_rows_hint=split_rows_hint["train"],
        max_switch_rows=args.max_switch_train_rows,
        negative_to_positive=args.negative_to_positive,
        max_destination_rows=args.max_destination_train_rows,
        chunksize=args.chunksize,
        seed=args.seed,
    )

    val_df = collect_eval_split(
        files=files,
        all_features=all_features,
        transition_features=transition_features,
        transition_profile=transition_profile,
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
        transition_features=transition_features,
        transition_profile=transition_profile,
        split_name="test",
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        chunksize=args.chunksize,
    )

    switch_rate_summary = pd.DataFrame(
        [
            {
                "split": "train",
                **train_switch_summary,
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
        ]
    )

    switch_rate_summary.to_csv(
        out_dir / "switch_rate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regular_thresholds = np.arange(
        args.threshold_min,
        args.threshold_max
        + 0.5 * args.threshold_step,
        args.threshold_step,
    )

    hierarchy_thresholds = np.concatenate(
        [
            regular_thresholds,
            np.array([1.01]),
        ]
    )

    switch_metric_rows = []
    destination_metric_rows = []
    hierarchy_metric_rows = []
    threshold_frames = []

    for set_no, set_name in enumerate(selected_sets, 1):
        features = feature_sets[set_name]

        print()
        print("=" * 80)
        print(
            f"[feature set {set_no}/{len(selected_sets)}] "
            f"{set_name}, features={len(features)}"
        )
        print("=" * 80)

        X_sw_df = prepare_X(
            switch_train,
            features,
        )

        imputer = fit_imputer(
            X_sw_df
        )

        X_sw = transform_X(
            X_sw_df,
            imputer,
        )

        y_sw_train = (
            switch_train["y_switch"]
            .to_numpy(np.int8)
        )

        X_dst = transform_X(
            prepare_X(
                destination_train,
                features,
            ),
            imputer,
        )

        y_dst = encode_template(
            destination_train[TARGET]
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
            val_df["y_switch"]
            .to_numpy(np.int8)
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
            test_df["y_switch"]
            .to_numpy(np.int8)
        )

        print("[stage 1] fit switch detector", flush=True)
        switch_models = fit_four_models(
            X_sw,
            y_sw_train,
            features,
            True,
            args,
        )

        print("[stage 2] fit destination classifier", flush=True)
        destination_models = fit_four_models(
            X_dst,
            y_dst,
            features,
            False,
            args,
        )

        joblib.dump(
            {
                "features": features,
                "imputer": imputer,
                "template_to_int": TEMPLATE_TO_INT,
            },
            out_dir
            / f"preprocess_{set_name}.joblib",
        )

        for algorithm in ALGORITHMS:
            print(
                f"[evaluate] {set_name} / {algorithm}",
                flush=True,
            )

            sw_model = switch_models[algorithm]
            dst_model = destination_models[algorithm]

            for stage, payload in [
                ("switch", sw_model),
                ("destination", dst_model),
            ]:
                if algorithm == "LightGBM":
                    payload["model"].save_model(
                        str(
                            out_dir
                            / f"model_{stage}_{set_name}_{algorithm}.txt"
                        )
                    )
                else:
                    joblib.dump(
                        payload,
                        out_dir
                        / f"model_{stage}_{set_name}_{algorithm}.joblib",
                    )

                save_importance(
                    out_dir,
                    stage,
                    set_name,
                    algorithm,
                    features,
                    payload,
                )

            sw_val_score = predict_binary_score(
                algorithm,
                sw_model,
                X_val,
            )

            sw_test_score = predict_binary_score(
                algorithm,
                sw_model,
                X_test,
            )

            dst_val_proba = predict_multiclass_proba(
                algorithm,
                dst_model,
                X_val,
            )

            dst_test_proba = predict_multiclass_proba(
                algorithm,
                dst_model,
                X_test,
            )

            (
                dst_val_pred,
                dst_val_forced,
            ) = force_destination_not_lag1(
                dst_val_proba,
                lag_val,
            )

            (
                dst_test_pred,
                dst_test_forced,
            ) = force_destination_not_lag1(
                dst_test_proba,
                lag_test,
            )

            # ---------------------------------------------------------
            # Binary switch threshold selected only on validation.
            # ---------------------------------------------------------
            (
                binary_threshold,
                binary_val_best,
            ) = select_binary_threshold(
                y_val_switch,
                sw_val_score,
                regular_thresholds,
            )

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
                    y_switch,
                    score,
                    binary_threshold,
                )

                switch_metric_rows.append(
                    {
                        "feature_set": set_name,
                        "algorithm": algorithm,
                        "split": split_name,
                        "binary_threshold": binary_threshold,
                        **bm,
                    }
                )

            # ---------------------------------------------------------
            # Destination metrics on TRUE switch rows only.
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
                    dst_val_forced,
                ),
                (
                    "test",
                    y_test_template,
                    y_test_switch,
                    dst_test_pred,
                    dst_test_forced,
                ),
            ]:
                mask = y_switch == 1
                yt = y_template[mask]
                yp = dst_pred[mask]
                pp = dst_proba[mask]

                mm = multiclass_metrics(
                    yt,
                    yp,
                )

                if len(yt):
                    eps = 1e-15
                    p_true = pp[
                        np.arange(len(yt)),
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
                            top2 == yt[:, None],
                            axis=1,
                        ).mean()
                    )
                else:
                    log_loss = np.nan
                    top2_acc = np.nan

                destination_metric_rows.append(
                    {
                        "feature_set": set_name,
                        "algorithm": algorithm,
                        "split": split_name,
                        "rows": int(mask.sum()),
                        "accuracy": mm["accuracy"],
                        "balanced_accuracy": mm["balanced_accuracy"],
                        "macro_f1": mm["macro_f1"],
                        "weighted_f1": mm["weighted_f1"],
                        "top2_accuracy": top2_acc,
                        "log_loss": log_loss,
                    }
                )

            # ---------------------------------------------------------
            # Final hierarchy threshold selected on validation.
            # Includes 1.01 = exact no-switch / lag1 baseline.
            # ---------------------------------------------------------
            (
                threshold_grid,
                best_hierarchy,
            ) = tune_hierarchy_threshold(
                y_val_template,
                lag_val,
                sw_val_score,
                dst_val_pred,
                hierarchy_thresholds,
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

            hierarchy_threshold = float(
                best_hierarchy["threshold"]
            )

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
                    >= hierarchy_threshold
                )

                final_pred = (
                    lag_template.copy()
                )
                final_pred[switch_hat] = (
                    destination_pred[switch_hat]
                )

                hm = multiclass_metrics(
                    y_template,
                    final_pred,
                )

                lag_m = multiclass_metrics(
                    y_template,
                    lag_template,
                )

                hierarchy_metric_rows.append(
                    {
                        "feature_set": set_name,
                        "algorithm": algorithm,
                        "split": split_name,
                        "hierarchy_threshold": hierarchy_threshold,
                        "rows": len(y_template),
                        "accuracy": hm["accuracy"],
                        "balanced_accuracy": hm["balanced_accuracy"],
                        "macro_f1": hm["macro_f1"],
                        "weighted_f1": hm["weighted_f1"],
                        "lag1_accuracy": lag_m["accuracy"],
                        "lag1_balanced_accuracy": lag_m["balanced_accuracy"],
                        "lag1_macro_f1": lag_m["macro_f1"],
                        "accuracy_gain_over_lag1": (
                            hm["accuracy"]
                            - lag_m["accuracy"]
                        ),
                        "balanced_accuracy_gain_over_lag1": (
                            hm["balanced_accuracy"]
                            - lag_m["balanced_accuracy"]
                        ),
                        "macro_f1_gain_over_lag1": (
                            hm["macro_f1"]
                            - lag_m["macro_f1"]
                        ),
                        "true_switch_rate": float(
                            np.mean(
                                y_template
                                != lag_template
                            )
                        ),
                        "predicted_switch_rate": float(
                            switch_hat.mean()
                        ),
                    }
                )

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
                f"  binary_threshold={binary_threshold:.2f}, "
                f"hierarchy_threshold={hierarchy_threshold:.2f}",
                flush=True,
            )

        del X_sw_df
        del X_sw
        del X_dst
        del X_val
        del X_test
        del switch_models
        del destination_models

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
            / "hierarchy_threshold_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------------------------------------------
    # Direct paired comparison: incremental value of Z_tr.
    # -------------------------------------------------------------
    pair_map = {
        "Z_base": "Z_base_tr",
        "Z_base_M_U": "Z_base_tr_M_U",
        "Z_base_H": "Z_base_tr_H",
        "Z_base_M_U_H": "Z_base_tr_M_U_H",
    }

    pair_rows = []

    for split_name in ["val", "test"]:
        for algorithm in ALGORITHMS:
            sw_sub = switch_df[
                (switch_df["split"] == split_name)
                & (switch_df["algorithm"] == algorithm)
            ].set_index("feature_set")

            dst_sub = destination_df[
                (destination_df["split"] == split_name)
                & (destination_df["algorithm"] == algorithm)
            ].set_index("feature_set")

            hier_sub = hierarchy_df[
                (hierarchy_df["split"] == split_name)
                & (hierarchy_df["algorithm"] == algorithm)
            ].set_index("feature_set")

            for base_set, tr_set in pair_map.items():
                if (
                    base_set not in sw_sub.index
                    or tr_set not in sw_sub.index
                    or base_set not in dst_sub.index
                    or tr_set not in dst_sub.index
                    or base_set not in hier_sub.index
                    or tr_set not in hier_sub.index
                ):
                    continue

                pair_rows.append(
                    {
                        "split": split_name,
                        "algorithm": algorithm,
                        "base_feature_set": base_set,
                        "transition_feature_set": tr_set,
                        "delta_switch_pr_auc_from_Ztr": (
                            sw_sub.loc[tr_set, "pr_auc"]
                            - sw_sub.loc[base_set, "pr_auc"]
                        ),
                        "delta_switch_f1_from_Ztr": (
                            sw_sub.loc[tr_set, "f1"]
                            - sw_sub.loc[base_set, "f1"]
                        ),
                        "delta_destination_accuracy_from_Ztr": (
                            dst_sub.loc[tr_set, "accuracy"]
                            - dst_sub.loc[base_set, "accuracy"]
                        ),
                        "delta_destination_macro_f1_from_Ztr": (
                            dst_sub.loc[tr_set, "macro_f1"]
                            - dst_sub.loc[base_set, "macro_f1"]
                        ),
                        "delta_hier_accuracy_from_Ztr": (
                            hier_sub.loc[tr_set, "accuracy"]
                            - hier_sub.loc[base_set, "accuracy"]
                        ),
                        "delta_hier_macro_f1_from_Ztr": (
                            hier_sub.loc[tr_set, "macro_f1"]
                            - hier_sub.loc[base_set, "macro_f1"]
                        ),
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(
        out_dir
        / "transition_profile_incremental_value.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        f"Hierarchical template switch prediction with Z_tr - {args.year}",
        "=" * 80,
        "",
        f"Train <= {train_end.date()}",
        f"Validation = {val_start.date()} .. {val_end.date()}",
        f"Test >= {test_start.date()}",
        "",
        f"Z_base = {len(components['Z_base'])}",
        f"Z_tr   = {len(components['Z_tr'])}",
        f"M      = {len(components['M'])}",
        f"U      = {len(components['U'])}",
        f"H      = {len(components['H'])}",
        "",
        "Observed switch rates:",
    ]

    for r in switch_rate_summary.itertuples():
        lines.append(
            f"  {r.split}: "
            f"{int(r.switch_rows):,}/{int(r.rows):,} "
            f"({r.switch_rate:.2%})"
        )

    lines.append("")

    for split_name in ["val", "test"]:
        lines.append(f"[{split_name}]")

        for set_name in selected_sets:
            lines.append(f"  {set_name}:")

            for algorithm in ALGORITHMS:
                s = switch_df[
                    (switch_df["split"] == split_name)
                    & (switch_df["feature_set"] == set_name)
                    & (switch_df["algorithm"] == algorithm)
                ]

                d = destination_df[
                    (destination_df["split"] == split_name)
                    & (destination_df["feature_set"] == set_name)
                    & (destination_df["algorithm"] == algorithm)
                ]

                h = hierarchy_df[
                    (hierarchy_df["split"] == split_name)
                    & (hierarchy_df["feature_set"] == set_name)
                    & (hierarchy_df["algorithm"] == algorithm)
                ]

                if s.empty or d.empty or h.empty:
                    continue

                s = s.iloc[0]
                d = d.iloc[0]
                h = h.iloc[0]

                lines.append(
                    f"    {algorithm}: "
                    f"SwitchPR-AUC={s['pr_auc']:.4f}, "
                    f"SwitchF1={s['f1']:.4f}, "
                    f"SwitchRecall={s['recall']:.4f}, "
                    f"binary_thr={s['binary_threshold']:.2f}; "
                    f"DestAcc={d['accuracy']:.4f}, "
                    f"DestMacroF1={d['macro_f1']:.4f}; "
                    f"HierAcc={h['accuracy']:.4f}, "
                    f"HierMacroF1={h['macro_f1']:.4f}, "
                    f"dAcc_vs_Lag1={h['accuracy_gain_over_lag1']:+.4f}, "
                    f"dMacroF1_vs_Lag1={h['macro_f1_gain_over_lag1']:+.4f}, "
                    f"hier_thr={h['hierarchy_threshold']:.2f}"
                )

            lines.append("")

        lines.append("  Increment from Z_tr:")
        if not pair_df.empty and "split" in pair_df.columns:
            psub = pair_df[
                pair_df["split"] == split_name
            ]

            for r in psub.itertuples():
                lines.append(
                    f"    {r.algorithm}/{r.base_feature_set}"
                    f" -> {r.transition_feature_set}: "
                    f"dPR-AUC={r.delta_switch_pr_auc_from_Ztr:+.4f}, "
                    f"dSwitchF1={r.delta_switch_f1_from_Ztr:+.4f}, "
                    f"dDestMacroF1={r.delta_destination_macro_f1_from_Ztr:+.4f}, "
                    f"dHierMacroF1={r.delta_hier_macro_f1_from_Ztr:+.4f}"
                )
        else:
            lines.append("    n/a (paired feature sets were not both selected)")

        lines.append("")

    lines += [
        "Interpretation rule:",
        "  1. First judge Z_tr by test dPR-AUC and dSwitchF1.",
        "  2. Then check destination Macro-F1.",
        "  3. Final hierarchy may rationally select threshold=1.01,",
        "     which means exact fallback to lag1 and is not treated as failure",
        "     of the threshold search itself.",
    ]

    summary = "\n".join(lines)

    (
        out_dir
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    config = {
        "year": args.year,
        "train_end": str(train_end.date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
        "test_start": str(test_start.date()),
        "components": components,
        "feature_sets": selected_sets,
        "algorithms": ALGORITHMS,
        "max_switch_train_rows": args.max_switch_train_rows,
        "negative_to_positive": args.negative_to_positive,
        "max_destination_train_rows": args.max_destination_train_rows,
        "binary_threshold_selection": "validation switch F1",
        "hierarchy_threshold_selection": (
            "validation final 13-class Macro-F1; "
            "threshold=1.01 is exact lag1 fallback"
        ),
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
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
