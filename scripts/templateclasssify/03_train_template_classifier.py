#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_train_template_classifier.py

Direct 13-class template-classification experiment after adding the
leakage-free transition strategy profile built by:

    scripts/bidprediction/02b_build_transition_features.py

Input decomposition
-------------------
Z_base : finalized 26-dimensional strategy profile
         = 9 LT + 9 ST + 8 Break

Z_tr   : 26-dimensional transition_strategy_profile
         built only from template history strictly before target day D

M      : market_environment
U      : unit_state_proxy
H      : participant_history

The original theoretical model remains:
    X = [Z, M, U]

The question tested here is whether Z should be extended from:
    Z_base
to:
    Z = [Z_base, Z_tr]

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

The pairs above use exactly the same target rows. Missing Z_tr values are not
used to remove rows; they are handled by TRAIN-only median imputation.

Models
------
Logistic Regression
Decision Tree
Random Forest
LightGBM

Baselines
---------
Majority
hist_lag1_template_id

Outputs
-------
data/processed/bidprediction/<year>/template_classifier_transition/
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
from sklearn.metrics import confusion_matrix
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


def unique_list(cols):
    out, seen = [], set()
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


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
    transition_features: list[str],
) -> pd.DataFrame:
    out = df.copy()

    out["participant_id"] = (
        out["participant_id"].astype("string").str.strip()
    )
    out["local_date"] = pd.to_datetime(
        out["local_date"], errors="coerce"
    ).dt.normalize()

    out = out.merge(
        transition_profile,
        on=["participant_id", "local_date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return out


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


def encode_target(s: pd.Series) -> np.ndarray:
    x = s.astype("string").str.strip()
    unknown = sorted(set(x.dropna().astype(str)) - set(TEMPLATE_TO_INT))
    if unknown:
        raise ValueError(f"Unknown template labels: {unknown}")

    y = x.map(TEMPLATE_TO_INT)
    if y.isna().any():
        raise ValueError("Target contains missing template labels.")
    return y.to_numpy(np.int16)


def collect_training_sample(
    files: list[Path],
    all_features: list[str],
    transition_features: list[str],
    transition_profile: pd.DataFrame,
    train_end: pd.Timestamp,
    max_train_rows: int | None,
    train_rows_hint: int,
    chunksize: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if max_train_rows is None or max_train_rows <= 0:
        keep_prob = 1.0
    else:
        keep_prob = min(
            1.0,
            1.10 * max_train_rows / max(train_rows_hint, 1),
        )

    tr_set = set(transition_features)
    dataset_features = [c for c in all_features if c not in tr_set]

    usecols = unique_list(
        [
            "participant_id",
            "local_date",
            "prediction_ready_flag",
            TARGET,
        ]
        + dataset_features
    )

    blocks = []
    seen_train_ready = 0
    joined_transition_rows = 0

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(file, nrows=0).columns.tolist()

        missing = [
            c for c in dataset_features + [TARGET, "participant_id", "local_date"]
            if c not in header
        ]
        if missing:
            raise KeyError(f"{file.name} missing columns: {missing}")

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
                chunk["local_date"], errors="coerce"
            ).dt.normalize()

            mask = ready & (d <= train_end)

            sub = chunk.loc[
                mask,
                unique_list(
                    ["participant_id", "local_date", TARGET] + dataset_features
                ),
            ].copy()

            seen_train_ready += len(sub)

            if sub.empty:
                continue

            if keep_prob < 1.0:
                take = rng.random(len(sub)) < keep_prob
                sub = sub.loc[take].copy()

            if sub.empty:
                continue

            sub = merge_transition(
                sub,
                transition_profile,
                transition_features,
            )

            if transition_features:
                joined_transition_rows += int(
                    sub[transition_features].notna().any(axis=1).sum()
                )

            blocks.append(
                sub[[TARGET] + all_features]
            )

    if not blocks:
        raise ValueError("No prediction-ready training rows collected.")

    train = pd.concat(blocks, ignore_index=True)

    if (
        max_train_rows is not None
        and max_train_rows > 0
        and len(train) > max_train_rows
    ):
        train = (
            train.sample(n=max_train_rows, random_state=seed)
            .reset_index(drop=True)
        )

    print(
        f"Shared train rows: available={seen_train_ready:,}, "
        f"sampled={len(train):,}",
        flush=True,
    )

    return train


def fit_imputer(X: pd.DataFrame) -> SimpleImputer:
    imp = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    imp.fit(X)
    return imp


def transform_X(X: pd.DataFrame, imputer: SimpleImputer) -> np.ndarray:
    return imputer.transform(X).astype(np.float32, copy=False)


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


def fit_lgb(
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


def predict_proba(algorithm, payload, X):
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


def metrics_from_confusion(cm: np.ndarray):
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
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_streaming(
    files: list[Path],
    feature_set: str,
    features: list[str],
    transition_features: list[str],
    transition_profile: pd.DataFrame,
    imputer: SimpleImputer,
    models: dict,
    split_name: str,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    chunksize: int,
):
    states = {
        a: {
            "cm": np.zeros(
                (len(TEMPLATE_ORDER), len(TEMPLATE_ORDER)),
                dtype=np.int64,
            ),
            "rows": 0,
            "top2_correct": 0,
            "logloss_sum": 0.0,
        }
        for a in models
    }

    lag_cm = np.zeros(
        (len(TEMPLATE_ORDER), len(TEMPLATE_ORDER)),
        dtype=np.int64,
    )
    lag_rows = 0

    tr_set = set(transition_features)
    dataset_features = [c for c in features if c not in tr_set]

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

    eps = 1e-15

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(file, nrows=0).columns.tolist()

        missing = [
            c
            for c in dataset_features + [TARGET, LAG1_TEMPLATE]
            if c not in header
        ]
        if missing:
            raise KeyError(f"{file.name} missing columns: {missing}")

        cols = [c for c in usecols if c in header]

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
                ready & sm,
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

            if d.empty:
                continue

            d = merge_transition(
                d,
                transition_profile,
                transition_features,
            )

            y = encode_target(d[TARGET])

            X = transform_X(
                prepare_X(d, features),
                imputer,
            )

            for algorithm, payload in models.items():
                p = predict_proba(
                    algorithm,
                    payload,
                    X,
                )

                pred = np.argmax(
                    p,
                    axis=1,
                ).astype(np.int16)

                st = states[algorithm]

                st["cm"] += confusion_matrix(
                    y,
                    pred,
                    labels=np.arange(len(TEMPLATE_ORDER)),
                )
                st["rows"] += len(y)

                p_true = p[
                    np.arange(len(y)),
                    y,
                ]
                st["logloss_sum"] += float(
                    -np.log(
                        np.clip(
                            p_true,
                            eps,
                            1.0,
                        )
                    ).sum()
                )

                top2 = np.argpartition(
                    p,
                    kth=-2,
                    axis=1,
                )[:, -2:]

                st["top2_correct"] += int(
                    np.any(
                        top2 == y[:, None],
                        axis=1,
                    ).sum()
                )

            lag = (
                d[LAG1_TEMPLATE]
                .astype("string")
                .str.strip()
                .map(TEMPLATE_TO_INT)
            )
            ok = lag.notna().to_numpy()

            if ok.any():
                lag_true = y[ok]
                lag_pred = lag.loc[ok].to_numpy(np.int16)

                lag_cm += confusion_matrix(
                    lag_true,
                    lag_pred,
                    labels=np.arange(len(TEMPLATE_ORDER)),
                )
                lag_rows += int(ok.sum())

    result = {}

    for algorithm, st in states.items():
        m = metrics_from_confusion(st["cm"])
        n = st["rows"]
        m["rows"] = n
        m["log_loss"] = (
            st["logloss_sum"] / n
            if n else np.nan
        )
        m["top2_accuracy"] = (
            st["top2_correct"] / n
            if n else np.nan
        )
        result[algorithm] = {
            "metrics": m,
            "cm": st["cm"],
        }

    lag_m = metrics_from_confusion(lag_cm)
    lag_m["rows"] = lag_rows

    return result, lag_m


def save_importance(
    out_dir,
    feature_set,
    algorithm,
    features,
    payload,
):
    model = payload["model"]

    if algorithm == "LogisticRegression":
        imp = np.mean(np.abs(model.coef_), axis=0)
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
        / f"feature_importance_{feature_set}_{algorithm}.csv",
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
    p.add_argument("--max-train-rows", type=int, default=600_000)
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

    p.add_argument("--logit-max-iter", type=int, default=300)
    p.add_argument("--logit-c", type=float, default=1.0)

    p.add_argument("--dt-max-depth", type=int, default=18)
    p.add_argument("--dt-min-samples-leaf", type=int, default=100)

    p.add_argument("--rf-trees", type=int, default=120)
    p.add_argument("--rf-max-depth", type=int, default=20)
    p.add_argument("--rf-min-samples-leaf", type=int, default=50)

    p.add_argument("--lgb-rounds", type=int, default=300)
    p.add_argument("--lgb-num-leaves", type=int, default=63)
    p.add_argument("--lgb-learning-rate", type=float, default=0.06)
    p.add_argument("--lgb-min-data-in-leaf", type=int, default=200)

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
        / "template_classifier_transition"
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
    print("Template classifier with Z_base + transition_strategy_profile")
    print("=" * 80)
    print(f"Year:              {args.year}")
    print(f"Train <=           {train_end.date()}")
    print(
        f"Validation:        "
        f"{val_start.date()} .. {val_end.date()}"
    )
    print(f"Test >=            {test_start.date()}")
    print()
    print(f"Z_base features:   {len(components['Z_base'])}")
    print(f"Z_tr features:     {len(components['Z_tr'])}")
    print(f"M features:        {len(components['M'])}")
    print(f"U features:        {len(components['U'])}")
    print(f"H features:        {len(components['H'])}")
    print(f"Feature sets:      {', '.join(selected_sets)}")
    print()

    all_features = unique_list(
        [
            c
            for set_name in selected_sets
            for c in feature_sets[set_name]
        ]
    )

    train_df = collect_training_sample(
        files=files,
        all_features=all_features,
        transition_features=transition_features,
        transition_profile=transition_profile,
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

    y_train = encode_target(
        train_df[TARGET]
    )

    train_counts = np.bincount(
        y_train,
        minlength=len(TEMPLATE_ORDER),
    )

    metric_rows = []
    baseline_rows = []
    training_rows = []

    for set_no, set_name in enumerate(selected_sets, 1):
        features = feature_sets[set_name]

        print()
        print("=" * 80)
        print(
            f"[feature set {set_no}/{len(selected_sets)}] "
            f"{set_name}, features={len(features)}"
        )
        print("=" * 80)

        X_train_df = prepare_X(
            train_df,
            features,
        )

        imputer = fit_imputer(
            X_train_df
        )

        X_train = transform_X(
            X_train_df,
            imputer,
        )

        models = {}

        print("[fit] LogisticRegression", flush=True)
        models["LogisticRegression"] = fit_logistic(
            X_train,
            y_train,
            args.seed,
            args.logit_max_iter,
            args.logit_c,
        )

        print("[fit] DecisionTree", flush=True)
        models["DecisionTree"] = fit_tree(
            X_train,
            y_train,
            args.seed,
            args.dt_max_depth,
            args.dt_min_samples_leaf,
        )

        print("[fit] RandomForest", flush=True)
        models["RandomForest"] = fit_rf(
            X_train,
            y_train,
            args.seed,
            args.rf_trees,
            args.rf_max_depth,
            args.rf_min_samples_leaf,
        )

        print("[fit] LightGBM", flush=True)
        models["LightGBM"] = fit_lgb(
            X_train,
            y_train,
            features,
            args.seed,
            args.lgb_rounds,
            args.lgb_num_leaves,
            args.lgb_learning_rate,
            args.lgb_min_data_in_leaf,
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

            save_importance(
                out_dir,
                set_name,
                algorithm,
                features,
                payload,
            )

            training_rows.append(
                {
                    "feature_set": set_name,
                    "algorithm": algorithm,
                    "feature_count": len(features),
                    "shared_train_sample_rows": len(train_df),
                }
            )

        for split_name in ["val", "test"]:
            result, lag_m = evaluate_streaming(
                files=files,
                feature_set=set_name,
                features=features,
                transition_features=transition_features,
                transition_profile=transition_profile,
                imputer=imputer,
                models=models,
                split_name=split_name,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                chunksize=args.chunksize,
            )

            ref_cm = result["LogisticRegression"]["cm"]
            majority = int(np.argmax(train_counts))
            support = ref_cm.sum(axis=1)

            majority_cm = np.zeros_like(ref_cm)
            for true_cls, n in enumerate(support):
                majority_cm[true_cls, majority] = n

            majority_m = metrics_from_confusion(
                majority_cm
            )

            if not any(
                r["split"] == split_name
                for r in baseline_rows
            ):
                baseline_rows += [
                    {
                        "baseline": LAG1_TEMPLATE,
                        "split": split_name,
                        "rows": lag_m["rows"],
                        "accuracy": lag_m["accuracy"],
                        "balanced_accuracy": lag_m["balanced_accuracy"],
                        "macro_f1": lag_m["macro_f1"],
                    },
                    {
                        "baseline": "majority",
                        "split": split_name,
                        "rows": int(support.sum()),
                        "accuracy": majority_m["accuracy"],
                        "balanced_accuracy": majority_m["balanced_accuracy"],
                        "macro_f1": majority_m["macro_f1"],
                    },
                ]

            for algorithm in ALGORITHMS:
                m = result[algorithm]["metrics"]

                metric_rows.append(
                    {
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
                        "lag1_accuracy": lag_m["accuracy"],
                        "lag1_balanced_accuracy": lag_m["balanced_accuracy"],
                        "lag1_macro_f1": lag_m["macro_f1"],
                        "accuracy_gain_over_lag1": (
                            m["accuracy"] - lag_m["accuracy"]
                        ),
                        "macro_f1_gain_over_lag1": (
                            m["macro_f1"] - lag_m["macro_f1"]
                        ),
                    }
                )

                pd.DataFrame(
                    result[algorithm]["cm"],
                    index=TEMPLATE_ORDER,
                    columns=TEMPLATE_ORDER,
                ).to_csv(
                    out_dir
                    / f"confusion_{set_name}_{algorithm}_{split_name}.csv",
                    encoding="utf-8-sig",
                )

                print(
                    f"  {split_name}/{algorithm}: "
                    f"Acc={m['accuracy']:.4f}, "
                    f"BalAcc={m['balanced_accuracy']:.4f}, "
                    f"MacroF1={m['macro_f1']:.4f}, "
                    f"Top2={m['top2_accuracy']:.4f}, "
                    f"LogLoss={m['log_loss']:.4f}",
                    flush=True,
                )

        del X_train_df
        del X_train
        del models

    metrics_df = pd.DataFrame(metric_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    metrics_df.to_csv(
        out_dir
        / "template_classifier_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_df.to_csv(
        out_dir
        / "baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(training_rows).to_csv(
        out_dir
        / "training_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_map = {
        "Z_base": "Z_base_tr",
        "Z_base_M_U": "Z_base_tr_M_U",
        "Z_base_H": "Z_base_tr_H",
        "Z_base_M_U_H": "Z_base_tr_M_U_H",
    }

    pair_rows = []

    for split_name in ["val", "test"]:
        for algorithm in ALGORITHMS:
            sub = metrics_df[
                (metrics_df["split"] == split_name)
                & (metrics_df["algorithm"] == algorithm)
            ].set_index("feature_set")

            for base_set, tr_set in pair_map.items():
                if base_set not in sub.index or tr_set not in sub.index:
                    continue

                a = sub.loc[base_set]
                b = sub.loc[tr_set]

                pair_rows.append(
                    {
                        "split": split_name,
                        "algorithm": algorithm,
                        "base_feature_set": base_set,
                        "transition_feature_set": tr_set,
                        "delta_accuracy_from_Ztr": (
                            b["accuracy"] - a["accuracy"]
                        ),
                        "delta_balanced_accuracy_from_Ztr": (
                            b["balanced_accuracy"]
                            - a["balanced_accuracy"]
                        ),
                        "delta_macro_f1_from_Ztr": (
                            b["macro_f1"] - a["macro_f1"]
                        ),
                        "delta_log_loss_from_Ztr": (
                            b["log_loss"] - a["log_loss"]
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
        f"Template classifier with transition strategy profile - {args.year}",
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
    ]

    for split_name in ["val", "test"]:
        lines.append(f"[{split_name}]")

        b = baseline_df[
            baseline_df["split"] == split_name
        ]

        for r in b.itertuples():
            lines.append(
                f"  Baseline/{r.baseline}: "
                f"Acc={r.accuracy:.4f}, "
                f"BalAcc={r.balanced_accuracy:.4f}, "
                f"MacroF1={r.macro_f1:.4f}"
            )

        for set_name in selected_sets:
            lines.append(f"  {set_name}:")
            sub = metrics_df[
                (metrics_df["split"] == split_name)
                & (metrics_df["feature_set"] == set_name)
            ]
            for r in sub.itertuples():
                lines.append(
                    f"    {r.algorithm}: "
                    f"Acc={r.accuracy:.4f}, "
                    f"BalAcc={r.balanced_accuracy:.4f}, "
                    f"MacroF1={r.macro_f1:.4f}, "
                    f"Top2={r.top2_accuracy:.4f}, "
                    f"LogLoss={r.log_loss:.4f}"
                )

        lines.append("")
        lines.append("  Increment from Z_tr:")
        if not pair_df.empty and "split" in pair_df.columns:
            subp = pair_df[
                pair_df["split"] == split_name
            ]
            for r in subp.itertuples():
                lines.append(
                    f"    {r.algorithm}/{r.base_feature_set}"
                    f" -> {r.transition_feature_set}: "
                    f"dAcc={r.delta_accuracy_from_Ztr:+.4f}, "
                    f"dBalAcc={r.delta_balanced_accuracy_from_Ztr:+.4f}, "
                    f"dMacroF1={r.delta_macro_f1_from_Ztr:+.4f}, "
                    f"dLogLoss={r.delta_log_loss_from_Ztr:+.4f}"
                )
        else:
            lines.append("    n/a (paired feature sets were not both selected)")

        lines.append("")

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
        "shared_train_rows": args.max_train_rows,
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
