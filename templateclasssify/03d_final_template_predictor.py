#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03d_final_template_predictor.py

Final template predictor lock-down.

This script DOES NOT search across model families or feature sets.
The components are fixed from the previous validation-stage experiments:

Switch detector
---------------
    RandomForest
    features = Z_base + Z_tr + H_full

Destination model
-----------------
    RandomForest GlobalOriginMasked
    features = Z_base + M + U + H_context + explicit origin code

where:
    Z_base    = finalized 9 LT + 9 ST + 8 Break = 26
    Z_tr      = transition_strategy_profile = 26
    H_full    = all participant_history features, including lag1 template
    H_context = participant_history excluding hist_lag1_template_id
    origin    = hist_lag1_template_id

Final decision
--------------
    default = lag1 template
    if P(switch) >= hierarchy_threshold:
        override with destination prediction

The hierarchy_threshold is selected ONLY on validation to maximize:
    1) final 13-class Macro-F1
    2) balanced accuracy
    3) accuracy

threshold=1.01 is included and means exact fallback to lag1.

Test is evaluated only after the threshold is fixed.

Outputs
-------
data/processed/bidprediction/<year>/final_template_predictor/

    final_template_metrics.csv
    switch_detector_metrics.csv
    destination_metrics.csv
    validation_threshold_search.csv
    transition_matrix_train_counts.csv
    transition_matrix_train_probabilities.csv
    confusion_final_val.csv
    confusion_final_test.csv
    confusion_lag1_val.csv
    confusion_lag1_test.csv
    feature_importance_switch.csv
    feature_importance_destination.csv
    final_template_model.joblib
    summary.txt
    config.json
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# Fixed project definitions
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
ORIGIN = "hist_lag1_template_id"

TEMPLATE_ORDER = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
TEMPLATE_TO_INT = {t: i for i, t in enumerate(TEMPLATE_ORDER)}
INT_TO_TEMPLATE = {i: t for t, i in TEMPLATE_TO_INT.items()}
N_TEMPLATE = len(TEMPLATE_ORDER)

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
    "tr_ready_flag",
}


# =============================================================================
# Helpers
# =============================================================================

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


def load_schema(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"column", "role", "feature_group"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"{path.name} missing schema columns: {sorted(missing)}"
        )

    return df


def load_split_dates(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun 02_validate_prediction_dataset.py first."
        )

    s = pd.read_csv(path).set_index("split")

    for name in ["train", "val", "test"]:
        if name not in s.index:
            raise ValueError(f"Missing temporal split: {name}")

    return (
        pd.Timestamp(s.loc["train", "last_date"]),
        pd.Timestamp(s.loc["val", "first_date"]),
        pd.Timestamp(s.loc["val", "last_date"]),
        pd.Timestamp(s.loc["test", "first_date"]),
    )


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
# Fixed feature components
# =============================================================================

def build_fixed_features(
    prediction_schema: pd.DataFrame,
    transition_schema: pd.DataFrame,
):
    p = prediction_schema[
        prediction_schema["role"].astype(str).eq("feature")
    ].copy()

    group = dict(
        zip(
            p["column"].astype(str),
            p["feature_group"].astype(str),
        )
    )

    available = set(group) - EXCLUDE_FEATURES

    z_base = LT_FEATURES + ST_FEATURES + BREAK_FEATURES

    missing_z = [
        c for c in z_base
        if c not in available
    ]

    if missing_z:
        raise KeyError(f"Missing Z_base columns: {missing_z}")

    m = [
        c for c, g in group.items()
        if g == "market_environment"
        and c in available
    ]

    u = [
        c for c, g in group.items()
        if g == "unit_state_proxy"
        and c in available
    ]

    h_full = [
        c for c, g in group.items()
        if g == "participant_history"
        and c in available
    ]

    if ORIGIN not in h_full:
        raise KeyError(
            f"{ORIGIN} must be present in participant_history."
        )

    h_context = [
        c for c in h_full
        if c != ORIGIN
    ]

    t = transition_schema[
        transition_schema["role"].astype(str).eq("feature")
    ].copy()

    z_tr = t["column"].astype(str).tolist()

    if not z_tr:
        raise ValueError(
            "No transition_strategy_profile features found."
        )

    z_base = unique_list(z_base)
    z_tr = unique_list(z_tr)
    m = unique_list(m)
    u = unique_list(u)
    h_full = unique_list(h_full)
    h_context = unique_list(h_context)

    # Fixed after 03b validation:
    # RF + Z_base_tr_H
    switch_features = unique_list(
        z_base + z_tr + h_full
    )

    # Fixed after 03c validation:
    # RF + Z_base_M_U_H_context + explicit origin code
    destination_features = unique_list(
        z_base + m + u + h_context
    )

    return {
        "Z_base": z_base,
        "Z_tr": z_tr,
        "M": m,
        "U": u,
        "H_full": h_full,
        "H_context": h_context,
        "switch_features": switch_features,
        "destination_features": destination_features,
    }


def load_transition_profile(
    path: Path,
    transition_features: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun 02b_build_transition_features.py first."
        )

    usecols = [
        "participant_id",
        "local_date",
    ] + transition_features

    tr = pd.read_csv(
        path,
        usecols=usecols,
        low_memory=False,
    )

    tr["participant_id"] = (
        tr["participant_id"]
        .astype("string")
        .str.strip()
    )

    tr["local_date"] = pd.to_datetime(
        tr["local_date"],
        errors="coerce",
    ).dt.normalize()

    if tr[
        ["participant_id", "local_date"]
    ].duplicated().any():
        raise ValueError(
            "transition_strategy_profile has duplicate "
            "participant_id/local_date."
        )

    for c in transition_features:
        tr[c] = safe_numeric(tr[c]).astype("float32")

    return tr


def merge_transition(
    df: pd.DataFrame,
    transition_profile: pd.DataFrame,
) -> pd.DataFrame:
    x = df.copy()

    x["participant_id"] = (
        x["participant_id"]
        .astype("string")
        .str.strip()
    )

    x["local_date"] = pd.to_datetime(
        x["local_date"],
        errors="coerce",
    ).dt.normalize()

    return x.merge(
        transition_profile,
        on=["participant_id", "local_date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )


# =============================================================================
# Encoding
# =============================================================================

def encode_template_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .map(TEMPLATE_TO_INT)
    )


def encode_target(s: pd.Series) -> np.ndarray:
    y = encode_template_series(s)

    if y.isna().any():
        bad = (
            s.loc[y.isna()]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Unknown/missing template labels: {bad[:20]}"
        )

    return y.to_numpy(np.int16)


def encode_categorical(
    s: pd.Series,
    col: str,
) -> pd.Series:
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
            out[c] = encode_categorical(
                df[c],
                c,
            )
        else:
            out[c] = safe_numeric(
                df[c]
            ).astype("float32")

    return pd.DataFrame(
        out,
        index=df.index,
    )


# =============================================================================
# Training data collection
# =============================================================================

def collect_training_data(
    files: list[Path],
    transition_profile: pd.DataFrame,
    transition_features: list[str],
    switch_features: list[str],
    destination_features: list[str],
    train_end: pd.Timestamp,
    chunksize: int,
    seed: int,
    max_switch_train_rows: int,
    negative_to_positive: float,
):
    """
    One streaming pass over TRAIN.

    Destination training:
        all TRUE switch rows.

    Switch detector training:
        all/capped switch positives + sampled no-switch negatives.
    """

    rng = np.random.default_rng(seed)

    all_features = unique_list(
        switch_features
        + destination_features
    )

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
            ORIGIN,
        ]
        + dataset_features
    )

    # Keep enough negative candidates without retaining the whole 2.8M train set.
    desired_neg = int(
        max_switch_train_rows
        * negative_to_positive
        / (1.0 + negative_to_positive)
    )

    # Conservative candidate probability.
    neg_keep_prob = 0.20

    positive_blocks = []
    negative_blocks = []

    train_rows = 0
    train_switch_rows = 0

    for file_no, file in enumerate(files, 1):
        header = pd.read_csv(file, nrows=0).columns.tolist()

        missing = [
            c
            for c in unique_list(
                [
                    "participant_id",
                    "local_date",
                    TARGET,
                    ORIGIN,
                ]
                + dataset_features
            )
            if c not in header
        ]

        if missing:
            raise KeyError(
                f"{file.name} missing columns: {missing}"
            )

        cols = [
            c for c in usecols
            if c in header
        ]

        print(
            f"[train {file_no}/{len(files)}] {file.name}",
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

            origin = encode_template_series(
                chunk[ORIGIN]
            )

            target = encode_template_series(
                chunk[TARGET]
            )

            mask = (
                ready
                & (d <= train_end)
                & origin.notna()
                & target.notna()
            )

            sub = chunk.loc[
                mask,
                unique_list(
                    [
                        "participant_id",
                        "local_date",
                        TARGET,
                        ORIGIN,
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

            y = encode_target(
                sub[TARGET]
            )

            lag = encode_target(
                sub[ORIGIN]
            )

            y_switch = (
                y != lag
            ).astype(np.int8)

            train_rows += len(sub)
            train_switch_rows += int(
                y_switch.sum()
            )

            sub["__y_switch__"] = y_switch

            pos = sub.loc[
                y_switch == 1,
                unique_list(
                    [
                        TARGET,
                        ORIGIN,
                        "__y_switch__",
                    ]
                    + all_features
                ),
            ]

            if not pos.empty:
                positive_blocks.append(
                    pos.copy()
                )

            neg = sub.loc[
                y_switch == 0,
                unique_list(
                    [
                        TARGET,
                        ORIGIN,
                        "__y_switch__",
                    ]
                    + switch_features
                ),
            ]

            if not neg.empty:
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
        raise ValueError(
            "No TRUE switch rows in training period."
        )

    positives = pd.concat(
        positive_blocks,
        ignore_index=True,
    )

    negatives = (
        pd.concat(
            negative_blocks,
            ignore_index=True,
        )
        if negative_blocks
        else positives.iloc[0:0][
            unique_list(
                [
                    TARGET,
                    ORIGIN,
                    "__y_switch__",
                ]
                + switch_features
            )
        ].copy()
    )

    # Destination uses all true switches.
    destination_train = positives[
        unique_list(
            [TARGET, ORIGIN]
            + destination_features
        )
    ].copy()

    # Reproduce the 03b 500k / 3:1 switch sample.
    n_pos = min(
        len(positives),
        int(
            max_switch_train_rows
            / (1.0 + negative_to_positive)
        ),
    )

    n_neg = min(
        len(negatives),
        max_switch_train_rows - n_pos,
    )

    if n_neg < (
        max_switch_train_rows - n_pos
    ):
        print(
            "WARNING: negative candidate sample smaller than requested; "
            "switch sample will contain fewer rows.",
            flush=True,
        )

    pos_sample = (
        positives
        if n_pos >= len(positives)
        else positives.sample(
            n=n_pos,
            random_state=seed,
        )
    )

    neg_sample = (
        negatives
        if n_neg >= len(negatives)
        else negatives.sample(
            n=n_neg,
            random_state=seed + 1,
        )
    )

    # Negative rows do not contain destination-only columns, which is fine.
    switch_train = pd.concat(
        [
            pos_sample[
                unique_list(
                    ["__y_switch__"]
                    + switch_features
                )
            ],
            neg_sample[
                unique_list(
                    ["__y_switch__"]
                    + switch_features
                )
            ],
        ],
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    summary = {
        "train_rows_with_lag1": train_rows,
        "train_switch_rows": train_switch_rows,
        "train_switch_rate": (
            train_switch_rows / train_rows
            if train_rows
            else np.nan
        ),
        "switch_train_sample_rows": len(
            switch_train
        ),
        "switch_train_sample_switch_rows": int(
            switch_train["__y_switch__"].sum()
        ),
        "destination_train_rows": len(
            destination_train
        ),
    }

    return (
        switch_train,
        destination_train,
        summary,
    )


# =============================================================================
# Fixed model fitting
# =============================================================================

def fit_switch_detector(
    switch_train: pd.DataFrame,
    switch_features: list[str],
    args,
):
    xdf = prepare_X(
        switch_train,
        switch_features,
    )

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X = imputer.fit_transform(
        xdf
    ).astype(
        np.float32,
        copy=False,
    )

    y = switch_train[
        "__y_switch__"
    ].to_numpy(
        np.int8
    )

    model = RandomForestClassifier(
        n_estimators=args.switch_rf_trees,
        max_depth=args.switch_rf_max_depth,
        min_samples_leaf=args.switch_rf_min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )

    model.fit(
        X,
        y,
    )

    return model, imputer


def fit_destination_model(
    destination_train: pd.DataFrame,
    destination_features: list[str],
    args,
):
    xdf = prepare_X(
        destination_train,
        destination_features,
    )

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X = imputer.fit_transform(
        xdf
    ).astype(
        np.float32,
        copy=False,
    )

    origin = encode_target(
        destination_train[ORIGIN]
    )

    y = encode_target(
        destination_train[TARGET]
    )

    # GlobalOriginMasked:
    # origin is explicit additional conditioning input.
    Xg = np.column_stack(
        [
            X,
            origin.astype(np.float32),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=args.destination_rf_trees,
        max_depth=args.destination_rf_max_depth,
        min_samples_leaf=args.destination_rf_min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )

    model.fit(
        Xg,
        y,
    )

    # TRAIN-only transition mask.
    counts = np.zeros(
        (N_TEMPLATE, N_TEMPLATE),
        dtype=np.int64,
    )

    np.add.at(
        counts,
        (origin, y),
        1,
    )

    np.fill_diagonal(
        counts,
        0,
    )

    row_sum = counts.sum(
        axis=1,
        keepdims=True,
    )

    probs = np.divide(
        counts,
        row_sum,
        out=np.zeros_like(
            counts,
            dtype=float,
        ),
        where=row_sum > 0,
    )

    allowed = counts > 0

    return (
        model,
        imputer,
        counts,
        probs,
        allowed,
    )


# =============================================================================
# Prediction and evaluation
# =============================================================================

def expand_proba(
    model: RandomForestClassifier,
    p: np.ndarray,
) -> np.ndarray:
    classes = np.asarray(
        model.classes_,
        dtype=np.int16,
    )

    full = np.zeros(
        (len(p), N_TEMPLATE),
        dtype=np.float64,
    )

    full[
        :,
        classes,
    ] = p

    return full


def mask_destination_proba(
    p: np.ndarray,
    origin: np.ndarray,
    allowed: np.ndarray,
):
    p = np.asarray(
        p,
        dtype=np.float64,
    ).copy()

    for i in range(N_TEMPLATE):
        rows = np.where(
            origin == i
        )[0]

        if len(rows) == 0:
            continue

        mask = allowed[
            i
        ].copy()

        if not mask.any():
            mask[:] = True
            mask[i] = False

        p[
            np.ix_(
                rows,
                ~mask,
            )
        ] = 0.0

    denom = p.sum(
        axis=1,
        keepdims=True,
    )

    bad = denom[
        :,
        0
    ] <= 0

    if bad.any():
        for r in np.where(
            bad
        )[0]:
            p[r, :] = 1.0
            p[r, origin[r]] = 0.0

        denom = p.sum(
            axis=1,
            keepdims=True,
        )

    p /= denom

    pred = np.argmax(
        p,
        axis=1,
    ).astype(
        np.int16
    )

    return pred, p


def predict_eval_split(
    files: list[Path],
    transition_profile: pd.DataFrame,
    transition_features: list[str],
    switch_features: list[str],
    destination_features: list[str],
    split_name: str,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    switch_model,
    switch_imputer,
    destination_model,
    destination_imputer,
    allowed_destinations,
    chunksize: int,
):
    all_features = unique_list(
        switch_features
        + destination_features
    )

    tr_set = set(
        transition_features
    )

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
            ORIGIN,
        ]
        + dataset_features
    )

    ys = []
    origins = []
    switch_scores = []
    destination_preds = []
    destination_probs_true_switch = []
    destination_true_labels = []
    destination_origins = []

    for file_no, file in enumerate(
        files,
        1,
    ):
        header = pd.read_csv(
            file,
            nrows=0,
        ).columns.tolist()

        cols = [
            c for c in usecols
            if c in header
        ]

        print(
            f"[eval {split_name} "
            f"{file_no}/{len(files)}] "
            f"{file.name}",
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

            origin_all = encode_template_series(
                chunk[ORIGIN]
            )

            target_all = encode_template_series(
                chunk[TARGET]
            )

            mask = (
                ready
                & sm
                & origin_all.notna()
                & target_all.notna()
            )

            sub = chunk.loc[
                mask,
                unique_list(
                    [
                        "participant_id",
                        "local_date",
                        TARGET,
                        ORIGIN,
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

            y = encode_target(
                sub[TARGET]
            )

            origin = encode_target(
                sub[ORIGIN]
            )

            # Switch detector.
            Xs = switch_imputer.transform(
                prepare_X(
                    sub,
                    switch_features,
                )
            ).astype(
                np.float32,
                copy=False,
            )

            sw_p = switch_model.predict_proba(
                Xs
            )[:, 1]

            # Destination model.
            Xd = destination_imputer.transform(
                prepare_X(
                    sub,
                    destination_features,
                )
            ).astype(
                np.float32,
                copy=False,
            )

            Xdg = np.column_stack(
                [
                    Xd,
                    origin.astype(
                        np.float32
                    ),
                ]
            )

            raw_p = destination_model.predict_proba(
                Xdg
            )

            full_p = expand_proba(
                destination_model,
                raw_p,
            )

            dest_pred, masked_p = mask_destination_proba(
                full_p,
                origin,
                allowed_destinations,
            )

            ys.append(
                y
            )
            origins.append(
                origin
            )
            switch_scores.append(
                sw_p
            )
            destination_preds.append(
                dest_pred
            )

            true_switch = (
                y != origin
            )

            if true_switch.any():
                destination_true_labels.append(
                    y[
                        true_switch
                    ]
                )
                destination_origins.append(
                    origin[
                        true_switch
                    ]
                )
                destination_probs_true_switch.append(
                    masked_p[
                        true_switch
                    ]
                )

    if not ys:
        raise ValueError(
            f"No evaluation rows for split={split_name}"
        )

    y = np.concatenate(
        ys
    )

    origin = np.concatenate(
        origins
    )

    sw_score = np.concatenate(
        switch_scores
    )

    dest_pred = np.concatenate(
        destination_preds
    )

    if destination_true_labels:
        dest_y = np.concatenate(
            destination_true_labels
        )
        dest_origin = np.concatenate(
            destination_origins
        )
        dest_p = np.concatenate(
            destination_probs_true_switch
        )
    else:
        dest_y = np.empty(
            0,
            dtype=np.int16,
        )
        dest_origin = np.empty(
            0,
            dtype=np.int16,
        )
        dest_p = np.empty(
            (0, N_TEMPLATE),
            dtype=float,
        )

    return {
        "y": y,
        "origin": origin,
        "switch_score": sw_score,
        "destination_pred": dest_pred,
        "destination_true_y": dest_y,
        "destination_true_origin": dest_origin,
        "destination_true_proba": dest_p,
    }


# =============================================================================
# Metrics
# =============================================================================

def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            N_TEMPLATE
        ),
    )

    total = cm.sum()
    diag = np.diag(
        cm
    )
    support = cm.sum(
        axis=1
    )
    predicted = cm.sum(
        axis=0
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
        predicted,
        out=np.zeros_like(
            diag,
            dtype=float,
        ),
        where=predicted > 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(
            recall,
            dtype=float,
        ),
        where=(
            precision + recall
        ) > 0,
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
                recall[
                    valid
                ].mean()
            )
            if valid.any()
            else np.nan
        ),
        "macro_f1": (
            float(
                f1[
                    valid
                ].mean()
            )
            if valid.any()
            else np.nan
        ),
        "weighted_f1": (
            float(
                np.sum(
                    f1
                    * support
                )
                / total
            )
            if total
            else np.nan
        ),
        "cm": cm,
    }


def switch_metrics(
    y_switch: np.ndarray,
    score: np.ndarray,
    threshold: float,
):
    pred = (
        score >= threshold
    ).astype(
        np.int8
    )

    return {
        "threshold": float(
            threshold
        ),
        "precision": precision_score(
            y_switch,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_switch,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_switch,
            pred,
            zero_division=0,
        ),
        "pr_auc": (
            average_precision_score(
                y_switch,
                score,
            )
            if len(
                np.unique(
                    y_switch
                )
            ) == 2
            else np.nan
        ),
        "roc_auc": (
            roc_auc_score(
                y_switch,
                score,
            )
            if len(
                np.unique(
                    y_switch
                )
            ) == 2
            else np.nan
        ),
        "true_switch_rate": float(
            y_switch.mean()
        ),
        "predicted_switch_rate": float(
            pred.mean()
        ),
    }


def choose_binary_threshold(
    y_switch: np.ndarray,
    score: np.ndarray,
    thresholds: np.ndarray,
):
    best = None

    for t in thresholds:
        m = switch_metrics(
            y_switch,
            score,
            float(t),
        )

        key = (
            m["f1"],
            m["recall"],
            m["precision"],
        )

        if (
            best is None
            or key > best[0]
        ):
            best = (
                key,
                float(t),
                m,
            )

    return best[1], best[2]


def destination_metrics(
    true_y: np.ndarray,
    p: np.ndarray,
):
    if len(
        true_y
    ) == 0:
        return {
            "rows": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "top2_accuracy": np.nan,
            "log_loss": np.nan,
        }

    pred = np.argmax(
        p,
        axis=1,
    ).astype(
        np.int16
    )

    m = multiclass_metrics(
        true_y,
        pred,
    )

    eps = 1e-15

    p_true = p[
        np.arange(
            len(
                true_y
            )
        ),
        true_y,
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
        p,
        kth=-2,
        axis=1,
    )[:, -2:]

    top2_accuracy = float(
        np.any(
            top2
            == true_y[:, None],
            axis=1,
        ).mean()
    )

    return {
        "rows": len(
            true_y
        ),
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "top2_accuracy": top2_accuracy,
        "log_loss": log_loss,
    }


def hierarchy_at_threshold(
    data: dict,
    threshold: float,
):
    y = data[
        "y"
    ]

    origin = data[
        "origin"
    ]

    sw_score = data[
        "switch_score"
    ]

    dest_pred = data[
        "destination_pred"
    ]

    override = (
        sw_score >= threshold
    )

    final_pred = origin.copy()

    final_pred[
        override
    ] = dest_pred[
        override
    ]

    final_m = multiclass_metrics(
        y,
        final_pred,
    )

    lag_m = multiclass_metrics(
        y,
        origin,
    )

    true_switch = (
        y != origin
    )

    fixed = (
        override
        & true_switch
        & (
            dest_pred == y
        )
    )

    damaged = (
        override
        & (~true_switch)
    )

    overrides_on_true_switch = (
        override
        & true_switch
    )

    correct_destination_given_override_switch = (
        fixed.sum()
        / overrides_on_true_switch.sum()
        if overrides_on_true_switch.sum()
        else np.nan
    )

    return {
        "threshold": float(
            threshold
        ),
        "rows": len(
            y
        ),
        "accuracy": final_m["accuracy"],
        "balanced_accuracy": final_m["balanced_accuracy"],
        "macro_f1": final_m["macro_f1"],
        "weighted_f1": final_m["weighted_f1"],
        "lag1_accuracy": lag_m["accuracy"],
        "lag1_balanced_accuracy": lag_m["balanced_accuracy"],
        "lag1_macro_f1": lag_m["macro_f1"],
        "accuracy_gain_over_lag1": (
            final_m["accuracy"]
            - lag_m["accuracy"]
        ),
        "balanced_accuracy_gain_over_lag1": (
            final_m["balanced_accuracy"]
            - lag_m["balanced_accuracy"]
        ),
        "macro_f1_gain_over_lag1": (
            final_m["macro_f1"]
            - lag_m["macro_f1"]
        ),
        "true_switch_rate": float(
            true_switch.mean()
        ),
        "predicted_override_rate": float(
            override.mean()
        ),
        "override_rows": int(
            override.sum()
        ),
        "corrected_switch_rows": int(
            fixed.sum()
        ),
        "damaged_persistence_rows": int(
            damaged.sum()
        ),
        "correct_destination_rate_given_override_true_switch": (
            correct_destination_given_override_switch
        ),
        "final_cm": final_m["cm"],
        "lag1_cm": lag_m["cm"],
    }


def choose_hierarchy_threshold(
    val_data: dict,
    thresholds: np.ndarray,
):
    rows = []
    best = None

    for t in thresholds:
        m = hierarchy_at_threshold(
            val_data,
            float(t),
        )

        rows.append(
            {
                k: v
                for k, v in m.items()
                if k not in {
                    "final_cm",
                    "lag1_cm",
                }
            }
        )

        key = (
            m["macro_f1"],
            m["balanced_accuracy"],
            m["accuracy"],
        )

        if (
            best is None
            or key > best[0]
        ):
            best = (
                key,
                float(t),
                m,
            )

    return (
        pd.DataFrame(
            rows
        ),
        best[1],
        best[2],
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Match 03b fixed switch detector.
    parser.add_argument(
        "--max-switch-train-rows",
        type=int,
        default=500_000,
    )

    parser.add_argument(
        "--negative-to-positive",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--switch-rf-trees",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--switch-rf-max-depth",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--switch-rf-min-samples-leaf",
        type=int,
        default=40,
    )

    # Match 03c validation-selected destination RF.
    parser.add_argument(
        "--destination-rf-trees",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--destination-rf-max-depth",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--destination-rf-min-samples-leaf",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    base = (
        Path(
            args.root
        )
        / str(
            args.year
        )
    )

    files = discover_parts(
        base
        / "dataset_parts"
    )

    pred_schema = load_schema(
        base
        / f"prediction_feature_schema_{args.year}.csv"
    )

    tr_schema = load_schema(
        base
        / f"transition_strategy_feature_schema_{args.year}.csv"
    )

    features = build_fixed_features(
        pred_schema,
        tr_schema,
    )

    tr_profile = load_transition_profile(
        base
        / f"transition_strategy_profile_{args.year}.csv",
        features[
            "Z_tr"
        ],
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
        / "final_template_predictor"
    )

    print("=" * 80)
    print("Final template predictor")
    print("=" * 80)
    print(f"Train <=      {train_end.date()}")
    print(
        f"Validation = "
        f"{val_start.date()} .. {val_end.date()}"
    )
    print(f"Test >=       {test_start.date()}")
    print()
    print(
        "Switch detector: "
        "RandomForest + Z_base_tr_H"
    )
    print(
        f"  features = "
        f"{len(features['switch_features'])}"
    )
    print(
        "Destination: "
        "RandomForest + Z_base_M_U_H + explicit origin + TRAIN mask"
    )
    print(
        f"  base features = "
        f"{len(features['destination_features'])}"
    )
    print()

    # -----------------------------------------------------------------
    # Train fixed components.
    # -----------------------------------------------------------------

    (
        switch_train,
        destination_train,
        train_summary,
    ) = collect_training_data(
        files=files,
        transition_profile=tr_profile,
        transition_features=features["Z_tr"],
        switch_features=features["switch_features"],
        destination_features=features["destination_features"],
        train_end=train_end,
        chunksize=args.chunksize,
        seed=args.seed,
        max_switch_train_rows=args.max_switch_train_rows,
        negative_to_positive=args.negative_to_positive,
    )

    print()
    print(
        f"Train rows with lag1: "
        f"{train_summary['train_rows_with_lag1']:,}"
    )
    print(
        f"Observed train switches: "
        f"{train_summary['train_switch_rows']:,} "
        f"({train_summary['train_switch_rate']:.2%})"
    )
    print(
        f"Switch detector sample: "
        f"{train_summary['switch_train_sample_rows']:,}"
    )
    print(
        f"Destination train rows: "
        f"{train_summary['destination_train_rows']:,}"
    )
    print()

    print(
        "[fit] switch detector "
        "RandomForest + Z_base_tr_H",
        flush=True,
    )

    (
        switch_model,
        switch_imputer,
    ) = fit_switch_detector(
        switch_train,
        features["switch_features"],
        args,
    )

    print(
        "[fit] destination "
        "RandomForest + Z_base_M_U_H + origin mask",
        flush=True,
    )

    (
        destination_model,
        destination_imputer,
        transition_counts,
        transition_probs,
        allowed_destinations,
    ) = fit_destination_model(
        destination_train,
        features["destination_features"],
        args,
    )

    # Save TRAIN-only transition structure.
    counts_df = pd.DataFrame(
        transition_counts,
        index=TEMPLATE_ORDER,
        columns=TEMPLATE_ORDER,
    )
    counts_df.index.name = "origin"
    counts_df.to_csv(
        out_dir
        / "transition_matrix_train_counts.csv",
        encoding="utf-8-sig",
    )

    probs_df = pd.DataFrame(
        transition_probs,
        index=TEMPLATE_ORDER,
        columns=TEMPLATE_ORDER,
    )
    probs_df.index.name = "origin"
    probs_df.to_csv(
        out_dir
        / "transition_matrix_train_probabilities.csv",
        encoding="utf-8-sig",
    )

    # Feature importance.
    sw_imp = pd.DataFrame(
        {
            "feature": features[
                "switch_features"
            ],
            "importance": switch_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )
    sw_imp[
        "importance_share"
    ] = (
        sw_imp["importance"]
        / sw_imp["importance"].sum()
    )
    sw_imp.to_csv(
        out_dir
        / "feature_importance_switch.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dest_feature_names = (
        features[
            "destination_features"
        ]
        + [
            "__origin_template_code__"
        ]
    )

    dest_imp = pd.DataFrame(
        {
            "feature": dest_feature_names,
            "importance": destination_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )
    dest_imp[
        "importance_share"
    ] = (
        dest_imp["importance"]
        / dest_imp["importance"].sum()
    )
    dest_imp.to_csv(
        out_dir
        / "feature_importance_destination.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Validation and test inference.
    # -----------------------------------------------------------------

    print()
    print(
        "[predict] validation",
        flush=True,
    )

    val_data = predict_eval_split(
        files=files,
        transition_profile=tr_profile,
        transition_features=features["Z_tr"],
        switch_features=features["switch_features"],
        destination_features=features["destination_features"],
        split_name="val",
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        switch_model=switch_model,
        switch_imputer=switch_imputer,
        destination_model=destination_model,
        destination_imputer=destination_imputer,
        allowed_destinations=allowed_destinations,
        chunksize=args.chunksize,
    )

    print()
    print(
        "[predict] test",
        flush=True,
    )

    test_data = predict_eval_split(
        files=files,
        transition_profile=tr_profile,
        transition_features=features["Z_tr"],
        switch_features=features["switch_features"],
        destination_features=features["destination_features"],
        split_name="test",
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        switch_model=switch_model,
        switch_imputer=switch_imputer,
        destination_model=destination_model,
        destination_imputer=destination_imputer,
        allowed_destinations=allowed_destinations,
        chunksize=args.chunksize,
    )

    # -----------------------------------------------------------------
    # Validation-only threshold selection.
    # -----------------------------------------------------------------

    regular_thresholds = np.arange(
        args.threshold_min,
        args.threshold_max
        + 0.5
        * args.threshold_step,
        args.threshold_step,
    )

    hierarchy_thresholds = np.concatenate(
        [
            regular_thresholds,
            np.array(
                [1.01],
                dtype=float,
            ),
        ]
    )

    y_val_switch = (
        val_data["y"]
        != val_data["origin"]
    ).astype(
        np.int8
    )

    binary_threshold, val_switch_diag = choose_binary_threshold(
        y_val_switch,
        val_data["switch_score"],
        regular_thresholds,
    )

    (
        threshold_df,
        final_threshold,
        val_final,
    ) = choose_hierarchy_threshold(
        val_data,
        hierarchy_thresholds,
    )

    threshold_df.to_csv(
        out_dir
        / "validation_threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Fixed-threshold test evaluation.
    # -----------------------------------------------------------------

    test_final = hierarchy_at_threshold(
        test_data,
        final_threshold,
    )

    y_test_switch = (
        test_data["y"]
        != test_data["origin"]
    ).astype(
        np.int8
    )

    val_switch = switch_metrics(
        y_val_switch,
        val_data["switch_score"],
        binary_threshold,
    )

    test_switch = switch_metrics(
        y_test_switch,
        test_data["switch_score"],
        binary_threshold,
    )

    val_dest = destination_metrics(
        val_data[
            "destination_true_y"
        ],
        val_data[
            "destination_true_proba"
        ],
    )

    test_dest = destination_metrics(
        test_data[
            "destination_true_y"
        ],
        test_data[
            "destination_true_proba"
        ],
    )

    # -----------------------------------------------------------------
    # Save metrics.
    # -----------------------------------------------------------------

    final_metric_rows = []

    for split_name, m in [
        (
            "val",
            val_final,
        ),
        (
            "test",
            test_final,
        ),
    ]:
        final_metric_rows.append(
            {
                k: v
                for k, v in {
                    "split": split_name,
                    **m,
                }.items()
                if k not in {
                    "final_cm",
                    "lag1_cm",
                }
            }
        )

    pd.DataFrame(
        final_metric_rows
    ).to_csv(
        out_dir
        / "final_template_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    switch_rows = []

    for split_name, m in [
        (
            "val",
            val_switch,
        ),
        (
            "test",
            test_switch,
        ),
    ]:
        switch_rows.append(
            {
                "split": split_name,
                **m,
            }
        )

    pd.DataFrame(
        switch_rows
    ).to_csv(
        out_dir
        / "switch_detector_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    destination_rows = []

    for split_name, m in [
        (
            "val",
            val_dest,
        ),
        (
            "test",
            test_dest,
        ),
    ]:
        destination_rows.append(
            {
                "split": split_name,
                **m,
            }
        )

    pd.DataFrame(
        destination_rows
    ).to_csv(
        out_dir
        / "destination_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Confusions.
    pd.DataFrame(
        val_final["final_cm"],
        index=TEMPLATE_ORDER,
        columns=TEMPLATE_ORDER,
    ).to_csv(
        out_dir
        / "confusion_final_val.csv",
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        test_final["final_cm"],
        index=TEMPLATE_ORDER,
        columns=TEMPLATE_ORDER,
    ).to_csv(
        out_dir
        / "confusion_final_test.csv",
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        val_final["lag1_cm"],
        index=TEMPLATE_ORDER,
        columns=TEMPLATE_ORDER,
    ).to_csv(
        out_dir
        / "confusion_lag1_val.csv",
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        test_final["lag1_cm"],
        index=TEMPLATE_ORDER,
        columns=TEMPLATE_ORDER,
    ).to_csv(
        out_dir
        / "confusion_lag1_test.csv",
        encoding="utf-8-sig",
    )

    # Save deployable final bundle.
    joblib.dump(
        {
            "year": args.year,
            "template_order": TEMPLATE_ORDER,
            "switch_model_type": "RandomForest",
            "switch_feature_set": "Z_base_tr_H",
            "switch_features": features[
                "switch_features"
            ],
            "switch_imputer": switch_imputer,
            "switch_model": switch_model,
            "destination_model_type": "RandomForest",
            "destination_feature_set": "Z_base_M_U_H_context_plus_origin",
            "destination_features": features[
                "destination_features"
            ],
            "destination_imputer": destination_imputer,
            "destination_model": destination_model,
            "allowed_destinations_train": allowed_destinations,
            "transition_counts_train": transition_counts,
            "hierarchy_threshold": final_threshold,
            "binary_diagnostic_threshold": binary_threshold,
            "decision_rule": (
                "default lag1; override with destination prediction "
                "when P(switch) >= hierarchy_threshold"
            ),
        },
        out_dir
        / "final_template_model.joblib",
    )

    # -----------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------

    summary_lines = [
        f"Final template predictor - {args.year}",
        "=" * 80,
        "",
        f"Train <= {train_end.date()}",
        f"Validation = {val_start.date()} .. {val_end.date()}",
        f"Test >= {test_start.date()}",
        "",
        "Fixed components selected before final TEST:",
        "  Switch detector:",
        "    RandomForest + Z_base_tr_H",
        f"    feature count = {len(features['switch_features'])}",
        "  Destination:",
        "    RandomForest + Z_base_M_U_H_context + explicit origin",
        "    TRAIN-only origin->destination candidate mask",
        f"    base feature count = {len(features['destination_features'])}",
        "",
        "Training:",
        (
            f"  train rows with lag1 = "
            f"{train_summary['train_rows_with_lag1']:,}"
        ),
        (
            f"  observed train switches = "
            f"{train_summary['train_switch_rows']:,} "
            f"({train_summary['train_switch_rate']:.2%})"
        ),
        (
            f"  switch detector sample = "
            f"{train_summary['switch_train_sample_rows']:,}"
        ),
        (
            f"  destination train rows = "
            f"{train_summary['destination_train_rows']:,}"
        ),
        "",
        "Validation-only thresholds:",
        (
            f"  switch diagnostic threshold "
            f"(best switch F1) = {binary_threshold:.2f}"
        ),
        (
            f"  FINAL hierarchy threshold "
            f"(best end-to-end MacroF1) = {final_threshold:.2f}"
        ),
        "",
        "[val]",
        (
            f"  Switch: PR-AUC={val_switch['pr_auc']:.4f}, "
            f"F1={val_switch['f1']:.4f}, "
            f"Recall={val_switch['recall']:.4f}, "
            f"Precision={val_switch['precision']:.4f}"
        ),
        (
            f"  Destination on true switches: "
            f"Acc={val_dest['accuracy']:.4f}, "
            f"BalAcc={val_dest['balanced_accuracy']:.4f}, "
            f"MacroF1={val_dest['macro_f1']:.4f}, "
            f"Top2={val_dest['top2_accuracy']:.4f}"
        ),
        (
            f"  Lag1: Acc={val_final['lag1_accuracy']:.4f}, "
            f"BalAcc={val_final['lag1_balanced_accuracy']:.4f}, "
            f"MacroF1={val_final['lag1_macro_f1']:.4f}"
        ),
        (
            f"  Final: Acc={val_final['accuracy']:.4f}, "
            f"BalAcc={val_final['balanced_accuracy']:.4f}, "
            f"MacroF1={val_final['macro_f1']:.4f}"
        ),
        (
            f"  Gain: dAcc={val_final['accuracy_gain_over_lag1']:+.4f}, "
            f"dBalAcc={val_final['balanced_accuracy_gain_over_lag1']:+.4f}, "
            f"dMacroF1={val_final['macro_f1_gain_over_lag1']:+.4f}"
        ),
        (
            f"  Override rate={val_final['predicted_override_rate']:.4%}, "
            f"corrected switches={val_final['corrected_switch_rows']:,}, "
            f"damaged persistence={val_final['damaged_persistence_rows']:,}"
        ),
        "",
        "[test]  (threshold fixed from validation)",
        (
            f"  Switch: PR-AUC={test_switch['pr_auc']:.4f}, "
            f"F1={test_switch['f1']:.4f}, "
            f"Recall={test_switch['recall']:.4f}, "
            f"Precision={test_switch['precision']:.4f}"
        ),
        (
            f"  Destination on true switches: "
            f"Acc={test_dest['accuracy']:.4f}, "
            f"BalAcc={test_dest['balanced_accuracy']:.4f}, "
            f"MacroF1={test_dest['macro_f1']:.4f}, "
            f"Top2={test_dest['top2_accuracy']:.4f}"
        ),
        (
            f"  Lag1: Acc={test_final['lag1_accuracy']:.4f}, "
            f"BalAcc={test_final['lag1_balanced_accuracy']:.4f}, "
            f"MacroF1={test_final['lag1_macro_f1']:.4f}"
        ),
        (
            f"  Final: Acc={test_final['accuracy']:.4f}, "
            f"BalAcc={test_final['balanced_accuracy']:.4f}, "
            f"MacroF1={test_final['macro_f1']:.4f}"
        ),
        (
            f"  Gain: dAcc={test_final['accuracy_gain_over_lag1']:+.4f}, "
            f"dBalAcc={test_final['balanced_accuracy_gain_over_lag1']:+.4f}, "
            f"dMacroF1={test_final['macro_f1_gain_over_lag1']:+.4f}"
        ),
        (
            f"  Override rate={test_final['predicted_override_rate']:.4%}, "
            f"corrected switches={test_final['corrected_switch_rows']:,}, "
            f"damaged persistence={test_final['damaged_persistence_rows']:,}"
        ),
        "",
        "Decision rule:",
        "  default template = hist_lag1_template_id",
        (
            f"  if P(switch) >= {final_threshold:.2f}: "
            f"override with RF GlobalOriginMasked destination"
        ),
        "  else: keep lag1 template",
        "",
        "Final acceptance:",
        "  Compare held-out TEST Final vs Lag1.",
        "  If gains are positive and nontrivial, lock the hierarchy.",
        "  If validation chooses threshold=1.01, the final locked predictor is lag1.",
    ]

    summary = "\n".join(
        summary_lines
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
        "switch_detector": {
            "model": "RandomForest",
            "feature_set": "Z_base_tr_H",
            "feature_count": len(
                features["switch_features"]
            ),
            "n_estimators": args.switch_rf_trees,
            "max_depth": args.switch_rf_max_depth,
            "min_samples_leaf": args.switch_rf_min_samples_leaf,
        },
        "destination": {
            "model": "RandomForest",
            "feature_set": "Z_base_M_U_H_context_plus_origin",
            "feature_count_excluding_origin": len(
                features[
                    "destination_features"
                ]
            ),
            "origin_condition": ORIGIN,
            "candidate_mask": "TRAIN-only transition matrix",
            "n_estimators": args.destination_rf_trees,
            "max_depth": args.destination_rf_max_depth,
            "min_samples_leaf": args.destination_rf_min_samples_leaf,
        },
        "threshold_selection": {
            "split": "validation only",
            "primary": "end-to-end MacroF1",
            "secondary": "balanced_accuracy",
            "tertiary": "accuracy",
            "allow_exact_lag1_fallback": True,
        },
        "selected_hierarchy_threshold": final_threshold,
        "seed": args.seed,
    }

    (
        out_dir
        / "config.json"
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
