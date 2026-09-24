#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04b_train_conditional_theta_model_v3.py

Train a generic destination-conditioned theta model for TRUE template-switch events.

Architecture
------------
Template predictor supplies, using only predictor-side information:
    P(switch)
    predicted destination template
    destination probabilities

This script trains theta only on rows where:
    true current template != historical template

Target:
    absolute current theta
    [p_base, slope, tail_uplift, q_base, log(q_span), ALR(q1..q5)]

Important:
- The 0.95 hierarchy threshold is NOT used to select training rows.
- True switch is used only to define the training population / target event.
- The true current template is NOT used as an input feature.
- Training conditions theta on the TRUE destination template.
- Deployment later substitutes the predicted destination template.
- Switch probability / destination probability are NOT theta-model inputs.
- No residual model, no gate, no stacking, no direct 21-point prediction.

Run
---
python scripts/bidprediction/04b_train_conditional_theta_model_v3.py --year 2025
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer


TEMPLATES = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
T2I = {t: i for i, t in enumerate(TEMPLATES)}
N_TEMPLATE = len(TEMPLATES)

STABLE_THETA = [
    "stable_theta_p_base",
    "stable_theta_slope",
    "stable_theta_tail_uplift",
    "stable_theta_q_base_mw",
    "stable_theta_q_span_mw",
    "stable_theta_q1",
    "stable_theta_q2",
    "stable_theta_q3",
    "stable_theta_q4",
    "stable_theta_q5",
]
HIST_THETA = ["hist_lag1_" + c for c in STABLE_THETA]

TARGET_NAMES = [
    "target_p_base",
    "target_slope",
    "target_tail_uplift",
    "target_q_base",
    "target_log_q_span",
    "target_qshare_alr1",
    "target_qshare_alr2",
    "target_qshare_alr3",
    "target_qshare_alr4",
]

BASE_ALLOWED_GROUPS = {
    "profile_LT",
    "profile_ST",
    "profile_Break",
    "transition_strategy_profile",
    "market_environment",
    "unit_state_proxy",
    "calendar",
    "stable_theta_history",
    "stable_theta_history_stats",
    "context_delta",
}

EXCLUDE = {
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

MODE2I = {
    "flat": 0,
    "block": 1,
    "sloped": 2,
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def norm(s):
    return s.astype("string").str.strip()


def history_ready(d):
    return (
        d[HIST_THETA]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )


def template_history_ready(d):
    if "hist_lag1_template_id" not in d.columns:
        return pd.Series(False, index=d.index)
    return norm(d["hist_lag1_template_id"]).isin(TEMPLATES)


def raw_theta(d, hist=False):
    cols = HIST_THETA if hist else STABLE_THETA
    return (
        d[cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )


def alr(q):
    q = np.asarray(q, dtype=float)
    q = np.maximum(q, 1e-8)
    q /= q.sum(axis=1, keepdims=True)
    return np.log(q[:, :4] / q[:, 4, None])


def inv_alr(z):
    z = np.asarray(z, dtype=float)
    x = np.c_[z, np.zeros(len(z))]
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def theta_to_target(d):
    theta = raw_theta(d, hist=False)
    return np.c_[
        theta[:, 0],
        theta[:, 1],
        theta[:, 2],
        theta[:, 3],
        np.log(np.maximum(theta[:, 4], 1e-6)),
        alr(theta[:, 5:10]),
    ]


def target_to_theta(z):
    z = np.asarray(z, dtype=float)
    out = np.empty((len(z), 10), dtype=float)
    out[:, 0:4] = z[:, 0:4]
    out[:, 4] = np.exp(np.clip(z[:, 4], -12.0, 12.0))
    out[:, 5:10] = inv_alr(z[:, 5:9])
    return out


# =============================================================================
# Template predictor
# =============================================================================

def template_encode_col(s, col):
    if col in {"hist_lag1_template_id", "hist30_dominant_template_id"}:
        return norm(s).map(T2I).astype("float32")
    if col == "hist_lag1_curve_mode":
        return norm(s).str.lower().map(MODE2I).astype("float32")
    return num(s).astype("float32")


def expand_proba(model, raw_p):
    classes = np.asarray(model.classes_, dtype=np.int16)
    full = np.zeros((len(raw_p), N_TEMPLATE), dtype=float)
    full[:, classes] = raw_p
    return full


def mask_destination_proba(p, origin, allowed):
    p = np.asarray(p, dtype=float).copy()

    for i in range(N_TEMPLATE):
        rows = np.where(origin == i)[0]
        if not len(rows):
            continue

        mask = allowed[i].copy()
        if not mask.any():
            mask[:] = True
            mask[i] = False

        p[np.ix_(rows, ~mask)] = 0.0

    den = p.sum(axis=1, keepdims=True)
    bad = den[:, 0] <= 0

    if bad.any():
        for r in np.where(bad)[0]:
            p[r, :] = 1.0
            p[r, origin[r]] = 0.0
        den = p.sum(axis=1, keepdims=True)

    p /= den
    return p


def template_context(d, bundle):
    origin_s = norm(d["hist_lag1_template_id"]).map(T2I)
    if origin_s.isna().any():
        raise ValueError("Invalid hist_lag1_template_id after filtering.")
    origin = origin_s.to_numpy(np.int16)

    # Switch probability
    sf = bundle["switch_features"]
    Xs = pd.DataFrame(
        {c: template_encode_col(d[c], c) for c in sf},
        index=d.index,
    )
    Xs = bundle["switch_imputer"].transform(Xs).astype(np.float32)

    sw = bundle["switch_model"]
    raw_sw = sw.predict_proba(Xs)
    sw_classes = list(sw.classes_)
    p_switch = (
        raw_sw[:, sw_classes.index(1)]
        if 1 in sw_classes
        else np.zeros(len(d), dtype=float)
    )

    # Destination probability
    df = bundle["destination_features"]
    Xd = pd.DataFrame(
        {c: template_encode_col(d[c], c) for c in df},
        index=d.index,
    )
    Xd = bundle["destination_imputer"].transform(Xd).astype(np.float32)

    Xdg = np.column_stack([Xd, origin.astype(np.float32)])
    dm = bundle["destination_model"]

    dest_p = expand_proba(dm, dm.predict_proba(Xdg))
    dest_p = mask_destination_proba(
        dest_p,
        origin,
        np.asarray(bundle["allowed_destinations_train"], dtype=bool),
    )

    destination = np.argmax(dest_p, axis=1).astype(np.int16)

    entropy = -np.sum(
        dest_p * np.log(np.clip(dest_p, 1e-12, 1.0)),
        axis=1,
    ) / np.log(N_TEMPLATE)

    return {
        "origin": origin,
        "switch_probability": p_switch.astype(np.float32),
        "destination_probability": dest_p.astype(np.float32),
        "destination_entropy": entropy.astype(np.float32),
        "destination_max_probability": dest_p.max(axis=1).astype(np.float32),
        "destination_template": np.asarray(TEMPLATES, dtype=object)[destination],
    }


# =============================================================================
# Features
# =============================================================================

def select_base_features(schema):
    s = schema[
        schema["role"].astype(str).str.lower().eq("feature")
    ].copy()

    out = []

    for row in s.itertuples(index=False):
        c = str(row.column)
        g = str(row.feature_group)
        if g in BASE_ALLOWED_GROUPS and c not in EXCLUDE:
            out.append(c)

    available = set(s["column"].astype(str))

    if "hist_days_since_prev_same_slot" in available:
        out.append("hist_days_since_prev_same_slot")

    out = list(dict.fromkeys(out))

    for c in HIST_THETA:
        if c not in out:
            raise KeyError(f"Missing history theta feature: {c}")

    return out


def prepare_X(
    d,
    base_features,
    origin_template,
    destination_template,
):
    """
    Generic conditional-regression features.

    The theta regressor learns:
        theta = f(context, history, origin_template, destination_template)

    During training destination_template is the TRUE current template.
    During deployment destination_template is supplied by the upstream
    destination classifier.

    Upstream probabilities are deliberately excluded to keep the downstream
    regression definition stable across datasets and probability calibration.
    """
    out = {
        c: num(d[c]).astype("float32")
        for c in base_features
    }

    origin = (
        pd.Series(
            origin_template,
            index=d.index,
            dtype="string",
        )
        .map(T2I)
    )

    destination = (
        pd.Series(
            destination_template,
            index=d.index,
            dtype="string",
        )
        .map(T2I)
    )

    if origin.isna().any():
        raise ValueError("Invalid origin template in theta features.")

    if destination.isna().any():
        raise ValueError("Invalid destination template in theta features.")

    origin = origin.to_numpy(np.int16)
    destination = destination.to_numpy(np.int16)

    for j, name in enumerate(TEMPLATES):
        out[f"origin_template_{name}"] = (
            origin == j
        ).astype("float32")

        out[f"destination_template_{name}"] = (
            destination == j
        ).astype("float32")

    return pd.DataFrame(
        out,
        index=d.index,
    )


# =============================================================================
# Data collection
# =============================================================================

def deterministic_cap(d, max_rows, seed):
    if max_rows is None or len(d) <= max_rows:
        return d.reset_index(drop=True)

    h = pd.util.hash_pandas_object(
        norm(d["sample_id"]),
        index=False,
    ).to_numpy(np.uint64)

    priority = h ^ np.uint64(seed * 0x9E3779B1)

    idx = np.argsort(
        priority,
        kind="mergesort",
    )[:max_rows]

    return d.iloc[idx].reset_index(drop=True)


def collect_true_switch_rows(
    frozen,
    manifest,
    split,
    template_bundle,
    max_rows,
    seed,
):
    blocks = []

    ready_rows = 0
    dropped_invalid_hist_template = 0
    true_switch_rows = 0
    destination_correct = 0

    for i, rel in enumerate(manifest["parts"][split], 1):
        p = frozen / rel

        print(
            f"[{split} {i}/{len(manifest['parts'][split])}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(p)

        ready_theta = history_ready(d)
        ready_template = template_history_ready(d)

        dropped_invalid_hist_template += int(
            (ready_theta & (~ready_template)).sum()
        )

        d = (
            d.loc[ready_theta & ready_template]
            .copy()
            .reset_index(drop=True)
        )

        if d.empty:
            continue

        ready_rows += len(d)

        origin = norm(d["hist_lag1_template_id"]).to_numpy(object)
        true_template = norm(d["y_template_id"]).to_numpy(object)
        true_switch = true_template != origin

        true_switch_rows += int(true_switch.sum())

        if not true_switch.any():
            continue

        ds = (
            d.loc[true_switch]
            .copy()
            .reset_index(drop=True)
        )

        tctx = template_context(ds, template_bundle)

        destination_correct += int(
            np.sum(
                tctx["destination_template"]
                == norm(ds["y_template_id"]).to_numpy(object)
            )
        )

        blocks.append(ds)

        del d, ds, tctx
        gc.collect()

    if not blocks:
        raise ValueError(f"No true-switch rows in split={split}.")

    out = pd.concat(blocks, ignore_index=True)
    out = deterministic_cap(out, max_rows, seed)

    stats = {
        "ready_rows": int(ready_rows),
        "dropped_invalid_hist_template_rows": int(
            dropped_invalid_hist_template
        ),
        "true_switch_rows": int(true_switch_rows),
        "true_switch_share": float(
            true_switch_rows / max(ready_rows, 1)
        ),
        "destination_accuracy_on_true_switch": float(
            destination_correct / max(true_switch_rows, 1)
        ),
        "retained_true_switch_rows": int(len(out)),
    }

    return out, stats


# =============================================================================
# Model
# =============================================================================

def fit_heads(
    X,
    Y,
    trees,
    max_depth,
    min_leaf,
    max_features,
    n_jobs,
    seed,
):
    heads = []

    for j, name in enumerate(TARGET_NAMES):
        print(f"  [head {j+1}/9] {name}", flush=True)

        model = ExtraTreesRegressor(
            n_estimators=trees,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=seed + 37 * j,
        )

        model.fit(X, Y[:, j])
        heads.append(model)

    return heads


def predict_heads(heads, X):
    return np.column_stack(
        [model.predict(X) for model in heads]
    )


def theta_metrics(d, theta_pred):
    true = raw_theta(d, hist=False)

    price_names = STABLE_THETA[:3]

    rows = []

    for j, name in enumerate(STABLE_THETA):
        err = theta_pred[:, j] - true[:, j]
        rows.append(
            {
                "theta": name,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--root",
        default="data/processed/bidprediction",
    )
    ap.add_argument(
        "--max-train-switch-rows",
        type=int,
        default=250_000,
    )
    ap.add_argument(
        "--max-val-switch-rows",
        type=int,
        default=120_000,
    )
    ap.add_argument("--trees", type=int, default=96)
    ap.add_argument("--max-depth", type=int, default=18)
    ap.add_argument("--min-leaf", type=int, default=5)
    ap.add_argument("--max-features", type=float, default=0.50)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    frozen = base / "frozen_stable_theta_continuous_dataset"

    manifest = json.loads(
        (frozen / "manifest.json").read_text(encoding="utf-8")
    )

    schema = pd.read_csv(frozen / "feature_schema.csv")
    base_features = select_base_features(schema)

    template_bundle = joblib.load(
        base
        / "final_template_predictor"
        / "final_template_model.joblib"
    )

    print("=" * 80)
    print(f"Destination-conditioned stable-theta model v3 - {args.year}")
    print("=" * 80)
    print(f"Base features = {len(base_features)}")
    print("Training population = TRUE template-switch rows")
    print("Training template condition = TRUE destination; deployment = predicted destination")
    print("Target = absolute theta")
    print()

    train, train_stats = collect_true_switch_rows(
        frozen,
        manifest,
        "train",
        template_bundle,
        args.max_train_switch_rows,
        args.seed,
    )

    val, val_stats = collect_true_switch_rows(
        frozen,
        manifest,
        "val",
        template_bundle,
        args.max_val_switch_rows,
        args.seed + 17,
    )

    print()
    print(f"Train true-switch rows = {len(train):,}")
    print(f"Validation true-switch rows = {len(val):,}")
    print()

    train_origin = norm(
        train["hist_lag1_template_id"]
    ).to_numpy(object)

    train_true_destination = norm(
        train["y_template_id"]
    ).to_numpy(object)

    Xdf = prepare_X(
        train,
        base_features,
        train_origin,
        train_true_destination,
    )

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X = imputer.fit_transform(Xdf).astype(np.float32)
    Y = theta_to_target(train)

    print("[fit] 9 destination-conditioned absolute-theta heads", flush=True)

    heads = fit_heads(
        X,
        Y,
        trees=args.trees,
        max_depth=args.max_depth,
        min_leaf=args.min_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    # -------------------------------------------------------------
    # Validation on TRUE-switch rows.
    # 1) Oracle destination: correct class condition supplied.
    # 2) Predicted destination: deployment-like upstream condition.
    # -------------------------------------------------------------
    val_origin = norm(
        val["hist_lag1_template_id"]
    ).to_numpy(object)

    val_true_destination = norm(
        val["y_template_id"]
    ).to_numpy(object)

    val_tctx = template_context(
        val,
        template_bundle,
    )

    val_pred_destination = val_tctx[
        "destination_template"
    ]

    Xv_oracle_df = prepare_X(
        val,
        base_features,
        val_origin,
        val_true_destination,
    )

    Xv_pred_df = prepare_X(
        val,
        base_features,
        val_origin,
        val_pred_destination,
    )

    Xv_oracle = imputer.transform(
        Xv_oracle_df
    ).astype(np.float32)

    Xv_pred = imputer.transform(
        Xv_pred_df
    ).astype(np.float32)

    theta_oracle_destination = target_to_theta(
        predict_heads(
            heads,
            Xv_oracle,
        )
    )

    theta_pred_destination = target_to_theta(
        predict_heads(
            heads,
            Xv_pred,
        )
    )

    metrics_oracle = theta_metrics(
        val,
        theta_oracle_destination,
    )
    metrics_oracle["conditioning"] = "oracle_destination"

    metrics_pred = theta_metrics(
        val,
        theta_pred_destination,
    )
    metrics_pred["conditioning"] = "predicted_destination"

    val_theta_metrics = pd.concat(
        [
            metrics_oracle,
            metrics_pred,
        ],
        ignore_index=True,
    )

    destination_accuracy_val = float(
        np.mean(
            val_pred_destination
            == val_true_destination
        )
    )

    out = base / "conditional_theta_model_v3"
    out.mkdir(parents=True, exist_ok=True)

    bundle = {
        "version": "conditional-theta-absolute-v3",
        "model_mode": "destination_conditioned_absolute_theta_on_true_switch_events",
        "stable_theta": STABLE_THETA,
        "history_theta": HIST_THETA,
        "target_names": TARGET_NAMES,
        "templates": TEMPLATES,
        "base_features": base_features,
        "imputer": imputer,
        "theta_heads": heads,
        "training_population": (
            "true current template != historical template"
        ),
        "conditioning_definition": (
            "theta = f(context, history, origin_template, destination_template)"
        ),
        "training_condition": (
            "destination_template = true current template on true-switch rows"
        ),
        "deployment_condition": (
            "destination_template = predicted destination template"
        ),
        "probability_features_used": False,
        "target_definition": (
            "absolute current theta: p_base, slope, tail_uplift, "
            "q_base, log(q_span), ALR(q1..q5)"
        ),
        "model_config": {
            "trees_per_head": args.trees,
            "max_depth": args.max_depth,
            "min_leaf": args.min_leaf,
            "max_features": args.max_features,
            "n_jobs": args.n_jobs,
        },
    }

    joblib.dump(
        bundle,
        out / "conditional_theta_model_v3.joblib",
        compress=3,
    )

    val_theta_metrics.to_csv(
        out / "validation_theta_metrics_true_switch.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics = {
        "train": train_stats,
        "validation": val_stats,
        "validation_destination_accuracy_on_true_switch": (
            destination_accuracy_val
        ),
    }

    (out / "diagnostics.json").write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            f"Destination-conditioned stable-theta model v3 - {args.year}",
            "=" * 80,
            "",
            f"Base features = {len(base_features)}",
            f"Train true-switch rows = {len(train):,}",
            f"Validation true-switch rows = {len(val):,}",
            "",
            "Train diagnostics:",
            json.dumps(train_stats, ensure_ascii=False, indent=2),
            "",
            "Validation diagnostics:",
            json.dumps(val_stats, ensure_ascii=False, indent=2),
            "",
            (
                "Validation destination accuracy on true switch = "
                f"{destination_accuracy_val:.6f}"
            ),
            "",
            "Validation theta metrics:",
            val_theta_metrics.to_string(index=False),
        ]
    )

    (out / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
