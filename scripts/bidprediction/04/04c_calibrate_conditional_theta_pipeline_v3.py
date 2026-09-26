#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04c_calibrate_conditional_theta_pipeline_v3.py

Calibrate the operational template-switch threshold on VALIDATION by the
FINAL reconstructed bid-curve WAPE, then freeze that threshold and evaluate
TEST exactly once.

Deployment logic
----------------
if P(switch) < tau*:
    final_template = historical_template
    theta_pred = historical_theta
else:
    final_template = predicted_destination_template
    theta_pred = switch_theta_model(X)

tau* is selected ONLY on validation.

No test-set tuning.
No binary/continuous theta gate.
No residual stacking.
No direct 21-point curve prediction.

Run
---
python scripts/bidprediction/04c_calibrate_conditional_theta_pipeline_v3.py --year 2025
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


GRID = np.linspace(0.0, 1.0, 21)
UK = np.linspace(0.0, 1.0, 6)

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

MODE2I = {
    "flat": 0,
    "block": 1,
    "sloped": 2,
}


# =============================================================================
# Basic helpers
# =============================================================================

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


def inv_alr(z):
    z = np.asarray(z, dtype=float)
    x = np.c_[z, np.zeros(len(z))]
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


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

    # Destination
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
# Switch-theta model
# =============================================================================

def prepare_X(
    d,
    model_bundle,
    origin_template,
    destination_template,
):
    """
    Match 04b v3:
        theta = f(context, history, origin_template, destination_template)

    The downstream theta model does not consume switch/destination
    probabilities. Training uses true destination; deployment uses predicted
    destination.
    """
    out = {
        c: num(d[c]).astype("float32")
        for c in model_bundle["base_features"]
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
        raise ValueError("Invalid origin template for theta model.")

    if destination.isna().any():
        raise ValueError("Invalid destination template for theta model.")

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


def predict_heads(heads, X):
    return np.column_stack(
        [model.predict(X) for model in heads]
    )


def predict_switch_theta(
    d,
    origin_template,
    destination_template,
    model_bundle,
):
    Xdf = prepare_X(
        d,
        model_bundle,
        origin_template,
        destination_template,
    )

    X = model_bundle["imputer"].transform(
        Xdf
    ).astype(np.float32)

    target = predict_heads(
        model_bundle["theta_heads"],
        X,
    )

    return target_to_theta(target)


# =============================================================================
# Curve reconstruction
# =============================================================================

def discover_centers(year_dir):
    preferred = (
        year_dir
        / "template_library"
        / "curve_template_library.csv"
    )

    candidates = (
        [preferred]
        if preferred.exists()
        else sorted(year_dir.rglob("*.csv"))
    )

    shape_cols = [
        f"shape_v{i:02d}"
        for i in range(21)
    ]

    for p in candidates:
        try:
            header = pd.read_csv(
                p,
                nrows=0,
            ).columns.tolist()
        except Exception:
            continue

        if not all(c in header for c in shape_cols):
            continue

        id_col = next(
            (
                c
                for c in [
                    "template_id",
                    "template",
                    "cluster_id",
                    "cluster_label",
                    "label",
                ]
                if c in header
            ),
            None,
        )

        if id_col is None:
            continue

        d = pd.read_csv(
            p,
            usecols=[id_col, *shape_cols],
        )

        if len(d) > 100:
            continue

        centers = {}

        for _, row in d.iterrows():
            raw = str(row[id_col]).strip()

            if raw.upper() == "FLAT":
                tid = "FLAT"
            else:
                try:
                    i = int(
                        float(
                            raw
                            .replace("T", "")
                            .replace("t", "")
                        )
                    )
                    tid = f"T{i:02d}"
                except Exception:
                    continue

            arr = pd.to_numeric(
                row[shape_cols],
                errors="coerce",
            ).to_numpy(float)

            if np.isfinite(arr).all():
                centers[tid] = arr

        centers["FLAT"] = np.zeros(21)

        if all(t in centers for t in TEMPLATES):
            return centers, p

    raise FileNotFoundError(
        "Cannot locate complete template center library."
    )


def inverse_warp(qshares):
    qshares = np.asarray(qshares, dtype=float)
    qshares = np.maximum(qshares, 1e-8)
    qshares /= qshares.sum(axis=1, keepdims=True)

    cum = np.c_[
        np.zeros(len(qshares)),
        np.cumsum(qshares, axis=1),
    ]
    cum[:, -1] = 1.0

    u = np.empty((len(qshares), 21), dtype=float)
    row = np.arange(len(qshares))

    for j, x in enumerate(GRID):
        k = np.sum(
            x >= cum[:, 1:],
            axis=1,
        )
        k = np.clip(k, 0, 4)

        x0 = cum[row, k]
        x1 = cum[row, k + 1]

        f = np.clip(
            (x - x0)
            / np.maximum(x1 - x0, 1e-12),
            0.0,
            1.0,
        )

        u[:, j] = (
            UK[k]
            + f * (UK[k + 1] - UK[k])
        )

    u[:, 0] = 0.0
    u[:, -1] = 1.0
    return u


def interp_center(center, u):
    pos = np.clip(
        u * 20.0,
        0.0,
        20.0,
    )
    lo = np.floor(pos).astype(np.int16)
    hi = np.minimum(lo + 1, 20)
    f = pos - lo

    return (
        center[lo]
        + f * (center[hi] - center[lo])
    )


def reconstruct_price(theta, templates, centers):
    theta = np.asarray(theta, dtype=float)
    templates = np.asarray(templates, dtype=object)

    U = inverse_warp(theta[:, 5:10])
    P = np.empty((len(theta), 21), dtype=float)

    for tid in TEMPLATES:
        idx = np.where(templates == tid)[0]

        if not len(idx):
            continue

        th = theta[idx]
        u = U[idx]

        p0 = th[:, 0]
        slope = th[:, 1]
        tail = th[:, 2]

        if tid == "FLAT":
            P[idx, :] = p0[:, None]
            continue

        center = centers[tid]
        c0 = float(center[0])
        c70 = float(
            interp_center(
                center,
                np.array([0.70]),
            )[0]
        )
        c1 = float(center[-1])

        p70 = p0 + 0.70 * slope
        p1 = p70 + 0.30 * slope + tail

        tc = interp_center(center, u)
        body = u <= 0.70

        if abs(c70 - c0) > 1e-10:
            wb = (tc - c0) / (c70 - c0)
        else:
            wb = u / 0.70

        wb = np.clip(wb, 0.0, 1.0)

        body_price = (
            p0[:, None]
            + (p70 - p0)[:, None] * wb
        )

        if abs(c1 - c70) > 1e-10:
            wt = (tc - c70) / (c1 - c70)
        else:
            wt = (u - 0.70) / 0.30

        wt = np.clip(wt, 0.0, 1.0)

        tail_price = (
            p70[:, None]
            + (p1 - p70)[:, None] * wt
        )

        P[idx, :] = np.where(
            body,
            body_price,
            tail_price,
        )

    return P


def true_curve(d):
    shape_cols = [
        f"shape_v{i:02d}"
        for i in range(21)
    ]

    shape = (
        d[shape_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    flat = (
        norm(d["y_template_id"])
        .eq("FLAT")
        .to_numpy()
    )

    if flat.any():
        shape[flat, :] = 0.0

    pa = num(d["p_anchor"]).to_numpy(float)
    ps = num(d["p_span"]).to_numpy(float)

    P = pa[:, None] + ps[:, None] * shape

    qb = num(d["q_anchor_mw"]).to_numpy(float)
    qs = num(d["q_span_mw"]).to_numpy(float)

    Q = qb[:, None] + qs[:, None] * GRID[None, :]

    return Q, P


def pred_price_on_true_q(true_q, theta, pred_p):
    q_base = theta[:, 3]
    q_span = np.maximum(theta[:, 4], 1e-8)

    x = (
        true_q
        - q_base[:, None]
    ) / q_span[:, None]

    pos = np.clip(x, 0.0, 1.0) * 20.0
    lo = np.floor(pos).astype(np.int16)
    hi = np.minimum(lo + 1, 20)
    f = pos - lo

    p_lo = np.take_along_axis(
        pred_p,
        lo,
        axis=1,
    )
    p_hi = np.take_along_axis(
        pred_p,
        hi,
        axis=1,
    )

    return p_lo + f * (p_hi - p_lo)


def per_row_curve_errors(
    d,
    theta,
    templates,
    centers,
):
    true_q, true_p = true_curve(d)

    pred_p = reconstruct_price(
        theta,
        templates,
        centers,
    )

    pred_on_true = pred_price_on_true_q(
        true_q,
        theta,
        pred_p,
    )

    err = pred_on_true - true_p
    ae = np.abs(err)

    denom_smape = (
        np.abs(pred_on_true)
        + np.abs(true_p)
    )

    smape = np.divide(
        2.0 * ae,
        denom_smape,
        out=np.zeros_like(ae),
        where=denom_smape > 1e-9,
    )

    return {
        "ae_sum": ae.sum(axis=1),
        "se_sum": np.square(err).sum(axis=1),
        "smape_sum": smape.sum(axis=1),
        "curve_mae": ae.mean(axis=1),
        "true_abs_sum": np.abs(true_p).sum(axis=1),
    }


# =============================================================================
# Validation threshold calibration
# =============================================================================

def parse_thresholds(spec):
    if spec:
        vals = [
            float(x)
            for x in spec.split(",")
            if x.strip()
        ]
    else:
        vals = list(
            np.round(
                np.arange(
                    0.20,
                    0.951,
                    0.01,
                ),
                2,
            )
        )

    vals = sorted(
        set(
            float(x)
            for x in vals
            if 0.0 <= float(x) <= 1.0
        )
    )

    if not vals:
        raise ValueError("Empty threshold grid.")

    return vals


def collect_validation_branch_errors(
    frozen,
    manifest,
    template_bundle,
    theta_bundle,
    centers,
    thresholds,
):
    min_tau = min(thresholds)

    p_blocks = []
    true_switch_blocks = []
    destination_correct_blocks = []
    persistence_correct_blocks = []

    persist_ae_blocks = []
    switch_ae_blocks = []
    true_abs_blocks = []

    total_rows = 0
    dropped_invalid_hist_template = 0

    parts = manifest["parts"]["val"]

    for i, rel in enumerate(parts, 1):
        path = frozen / rel

        print(
            f"[VAL {i}/{len(parts)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)

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

        tctx = template_context(d, template_bundle)

        origin = np.asarray(
            TEMPLATES,
            dtype=object,
        )[tctx["origin"]]

        true_template = norm(
            d["y_template_id"]
        ).to_numpy(object)

        destination = tctx["destination_template"]
        p_switch = tctx["switch_probability"].astype(float)

        theta_p = raw_theta(d, hist=True)

        persist_err = per_row_curve_errors(
            d,
            theta_p,
            origin,
            centers,
        )

        # Default switch branch equals persistence for rows that can never be
        # selected by this threshold grid. This keeps arrays dense and simple.
        switch_ae = persist_err["ae_sum"].copy()

        candidate = p_switch >= min_tau

        if candidate.any():
            ds = (
                d.loc[candidate]
                .copy()
                .reset_index(drop=True)
            )

            theta_s = predict_switch_theta(
                ds,
                origin[candidate],
                destination[candidate],
                theta_bundle,
            )

            switch_err = per_row_curve_errors(
                ds,
                theta_s,
                destination[candidate],
                centers,
            )

            switch_ae[candidate] = switch_err["ae_sum"]

        p_blocks.append(p_switch.astype(np.float32))
        true_switch_blocks.append(
            (true_template != origin)
        )
        destination_correct_blocks.append(
            destination == true_template
        )
        persistence_correct_blocks.append(
            origin == true_template
        )

        persist_ae_blocks.append(
            persist_err["ae_sum"].astype(np.float32)
        )
        switch_ae_blocks.append(
            switch_ae.astype(np.float32)
        )
        true_abs_blocks.append(
            persist_err["true_abs_sum"].astype(np.float32)
        )

        total_rows += len(d)

        del d, tctx, theta_p, persist_err
        gc.collect()

    return {
        "p_switch": np.concatenate(p_blocks).astype(float),
        "true_switch": np.concatenate(true_switch_blocks).astype(bool),
        "destination_correct": np.concatenate(
            destination_correct_blocks
        ).astype(bool),
        "persistence_correct": np.concatenate(
            persistence_correct_blocks
        ).astype(bool),
        "persist_ae": np.concatenate(persist_ae_blocks).astype(float),
        "switch_ae": np.concatenate(switch_ae_blocks).astype(float),
        "true_abs": np.concatenate(true_abs_blocks).astype(float),
        "rows": int(total_rows),
        "dropped_invalid_hist_template_rows": int(
            dropped_invalid_hist_template
        ),
    }


def threshold_sweep_from_branch_errors(data, thresholds):
    p = data["p_switch"]
    true_switch = data["true_switch"]
    dest_ok = data["destination_correct"]
    persist_ok = data["persistence_correct"]

    persist_ae = data["persist_ae"]
    switch_ae = data["switch_ae"]

    denominator = max(
        float(data["true_abs"].sum()),
        1e-12,
    )

    base_ae = float(persist_ae.sum())

    rows = []

    for tau in thresholds:
        mask = p >= tau

        total_ae = (
            base_ae
            + float(
                (
                    switch_ae[mask]
                    - persist_ae[mask]
                ).sum()
            )
        )

        tp = int(np.sum(true_switch & mask))
        fp = int(np.sum((~true_switch) & mask))
        fn = int(np.sum(true_switch & (~mask)))
        tn = int(np.sum((~true_switch) & (~mask)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        f1 = (
            2.0 * precision * recall
            / max(
                precision + recall,
                1e-12,
            )
        )

        tnr = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (recall + tnr)

        final_correct = (
            ((~mask) & persist_ok)
            | (mask & dest_ok)
        )

        rows.append(
            {
                "threshold": float(tau),
                "rows": int(len(p)),
                "true_switch_share": float(
                    true_switch.mean()
                ),
                "pred_switch_share": float(
                    mask.mean()
                ),
                "switch_precision": float(precision),
                "switch_recall": float(recall),
                "switch_f1": float(f1),
                "switch_balanced_accuracy": float(bal_acc),
                "final_template_accuracy": float(
                    final_correct.mean()
                ),
                "price_wape_pct": float(
                    100.0
                    * total_ae
                    / denominator
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Test metrics
# =============================================================================

def init_state():
    return {
        "rows": 0,
        "price_ae": 0.0,
        "price_se": 0.0,
        "price_abs_true": 0.0,
        "smape_sum": 0.0,
        "smape_points": 0,
        "curve_mae": [],
        "q_anchor_ae": 0.0,
        "q_anchor_abs_true": 0.0,
        "q_span_ae": 0.0,
        "q_span_abs_true": 0.0,
        "q_share_ae": 0.0,
        "q_share_points": 0,
    }


def update_state(
    state,
    d,
    theta,
    templates,
    centers,
):
    if len(d) == 0:
        return

    true_theta = raw_theta(d, hist=False)
    true_q, true_p = true_curve(d)

    pred_p = reconstruct_price(
        theta,
        templates,
        centers,
    )

    pred_on_true = pred_price_on_true_q(
        true_q,
        theta,
        pred_p,
    )

    err = pred_on_true - true_p
    ae = np.abs(err)

    denom = (
        np.abs(pred_on_true)
        + np.abs(true_p)
    )

    smape = np.divide(
        2.0 * ae,
        denom,
        out=np.zeros_like(ae),
        where=denom > 1e-9,
    )

    state["rows"] += len(d)
    state["price_ae"] += float(ae.sum())
    state["price_se"] += float(np.square(err).sum())
    state["price_abs_true"] += float(np.abs(true_p).sum())
    state["smape_sum"] += float(smape.sum())
    state["smape_points"] += int(smape.size)

    state["curve_mae"].append(
        ae.mean(axis=1).astype(np.float32)
    )

    state["q_anchor_ae"] += float(
        np.abs(
            theta[:, 3]
            - true_theta[:, 3]
        ).sum()
    )
    state["q_anchor_abs_true"] += float(
        np.abs(true_theta[:, 3]).sum()
    )

    state["q_span_ae"] += float(
        np.abs(
            theta[:, 4]
            - true_theta[:, 4]
        ).sum()
    )
    state["q_span_abs_true"] += float(
        np.abs(true_theta[:, 4]).sum()
    )

    state["q_share_ae"] += float(
        np.abs(
            theta[:, 5:10]
            - true_theta[:, 5:10]
        ).sum()
    )
    state["q_share_points"] += int(len(d) * 5)


def finalize_state(state):
    curve = (
        np.concatenate(state["curve_mae"])
        if state["curve_mae"]
        else np.empty(0, dtype=float)
    )

    price_points = state["rows"] * 21

    return {
        "rows": state["rows"],
        "price_mae": (
            state["price_ae"]
            / max(price_points, 1)
        ),
        "price_rmse": float(
            np.sqrt(
                state["price_se"]
                / max(price_points, 1)
            )
        ),
        "price_wape_pct": (
            100.0
            * state["price_ae"]
            / max(
                state["price_abs_true"],
                1e-12,
            )
        ),
        "price_smape_pct": (
            100.0
            * state["smape_sum"]
            / max(
                state["smape_points"],
                1,
            )
        ),
        "curve_mae_p50": (
            float(np.quantile(curve, 0.50))
            if len(curve)
            else np.nan
        ),
        "curve_mae_p90": (
            float(np.quantile(curve, 0.90))
            if len(curve)
            else np.nan
        ),
        "curve_mae_p95": (
            float(np.quantile(curve, 0.95))
            if len(curve)
            else np.nan
        ),
        "curve_mae_le20_share": (
            float(np.mean(curve <= 20.0))
            if len(curve)
            else np.nan
        ),
        "q_anchor_wape_pct": (
            100.0
            * state["q_anchor_ae"]
            / max(
                state["q_anchor_abs_true"],
                1e-12,
            )
        ),
        "q_span_wape_pct": (
            100.0
            * state["q_span_ae"]
            / max(
                state["q_span_abs_true"],
                1e-12,
            )
        ),
        "q_share_mae": (
            state["q_share_ae"]
            / max(
                state["q_share_points"],
                1,
            )
        ),
    }


def evaluate_test(
    frozen,
    manifest,
    template_bundle,
    theta_bundle,
    centers,
    threshold,
):
    states = {
        "persistence_origin_template": init_state(),
        "persistence_oracle_template": init_state(),
        "hybrid_full_pipeline": init_state(),
        "hybrid_oracle_template": init_state(),
        "persistence_on_predicted_switch_rows": init_state(),
        "switch_theta_on_predicted_switch_rows": init_state(),
        "persistence_on_true_switch_rows": init_state(),
        "hybrid_on_true_switch_rows": init_state(),
        "pred_destination_conditional_theta_on_true_switch_rows": init_state(),
        "oracle_destination_conditional_theta_on_true_switch_rows": init_state(),
        "persistence_on_true_unchanged_rows": init_state(),
        "hybrid_on_true_unchanged_rows": init_state(),
    }

    total_rows = 0
    dropped_invalid_hist_template = 0
    true_switch_rows = 0
    pred_switch_rows = 0
    tp = fp = fn = 0
    final_template_correct = 0
    destination_correct_detected = 0

    parts = manifest["parts"]["test"]

    for i, rel in enumerate(parts, 1):
        path = frozen / rel

        print(
            f"[TEST {i}/{len(parts)}] {path.name}",
            flush=True,
        )

        d = pd.read_pickle(path)

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

        tctx = template_context(d, template_bundle)

        origin = np.asarray(
            TEMPLATES,
            dtype=object,
        )[tctx["origin"]]

        true_template = norm(
            d["y_template_id"]
        ).to_numpy(object)

        destination = tctx["destination_template"]
        p_switch = tctx["switch_probability"].astype(float)

        true_switch = true_template != origin
        pred_switch = p_switch >= threshold

        final_template = origin.copy()
        final_template[pred_switch] = destination[pred_switch]

        theta_p = raw_theta(d, hist=True)
        theta_h = theta_p.copy()

        if pred_switch.any():
            ds = (
                d.loc[pred_switch]
                .copy()
                .reset_index(drop=True)
            )

            theta_s = predict_switch_theta(
                ds,
                origin[pred_switch],
                destination[pred_switch],
                theta_bundle,
            )

            theta_h[pred_switch, :] = theta_s

            update_state(
                states["persistence_on_predicted_switch_rows"],
                ds,
                theta_p[pred_switch],
                origin[pred_switch],
                centers,
            )

            update_state(
                states["switch_theta_on_predicted_switch_rows"],
                ds,
                theta_s,
                destination[pred_switch],
                centers,
            )

        # overall
        update_state(
            states["persistence_origin_template"],
            d,
            theta_p,
            origin,
            centers,
        )

        update_state(
            states["persistence_oracle_template"],
            d,
            theta_p,
            true_template,
            centers,
        )

        update_state(
            states["hybrid_full_pipeline"],
            d,
            theta_h,
            final_template,
            centers,
        )

        update_state(
            states["hybrid_oracle_template"],
            d,
            theta_h,
            true_template,
            centers,
        )

        # true switch subset
        if true_switch.any():
            ds_true = (
                d.loc[true_switch]
                .copy()
                .reset_index(drop=True)
            )

            update_state(
                states["persistence_on_true_switch_rows"],
                ds_true,
                theta_p[true_switch],
                origin[true_switch],
                centers,
            )

            update_state(
                states["hybrid_on_true_switch_rows"],
                ds_true,
                theta_h[true_switch],
                final_template[true_switch],
                centers,
            )

            # Clean conditional-regression attribution on ALL true switches:
            # same regressor, re-evaluated under predicted vs oracle destination.
            theta_true_switch_pred_dest = predict_switch_theta(
                ds_true,
                origin[true_switch],
                destination[true_switch],
                theta_bundle,
            )

            theta_true_switch_oracle_dest = predict_switch_theta(
                ds_true,
                origin[true_switch],
                true_template[true_switch],
                theta_bundle,
            )

            update_state(
                states[
                    "pred_destination_conditional_theta_on_true_switch_rows"
                ],
                ds_true,
                theta_true_switch_pred_dest,
                destination[true_switch],
                centers,
            )

            update_state(
                states[
                    "oracle_destination_conditional_theta_on_true_switch_rows"
                ],
                ds_true,
                theta_true_switch_oracle_dest,
                true_template[true_switch],
                centers,
            )

        unchanged = ~true_switch

        if unchanged.any():
            du = (
                d.loc[unchanged]
                .copy()
                .reset_index(drop=True)
            )

            update_state(
                states["persistence_on_true_unchanged_rows"],
                du,
                theta_p[unchanged],
                origin[unchanged],
                centers,
            )

            update_state(
                states["hybrid_on_true_unchanged_rows"],
                du,
                theta_h[unchanged],
                final_template[unchanged],
                centers,
            )

        total_rows += len(d)
        true_switch_rows += int(true_switch.sum())
        pred_switch_rows += int(pred_switch.sum())

        tp_part = int(
            np.sum(
                true_switch
                & pred_switch
            )
        )
        fp_part = int(
            np.sum(
                (~true_switch)
                & pred_switch
            )
        )
        fn_part = int(
            np.sum(
                true_switch
                & (~pred_switch)
            )
        )

        tp += tp_part
        fp += fp_part
        fn += fn_part

        final_template_correct += int(
            np.sum(
                final_template
                == true_template
            )
        )

        destination_correct_detected += int(
            np.sum(
                true_switch
                & pred_switch
                & (
                    destination
                    == true_template
                )
            )
        )

        del d, tctx, theta_p, theta_h
        gc.collect()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    switch_diag = {
        "selected_threshold_from_validation": float(threshold),
        "rows": int(total_rows),
        "dropped_invalid_hist_template_rows": int(
            dropped_invalid_hist_template
        ),
        "true_switch_rows": int(true_switch_rows),
        "pred_switch_rows": int(pred_switch_rows),
        "true_switch_share": float(
            true_switch_rows / max(total_rows, 1)
        ),
        "pred_switch_share": float(
            pred_switch_rows / max(total_rows, 1)
        ),
        "switch_precision": float(precision),
        "switch_recall": float(recall),
        "switch_f1": float(
            2.0
            * precision
            * recall
            / max(
                precision + recall,
                1e-12,
            )
        ),
        "final_template_accuracy": float(
            final_template_correct
            / max(total_rows, 1)
        ),
        "destination_accuracy_when_true_switch_detected": float(
            destination_correct_detected
            / max(tp, 1)
        ),
    }

    rows = []

    for mode, state in states.items():
        row = finalize_state(state)
        row["mode"] = mode
        rows.append(row)

    return pd.DataFrame(rows), switch_diag


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
        "--bidtemplate-root",
        default="data/processed/bidtemplate",
    )

    ap.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Comma-separated validation threshold grid. "
            "Default = 0.20..0.95 by 0.01."
        ),
    )

    args = ap.parse_args()

    thresholds = parse_thresholds(args.thresholds)

    base = Path(args.root) / str(args.year)
    frozen = base / "frozen_stable_theta_continuous_dataset"

    manifest = json.loads(
        (frozen / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    theta_bundle = joblib.load(
        base
        / "conditional_theta_model_v3"
        / "conditional_theta_model_v3.joblib"
    )

    template_bundle = joblib.load(
        base
        / "final_template_predictor"
        / "final_template_model.joblib"
    )

    centers, center_file = discover_centers(
        Path(args.bidtemplate_root)
        / str(args.year)
    )

    out = base / "conditional_theta_operational_calibration_v3"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(
        f"Conditional-theta operational calibration v3 - {args.year}"
    )
    print("=" * 80)
    print(
        f"Validation thresholds: "
        f"{thresholds[0]:.2f} .. {thresholds[-1]:.2f} "
        f"({len(thresholds)} values)"
    )
    print()

    # -----------------------------------------------------------------
    # Validation calibration.
    # -----------------------------------------------------------------
    val_data = collect_validation_branch_errors(
        frozen,
        manifest,
        template_bundle,
        theta_bundle,
        centers,
        thresholds,
    )

    sweep = threshold_sweep_from_branch_errors(
        val_data,
        thresholds,
    )

    sweep = sweep.sort_values(
        [
            "price_wape_pct",
            "threshold",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    selected_threshold = float(
        sweep.iloc[0]["threshold"]
    )

    sweep.to_csv(
        out / "validation_threshold_curve_wape.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_row = sweep.iloc[0].to_dict()

    (out / "selected_threshold.json").write_text(
        json.dumps(
            {
                "selected_threshold": selected_threshold,
                "selection_split": "validation",
                "selection_metric": "final reconstructed price WAPE",
                "theta_model": (
                    "destination-conditioned regression trained with true "
                    "destination and deployed with predicted destination"
                ),
                "selected_validation_metrics": selected_row,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Validation best threshold:")
    print(
        pd.DataFrame(
            [selected_row]
        ).to_string(
            index=False
        )
    )
    print()

    # -----------------------------------------------------------------
    # Test exactly once with frozen validation-selected threshold.
    # -----------------------------------------------------------------
    test_metrics, test_switch_diag = evaluate_test(
        frozen,
        manifest,
        template_bundle,
        theta_bundle,
        centers,
        selected_threshold,
    )

    test_metrics.to_csv(
        out / "test_reconstruction_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (out / "test_switch_diagnostics.json").write_text(
        json.dumps(
            test_switch_diag,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            (
                f"Conditional-theta operational calibration v3 - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            f"Template center file: {center_file}",
            (
                f"Selected threshold from VALIDATION = "
                f"{selected_threshold:.3f}"
            ),
            (
                "Selection metric = final reconstructed "
                "price-curve WAPE"
            ),
            "",
            "Selected validation row:",
            pd.DataFrame(
                [selected_row]
            ).to_string(
                index=False
            ),
            "",
            "TEST switch diagnostics:",
            json.dumps(
                test_switch_diag,
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "TEST reconstruction metrics:",
            test_metrics.to_string(
                index=False
            ),
        ]
    )

    (out / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
