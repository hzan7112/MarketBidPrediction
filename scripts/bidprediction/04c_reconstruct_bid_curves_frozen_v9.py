#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04c_reconstruct_bid_curves_frozen.py

Final bid-curve reconstruction on the frozen TEST set using the v8 PCA-residual price representation.

Quantity evaluation policy
--------------------------
The 21 Stage2 shape points are sampled on a UNIFORM normalized physical-
quantity grid. The reconstructed theta curve uses q1..q5 to warp its internal
template coordinate. Therefore pointwise

    abs(q_pred[j] - q_true[j])

does not compare like-for-like points and must not be reported as a quantity
prediction error.

This version evaluates quantity with physically interpretable theta quantities:
    q_anchor_mae_mw = |q_base_pred - q_base_true|
    q_span_mae_mw   = |q_span_pred - q_span_true|
    q_max_mae_mw    = |(q_base+q_span)_pred - (...)_true|
    q_share_mae      = mean_k |qk_pred - qk_true|

Price is evaluated on the TRUE physical quantity grid by interpolating the
predicted curve onto q_true.

Final parameter prediction policy:
    - one 04b bundle predicts all theta parameters;
    - price / quantity-scale / quantity-shape may use different model families;
    - oracle-template evaluation conditions the parameter bundle on TRUE template;
    - full-pipeline evaluation conditions it on PREDICTED template.

No participant-specific or dataset-specific rule is used in reconstruction.

Run:
python scripts/bidprediction/04c_reconstruct_bid_curves_frozen.py --year 2025
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
U_KNOTS = np.linspace(0.0, 1.0, 6)
TAIL_START = 0.70

TEMPLATES = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
T2I = {t: i for i, t in enumerate(TEMPLATES)}
I2T = {i: t for t, i in T2I.items()}
N_TEMPLATE = len(TEMPLATES)

MODE2I = {
    "flat": 0,
    "block": 1,
    "sloped": 2,
}

RAW_THETA = [
    "theta_p_base",
    "theta_alpha",
    "theta_beta",
    "theta_q_base_mw",
    "theta_q_span_mw",
    "theta_q1",
    "theta_q2",
    "theta_q3",
    "theta_q4",
    "theta_q5",
]

TARGET_TEMPLATE = "y_template_id"
ORIGIN_TEMPLATE = "hist_lag1_template_id"


def norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def uniq(xs):
    return list(dict.fromkeys(xs))


def signed_expm1(z):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.expm1(np.abs(z))


def target_p_baseline(d, spec):
    col = spec.get("p_baseline_column")

    if not col:
        return np.zeros(len(d), dtype=float)

    x = num(d[col]).to_numpy(float)
    fill = float(spec.get("p_baseline_fill", 0.0))

    return np.where(
        np.isfinite(x),
        x,
        fill,
    )


def target_q_scale(d, spec):
    col = spec.get("q_scale_column")

    if not col:
        return np.ones(len(d), dtype=float)

    x = num(d[col]).to_numpy(float)
    fill = float(spec.get("q_scale_fill", 1.0))
    floor = float(spec.get("q_scale_floor", 1.0))

    x = np.where(
        np.isfinite(x) & (x > floor),
        x,
        fill,
    )

    return np.maximum(x, floor)


def load_manifest(frozen: Path):
    p = frozen / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p}\nRun 04a3_freeze_modeling_dataset.py first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def shape_cols(header):
    exact = [f"shape_v{i:02d}" for i in range(21)]
    return exact if all(c in header for c in exact) else None


def normalize_template_id(v):
    s = str(v).strip()

    if s.upper() == "FLAT":
        return "FLAT"

    if s.upper().startswith("T"):
        s = s[1:]

    try:
        i = int(float(s))
        if 0 <= i <= 11:
            return f"T{i:02d}"
    except Exception:
        pass

    return None


def discover_centers(year_dir: Path):
    preferred = (
        year_dir
        / "template_library"
        / "curve_template_library.csv"
    )

    candidates = [preferred] if preferred.exists() else []

    if not candidates:
        candidates = sorted(year_dir.rglob("*.csv"))

    for p in candidates:
        try:
            h = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue

        sc = shape_cols(h)
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
                if c in h
            ),
            None,
        )

        if sc is None or id_col is None:
            continue

        d = pd.read_csv(
            p,
            usecols=[id_col] + sc,
        )

        if len(d) > 100:
            continue

        centers = {}

        for _, r in d.iterrows():
            t = normalize_template_id(r[id_col])
            a = pd.to_numeric(
                r[sc],
                errors="coerce",
            ).to_numpy(float)

            if t and np.isfinite(a).all():
                centers[t] = a

        centers["FLAT"] = np.zeros(21)

        if all(t in centers for t in TEMPLATES):
            return centers, p

    raise FileNotFoundError(
        "Cannot locate Stage2 curve_template_library.csv"
    )


# ---------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------

def enc_col(s, col):
    # Needed by the currently frozen final-template predictor.
    if col in {
        "hist_lag1_template_id",
        "hist30_dominant_template_id",
    }:
        return (
            norm(s)
            .map(T2I)
            .astype("float32")
        )

    if col == "hist_lag1_curve_mode":
        return (
            norm(s)
            .str.lower()
            .map(MODE2I)
            .astype("float32")
        )

    return num(s).astype("float32")


def prepare_plain_X(d, features):
    return pd.DataFrame(
        {
            c: enc_col(d[c], c)
            for c in features
        },
        index=d.index,
    )


def prepare_parameter_X(
    d,
    features,
    template,
):
    out = {
        c: enc_col(d[c], c)
        for c in features
    }

    t = norm(template).map(T2I)

    if t.isna().any():
        bad = norm(template)[t.isna()].unique().tolist()
        raise ValueError(
            f"Unknown template condition: {bad[:20]}"
        )

    a = t.to_numpy(np.int16)

    for j, z in enumerate(TEMPLATES):
        out[f"cond_template_{z}"] = (
            a == j
        ).astype("float32")

    return pd.DataFrame(
        out,
        index=d.index,
    )


# ---------------------------------------------------------------------
# Final template predictor
# ---------------------------------------------------------------------

def expand_proba(model, raw_p):
    classes = np.asarray(
        model.classes_,
        dtype=np.int16,
    )

    full = np.zeros(
        (len(raw_p), N_TEMPLATE),
        dtype=float,
    )
    full[:, classes] = raw_p
    return full


def mask_destination_proba(
    p,
    origin,
    allowed,
):
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

    return np.argmax(
        p,
        axis=1,
    ).astype(np.int16)


def predict_template(
    d,
    bundle,
):
    switch_features = bundle["switch_features"]
    destination_features = bundle["destination_features"]

    required = uniq(
        switch_features
        + destination_features
        + [ORIGIN_TEMPLATE]
    )

    missing = [
        c for c in required
        if c not in d.columns
    ]

    if missing:
        raise KeyError(
            f"Frozen test set is missing final-template features: {missing}"
        )

    origin_s = norm(
        d[ORIGIN_TEMPLATE]
    ).map(T2I)

    if origin_s.isna().any():
        raise ValueError(
            "Final template predictor requires valid hist_lag1_template_id."
        )

    origin = origin_s.to_numpy(np.int16)

    Xs = bundle[
        "switch_imputer"
    ].transform(
        prepare_plain_X(
            d,
            switch_features,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    sw_model = bundle["switch_model"]
    p_raw = sw_model.predict_proba(Xs)
    classes = list(sw_model.classes_)

    p_switch = (
        p_raw[:, classes.index(1)]
        if 1 in classes
        else np.zeros(len(d))
    )

    Xd = bundle[
        "destination_imputer"
    ].transform(
        prepare_plain_X(
            d,
            destination_features,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    Xdg = np.column_stack(
        [
            Xd,
            origin.astype(np.float32),
        ]
    )

    dest_model = bundle["destination_model"]

    full_p = expand_proba(
        dest_model,
        dest_model.predict_proba(Xdg),
    )

    dest_pred = mask_destination_proba(
        full_p,
        origin,
        np.asarray(
            bundle["allowed_destinations_train"],
            dtype=bool,
        ),
    )

    final = origin.copy()

    override = (
        p_switch
        >= float(bundle["hierarchy_threshold"])
    )

    final[override] = dest_pred[override]

    names = np.asarray(
        [
            I2T[int(v)]
            for v in final
        ],
        dtype=object,
    )

    return names, p_switch


# ---------------------------------------------------------------------
# Parameter model inverse transform
# ---------------------------------------------------------------------

def latent_to_raw(
    z,
    clip_lo,
    clip_hi,
    d=None,
    target_transform=None,
):
    z = np.asarray(z, dtype=float)

    z = np.clip(
        z,
        np.asarray(clip_lo)[None, :],
        np.asarray(clip_hi)[None, :],
    )

    out = np.empty(
        (len(z), 10),
        dtype=float,
    )

    spec = target_transform or {
        "p_base_mode": "absolute",
        "q_scale_mode": "absolute",
        "p_baseline_column": None,
        "p_baseline_fill": 0.0,
        "q_scale_column": None,
        "q_scale_fill": 1.0,
        "q_scale_floor": 1.0,
    }

    if d is None:
        pbase = np.zeros(len(z), dtype=float)
        qscale = np.ones(len(z), dtype=float)
    else:
        pbase = target_p_baseline(d, spec)
        qscale = target_q_scale(d, spec)

    out[:, 0] = z[:, 0] + pbase

    out[:, 1] = np.maximum(
        np.expm1(z[:, 1]),
        0.0,
    )

    out[:, 2] = np.maximum(
        np.expm1(z[:, 2]),
        0.0,
    )

    out[:, 3] = (
        signed_expm1(z[:, 3])
        * qscale
    )

    out[:, 4] = (
        np.maximum(
            np.expm1(z[:, 4]),
            1e-8,
        )
        * qscale
    )

    logits = np.c_[
        z[:, 5:9],
        np.zeros(len(z)),
    ]

    logits -= logits.max(
        axis=1,
        keepdims=True,
    )

    e = np.exp(logits)

    out[:, 5:10] = (
        e
        / e.sum(axis=1, keepdims=True)
    )

    return out


def predict_theta(
    bundle,
    model_name,
    d,
    template,
):
    """
    Predict theta from the final 04b bundle.

    Supports:
      - legacy single-model bundles;
      - v7 group-specialist bundles, where price / q-scale / q-shape
        independently use the validation-selected model family.
    """
    X = bundle[
        "imputer"
    ].transform(
        prepare_parameter_X(
            d,
            bundle["features"],
            template,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    Xs = bundle[
        "x_scaler"
    ].transform(
        X
    ).astype(
        np.float32,
        copy=False,
    )

    if (
        bundle.get("model_mode")
        == "group_specialist"
    ):
        latent_targets = bundle[
            "latent_targets"
        ]

        Z = np.zeros(
            (
                len(d),
                len(latent_targets),
            ),
            dtype=float,
        )

        for group, info in bundle[
            "group_models"
        ].items():
            candidate = str(
                info["candidate"]
            )

            Xin = (
                Xs
                if bool(
                    info.get(
                        "uses_scaled_x",
                        False,
                    )
                )
                else X
            )

            pred = np.asarray(
                info["model"].predict(
                    Xin
                ),
                dtype=float,
            )

            if pred.ndim == 1:
                pred = pred[:, None]

            Z[
                :,
                info["indices"],
            ] = pred

        z = (
            Z
            * np.asarray(
                bundle[
                    "latent_std"
                ]
            )[None, :]
            + np.asarray(
                bundle[
                    "latent_mean"
                ]
            )[None, :]
        )

    else:
        Xin = (
            Xs
            if model_name in {
                "LinearRegression",
                "Ridge",
            }
            else X
        )

        z = np.asarray(
            bundle[
                "models"
            ][
                model_name
            ].predict(
                Xin
            ),
            dtype=float,
        )

        z = (
            z
            * np.asarray(
                bundle[
                    "latent_std"
                ]
            )[None, :]
            + np.asarray(
                bundle[
                    "latent_mean"
                ]
            )[None, :]
        )

    theta = latent_to_raw(
        z,
        bundle["latent_clip_lo"],
        bundle["latent_clip_hi"],
        d=d,
        target_transform=bundle.get(
            "target_transform"
        ),
    )

    flat = norm(
        template
    ).eq(
        "FLAT"
    ).to_numpy()

    theta[flat, 1] = 0.0
    theta[flat, 2] = 0.0

    return theta



def predict_pbase_specialist(
    bundle,
    d,
    template,
):
    """
    Predict theta_p_base with the final 04g specialist.

    The specialist was trained on:
        residual = theta_p_base - lt_bid_level

    Its template condition MUST match the scenario being evaluated:
        oracle-template mode -> true template
        full pipeline        -> predicted template
    """
    required = [
        "features",
        "imputer",
        "model",
        "baseline_state",
        "target_mean",
        "target_std",
    ]

    missing = [
        k for k in required
        if k not in bundle
    ]

    if missing:
        raise KeyError(
            f"Invalid 04g p_base specialist bundle; missing: {missing}"
        )

    if bundle.get(
        "history_features_used",
        False,
    ):
        raise RuntimeError(
            "04g p_base specialist must not use H/direct bid-history features."
        )

    X = bundle[
        "imputer"
    ].transform(
        prepare_parameter_X(
            d,
            bundle["features"],
            template,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    z = np.asarray(
        bundle["model"].predict(X),
        dtype=float,
    )

    residual = (
        z * float(bundle["target_std"])
        + float(bundle["target_mean"])
    )

    state = bundle["baseline_state"]
    col = state.get(
        "baseline_column",
        "lt_bid_level",
    )
    fill = float(
        state.get(
            "baseline_fill",
            0.0,
        )
    )

    baseline = num(
        d[col]
    ).to_numpy(float)

    baseline = np.where(
        np.isfinite(baseline),
        baseline,
        fill,
    )

    return baseline + residual



# ---------------------------------------------------------------------
# v8 PCA-residual price + quantity specialist prediction
# ---------------------------------------------------------------------

def v8_p_anchor_baseline(
    d,
    state,
):
    if (
        state["anchor_mode"]
        == "absolute"
    ):
        return np.zeros(
            len(d),
            dtype=float,
        )

    x = num(
        d["lt_bid_level"]
    ).to_numpy(float)

    fill = float(
        state["baseline_fill"]
    )

    return np.where(
        np.isfinite(x),
        x,
        fill,
    )


def v8_q_scale_value(
    d,
    spec,
):
    col = spec.get(
        "q_scale_column"
    )

    if not col:
        return np.ones(
            len(d),
            dtype=float,
        )

    x = num(
        d[col]
    ).to_numpy(float)

    fill = float(
        spec.get(
            "q_scale_fill",
            1.0,
        )
    )

    floor = float(
        spec.get(
            "q_scale_floor",
            1.0,
        )
    )

    x = np.where(
        np.isfinite(x)
        & (x > floor),
        x,
        fill,
    )

    return np.maximum(
        x,
        floor,
    )


def v8_predict_standardized(
    model_info,
    X,
    Xs,
):
    Xin = (
        Xs
        if bool(
            model_info.get(
                "uses_scaled_x",
                False,
            )
        )
        else X
    )

    z = np.asarray(
        model_info[
            "model"
        ].predict(
            Xin
        ),
        dtype=float,
    )

    if z.ndim == 1:
        z = z[:, None]

    return z


def v8_reconstruct_price(
    bundle,
    d,
    template,
    standardized_pred,
    centers,
):
    state = bundle[
        "price_target_state"
    ]

    basis = bundle[
        "price_basis"
    ]

    Y = (
        np.asarray(
            standardized_pred,
            dtype=float,
        )
        * np.asarray(
            state[
                "target_std"
            ],
            dtype=float,
        )[None, :]
        + np.asarray(
            state[
                "target_mean"
            ],
            dtype=float,
        )[None, :]
    )

    Y = np.clip(
        Y,
        np.asarray(
            state[
                "clip_lo"
            ],
            dtype=float,
        )[None, :],
        np.asarray(
            state[
                "clip_hi"
            ],
            dtype=float,
        )[None, :],
    )

    p_anchor = (
        Y[:, 0]
        + v8_p_anchor_baseline(
            d,
            state,
        )
    )

    p_span = signed_expm1(
        Y[:, 1]
    )

    coeff = Y[:, 2:]

    residual = (
        np.asarray(
            basis[
                "mean"
            ],
            dtype=float,
        )[None, :]
        + coeff
        @ np.asarray(
            basis[
                "components"
            ],
            dtype=float,
        )
    )

    ts = (
        norm(template)
        if isinstance(
            template,
            pd.Series,
        )
        else pd.Series(
            template,
            index=d.index,
            dtype="string",
        )
    )

    tid = ts.to_numpy(object)

    C = np.vstack(
        [
            centers[str(t)]
            for t in tid
        ]
    )

    shape = (
        C + residual
    )

    flat = (
        tid == "FLAT"
    )

    shape[
        flat
    ] = 0.0

    p_span[
        flat
    ] = 0.0

    nonflat = ~flat

    if nonflat.any():
        shape[
            nonflat,
            0,
        ] = 0.0

        shape[
            nonflat,
            -1,
        ] = 1.0

    p = (
        p_anchor[:, None]
        + p_span[:, None]
        * shape
    )

    return {
        "p_anchor": p_anchor,
        "p_span": p_span,
        "shape": shape,
        "price": p,
    }


def v8_inverse_quantity(
    bundle,
    d,
    z_scale,
    z_shape,
):
    state = bundle[
        "q_state"
    ]

    spec = bundle[
        "q_spec"
    ]

    Z = np.zeros(
        (
            len(d),
            6,
        ),
        dtype=float,
    )

    Z[
        :,
        :2,
    ] = z_scale

    Z[
        :,
        2:6,
    ] = z_shape

    Y = (
        Z
        * np.asarray(
            state[
                "std"
            ],
            dtype=float,
        )[None, :]
        + np.asarray(
            state[
                "mean"
            ],
            dtype=float,
        )[None, :]
    )

    Y = np.clip(
        Y,
        np.asarray(
            state[
                "clip_lo"
            ],
            dtype=float,
        )[None, :],
        np.asarray(
            state[
                "clip_hi"
            ],
            dtype=float,
        )[None, :],
    )

    scale = v8_q_scale_value(
        d,
        spec,
    )

    q_base = (
        signed_expm1(
            Y[:, 0]
        )
        * scale
    )

    q_span = (
        np.maximum(
            np.expm1(
                Y[:, 1]
            ),
            1e-8,
        )
        * scale
    )

    L = np.c_[
        Y[:, 2:6],
        np.zeros(
            len(Y)
        ),
    ]

    L -= L.max(
        axis=1,
        keepdims=True,
    )

    E = np.exp(L)

    q_share = (
        E
        / E.sum(
            axis=1,
            keepdims=True,
        )
    )

    return (
        q_base,
        q_span,
        q_share,
    )


def v8_theta_for_metrics(
    q_base,
    q_span,
    q_share,
):
    theta = np.zeros(
        (
            len(q_base),
            10,
        ),
        dtype=float,
    )

    theta[:, 3] = q_base
    theta[:, 4] = q_span
    theta[:, 5:10] = q_share

    return theta


def v8_predict_parameters(
    bundle,
    d,
    template,
    centers,
):
    X = bundle[
        "imputer"
    ].transform(
        prepare_parameter_X(
            d,
            bundle[
                "features"
            ],
            template,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    Xs = bundle[
        "x_scaler"
    ].transform(
        X
    ).astype(
        np.float32,
        copy=False,
    )

    z_price = v8_predict_standardized(
        bundle[
            "price_model"
        ],
        X,
        Xs,
    )

    price = v8_reconstruct_price(
        bundle,
        d,
        template,
        z_price,
        centers,
    )

    z_qscale = v8_predict_standardized(
        bundle[
            "q_scale_model"
        ],
        X,
        Xs,
    )

    z_qshape = v8_predict_standardized(
        bundle[
            "q_shape_model"
        ],
        X,
        Xs,
    )

    (
        q_base,
        q_span,
        q_share,
    ) = v8_inverse_quantity(
        bundle,
        d,
        z_qscale,
        z_qshape,
    )

    # Stage2 observed price shape is defined on a UNIFORM physical-q grid.
    # q1..q5 are evaluated as segment/breakpoint quantities separately.
    q_curve = (
        q_base[:, None]
        + q_span[:, None]
        * GRID[None, :]
    )

    theta = v8_theta_for_metrics(
        q_base,
        q_span,
        q_share,
    )

    return {
        "q": q_curve,
        "p": price[
            "price"
        ],
        "theta": theta,
        "price": price,
    }


def v8_oracle_representation(
    bundle,
    d,
    template,
    centers,
):
    pa = num(
        d[
            "p_anchor"
        ]
    ).to_numpy(float)

    ps = num(
        d[
            "p_span"
        ]
    ).to_numpy(float)

    S = d[
        [
            f"shape_v{i:02d}"
            for i in range(21)
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)

    ts = norm(
        template
    )

    tid = ts.to_numpy(object)

    flat = (
        tid == "FLAT"
    )

    S[
        flat
    ] = 0.0

    C = np.vstack(
        [
            centers[str(t)]
            for t in tid
        ]
    )

    R = (
        S - C
    )

    R[
        flat
    ] = 0.0

    basis = bundle[
        "price_basis"
    ]

    mean = np.asarray(
        basis[
            "mean"
        ],
        dtype=float,
    )

    comp = np.asarray(
        basis[
            "components"
        ],
        dtype=float,
    )

    coeff = (
        R - mean[None, :]
    ) @ comp.T

    R_hat = (
        mean[None, :]
        + coeff @ comp
    )

    S_hat = (
        C + R_hat
    )

    S_hat[
        flat
    ] = 0.0

    nonflat = ~flat

    if nonflat.any():
        S_hat[
            nonflat,
            0,
        ] = 0.0

        S_hat[
            nonflat,
            -1,
        ] = 1.0

    ps_oracle = ps.copy()

    ps_oracle[
        flat
    ] = 0.0

    p_curve = (
        pa[:, None]
        + ps_oracle[:, None]
        * S_hat
    )

    q_base = num(
        d[
            "theta_q_base_mw"
        ]
    ).to_numpy(float)

    q_span = num(
        d[
            "theta_q_span_mw"
        ]
    ).to_numpy(float)

    Q = d[
        [
            "theta_q1",
            "theta_q2",
            "theta_q3",
            "theta_q4",
            "theta_q5",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)

    Q = np.maximum(
        Q,
        1e-8,
    )

    Q /= Q.sum(
        axis=1,
        keepdims=True,
    )

    q_curve = (
        q_base[:, None]
        + q_span[:, None]
        * GRID[None, :]
    )

    theta = v8_theta_for_metrics(
        q_base,
        q_span,
        Q,
    )

    return {
        "q": q_curve,
        "p": p_curve,
        "theta": theta,
        "price": {
            "p_anchor": pa,
            "p_span": ps_oracle,
            "shape": S_hat,
        },
    }



# ---------------------------------------------------------------------
# v9 direct 21-point price prediction helpers
# ---------------------------------------------------------------------

def v9_predict_standardized(
    model_info,
    X,
    Xs,
):
    Xin = (
        Xs
        if bool(
            model_info.get(
                "uses_scaled_x",
                False,
            )
        )
        else X
    )

    z = np.asarray(
        model_info[
            "model"
        ].predict(
            Xin
        ),
        dtype=float,
    )

    if z.ndim == 1:
        z = z[:, None]

    return z


def v9_q_scale_value(
    d,
    spec,
):
    col = spec.get(
        "q_scale_column"
    )

    if not col:
        return np.ones(
            len(d),
            dtype=float,
        )

    x = num(
        d[col]
    ).to_numpy(float)

    fill = float(
        spec.get(
            "q_scale_fill",
            1.0,
        )
    )

    floor = float(
        spec.get(
            "q_scale_floor",
            1.0,
        )
    )

    x = np.where(
        np.isfinite(x)
        & (x > floor),
        x,
        fill,
    )

    return np.maximum(
        x,
        floor,
    )


def v9_inverse_quantity(
    bundle,
    d,
    z_scale,
    z_shape,
):
    state = bundle[
        "q_state"
    ]

    spec = bundle[
        "q_spec"
    ]

    Z = np.zeros(
        (
            len(d),
            6,
        ),
        dtype=float,
    )

    Z[:, :2] = z_scale
    Z[:, 2:6] = z_shape

    Y = (
        Z
        * np.asarray(
            state[
                "std"
            ],
            dtype=float,
        )[None, :]
        + np.asarray(
            state[
                "mean"
            ],
            dtype=float,
        )[None, :]
    )

    scale = v9_q_scale_value(
        d,
        spec,
    )

    q_base = (
        signed_expm1(
            Y[:, 0]
        )
        * scale
    )

    q_span = (
        np.maximum(
            np.expm1(
                Y[:, 1]
            ),
            1e-8,
        )
        * scale
    )

    L = np.c_[
        Y[:, 2:6],
        np.zeros(
            len(Y)
        ),
    ]

    L -= L.max(
        axis=1,
        keepdims=True,
    )

    E = np.exp(L)

    q_share = (
        E
        / E.sum(
            axis=1,
            keepdims=True,
        )
    )

    return (
        q_base,
        q_span,
        q_share,
    )


def v9_theta_for_metrics(
    q_base,
    q_span,
    q_share,
):
    theta = np.zeros(
        (
            len(q_base),
            10,
        ),
        dtype=float,
    )

    theta[:, 3] = q_base
    theta[:, 4] = q_span
    theta[:, 5:10] = q_share

    return theta


def v9_reconstruct_price(
    bundle,
    d,
    standardized_pred,
):
    state = bundle[
        "price_state"
    ]

    residual = (
        np.asarray(
            standardized_pred,
            dtype=float,
        )
        * np.asarray(
            state[
                "target_std"
            ],
            dtype=float,
        )[None, :]
        + np.asarray(
            state[
                "target_mean"
            ],
            dtype=float,
        )[None, :]
    )

    baseline = num(
        d[
            state[
                "baseline_column"
            ]
        ]
    ).to_numpy(float)

    baseline = np.where(
        np.isfinite(
            baseline
        ),
        baseline,
        float(
            state[
                "baseline_fill"
            ]
        ),
    )

    return (
        baseline[:, None]
        + residual
    )


def v9_predict_curve(
    bundle,
    d,
    template,
):
    X = bundle[
        "imputer"
    ].transform(
        prepare_parameter_X(
            d,
            bundle[
                "features"
            ],
            template,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    Xs = bundle[
        "x_scaler"
    ].transform(
        X
    ).astype(
        np.float32,
        copy=False,
    )

    z_price = v9_predict_standardized(
        bundle[
            "price_model"
        ],
        X,
        Xs,
    )

    price = v9_reconstruct_price(
        bundle,
        d,
        z_price,
    )

    z_qscale = v9_predict_standardized(
        bundle[
            "q_scale_model"
        ],
        X,
        Xs,
    )

    z_qshape = v9_predict_standardized(
        bundle[
            "q_shape_model"
        ],
        X,
        Xs,
    )

    (
        q_base,
        q_span,
        q_share,
    ) = v9_inverse_quantity(
        bundle,
        d,
        z_qscale,
        z_qshape,
    )

    q_curve = (
        q_base[:, None]
        + q_span[:, None]
        * GRID[None, :]
    )

    theta = v9_theta_for_metrics(
        q_base,
        q_span,
        q_share,
    )

    return {
        "q": q_curve,
        "p": price,
        "theta": theta,
    }


# ---------------------------------------------------------------------
# Curve reconstruction
# ---------------------------------------------------------------------

def tail_basis(u):
    z = np.maximum(
        0.0,
        (u - TAIL_START)
        / (1.0 - TAIL_START),
    )
    return z * z


TAIL = tail_basis(GRID)


def reconstruct(
    theta,
    templates,
    centers,
):
    theta = np.asarray(theta, dtype=float)
    n = len(theta)

    Q = np.maximum(
        theta[:, 5:10],
        1e-10,
    )
    Q /= Q.sum(
        axis=1,
        keepdims=True,
    )

    cum = np.c_[
        np.zeros(n),
        np.cumsum(
            Q,
            axis=1,
        ),
    ]
    cum[:, -1] = 1.0

    x = np.empty((n, 21), dtype=float)

    for j, u in enumerate(GRID):
        k = min(
            int(np.floor(u * 5)),
            4,
        )

        frac = (
            u - U_KNOTS[k]
        ) / (
            U_KNOTS[k + 1]
            - U_KNOTS[k]
        )

        frac = float(
            np.clip(frac, 0.0, 1.0)
        )

        x[:, j] = (
            cum[:, k]
            + frac * Q[:, k]
        )

    x[:, 0] = 0.0
    x[:, -1] = 1.0

    q = (
        theta[:, 3, None]
        + theta[:, 4, None] * x
    )

    C = np.vstack(
        [
            centers[str(t)]
            for t in templates
        ]
    )

    p = (
        theta[:, 0, None]
        + theta[:, 1, None] * C
        + theta[:, 2, None] * TAIL[None, :]
    )

    return q, p


def actual_curve(d):
    qa = num(
        d["q_anchor_mw"]
    ).to_numpy(float)

    qs = num(
        d["q_span_mw"]
    ).to_numpy(float)

    pa = num(
        d["p_anchor"]
    ).to_numpy(float)

    ps = num(
        d["p_span"]
    ).to_numpy(float)

    sc = [
        f"shape_v{i:02d}"
        for i in range(21)
    ]

    S = d[
        sc
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)

    tid = norm(
        d[TARGET_TEMPLATE]
    ).to_numpy(object)

    # Stage2 FLAT curves intentionally carry no normalized shape.
    S[tid == "FLAT"] = 0.0

    valid = (
        np.isfinite(S).all(axis=1)
        & np.isfinite(qa)
        & np.isfinite(qs)
        & np.isfinite(pa)
        & np.isfinite(ps)
        & (qs > 0)
    )

    # Stage2 observed shape_v00..20 is defined on uniform physical q.
    q = (
        qa[:, None]
        + qs[:, None] * GRID
    )

    p = (
        pa[:, None]
        + ps[:, None] * S
    )

    return valid, q, p, tid


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

METRIC_EPS = 1e-8


def interp_uniform_rows(values, frac):
    """Row-wise linear interpolation on GRID=[0,0.05,...,1]."""
    values = np.asarray(values, dtype=float)
    frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)

    z = frac * (len(GRID) - 1)
    lo = np.floor(z).astype(np.int64)
    hi = np.minimum(lo + 1, len(GRID) - 1)
    w = z - lo

    vlo = np.take_along_axis(values, lo, axis=1)
    vhi = np.take_along_axis(values, hi, axis=1)

    return (1.0 - w) * vlo + w * vhi


class Metrics:
    def __init__(self):
        self.n = 0

        # Price point metrics.
        self.price_abs = 0.0
        self.price_sq = 0.0
        self.price_true_abs = 0.0
        self.price_mape_sum = 0.0
        self.price_mape_count = 0
        self.price_smape_sum = 0.0
        self.price_smape_count = 0

        # Per-curve price MAE.
        self.curve_mae = []

        # Quantity scale.
        self.q_anchor_abs = 0.0
        self.q_span_abs = 0.0
        self.q_max_abs = 0.0
        self.q_anchor_true_abs = 0.0
        self.q_span_true_abs = 0.0
        self.q_max_true_abs = 0.0
        self.q_anchor_mape_sum = 0.0
        self.q_span_mape_sum = 0.0
        self.q_max_mape_sum = 0.0
        self.q_anchor_mape_count = 0
        self.q_span_mape_count = 0
        self.q_max_mape_count = 0

        # Segment / breakpoint.
        self.q_share_abs = 0.0
        self.break_frac_abs = np.zeros(4, dtype=float)
        self.break_mw_abs = np.zeros(4, dtype=float)
        self.break_price_abs = np.zeros(4, dtype=float)
        self.segment_mid_price_abs = np.zeros(5, dtype=float)

        # Normalized price shape.
        self.shape_abs = 0.0
        self.shape_sq = 0.0
        self.shape_count = 0

        self.area_abs = 0.0
        self.template_correct = 0

    def update(
        self,
        q_true,
        p_true,
        q_pred,
        p_pred,
        t_true,
        t_pred,
        theta_true,
        theta_pred,
    ):
        q_true = np.asarray(q_true, dtype=float)
        p_true = np.asarray(p_true, dtype=float)
        q_pred = np.asarray(q_pred, dtype=float)
        p_pred = np.asarray(p_pred, dtype=float)
        theta_true = np.asarray(theta_true, dtype=float)
        theta_pred = np.asarray(theta_pred, dtype=float)

        n = len(q_true)
        if n == 0:
            return

        # -------------------------------------------------------------
        # Interpolate predicted curve onto the TRUE physical q grid.
        # -------------------------------------------------------------
        p_est_all = np.empty_like(p_true, dtype=float)

        for i in range(n):
            order = np.argsort(q_pred[i])
            qp = q_pred[i, order]
            pp = p_pred[i, order]

            uq, idx = np.unique(qp, return_index=True)
            up = pp[idx]

            if len(uq) == 0:
                p_est_all[i] = np.nan
            elif len(uq) == 1:
                p_est_all[i] = up[0]
            else:
                p_est_all[i] = np.interp(
                    q_true[i],
                    uq,
                    up,
                    left=up[0],
                    right=up[-1],
                )

        pe = p_est_all - p_true
        ae = np.abs(pe)

        self.n += n
        self.price_abs += float(np.sum(ae))
        self.price_sq += float(np.sum(pe * pe))
        self.price_true_abs += float(np.sum(np.abs(p_true)))

        # Standard MAPE: only mathematically valid where true price != 0.
        mape_valid = np.abs(p_true) > METRIC_EPS
        if mape_valid.any():
            self.price_mape_sum += float(
                np.sum(ae[mape_valid] / np.abs(p_true[mape_valid]))
            )
            self.price_mape_count += int(mape_valid.sum())

        # sMAPE is retained because electricity prices may be zero/negative.
        smape_den = np.abs(p_true) + np.abs(p_est_all)
        smape_valid = smape_den > METRIC_EPS
        if smape_valid.any():
            self.price_smape_sum += float(
                np.sum(
                    2.0 * ae[smape_valid] / smape_den[smape_valid]
                )
            )
            self.price_smape_count += int(smape_valid.sum())

        self.curve_mae.extend(
            np.mean(ae, axis=1).astype(np.float64).tolist()
        )

        # -------------------------------------------------------------
        # Quantity scale.
        # -------------------------------------------------------------
        qb_t = theta_true[:, 3]
        qs_t = theta_true[:, 4]
        qb_p = theta_pred[:, 3]
        qs_p = theta_pred[:, 4]

        qm_t = qb_t + qs_t
        qm_p = qb_p + qs_p

        qb_err = np.abs(qb_p - qb_t)
        qs_err = np.abs(qs_p - qs_t)
        qm_err = np.abs(qm_p - qm_t)

        self.q_anchor_abs += float(np.sum(qb_err))
        self.q_span_abs += float(np.sum(qs_err))
        self.q_max_abs += float(np.sum(qm_err))

        self.q_anchor_true_abs += float(np.sum(np.abs(qb_t)))
        self.q_span_true_abs += float(np.sum(np.abs(qs_t)))
        self.q_max_true_abs += float(np.sum(np.abs(qm_t)))

        for true_v, err_v, sum_name, count_name in [
            (qb_t, qb_err, "q_anchor_mape_sum", "q_anchor_mape_count"),
            (qs_t, qs_err, "q_span_mape_sum", "q_span_mape_count"),
            (qm_t, qm_err, "q_max_mape_sum", "q_max_mape_count"),
        ]:
            valid = np.abs(true_v) > METRIC_EPS
            if valid.any():
                setattr(
                    self,
                    sum_name,
                    getattr(self, sum_name)
                    + float(np.sum(err_v[valid] / np.abs(true_v[valid]))),
                )
                setattr(
                    self,
                    count_name,
                    getattr(self, count_name) + int(valid.sum()),
                )

        # -------------------------------------------------------------
        # Segment shares and cumulative quantity breakpoints.
        # q1..q5 are model quantity-segment shares.
        # -------------------------------------------------------------
        Q_t = np.maximum(theta_true[:, 5:10], 0.0)
        Q_p = np.maximum(theta_pred[:, 5:10], 0.0)

        Q_t = Q_t / np.maximum(
            Q_t.sum(axis=1, keepdims=True),
            METRIC_EPS,
        )
        Q_p = Q_p / np.maximum(
            Q_p.sum(axis=1, keepdims=True),
            METRIC_EPS,
        )

        self.q_share_abs += float(np.sum(np.abs(Q_p - Q_t)))

        B_t = np.cumsum(Q_t, axis=1)[:, :4]
        B_p = np.cumsum(Q_p, axis=1)[:, :4]

        self.break_frac_abs += np.sum(np.abs(B_p - B_t), axis=0)

        true_break_mw = qb_t[:, None] + qs_t[:, None] * B_t
        pred_break_mw = qb_p[:, None] + qs_p[:, None] * B_p

        self.break_mw_abs += np.sum(
            np.abs(pred_break_mw - true_break_mw),
            axis=0,
        )

        # -------------------------------------------------------------
        # Price fit at true segment boundaries and segment midpoints.
        # p_est_all and p_true are both sampled on the TRUE normalized
        # physical-q grid, so row-wise interpolation is like-for-like.
        # -------------------------------------------------------------
        edges_t = np.c_[
            np.zeros(n),
            B_t,
            np.ones(n),
        ]
        mid_t = 0.5 * (edges_t[:, :-1] + edges_t[:, 1:])

        true_bp_price = interp_uniform_rows(p_true, B_t)
        pred_bp_price = interp_uniform_rows(p_est_all, B_t)
        true_mid_price = interp_uniform_rows(p_true, mid_t)
        pred_mid_price = interp_uniform_rows(p_est_all, mid_t)

        self.break_price_abs += np.sum(
            np.abs(pred_bp_price - true_bp_price),
            axis=0,
        )

        self.segment_mid_price_abs += np.sum(
            np.abs(pred_mid_price - true_mid_price),
            axis=0,
        )

        # -------------------------------------------------------------
        # Normalized price-shape error.
        # Normalize true and predicted curves by their OWN endpoint span.
        # This removes p_base / absolute price-scale error.
        # -------------------------------------------------------------
        true_span = p_true[:, -1] - p_true[:, 0]
        pred_span = p_est_all[:, -1] - p_est_all[:, 0]

        true_shape = np.zeros_like(p_true, dtype=float)
        pred_shape = np.zeros_like(p_est_all, dtype=float)

        true_nonflat = np.abs(true_span) > METRIC_EPS
        pred_nonflat = np.abs(pred_span) > METRIC_EPS

        if true_nonflat.any():
            true_shape[true_nonflat] = (
                p_true[true_nonflat]
                - p_true[true_nonflat, 0][:, None]
            ) / true_span[true_nonflat, None]

        if pred_nonflat.any():
            pred_shape[pred_nonflat] = (
                p_est_all[pred_nonflat]
                - p_est_all[pred_nonflat, 0][:, None]
            ) / pred_span[pred_nonflat, None]

        shape_err = pred_shape - true_shape

        self.shape_abs += float(np.sum(np.abs(shape_err)))
        self.shape_sq += float(np.sum(shape_err * shape_err))
        self.shape_count += int(shape_err.size)

        # -------------------------------------------------------------
        # Area and template.
        # -------------------------------------------------------------
        area_pred = np.trapz(p_pred, q_pred, axis=1)
        area_true = np.trapz(p_true, q_true, axis=1)

        self.area_abs += float(
            np.sum(np.abs(area_pred - area_true))
        )

        self.template_correct += int(
            np.sum(
                np.asarray(
                    [
                        str(a) == str(b)
                        for a, b in zip(t_true, t_pred)
                    ],
                    dtype=np.int8,
                )
            )
        )

    @staticmethod
    def _pct(num_value, den_value):
        if den_value <= METRIC_EPS:
            return np.nan
        return 100.0 * num_value / den_value

    def row(self, mode):
        pts = self.n * 21
        qpts = self.n * 5
        curves = np.asarray(self.curve_mae, dtype=float)

        row = {
            "evaluation_mode": mode,
            "rows": self.n,
            "template_accuracy": (
                self.template_correct / self.n
                if self.n else np.nan
            ),

            # Price.
            "price_mae": self.price_abs / pts if pts else np.nan,
            "price_rmse": (
                np.sqrt(self.price_sq / pts)
                if pts else np.nan
            ),
            "price_mape_pct": (
                100.0 * self.price_mape_sum / self.price_mape_count
                if self.price_mape_count else np.nan
            ),
            "price_mape_valid_share": (
                self.price_mape_count / pts
                if pts else np.nan
            ),
            "price_smape_pct": (
                100.0 * self.price_smape_sum / self.price_smape_count
                if self.price_smape_count else np.nan
            ),
            "price_wape_pct": self._pct(
                self.price_abs,
                self.price_true_abs,
            ),

            # Quantity.
            "q_anchor_mae_mw": (
                self.q_anchor_abs / self.n
                if self.n else np.nan
            ),
            "q_anchor_mape_pct": (
                100.0 * self.q_anchor_mape_sum / self.q_anchor_mape_count
                if self.q_anchor_mape_count else np.nan
            ),
            "q_anchor_wape_pct": self._pct(
                self.q_anchor_abs,
                self.q_anchor_true_abs,
            ),
            "q_span_mae_mw": (
                self.q_span_abs / self.n
                if self.n else np.nan
            ),
            "q_span_mape_pct": (
                100.0 * self.q_span_mape_sum / self.q_span_mape_count
                if self.q_span_mape_count else np.nan
            ),
            "q_span_wape_pct": self._pct(
                self.q_span_abs,
                self.q_span_true_abs,
            ),
            "q_max_mae_mw": (
                self.q_max_abs / self.n
                if self.n else np.nan
            ),
            "q_max_mape_pct": (
                100.0 * self.q_max_mape_sum / self.q_max_mape_count
                if self.q_max_mape_count else np.nan
            ),
            "q_max_wape_pct": self._pct(
                self.q_max_abs,
                self.q_max_true_abs,
            ),

            # Segments / breakpoints.
            "q_share_mae": (
                self.q_share_abs / qpts
                if qpts else np.nan
            ),
            "breakpoint_fraction_mae": (
                float(np.sum(self.break_frac_abs)) / (self.n * 4)
                if self.n else np.nan
            ),
            "breakpoint_mw_mae": (
                float(np.sum(self.break_mw_abs)) / (self.n * 4)
                if self.n else np.nan
            ),
            "breakpoint_price_mae": (
                float(np.sum(self.break_price_abs)) / (self.n * 4)
                if self.n else np.nan
            ),
            "segment_mid_price_mae": (
                float(np.sum(self.segment_mid_price_abs)) / (self.n * 5)
                if self.n else np.nan
            ),

            # Shape.
            "normalized_shape_mae": (
                self.shape_abs / self.shape_count
                if self.shape_count else np.nan
            ),
            "normalized_shape_rmse": (
                np.sqrt(self.shape_sq / self.shape_count)
                if self.shape_count else np.nan
            ),

            # Curve-level distribution.
            "curve_mae_p50": (
                float(np.quantile(curves, 0.50))
                if len(curves) else np.nan
            ),
            "curve_mae_p75": (
                float(np.quantile(curves, 0.75))
                if len(curves) else np.nan
            ),
            "curve_mae_p90": (
                float(np.quantile(curves, 0.90))
                if len(curves) else np.nan
            ),
            "curve_mae_p95": (
                float(np.quantile(curves, 0.95))
                if len(curves) else np.nan
            ),
            "curve_mae_le_5_share": (
                float(np.mean(curves <= 5.0))
                if len(curves) else np.nan
            ),
            "curve_mae_le_10_share": (
                float(np.mean(curves <= 10.0))
                if len(curves) else np.nan
            ),
            "curve_mae_le_20_share": (
                float(np.mean(curves <= 20.0))
                if len(curves) else np.nan
            ),
            "curve_mae_le_50_share": (
                float(np.mean(curves <= 50.0))
                if len(curves) else np.nan
            ),

            "curve_area_abs_error": (
                self.area_abs / self.n
                if self.n else np.nan
            ),
        }

        for k in range(4):
            row[f"b{k+1}_fraction_mae"] = (
                self.break_frac_abs[k] / self.n
                if self.n else np.nan
            )
            row[f"b{k+1}_mw_mae"] = (
                self.break_mw_abs[k] / self.n
                if self.n else np.nan
            )
            row[f"b{k+1}_price_mae"] = (
                self.break_price_abs[k] / self.n
                if self.n else np.nan
            )

        for k in range(5):
            row[f"segment{k+1}_mid_price_mae"] = (
                self.segment_mid_price_abs[k] / self.n
                if self.n else np.nan
            )

        return row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------



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
        "--bidtemplate-root",
        default="data/processed/bidtemplate",
    )

    ap.add_argument(
        "--save-sample-rows",
        type=int,
        default=200,
    )

    args = ap.parse_args()

    base = (
        Path(args.root)
        / str(args.year)
    )

    frozen = (
        base
        / "frozen_modeling_dataset"
    )

    manifest = load_manifest(
        frozen
    )

    test_curve_parts = [
        frozen / p
        for p in manifest[
            "parts"
        ][
            "test_curve"
        ]
    ]

    parameter_dir = (
        base
        / "template_parameter_models"
    )

    selected = pd.read_csv(
        parameter_dir
        / "selected_template_parameter_model.csv"
    ).iloc[0]

    fs = str(
        selected[
            "selected_feature_set"
        ]
    )

    model = str(
        selected[
            "selected_model"
        ]
    )

    parameter_bundle = joblib.load(
        parameter_dir
        / "models"
        / f"{fs}.joblib"
    )

    if (
        parameter_bundle.get(
            "model_mode"
        )
        != "direct_21_price_q_specialist"
    ):
        raise RuntimeError(
            "This 04c expects the v9 direct-price 04b bundle. "
            "Run 04b_train_template_parameter_models_frozen_v9.py first."
        )

    template_bundle = joblib.load(
        base
        / "final_template_predictor"
        / "final_template_model.joblib"
    )

    states = {
        "representation_oracle": Metrics(),
        "oracle_template_model": Metrics(),
        "full_pipeline": Metrics(),
    }

    samples = []

    print("=" * 80)
    print(
        "Final bid-curve prediction - direct 21-point price curve"
    )
    print("=" * 80)
    print(
        f"Year:                  {args.year}"
    )
    print(
        f"Frozen test rows:      {manifest['test_rows']:,}"
    )
    print(
        f"Parameter regressor:   {model} / {fs}"
    )
    print(
        "Price representation:  direct 21-point residual to lt_bid_level"
    )
    print(
        "Price model:           "
        f"{parameter_bundle['price_model']['candidate']}"
    )
    print(
        "Quantity scale target: "
        f"{parameter_bundle['q_spec']['q_scale_mode']}"
    )
    print(
        "q-scale model:         "
        f"{parameter_bundle['q_scale_model']['candidate']}"
    )
    print(
        "q-shape model:         "
        f"{parameter_bundle['q_shape_model']['candidate']}"
    )
    print(
        "H/direct bid history:  disabled"
    )
    print()

    for i, p in enumerate(
        test_curve_parts,
        1,
    ):
        print(
            f"[test frozen {i}/{len(test_curve_parts)}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(
            p
        )

        if d.empty:
            continue

        (
            good,
            q_true,
            p_true,
            t_true,
        ) = actual_curve(
            d
        )

        d = d.loc[
            good
        ].copy()

        q_true = q_true[
            good
        ]

        p_true = p_true[
            good
        ]

        t_true = t_true[
            good
        ]

        if d.empty:
            continue

        true_t_s = norm(
            d[
                TARGET_TEMPLATE
            ]
        )

        true_t = true_t_s.to_numpy(
            object
        )

        true_q_base = num(
            d[
                "theta_q_base_mw"
            ]
        ).to_numpy(float)

        true_q_span = num(
            d[
                "theta_q_span_mw"
            ]
        ).to_numpy(float)

        true_q_share = (
            d[
                [
                    "theta_q1",
                    "theta_q2",
                    "theta_q3",
                    "theta_q4",
                    "theta_q5",
                ]
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(float)
        )

        true_q_share = np.maximum(
            true_q_share,
            1e-8,
        )

        true_q_share /= np.maximum(
            true_q_share.sum(
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )

        true_theta = v9_theta_for_metrics(
            true_q_base,
            true_q_span,
            true_q_share,
        )

        # 1) Representation oracle:
        # direct 21-point representation is the Stage2 curve itself.
        states[
            "representation_oracle"
        ].update(
            q_true,
            p_true,
            q_true,
            p_true,
            t_true,
            true_t,
            true_theta,
            true_theta,
        )

        # 2) True template condition + predicted price/quantity curve.
        oracle_pred = v9_predict_curve(
            parameter_bundle,
            d,
            true_t_s,
        )

        states[
            "oracle_template_model"
        ].update(
            q_true,
            p_true,
            oracle_pred[
                "q"
            ],
            oracle_pred[
                "p"
            ],
            t_true,
            true_t,
            true_theta,
            oracle_pred[
                "theta"
            ],
        )

        # 3) Predicted template condition + predicted price/quantity curve.
        origin_valid = (
            norm(
                d[
                    ORIGIN_TEMPLATE
                ]
            )
            .isin(
                TEMPLATES
            )
            .to_numpy()
        )

        if origin_valid.any():
            df = d.loc[
                origin_valid
            ].copy()

            qt = q_true[
                origin_valid
            ]

            pt = p_true[
                origin_valid
            ]

            tt = t_true[
                origin_valid
            ]

            true_theta_fp = true_theta[
                origin_valid
            ]

            (
                pred_t,
                p_switch,
            ) = predict_template(
                df,
                template_bundle,
            )

            pred_t_s = pd.Series(
                pred_t,
                index=df.index,
                dtype="string",
            )

            pred = v9_predict_curve(
                parameter_bundle,
                df,
                pred_t_s,
            )

            states[
                "full_pipeline"
            ].update(
                qt,
                pt,
                pred[
                    "q"
                ],
                pred[
                    "p"
                ],
                tt,
                pred_t,
                true_theta_fp,
                pred[
                    "theta"
                ],
            )

            remain = (
                args.save_sample_rows
                - len(samples)
            )

            if remain > 0:
                take = min(
                    remain,
                    len(df),
                )

                for j in range(
                    take
                ):
                    samples.append(
                        {
                            "sample_id": df.iloc[j][
                                "sample_id"
                            ],
                            "participant_id": df.iloc[j][
                                "participant_id"
                            ],
                            "local_date": df.iloc[j][
                                "local_date"
                            ],
                            "true_template_id": tt[j],
                            "pred_template_id": pred_t[j],
                            "switch_probability": float(
                                p_switch[j]
                            ),
                            "pred_q_base_mw": float(
                                pred[
                                    "theta"
                                ][
                                    j,
                                    3,
                                ]
                            ),
                            "pred_q_span_mw": float(
                                pred[
                                    "theta"
                                ][
                                    j,
                                    4,
                                ]
                            ),
                            "pred_q_share_json": json.dumps(
                                pred[
                                    "theta"
                                ][
                                    j,
                                    5:10,
                                ].tolist()
                            ),
                            "pred_price_21_json": json.dumps(
                                pred[
                                    "p"
                                ][j].tolist()
                            ),
                        }
                    )

        del d
        gc.collect()

    metrics = pd.DataFrame(
        [
            states[name].row(name)
            for name in [
                "representation_oracle",
                "oracle_template_model",
                "full_pipeline",
            ]
        ]
    )

    out = (
        base
        / "curve_reconstruction"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        out
        / "reconstruction_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        samples
    ).to_csv(
        out
        / "reconstructed_samples_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oracle_rows = int(
        metrics.loc[
            metrics[
                "evaluation_mode"
            ].eq(
                "oracle_template_model"
            ),
            "rows",
        ].iloc[0]
    )

    full_rows = int(
        metrics.loc[
            metrics[
                "evaluation_mode"
            ].eq(
                "full_pipeline"
            ),
            "rows",
        ].iloc[0]
    )

    coverage = (
        full_rows
        / oracle_rows
        if oracle_rows
        else np.nan
    )

    lines = [
        (
            "Final bid-curve prediction - "
            f"direct 21-point price curve - {args.year}"
        ),
        "=" * 80,
        "",
        (
            "Full-pipeline eligible coverage = "
            f"{full_rows:,}/{oracle_rows:,} "
            f"({coverage:.2%})"
        ),
        "",
        (
            f"Parameter regressor: "
            f"{model} / {fs}"
        ),
        "",
        "Price representation:",
        (
            "  direct P(x0..x20) residual to lt_bid_level"
        ),
        (
            "  selected price model = "
            f"{parameter_bundle['price_model']['candidate']}"
        ),
        "",
        "Quantity representation:",
        (
            "  q scale target = "
            f"{parameter_bundle['q_spec']['q_scale_mode']}"
        ),
        (
            "  q_scale model = "
            f"{parameter_bundle['q_scale_model']['candidate']}"
        ),
        (
            "  q_shape model = "
            f"{parameter_bundle['q_shape_model']['candidate']}"
        ),
        "  q1..q5 = simplex shares",
        "",
        "Final TEST curve-quality metrics:",
    ]

    for r in metrics.itertuples():
        lines += [
            f"  [{r.evaluation_mode}]",
            f"    rows={int(r.rows):,}",
            (
                "    template_accuracy="
                f"{r.template_accuracy:.6f}"
            ),
            "",
            "    Price:",
            f"      MAE={r.price_mae:.6f}",
            f"      RMSE={r.price_rmse:.6f}",
            f"      MAPE={r.price_mape_pct:.4f}%",
            (
                "      MAPE_valid_point_share="
                f"{r.price_mape_valid_share:.4%}"
            ),
            f"      sMAPE={r.price_smape_pct:.4f}%",
            f"      WAPE={r.price_wape_pct:.4f}%",
            "",
            "    Quantity:",
            (
                f"      q_anchor: MAE={r.q_anchor_mae_mw:.6f} MW, "
                f"MAPE={r.q_anchor_mape_pct:.4f}%, "
                f"WAPE={r.q_anchor_wape_pct:.4f}%"
            ),
            (
                f"      q_span:   MAE={r.q_span_mae_mw:.6f} MW, "
                f"MAPE={r.q_span_mape_pct:.4f}%, "
                f"WAPE={r.q_span_wape_pct:.4f}%"
            ),
            (
                f"      q_max:    MAE={r.q_max_mae_mw:.6f} MW, "
                f"MAPE={r.q_max_mape_pct:.4f}%, "
                f"WAPE={r.q_max_wape_pct:.4f}%"
            ),
            "",
            "    Segment / breakpoint fit:",
            f"      q_share_MAE={r.q_share_mae:.6f}",
            (
                "      breakpoint_fraction_MAE="
                f"{r.breakpoint_fraction_mae:.6f}"
            ),
            (
                "      breakpoint_MW_MAE="
                f"{r.breakpoint_mw_mae:.6f} MW"
            ),
            (
                "      breakpoint_price_MAE="
                f"{r.breakpoint_price_mae:.6f}"
            ),
            (
                "      segment_mid_price_MAE="
                f"{r.segment_mid_price_mae:.6f}"
            ),
            (
                "      b1..b4 fraction MAE="
                f"[{r.b1_fraction_mae:.6f}, "
                f"{r.b2_fraction_mae:.6f}, "
                f"{r.b3_fraction_mae:.6f}, "
                f"{r.b4_fraction_mae:.6f}]"
            ),
            (
                "      b1..b4 MW MAE="
                f"[{r.b1_mw_mae:.3f}, "
                f"{r.b2_mw_mae:.3f}, "
                f"{r.b3_mw_mae:.3f}, "
                f"{r.b4_mw_mae:.3f}]"
            ),
            (
                "      segment1..5 midpoint price MAE="
                f"[{r.segment1_mid_price_mae:.3f}, "
                f"{r.segment2_mid_price_mae:.3f}, "
                f"{r.segment3_mid_price_mae:.3f}, "
                f"{r.segment4_mid_price_mae:.3f}, "
                f"{r.segment5_mid_price_mae:.3f}]"
            ),
            "",
            "    Normalized shape:",
            (
                f"      MAE="
                f"{r.normalized_shape_mae:.6f}"
            ),
            (
                f"      RMSE="
                f"{r.normalized_shape_rmse:.6f}"
            ),
            "",
            "    Curve-level price MAE distribution:",
            (
                "      P50/P75/P90/P95="
                f"{r.curve_mae_p50:.3f} / "
                f"{r.curve_mae_p75:.3f} / "
                f"{r.curve_mae_p90:.3f} / "
                f"{r.curve_mae_p95:.3f}"
            ),
            (
                "      <=5/10/20/50 share="
                f"{r.curve_mae_le_5_share:.2%} / "
                f"{r.curve_mae_le_10_share:.2%} / "
                f"{r.curve_mae_le_20_share:.2%} / "
                f"{r.curve_mae_le_50_share:.2%}"
            ),
            "",
            (
                "    curve_area_abs_error="
                f"{r.curve_area_abs_error:.6f}"
            ),
            "",
        ]

    summary = "\n".join(lines)

    (
        out / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    config = {
        "version": "04c-direct-price-v9",
        "year": args.year,
        "parameter_model_mode": parameter_bundle[
            "model_mode"
        ],
        "price_representation": (
            "direct 21-point residual to lt_bid_level"
        ),
        "parameter_model_file": str(
            parameter_dir
            / "models"
            / f"{fs}.joblib"
        ),
    }

    (
        out / "config.json"
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
        f"Outputs: {out}"
    )


if __name__ == "__main__":
    main()
