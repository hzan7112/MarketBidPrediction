#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04c_reconstruct_bid_curves.py

Frozen-data final curve reconstruction.

Reads ONLY:
data/processed/bidprediction/<year>/frozen_modeling_dataset/test_curve_parts/*.pkl

It does NOT:
- rescan monthly modeling CSVs,
- recompute test split,
- rejoin Stage2 curve samples,
- reload transition features from a separate table.

The frozen test_curve parts already contain:
- all candidate features,
- theta targets,
- Stage2 shape_v00..shape_v20,
- fixed test membership.

Run:
python scripts/bidprediction/04c_reconstruct_bid_curves.py --year 2025
"""

from __future__ import annotations

import argparse
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


def load_manifest(frozen: Path):
    p = frozen / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p}\nRun 04a3_freeze_modeling_dataset.py first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def shape_cols(header):
    exact = [f"shape_v{i:02d}" for i in range(21)]
    if all(c in header for c in exact):
        return exact
    return None


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
        candidates = sorted(
            year_dir.rglob("*.csv")
        )

    for p in candidates:
        try:
            h = pd.read_csv(
                p,
                nrows=0,
            ).columns.tolist()
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
            t = normalize_template_id(
                r[id_col]
            )

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


def enc_col(s, col):
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
            c: enc_col(
                d[c],
                c,
            )
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
        c: enc_col(
            d[c],
            c,
        )
        for c in features
    }

    t = norm(
        template
    ).map(
        T2I
    )

    if t.isna().any():
        raise ValueError(
            "Unknown template condition."
        )

    a = t.to_numpy(
        np.int16
    )

    for j, z in enumerate(TEMPLATES):
        out[
            f"cond_template_{z}"
        ] = (
            a == j
        ).astype(
            "float32"
        )

    return pd.DataFrame(
        out,
        index=d.index,
    )


def expand_proba(
    model,
    raw_p,
):
    classes = np.asarray(
        model.classes_,
        dtype=np.int16,
    )

    full = np.zeros(
        (
            len(raw_p),
            N_TEMPLATE,
        ),
        dtype=float,
    )

    full[
        :,
        classes,
    ] = raw_p

    return full


def mask_destination_proba(
    p,
    origin,
    allowed,
):
    p = np.asarray(
        p,
        dtype=float,
    ).copy()

    for i in range(
        N_TEMPLATE
    ):
        rows = np.where(
            origin == i
        )[0]

        if not len(rows):
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

    den = p.sum(
        axis=1,
        keepdims=True,
    )

    bad = (
        den[:, 0]
        <= 0
    )

    if bad.any():
        for r in np.where(
            bad
        )[0]:
            p[
                r,
                :
            ] = 1.0

            p[
                r,
                origin[r],
            ] = 0.0

        den = p.sum(
            axis=1,
            keepdims=True,
        )

    p /= den

    return np.argmax(
        p,
        axis=1,
    ).astype(
        np.int16
    )


def predict_template(
    d,
    bundle,
):
    switch_features = bundle[
        "switch_features"
    ]

    destination_features = bundle[
        "destination_features"
    ]

    required = uniq(
        switch_features
        + destination_features
        + [ORIGIN_TEMPLATE]
    )

    missing = [
        c
        for c in required
        if c not in d.columns
    ]

    if missing:
        raise KeyError(
            f"Frozen test set is missing final-template features: {missing}"
        )

    origin_s = norm(
        d[
            ORIGIN_TEMPLATE
        ]
    ).map(
        T2I
    )

    if origin_s.isna().any():
        raise ValueError(
            "Final template predictor requires valid hist_lag1_template_id."
        )

    origin = origin_s.to_numpy(
        np.int16
    )

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

    sw_model = bundle[
        "switch_model"
    ]

    p_raw = sw_model.predict_proba(
        Xs
    )

    classes = list(
        sw_model.classes_
    )

    p_switch = (
        p_raw[
            :,
            classes.index(1),
        ]
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
            origin.astype(
                np.float32
            ),
        ]
    )

    dest_model = bundle[
        "destination_model"
    ]

    full_p = expand_proba(
        dest_model,
        dest_model.predict_proba(
            Xdg
        ),
    )

    dest_pred = mask_destination_proba(
        full_p,
        origin,
        np.asarray(
            bundle[
                "allowed_destinations_train"
            ],
            dtype=bool,
        ),
    )

    final = origin.copy()

    override = (
        p_switch
        >= float(
            bundle[
                "hierarchy_threshold"
            ]
        )
    )

    final[
        override
    ] = dest_pred[
        override
    ]

    names = np.asarray(
        [
            I2T[
                int(v)
            ]
            for v in final
        ],
        dtype=object,
    )

    return names, p_switch


def latent_to_raw(
    z,
    clip_lo,
    clip_hi,
):
    z = np.asarray(
        z,
        dtype=float,
    )

    z = np.clip(
        z,
        np.asarray(
            clip_lo
        )[None, :],
        np.asarray(
            clip_hi
        )[None, :],
    )

    out = np.empty(
        (
            len(z),
            10,
        ),
        dtype=float,
    )

    out[
        :,
        0,
    ] = z[:, 0]

    out[
        :,
        1,
    ] = np.maximum(
        np.expm1(
            z[:, 1]
        ),
        0.0,
    )

    out[
        :,
        2,
    ] = np.maximum(
        np.expm1(
            z[:, 2]
        ),
        0.0,
    )

    qmax = np.maximum(
        np.expm1(
            z[:, 3]
        ),
        1e-8,
    )

    zr = np.clip(
        z[:, 4],
        -30.0,
        30.0,
    )

    rbase = 1.0 / (
        1.0
        + np.exp(
            -zr
        )
    )

    rbase = np.clip(
        rbase,
        1e-6,
        1.0 - 1e-6,
    )

    out[
        :,
        3,
    ] = (
        qmax
        * rbase
    )

    out[
        :,
        4,
    ] = (
        qmax
        * (
            1.0 - rbase
        )
    )

    logits = np.c_[
        z[:, 5:9],
        np.zeros(len(z)),
    ]

    logits -= logits.max(
        axis=1,
        keepdims=True,
    )

    e = np.exp(
        logits
    )

    out[
        :,
        5:10,
    ] = (
        e
        / e.sum(
            axis=1,
            keepdims=True,
        )
    )

    return out


def predict_theta(
    bundle,
    model_name,
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

    Xin = (
        bundle[
            "x_scaler"
        ].transform(
            X
        ).astype(
            np.float32,
            copy=False,
        )
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
        bundle[
            "latent_clip_lo"
        ],
        bundle[
            "latent_clip_hi"
        ],
    )

    flat = norm(
        template
    ).eq(
        "FLAT"
    ).to_numpy()

    theta[
        flat,
        1,
    ] = 0.0

    theta[
        flat,
        2,
    ] = 0.0

    return theta


def tail_basis(
    u,
):
    z = np.maximum(
        0.0,
        (
            u - TAIL_START
        )
        / (
            1.0 - TAIL_START
        ),
    )

    return z * z


TAIL = tail_basis(
    GRID
)


def reconstruct(
    theta,
    templates,
    centers,
):
    theta = np.asarray(
        theta,
        dtype=float,
    )

    n = len(theta)

    Q = np.maximum(
        theta[
            :,
            5:10,
        ],
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

    cum[
        :,
        -1,
    ] = 1.0

    x = np.empty(
        (
            n,
            21,
        )
    )

    for j, u in enumerate(
        GRID
    ):
        k = min(
            int(
                np.floor(
                    u * 5
                )
            ),
            4,
        )

        frac = (
            u
            - U_KNOTS[k]
        ) / (
            U_KNOTS[
                k + 1
            ]
            - U_KNOTS[k]
        )

        frac = float(
            np.clip(
                frac,
                0.0,
                1.0,
            )
        )

        x[
            :,
            j,
        ] = (
            cum[
                :,
                k,
            ]
            + frac
            * Q[
                :,
                k,
            ]
        )

    x[
        :,
        0,
    ] = 0.0

    x[
        :,
        -1,
    ] = 1.0

    q = (
        theta[
            :,
            3,
            None,
        ]
        + theta[
            :,
            4,
            None,
        ]
        * x
    )

    C = np.vstack(
        [
            centers[
                str(t)
            ]
            for t in templates
        ]
    )

    p = (
        theta[
            :,
            0,
            None,
        ]
        + theta[
            :,
            1,
            None,
        ]
        * C
        + theta[
            :,
            2,
            None,
        ]
        * TAIL[
            None,
            :,
        ]
    )

    return q, p


def actual_curve(
    d,
):
    qa = num(
        d[
            "q_anchor_mw"
        ]
    ).to_numpy(float)

    qs = num(
        d[
            "q_span_mw"
        ]
    ).to_numpy(float)

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
        d[
            TARGET_TEMPLATE
        ]
    ).to_numpy(object)

    S[
        tid == "FLAT"
    ] = 0.0

    valid = (
        np.isfinite(
            S
        ).all(
            axis=1
        )
        & np.isfinite(
            qa
        )
        & np.isfinite(
            qs
        )
        & np.isfinite(
            pa
        )
        & np.isfinite(
            ps
        )
        & (
            qs > 0
        )
    )

    q = (
        qa[
            :,
            None,
        ]
        + qs[
            :,
            None,
        ]
        * GRID
    )

    p = (
        pa[
            :,
            None,
        ]
        + ps[
            :,
            None,
        ]
        * S
    )

    return (
        valid,
        q,
        p,
        tid,
    )


class Metrics:
    def __init__(self):
        self.n = 0
        self.price_abs = 0.0
        self.price_sq = 0.0
        self.quantity_abs = 0.0
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
    ):
        for i in range(
            len(q_true)
        ):
            order = np.argsort(
                q_pred[i]
            )

            qp = q_pred[
                i,
                order,
            ]

            pp = p_pred[
                i,
                order,
            ]

            uq, idx = np.unique(
                qp,
                return_index=True,
            )

            up = pp[
                idx
            ]

            if len(uq) < 2:
                p_est = np.full_like(
                    p_true[i],
                    up[0]
                    if len(up)
                    else np.nan,
                )
            else:
                p_est = np.interp(
                    q_true[i],
                    uq,
                    up,
                    left=up[0],
                    right=up[-1],
                )

            pe = (
                p_est
                - p_true[i]
            )

            self.n += 1

            self.price_abs += float(
                np.abs(
                    pe
                ).sum()
            )

            self.price_sq += float(
                np.square(
                    pe
                ).sum()
            )

            self.quantity_abs += float(
                np.abs(
                    q_pred[i]
                    - q_true[i]
                ).sum()
            )

            self.area_abs += abs(
                float(
                    np.trapz(
                        p_pred[i],
                        q_pred[i],
                    )
                )
                - float(
                    np.trapz(
                        p_true[i],
                        q_true[i],
                    )
                )
            )

            self.template_correct += int(
                str(
                    t_true[i]
                )
                == str(
                    t_pred[i]
                )
            )

    def row(
        self,
        mode,
    ):
        pts = (
            self.n
            * 21
        )

        return {
            "evaluation_mode": mode,
            "rows": self.n,
            "template_accuracy": (
                self.template_correct
                / self.n
                if self.n
                else np.nan
            ),
            "price_mae": (
                self.price_abs
                / pts
                if pts
                else np.nan
            ),
            "price_rmse": (
                np.sqrt(
                    self.price_sq
                    / pts
                )
                if pts
                else np.nan
            ),
            "quantity_grid_mae_mw": (
                self.quantity_abs
                / pts
                if pts
                else np.nan
            ),
            "curve_area_abs_error": (
                self.area_abs
                / self.n
                if self.n
                else np.nan
            ),
        }


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
        "--save-sample-rows",
        type=int,
        default=5000,
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

    frozen = (
        base
        / "frozen_modeling_dataset"
    )

    manifest = load_manifest(
        frozen
    )

    test_curve_parts = [
        frozen
        / p
        for p in manifest[
            "parts"
        ][
            "test_curve"
        ]
    ]

    if not test_curve_parts:
        raise ValueError(
            "Frozen test_curve_parts are empty."
        )

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

    template_bundle = joblib.load(
        base
        / "final_template_predictor"
        / "final_template_model.joblib"
    )

    centers, center_file = discover_centers(
        Path(
            args.bidtemplate_root
        )
        / str(
            args.year
        )
    )

    states = {
        "representation_oracle": Metrics(),
        "oracle_template_model": Metrics(),
        "full_pipeline": Metrics(),
    }

    samples = []

    print("=" * 80)
    print("Final bid-curve prediction - frozen dataset")
    print("=" * 80)
    print(f"Year:                  {args.year}")
    print(f"Frozen test rows:      {manifest['test_rows']:,}")
    print(
        f"Parameter regressor:   {model} / {fs}"
    )
    print(
        "q_base/q_span source:  regression model"
    )
    print(
        "Lag1 scale replacement: disabled"
    )
    print(
        f"Template centers:      {center_file}"
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

        good, q_true, p_true, t_true = actual_curve(
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

        true_theta = d[
            RAW_THETA
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(float)

        true_t_s = norm(
            d[
                TARGET_TEMPLATE
            ]
        )

        true_t = true_t_s.to_numpy(
            object
        )

        q0, p0 = reconstruct(
            true_theta,
            true_t,
            centers,
        )

        states[
            "representation_oracle"
        ].update(
            q_true,
            p_true,
            q0,
            p0,
            t_true,
            true_t,
        )

        theta_o = predict_theta(
            parameter_bundle,
            model,
            d,
            true_t_s,
        )

        q1, p1 = reconstruct(
            theta_o,
            true_t,
            centers,
        )

        states[
            "oracle_template_model"
        ].update(
            q_true,
            p_true,
            q1,
            p1,
            t_true,
            true_t,
        )

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

            pred_t, p_switch = predict_template(
                df,
                template_bundle,
            )

            pred_t_s = pd.Series(
                pred_t,
                index=df.index,
                dtype="string",
            )

            theta_f = predict_theta(
                parameter_bundle,
                model,
                df,
                pred_t_s,
            )

            q2, p2 = reconstruct(
                theta_f,
                pred_t,
                centers,
            )

            states[
                "full_pipeline"
            ].update(
                qt,
                pt,
                q2,
                p2,
                tt,
                pred_t,
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
                            "sample_id": df.iloc[j]["sample_id"],
                            "participant_id": df.iloc[j]["participant_id"],
                            "local_date": df.iloc[j]["local_date"],
                            "true_template_id": tt[j],
                            "pred_template_id": pred_t[j],
                            "switch_probability": float(p_switch[j]),
                            "true_theta_json": json.dumps(
                                true_theta_fp[j].tolist()
                            ),
                            "pred_theta_json": json.dumps(
                                theta_f[j].tolist()
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
        f"Final bid-curve prediction - frozen dataset - {args.year}",
        "=" * 80,
        "",
        (
            f"Full-pipeline eligible coverage = "
            f"{full_rows:,}/{oracle_rows:,} ({coverage:.2%})"
        ),
        "",
        f"Parameter regressor: {model} / {fs}",
        "q_base_mw: regression",
        "q_span_mw: regression",
        "Lag1 scale replacement: disabled",
        "",
        "Curve-level TEST metrics:",
    ]

    for r in metrics.itertuples():
        lines += [
            f"  [{r.evaluation_mode}]",
            f"    rows={int(r.rows):,}",
            f"    template_accuracy={r.template_accuracy:.6f}",
            f"    price_MAE={r.price_mae:.6f}",
            f"    price_RMSE={r.price_rmse:.6f}",
            f"    quantity_grid_MAE_MW={r.quantity_grid_mae_mw:.6f}",
            f"    curve_area_abs_error={r.curve_area_abs_error:.6f}",
            "",
        ]

    summary = "\n".join(
        lines
    )

    (
        out
        / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    (
        out
        / "config.json"
    ).write_text(
        json.dumps(
            {
                "year": args.year,
                "frozen_dataset": str(frozen),
                "selected_parameter_model": model,
                "selected_feature_set": fs,
                "template_centers": str(center_file),
                "full_pipeline_coverage": (
                    float(coverage)
                    if np.isfinite(coverage)
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    import gc
    main()
