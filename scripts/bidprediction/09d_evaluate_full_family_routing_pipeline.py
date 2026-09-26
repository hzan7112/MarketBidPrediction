#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09d_evaluate_full_family_routing_pipeline.py

Evaluate the deployable hard-routing pipeline:

83 feature-only inputs
    -> selected 09b family classifier
    -> predicted K=5 family
    -> validation-selected family expert from 09c
    -> absolute latent
    -> bid curve

References reported in the same task:
- global_random_forest: the 08c global feature-only RF
- oracle_family_experts: TRUE family + selected family expert
- predicted_family_experts: deployable hard routing

Also outputs true-family vs predicted-family routing-pair curve errors.
"""

from __future__ import annotations

import argparse, gc, json, shutil
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


GRID = np.linspace(0.0, 1.0, 21)
SHAPE = [f"shape_v{i:02d}" for i in range(21)]
META = [
    "sample_id",
    "participant_id",
    "timestamp_utc",
    "timestamp_local",
    "local_date",
    "local_slot_seconds",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


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


def true_curve(d):
    sh = d[SHAPE].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)

    pa = num(d["p_anchor"]).to_numpy(float)
    ps = num(d["p_span"]).to_numpy(float)

    flat = np.abs(ps) <= 1e-12

    if flat.any():
        sh[flat] = np.nan_to_num(
            sh[flat],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    p = pa[:, None] + ps[:, None] * sh
    qa = num(d["q_anchor_mw"]).to_numpy(float)
    qs = num(d["q_span_mw"]).to_numpy(float)
    q = qa[:, None] + qs[:, None] * GRID[None, :]

    return q, p, qa, qs


def unpack(v):
    return (
        v[:, :21],
        v[:, 21],
        np.exp(
            np.clip(
                v[:, 22],
                -20.0,
                20.0,
            )
        ),
    )


def on_q(q, p, qa, qs):
    pos = np.clip(
        (q - qa[:, None])
        / np.maximum(
            qs[:, None],
            1e-8,
        ),
        0.0,
        1.0,
    ) * 20.0

    lo = np.floor(pos).astype(np.int16)
    hi = np.minimum(lo + 1, 20)
    f = pos - lo

    return (
        np.take_along_axis(p, lo, 1)
        + f
        * (
            np.take_along_axis(p, hi, 1)
            - np.take_along_axis(p, lo, 1)
        )
    )


def decode(z, bundle):
    pca = bundle["pca"]
    sc = bundle["scaler"]

    full = np.zeros(
        (
            len(z),
            int(pca.n_components_),
        ),
        float,
    )

    full[:, : z.shape[1]] = z

    return sc.inverse_transform(
        pca.inverse_transform(
            full
        )
    )


def true_family(
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

    return cluster_model.predict(
        z
    ).astype(
        np.int16
    )


def pred_gam(d, gam, targets):
    out = np.empty(
        (
            len(d),
            len(targets),
        ),
        dtype=np.float32,
    )

    for j, t in enumerate(targets):
        spec = gam[t]

        out[:, j] = spec[
            "model"
        ].predict(
            nframe(
                d,
                spec[
                    "features"
                ],
            )
        ).astype(
            np.float32
        )

    return out


def predict_expert(
    d,
    family_spec,
    expert_name,
    features,
    targets,
):
    if expert_name == "family_mean":
        return np.repeat(
            family_spec[
                "family_mean"
            ][
                None,
                :,
            ],
            len(d),
            axis=0,
        ).astype(
            np.float32
        )

    if expert_name == "ridge":
        return family_spec[
            "ridge"
        ].predict(
            nframe(
                d,
                features,
            )
        ).astype(
            np.float32
        )

    if expert_name == "spline_gam":
        return pred_gam(
            d,
            family_spec[
                "spline_gam"
            ],
            targets,
        )

    if expert_name == "random_forest":
        return family_spec[
            "random_forest"
        ].predict(
            nframe(
                d,
                features,
            )
        ).astype(
            np.float32
        )

    raise KeyError(
        expert_name
    )


def routed_latent(
    d,
    route_family,
    family_specs,
    expert_selection,
    features,
    targets,
):
    out = np.empty(
        (
            len(d),
            len(targets),
        ),
        dtype=np.float32,
    )

    for fam in sorted(
        family_specs
    ):
        idx = np.flatnonzero(
            route_family
            == fam
        )

        if not len(idx):
            continue

        sub = d.iloc[
            idx
        ]

        fam_id = f"F{fam:02d}"

        expert_name = expert_selection[
            fam_id
        ]

        out[
            idx
        ] = predict_expert(
            sub,
            family_specs[
                fam
            ],
            expert_name,
            features,
            targets,
        )

    return out


def state():
    return {
        "rows": 0,
        "ae": 0.0,
        "se": 0.0,
        "abst": 0.0,
        "n": 0,
        "qa": 0.0,
        "qaa": 0.0,
        "qs": 0.0,
        "qsa": 0.0,
    }


def row_errors(
    tq,
    tp,
    tqa,
    tqs,
    v,
):
    p, qa, qs = unpack(
        v
    )

    pred = on_q(
        tq,
        p,
        qa,
        qs,
    )

    e = pred - tp
    ae = np.abs(
        e
    )

    return {
        "row_ae_sum": ae.sum(
            axis=1
        ),
        "row_abs_true_sum": np.abs(
            tp
        ).sum(
            axis=1
        ),
        "row_se_sum": np.square(
            e
        ).sum(
            axis=1
        ),
        "row_curve_mae": ae.mean(
            axis=1
        ),
        "row_qa_ae": np.abs(
            qa - tqa
        ),
        "row_qa_abs": np.abs(
            tqa
        ),
        "row_qs_ae": np.abs(
            qs - tqs
        ),
        "row_qs_abs": np.abs(
            tqs
        ),
    }


def update_from_rows(
    st,
    r,
    mask=None,
):
    if mask is None:
        mask = slice(
            None
        )

    ae = r[
        "row_ae_sum"
    ][
        mask
    ]

    if not len(
        np.atleast_1d(
            ae
        )
    ):
        return

    abst = r[
        "row_abs_true_sum"
    ][
        mask
    ]

    se = r[
        "row_se_sum"
    ][
        mask
    ]

    qa = r[
        "row_qa_ae"
    ][
        mask
    ]

    qaa = r[
        "row_qa_abs"
    ][
        mask
    ]

    qs = r[
        "row_qs_ae"
    ][
        mask
    ]

    qsa = r[
        "row_qs_abs"
    ][
        mask
    ]

    nrows = len(
        np.atleast_1d(
            ae
        )
    )

    st[
        "rows"
    ] += int(
        nrows
    )

    st[
        "ae"
    ] += float(
        np.sum(
            ae
        )
    )

    st[
        "abst"
    ] += float(
        np.sum(
            abst
        )
    )

    st[
        "se"
    ] += float(
        np.sum(
            se
        )
    )

    st[
        "n"
    ] += int(
        nrows
        * 21
    )

    st[
        "qa"
    ] += float(
        np.sum(
            qa
        )
    )

    st[
        "qaa"
    ] += float(
        np.sum(
            qaa
        )
    )

    st[
        "qs"
    ] += float(
        np.sum(
            qs
        )
    )

    st[
        "qsa"
    ] += float(
        np.sum(
            qsa
        )
    )


def finish(
    st,
    split,
    route,
):
    return {
        "split": split,
        "route": route,
        "rows": int(
            st[
                "rows"
            ]
        ),
        "price_mae": float(
            st[
                "ae"
            ]
            / max(
                st[
                    "n"
                ],
                1,
            )
        ),
        "price_rmse": float(
            np.sqrt(
                st[
                    "se"
                ]
                / max(
                    st[
                        "n"
                    ],
                    1,
                )
            )
        ),
        "price_wape_pct": float(
            100.0
            * st[
                "ae"
            ]
            / max(
                st[
                    "abst"
                ],
                1e-12,
            )
        ),
        "q_anchor_wape_pct": float(
            100.0
            * st[
                "qa"
            ]
            / max(
                st[
                    "qaa"
                ],
                1e-12,
            )
        ),
        "q_span_wape_pct": float(
            100.0
            * st[
                "qs"
            ]
            / max(
                st[
                    "qsa"
                ],
                1e-12,
            )
        ),
    }


def evaluate_split(
    dataset,
    manifest,
    split,
    out_dir,
    features,
    targets,
    cluster_model,
    classifier,
    family_specs,
    expert_selection,
    rep_bundle,
    global_rf,
    k,
):
    routes = [
        "global_random_forest",
        "oracle_family_experts",
        "predicted_family_experts",
    ]

    states = {
        x: state()
        for x in routes
    }

    pair_states = defaultdict(
        state
    )

    correct_state = state()
    wrong_state = state()

    y_true_all = []
    y_pred_all = []

    pred_dir = (
        out_dir
        / "prediction_parts"
        / split
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            f"[full pipeline {split} {i}/{len(fs)}] "
            f"{path.name}",
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

        y_true = true_family(
            d,
            targets,
            cluster_model,
        )

        y_pred = classifier.predict(
            X
        ).astype(
            np.int16
        )

        global_z = global_rf.predict(
            X
        ).astype(
            np.float32
        )

        oracle_z = routed_latent(
            d,
            y_true,
            family_specs,
            expert_selection,
            features,
            targets,
        )

        predicted_z = routed_latent(
            d,
            y_pred,
            family_specs,
            expert_selection,
            features,
            targets,
        )

        tq, tp, tqa, tqs = true_curve(
            d
        )

        route_vectors = {
            "global_random_forest": decode(
                global_z,
                rep_bundle,
            ),
            "oracle_family_experts": decode(
                oracle_z,
                rep_bundle,
            ),
            "predicted_family_experts": decode(
                predicted_z,
                rep_bundle,
            ),
        }

        errors = {}

        for route, v in route_vectors.items():
            r = row_errors(
                tq,
                tp,
                tqa,
                tqs,
                v,
            )

            errors[
                route
            ] = r

            update_from_rows(
                states[
                    route
                ],
                r,
            )

        hard_r = errors[
            "predicted_family_experts"
        ]

        correct = (
            y_true
            == y_pred
        )

        update_from_rows(
            correct_state,
            hard_r,
            correct,
        )

        update_from_rows(
            wrong_state,
            hard_r,
            ~correct,
        )

        for t in range(
            k
        ):
            mt = (
                y_true
                == t
            )

            if not mt.any():
                continue

            for p in range(
                k
            ):
                mask = (
                    mt
                    & (
                        y_pred
                        == p
                    )
                )

                if mask.any():
                    update_from_rows(
                        pair_states[
                            (
                                t,
                                p,
                            )
                        ],
                        hard_r,
                        mask,
                    )

        y_true_all.append(
            y_true
        )

        y_pred_all.append(
            y_pred
        )

        keep = [
            c
            for c in META
            if c in d.columns
        ]

        out_d = d[
            keep
        ].copy()

        out_d[
            "true_family_id"
        ] = [
            f"F{x:02d}"
            for x in y_true
        ]

        out_d[
            "pred_family_id"
        ] = [
            f"F{x:02d}"
            for x in y_pred
        ]

        out_d[
            "routing_correct"
        ] = correct

        out_d[
            "global_rf_curve_mae"
        ] = errors[
            "global_random_forest"
        ][
            "row_curve_mae"
        ].astype(
            np.float32
        )

        out_d[
            "oracle_family_curve_mae"
        ] = errors[
            "oracle_family_experts"
        ][
            "row_curve_mae"
        ].astype(
            np.float32
        )

        out_d[
            "predicted_family_curve_mae"
        ] = errors[
            "predicted_family_experts"
        ][
            "row_curve_mae"
        ].astype(
            np.float32
        )

        name = (
            f"{split}_full_family_pipeline_"
            f"{i:04d}.pkl"
        )

        out_path = (
            pred_dir
            / name
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
            y_pred,
            global_z,
            oracle_z,
            predicted_z,
            tq,
            tp,
            tqa,
            tqs,
            route_vectors,
            errors,
            out_d,
        )

        gc.collect()

    yt = np.concatenate(
        y_true_all
    )

    yp = np.concatenate(
        y_pred_all
    )

    cls = {
        "split": split,
        "rows": int(
            len(
                yt
            )
        ),
        "accuracy": float(
            accuracy_score(
                yt,
                yp,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                yt,
                yp,
            )
        ),
        "macro_f1": float(
            f1_score(
                yt,
                yp,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                yt,
                yp,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    metrics = pd.DataFrame(
        [
            finish(
                states[
                    route
                ],
                split,
                route,
            )
            for route in routes
        ]
    )

    condition = pd.DataFrame(
        [
            {
                **finish(
                    correct_state,
                    split,
                    "predicted_family_experts",
                ),
                "routing_condition": "correct_family",
            },
            {
                **finish(
                    wrong_state,
                    split,
                    "predicted_family_experts",
                ),
                "routing_condition": "wrong_family",
            },
        ]
    )

    pairs = []

    for (
        true_f,
        pred_f,
    ), st in sorted(
        pair_states.items()
    ):
        row = finish(
            st,
            split,
            "predicted_family_experts",
        )

        row[
            "true_family_id"
        ] = f"F{true_f:02d}"

        row[
            "pred_family_id"
        ] = f"F{pred_f:02d}"

        pairs.append(
            row
        )

    return (
        metrics,
        pd.DataFrame(
            [cls]
        ),
        condition,
        pd.DataFrame(
            pairs
        ),
        written,
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
        "--representation-dir",
        default="macro_b_absolute_curve_representation",
    )

    ap.add_argument(
        "--family-dir",
        default="absolute_prediction_family_diagnostics",
    )

    ap.add_argument(
        "--classifier-dir",
        default="prediction_family_classifier",
    )

    ap.add_argument(
        "--expert-dir",
        default="oracle_family_regressors",
    )

    ap.add_argument(
        "--global-model-dir",
        default="macro_b_feature_only_regression_models",
    )

    ap.add_argument(
        "--k",
        type=int,
        default=5,
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

    rep_dir = (
        base
        / args.representation_dir
    )

    fam_dir = (
        base
        / args.family_dir
    )

    cls_dir = (
        base
        / args.classifier_dir
    )

    exp_dir = (
        base
        / args.expert_dir
    )

    global_dir = (
        base
        / args.global_model_dir
    )

    manifest = json.loads(
        (
            dataset
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    features = list(
        manifest[
            "model_features"
        ]
    )

    targets = list(
        manifest[
            "latent_columns"
        ]
    )

    family_bundle = joblib.load(
        fam_dir
        / "cluster_models.joblib"
    )

    cluster_model = family_bundle[
        "models"
    ][
        args.k
    ]

    cls_selection = json.loads(
        (
            cls_dir
            / "selected_classifier.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    classifier_name = cls_selection[
        "selected_model"
    ]

    classifier_bundle = joblib.load(
        cls_dir
        / cls_selection[
            "model_file"
        ]
    )

    classifier = classifier_bundle[
        "model"
    ]

    expert_bundle = joblib.load(
        exp_dir
        / "family_regressors.joblib"
    )

    family_specs = expert_bundle[
        "family_specs"
    ]

    expert_selection_payload = json.loads(
        (
            exp_dir
            / "selected_experts.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    expert_selection = expert_selection_payload[
        "selected_expert_by_family"
    ]

    global_rf = joblib.load(
        global_dir
        / "random_forest_model.joblib"
    )[
        "model"
    ]

    rep_bundle = joblib.load(
        rep_dir
        / "absolute_pca_bundle.joblib"
    )

    out_dir = (
        base
        / "full_family_routing_evaluation"
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
        f"09d Full family-routing pipeline - "
        f"{args.year}"
    )
    print("=" * 80)
    print(
        f"Selected family classifier = "
        f"{classifier_name}"
    )
    print(
        "Selected expert by family = "
        + json.dumps(
            expert_selection,
            ensure_ascii=False,
        )
    )
    print()

    all_metrics = []
    all_cls = []
    all_condition = []
    all_pairs = []
    parts_out = {}

    for split in [
        "val",
        "test",
    ]:
        (
            metrics,
            cls,
            condition,
            pairs,
            written,
        ) = evaluate_split(
            dataset,
            manifest,
            split,
            out_dir,
            features,
            targets,
            cluster_model,
            classifier,
            family_specs,
            expert_selection,
            rep_bundle,
            global_rf,
            args.k,
        )

        all_metrics.append(
            metrics
        )

        all_cls.append(
            cls
        )

        all_condition.append(
            condition
        )

        all_pairs.append(
            pairs
        )

        parts_out[
            split
        ] = written

    metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    cls_metrics = pd.concat(
        all_cls,
        ignore_index=True,
    )

    conditions = pd.concat(
        all_condition,
        ignore_index=True,
    )

    pairs = pd.concat(
        all_pairs,
        ignore_index=True,
    )

    metrics.to_csv(
        out_dir
        / "full_pipeline_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cls_metrics.to_csv(
        out_dir
        / "full_pipeline_classifier_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    conditions.to_csv(
        out_dir
        / "routing_correctness_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pairs.to_csv(
        out_dir
        / "routing_pair_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_out = {
        "year": args.year,
        "k": args.k,
        "selected_classifier": classifier_name,
        "selected_expert_by_family": expert_selection,
        "routes": [
            "global_random_forest",
            "oracle_family_experts",
            "predicted_family_experts",
        ],
        "parts": parts_out,
    }

    (
        out_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest_out,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            (
                f"09d Full family-routing pipeline - "
                f"{args.year}"
            ),
            "=" * 80,
            "",
            (
                f"Selected family classifier = "
                f"{classifier_name}"
            ),
            (
                "Selected expert by family = "
                + json.dumps(
                    expert_selection,
                    ensure_ascii=False,
                )
            ),
            "",
            "Classifier metrics:",
            cls_metrics.to_string(
                index=False
            ),
            "",
            "Curve metrics:",
            metrics.to_string(
                index=False
            ),
            "",
            "Predicted-route error conditional on routing correctness:",
            conditions.to_string(
                index=False
            ),
            "",
            (
                "global_random_forest = 08 feature-only global RF; "
                "oracle_family_experts = true family + family expert; "
                "predicted_family_experts = deployable hard routing."
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
    print(summary)
    print()
    print(
        f"Outputs: {out_dir}"
    )


if __name__ == "__main__":
    main()
