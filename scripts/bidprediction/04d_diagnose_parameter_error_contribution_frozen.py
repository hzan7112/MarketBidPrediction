#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04d_diagnose_parameter_error_contribution.py

Frozen-data parameter-error diagnosis.

Reads ONLY:
data/processed/bidprediction/<year>/frozen_modeling_dataset/test_curve_parts/*.pkl

It imports the active 04c_reconstruct_bid_curves.py so the diagnosis uses
exactly the same:
- template predictor,
- theta inverse transform,
- curve reconstruction,
- frozen test rows.

Counterfactual modes:
    pred_all
    true_p_base
    true_alpha
    true_beta
    true_q_base
    true_q_span
    true_q_shares
    true_price_params
    true_quantity_scale
    true_quantity_shape
    true_all_theta

Run:
python scripts/bidprediction/04d_diagnose_parameter_error_contribution.py --year 2025
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PARAM_GROUPS = {
    "pred_all": [],
    "true_p_base": [0],
    "true_alpha": [1],
    "true_beta": [2],
    "true_q_base": [3],
    "true_q_span": [4],
    "true_q_shares": [5, 6, 7, 8, 9],
    "true_price_params": [0, 1, 2],
    "true_quantity_scale": [3, 4],
    "true_quantity_shape": [5, 6, 7, 8, 9],
    "true_all_theta": list(range(10)),
}


def load_04c(project_root: Path):
    path = (
        project_root
        / "scripts"
        / "bidprediction"
        / "04c_reconstruct_bid_curves.py"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "bidprediction_04c",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    required = [
        "TEMPLATES",
        "RAW_THETA",
        "TARGET_TEMPLATE",
        "ORIGIN_TEMPLATE",
        "norm",
        "actual_curve",
        "predict_template",
        "predict_theta",
        "reconstruct",
        "discover_centers",
    ]

    missing = [
        x
        for x in required
        if not hasattr(
            module,
            x,
        )
    ]

    if missing:
        raise AttributeError(
            f"Active 04c missing helpers: {missing}"
        )

    return module, path


def replace_theta(
    pred,
    true,
    indices,
):
    out = pred.copy()

    if indices:
        out[
            :,
            indices,
        ] = true[
            :,
            indices,
        ]

    return out


class Metrics:
    def __init__(self):
        self.n = 0
        self.price_abs = 0.0
        self.price_sq = 0.0
        self.quantity_abs = 0.0
        self.quantity_sq = 0.0
        self.area_abs = 0.0

    def update(
        self,
        q_true,
        p_true,
        q_pred,
        p_pred,
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

            qe = (
                q_pred[i]
                - q_true[i]
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
                    qe
                ).sum()
            )

            self.quantity_sq += float(
                np.square(
                    qe
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

    def row(
        self,
        mode,
    ):
        pts = (
            self.n
            * 21
        )

        return {
            "mode": mode,
            "rows": self.n,
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
            "quantity_grid_rmse_mw": (
                np.sqrt(
                    self.quantity_sq
                    / pts
                )
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
        "--project-root",
        default=".",
    )

    args = ap.parse_args()

    project_root = Path(
        args.project_root
    ).resolve()

    c04, c04_path = load_04c(
        project_root
    )

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

    manifest_file = (
        frozen
        / "manifest.json"
    )

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"{manifest_file}\nRun 04a3_freeze_modeling_dataset.py first."
        )

    manifest = json.loads(
        manifest_file.read_text(
            encoding="utf-8"
        )
    )

    test_parts = [
        frozen
        / p
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

    feature_set = str(
        selected[
            "selected_feature_set"
        ]
    )

    model_name = str(
        selected[
            "selected_model"
        ]
    )

    parameter_bundle = joblib.load(
        parameter_dir
        / "models"
        / f"{feature_set}.joblib"
    )

    template_bundle = joblib.load(
        base
        / "final_template_predictor"
        / "final_template_model.joblib"
    )

    centers, center_file = c04.discover_centers(
        Path(
            args.bidtemplate_root
        )
        / str(
            args.year
        )
    )

    states = {
        mode: Metrics()
        for mode in PARAM_GROUPS
    }

    template_correct = 0
    template_rows = 0

    print("=" * 80)
    print("Parameter error contribution diagnosis - frozen dataset")
    print("=" * 80)
    print(f"Year:                 {args.year}")
    print(f"Active 04c:           {c04_path}")
    print(
        f"Parameter model:      {model_name} / {feature_set}"
    )
    print(
        f"Frozen test rows:     {manifest['test_rows']:,}"
    )
    print()

    for i, p in enumerate(
        test_parts,
        1,
    ):
        print(
            f"[frozen test {i}/{len(test_parts)}] {p.name}",
            flush=True,
        )

        d = pd.read_pickle(
            p
        )

        if d.empty:
            continue

        valid, q_true, p_true, t_true = (
            c04.actual_curve(
                d
            )
        )

        d = d.loc[
            valid
        ].copy()

        q_true = q_true[
            valid
        ]

        p_true = p_true[
            valid
        ]

        t_true = t_true[
            valid
        ]

        if d.empty:
            continue

        origin_valid = (
            c04.norm(
                d[
                    c04.ORIGIN_TEMPLATE
                ]
            )
            .isin(
                c04.TEMPLATES
            )
            .to_numpy()
        )

        if not origin_valid.any():
            continue

        d = d.loc[
            origin_valid
        ].copy()

        q_true = q_true[
            origin_valid
        ]

        p_true = p_true[
            origin_valid
        ]

        t_true = t_true[
            origin_valid
        ]

        true_theta = d[
            c04.RAW_THETA
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(float)

        pred_t, _ = c04.predict_template(
            d,
            template_bundle,
        )

        pred_t_s = pd.Series(
            pred_t,
            index=d.index,
            dtype="string",
        )

        pred_theta = c04.predict_theta(
            parameter_bundle,
            model_name,
            d,
            pred_t_s,
        )

        template_correct += int(
            np.sum(
                pred_t
                == t_true
            )
        )

        template_rows += len(d)

        for mode, indices in PARAM_GROUPS.items():
            theta_cf = replace_theta(
                pred_theta,
                true_theta,
                indices,
            )

            q_pred, p_pred = c04.reconstruct(
                theta_cf,
                pred_t,
                centers,
            )

            states[
                mode
            ].update(
                q_true,
                p_true,
                q_pred,
                p_pred,
            )

    result = pd.DataFrame(
        [
            state.row(
                mode
            )
            for mode, state in states.items()
        ]
    )

    baseline = result.loc[
        result[
            "mode"
        ].eq(
            "pred_all"
        )
    ].iloc[0]

    metrics = [
        "price_mae",
        "price_rmse",
        "quantity_grid_mae_mw",
        "quantity_grid_rmse_mw",
        "curve_area_abs_error",
    ]

    for metric in metrics:
        base_value = float(
            baseline[
                metric
            ]
        )

        result[
            f"{metric}_reduction"
        ] = (
            base_value
            - result[
                metric
            ]
        )

        result[
            f"{metric}_reduction_pct"
        ] = (
            100.0
            * (
                base_value
                - result[
                    metric
                ]
            )
            / base_value
            if base_value != 0
            else np.nan
        )

    out = (
        base
        / "parameter_error_diagnosis"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        out
        / "parameter_error_contribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rank_modes = [
        mode
        for mode in PARAM_GROUPS
        if mode not in {
            "pred_all",
            "true_all_theta",
        }
    ]

    price_rank = (
        result[
            result[
                "mode"
            ].isin(
                rank_modes
            )
        ]
        .sort_values(
            "price_mae_reduction",
            ascending=False,
        )
        [
            [
                "mode",
                "price_mae",
                "price_mae_reduction",
                "price_mae_reduction_pct",
            ]
        ]
    )

    quantity_rank = (
        result[
            result[
                "mode"
            ].isin(
                rank_modes
            )
        ]
        .sort_values(
            "quantity_grid_mae_mw_reduction",
            ascending=False,
        )
        [
            [
                "mode",
                "quantity_grid_mae_mw",
                "quantity_grid_mae_mw_reduction",
                "quantity_grid_mae_mw_reduction_pct",
            ]
        ]
    )

    price_rank.to_csv(
        out
        / "price_error_contribution_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quantity_rank.to_csv(
        out
        / "quantity_error_contribution_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        f"Parameter error contribution diagnosis - frozen dataset - {args.year}",
        "=" * 80,
        "",
        f"Rows = {template_rows:,}",
        (
            f"Template accuracy = "
            f"{template_correct / template_rows:.6f}"
            if template_rows
            else "Template accuracy = nan"
        ),
        (
            f"Parameter model = "
            f"{model_name} / {feature_set}"
        ),
        "",
        "Baseline (all theta predicted):",
        (
            f"  price MAE = "
            f"{baseline['price_mae']:.6f}"
        ),
        (
            f"  price RMSE = "
            f"{baseline['price_rmse']:.6f}"
        ),
        (
            f"  quantity-grid MAE MW = "
            f"{baseline['quantity_grid_mae_mw']:.6f}"
        ),
        (
            f"  curve-area abs error = "
            f"{baseline['curve_area_abs_error']:.6f}"
        ),
        "",
        "Price-MAE contribution ranking:",
    ]

    for r in price_rank.itertuples():
        lines.append(
            f"  {r.mode}: "
            f"MAE={r.price_mae:.6f}, "
            f"reduction={r.price_mae_reduction:.6f} "
            f"({r.price_mae_reduction_pct:.2f}%)"
        )

    lines += [
        "",
        "Quantity-MAE contribution ranking:",
    ]

    for r in quantity_rank.itertuples():
        lines.append(
            f"  {r.mode}: "
            f"MAE={r.quantity_grid_mae_mw:.6f}, "
            f"reduction={r.quantity_grid_mae_mw_reduction:.6f} "
            f"({r.quantity_grid_mae_mw_reduction_pct:.2f}%)"
        )

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
                "active_04c": str(c04_path),
                "selected_parameter_model": model_name,
                "selected_feature_set": feature_set,
                "template_centers": str(center_file),
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
    main()
