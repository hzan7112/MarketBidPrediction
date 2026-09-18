#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_validate_profile_template_relation.py

Validate whether the interpretable strategy profile is materially associated
with the bid-curve template library.

Three complementary checks are used:
1. LT relation: Spearman correlation between each of the 9 LT dimensions and
   each participant's annual template usage share.
2. ST relation: Spearman correlation between each of the 9 ST states and the
   participant-day template usage shares; Break features are evaluated by the
   shift in template distribution between break=0 and break=1 days.
3. Multivariate diagnostic: predict the daily dominant template using
   LT / ST / Break / combined feature groups under both temporal holdout and
   participant holdout. This is a relation diagnostic, not the final forecasting
   model.

Outputs
-------
data/processed/bidtemplate/<year>/profile_template_relation/validation/
    lt_template_spearman.csv
    st_template_spearman.csv
    break_template_distribution_shift.csv
    feature_template_mutual_information.csv
    relation_model_metrics.csv
    relation_summary.csv
    lt_template_correlation_heatmap.png
    st_template_correlation_heatmap.png
    relation_model_metrics.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def template_share_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("template_share_")]


def safe_spearman(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < 10:
        return n, np.nan, np.nan
    xv = x[ok].to_numpy(float)
    yv = y[ok].to_numpy(float)
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return n, np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = spearmanr(xv, yv)
    return n, float(r), float(p)


def build_spearman_table(df, features, share_cols, scope):
    rows = []
    for f in features:
        for s in share_cols:
            n, rho, p = safe_spearman(df[f], df[s])
            rows.append(
                {
                    "scope": scope,
                    "feature": f,
                    "template_id": s.replace("template_share_", ""),
                    "n": n,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
                    "p_value": p,
                }
            )
    return pd.DataFrame(rows)


def break_shift_table(df, break_features, share_cols):
    rows = []
    for f in break_features:
        b = pd.to_numeric(df[f], errors="coerce")
        valid = b.isin([0, 1])
        d = df.loc[valid].copy()
        b = b.loc[valid].astype(int)
        n0 = int((b == 0).sum())
        n1 = int((b == 1).sum())

        if n0 == 0 or n1 == 0:
            rows.append(
                {
                    "break_feature": f,
                    "n_break0": n0,
                    "n_break1": n1,
                    "break_rate": n1 / (n0 + n1) if n0 + n1 else np.nan,
                    "template_total_variation_distance": np.nan,
                    "max_abs_template_share_shift": np.nan,
                    "most_shifted_template": "",
                }
            )
            continue

        m0 = d.loc[b == 0, share_cols].apply(pd.to_numeric, errors="coerce").mean()
        m1 = d.loc[b == 1, share_cols].apply(pd.to_numeric, errors="coerce").mean()
        delta = m1 - m0
        tv = 0.5 * float(np.nansum(np.abs(delta.to_numpy(float))))
        idx = delta.abs().idxmax()

        rows.append(
            {
                "break_feature": f,
                "n_break0": n0,
                "n_break1": n1,
                "break_rate": n1 / (n0 + n1),
                "template_total_variation_distance": tv,
                "max_abs_template_share_shift": float(abs(delta[idx])),
                "most_shifted_template": idx.replace("template_share_", ""),
            }
        )
    return pd.DataFrame(rows)


def mutual_information_table(df, features, target, max_rows, seed):
    cols = features + [target]
    d = df[cols].copy()
    d = d[d[target].notna()]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)

    X = d[features].apply(pd.to_numeric, errors="coerce")
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=features,
        index=X.index,
    )

    le = LabelEncoder()
    y = le.fit_transform(d[target].astype(str))

    discrete = np.array([f in BREAK_FEATURES for f in features], dtype=bool)
    mi = mutual_info_classif(
        X.to_numpy(float),
        y,
        discrete_features=discrete,
        random_state=seed,
    )

    probs = np.bincount(y) / len(y)
    h = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))

    out = pd.DataFrame(
        {
            "feature": features,
            "mutual_information_nats": mi,
            "fraction_of_target_entropy": mi / h if h > 0 else np.nan,
            "sample_n": len(d),
        }
    ).sort_values("mutual_information_nats", ascending=False)
    return out


def subsample(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    if max_n is None or len(df) <= max_n:
        return df
    return df.sample(max_n, random_state=seed)


def evaluate_predictions(y_true, y_pred, proba, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
    }


def fit_eval(train, test, features, target, seed):
    train = train.dropna(subset=[target]).copy()
    test = test.dropna(subset=[target]).copy()

    le = LabelEncoder()
    le.fit(pd.concat([train[target], test[target]], ignore_index=True).astype(str))
    y_train = le.transform(train[target].astype(str))
    y_test = le.transform(test[target].astype(str))

    X_train = train[features].apply(pd.to_numeric, errors="coerce")
    X_test = test[features].apply(pd.to_numeric, errors="coerce")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=120,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    model_proba = model.predict_proba(X_test)
    model_classes = model.named_steps["model"].classes_.astype(int)

    proba = np.full((len(test), len(le.classes_)), 1e-15, dtype=float)
    proba[:, model_classes] = model_proba
    proba /= proba.sum(axis=1, keepdims=True)

    metrics = evaluate_predictions(
        y_test,
        pred,
        proba,
        labels=np.arange(len(le.classes_)),
    )

    # Majority baseline, with a smoothed train-frequency probability vector.
    counts = np.bincount(y_train, minlength=len(le.classes_)).astype(float)
    majority = int(np.argmax(counts))
    base_pred = np.full(len(y_test), majority, dtype=int)
    base_prob_vec = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
    base_proba = np.repeat(base_prob_vec[None, :], len(y_test), axis=0)
    baseline = evaluate_predictions(
        y_test,
        base_pred,
        base_proba,
        labels=np.arange(len(le.classes_)),
    )

    return metrics, baseline, len(train), len(test), len(le.classes_)


def model_diagnostics(df, feature_sets, target, max_train, max_test, seed, time_frac):
    rows = []

    # Temporal holdout.
    dates = np.sort(pd.to_datetime(df["local_date"], errors="coerce").dropna().unique())
    cut_idx = max(1, min(len(dates) - 1, int(len(dates) * time_frac)))
    cutoff = pd.Timestamp(dates[cut_idx])
    train_time = df[pd.to_datetime(df["local_date"]) < cutoff]
    test_time = df[pd.to_datetime(df["local_date"]) >= cutoff]
    train_time = subsample(train_time, max_train, seed)
    test_time = subsample(test_time, max_test, seed + 1)

    # Participant holdout.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    idx_train, idx_test = next(
        gss.split(df, groups=df["participant_id"].astype(str))
    )
    train_group = subsample(df.iloc[idx_train], max_train, seed + 2)
    test_group = subsample(df.iloc[idx_test], max_test, seed + 3)

    splits = [
        ("temporal_holdout", train_time, test_time, str(cutoff.date())),
        ("participant_holdout", train_group, test_group, "20% participants held out"),
    ]

    for split_name, train, test, detail in splits:
        for set_name, features in feature_sets.items():
            print(
                f"[model] split={split_name:19s} features={set_name:12s} "
                f"train={len(train):,} test={len(test):,}",
                flush=True,
            )
            metrics, baseline, ntr, nte, ncls = fit_eval(
                train, test, features, target, seed
            )
            row = {
                "split": split_name,
                "split_detail": detail,
                "feature_set": set_name,
                "feature_count": len(features),
                "train_n": ntr,
                "test_n": nte,
                "target_classes": ncls,
                **metrics,
                "baseline_accuracy": baseline["accuracy"],
                "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                "baseline_macro_f1": baseline["macro_f1"],
                "baseline_weighted_f1": baseline["weighted_f1"],
                "baseline_log_loss": baseline["log_loss"],
            }
            row["accuracy_gain"] = row["accuracy"] - row["baseline_accuracy"]
            row["macro_f1_gain"] = row["macro_f1"] - row["baseline_macro_f1"]
            rows.append(row)

    return pd.DataFrame(rows)


def plot_corr_heatmap(table, features, templates, out_file, title):
    mat = (
        table.pivot(index="feature", columns="template_id", values="spearman_rho")
        .reindex(index=features, columns=templates)
    )
    fig_w = max(10, 0.75 * len(templates) + 4)
    fig_h = max(6, 0.55 * len(features) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.to_numpy(float), aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(templates)), labels=templates, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(features)), labels=features)
    ax.set_title(title)
    ax.set_xlabel("Template")
    ax.set_ylabel("Profile feature")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_model_metrics(metrics, out_file):
    plot = metrics.copy()
    labels = [f"{r.split}\n{r.feature_set}" for r in plot.itertuples()]
    x = np.arange(len(plot))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, 0.9 * len(plot)), 6.5))
    ax.bar(x - width / 2, plot["accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, plot["macro_f1"], width, label="Macro F1")
    ax.axhline(plot["baseline_accuracy"].iloc[0], linestyle="--", linewidth=1, label="Majority accuracy (reference)")
    ax.set_xticks(x, labels=labels, rotation=55, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Profile-template relation diagnostic")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument(
        "--root",
        default="data/processed/bidtemplate",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mi-max-rows", type=int, default=100_000)
    p.add_argument("--model-max-train", type=int, default=150_000)
    p.add_argument("--model-max-test", type=int, default=60_000)
    p.add_argument("--time-train-frac", type=float, default=0.80)
    args = p.parse_args()

    base = Path(args.root) / str(args.year) / "profile_template_relation"
    daily_file = base / f"profile_template_daily_{args.year}.csv"
    participant_file = base / f"profile_template_participant_{args.year}.csv"
    if not daily_file.exists():
        raise FileNotFoundError(daily_file)
    if not participant_file.exists():
        raise FileNotFoundError(participant_file)

    out_dir = ensure_dir(base / "validation")

    daily = pd.read_csv(daily_file, low_memory=False)
    participant = pd.read_csv(participant_file, low_memory=False)
    daily["local_date"] = pd.to_datetime(daily["local_date"], errors="coerce")

    share_cols_daily = template_share_columns(daily)
    share_cols_participant = template_share_columns(participant)
    templates = [c.replace("template_share_", "") for c in share_cols_daily]

    # Use ready rows for the intended profile definitions.
    lt_df = participant[participant["lt_ready_flag"].eq(1)].copy()
    st_df = daily[daily["st_ready_flag"].eq(1)].copy()
    full_df = daily[daily["profile_ready_flag"].eq(1)].copy()

    print("=" * 72)
    print("Profile-template relation validation")
    print("=" * 72)
    print(f"LT-ready participants: {len(lt_df):,}")
    print(f"ST-ready days:         {len(st_df):,}")
    print(f"Full-profile days:     {len(full_df):,}")
    print(f"Templates:             {len(templates)}")

    lt_corr = build_spearman_table(
        lt_df, LT_FEATURES, share_cols_participant, "participant_LT"
    )
    st_corr = build_spearman_table(
        st_df, ST_FEATURES, share_cols_daily, "participant_day_ST"
    )
    br_shift = break_shift_table(st_df, BREAK_FEATURES, share_cols_daily)

    mi = mutual_information_table(
        full_df,
        LT_FEATURES + ST_FEATURES + BREAK_FEATURES,
        "dominant_template_id",
        args.mi_max_rows,
        args.seed,
    )

    feature_sets = {
        "LT_9": LT_FEATURES,
        "ST_9": ST_FEATURES,
        "Break_8": BREAK_FEATURES,
        "STBreak_17": ST_FEATURES + BREAK_FEATURES,
        "Full_26": LT_FEATURES + ST_FEATURES + BREAK_FEATURES,
    }
    metrics = model_diagnostics(
        full_df,
        feature_sets,
        "dominant_template_id",
        args.model_max_train,
        args.model_max_test,
        args.seed,
        args.time_train_frac,
    )

    lt_file = out_dir / "lt_template_spearman.csv"
    st_file = out_dir / "st_template_spearman.csv"
    br_file = out_dir / "break_template_distribution_shift.csv"
    mi_file = out_dir / "feature_template_mutual_information.csv"
    metric_file = out_dir / "relation_model_metrics.csv"

    lt_corr.to_csv(lt_file, index=False, encoding="utf-8-sig")
    st_corr.to_csv(st_file, index=False, encoding="utf-8-sig")
    br_shift.to_csv(br_file, index=False, encoding="utf-8-sig")
    mi.to_csv(mi_file, index=False, encoding="utf-8-sig")
    metrics.to_csv(metric_file, index=False, encoding="utf-8-sig")

    plot_corr_heatmap(
        lt_corr,
        LT_FEATURES,
        templates,
        out_dir / "lt_template_correlation_heatmap.png",
        "Long-term profile vs annual template preference",
    )
    plot_corr_heatmap(
        st_corr,
        ST_FEATURES,
        templates,
        out_dir / "st_template_correlation_heatmap.png",
        "Short-term state vs daily template share",
    )
    plot_model_metrics(metrics, out_dir / "relation_model_metrics.png")

    # Compact summary for quick review.
    lt_rank = (
        lt_corr.groupby("feature", observed=True)["abs_spearman_rho"]
        .max()
        .sort_values(ascending=False)
    )
    st_rank = (
        st_corr.groupby("feature", observed=True)["abs_spearman_rho"]
        .max()
        .sort_values(ascending=False)
    )
    best_full = metrics[metrics["feature_set"].eq("Full_26")].copy()

    summary_rows = []
    for f, v in lt_rank.items():
        summary_rows.append({"section": "LT_max_abs_spearman", "item": f, "value": v})
    for f, v in st_rank.items():
        summary_rows.append({"section": "ST_max_abs_spearman", "item": f, "value": v})
    for r in br_shift.sort_values("template_total_variation_distance", ascending=False).itertuples():
        summary_rows.append(
            {
                "section": "Break_template_TV_distance",
                "item": r.break_feature,
                "value": r.template_total_variation_distance,
            }
        )
    for r in best_full.itertuples():
        summary_rows.append(
            {
                "section": f"Full26_{r.split}",
                "item": "accuracy",
                "value": r.accuracy,
            }
        )
        summary_rows.append(
            {
                "section": f"Full26_{r.split}",
                "item": "macro_f1",
                "value": r.macro_f1,
            }
        )
        summary_rows.append(
            {
                "section": f"Full26_{r.split}",
                "item": "accuracy_gain_over_majority",
                "value": r.accuracy_gain,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_file = out_dir / "relation_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print()
    print("[Top LT associations: max |Spearman rho| across templates]")
    print(lt_rank.head(9).to_string())
    print()
    print("[Top ST associations: max |Spearman rho| across templates]")
    print(st_rank.head(9).to_string())
    print()
    print("[Break distribution shifts: total variation distance]")
    print(
        br_shift[["break_feature", "break_rate", "template_total_variation_distance"]]
        .sort_values("template_total_variation_distance", ascending=False)
        .to_string(index=False)
    )
    print()
    print("[Model diagnostic]")
    print(
        metrics[[
            "split", "feature_set", "train_n", "test_n",
            "accuracy", "balanced_accuracy", "macro_f1",
            "baseline_accuracy", "accuracy_gain",
        ]].to_string(index=False)
    )
    print()
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
