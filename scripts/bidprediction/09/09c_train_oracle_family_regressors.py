#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09c_train_oracle_family_regressors.py

Train K=5 family-specific feature-only absolute-latent regressors.

TRUE current family labels are derived from the frozen 09a KMeans model.
Therefore this script measures the UPPER BOUND of family-specific regression
under oracle routing.

Per family candidates:
- family_mean
- Ridge
- Spline-GAM
- Random Forest

For each family, validation reconstructed-curve WAPE selects its expert.
TEST is never used for expert selection.
"""

from __future__ import annotations

import argparse, gc, json, math, shutil
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


GRID = np.linspace(0.0, 1.0, 21)
SHAPE = [f"shape_v{i:02d}" for i in range(21)]
MODELS = ["family_mean", "ridge", "spline_gam", "random_forest"]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def parts(m, split):
    return [
        x["file"] if isinstance(x, dict) else x
        for x in m["parts"][split]
    ]


def nframe(d, cols):
    return d[cols].apply(pd.to_numeric, errors="coerce")


def true_curve(d):
    sh = d[SHAPE].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pa = num(d["p_anchor"]).to_numpy(float)
    ps = num(d["p_span"]).to_numpy(float)
    flat = np.abs(ps) <= 1e-12
    if flat.any():
        sh[flat] = np.nan_to_num(sh[flat], nan=0.0, posinf=0.0, neginf=0.0)
    p = pa[:, None] + ps[:, None] * sh
    qa = num(d["q_anchor_mw"]).to_numpy(float)
    qs = num(d["q_span_mw"]).to_numpy(float)
    q = qa[:, None] + qs[:, None] * GRID[None, :]
    return q, p, qa, qs


def unpack(v):
    return v[:, :21], v[:, 21], np.exp(np.clip(v[:, 22], -20.0, 20.0))


def on_q(q, p, qa, qs):
    pos = np.clip((q - qa[:, None]) / np.maximum(qs[:, None], 1e-8), 0.0, 1.0) * 20.0
    lo = np.floor(pos).astype(np.int16)
    hi = np.minimum(lo + 1, 20)
    f = pos - lo
    return (
        np.take_along_axis(p, lo, 1)
        + f * (
            np.take_along_axis(p, hi, 1)
            - np.take_along_axis(p, lo, 1)
        )
    )


def decode(z, bundle):
    pca = bundle["pca"]
    sc = bundle["scaler"]
    full = np.zeros((len(z), int(pca.n_components_)), float)
    full[:, : z.shape[1]] = z
    return sc.inverse_transform(pca.inverse_transform(full))


def label_rows(d, targets, cluster_model):
    z = nframe(d, targets).to_numpy(np.float64)
    z = np.asarray(z, dtype=cluster_model.cluster_centers_.dtype)
    return cluster_model.predict(z).astype(np.int16)


def top_spearman(d, features, target, k):
    y = pd.to_numeric(d[target], errors="coerce")
    scores = []
    for c in features:
        x = pd.to_numeric(d[c], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 100 or x.loc[ok].nunique() <= 1:
            continue
        r = x.loc[ok].corr(y.loc[ok], method="spearman")
        if pd.notna(r):
            scores.append((c, abs(float(r)), float(r)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def collect_family_samples(
    dataset,
    manifest,
    features,
    targets,
    cluster_model,
    k,
    max_rows_per_family,
    seed,
):
    fs = parts(manifest, "train")
    per_part_family = max(
        1,
        int(math.ceil(max_rows_per_family / max(len(fs), 1))),
    )
    buckets = {i: [] for i in range(k)}

    for i, rel in enumerate(fs, 1):
        path = dataset / rel
        print(f"[TRAIN family sample {i}/{len(fs)}] {path.name}", flush=True)
        d = pd.read_pickle(path)
        if d.empty:
            continue
        y = label_rows(d, targets, cluster_model)

        for fam in range(k):
            idx = np.flatnonzero(y == fam)
            if not len(idx):
                continue
            if len(idx) > per_part_family:
                rng = np.random.default_rng(seed + 100003 * i + fam)
                idx = rng.choice(idx, size=per_part_family, replace=False)
            block = d.iloc[idx][[*features, *targets]].copy()
            buckets[fam].append(block)

        del d, y
        gc.collect()

    out = {}
    for fam in range(k):
        if not buckets[fam]:
            raise ValueError(f"No TRAIN rows for family F{fam:02d}.")
        x = pd.concat(buckets[fam], ignore_index=True)
        if len(x) > max_rows_per_family:
            x = x.sample(
                n=max_rows_per_family,
                random_state=seed + fam,
            ).reset_index(drop=True)
        out[fam] = x

    return out


def fit_family_models(train, features, targets, args, fam):
    mean_z = nframe(train, targets).mean(axis=0).to_numpy(float)

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.ridge_alpha)),
        ]
    )
    ridge.fit(
        nframe(train, features),
        nframe(train, targets).to_numpy(float),
    )

    if len(train) > args.gam_rows_per_family:
        gam_train = train.sample(
            n=args.gam_rows_per_family,
            random_state=args.seed + 17 + fam,
        ).reset_index(drop=True)
    else:
        gam_train = train

    gam = {}
    gam_selection = {}

    for j, t in enumerate(targets, 1):
        print(f"[F{fam:02d} GAM {j}/{len(targets)}] {t}", flush=True)
        top = top_spearman(
            gam_train,
            features,
            t,
            args.gam_top_context,
        )
        f = [x[0] for x in top]
        if not f:
            f = features[: min(args.gam_top_context, len(features))]

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=args.gam_knots,
                        degree=args.gam_degree,
                        knots="quantile",
                        extrapolation="constant",
                        include_bias=False,
                        sparse_output=True,
                    ),
                ),
                ("ridge", Ridge(alpha=args.gam_alpha, solver="lsqr")),
            ]
        )
        model.fit(
            nframe(gam_train, f),
            pd.to_numeric(gam_train[t], errors="coerce").to_numpy(float),
        )
        gam[t] = {"features": f, "model": model}
        gam_selection[t] = {
            "features": f,
            "top_context": [
                {
                    "feature": c,
                    "abs_spearman": a,
                    "spearman": r,
                }
                for c, a, r in top
            ],
        }

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=args.rf_trees,
                    max_depth=args.rf_max_depth,
                    min_samples_leaf=args.rf_min_leaf,
                    max_features=args.rf_max_features,
                    n_jobs=args.n_jobs,
                    random_state=args.seed + fam,
                ),
            ),
        ]
    )
    rf.fit(
        nframe(train, features),
        nframe(train, targets).to_numpy(np.float32),
    )

    return {
        "family_mean": mean_z,
        "ridge": ridge,
        "spline_gam": gam,
        "spline_gam_selection": gam_selection,
        "random_forest": rf,
        "train_rows": int(len(train)),
        "gam_train_rows": int(len(gam_train)),
    }


def pred_gam(d, gam, targets):
    out = np.empty((len(d), len(targets)), dtype=np.float32)
    for j, t in enumerate(targets):
        spec = gam[t]
        out[:, j] = spec["model"].predict(
            nframe(d, spec["features"])
        ).astype(np.float32)
    return out


def predict_model(d, spec, model_name, features, targets):
    if model_name == "family_mean":
        return np.repeat(
            spec["family_mean"][None, :],
            len(d),
            axis=0,
        ).astype(np.float32)
    if model_name == "ridge":
        return spec["ridge"].predict(
            nframe(d, features)
        ).astype(np.float32)
    if model_name == "spline_gam":
        return pred_gam(
            d,
            spec["spline_gam"],
            targets,
        )
    if model_name == "random_forest":
        return spec["random_forest"].predict(
            nframe(d, features)
        ).astype(np.float32)
    raise KeyError(model_name)


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


def update(st, tq, tp, tqa, tqs, v):
    p, qa, qs = unpack(v)
    pred = on_q(tq, p, qa, qs)
    e = pred - tp
    ae = np.abs(e)
    st["rows"] += len(tp)
    st["ae"] += float(ae.sum())
    st["se"] += float((e * e).sum())
    st["abst"] += float(np.abs(tp).sum())
    st["n"] += int(ae.size)
    st["qa"] += float(np.abs(qa - tqa).sum())
    st["qaa"] += float(np.abs(tqa).sum())
    st["qs"] += float(np.abs(qs - tqs).sum())
    st["qsa"] += float(np.abs(tqs).sum())


def finish(st, split, fam, model):
    return {
        "split": split,
        "family_id": f"F{fam:02d}" if isinstance(fam, int) else str(fam),
        "model": model,
        "rows": int(st["rows"]),
        "price_ae_sum": float(st["ae"]),
        "price_abs_true_sum": float(st["abst"]),
        "price_se_sum": float(st["se"]),
        "price_points": int(st["n"]),
        "price_mae": st["ae"] / max(st["n"], 1),
        "price_rmse": float(np.sqrt(st["se"] / max(st["n"], 1))),
        "price_wape_pct": 100.0 * st["ae"] / max(st["abst"], 1e-12),
        "q_anchor_wape_pct": 100.0 * st["qa"] / max(st["qaa"], 1e-12),
        "q_span_wape_pct": 100.0 * st["qs"] / max(st["qsa"], 1e-12),
    }


def evaluate_oracle(
    dataset,
    manifest,
    split,
    features,
    targets,
    cluster_model,
    family_specs,
    bundle,
    k,
):
    states = {
        (fam, model): state()
        for fam in range(k)
        for model in MODELS
    }

    fs = parts(manifest, split)

    for i, rel in enumerate(fs, 1):
        path = dataset / rel
        print(f"[oracle {split} {i}/{len(fs)}] {path.name}", flush=True)
        d = pd.read_pickle(path)
        if d.empty:
            continue

        fam_y = label_rows(
            d,
            targets,
            cluster_model,
        )
        tq, tp, tqa, tqs = true_curve(d)

        for fam in range(k):
            idx = np.flatnonzero(
                fam_y == fam
            )
            if not len(idx):
                continue
            sub = d.iloc[idx]
            spec = family_specs[fam]

            for model in MODELS:
                z = predict_model(
                    sub,
                    spec,
                    model,
                    features,
                    targets,
                )
                v = decode(
                    z,
                    bundle,
                )
                update(
                    states[(fam, model)],
                    tq[idx],
                    tp[idx],
                    tqa[idx],
                    tqs[idx],
                    v,
                )

        del d, fam_y, tq, tp, tqa, tqs
        gc.collect()

    return pd.DataFrame(
        [
            finish(
                st,
                split,
                fam,
                model,
            )
            for (fam, model), st in states.items()
            if st["rows"] > 0
        ]
    )


def aggregate_selected(
    table,
    selection,
    split,
):
    rows = []

    for fam_id, model in selection.items():
        x = table.loc[
            table["family_id"].eq(fam_id)
            & table["model"].eq(model)
        ]
        if x.empty:
            raise RuntimeError(
                f"Missing {split} metrics for {fam_id}/{model}"
            )
        rows.append(
            x.iloc[0]
        )

    df = pd.DataFrame(rows)
    ae = float(df["price_ae_sum"].sum())
    abst = float(df["price_abs_true_sum"].sum())
    se = float(df["price_se_sum"].sum())
    n = int(df["price_points"].sum())

    return {
        "split": split,
        "routing": "oracle_true_family",
        "expert_selection": "per-family validation WAPE",
        "rows": int(df["rows"].sum()),
        "price_mae": ae / max(n, 1),
        "price_rmse": float(np.sqrt(se / max(n, 1))),
        "price_wape_pct": 100.0 * ae / max(abst, 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--root", default="data/processed/bidprediction")
    ap.add_argument("--dataset-dir", default="macro_b_feature_only_dataset")
    ap.add_argument("--representation-dir", default="macro_b_absolute_curve_representation")
    ap.add_argument("--family-dir", default="absolute_prediction_family_diagnostics")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-rows-per-family", type=int, default=150_000)
    ap.add_argument("--gam-rows-per-family", type=int, default=75_000)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--gam-alpha", type=float, default=1.0)
    ap.add_argument("--gam-top-context", type=int, default=24)
    ap.add_argument("--gam-knots", type=int, default=5)
    ap.add_argument("--gam-degree", type=int, default=2)
    ap.add_argument("--rf-trees", type=int, default=128)
    ap.add_argument("--rf-max-depth", type=int, default=18)
    ap.add_argument("--rf-min-leaf", type=int, default=8)
    ap.add_argument("--rf-max-features", type=float, default=0.5)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    dataset = base / args.dataset_dir
    rep_dir = base / args.representation_dir
    fam_dir = base / args.family_dir

    manifest = json.loads(
        (dataset / "manifest.json").read_text(encoding="utf-8")
    )
    features = list(manifest["model_features"])
    targets = list(manifest["latent_columns"])

    family_bundle = joblib.load(
        fam_dir / "cluster_models.joblib"
    )
    cluster_model = family_bundle["models"][args.k]

    bundle = joblib.load(
        rep_dir / "absolute_pca_bundle.joblib"
    )

    out_dir = base / "oracle_family_regressors"
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print("=" * 80)
    print(f"09c Oracle-family regressors - {args.year}")
    print("=" * 80)
    print(f"K = {args.k}")
    print(f"Features = {len(features)}")
    print()

    samples = collect_family_samples(
        dataset,
        manifest,
        features,
        targets,
        cluster_model,
        args.k,
        args.max_rows_per_family,
        args.seed,
    )

    family_specs = {}

    for fam in range(args.k):
        print()
        print(f"[fit family F{fam:02d}] rows={len(samples[fam]):,}")
        family_specs[fam] = fit_family_models(
            samples[fam],
            features,
            targets,
            args,
            fam,
        )
        del samples[fam]
        gc.collect()

    model_bundle = {
        "k": args.k,
        "features": features,
        "targets": targets,
        "family_specs": family_specs,
    }

    joblib.dump(
        model_bundle,
        out_dir / "family_regressors.joblib",
        compress=3,
    )

    val = evaluate_oracle(
        dataset,
        manifest,
        "val",
        features,
        targets,
        cluster_model,
        family_specs,
        bundle,
        args.k,
    )

    selection = {}

    for fam in range(args.k):
        fam_id = f"F{fam:02d}"
        x = (
            val.loc[
                val["family_id"].eq(fam_id)
            ]
            .sort_values(
                ["price_wape_pct", "price_mae"]
            )
            .reset_index(drop=True)
        )
        if x.empty:
            raise RuntimeError(
                f"No validation rows for {fam_id}."
            )
        selection[fam_id] = str(
            x.iloc[0]["model"]
        )

    test = evaluate_oracle(
        dataset,
        manifest,
        "test",
        features,
        targets,
        cluster_model,
        family_specs,
        bundle,
        args.k,
    )

    val.to_csv(
        out_dir / "validation_oracle_family_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test.to_csv(
        out_dir / "test_oracle_family_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall = pd.DataFrame(
        [
            aggregate_selected(
                val,
                selection,
                "val",
            ),
            aggregate_selected(
                test,
                selection,
                "test",
            ),
        ]
    )

    overall.to_csv(
        out_dir / "oracle_selected_expert_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selection_payload = {
        "selection_split": "validation",
        "selection_metric": "reconstructed curve WAPE within true family",
        "test_used_for_selection": False,
        "selected_expert_by_family": selection,
    }

    (
        out_dir / "selected_experts.json"
    ).write_text(
        json.dumps(
            selection_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest_out = {
        "year": args.year,
        "k": args.k,
        "source_dataset": str(dataset),
        "model_features": features,
        "latent_columns": targets,
        "candidate_experts": MODELS,
        "selected_expert_by_family": selection,
    }

    (
        out_dir / "manifest.json"
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
            f"09c Oracle-family regressors - {args.year}",
            "=" * 80,
            "",
            f"K = {args.k}",
            f"Features = {len(features)}",
            "",
            "Selected expert by family (VALIDATION only):",
            json.dumps(
                selection,
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "VALIDATION per-family candidates:",
            val.to_string(index=False),
            "",
            "TEST per-family candidates:",
            test.to_string(index=False),
            "",
            "Oracle-routing selected-expert overall:",
            overall.to_string(index=False),
        ]
    )

    (
        out_dir / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
