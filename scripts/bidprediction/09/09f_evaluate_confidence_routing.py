#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, gc, json, shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

GRID = np.linspace(0.0, 1.0, 21)
SHAPE = [f"shape_v{i:02d}" for i in range(21)]

def num(s):
    return pd.to_numeric(s, errors="coerce")

def parts(m, split):
    return [x["file"] if isinstance(x, dict) else x for x in m["parts"][split]]

def nframe(d, cols):
    return d[cols].apply(pd.to_numeric, errors="coerce")

def true_curve(d):
    sh = d[SHAPE].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    pa = num(d["p_anchor"]).to_numpy(np.float64)
    ps = num(d["p_span"]).to_numpy(np.float64)
    flat = np.abs(ps) <= 1e-12
    if flat.any():
        sh[flat] = np.nan_to_num(sh[flat], nan=0.0, posinf=0.0, neginf=0.0)
    p = pa[:, None] + ps[:, None] * sh
    qa = num(d["q_anchor_mw"]).to_numpy(np.float64)
    qs = num(d["q_span_mw"]).to_numpy(np.float64)
    q = qa[:, None] + qs[:, None] * GRID[None, :]
    return q, p, qa, qs

def unpack(v):
    v = np.asarray(v, dtype=np.float64)
    return v[:, :21], v[:, 21], np.exp(np.clip(v[:, 22], -20.0, 20.0))

def on_true_q(q, p, qa, qs):
    pos = np.clip((q - qa[:, None]) / np.maximum(qs[:, None], 1e-8), 0.0, 1.0) * 20.0
    lo = np.floor(pos).astype(np.int16)
    hi = np.minimum(lo + 1, 20)
    f = pos - lo
    plo = np.take_along_axis(p, lo, axis=1)
    phi = np.take_along_axis(p, hi, axis=1)
    return plo + f * (phi - plo)

def decode(z, bundle):
    z = np.asarray(z, dtype=np.float64)
    pca = bundle["pca"]
    sc = bundle["scaler"]
    full = np.zeros((len(z), int(pca.n_components_)), dtype=np.float64)
    full[:, :z.shape[1]] = z
    return sc.inverse_transform(pca.inverse_transform(full))

def get_true_family(d, targets, cluster_model):
    z = nframe(d, targets).to_numpy(np.float64)
    z = np.asarray(z, dtype=cluster_model.cluster_centers_.dtype)
    return cluster_model.predict(z).astype(np.int16)

def pred_gam(d, gam, targets):
    out = np.empty((len(d), len(targets)), dtype=np.float32)
    for j, t in enumerate(targets):
        spec = gam[t]
        out[:, j] = spec["model"].predict(
            nframe(d, spec["features"])
        ).astype(np.float32)
    return out

def pred_expert(d, spec, name, features, targets):
    if name == "family_mean":
        return np.repeat(spec["family_mean"][None, :], len(d), axis=0).astype(np.float32)
    if name == "ridge":
        return spec["ridge"].predict(nframe(d, features)).astype(np.float32)
    if name == "spline_gam":
        return pred_gam(d, spec["spline_gam"], targets)
    if name == "random_forest":
        return spec["random_forest"].predict(nframe(d, features)).astype(np.float32)
    raise KeyError(name)

def routed_latent(d, fam, specs, selected, features, targets):
    out = np.empty((len(d), len(targets)), dtype=np.float32)
    for f in sorted(specs):
        idx = np.flatnonzero(fam == f)
        if not len(idx):
            continue
        fid = f"F{f:02d}"
        out[idx] = pred_expert(
            d.iloc[idx],
            specs[f],
            selected[fid],
            features,
            targets,
        )
    return out

def row_errors(tq, tp, tqa, tqs, vec):
    p, qa, qs = unpack(vec)
    pred = on_true_q(tq, p, qa, qs)
    e = pred - tp
    ae = np.abs(e)
    return {
        "ae": ae.sum(axis=1),
        "se": np.square(e).sum(axis=1),
        "abst": np.abs(tp).sum(axis=1),
        "qa": np.abs(qa - tqa),
        "qaa": np.abs(tqa),
        "qs": np.abs(qs - tqs),
        "qsa": np.abs(tqs),
    }

def state():
    return dict(rows=0, ae=0.0, se=0.0, abst=0.0, points=0,
                qa=0.0, qaa=0.0, qs=0.0, qsa=0.0)

def update(st, r, mask=None):
    if mask is None:
        mask = np.ones(len(r["ae"]), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return
    st["rows"] += n
    st["ae"] += float(r["ae"][mask].sum())
    st["se"] += float(r["se"][mask].sum())
    st["abst"] += float(r["abst"][mask].sum())
    st["points"] += n * 21
    st["qa"] += float(r["qa"][mask].sum())
    st["qaa"] += float(r["qaa"][mask].sum())
    st["qs"] += float(r["qs"][mask].sum())
    st["qsa"] += float(r["qsa"][mask].sum())

def finish(st, split, route, **extra):
    row = {
        "split": split,
        "route": route,
        "rows": int(st["rows"]),
        "price_mae": st["ae"] / max(st["points"], 1),
        "price_rmse": float(np.sqrt(st["se"] / max(st["points"], 1))),
        "price_wape_pct": 100.0 * st["ae"] / max(st["abst"], 1e-12),
        "q_anchor_wape_pct": 100.0 * st["qa"] / max(st["qaa"], 1e-12),
        "q_span_wape_pct": 100.0 * st["qs"] / max(st["qsa"], 1e-12),
    }
    row.update(extra)
    return row

def parse_thresholds(s):
    x = sorted({float(v.strip()) for v in s.split(",") if v.strip()})
    if not x or any(v < 0.0 or v > 1.0 for v in x):
        raise ValueError("Invalid thresholds.")
    return x

def evaluate_split(
    dataset, manifest, split, features, targets,
    cluster_model, classifier, specs, selected,
    global_rf, rep_bundle, thresholds, bins,
):
    base = {
        "global_random_forest": state(),
        "hard_family_routing": state(),
        "oracle_family_routing": state(),
    }
    sweep = {t: state() for t in thresholds}
    sweep_family_rows = {t: 0 for t in thresholds}
    correct_st, wrong_st = state(), state()

    bin_stats = {}
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j + 1]
        label = f"[{lo:.2f},{hi:.2f})" if j < len(bins) - 2 else f"[{lo:.2f},{hi:.2f}]"
        bin_stats[label] = {
            "lo": lo, "hi": hi, "rows": 0, "correct": 0,
            "global": state(), "hard": state(), "oracle": state(),
        }

    total = 0
    correct_total = 0
    fs = parts(manifest, split)

    for i, rel in enumerate(fs, 1):
        path = dataset / rel
        print(f"[09f {split} {i}/{len(fs)}] {path.name}", flush=True)
        d = pd.read_pickle(path)
        if d.empty:
            continue

        X = nframe(d, features)
        y_true = get_true_family(d, targets, cluster_model)

        if not hasattr(classifier, "predict_proba"):
            raise RuntimeError("Selected classifier has no predict_proba().")

        proba = classifier.predict_proba(X)
        classes = np.asarray(classifier.classes_, dtype=int)
        arg = np.argmax(proba, axis=1)
        y_pred = classes[arg].astype(np.int16)
        pmax = proba[np.arange(len(d)), arg].astype(np.float64)

        z_global = global_rf.predict(X).astype(np.float32)
        z_hard = routed_latent(d, y_pred, specs, selected, features, targets)
        z_oracle = routed_latent(d, y_true, specs, selected, features, targets)

        tq, tp, tqa, tqs = true_curve(d)

        err_global = row_errors(tq, tp, tqa, tqs, decode(z_global, rep_bundle))
        err_hard = row_errors(tq, tp, tqa, tqs, decode(z_hard, rep_bundle))
        err_oracle = row_errors(tq, tp, tqa, tqs, decode(z_oracle, rep_bundle))

        update(base["global_random_forest"], err_global)
        update(base["hard_family_routing"], err_hard)
        update(base["oracle_family_routing"], err_oracle)

        correct = y_true == y_pred
        update(correct_st, err_hard, correct)
        update(wrong_st, err_hard, ~correct)

        total += len(d)
        correct_total += int(correct.sum())

        for t in thresholds:
            use_family = pmax >= t
            sweep_family_rows[t] += int(use_family.sum())
            mixed = {
                k: np.where(use_family, err_hard[k], err_global[k])
                for k in err_global
            }
            update(sweep[t], mixed)

        for j in range(len(bins) - 1):
            lo, hi = bins[j], bins[j + 1]
            label = f"[{lo:.2f},{hi:.2f})" if j < len(bins) - 2 else f"[{lo:.2f},{hi:.2f}]"
            mask = ((pmax >= lo) & (pmax < hi)) if j < len(bins) - 2 else ((pmax >= lo) & (pmax <= hi))
            n = int(mask.sum())
            if n == 0:
                continue
            bs = bin_stats[label]
            bs["rows"] += n
            bs["correct"] += int(correct[mask].sum())
            update(bs["global"], err_global, mask)
            update(bs["hard"], err_hard, mask)
            update(bs["oracle"], err_oracle, mask)

        del d, X, y_true, proba, y_pred, pmax, z_global, z_hard, z_oracle
        del tq, tp, tqa, tqs, err_global, err_hard, err_oracle, correct
        gc.collect()

    base_df = pd.DataFrame([
        finish(base[k], split, k)
        for k in ["global_random_forest", "hard_family_routing", "oracle_family_routing"]
    ])

    sweep_rows = []
    for t in thresholds:
        fr = sweep_family_rows[t]
        sweep_rows.append(
            finish(
                sweep[t],
                split,
                "confidence_family_routing",
                threshold=float(t),
                family_expert_rows=int(fr),
                global_fallback_rows=int(total - fr),
                family_expert_share=fr / max(total, 1),
                global_fallback_share=(total - fr) / max(total, 1),
            )
        )
    sweep_df = pd.DataFrame(sweep_rows)

    bin_rows = []
    for label, bs in bin_stats.items():
        if bs["rows"] == 0:
            continue
        g = finish(bs["global"], split, "global")
        h = finish(bs["hard"], split, "hard")
        o = finish(bs["oracle"], split, "oracle")
        bin_rows.append({
            "split": split,
            "confidence_bin": label,
            "pmax_lo": bs["lo"],
            "pmax_hi": bs["hi"],
            "rows": int(bs["rows"]),
            "row_share": bs["rows"] / max(total, 1),
            "family_accuracy": bs["correct"] / max(bs["rows"], 1),
            "global_rf_wape_pct": g["price_wape_pct"],
            "hard_routing_wape_pct": h["price_wape_pct"],
            "oracle_routing_wape_pct": o["price_wape_pct"],
            "hard_minus_global_wape_pp": h["price_wape_pct"] - g["price_wape_pct"],
        })

    cond_df = pd.DataFrame([
        finish(correct_st, split, "hard_family_routing", routing_condition="correct_family"),
        finish(wrong_st, split, "hard_family_routing", routing_condition="wrong_family"),
    ])

    return base_df, sweep_df, pd.DataFrame(bin_rows), cond_df, correct_total / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--root", default="data/processed/bidprediction")
    ap.add_argument("--dataset-dir", default="macro_b_feature_only_dataset")
    ap.add_argument("--representation-dir", default="macro_b_absolute_curve_representation")
    ap.add_argument("--family-dir", default="absolute_prediction_family_diagnostics")
    ap.add_argument("--classifier-dir", default="prediction_family_classifier")
    ap.add_argument("--expert-dir", default="oracle_family_regressors")
    ap.add_argument("--global-model-dir", default="macro_b_feature_only_regression_models")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--thresholds",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    bins = [0.00, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]

    base = Path(args.root) / str(args.year)
    dataset = base / args.dataset_dir
    rep_dir = base / args.representation_dir
    fam_dir = base / args.family_dir
    cls_dir = base / args.classifier_dir
    exp_dir = base / args.expert_dir
    global_dir = base / args.global_model_dir

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    features = list(manifest["model_features"])
    targets = list(manifest["latent_columns"])

    family_bundle = joblib.load(fam_dir / "cluster_models.joblib")
    cluster_model = family_bundle["models"][args.k]

    cls_sel = json.loads((cls_dir / "selected_classifier.json").read_text(encoding="utf-8"))
    classifier_name = cls_sel["selected_model"]
    classifier = joblib.load(cls_dir / cls_sel["model_file"])["model"]

    exp_bundle = joblib.load(exp_dir / "family_regressors.joblib")
    specs = exp_bundle["family_specs"]
    selected = json.loads((exp_dir / "selected_experts.json").read_text(encoding="utf-8"))[
        "selected_expert_by_family"
    ]

    global_rf = joblib.load(global_dir / "random_forest_model.joblib")["model"]
    rep_bundle = joblib.load(rep_dir / "absolute_pca_bundle.joblib")

    out_dir = base / "confidence_family_routing"
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print("=" * 80)
    print(f"09f Confidence family routing - {args.year}")
    print("=" * 80)
    print(f"K = {args.k}")
    print(f"Selected classifier = {classifier_name}")
    print(f"Selected experts = {json.dumps(selected, ensure_ascii=False)}")
    print(f"Thresholds = {thresholds}")
    print()

    val_base, val_sweep, val_bins, val_cond, val_acc = evaluate_split(
        dataset, manifest, "val", features, targets,
        cluster_model, classifier, specs, selected,
        global_rf, rep_bundle, thresholds, bins,
    )

    val_sweep_sorted = val_sweep.sort_values(
        ["price_wape_pct", "price_mae", "threshold"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    tau = float(val_sweep_sorted.iloc[0]["threshold"])

    test_base, test_sweep, test_bins, test_cond, test_acc = evaluate_split(
        dataset, manifest, "test", features, targets,
        cluster_model, classifier, specs, selected,
        global_rf, rep_bundle, thresholds, bins,
    )

    val_selected = val_sweep.loc[val_sweep["threshold"].eq(tau)].iloc[0]
    test_selected = test_sweep.loc[test_sweep["threshold"].eq(tau)].iloc[0]

    val_sweep.sort_values("threshold").to_csv(
        out_dir / "validation_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_sweep.sort_values("threshold").to_csv(
        out_dir / "test_threshold_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bin_df = pd.concat([val_bins, test_bins], ignore_index=True)
    bin_df.to_csv(
        out_dir / "confidence_bin_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat([val_cond, test_cond], ignore_index=True).to_csv(
        out_dir / "routing_condition_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_rows = []
    for split, bdf, sel in [
        ("val", val_base, val_selected),
        ("test", test_base, test_selected),
    ]:
        final_rows.extend(bdf.to_dict(orient="records"))
        r = sel.to_dict()
        r["route"] = "confidence_family_routing"
        r["selected_on_validation"] = True
        final_rows.append(r)

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(
        out_dir / "final_route_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selection = {
        "selected_threshold": tau,
        "selection_split": "validation",
        "selection_metric": "reconstructed curve price WAPE",
        "test_used_for_selection": False,
        "selected_classifier": classifier_name,
        "selected_expert_by_family": selected,
        "validation_selected_metrics": val_selected.to_dict(),
        "test_frozen_metrics": test_selected.to_dict(),
    }

    (out_dir / "selected_threshold.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def wape(df, route):
        return float(df.loc[df["route"].eq(route), "price_wape_pct"].iloc[0])

    vg = wape(val_base, "global_random_forest")
    vh = wape(val_base, "hard_family_routing")
    vo = wape(val_base, "oracle_family_routing")
    tg = wape(test_base, "global_random_forest")
    th = wape(test_base, "hard_family_routing")
    to = wape(test_base, "oracle_family_routing")
    vc = float(val_selected["price_wape_pct"])
    tc = float(test_selected["price_wape_pct"])

    summary = "\n".join([
        f"09f Confidence family routing - {args.year}",
        "=" * 80,
        "",
        f"K = {args.k}",
        f"Selected classifier = {classifier_name}",
        f"Validation classifier accuracy = {val_acc:.6f}",
        f"Test classifier accuracy = {test_acc:.6f}",
        "",
        f"Selected threshold from VALIDATION = {tau:.2f}",
        "",
        "VALIDATION:",
        f"Global RF WAPE             = {vg:.6f}%",
        f"Hard family routing WAPE   = {vh:.6f}%",
        f"Confidence routing WAPE    = {vc:.6f}%",
        f"Oracle family routing WAPE = {vo:.6f}%",
        f"Family-expert share        = {float(val_selected['family_expert_share']):.6f}",
        "",
        "TEST (threshold frozen from VALIDATION):",
        f"Global RF WAPE             = {tg:.6f}%",
        f"Hard family routing WAPE   = {th:.6f}%",
        f"Confidence routing WAPE    = {tc:.6f}%",
        f"Oracle family routing WAPE = {to:.6f}%",
        f"Family-expert share        = {float(test_selected['family_expert_share']):.6f}",
        f"TEST gain vs Global RF     = {tg - tc:.6f} percentage points",
        "",
        "Confidence-bin diagnostics:",
        bin_df.to_string(index=False),
        "",
        "Threshold is selected only on VALIDATION; TEST sweep is diagnostic only.",
    ])

    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print(f"Outputs: {out_dir}")

if __name__ == "__main__":
    main()
