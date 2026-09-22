#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03c_train_template_transition_destination.py

Conditional destination prediction on TRUE switch samples only:

    P(T_t=j | T_{t-1}=i, switch=1, Z_base, Z_tr, M, U, H)

Methods:
1) OriginPrior
2) GlobalOriginMasked
3) ConditionalOriginModel

Important:
- origin = hist_lag1_template_id is an explicit condition.
- origin is removed from H_context to avoid double counting.
- all transition masks/priors are built from TRAIN only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("Install LightGBM first: pip install lightgbm") from e


LT = [
    "lt_bid_level", "lt_adjustment_magnitude", "lt_strategy_persistence",
    "lt_quantity_hhi", "lt_effective_segment_count", "lt_flat_curve_rate",
    "lt_tail_uplift_ratio", "lt_curve_bend_ratio", "lt_shape_variability",
]
ST = [
    "st_bid_level_z", "st_adjustment_bias_z", "st_adjustment_magnitude_z",
    "st_quantity_hhi_z", "st_effective_segment_count_z",
    "st_flat_curve_rate_z", "st_tail_uplift_ratio_z",
    "st_curve_bend_ratio_z", "st_shape_shift",
]
BR = [
    "break_bid_level", "break_adjustment_bias", "break_adjustment_magnitude",
    "break_quantity_hhi", "break_effective_segment_count",
    "break_flat_curve_rate", "break_tail_uplift_ratio",
    "break_curve_bend_ratio",
]

TARGET = "y_template_id"
ORIGIN = "hist_lag1_template_id"

TEMPLATES = [f"T{i:02d}" for i in range(12)] + ["FLAT"]
T2I = {t: i for i, t in enumerate(TEMPLATES)}
I2T = {i: t for t, i in T2I.items()}
K = len(TEMPLATES)

MODE2I = {"flat": 0, "block": 1, "sloped": 2}

CAT = {
    "hist30_dominant_template_id",
    "hist_lag1_curve_mode",
}

EXCLUDE = {
    "rolling_lt_ready_flag", "st_ready_flag", "profile_ready_flag",
    "market_ready_flag", "unit_state_ready_flag", "prediction_ready_flag",
    "market_nonmissing_count", "rolling_lt_nonmissing_count",
    "hist_prev_available_flag", "tr_ready_flag",
}

ALGORITHMS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "LightGBM",
]


def uq(xs):
    out, seen = [], set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load_schema(path):
    df = pd.read_csv(path)
    need = {"column", "role", "feature_group"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{path}: missing {sorted(miss)}")
    return df


def load_splits(path):
    s = pd.read_csv(path).set_index("split")
    return (
        pd.Timestamp(s.loc["train", "last_date"]),
        pd.Timestamp(s.loc["val", "first_date"]),
        pd.Timestamp(s.loc["val", "last_date"]),
        pd.Timestamp(s.loc["test", "first_date"]),
    )


def feature_sets(pred_schema, tr_schema):
    p = pred_schema[pred_schema["role"].astype(str).eq("feature")].copy()
    gmap = dict(zip(p["column"].astype(str), p["feature_group"].astype(str)))
    avail = set(gmap) - EXCLUDE

    z = LT + ST + BR
    missing = [c for c in z if c not in avail]
    if missing:
        raise KeyError(f"Missing Z_base: {missing}")

    m = [c for c, g in gmap.items() if g == "market_environment" and c in avail]
    u = [c for c, g in gmap.items() if g == "unit_state_proxy" and c in avail]
    h = [
        c for c, g in gmap.items()
        if g == "participant_history" and c in avail and c != ORIGIN
    ]
    tr = tr_schema[tr_schema["role"].astype(str).eq("feature")]
    ztr = tr["column"].astype(str).tolist()

    z, ztr, m, u, h = map(uq, [z, ztr, m, u, h])

    sets = {
        "Z_base": z,
        "Z_base_tr": uq(z + ztr),
        "Z_base_M_U": uq(z + m + u),
        "Z_base_tr_M_U": uq(z + ztr + m + u),
        "Z_base_H": uq(z + h),
        "Z_base_tr_H": uq(z + ztr + h),
        "Z_base_M_U_H": uq(z + m + u + h),
        "Z_base_tr_M_U_H": uq(z + ztr + m + u + h),
    }
    comps = {"Z_base": z, "Z_tr": ztr, "M": m, "U": u, "H_context": h}
    for c in ztr:
        gmap[c] = "transition_strategy_profile"
    return sets, comps, gmap


def load_transition(path, features):
    cols = ["participant_id", "local_date"] + features
    tr = pd.read_csv(path, usecols=cols, low_memory=False)
    tr["participant_id"] = tr["participant_id"].astype("string").str.strip()
    tr["local_date"] = pd.to_datetime(tr["local_date"], errors="coerce").dt.normalize()
    if tr[["participant_id", "local_date"]].duplicated().any():
        raise ValueError("Duplicate participant_id/local_date in transition profile")
    for c in features:
        tr[c] = num(tr[c]).astype("float32")
    return tr


def merge_tr(df, tr):
    x = df.copy()
    x["participant_id"] = x["participant_id"].astype("string").str.strip()
    x["local_date"] = pd.to_datetime(x["local_date"], errors="coerce").dt.normalize()
    return x.merge(
        tr,
        on=["participant_id", "local_date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )


def enc_template(s):
    return s.astype("string").str.strip().map(T2I)


def enc_target(s):
    y = enc_template(s)
    if y.isna().any():
        raise ValueError(f"Unknown template label: {s[y.isna()].head().tolist()}")
    return y.to_numpy(np.int16)


def split_mask(s, name, train_end, val_start, val_end, test_start):
    d = pd.to_datetime(s, errors="coerce").dt.normalize()
    if name == "train":
        return d <= train_end
    if name == "val":
        return (d >= val_start) & (d <= val_end)
    if name == "test":
        return d >= test_start
    raise ValueError(name)


def collect_switch_rows(
    files, all_features, tr_features, tr_table, split_name,
    train_end, val_start, val_end, test_start, chunksize
):
    trset = set(tr_features)
    ds_features = [c for c in all_features if c not in trset]
    usecols = uq([
        "participant_id", "local_date", "prediction_ready_flag",
        TARGET, ORIGIN
    ] + ds_features)

    blocks = []
    n_ready = n_lag = 0

    for fi, f in enumerate(files, 1):
        header = pd.read_csv(f, nrows=0).columns.tolist()
        miss = [c for c in uq([TARGET, ORIGIN] + ds_features) if c not in header]
        if miss:
            raise KeyError(f"{f.name} missing {miss}")

        cols = [c for c in usecols if c in header]
        print(f"[{split_name} {fi}/{len(files)}] {f.name}", flush=True)

        for ch in pd.read_csv(f, usecols=cols, chunksize=chunksize, low_memory=False):
            ready = num(ch["prediction_ready_flag"]).fillna(0).eq(1)
            sm = split_mask(ch["local_date"], split_name, train_end, val_start, val_end, test_start)
            base = ready & sm
            n_ready += int(base.sum())

            oi = enc_template(ch[ORIGIN])
            yi = enc_template(ch[TARGET])
            ok = base & oi.notna() & yi.notna()
            n_lag += int(ok.sum())

            sw = ok & oi.ne(yi)
            sub = ch.loc[
                sw,
                uq(["participant_id", "local_date", TARGET, ORIGIN] + ds_features),
            ].copy()
            if sub.empty:
                continue

            sub = merge_tr(sub, tr_table)
            blocks.append(sub[uq([TARGET, ORIGIN] + all_features)])

    if not blocks:
        raise ValueError(f"No switch rows for {split_name}")

    out = pd.concat(blocks, ignore_index=True)
    print(
        f"{split_name}: ready={n_ready:,}, lag1={n_lag:,}, "
        f"switch={len(out):,} ({len(out)/max(n_lag,1):.2%})",
        flush=True,
    )
    return out


def prepare_X(df, features):
    out = {}
    for c in features:
        if c == "hist30_dominant_template_id":
            out[c] = df[c].astype("string").str.strip().map(T2I).astype("float32")
        elif c == "hist_lag1_curve_mode":
            out[c] = df[c].astype("string").str.strip().str.lower().map(MODE2I).astype("float32")
        else:
            out[c] = num(df[c]).astype("float32")
    return pd.DataFrame(out, index=df.index)


def transition_matrix(train):
    o = enc_target(train[ORIGIN])
    y = enc_target(train[TARGET])

    counts = np.zeros((K, K), dtype=np.int64)
    np.add.at(counts, (o, y), 1)
    np.fill_diagonal(counts, 0)

    rs = counts.sum(axis=1, keepdims=True)
    prob = np.divide(
        counts, rs,
        out=np.zeros_like(counts, dtype=float),
        where=rs > 0,
    )
    allowed = counts > 0
    prior = np.full(K, -1, dtype=np.int16)

    for i in range(K):
        if counts[i].sum() > 0:
            prior[i] = int(np.argmax(counts[i]))

    return counts, prob, allowed, prior


def sqrt_weights(y_local, nclass):
    cnt = np.bincount(y_local, minlength=nclass).astype(float)
    valid = cnt > 0
    cls = np.ones(nclass, dtype=float)
    raw = np.sqrt(cnt[valid].sum() / cnt[valid])
    cls[valid] = raw / (np.sum(cnt[valid] * raw) / cnt[valid].sum())
    return cls[y_local]


def fit_model(name, X, y, feature_names, args):
    classes = np.sort(np.unique(y)).astype(np.int16)

    if len(classes) == 1:
        return {"constant": int(classes[0]), "classes": classes}

    if name == "LogisticRegression":
        scaler = StandardScaler()
        xs = scaler.fit_transform(X)
        model = LogisticRegression(
            solver="lbfgs",
            C=args.logit_c,
            max_iter=args.logit_max_iter,
            class_weight="balanced",
            random_state=args.seed,
        )
        model.fit(xs, y)
        return {"model": model, "scaler": scaler, "classes": model.classes_.astype(np.int16)}

    if name == "DecisionTree":
        model = DecisionTreeClassifier(
            max_depth=args.dt_max_depth,
            min_samples_leaf=args.dt_min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed,
        )
        model.fit(X, y)
        return {"model": model, "classes": model.classes_.astype(np.int16)}

    if name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        )
        model.fit(X, y)
        return {"model": model, "classes": model.classes_.astype(np.int16)}

    if name == "LightGBM":
        g2l = {int(v): j for j, v in enumerate(classes)}
        yl = np.asarray([g2l[int(v)] for v in y], dtype=np.int16)
        w = sqrt_weights(yl, len(classes))
        ds = lgb.Dataset(
            X,
            label=yl,
            weight=w,
            feature_name=feature_names,
            free_raw_data=True,
        )
        params = {
            "objective": "multiclass",
            "num_class": len(classes),
            "metric": "multi_logloss",
            "learning_rate": args.lgb_learning_rate,
            "num_leaves": args.lgb_num_leaves,
            "min_data_in_leaf": args.lgb_min_data_in_leaf,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "max_bin": 127,
            "verbosity": -1,
            "seed": args.seed,
            "num_threads": 0,
        }
        model = lgb.train(params, ds, num_boost_round=args.lgb_rounds)
        return {"model": model, "classes": classes}

    raise ValueError(name)


def predict_proba(name, payload, X):
    full = np.zeros((len(X), K), dtype=np.float64)

    if "constant" in payload:
        full[:, payload["constant"]] = 1.0
        return full

    model = payload["model"]
    classes = np.asarray(payload["classes"], dtype=np.int16)

    if name == "LogisticRegression":
        p = model.predict_proba(payload["scaler"].transform(X))
    elif name in {"DecisionTree", "RandomForest"}:
        p = model.predict_proba(X)
    elif name == "LightGBM":
        p = np.asarray(model.predict(X), dtype=np.float64)
        if p.ndim == 1:
            p = p[:, None]
    else:
        raise ValueError(name)

    full[:, classes] = p
    return full


def mask_by_origin(p, origin, allowed):
    p = np.asarray(p, dtype=np.float64).copy()

    for i in range(K):
        rows = np.where(origin == i)[0]
        if len(rows) == 0:
            continue
        mask = allowed[i].copy()
        if not mask.any():
            mask[:] = True
            mask[i] = False
        p[np.ix_(rows, ~mask)] = 0.0

    denom = p.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= 0
    if bad.any():
        for r in np.where(bad)[0]:
            p[r, :] = 1.0
            p[r, origin[r]] = 0.0
        denom = p.sum(axis=1, keepdims=True)

    p /= denom
    pred = np.argmax(p, axis=1).astype(np.int16)
    return pred, p


def metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=np.arange(K))
    total = cm.sum()
    diag = np.diag(cm)
    sup = cm.sum(axis=1)
    prd = cm.sum(axis=0)

    rec = np.divide(diag, sup, out=np.zeros_like(diag, dtype=float), where=sup > 0)
    pre = np.divide(diag, prd, out=np.zeros_like(diag, dtype=float), where=prd > 0)
    f1 = np.divide(
        2 * pre * rec, pre + rec,
        out=np.zeros_like(rec, dtype=float),
        where=(pre + rec) > 0,
    )
    valid = sup > 0

    return {
        "accuracy": float(diag.sum() / total) if total else np.nan,
        "balanced_accuracy": float(rec[valid].mean()) if valid.any() else np.nan,
        "macro_f1": float(f1[valid].mean()) if valid.any() else np.nan,
        "weighted_f1": float(np.sum(f1 * sup) / total) if total else np.nan,
        "cm": cm,
    }


def prob_metrics(y, p):
    eps = 1e-15
    pt = p[np.arange(len(y)), y]
    ll = float(-np.log(np.clip(pt, eps, 1.0)).mean())
    top2 = np.argpartition(p, kth=-2, axis=1)[:, -2:]
    t2 = float(np.any(top2 == y[:, None], axis=1).mean())
    return t2, ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--root", default="data/processed/bidprediction")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--feature-set",
        choices=[
            "Z_base", "Z_base_tr",
            "Z_base_M_U", "Z_base_tr_M_U",
            "Z_base_H", "Z_base_tr_H",
            "Z_base_M_U_H", "Z_base_tr_M_U_H",
            "all",
        ],
        default="all",
    )

    ap.add_argument("--min-origin-train-rows", type=int, default=300)
    ap.add_argument("--min-origin-destination-classes", type=int, default=2)

    ap.add_argument("--logit-max-iter", type=int, default=300)
    ap.add_argument("--logit-c", type=float, default=1.0)

    ap.add_argument("--dt-max-depth", type=int, default=16)
    ap.add_argument("--dt-min-samples-leaf", type=int, default=30)

    ap.add_argument("--rf-trees", type=int, default=160)
    ap.add_argument("--rf-max-depth", type=int, default=18)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=20)

    ap.add_argument("--lgb-rounds", type=int, default=300)
    ap.add_argument("--lgb-num-leaves", type=int, default=31)
    ap.add_argument("--lgb-learning-rate", type=float, default=0.06)
    ap.add_argument("--lgb-min-data-in-leaf", type=int, default=40)

    args = ap.parse_args()

    base = Path(args.root) / str(args.year)
    parts = sorted((base / "dataset_parts").glob("prediction_dataset_*.csv"))
    if not parts:
        raise FileNotFoundError(base / "dataset_parts")

    pred_schema = load_schema(base / f"prediction_feature_schema_{args.year}.csv")
    tr_schema = load_schema(base / f"transition_strategy_feature_schema_{args.year}.csv")
    sets, comps, gmap = feature_sets(pred_schema, tr_schema)

    selected = list(sets) if args.feature_set == "all" else [args.feature_set]
    tr_features = comps["Z_tr"]
    tr_table = load_transition(
        base / f"transition_strategy_profile_{args.year}.csv",
        tr_features,
    )

    train_end, val_start, val_end, test_start = load_splits(
        base / "validation" / "temporal_split_summary.csv"
    )

    out = base / "template_destination_conditional"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for comp, cols in comps.items():
        for j, c in enumerate(cols):
            rows.append({
                "component": comp,
                "order": j,
                "feature": c,
                "feature_group": gmap.get(c, ""),
            })
    rows.append({
        "component": "OriginCondition",
        "order": 0,
        "feature": ORIGIN,
        "feature_group": "explicit_transition_origin",
    })
    pd.DataFrame(rows).to_csv(
        out / "input_components_Zbase_Ztr_M_U_H.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fsrows = []
    for s, cols in sets.items():
        for j, c in enumerate(cols):
            fsrows.append({
                "feature_set": s,
                "order": j,
                "feature": c,
                "feature_group": gmap.get(c, ""),
            })
    pd.DataFrame(fsrows).to_csv(
        out / "feature_sets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Conditional template destination prediction")
    print("=" * 80)
    print(f"Z_base={len(comps['Z_base'])}, Z_tr={len(comps['Z_tr'])}, "
          f"M={len(comps['M'])}, U={len(comps['U'])}, "
          f"H_context={len(comps['H_context'])}")
    print(f"Origin condition: {ORIGIN}")
    print(f"Feature sets: {', '.join(selected)}")
    print()

    all_features = uq([c for s in selected for c in sets[s]])

    train = collect_switch_rows(
        parts, all_features, tr_features, tr_table, "train",
        train_end, val_start, val_end, test_start, args.chunksize,
    )
    val = collect_switch_rows(
        parts, all_features, tr_features, tr_table, "val",
        train_end, val_start, val_end, test_start, args.chunksize,
    )
    test = collect_switch_rows(
        parts, all_features, tr_features, tr_table, "test",
        train_end, val_start, val_end, test_start, args.chunksize,
    )

    ytr, otr = enc_target(train[TARGET]), enc_target(train[ORIGIN])
    yv, ov = enc_target(val[TARGET]), enc_target(val[ORIGIN])
    yt, ot = enc_target(test[TARGET]), enc_target(test[ORIGIN])

    counts, probs, allowed, prior = transition_matrix(train)

    cdf = pd.DataFrame(counts, index=TEMPLATES, columns=TEMPLATES)
    cdf.index.name = "origin"
    cdf.to_csv(out / "transition_matrix_train_counts.csv", encoding="utf-8-sig")

    pdf = pd.DataFrame(probs, index=TEMPLATES, columns=TEMPLATES)
    pdf.index.name = "origin"
    pdf.to_csv(out / "transition_matrix_train_probabilities.csv", encoding="utf-8-sig")

    origin_rows = []
    for i in range(K):
        cand = np.flatnonzero(counts[i] > 0)
        def unseen_rate(oarr, yarr):
            m = oarr == i
            if not m.any():
                return np.nan
            return float((~np.isin(yarr[m], cand)).mean())

        n = int(counts[i].sum())
        dom = int(np.argmax(counts[i])) if n else -1
        origin_rows.append({
            "origin_template": I2T[i],
            "train_switch_rows": n,
            "train_destination_classes": int(len(cand)),
            "train_dominant_destination": I2T[dom] if dom >= 0 else "",
            "train_dominant_destination_share": (
                float(counts[i, dom] / n) if n else np.nan
            ),
            "val_switch_rows": int((ov == i).sum()),
            "val_unseen_destination_rate": unseen_rate(ov, yv),
            "test_switch_rows": int((ot == i).sum()),
            "test_unseen_destination_rate": unseen_rate(ot, yt),
        })
    origin_summary = pd.DataFrame(origin_rows)
    origin_summary.to_csv(
        out / "origin_transition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def prior_pred(oarr):
        pred = np.full(len(oarr), -1, dtype=np.int16)
        for i in range(K):
            r = np.where(oarr == i)[0]
            if len(r) == 0:
                continue
            if prior[i] >= 0:
                pred[r] = prior[i]
            else:
                pred[r] = 0 if i != 0 else 1
        return pred

    metric_rows = []
    per_origin_rows = []

    for split_name, yy, oo in [("val", yv, ov), ("test", yt, ot)]:
        pred = prior_pred(oo)
        mm = metrics(yy, pred)
        coverage = float(np.mean([allowed[oo[r], yy[r]] for r in range(len(yy))]))
        metric_rows.append({
            "feature_set": "TRAIN_TRANSITION_MATRIX_ONLY",
            "algorithm": "None",
            "method": "OriginPrior",
            "split": split_name,
            "rows": len(yy),
            "accuracy": mm["accuracy"],
            "balanced_accuracy": mm["balanced_accuracy"],
            "macro_f1": mm["macro_f1"],
            "weighted_f1": mm["weighted_f1"],
            "top2_accuracy": np.nan,
            "log_loss": np.nan,
            "fallback_share": 0.0,
            "candidate_coverage": coverage,
        })

    for si, set_name in enumerate(selected, 1):
        feats = sets[set_name]
        print()
        print("=" * 80)
        print(f"[{si}/{len(selected)}] {set_name}, features={len(feats)}")
        print("=" * 80)

        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        Xtr = imp.fit_transform(prepare_X(train, feats)).astype(np.float32)
        Xv = imp.transform(prepare_X(val, feats)).astype(np.float32)
        Xt = imp.transform(prepare_X(test, feats)).astype(np.float32)

        # Global model receives origin explicitly.
        Xtrg = np.column_stack([Xtr, otr.astype(np.float32)])
        Xvg = np.column_stack([Xv, ov.astype(np.float32)])
        Xtg = np.column_stack([Xt, ot.astype(np.float32)])
        global_feats = feats + ["__origin_template_code__"]

        for algo in ALGORITHMS:
            print(f"[fit] {set_name}/{algo}", flush=True)

            gp = fit_model(algo, Xtrg, ytr, global_feats, args)
            gvp = predict_proba(algo, gp, Xvg)
            gtp = predict_proba(algo, gp, Xtg)

            gv_pred, gvp = mask_by_origin(gvp, ov, allowed)
            gt_pred, gtp = mask_by_origin(gtp, ot, allowed)

            for split_name, yy, oo, pred, pp in [
                ("val", yv, ov, gv_pred, gvp),
                ("test", yt, ot, gt_pred, gtp),
            ]:
                mm = metrics(yy, pred)
                top2, ll = prob_metrics(yy, pp)
                coverage = float(np.mean([allowed[oo[r], yy[r]] for r in range(len(yy))]))
                metric_rows.append({
                    "feature_set": set_name,
                    "algorithm": algo,
                    "method": "GlobalOriginMasked",
                    "split": split_name,
                    "rows": len(yy),
                    "accuracy": mm["accuracy"],
                    "balanced_accuracy": mm["balanced_accuracy"],
                    "macro_f1": mm["macro_f1"],
                    "weighted_f1": mm["weighted_f1"],
                    "top2_accuracy": top2,
                    "log_loss": ll,
                    "fallback_share": 0.0,
                    "candidate_coverage": coverage,
                })

            # Per-origin classifiers.
            origin_models = {}
            for i in range(K):
                m = otr == i
                n = int(m.sum())
                classes = np.unique(ytr[m])
                if (
                    n < args.min_origin_train_rows
                    or len(classes) < args.min_origin_destination_classes
                ):
                    origin_models[i] = None
                    continue

                print(
                    f"  origin={I2T[i]} train={n:,} "
                    f"dest_classes={len(classes)}",
                    flush=True,
                )
                origin_models[i] = fit_model(
                    algo, Xtr[m], ytr[m], feats, args
                )

            def cond_predict(X, oo, global_pred, global_p):
                pred = global_pred.copy()
                pp = global_p.copy()
                fallback = np.ones(len(oo), dtype=bool)

                for i in range(K):
                    rows_i = np.where(oo == i)[0]
                    if len(rows_i) == 0:
                        continue
                    payload = origin_models.get(i)
                    if payload is None:
                        continue

                    pi = predict_proba(algo, payload, X[rows_i])
                    pr_i, pi = mask_by_origin(
                        pi,
                        np.full(len(rows_i), i, dtype=np.int16),
                        allowed,
                    )
                    pred[rows_i] = pr_i
                    pp[rows_i] = pi
                    fallback[rows_i] = False

                return pred, pp, fallback

            cv_pred, cvp, cv_fb = cond_predict(Xv, ov, gv_pred, gvp)
            ct_pred, ctp, ct_fb = cond_predict(Xt, ot, gt_pred, gtp)

            for split_name, yy, oo, pred, pp, fb in [
                ("val", yv, ov, cv_pred, cvp, cv_fb),
                ("test", yt, ot, ct_pred, ctp, ct_fb),
            ]:
                mm = metrics(yy, pred)
                top2, ll = prob_metrics(yy, pp)
                coverage = float(np.mean([allowed[oo[r], yy[r]] for r in range(len(yy))]))
                metric_rows.append({
                    "feature_set": set_name,
                    "algorithm": algo,
                    "method": "ConditionalOriginModel",
                    "split": split_name,
                    "rows": len(yy),
                    "accuracy": mm["accuracy"],
                    "balanced_accuracy": mm["balanced_accuracy"],
                    "macro_f1": mm["macro_f1"],
                    "weighted_f1": mm["weighted_f1"],
                    "top2_accuracy": top2,
                    "log_loss": ll,
                    "fallback_share": float(fb.mean()),
                    "candidate_coverage": coverage,
                })

                for i in range(K):
                    m = oo == i
                    if not m.any():
                        continue
                    om = metrics(yy[m], pred[m])
                    per_origin_rows.append({
                        "feature_set": set_name,
                        "algorithm": algo,
                        "split": split_name,
                        "origin_template": I2T[i],
                        "rows": int(m.sum()),
                        "accuracy": om["accuracy"],
                        "balanced_accuracy": om["balanced_accuracy"],
                        "macro_f1": om["macro_f1"],
                        "fallback": origin_models.get(i) is None,
                        "train_origin_rows": int((otr == i).sum()),
                        "train_destination_classes": int((counts[i] > 0).sum()),
                    })

    mdf = pd.DataFrame(metric_rows)
    podf = pd.DataFrame(per_origin_rows)

    mdf.to_csv(
        out / "destination_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    podf.to_csv(
        out / "per_origin_destination_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_map = {
        "Z_base": "Z_base_tr",
        "Z_base_M_U": "Z_base_tr_M_U",
        "Z_base_H": "Z_base_tr_H",
        "Z_base_M_U_H": "Z_base_tr_M_U_H",
    }

    pairs = []
    for split_name in ["val", "test"]:
        for method in ["GlobalOriginMasked", "ConditionalOriginModel"]:
            for algo in ALGORITHMS:
                sub = mdf[
                    (mdf["split"] == split_name)
                    & (mdf["method"] == method)
                    & (mdf["algorithm"] == algo)
                ].set_index("feature_set")

                for a, b in pair_map.items():
                    if a not in sub.index or b not in sub.index:
                        continue
                    ra, rb = sub.loc[a], sub.loc[b]
                    pairs.append({
                        "split": split_name,
                        "method": method,
                        "algorithm": algo,
                        "base_feature_set": a,
                        "transition_feature_set": b,
                        "delta_accuracy_from_Ztr": rb["accuracy"] - ra["accuracy"],
                        "delta_macro_f1_from_Ztr": rb["macro_f1"] - ra["macro_f1"],
                        "delta_top2_from_Ztr": rb["top2_accuracy"] - ra["top2_accuracy"],
                        "delta_log_loss_from_Ztr": rb["log_loss"] - ra["log_loss"],
                    })

    pairdf = pd.DataFrame(pairs)
    pairdf.to_csv(
        out / "transition_profile_incremental_value.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        f"Conditional template destination prediction - {args.year}",
        "=" * 80,
        "",
        f"Train <= {train_end.date()}",
        f"Validation = {val_start.date()} .. {val_end.date()}",
        f"Test >= {test_start.date()}",
        "",
        f"Train TRUE switch rows = {len(train):,}",
        f"Val TRUE switch rows   = {len(val):,}",
        f"Test TRUE switch rows  = {len(test):,}",
        "",
        f"Z_base={len(comps['Z_base'])}, Z_tr={len(comps['Z_tr'])}, "
        f"M={len(comps['M'])}, U={len(comps['U'])}, "
        f"H_context={len(comps['H_context'])}",
        f"Origin condition = {ORIGIN}",
        "",
        "Origin TRAIN transition structure:",
    ]

    for r in origin_summary.itertuples():
        lines.append(
            f"  {r.origin_template}: train={r.train_switch_rows:,}, "
            f"dest_classes={r.train_destination_classes}, "
            f"dominant={r.train_dominant_destination or '-'}, "
            f"dominant_share={r.train_dominant_destination_share:.4f}"
            if pd.notna(r.train_dominant_destination_share)
            else
            f"  {r.origin_template}: train=0"
        )

    for split_name in ["val", "test"]:
        lines += ["", f"[{split_name}]"]

        pr = mdf[
            (mdf["split"] == split_name)
            & (mdf["method"] == "OriginPrior")
        ]
        if not pr.empty:
            r = pr.iloc[0]
            lines.append(
                f"  OriginPrior: Acc={r['accuracy']:.4f}, "
                f"BalAcc={r['balanced_accuracy']:.4f}, "
                f"MacroF1={r['macro_f1']:.4f}, "
                f"candidate_coverage={r['candidate_coverage']:.4f}"
            )

        for s in selected:
            lines.append(f"  {s}:")
            for algo in ALGORITHMS:
                for method in ["GlobalOriginMasked", "ConditionalOriginModel"]:
                    q = mdf[
                        (mdf["split"] == split_name)
                        & (mdf["feature_set"] == s)
                        & (mdf["algorithm"] == algo)
                        & (mdf["method"] == method)
                    ]
                    if q.empty:
                        continue
                    r = q.iloc[0]
                    lines.append(
                        f"    {algo}/{method}: "
                        f"Acc={r['accuracy']:.4f}, "
                        f"BalAcc={r['balanced_accuracy']:.4f}, "
                        f"MacroF1={r['macro_f1']:.4f}, "
                        f"Top2={r['top2_accuracy']:.4f}, "
                        f"LogLoss={r['log_loss']:.4f}, "
                        f"fallback={r['fallback_share']:.2%}"
                    )

        lines.append("")
        lines.append("  Increment from Z_tr (ConditionalOriginModel):")
        q = pairdf[
            (pairdf["split"] == split_name)
            & (pairdf["method"] == "ConditionalOriginModel")
        ]
        for r in q.itertuples():
            lines.append(
                f"    {r.algorithm}/{r.base_feature_set}"
                f" -> {r.transition_feature_set}: "
                f"dAcc={r.delta_accuracy_from_Ztr:+.4f}, "
                f"dMacroF1={r.delta_macro_f1_from_Ztr:+.4f}, "
                f"dTop2={r.delta_top2_from_Ztr:+.4f}, "
                f"dLogLoss={r.delta_log_loss_from_Ztr:+.4f}"
            )

    lines += [
        "",
        "Interpretation:",
        "  OriginPrior = TRAIN transition-matrix prior only.",
        "  GlobalOriginMasked = shared classifier + TRAIN origin candidate mask.",
        "  ConditionalOriginModel = one classifier per origin when data is sufficient.",
        "  ConditionalOriginModel should beat both baselines on TEST to justify per-origin modeling.",
    ]

    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary, encoding="utf-8")

    cfg = {
        "year": args.year,
        "feature_sets": selected,
        "algorithms": ALGORITHMS,
        "origin_condition": ORIGIN,
        "origin_removed_from_H_context": True,
        "min_origin_train_rows": args.min_origin_train_rows,
        "min_origin_destination_classes": args.min_origin_destination_classes,
        "candidate_mask_source": "TRAIN transition matrix only",
        "seed": args.seed,
    }
    (out / "training_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
