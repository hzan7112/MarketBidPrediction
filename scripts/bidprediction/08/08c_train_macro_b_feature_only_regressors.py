#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08c: Train feature-only absolute-latent regressors on Macro-B.
Models: train_mean baseline, Ridge, Spline-GAM, Random Forest.
No previous bid curve or historical latent is used as model input.
"""
from __future__ import annotations
import argparse, gc, json, math, shutil
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

META=["sample_id","participant_id","timestamp_utc","timestamp_local","local_date","local_slot_seconds"]
SHAPE=[f"shape_v{i:02d}" for i in range(21)]
CURVE=[*SHAPE,"p_anchor","p_span","q_anchor_mw","q_span_mw"]
FORBIDDEN=("latent","curvevec","theta","template","shape_v","p_anchor","p_span","q_anchor","q_span")
def parts(m,s): return [x["file"] if isinstance(x,dict) else x for x in m["parts"][s]]
def nframe(d,c): return d[c].apply(pd.to_numeric,errors="coerce")
def leakage_check(features):
    bad=[f for f in features if any(t in f.lower() for t in FORBIDDEN)]
    if bad: raise RuntimeError("Historical-bid leakage in model_features: "+", ".join(bad[:30]))
def load_sample(ds,m,features,targets,max_rows,seed):
    fs=parts(m,"train"); per=max(1,math.ceil(max_rows/max(len(fs),1))); blocks=[]
    for i,r in enumerate(fs,1):
        d=pd.read_pickle(ds/r)
        if len(d)>per: d=d.sample(per,random_state=seed+7919*i)
        blocks.append(d[[*features,*targets]].copy()); del d; gc.collect()
    x=pd.concat(blocks,ignore_index=True)
    if len(x)>max_rows: x=x.sample(max_rows,random_state=seed).reset_index(drop=True)
    return x
def top_spearman(d,features,target,k):
    y=pd.to_numeric(d[target],errors="coerce"); scores=[]
    for c in features:
        x=pd.to_numeric(d[c],errors="coerce"); ok=x.notna()&y.notna()
        if ok.sum()<100 or x[ok].nunique()<=1: continue
        r=x[ok].corr(y[ok],method="spearman")
        if pd.notna(r): scores.append((c,abs(float(r)),float(r)))
    scores.sort(key=lambda z:z[1],reverse=True); return scores[:k]
def fit_ridge(train,features,targets,alpha):
    p=Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),("scaler",StandardScaler()),("ridge",Ridge(alpha=alpha))])
    p.fit(nframe(train,features),nframe(train,targets).to_numpy(float)); return p
def fit_gam(train,features,targets,topk,knots,degree,alpha):
    models={}; selection={}
    for i,t in enumerate(targets,1):
        print(f"[GAM {i}/{len(targets)}] {t}",flush=True); top=top_spearman(train,features,t,topk); f=[x[0] for x in top]
        if not f: f=features[:min(topk,len(features))]
        p=Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),("spline",SplineTransformer(n_knots=knots,degree=degree,knots="quantile",extrapolation="constant",include_bias=False,sparse_output=True)),("ridge",Ridge(alpha=alpha,solver="lsqr"))])
        p.fit(nframe(train,f),pd.to_numeric(train[t],errors="coerce").to_numpy(float)); models[t]=dict(features=f,model=p); selection[t]=dict(features=f,top_context=[dict(feature=c,abs_spearman=a,spearman=r) for c,a,r in top])
    return models,selection
def pred_gam(d,models,targets):
    out=np.empty((len(d),len(targets)),np.float32)
    for j,t in enumerate(targets): out[:,j]=models[t]["model"].predict(nframe(d,models[t]["features"])).astype(np.float32)
    return out
def latent_metrics(true,preds):
    rows=[]
    for name,p in preds.items():
        e=p-true; rows.append(dict(model=name,rows=len(true),latent_mae=float(np.mean(np.abs(e))),latent_rmse=float(np.sqrt(np.mean(e*e)))))
    return pd.DataFrame(rows)
def predict_split(ds,m,split,out,features,targets,mean_z,ridge,gam,rf):
    od=out/"prediction_parts"/split; od.mkdir(parents=True,exist_ok=True); written=[]; metrics=[]
    fs=parts(m,split)
    for i,r in enumerate(fs,1):
        print(f"[predict {split} {i}/{len(fs)}] {Path(r).name}",flush=True); d=pd.read_pickle(ds/r); X=nframe(d,features); true=nframe(d,targets).to_numpy(np.float32)
        preds={"train_mean":np.repeat(mean_z[None,:],len(d),axis=0).astype(np.float32),"ridge":ridge.predict(X).astype(np.float32),"spline_gam":pred_gam(d,gam,targets),"random_forest":rf.predict(X).astype(np.float32)}
        metrics.append(latent_metrics(true,preds)); keep=[c for c in [*META,*CURVE] if c in d.columns]+[c for c in d.columns if c.startswith("reference_curvevec_") or c.startswith("reference_curvevec") or c.startswith("reference_curvevec")]
        # 08b reference columns begin with 'reference_curvevec...' because source names begin with '_curvevec'.
        keep=[c for c in d.columns if c in set(keep) or c.startswith("reference_curvevec") or c.startswith("reference_curvevec") or c.startswith("reference_curvevec") or c.startswith("reference_curvevec")]
        # Generic reference prefix catch.
        keep=list(dict.fromkeys([*keep,*[c for c in d.columns if c.startswith("reference_") or c.startswith("reference")]]))
        odf=d[keep].copy()
        for j,t in enumerate(targets):
            odf[f"true_{t}"]=true[:,j]
            for name,p in preds.items(): odf[f"pred_{name}_{t}"]=p[:,j]
        name=f"{split}_feature_only_predictions_{i:04d}.pkl"; path=od/name; odf.to_pickle(path); written.append(dict(file=str(path.relative_to(out)),rows=len(odf))); del d,X,true,odf,preds; gc.collect()
    mm=pd.concat(metrics,ignore_index=True); agg=[]
    for model,g in mm.groupby("model",sort=False):
        w=g.rows.to_numpy(float); agg.append(dict(split=split,model=model,rows=int(w.sum()),latent_mae=float(np.average(g.latent_mae,weights=w)),latent_rmse=float(np.average(g.latent_rmse,weights=w))))
    return written,pd.DataFrame(agg)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int,default=2025); ap.add_argument("--root",default="data/processed/bidprediction"); ap.add_argument("--dataset-dir",default="macro_b_feature_only_dataset"); ap.add_argument("--max-train-rows",type=int,default=300000); ap.add_argument("--gam-train-rows",type=int,default=150000); ap.add_argument("--ridge-alpha",type=float,default=10.0); ap.add_argument("--gam-alpha",type=float,default=1.0); ap.add_argument("--gam-top-context",type=int,default=24); ap.add_argument("--gam-knots",type=int,default=5); ap.add_argument("--gam-degree",type=int,default=2); ap.add_argument("--rf-trees",type=int,default=128); ap.add_argument("--rf-max-depth",type=int,default=18); ap.add_argument("--rf-min-leaf",type=int,default=8); ap.add_argument("--rf-max-features",type=float,default=.5); ap.add_argument("--n-jobs",type=int,default=8); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--overwrite",action="store_true"); a=ap.parse_args()
    base=Path(a.root)/str(a.year); ds=base/a.dataset_dir; m=json.loads((ds/"manifest.json").read_text(encoding="utf-8")); features=list(m["model_features"]); targets=list(m["latent_columns"]); leakage_check(features); out=base/"macro_b_feature_only_regression_models"
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    train=load_sample(ds,m,features,targets,a.max_train_rows,a.seed); mean_z=nframe(train,targets).mean(axis=0).to_numpy(float); ridge=fit_ridge(train,features,targets,a.ridge_alpha)
    gt=train.sample(a.gam_train_rows,random_state=a.seed+17).reset_index(drop=True) if len(train)>a.gam_train_rows else train
    gam,selection=fit_gam(gt,features,targets,a.gam_top_context,a.gam_knots,a.gam_degree,a.gam_alpha)
    rf=Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),("rf",RandomForestRegressor(n_estimators=a.rf_trees,max_depth=a.rf_max_depth,min_samples_leaf=a.rf_min_leaf,max_features=a.rf_max_features,n_jobs=a.n_jobs,random_state=a.seed))]); rf.fit(nframe(train,features),nframe(train,targets).to_numpy(np.float32))
    joblib.dump(dict(mean_z=mean_z,targets=targets),out/"train_mean_baseline.joblib",compress=3); joblib.dump(dict(model=ridge,features=features,targets=targets),out/"ridge_model.joblib",compress=3); joblib.dump(dict(models=gam,targets=targets,selection=selection),out/"spline_gam_models.joblib",compress=3); joblib.dump(dict(model=rf,features=features,targets=targets),out/"random_forest_model.joblib",compress=3)
    (out/"spline_gam_feature_selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    pm=dict(year=a.year,source_dataset=str(ds),latent_columns=targets,model_features=features,models=["train_mean","ridge","spline_gam","random_forest"],parts={}); tables=[]
    for s in ["val","test"]:
        wr,met=predict_split(ds,m,s,out,features,targets,mean_z,ridge,gam,rf); pm["parts"][s]=wr; met.to_csv(out/f"{s}_latent_metrics.csv",index=False,encoding="utf-8-sig"); tables.append(met)
    (out/"manifest.json").write_text(json.dumps(pm,ensure_ascii=False,indent=2),encoding="utf-8"); allm=pd.concat(tables,ignore_index=True)
    summary="\n".join([f"08c Macro-B feature-only regressors - {a.year}","="*80,"",f"Model features = {len(features)}",f"Absolute latent targets = {len(targets)}",f"Shared TRAIN sample = {len(train):,}",f"Spline-GAM TRAIN sample = {len(gt):,}","Historical-bid leakage check = PASS","","Latent diagnostics:",allm.to_string(index=False),"","Final selection is done in 08d by validation reconstructed-curve WAPE."])
    (out/"summary.txt").write_text(summary,encoding="utf-8"); print(summary); print(f"Outputs: {out}")
if __name__=="__main__": main()
