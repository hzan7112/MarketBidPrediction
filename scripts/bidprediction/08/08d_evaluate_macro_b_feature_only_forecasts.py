#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08d: Reconstruct and evaluate Macro-B feature-only absolute-curve forecasts.

Primary same-task comparison:
    train_mean vs Ridge vs Spline-GAM vs Random Forest
Validation WAPE selects the final feature-only model.

Cross-task reference only:
    previous raw curve (Persistence)
This is NOT eligible for feature-only model selection because it has extra
information unavailable to the feature-only task.
"""
from __future__ import annotations
import argparse, gc, json, shutil
from pathlib import Path
import joblib, numpy as np, pandas as pd

GRID=np.linspace(0,1,21); SHAPE=[f"shape_v{i:02d}" for i in range(21)]
MODELS=["train_mean","ridge","spline_gam","random_forest"]
REFP=[f"reference_curvevec_p{i:02d}_lag1" for i in range(21)]

def num(s): return pd.to_numeric(s,errors="coerce")
def parts(m,s): return [x["file"] if isinstance(x,dict) else x for x in m["parts"][s]]
def true_curve(d):
    sh=d[SHAPE].apply(pd.to_numeric,errors="coerce").to_numpy(float); pa=num(d.p_anchor).to_numpy(float); ps=num(d.p_span).to_numpy(float); z=np.abs(ps)<=1e-12
    if z.any(): sh[z]=np.nan_to_num(sh[z],nan=0.0,posinf=0.0,neginf=0.0)
    p=pa[:,None]+ps[:,None]*sh; qa=num(d.q_anchor_mw).to_numpy(float); qs=num(d.q_span_mw).to_numpy(float); q=qa[:,None]+qs[:,None]*GRID[None,:]
    return q,p,qa,qs
def unpack(v): return v[:,:21],v[:,21],np.exp(np.clip(v[:,22],-20,20))
def on_q(q,p,qa,qs):
    pos=np.clip((q-qa[:,None])/np.maximum(qs[:,None],1e-8),0,1)*20; lo=np.floor(pos).astype(np.int16); hi=np.minimum(lo+1,20); f=pos-lo
    return np.take_along_axis(p,lo,1)+f*(np.take_along_axis(p,hi,1)-np.take_along_axis(p,lo,1))
def decode(z,bundle):
    pca=bundle["pca"]; sc=bundle["scaler"]; full=np.zeros((len(z),int(pca.n_components_)),float); full[:,:z.shape[1]]=z; return sc.inverse_transform(pca.inverse_transform(full))
def zmat(d,prefix,targets): return np.column_stack([num(d[f"{prefix}{t}"]).to_numpy(float) for t in targets])
def prev_vec(d):
    req=[*REFP,"reference_curvevec_q_anchor_lag1","reference_curvevec_log_q_span_lag1"]
    miss=[c for c in req if c not in d.columns]
    if miss: return None
    return np.column_stack([*[num(d[c]).to_numpy(float) for c in REFP],num(d.reference_curvevec_q_anchor_lag1).to_numpy(float),num(d.reference_curvevec_log_q_span_lag1).to_numpy(float)])
def state(): return dict(rows=0,ae=0.,se=0.,abst=0.,n=0,smape=0.,curves=[],qa=0.,qaa=0.,qs=0.,qsa=0.)
def update(st,tq,tp,tqa,tqs,v):
    p,qa,qs=unpack(v); pred=on_q(tq,p,qa,qs); e=pred-tp; ae=np.abs(e); den=np.abs(pred)+np.abs(tp); sm=np.divide(2*ae,den,out=np.zeros_like(ae),where=den>1e-9)
    st["rows"]+=len(tp); st["ae"]+=ae.sum(); st["se"]+=(e*e).sum(); st["abst"]+=np.abs(tp).sum(); st["n"]+=ae.size; st["smape"]+=sm.sum(); st["curves"].append(ae.mean(axis=1).astype(np.float32)); st["qa"]+=np.abs(qa-tqa).sum(); st["qaa"]+=np.abs(tqa).sum(); st["qs"]+=np.abs(qs-tqs).sum(); st["qsa"]+=np.abs(tqs).sum()
def finish(st,split,model,eligible):
    c=np.concatenate(st["curves"]) if st["curves"] else np.empty(0)
    return dict(split=split,model=model,eligible_for_feature_only_selection=eligible,rows=st["rows"],price_mae=st["ae"]/max(st["n"],1),price_rmse=float(np.sqrt(st["se"]/max(st["n"],1))),price_wape_pct=100*st["ae"]/max(st["abst"],1e-12),price_smape_pct=100*st["smape"]/max(st["n"],1),curve_mae_p50=float(np.quantile(c,.5)) if len(c) else np.nan,curve_mae_p90=float(np.quantile(c,.9)) if len(c) else np.nan,curve_mae_p95=float(np.quantile(c,.95)) if len(c) else np.nan,q_anchor_wape_pct=100*st["qa"]/max(st["qaa"],1e-12),q_span_wape_pct=100*st["qs"]/max(st["qsa"],1e-12))
def evaluate(model_dir,m,split,bundle,targets):
    names=[*MODELS,"absolute_latent_oracle","persistence_reference"]; sts={n:state() for n in names}; fs=parts(m,split)
    for i,r in enumerate(fs,1):
        print(f"[{split} {i}/{len(fs)}] {Path(r).name}",flush=True); d=pd.read_pickle(model_dir/r); tq,tp,tqa,tqs=true_curve(d)
        for name in MODELS:
            v=decode(zmat(d,f"pred_{name}_",targets),bundle); update(sts[name],tq,tp,tqa,tqs,v)
        update(sts["absolute_latent_oracle"],tq,tp,tqa,tqs,decode(zmat(d,"true_",targets),bundle))
        pv=prev_vec(d)
        if pv is not None: update(sts["persistence_reference"],tq,tp,tqa,tqs,pv)
        del d; gc.collect()
    return pd.DataFrame([finish(sts[n],split,n,n in MODELS) for n in names if sts[n]["rows"]>0])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int,default=2025); ap.add_argument("--root",default="data/processed/bidprediction"); ap.add_argument("--representation-dir",default="macro_b_absolute_curve_representation"); ap.add_argument("--model-dir",default="macro_b_feature_only_regression_models"); ap.add_argument("--overwrite",action="store_true"); a=ap.parse_args()
    base=Path(a.root)/str(a.year); rep=base/a.representation_dir; md=base/a.model_dir; bundle=joblib.load(rep/"absolute_pca_bundle.joblib"); m=json.loads((md/"manifest.json").read_text(encoding="utf-8")); targets=list(m["latent_columns"]); out=base/"macro_b_feature_only_evaluation"
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    val=evaluate(md,m,"val",bundle,targets); cand=val[val.eligible_for_feature_only_selection].sort_values(["price_wape_pct","price_mae"]).reset_index(drop=True); selected=str(cand.iloc[0].model); val.to_csv(out/"validation_curve_metrics.csv",index=False,encoding="utf-8-sig")
    test=evaluate(md,m,"test",bundle,targets); test["selected_by_validation"]=test.model.eq(selected); test.to_csv(out/"test_curve_metrics.csv",index=False,encoding="utf-8-sig")
    selection=dict(selection_split="validation",selection_metric="feature-only reconstructed full-curve price WAPE",selected_model=selected,eligible_models=MODELS,persistence_eligible=False,test_used_for_selection=False); (out/"selected_model.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    def w(t,n):
        x=t.loc[t.model.eq(n),"price_wape_pct"]; return float(x.iloc[0]) if len(x) else np.nan
    vm=w(val,"train_mean"); vs=w(val,selected); tm=w(test,"train_mean"); ts=w(test,selected); tp=w(test,"persistence_reference"); oracle=w(test,"absolute_latent_oracle")
    gain=100*(tm-ts)/tm if np.isfinite(tm) and tm else np.nan
    summary="\n".join([f"08d Macro-B feature-only forecast evaluation - {a.year}","="*80,"","Primary question: can profile/transition/market/unit/calendar features predict the absolute bid curve WITHOUT previous bid-curve input?","",f"Selected feature-only model from VALIDATION = {selected}","","VALIDATION:",val.to_string(index=False),"","TEST:",test.to_string(index=False),"",f"VAL train-mean WAPE = {vm:.6f}%",f"VAL selected WAPE = {vs:.6f}%",f"TEST train-mean WAPE = {tm:.6f}%",f"TEST selected WAPE = {ts:.6f}%",f"TEST relative improvement vs no-feature train mean = {gain:.4f}%",f"TEST absolute-latent oracle WAPE = {oracle:.6f}%",f"TEST Persistence reference WAPE = {tp:.6f}%","","Persistence is shown only as an information-rich cross-task reference and is NOT eligible for feature-only model selection."])
    (out/"summary.txt").write_text(summary,encoding="utf-8"); print(summary); print(f"Outputs: {out}")
if __name__=="__main__": main()
