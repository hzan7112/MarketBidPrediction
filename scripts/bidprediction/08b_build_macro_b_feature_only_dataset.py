#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08b: Build Macro-B feature-only absolute-curve forecasting dataset.

Allowed model inputs:
- strategy profile / transition features
- market features
- unit features
- calendar / slot features

Forbidden as MODEL INPUTS:
- previous raw bid curve
- historical latent features
- DeltaZ history
- historical theta / template inertia

Previous raw curve is copied only to reference_* columns so 08d can report
Persistence as a cross-task reference; those columns are never in model_features.
"""
from __future__ import annotations
import argparse, gc, json, shutil
from pathlib import Path
import joblib, numpy as np, pandas as pd

SHAPE=[f"shape_v{i:02d}" for i in range(21)]
PREV_P=[f"_curvevec_p{i:02d}_lag1" for i in range(21)]
PREV_RAW=[*PREV_P,"_curvevec_q_anchor_lag1","_curvevec_log_q_span_lag1"]
META=["sample_id","participant_id","timestamp_utc","timestamp_local","local_date","local_slot_seconds","prediction_cutoff_utc"]
CURVE=[*SHAPE,"p_anchor","p_span","q_anchor_mw","q_span_mw"]
FORBIDDEN=("latent","curvevec","theta","template","shape_v","p_anchor","p_span","q_anchor","q_span")

def num(s): return pd.to_numeric(s,errors="coerce")
def parts(m,s): return [x["file"] if isinstance(x,dict) else x for x in m["parts"][s]]
def current_vec(d):
    sh=d[SHAPE].apply(pd.to_numeric,errors="coerce").to_numpy(float); pa=num(d.p_anchor).to_numpy(float); ps=num(d.p_span).to_numpy(float)
    zero=np.abs(ps)<=1e-12
    if zero.any(): sh[zero]=np.nan_to_num(sh[zero],nan=0.0,posinf=0.0,neginf=0.0)
    p=pa[:,None]+ps[:,None]*sh; qa=num(d.q_anchor_mw).to_numpy(float); qs=num(d.q_span_mw).to_numpy(float)
    return np.c_[p,qa,np.log(np.maximum(qs,1e-8))]
def load_reference_base_features(base,source_features):
    candidates=[
        base/"curve_latent_forecasting_dataset"/"manifest.json",
        base/"curve_latent_dataset"/"manifest.json",
    ]
    keys=["base_context_features","base_features","context_features"]
    for p in candidates:
        if not p.exists(): continue
        try: m=json.loads(p.read_text(encoding="utf-8"))
        except Exception: continue
        for k in keys:
            vals=m.get(k)
            if isinstance(vals,list) and vals:
                keep=[x for x in vals if x in source_features]
                if keep: return keep,f"{p}:{k}"
    keep=[f for f in source_features if not any(t in f.lower() for t in FORBIDDEN)]
    return keep,"fallback: source_model_features minus direct historical-bid tokens"
def add_calendar(d):
    added=[]
    if "local_slot_seconds" in d.columns:
        slot=num(d["local_slot_seconds"]).to_numpy(float); ang=2*np.pi*slot/86400.0
        d["calendar_slot_sin"]=np.sin(ang).astype(np.float32); d["calendar_slot_cos"]=np.cos(ang).astype(np.float32); added += ["calendar_slot_sin","calendar_slot_cos"]
    if "local_date" in d.columns:
        dt=pd.to_datetime(d["local_date"],errors="coerce"); dow=dt.dt.dayofweek.to_numpy(float); doy=dt.dt.dayofyear.to_numpy(float)
        d["calendar_dow_sin"]=np.sin(2*np.pi*dow/7).astype(np.float32); d["calendar_dow_cos"]=np.cos(2*np.pi*dow/7).astype(np.float32)
        d["calendar_doy_sin"]=np.sin(2*np.pi*doy/365.25).astype(np.float32); d["calendar_doy_cos"]=np.cos(2*np.pi*doy/365.25).astype(np.float32)
        added += ["calendar_dow_sin","calendar_dow_cos","calendar_doy_sin","calendar_doy_cos"]
    return added
def leakage_check(features):
    bad=[f for f in features if any(t in f.lower() for t in FORBIDDEN)]
    if bad: raise RuntimeError("Forbidden direct historical-bid information entered model_features: "+", ".join(bad[:30]))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int,default=2025); ap.add_argument("--root",default="data/processed/bidprediction"); ap.add_argument("--source-dir",default="macro_b_filtered_dataset"); ap.add_argument("--representation-dir",default="macro_b_absolute_curve_representation"); ap.add_argument("--overwrite",action="store_true"); a=ap.parse_args()
    base=Path(a.root)/str(a.year); src=base/a.source_dir; rep=base/a.representation_dir; sm=json.loads((src/"manifest.json").read_text(encoding="utf-8")); rb=joblib.load(rep/"absolute_pca_bundle.joblib"); k=int(rb["selected_latent_dim"]); targets=[f"absolute_latent_z{i+1:02d}" for i in range(k)]
    source_features=list(sm.get("source_model_features",[])); base_features,feature_source=load_reference_base_features(base,source_features); leakage_check(base_features)
    out=base/"macro_b_feature_only_dataset"
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    out_parts={}; final_features=None; stats={}
    for split in ["train","val","test"]:
        od=out/"parts"/split; od.mkdir(parents=True,exist_ok=True); written=[]; rows=0
        for i,r in enumerate(parts(sm,split),1):
            print(f"[{split} {i}/{len(parts(sm,split))}] {Path(r).name}",flush=True); d=pd.read_pickle(src/r); cal=add_calendar(d); feats=list(dict.fromkeys([*base_features,*cal])); leakage_check(feats)
            missing=[f for f in feats if f not in d.columns]
            if missing: raise KeyError(f"Missing model features in {r}: {missing[:20]}")
            if final_features is None: final_features=feats
            elif feats!=final_features: raise RuntimeError("Feature list changed between parts.")
            z=rb["pca"].transform(rb["scaler"].transform(current_vec(d)))[:,:k].astype(np.float32)
            keep=[c for c in META if c in d.columns]+feats+[c for c in CURVE if c in d.columns]
            outd=d[keep].copy()
            for j,t in enumerate(targets): outd[t]=z[:,j]
            # Reference-only previous curve columns: never part of model_features.
            for c in PREV_RAW:
                if c in d.columns: outd[f"reference{c}"]=d[c].to_numpy()
            name=f"{split}_feature_only_{i:04d}.pkl"; path=od/name; outd.to_pickle(path); written.append(dict(file=str(path.relative_to(out)),rows=len(outd))); rows+=len(outd)
            del d,outd,z; gc.collect()
        out_parts[split]=written; stats[split]=dict(rows=rows)
    leakage_check(final_features or [])
    mani=dict(year=a.year,source_dataset=str(src),representation=str(rep),feature_definition="Strategy profile/transition + market + unit + calendar/slot; no direct historical bid curve",feature_source=feature_source,source_model_features=source_features,model_features=final_features,latent_columns=targets,reference_only_columns=[f"reference{c}" for c in PREV_RAW],parts=out_parts,split_stats=stats)
    (out/"manifest.json").write_text(json.dumps(mani,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame({"feature":final_features}).to_csv(out/"model_features.csv",index=False,encoding="utf-8-sig")
    summary="\n".join([f"08b Macro-B feature-only dataset - {a.year}","="*80,"",f"Feature source = {feature_source}",f"Source model features = {len(source_features)}",f"Feature-only model features = {len(final_features)}",f"Absolute latent targets = {k}","","Historical bid leakage check = PASS","","Features:","\n".join(final_features),"","Split rows:",json.dumps(stats,ensure_ascii=False,indent=2)])
    (out/"summary.txt").write_text(summary,encoding="utf-8"); print(summary); print(f"Outputs: {out}")
if __name__=="__main__": main()
