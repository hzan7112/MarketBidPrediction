#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""08a: Macro-B absolute curve PCA representation (no historical curve input)."""
from __future__ import annotations
import argparse, gc, json, math, shutil
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GRID=np.linspace(0,1,21); SHAPE=[f"shape_v{i:02d}" for i in range(21)]
def num(s): return pd.to_numeric(s,errors="coerce")
def parts(m,s): return [x["file"] if isinstance(x,dict) else x for x in m["parts"][s]]
def vec(d):
    sh=d[SHAPE].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    pa=num(d.p_anchor).to_numpy(float); ps=num(d.p_span).to_numpy(float)
    z=np.abs(ps)<=1e-12
    if z.any(): sh[z]=np.nan_to_num(sh[z],nan=0.0,posinf=0.0,neginf=0.0)
    p=pa[:,None]+ps[:,None]*sh
    qa=num(d.q_anchor_mw).to_numpy(float); qs=num(d.q_span_mw).to_numpy(float)
    return np.c_[p,qa,np.log(np.maximum(qs,1e-8))]
def qtrue(d):
    qa=num(d.q_anchor_mw).to_numpy(float); qs=num(d.q_span_mw).to_numpy(float)
    return qa[:,None]+qs[:,None]*GRID[None,:]
def unpack(v): return v[:,:21],v[:,21],np.exp(np.clip(v[:,22],-20,20))
def on_q(q,p,qa,qs):
    pos=np.clip((q-qa[:,None])/np.maximum(qs[:,None],1e-8),0,1)*20
    lo=np.floor(pos).astype(np.int16); hi=np.minimum(lo+1,20); f=pos-lo
    return np.take_along_axis(p,lo,1)+f*(np.take_along_axis(p,hi,1)-np.take_along_axis(p,lo,1))
def fit_sample(src,m,max_rows,seed):
    fs=parts(m,"train"); per=max(1,math.ceil(max_rows/max(len(fs),1))); blocks=[]
    for i,r in enumerate(fs,1):
        d=pd.read_pickle(src/r)
        if len(d)>per: d=d.sample(per,random_state=seed+1009*i)
        blocks.append(vec(d)); del d; gc.collect()
    x=np.vstack(blocks)
    if len(x)>max_rows:
        rng=np.random.default_rng(seed); x=x[rng.choice(len(x),max_rows,replace=False)]
    return x
def evaluate(src,m,split,sc,pca,dims):
    st={k:dict(rows=0,ae=0.,se=0.,abst=0.,n=0,vae=0.,vn=0,qa=0.,qaa=0.,qs=0.,qsa=0.) for k in dims}
    for r in parts(m,split):
        d=pd.read_pickle(src/r); tv=vec(d); tq=qtrue(d); tp=tv[:,:21]; tqa=tv[:,21]; tqs=np.exp(np.clip(tv[:,22],-20,20))
        z=pca.transform(sc.transform(tv))
        for k in dims:
            zz=np.zeros_like(z); zz[:,:k]=z[:,:k]; rec=sc.inverse_transform(pca.inverse_transform(zz)); pp,pqa,pqs=unpack(rec); pred=on_q(tq,pp,pqa,pqs)
            e=pred-tp; a=np.abs(e); s=st[k]
            s["rows"]+=len(d); s["ae"]+=a.sum(); s["se"]+=(e*e).sum(); s["abst"]+=np.abs(tp).sum(); s["n"]+=a.size
            s["vae"]+=np.abs(rec-tv).sum(); s["vn"]+=rec.size; s["qa"]+=np.abs(pqa-tqa).sum(); s["qaa"]+=np.abs(tqa).sum(); s["qs"]+=np.abs(pqs-tqs).sum(); s["qsa"]+=np.abs(tqs).sum()
        del d,tv,tq,tp,z; gc.collect()
    cum=np.cumsum(pca.explained_variance_ratio_); rows=[]
    for k,s in st.items():
        rows.append(dict(split=split,latent_dim=k,rows=s["rows"],cumulative_explained_variance=float(cum[k-1]),absolute_vector_mae=s["vae"]/s["vn"],price_mae=s["ae"]/s["n"],price_rmse=float(np.sqrt(s["se"]/s["n"])),price_wape_pct=100*s["ae"]/s["abst"],q_anchor_wape_pct=100*s["qa"]/max(s["qaa"],1e-12),q_span_wape_pct=100*s["qs"]/max(s["qsa"],1e-12)))
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int,default=2025); ap.add_argument("--root",default="data/processed/bidprediction"); ap.add_argument("--source-dir",default="macro_b_filtered_dataset"); ap.add_argument("--max-fit-rows",type=int,default=500000); ap.add_argument("--max-components",type=int,default=20); ap.add_argument("--variance-target",type=float,default=.995); ap.add_argument("--candidate-dims",default="2,3,4,5,6,8,10,12,16,20"); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--overwrite",action="store_true"); a=ap.parse_args()
    base=Path(a.root)/str(a.year); src=base/a.source_dir; m=json.loads((src/"manifest.json").read_text(encoding="utf-8")); out=base/"macro_b_absolute_curve_representation"
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    x=fit_sample(src,m,a.max_fit_rows,a.seed); sc=StandardScaler().fit(x); pca=PCA(n_components=min(a.max_components,x.shape[1]),svd_solver="randomized",random_state=a.seed).fit(sc.transform(x)); cum=np.cumsum(pca.explained_variance_ratio_); hit=np.where(cum>=a.variance_target)[0]; k=int(hit[0]+1) if len(hit) else pca.n_components_
    dims=sorted({k,pca.n_components_,*[int(v) for v in a.candidate_dims.split(",") if v.strip()]}); dims=[d for d in dims if d<=pca.n_components_]
    met=pd.concat([evaluate(src,m,s,sc,pca,dims) for s in ["val","test"]],ignore_index=True); met.to_csv(out/"representation_metrics.csv",index=False,encoding="utf-8-sig")
    joblib.dump(dict(scaler=sc,pca=pca,selected_latent_dim=k,vector_definition="21 absolute prices + q_anchor_mw + log(q_span_mw)"),out/"absolute_pca_bundle.joblib",compress=3)
    mani=dict(year=a.year,source_dataset=str(src),selected_latent_dim=k,latent_columns=[f"absolute_latent_z{i+1:02d}" for i in range(k)],vector_definition="21 absolute prices + q_anchor_mw + log(q_span_mw)"); (out/"manifest.json").write_text(json.dumps(mani,ensure_ascii=False,indent=2),encoding="utf-8")
    sel=met[met.latent_dim.eq(k)]; summary="\n".join([f"08a Macro-B absolute curve representation - {a.year}","="*80,"",f"Fit rows = {len(x):,}",f"Selected latent dimension = {k}",f"TRAIN cumulative explained variance = {cum[k-1]:.6f}","","Selected dimension:",sel.to_string(index=False),"","All dimensions:",met.to_string(index=False)]); (out/"summary.txt").write_text(summary,encoding="utf-8"); print(summary); print(f"Outputs: {out}")
if __name__=="__main__": main()
