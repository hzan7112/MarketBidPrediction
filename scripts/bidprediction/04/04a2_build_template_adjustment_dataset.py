#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build low-dimensional template-adjustment targets.

Requires the validated output from 04a_build_curve_parameter_dataset.py.
Final theta:
    p_base, alpha, beta, q_base_mw, q_span_mw, q1..q5
with alpha,beta>=0; qk>0 and sum(qk)=1.

No curve_mode / segment_count / breakpoint_count classifier is used.
"""
from __future__ import annotations

import argparse, json, re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

GRID=np.linspace(0.0,1.0,21)
UK=np.linspace(0.0,1.0,6)
TAIL_START=0.70
TEMPLATES=[f"T{i:02d}" for i in range(12)]+["FLAT"]
BASE_TARGETS=["q_anchor_mw","q_span_mw","p_anchor","p_span"]
THETA=["theta_p_base","theta_alpha","theta_beta","theta_q_base_mw","theta_q_span_mw",
       "theta_q1","theta_q2","theta_q3","theta_q4","theta_q5"]


def num(s): return pd.to_numeric(s,errors="coerce")
def norm_id(s): return s.astype("string").str.strip()
def mkdir(p): p.mkdir(parents=True,exist_ok=True); return p

def month_key(p:Path):
    m=re.search(r"(\d{4})[_-](\d{2})",p.stem)
    return f"{m.group(1)}-{m.group(2)}" if m else None

def shape_cols(header):
    exact=[f"shape_v{i:02d}" for i in range(21)]
    if all(c in header for c in exact): return exact
    groups={}
    for c in header:
        m=re.match(r"^(.*?)(?:_)?v(\d{2})$",str(c),flags=re.I)
        if m and 0<=int(m.group(2))<=20:
            groups.setdefault(m.group(1),{})[int(m.group(2))]=c
    for g in groups.values():
        if len(g)==21 and all(i in g for i in range(21)):
            return [g[i] for i in range(21)]
    return None

def template_id(v):
    s=str(v).strip()
    if s.upper()=="FLAT": return "FLAT"
    m=re.fullmatch(r"T?(\d{1,2})",s,flags=re.I)
    if m and 0<=int(m.group(1))<=11: return f"T{int(m.group(1)):02d}"
    try:
        i=int(float(s))
        return f"T{i:02d}" if 0<=i<=11 else None
    except: return None

def discover_samples(year_dir:Path, explicit=None):
    root=Path(explicit) if explicit else year_dir/"curve_samples"
    if not root.exists(): raise FileNotFoundError(f"Curve samples not found: {root}")
    out={}
    for p in sorted(root.rglob("*.csv")):
        try: h=pd.read_csv(p,nrows=0).columns.tolist()
        except: continue
        if "sample_id" not in h or shape_cols(h) is None: continue
        mk=month_key(p)
        if mk: out.setdefault(mk,[]).append(p)
    if not out: raise FileNotFoundError(f"No sample_id + 21-shape CSV found under {root}")
    return out

def discover_centers(year_dir:Path, explicit=None):
    cand=[Path(explicit)] if explicit else []
    if not explicit:
        scored=[]
        for p in year_dir.rglob("*.csv"):
            try: sz=p.stat().st_size
            except: continue
            if sz>5_000_000: continue
            n=p.name.lower(); sc=0
            sc+=5 if ("center" in n or "centroid" in n) else 0
            sc+=4 if "template" in n else 0
            sc+=2 if "cluster" in n else 0
            sc-=5 if "sample" in n else 0
            sc-=3 if "parameter" in n else 0
            scored.append((sc,p))
        cand=[p for _,p in sorted(scored,key=lambda z:(-z[0],str(z[1])))]
    for p in cand:
        if not p.exists(): continue
        try: h=pd.read_csv(p,nrows=0).columns.tolist()
        except: continue
        scols=shape_cols(h)
        if scols is None: continue
        idc=next((c for c in ["template_id","template","cluster_id","cluster_label","label"] if c in h),None)
        if idc is None: continue
        d=pd.read_csv(p,usecols=[idc]+scols)
        if len(d)>100: continue
        centers={}
        for _,r in d.iterrows():
            tid=template_id(r[idc])
            if tid is None: continue
            a=pd.to_numeric(r[scols],errors="coerce").to_numpy(float)
            if np.isfinite(a).all(): centers[tid]=a
        centers["FLAT"]=np.zeros(21)
        if all(t in centers for t in TEMPLATES): return centers,p
    raise FileNotFoundError("Template-center CSV not found; pass --template-centers <csv>.")

def load_shapes(files):
    blocks=[]
    for p in files:
        h=pd.read_csv(p,nrows=0).columns.tolist(); sc=shape_cols(h)
        if sc is None: continue
        d=pd.read_csv(p,usecols=["sample_id"]+sc,low_memory=False)
        d["sample_id"]=norm_id(d["sample_id"])
        d=d.rename(columns={c:f"__shape_{i:02d}" for i,c in enumerate(sc)})
        blocks.append(d)
    if not blocks: return pd.DataFrame()
    d=pd.concat(blocks,ignore_index=True)
    if d["sample_id"].duplicated().any(): raise ValueError("Duplicate sample_id in monthly curve samples")
    return d

def parse_bp(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return []
    try:
        a=json.loads(v.strip()) if isinstance(v,str) and v.strip() else (v if not isinstance(v,str) else [])
    except: return []
    if a is None: return []
    z=[]
    for x in a:
        try: f=float(x)
        except: continue
        if np.isfinite(f) and 0<f<1: z.append(f)
    return sorted(set(z))

def qshares(v):
    bp=parse_bp(v); b=np.array([0.0,*bp,1.0],float)
    w=np.maximum(np.diff(b),1e-8); w/=w.sum()
    cq=np.r_[0.0,np.cumsum(w)]; rank=np.linspace(0,1,len(cq))
    q=np.diff(np.interp(UK,rank,cq)); q=np.maximum(q,1e-6); q/=q.sum()
    return q

def inverse_warp(Q):
    n=len(Q); cum=np.c_[np.zeros(n),np.cumsum(Q,axis=1)]; cum[:,-1]=1.0
    U=np.empty((n,21),float); row=np.arange(n)
    for j,x in enumerate(GRID):
        k=np.sum(x>=cum[:,1:],axis=1); k=np.clip(k,0,4)
        x0=cum[row,k]; x1=cum[row,k+1]; f=np.clip((x-x0)/np.maximum(x1-x0,1e-12),0,1)
        U[:,j]=UK[k]+f*(UK[k+1]-UK[k])
    U[:,0]=0; U[:,-1]=1
    return U

def interp_centers(C,U):
    pos=np.clip(U*20,0,20); lo=np.floor(pos).astype(np.int16); hi=np.minimum(lo+1,20); f=pos-lo
    row=np.arange(len(U))[:,None]
    return C[row,lo]+f*(C[row,hi]-C[row,lo])

def tail(U):
    z=np.maximum(0,(U-TAIL_START)/(1-TAIL_START)); return z*z

def fit_price(y,s,h):
    """Vectorized active-set LS for y=c+a*s+b*h, a,b>=0."""
    n,m=y.shape; sy=s.sum(1); hh=h.sum(1); yy=y.sum(1)
    ss=(s*s).sum(1); h2=(h*h).sum(1); sh=(s*h).sum(1); syy=(s*y).sum(1); hyy=(h*y).sum(1)
    A=np.empty((n,3,3)); B=np.c_[yy,syy,hyy]
    A[:,0,0]=m; A[:,0,1]=A[:,1,0]=sy; A[:,0,2]=A[:,2,0]=hh
    A[:,1,1]=ss+1e-10; A[:,1,2]=A[:,2,1]=sh; A[:,2,2]=h2+1e-10
    try: cf=np.linalg.solve(A,B[...,None])[...,0]
    except np.linalg.LinAlgError:
        cf=np.vstack([np.linalg.lstsq(np.c_[np.ones(m),s[i],h[i]],y[i],rcond=None)[0] for i in range(n)])
    c0,a0,b0=cf[:,0],cf[:,1],cf[:,2]
    e0=((c0[:,None]+a0[:,None]*s+b0[:,None]*h-y)**2).sum(1)
    e0=np.where((a0>=0)&(b0>=0),e0,np.inf)
    ym=y.mean(1); sm=s.mean(1); hm=h.mean(1); yc=y-ym[:,None]
    sc=s-sm[:,None]; hc=h-hm[:,None]
    ds=(sc*sc).sum(1); dh=(hc*hc).sum(1)
    a1=np.maximum(np.divide((sc*yc).sum(1),ds,out=np.zeros(n),where=ds>1e-12),0); c1=ym-a1*sm
    e1=((c1[:,None]+a1[:,None]*s-y)**2).sum(1)
    b2=np.maximum(np.divide((hc*yc).sum(1),dh,out=np.zeros(n),where=dh>1e-12),0); c2=ym-b2*hm
    e2=((c2[:,None]+b2[:,None]*h-y)**2).sum(1)
    c3=ym; e3=((c3[:,None]-y)**2).sum(1)
    E=np.c_[e0,e1,e2,e3]; ch=np.argmin(E,1); row=np.arange(n)
    c=np.where(ch==0,c0,np.where(ch==1,c1,np.where(ch==2,c2,c3)))
    a=np.where(ch==0,a0,np.where(ch==1,a1,0.0)); b=np.where(ch==0,b0,np.where(ch==2,b2,0.0))
    rmse=np.sqrt(E[row,ch]/m)
    return c,a,b,rmse

def build_theta(d,centers):
    n=len(d); tid=d["y_template_id"].astype("string").str.strip()
    qb=num(d["q_anchor_mw"]).to_numpy(float); qs=num(d["q_span_mw"]).to_numpy(float)
    pa=num(d["p_anchor"]).to_numpy(float); ps=num(d["p_span"]).to_numpy(float)
    bpcol=next((c for c in ["label_breakpoint_x_json","y_breakpoint_x_json"] if c in d.columns),None)
    if bpcol is None: raise KeyError("Need label_breakpoint_x_json or y_breakpoint_x_json")
    Q=np.vstack([qshares(v) for v in d[bpcol].tolist()])
    c=np.full(n,np.nan); a=np.full(n,np.nan); b=np.full(n,np.nan); rm=np.full(n,np.nan)
    flat=tid.eq("FLAT").to_numpy(); c[flat]=pa[flat]; a[flat]=0; b[flat]=0; rm[flat]=0
    idx=np.where(~flat)[0]
    if len(idx):
        sc=[f"__shape_{i:02d}" for i in range(21)]
        Yshape=d.iloc[idx][sc].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        good=np.isfinite(Yshape).all(1); gi=idx[good]
        if len(gi):
            U=inverse_warp(Q[gi]); C=np.vstack([centers[str(tid.iloc[i])] for i in gi])
            S=interp_centers(C,U); H=tail(U); Y=pa[gi,None]+ps[gi,None]*Yshape[good]
            cc,aa,bb,rr=fit_price(Y,S,H); c[gi]=cc; a[gi]=aa; b[gi]=bb; rm[gi]=rr
    d=d.copy(); d["theta_p_base"]=c; d["theta_alpha"]=a; d["theta_beta"]=b
    d["theta_q_base_mw"]=qb; d["theta_q_span_mw"]=qs
    for k in range(5): d[f"theta_q{k+1}"]=Q[:,k]
    d["theta_price_fit_rmse"]=rm
    valid=(np.isfinite(c)&np.isfinite(a)&np.isfinite(b)&(a>=0)&(b>=0)&np.isfinite(qb)&np.isfinite(qs)&(qs>0)&np.isfinite(Q).all(1)&(Q>0).all(1)&(np.abs(Q.sum(1)-1)<1e-6))
    pr=num(d["prediction_ready_flag"]).fillna(0).eq(1).to_numpy()
    d["theta_target_valid_flag"]=valid.astype("int8"); d["theta_ready_flag"]=(valid&pr).astype("int8")
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",type=int,default=2025)
    ap.add_argument("--prediction-root",default="data/processed/bidprediction")
    ap.add_argument("--bidtemplate-root",default="data/processed/bidtemplate")
    ap.add_argument("--curve-samples",default=None)
    ap.add_argument("--template-centers",default=None)
    ap.add_argument("--chunksize",type=int,default=100000)
    args=ap.parse_args()
    pbase=Path(args.prediction_root)/str(args.year); old=pbase/"curve_parameter_dataset"; parts=sorted((old/"dataset_parts").glob("curve_parameter_dataset_*.csv"))
    if not parts: raise FileNotFoundError("Run 04a_build_curve_parameter_dataset.py first")
    bty=Path(args.bidtemplate_root)/str(args.year); sf=discover_samples(bty,args.curve_samples); centers,cp=discover_centers(bty,args.template_centers)
    out=mkdir(pbase/"template_adjustment_dataset"); outparts=mkdir(out/"dataset_parts")
    old_schema=pd.read_csv(old/f"curve_parameter_dataset_schema_{args.year}.csv")
    totals=Counter(); manifest=[]; print("="*80); print("Build template-adjustment dataset"); print("="*80); print(f"Template centers: {cp}")
    for i,part in enumerate(parts,1):
        mk=month_key(part)
        if mk not in sf: raise FileNotFoundError(f"No curve samples for {mk}")
        print(f"[part {i}/{len(parts)}] {part.name}",flush=True); shapes=load_shapes(sf[mk])
        of=outparts/part.name.replace("curve_parameter_dataset_","template_adjustment_dataset_",1)
        if of.exists(): of.unlink()
        first=True; rows=ready=0
        for ch in pd.read_csv(part,chunksize=args.chunksize,low_memory=False):
            ch["sample_id"]=norm_id(ch["sample_id"]); x=ch.merge(shapes,on="sample_id",how="left",validate="one_to_one",sort=False)
            x=build_theta(x,centers); rows+=len(x); ready+=int(x["theta_ready_flag"].sum()); totals["rows"]+=len(x); totals["ready"]+=int(x["theta_ready_flag"].sum()); totals["pred"]+=int(num(x["prediction_ready_flag"]).fillna(0).eq(1).sum())
            x=x.drop(columns=[f"__shape_{j:02d}" for j in range(21)],errors="ignore")
            x.to_csv(of,mode="w" if first else "a",header=first,index=False,encoding="utf-8-sig"); first=False
        manifest.append({"input_part":str(part),"output_part":str(of),"month":mk,"rows":rows,"theta_ready_rows":ready})
    schema=old_schema.to_dict("records"); existing=set(old_schema["column"].astype(str))
    for c in THETA:
        if c not in existing: schema.append({"column":c,"role":"target","feature_group":"template_adjustment_target","source":"04a2_derived","leakage_use":"target_only"})
    for c in ["theta_price_fit_rmse","theta_target_valid_flag","theta_ready_flag"]:
        if c not in existing: schema.append({"column":c,"role":"quality_flag","feature_group":"template_adjustment_quality","source":"04a2_derived","leakage_use":"audit_only"})
    pd.DataFrame(schema).to_csv(out/f"template_adjustment_schema_{args.year}.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(out/f"template_adjustment_manifest_{args.year}.csv",index=False,encoding="utf-8-sig")
    cfg={"year":args.year,"template_center_file":str(cp),"tail_start":TAIL_START,"quantity_groups":5,"theta_targets":THETA,"constraints":{"alpha":">=0","beta":">=0","q_span_mw":">0","qk":">0, sum=1"}}
    (out/f"template_adjustment_build_config_{args.year}.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
    summary="\n".join([f"Template adjustment dataset - {args.year}","="*80,"",f"Rows: {totals['rows']:,}",f"Prediction-ready rows: {totals['pred']:,}",f"Theta-ready rows: {totals['ready']:,}",f"Theta-ready / prediction-ready: {totals['ready']/totals['pred']:.2%}","","Final theta:","  p_base, alpha, beta","  q_base_mw, q_span_mw","  q1..q5 (continuous shares, sum=1)","","No curve_mode / segment_count / breakpoint_count classifier."])
    (out/"summary.txt").write_text(summary,encoding="utf-8"); print("\n"+summary); print(f"\nOutputs: {out}")

if __name__=="__main__": main()
