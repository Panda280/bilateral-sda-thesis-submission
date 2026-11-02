#!/usr/bin/env python3
"""Role-safe, sign-stratified descriptive case selection (contract v2)."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED={"year_current","effect","attribution_role","resolution","selection_sign","entity_type","canonical_entity","mapping_status","comparison_eligible","value","units","anomaly_score","zero_mad","cancellation_ratio","upstream_result_id"}
ALLOWED_MAPPING={"reviewed_unique","reviewed_identity","reviewed","reviewed_native"}
RESOLUTIONS={"2013","2016","R3"}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable_id(row):
    payload="|".join(str(row[k]) for k in ("effect","attribution_role","resolution","selection_sign","selection_scope","entity_type","canonical_entity","units"))
    return "CASE-"+hashlib.sha256(("case-selection-v2|"+payload).encode()).hexdigest()[:16].upper()

def build(table:Path, upstream:list[Path], *, min_materiality=.01,min_persistence=.6,max_resolution_sensitivity=.5,top_n=5):
    blockers=[]; manifests=[]
    if not upstream: blockers.append("at least one upstream manifest is required")
    for p in upstream:
        if not p.is_file(): blockers.append(f"missing upstream manifest: {p}"); continue
        m=json.loads(p.read_text()); manifests.append({"path":str(p),"sha256":digest(p)})
        if m.get("status") not in {"complete","accepted_provisional"}: blockers.append(f"upstream manifest not accepted: {p}")
    if not table.is_file(): return None,None,None,pd.DataFrame(),[f"missing table: {table}",*blockers],manifests
    f=pd.read_csv(table,dtype={"resolution":str}); missing=REQUIRED-set(f)
    if missing: blockers.append(f"missing columns: {sorted(missing)}")
    if blockers: return None,None,None,pd.DataFrame(),blockers,manifests
    if set(f.resolution)!=RESOLUTIONS: blockers.append("resolution coverage must equal 2013, 2016, R3")
    if f.units.nunique()!=1: blockers.append("mixed units are not permitted")
    for c in ("value","anomaly_score","cancellation_ratio"):
        f[c]=pd.to_numeric(f[c],errors="coerce")
        if not np.isfinite(f[c]).all(): blockers.append(f"non-finite {c}")
    if not set(f.selection_sign).issubset({"positive","negative"}): blockers.append("selection_sign must be positive or negative")
    if ((f.selection_sign.eq("positive")&f.value.lt(0))|(f.selection_sign.eq("negative")&f.value.gt(0))).any(): blockers.append("value sign conflicts with selection_sign")
    if f.duplicated(["year_current","effect","attribution_role","resolution","selection_sign","entity_type","canonical_entity"]).any(): blockers.append("duplicate annual semantic key")
    if not (0<=min_materiality<=1 and 0<=min_persistence<=1 and max_resolution_sensitivity>=0 and top_n>0): blockers.append("invalid thresholds")
    if blockers: return None,None,None,pd.DataFrame(),blockers,manifests
    excluded=f[~f.mapping_status.isin(ALLOWED_MAPPING)].copy(); excluded["exclusion_reason"]="unresolved_cross_resolution_mapping"
    f=f[f.mapping_status.isin(ALLOWED_MAPPING)].copy()
    f["anomaly_score_used"]=np.where(f.zero_mad.astype(str).str.lower().isin(["true","1"]),0.0,f.anomaly_score.abs())
    f["exceptional_deviation_zero_mad"]=f.zero_mad.astype(str).str.lower().isin(["true","1"]) & f.value.ne(0)
    keys=["effect","attribution_role","resolution","selection_sign","entity_type","canonical_entity","units"]
    rows=[]
    annual_den=f.assign(abs_value=f.value.abs()).groupby(["year_current","effect","attribution_role","resolution"],as_index=False).abs_value.sum()
    f=f.merge(annual_den,on=["year_current","effect","attribution_role","resolution"],how="left")
    for key,p in f.groupby(keys,sort=True):
        total=p.value.sum(); sign=key[3]
        nonzero=p.value.ne(0); n=int(nonzero.sum()); persistence=max(int((p.value>0).sum()),int((p.value<0).sum()))/n if n else 0
        absolute=float(p.value.abs().sum()); directional=abs(float(total))/absolute if absolute else 0
        rows.append(dict(zip(keys,key),year_count=p.year_current.nunique(),signed_total=total,absolute_total=absolute,comparison_eligible=bool(p.comparison_eligible.astype(str).str.lower().isin(["true","1"]).all()),
                         materiality=float((p.value.abs()/p.abs_value.replace(0,np.nan)).fillna(0).mean()),sign_persistence=persistence,directional_balance=directional,
                         anomaly=float(p.anomaly_score_used.mean()),cancellation=float(p.cancellation_ratio.mean()),upstream_result_ids=" | ".join(sorted(set(p.upstream_result_id.astype(str))))))
    scores=pd.DataFrame(rows); zero=scores.absolute_total.eq(0)
    if zero.any():
        z=scores[zero].copy(); z["exclusion_reason"]="zero_cumulative_contribution"; excluded=pd.concat([excluded,z],ignore_index=True,sort=False)
    scores=scores[~zero].copy()
    comparable=scores[scores.comparison_eligible]
    sens=comparable.groupby(["effect","attribution_role","selection_sign","entity_type","canonical_entity","units"]).filter(lambda x:set(x.resolution)==RESOLUTIONS)
    sensitivity=sens.groupby(["effect","attribution_role","selection_sign","entity_type","canonical_entity","units"]).absolute_total.agg(lambda x:(x.max()-x.min())/x.mean() if x.mean() else 0).rename("resolution_sensitivity").reset_index()
    scores=scores.merge(sensitivity,how="left"); scores["complete_resolution_set"]=scores.resolution_sensitivity.notna()
    scores["selection_scope"]=np.where(scores.complete_resolution_set,"cross_resolution","native_resolution")
    scores["eligible"]=(scores.materiality>=min_materiality)&(scores.sign_persistence>=min_persistence)&((scores.selection_scope.eq("native_resolution"))|(scores.resolution_sensitivity<=max_resolution_sensitivity))
    scores["descriptive_score"]=.5*scores.materiality+.3*scores.sign_persistence+.2*(1-scores.resolution_sensitivity.clip(0,1).fillna(1))
    scores["result_id"]=scores.apply(stable_id,axis=1)
    sort=["effect","attribution_role","resolution","selection_sign","descriptive_score","canonical_entity"]
    scores=scores.sort_values(sort,ascending=[True,True,True,True,False,True])
    selected=scores[scores.eligible].groupby(["effect","attribution_role","resolution","selection_sign","selection_scope"],group_keys=False).head(top_n).copy()
    chosen=set(selected.result_id); plot=f.merge(scores[keys+["result_id"]],on=keys,how="inner"); plot=plot[plot.result_id.isin(chosen)].sort_values(["result_id","year_current"])
    return scores,selected,plot,excluded,blockers,manifests

def main():
    p=argparse.ArgumentParser(); p.add_argument("--table",type=Path,required=True); p.add_argument("--upstream-manifest",type=Path,action="append",default=[]); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--min-materiality",type=float,default=.01); p.add_argument("--min-persistence",type=float,default=.6); p.add_argument("--max-resolution-sensitivity",type=float,default=.5); p.add_argument("--top-n",type=int,default=5); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    scores,selected,plot,excluded,blockers,inputs=build(a.table,a.upstream_manifest,min_materiality=a.min_materiality,min_persistence=a.min_persistence,max_resolution_sensitivity=a.max_resolution_sensitivity,top_n=a.top_n)
    outputs=[]
    if not blockers:
        for name,data in (("case_scores.csv",scores),("selected_cases.csv",selected),("selected_case_plot_data.csv",plot),("exclusions.csv",excluded)):
            q=a.output_dir/name; data.to_csv(q,index=False); outputs.append({"path":name,"sha256":digest(q),"rows":len(data)})
    manifest={"schema_version":2,"analysis_id":"sign-stratified-case-selection-v2","status":"blocked" if blockers else "complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"inputs":inputs,"outputs":outputs,"coverage":{"excluded":0 if blockers else len(excluded)},"validation":{"passed":not blockers,"gates":{"sign_stratified":not blockers,"canonical_ids":not blockers,"unresolved_excluded":not blockers,"finite":not blockers,"units_preserved":not blockers,"stable_ids_unique":False if blockers else scores.result_id.is_unique},"thesis_status":"ineligible" if blockers else "accepted_provisional"},"parameters":{"min_materiality":a.min_materiality,"min_sign_persistence":a.min_persistence,"max_resolution_sensitivity":a.max_resolution_sensitivity,"top_n_per_effect_role_sign_resolution":a.top_n},"units":{"conversion":"none; input units preserved"},"blockers":blockers}
    (a.output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); return 2 if blockers else 0
if __name__=="__main__": raise SystemExit(main())
