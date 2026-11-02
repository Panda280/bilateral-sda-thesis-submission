#!/usr/bin/env python3
"""Predeclared, role-safe case selection from completed analysis tables."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from sda.analysis.integration import ScoreConfig, rank_descriptive_cases

REQUIRED = {"year", "effect", "role", "resolution", "entity", "value", "anomaly_score", "cancellation_ratio"}
RESOLUTIONS = {"2013", "2016", "R3"}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(table: Path, upstream: list[Path], *, min_materiality=.01, min_persistence=.6,
          max_resolution_sensitivity=.5, top_n=5):
    blockers=[]; inputs=[]
    if not upstream:
        blockers.append("at least one completed upstream analysis manifest is required")
    for path in upstream:
        if not path.is_file(): blockers.append(f"missing upstream manifest: {path}"); continue
        manifest=json.loads(path.read_text(encoding="utf-8")); inputs.append({"role":"upstream_manifest","path":str(path),"sha256":digest(path)})
        if manifest.get("status") not in {"complete", "accepted_provisional"}: blockers.append(f"upstream manifest is not complete: {path}")
        validation=manifest.get("validation", {})
        if validation and not validation.get("passed", False) and validation.get("thesis_status") != "accepted_provisional": blockers.append(f"upstream validation not passed/accepted provisional: {path}")
    if not table.is_file(): blockers.append(f"missing case-input table: {table}"); frame=pd.DataFrame()
    else:
        inputs.append({"role":"case_input_table","path":str(table),"sha256":digest(table)}); frame=pd.read_csv(table)
        missing=REQUIRED-set(frame.columns)
        if missing: blockers.append(f"case-input table lacks columns: {sorted(missing)}")
    if blockers: return None, None, None, blockers, inputs
    if frame.duplicated(["year","effect","role","resolution","entity"]).any(): blockers.append("duplicate case/year/effect/role/resolution rows")
    if set(frame.resolution.astype(str)) != RESOLUTIONS: blockers.append(f"resolution coverage must equal {sorted(RESOLUTIONS)}")
    if blockers: return None,None,None,blockers,inputs
    rows=[]
    for resolution, part in frame.groupby("resolution"):
        ranked=rank_descriptive_cases(part.to_dict("records"), case_dimensions=["entity"])
        for row in ranked: rows.append({"resolution":str(resolution), **row})
    scores=pd.DataFrame(rows)
    sensitivity=[]
    for (effect,role,entity), part in scores.groupby(["effect","role","entity"]):
        if set(part.resolution)!=RESOLUTIONS: continue
        denom=part.absolute_total.mean()
        value=(part.signed_total.max()-part.signed_total.min())/denom if denom else 0.0
        sensitivity.append({"effect":effect,"role":role,"entity":entity,"resolution_sensitivity":value})
    scores=scores.merge(pd.DataFrame(sensitivity), on=["effect","role","entity"], how="left")
    scores["meets_predeclared_criteria"]=(scores.materiality>=min_materiality)&(scores.persistence>=min_persistence)&(scores.resolution_sensitivity<=max_resolution_sensitivity)
    selected=[]
    eligible=scores[scores.meets_predeclared_criteria]
    for (_effect,_role,res), part in eligible.groupby(["effect","role","resolution"]):
        chosen=part.sort_values(["descriptive_score","entity"],ascending=[False,True]).head(top_n).copy()
        chosen["selection_rationale"]=("Passed predeclared materiality, persistence, and cross-resolution sensitivity thresholds; then ranked by descriptive score")
        selected.append(chosen)
    selected=pd.concat(selected,ignore_index=True) if selected else scores.head(0).assign(selection_rationale="")
    keys=set(map(tuple,selected[["effect","role","resolution","entity"]].itertuples(index=False,name=None)))
    plot=frame[frame.apply(lambda x:(x.effect,x.role,str(x.resolution),x.entity) in keys,axis=1)].copy()
    return scores,selected,plot,blockers,inputs

def main():
    p=argparse.ArgumentParser(); p.add_argument("--table",type=Path,required=True); p.add_argument("--upstream-manifest",type=Path,action="append",default=[]); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--min-materiality",type=float,default=.01); p.add_argument("--min-persistence",type=float,default=.6); p.add_argument("--max-resolution-sensitivity",type=float,default=.5); p.add_argument("--top-n",type=int,default=5); a=p.parse_args()
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    scores,selected,plot,blockers,inputs=build(a.table,a.upstream_manifest,min_materiality=a.min_materiality,min_persistence=a.min_persistence,max_resolution_sensitivity=a.max_resolution_sensitivity,top_n=a.top_n)
    outputs=[]
    if not blockers:
        for name,data in (("case_scores.csv",scores),("selected_cases.csv",selected),("selected_case_plot_data.csv",plot)):
            path=out/name; data.to_csv(path,index=False); outputs.append({"role":name.removesuffix(".csv"),"path":name,"sha256":digest(path),"rows":len(data)})
        lines=["# Integrated descriptive case selection","",f"Selected {len(selected)} resolution-specific cases using thresholds declared in the manifest.","","Selections are descriptive candidates, not causal findings. No ranking crosses effect or attribution-role strata."]
    else: lines=["# Integrated descriptive case selection","","Status: **BLOCKED**","","## Exact blockers",""]+[f"- {b}" for b in blockers]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest={"schema_version":1,"analysis_id":"integrated-case-selection-v1","status":"blocked" if blockers else "complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"generator":{"script":"scripts/build_integrated_case_selection.py","git_commit":"dirty-state-not-resolved-by-pipeline","command":sys.argv},"scope":{"price_basis":"CYP_(t-1) -> PYP_t","indicator":"fossil-origin CO2","units_input":"Mt CO2","units_output":"Mt CO2","effects":sorted(frame.effect.unique()) if not blockers else [],"resolutions":sorted(RESOLUTIONS),"year_previous_min":2000,"year_current_max":2014},"coverage":{"expected":None,"accepted":0 if blockers else len(frame),"missing":blockers,"excluded":[]},"inputs":inputs,"outputs":outputs,"validation":{"passed":not blockers,"gates":{"role_safe":not blockers,"complete_resolution_set":not blockers},"upstream_input_status":"accepted_provisional","thesis_status":"accepted_provisional" if not blockers else "ineligible"},"parameters":{"min_materiality":a.min_materiality,"min_persistence":a.min_persistence,"max_resolution_sensitivity":a.max_resolution_sensitivity,"top_n_per_effect_role_resolution":a.top_n,"score_weights":{"materiality":.4,"anomaly":.25,"persistence":.2,"cancellation":.15}},"limitations":["Predeclared descriptive selection does not establish causality.","Fossil-origin CO2 results remain accepted provisional.","Ranks are separate by effect, role, and resolution."]}
    (out/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); return 2 if blockers else 0
if __name__=="__main__": raise SystemExit(main())
