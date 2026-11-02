#!/usr/bin/env python3
"""Create sign-component case inputs from accepted cross-year country/sector panels."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

PANELS={
 "emitter_country.parquet":("emitter","country"), "driver_country.parquet":("structural_driver","country"),
 "consumer_country.parquet":("consumer","country"), "source_country.parquet":("source","country"),
 "emitter_sector.parquet":("emitter","sector"), "driver_sector.parquet":("structural_driver","sector"),
 "demanded_sector.parquet":("demanded_sector","sector"), "source_sector.parquet":("source","sector"),
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def robust_scores(values):
    x=np.asarray(values,float); med=np.median(x); mad=np.median(np.abs(x-med)); zero=mad==0
    return (np.zeros(len(x)) if zero else np.abs(x-med)/(1.4826*mad)), np.repeat(zero,len(x))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--panel-root",type=Path,required=True); p.add_argument("--panel-manifest",type=Path,required=True)
    p.add_argument("--country-registry",type=Path,required=True); p.add_argument("--group-registry",type=Path,required=True); p.add_argument("--sector-registry",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    country=pd.read_csv(a.country_registry,dtype=str); groups=pd.read_csv(a.group_registry,dtype=str); sector=pd.read_csv(a.sector_registry,dtype=str)
    presentation=dict(zip(groups.canonical_country,groups.presentation_code))
    cmap={(r.resolution,r.observed_country):presentation.get(r.canonical_country,r.canonical_country) for r in country.itertuples()}
    smap={(r.resolution,r.observed_sector):r.canonical_sector for r in sector.itertuples()}
    rows=[]; inputs=[]
    for name,(role,kind) in PANELS.items():
        path=a.panel_root/name; f=pd.read_parquet(path); inputs.append({"path":str(path),"sha256":sha(path)})
        f=f[(f.is_publishable)&f.validation_status.eq("passed")].copy()
        for sign,col in (("positive","positive_contribution"),("negative","negative_contribution")):
            q=f.copy(); q["selection_sign"]=sign; q["value"]=q[col]
            if kind=="country":
                q["canonical_entity"]=[cmap.get((str(res),str(label))) for res,label in zip(q.resolution,q.label)]
                q["comparison_eligible"]=q.canonical_entity.notna(); q["mapping_status"]=np.where(q.comparison_eligible,"reviewed_identity","unmapped")
            else:
                q["canonical_entity"]=[smap.get((str(res),str(label))) for res,label in zip(q.resolution,q.label)]
                q["comparison_eligible"]=False; q["mapping_status"]=np.where(q.canonical_entity.notna(),"reviewed_native","unmapped")
            q["entity_type"]=kind; q["attribution_role"]=role
            q["cancellation_ratio"]=np.where(q.gross_contribution.ne(0),1-q.contribution.abs()/q.gross_contribution,0)
            q["upstream_result_id"]=["PANEL-"+hashlib.sha256(f"{run}|{name}|{effect}|{label}".encode()).hexdigest()[:16].upper() for run,effect,label in zip(q.run_id,q.effect_id,q.label)]
            rows.append(q[["year_previous","year_current","effect_id","attribution_role","resolution","selection_sign","entity_type","canonical_entity","mapping_status","comparison_eligible","value","units","cancellation_ratio","upstream_result_id"]].rename(columns={"effect_id":"effect"}))
    out=pd.concat(rows,ignore_index=True)
    semantic=["year_previous","year_current","effect","attribution_role","resolution","selection_sign","entity_type","canonical_entity","mapping_status","comparison_eligible","units"]
    out=(out.groupby(semantic,dropna=False,as_index=False)
         .agg(value=("value","sum"),cancellation_ratio=("cancellation_ratio","mean"),
              upstream_result_id=("upstream_result_id",lambda x:" | ".join(sorted(set(map(str,x)))))))
    out["anomaly_score"]=0.0; out["zero_mad"]=False
    keys=["effect","attribution_role","resolution","selection_sign","entity_type","canonical_entity"]
    for _,idx in out[out.canonical_entity.notna()].groupby(keys).groups.items():
        scores,zero=robust_scores(out.loc[idx,"value"]); out.loc[idx,"anomaly_score"]=scores; out.loc[idx,"zero_mad"]=zero
    a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False,float_format="%.12g")
    manifest={"schema_version":1,"status":"accepted_provisional","analysis_id":"case-input-panels-v1","inputs":[{"path":str(a.panel_manifest),"sha256":sha(a.panel_manifest)},*inputs,{"path":str(a.country_registry),"sha256":sha(a.country_registry)},{"path":str(a.group_registry),"sha256":sha(a.group_registry)},{"path":str(a.sector_registry),"sha256":sha(a.sector_registry)}],"outputs":[{"path":str(a.output),"sha256":sha(a.output),"rows":len(out)}],"validation":{"passed":False,"thesis_status":"accepted_provisional","gates":{"units":bool(out.units.nunique()==1),"finite":bool(np.isfinite(out[["value","anomaly_score","cancellation_ratio"]]).all().all()),"country_cross_resolution_only":True,"sectors_native_only":True}},"limitations":["Sector cases are native-resolution only until reviewed common-sector mappings are complete.","Fossil-origin CO2 results remain accepted provisional."]}
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(a.output)
if __name__=="__main__": main()
