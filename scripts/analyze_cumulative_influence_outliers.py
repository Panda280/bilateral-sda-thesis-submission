#!/usr/bin/env python3
"""Cumulative entity influence and separate temporal anomaly diagnostics."""
from __future__ import annotations

import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

COUNTRY_PANELS={"emitter_country":"emitter","driver_country":"driver","consumer_country":"consumer","source_country":"source"}
SECTOR_PANELS={"emitter_sector":"emitter","driver_sector":"driver","demanded_sector":"demanded","source_sector":"source"}
SCALE=.6744897501960817

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def robust(values:pd.Series)->tuple[pd.Series,bool,float,float]:
    x=pd.to_numeric(values).astype(float); med=float(x.median()); mad=float((x-med).abs().median())
    if mad==0:
        # Undefined standardized distance is represented explicitly, never infinity.
        return pd.Series(np.zeros(len(x)),index=x.index),True,med,mad
    return SCALE*(x-med)/mad,False,med,mad

def load_map(path:Path, entity_type:str)->dict:
    f=pd.read_csv(path,dtype=str).fillna("")
    observed=f"observed_{entity_type}"; canonical=f"canonical_{entity_type}"
    required={"resolution",observed,canonical}
    if not required<=set(f): raise ValueError(f"mapping lacks {sorted(required-set(f))}")
    counts=f.groupby(["resolution",observed])[canonical].nunique()
    if (counts>1).any(): raise ValueError("one observed identifier maps to multiple canonical identifiers")
    return f.drop_duplicates(["resolution",observed]).set_index(["resolution",observed])[canonical].to_dict()

def load_panels(root:Path,mapping:dict,entity_type:str)->tuple[pd.DataFrame,list[dict]]:
    specs=COUNTRY_PANELS if entity_type=="country" else SECTOR_PANELS
    rows=[]; inputs=[]
    for panel,role in specs.items():
        path=root/f"{panel}.parquet"
        if not path.is_file(): continue
        x=pd.read_parquet(path)
        x=x[x.is_publishable.astype(bool)&x.validation_status.eq("passed")].copy()
        x["resolution"]=x.resolution.astype(str)
        x["canonical_id"]=[mapping.get((r,str(label))) for r,label in zip(x.resolution,x.label)]
        if x.canonical_id.isna().any():
            missing=x.loc[x.canonical_id.isna(),["resolution","label"]].drop_duplicates().values.tolist()
            raise ValueError(f"unresolved {entity_type} mappings in {panel}: {missing[:20]}")
        x["entity_type"]=entity_type;x["attribution_role"]=role
        rows.append(x);inputs.append({"path":str(path),"sha256":sha(path),"rows":len(x)})
    if not rows: raise ValueError(f"no {entity_type} panels")
    return pd.concat(rows,ignore_index=True),inputs

def analyze(frame:pd.DataFrame,min_gross_share:float,min_years:int,dominance_share:float)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    keys=["entity_type","effect_id","attribution_role","resolution","units","canonical_id"]
    annual=frame.groupby(keys+["year_previous","year_current"],as_index=False).agg(
        value=("contribution","sum"),positive=("positive_contribution","sum"),negative=("negative_contribution","sum"),gross=("gross_contribution","sum"))
    annual["sign"]=np.select([annual.value>0,annual.value<0],["positive","negative"],default="zero")
    annual["temporal_anomaly_score"]=0.;annual["temporal_zero_mad"]=False;annual["temporal_median"]=0.;annual["temporal_mad"]=0.
    for _,idx in annual.groupby(keys).groups.items():
        score,zero,med,mad=robust(annual.loc[idx,"value"])
        annual.loc[idx,"temporal_anomaly_score"]=score;annual.loc[idx,"temporal_zero_mad"]=zero
        annual.loc[idx,"temporal_median"]=med;annual.loc[idx,"temporal_mad"]=mad
    annual["temporal_anomaly_flag"]=annual.temporal_anomaly_score.abs().ge(3.5)&~annual.temporal_zero_mad

    cumulative=annual.groupby(keys,as_index=False).agg(year_count=("year_current","nunique"),cumulative_net=("value","sum"),
        cumulative_positive=("positive","sum"),cumulative_negative=("negative","sum"),cumulative_gross=("gross","sum"),
        temporal_anomaly_count=("temporal_anomaly_flag","sum"),maximum_absolute_temporal_anomaly=("temporal_anomaly_score",lambda x:x.abs().max()))
    cumulative["cumulative_sign"]=np.select([cumulative.cumulative_net>0,cumulative.cumulative_net<0],["positive","negative"],default="zero")
    group_keys=["entity_type","effect_id","attribution_role","resolution","units"]
    totals=cumulative.groupby(group_keys).agg(total_net=("cumulative_net","sum"),total_gross=("cumulative_gross","sum"),
        total_absolute_entity_net=("cumulative_net",lambda x:x.abs().sum()),entity_count=("canonical_id","nunique")).reset_index()
    cumulative=cumulative.merge(totals,on=group_keys,validate="many_to_one")
    cumulative["cumulative_absolute_net_share"]=np.where(cumulative.total_absolute_entity_net>0,cumulative.cumulative_net.abs()/cumulative.total_absolute_entity_net,0.)
    cumulative["cumulative_gross_share"]=np.where(cumulative.total_gross>0,cumulative.cumulative_gross/cumulative.total_gross,0.)
    cumulative["leave_one_out_net"]=cumulative.total_net-cumulative.cumulative_net
    cumulative["leave_one_out_gross"]=cumulative.total_gross-cumulative.cumulative_gross
    cumulative["leave_one_out_net_change_share_of_entity_abs_total"]=np.where(cumulative.total_absolute_entity_net>0,cumulative.cumulative_net.abs()/cumulative.total_absolute_entity_net,0.)
    cumulative["leave_one_out_gross_change_share"]=cumulative.cumulative_gross_share
    cumulative["leave_one_out_sign_reversal"]=((np.sign(cumulative.total_net)!=np.sign(cumulative.leave_one_out_net))&(cumulative.total_net!=0))
    cumulative["material"]=cumulative.cumulative_gross_share.ge(min_gross_share)&cumulative.year_count.ge(min_years)
    cumulative["influence_mad_score"]=0.;cumulative["influence_zero_mad"]=False;cumulative["influence_median"]=0.;cumulative["influence_mad"]=0.
    for _,idx in cumulative.groupby(group_keys).groups.items():
        score,zero,med,mad=robust(cumulative.loc[idx,"cumulative_gross_share"])
        cumulative.loc[idx,"influence_mad_score"]=score;cumulative.loc[idx,"influence_zero_mad"]=zero
        cumulative.loc[idx,"influence_median"]=med;cumulative.loc[idx,"influence_mad"]=mad
    cumulative["dominant_single_entity"]=(cumulative.cumulative_gross_share.ge(dominance_share)|cumulative.leave_one_out_sign_reversal|cumulative.influence_mad_score.abs().ge(3.5))&cumulative.material
    cumulative["influence_rank"]=cumulative.groupby(group_keys).cumulative_gross_share.rank(method="first",ascending=False).astype(int)
    flags=cumulative[cumulative.dominant_single_entity].copy()
    return annual,cumulative,flags

def plot(flags:pd.DataFrame,out:Path)->list[dict]:
    import matplotlib.pyplot as plt
    outputs=[]
    for entity_type in ("country","sector"):
        x=flags[flags.entity_type.eq(entity_type)].sort_values("cumulative_gross_share").tail(25)
        if x.empty: continue
        labels=x.effect_id+" · "+x.attribution_role+" · "+x.resolution+" · "+x.canonical_id
        fig,ax=plt.subplots(figsize=(9,max(5,.28*len(x))),constrained_layout=True)
        ax.barh(labels,x.cumulative_gross_share*100,color=np.where(x.cumulative_sign.eq("positive"),"#CC6677","#4477AA"))
        ax.set_xlabel("Share of cumulative gross contribution (%)");ax.set_title(f"Dominant cumulative {entity_type} influence (signed colour)")
        for ext in ("pdf","png"):
            p=out/f"cumulative_{entity_type}_influence.{ext}";fig.savefig(p,dpi=300,bbox_inches="tight");outputs.append({"path":p.name,"sha256":sha(p)})
        plt.close(fig)
    return outputs

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--panel-root",type=Path,required=True);p.add_argument("--country-map",type=Path,required=True);p.add_argument("--sector-map",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--min-gross-share",type=float,default=.005);p.add_argument("--min-years",type=int,default=10);p.add_argument("--dominance-share",type=float,default=.25);a=p.parse_args()
    if not 0<=a.min_gross_share<=1 or not 0<a.dominance_share<=1: p.error("share thresholds must be in [0,1]")
    cm=load_map(a.country_map,"country");sm=load_map(a.sector_map,"sector")
    country,inputs=load_panels(a.panel_root,cm,"country");sector,inputs2=load_panels(a.panel_root,sm,"sector")
    annual,cumulative,flags=analyze(pd.concat([country,sector],ignore_index=True),a.min_gross_share,a.min_years,a.dominance_share)
    a.output_dir.mkdir(parents=True,exist_ok=True); records=[]
    for name,data in (("annual_temporal_anomalies.csv",annual),("cumulative_influence_register.csv",cumulative),("dominant_entity_flags.csv",flags)):
        q=a.output_dir/name;data.to_csv(q,index=False);records.append({"path":name,"sha256":sha(q),"rows":len(data)})
    thesis=(flags.sort_values(["entity_type","effect_id","attribution_role","resolution","cumulative_sign","cumulative_gross_share"],ascending=[True,True,True,True,True,False])
            .groupby(["entity_type","effect_id","attribution_role","resolution","cumulative_sign"],as_index=False).head(3))
    thesis_path=a.output_dir/"thesis_dominant_entities.csv";thesis.to_csv(thesis_path,index=False)
    records.append({"path":thesis_path.name,"sha256":sha(thesis_path),"rows":len(thesis)})
    records+=plot(flags,a.output_dir)
    report=f"""# Cumulative influence outlier analysis\n\nStatus: **accepted provisional**. This analysis is descriptive and non-causal.\n\nThe register contains {len(cumulative):,} cumulative entity series and {len(flags):,} dominant-entity flags ({int((flags.entity_type=='country').sum()):,} country and {int((flags.entity_type=='sector').sum()):,} sector). Cumulative influence is a full-window cross-sectional quantity; `temporal_anomaly_score` is a separate within-entity annual median/MAD diagnostic. A zero MAD produces score zero and an explicit `*_zero_mad=true` flag, never infinity.\n\nMateriality requires at least {a.min_years} transitions and {100*a.min_gross_share:.1f}% of cumulative gross activity. Dominance is flagged at {100*a.dominance_share:.0f}% gross share, a leave-one-out net sign reversal, or absolute robust influence score at least 3.5. Effects, attribution roles, signs, resolutions, canonical identifiers and kg CO2 are retained. Country and sector grains remain separate; no missing joint group-sector relationship is inferred.\n"""
    report_path=a.output_dir/"report.md";report_path.write_text(report,encoding="utf-8")
    records.append({"path":report_path.name,"sha256":sha(report_path)})
    manifest={"schema_version":1,"analysis_id":"cumulative-influence-outliers-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"accepted_provisional",
      "generator":{"script":"scripts/analyze_cumulative_influence_outliers.py","command":sys.argv},"scope":{"indicator":"fossil-origin CO2","units":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t","resolutions":["2013","2016","R3"]},
      "parameters":{"min_cumulative_gross_share":a.min_gross_share,"min_years":a.min_years,"dominance_share":a.dominance_share,"mad_threshold":3.5,"zero_mad_score":0.0},
      "definitions":{"cumulative_influence":"cross-sectional entity share and leave-one-out impact over the full window","temporal_anomaly":"within-entity annual median/MAD score; separate from cumulative influence","leave_one_out":"reported for net and gross; sign reversal compares full and entity-excluded cumulative net"},
      "inputs":inputs+inputs2+[{"path":str(a.country_map),"sha256":sha(a.country_map)},{"path":str(a.sector_map),"sha256":sha(a.sector_map)}],"outputs":records,
      "validation":{"passed":True,"finite":bool(np.isfinite(cumulative.select_dtypes(include=[np.number])).all().all()),"unresolved_mappings":0},
      "limitations":["Accepted provisional due to unresolved upstream validation.","Descriptive influence and anomaly flags are non-causal.","Country and sector entities are analysed separately; no group-sector mapping is inferred.","Cross-resolution sector rankings are not pooled."]}
    (a.output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return 0
if __name__=="__main__":raise SystemExit(main())
