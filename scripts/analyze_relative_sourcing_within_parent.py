#!/usr/bin/env python3
"""Relative sourcing contributions within reconstructed parent components."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

SPECS={"leontief":{"sourcing":"D_C","companions":["D_T_coef","D_w_coef"],"roles":{"emitter":"emitter_country.parquet","driver":"driver_country.parquet"}},
       "final_demand":{"sourcing":"D_YF","companions":["D_YB","D_y"],"roles":{"emitter":"emitter_country.parquet","consumer":"consumer_country.parquet"}}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sign(x,eps):return np.select([x>eps,x<-eps],["positive","negative"],default="near_zero")

def derive(x,eps,materiality):
    x=x.copy();x["reconstructed_parent_net"]=x.sourcing_net+x.companion_net;x["parent_gross_activity"]=x.sourcing_gross+x.companion_gross
    x["component_absolute_net"]=x.sourcing_net.abs()+x.companion_net.abs()
    x["parent_gross_denominator_status"]=np.where(x.parent_gross_activity>eps,"ok","near_zero")
    x["component_net_denominator_status"]=np.where(x.component_absolute_net>eps,"ok","near_zero")
    x["sourcing_share_of_parent_gross_activity"]=np.where(x.parent_gross_activity>eps,x.sourcing_gross/x.parent_gross_activity,0.)
    x["absolute_sourcing_share_of_component_absolute_net"]=np.where(x.component_absolute_net>eps,x.sourcing_net.abs()/x.component_absolute_net,0.)
    x["offset_ratio"]=np.where(x.component_absolute_net>eps,1-x.reconstructed_parent_net.abs()/x.component_absolute_net,0.)
    x["sourcing_sign"]=sign(x.sourcing_net,eps);x["companion_sign"]=sign(x.companion_net,eps);x["parent_sign_without_sourcing"]=x.companion_sign;x["parent_sign_with_sourcing"]=sign(x.reconstructed_parent_net,eps)
    x["signed_alignment"]=np.where((x.sourcing_sign=="near_zero")|(x.companion_sign=="near_zero"),"indeterminate",np.where(x.sourcing_sign==x.companion_sign,"aligned","opposed"))
    x["interaction_class"]=np.select([x.signed_alignment=="aligned",x.signed_alignment=="opposed",(x.sourcing_sign=="near_zero")&(x.companion_sign!="near_zero"),(x.companion_sign=="near_zero")&(x.sourcing_sign!="near_zero")],["reinforce","counteract","companion_only","sourcing_only"],default="both_near_zero")
    x["sign_switch_flag"]=(x.parent_sign_without_sourcing!=x.parent_sign_with_sourcing)&(x.parent_sign_without_sourcing!="near_zero")
    x["material"]=x.parent_gross_activity.gt(eps)&x.sourcing_share_of_parent_gross_activity.ge(materiality)
    return x

def mappings(country_path,group_path):
    c=pd.read_csv(country_path,dtype=str).fillna("");g=pd.read_csv(group_path,dtype=str).fillna("")
    if (c.groupby(["resolution","observed_country"]).canonical_country.nunique()>1).any():raise ValueError("ambiguous country map")
    cm=c.drop_duplicates(["resolution","observed_country"]).set_index(["resolution","observed_country"]).canonical_country.to_dict();gm=g.set_index("canonical_country").analysis_group.to_dict();return cm,gm

def load(root,cm,gm):
    rows=[];inputs=[]
    for parent,spec in SPECS.items():
        effects=[spec["sourcing"],*spec["companions"]]
        for role,name in spec["roles"].items():
            path=root/name;x=pd.read_parquet(path);x=x[x.effect_id.isin(effects)&x.is_publishable.astype(bool)&x.validation_status.eq("passed")].copy();x["resolution"]=x.resolution.astype(str)
            x["canonical_country"]=[cm.get((r,str(v))) for r,v in zip(x.resolution,x.label)]
            if x.canonical_country.isna().any():raise ValueError("unresolved country mapping")
            x["analysis_group"]=x.canonical_country.map(gm)
            if x.analysis_group.isna().any():raise ValueError("unresolved analytical group")
            x["parent_component"]=parent;x["attribution_role"]=role;x["component_kind"]=np.where(x.effect_id.eq(spec["sourcing"]),"sourcing","companion")
            rows.append(x);inputs.append({"path":str(path),"sha256":sha(path),"parent_component":parent,"role":role})
    return pd.concat(rows,ignore_index=True),inputs

def aggregate(raw,entity_col,level,eps,materiality):
    keys=["year_previous","year_current","parent_component","attribution_role","resolution","units",entity_col]
    a=raw.groupby(keys+["component_kind"],as_index=False).agg(net=("contribution","sum"),gross=("gross_contribution","sum"))
    net=a.pivot(index=keys,columns="component_kind",values="net").fillna(0).rename(columns={"sourcing":"sourcing_net","companion":"companion_net"})
    gross=a.pivot(index=keys,columns="component_kind",values="gross").fillna(0).rename(columns={"sourcing":"sourcing_gross","companion":"companion_gross"})
    annual=net.join(gross).reset_index();annual["entity_level"]=level;annual=derive(annual,eps,materiality)
    ckeys=["parent_component","attribution_role","resolution","units",entity_col,"entity_level"]
    cumulative=annual.groupby(ckeys,as_index=False).agg(year_count=("year_current","nunique"),sourcing_net=("sourcing_net","sum"),sourcing_gross=("sourcing_gross","sum"),companion_net=("companion_net","sum"),companion_gross=("companion_gross","sum"))
    cumulative=derive(cumulative,eps,materiality)
    return annual,cumulative

def plot(cumulative,out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    records=[]
    for level in ["analytical_group","canonical_country"]:
        entity="analysis_group" if level=="analytical_group" else "canonical_country";x=cumulative[cumulative.entity_level.eq(level)&cumulative.material].nlargest(30,"sourcing_share_of_parent_gross_activity").sort_values("sourcing_share_of_parent_gross_activity")
        labels=x.parent_component+" · "+x.attribution_role+" · "+x.resolution+" · "+x[entity]
        fig,ax=plt.subplots(figsize=(10,max(6,.28*len(x))),constrained_layout=True);ax.barh(labels,x.sourcing_share_of_parent_gross_activity*100,color=np.where(x.interaction_class.eq("reinforce"),"#CC6677","#4477AA"));ax.set_xlabel("Sourcing share of reconstructed parent gross activity (%)");ax.set_title(f"Relative sourcing within parent: {level.replace('_',' ')}");ax.legend(handles=[Patch(color="#CC6677",label="reinforce"),Patch(color="#4477AA",label="counteract / other")],frameon=False,loc="lower right")
        for ext in ["pdf","png"]:
            p=out/f"relative_sourcing_{level}.{ext}";fig.savefig(p,dpi=300,bbox_inches="tight");records.append({"path":p.name,"sha256":sha(p)})
        plt.close(fig)
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument("--panel-root",type=Path,required=True);p.add_argument("--country-map",type=Path,required=True);p.add_argument("--group-registry",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--epsilon-kg",type=float,default=1.);p.add_argument("--materiality-share",type=float,default=.05);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    cm,gm=mappings(a.country_map,a.group_registry);raw,inputs=load(a.panel_root,cm,gm)
    ac,cc=aggregate(raw,"canonical_country","canonical_country",a.epsilon_kg,a.materiality_share)
    groupraw=raw.copy();ag,cg=aggregate(groupraw,"analysis_group","analytical_group",a.epsilon_kg,a.materiality_share)
    annual=pd.concat([ac,ag],ignore_index=True);cumulative=pd.concat([cc,cg],ignore_index=True);flags=cumulative[cumulative.material&(cumulative.sign_switch_flag|cumulative.interaction_class.eq("counteract"))].copy()
    outputs=[]
    for name,x in (("annual_relative_sourcing.csv",annual),("cumulative_relative_sourcing.csv",cumulative),("material_counteracting_and_sign_switch_cases.csv",flags)):
        q=a.output_dir/name;x.to_csv(q,index=False);outputs.append({"path":name,"sha256":sha(q),"rows":len(x)})
    table=cumulative[cumulative.material].sort_values("sourcing_share_of_parent_gross_activity",ascending=False).groupby(["entity_level","parent_component","attribution_role","resolution"],as_index=False).head(5);q=a.output_dir/"thesis_relative_sourcing_table.csv";table.to_csv(q,index=False);outputs.append({"path":q.name,"sha256":sha(q),"rows":len(table)})
    outputs+=plot(cumulative,a.output_dir)
    report=f"""# Relative sourcing within reconstructed parent components\n\nStatus: **accepted provisional**, descriptive and non-causal. Units remain kg CO2 and the price basis is `CYP_(t-1) -> PYP_t`.\n\n`D_C` is compared with `D_T_coef + D_w_coef` only for emitter and driver roles. `D_YF` is compared with `D_YB + D_y` only for emitter and consumer roles. Roles and resolutions are never pooled. Source-country attribution is unavailable for all companion subeffects and is therefore not inferred.\n\nParent quantities are reconstructed from the registered subeffects; they are not independent parent-effect outputs. Sourcing gross share uses `sourcing_gross / (sourcing_gross + companion_gross)`. Absolute-net share uses `abs(sourcing_net) / (abs(sourcing_net) + abs(companion_net))`. Offset ratio is `1 - abs(parent_net) / (abs(sourcing_net) + abs(companion_net))`. Denominators at or below {a.epsilon_kg:g} kg CO2 are explicitly marked `near_zero` and receive a finite ratio of zero. Material cases require at least {100*a.materiality_share:g}% sourcing share of parent gross activity.\n\nThe outputs contain {len(annual):,} annual and {len(cumulative):,} cumulative comparisons; {int(cumulative.sign_switch_flag.sum()):,} cumulative cases switch parent sign when sourcing is included.\n""";rp=a.output_dir/"report.md";rp.write_text(report,encoding="utf-8");outputs.append({"path":rp.name,"sha256":sha(rp)})
    m={"schema_version":1,"analysis_id":"relative-sourcing-within-parent-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"accepted_provisional","generator":{"script":"scripts/analyze_relative_sourcing_within_parent.py","command":sys.argv},"scope":{"units":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t","indicator":"fossil-origin CO2","resolutions":["2013","2016","R3"],"roles":{"leontief":["emitter","driver"],"final_demand":["emitter","consumer"]}},"parameters":{"near_zero_epsilon_kg":a.epsilon_kg,"materiality_sourcing_gross_share":a.materiality_share},"definitions":{"leontief_parent":"D_C + D_T_coef + D_w_coef","final_demand_parent":"D_YF + D_YB + D_y","parent_status":"reconstructed from registered subeffects"},"inputs":inputs+[{"path":str(a.country_map),"sha256":sha(a.country_map)},{"path":str(a.group_registry),"sha256":sha(a.group_registry)}],"outputs":outputs,"validation":{"passed":True,"finite":bool(np.isfinite(cumulative.select_dtypes('number')).all().all()),"unresolved_mappings":0},"limitations":["Accepted provisional due to upstream validation.","Descriptive and non-causal.","Parent is reconstructed, not independently observed.","Roles and grids remain separate."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8");return 0
if __name__=="__main__":raise SystemExit(main())
