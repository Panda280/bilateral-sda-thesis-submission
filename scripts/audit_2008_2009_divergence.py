#!/usr/bin/env python3
"""Reconcile the 2008->2009 hierarchy divergence across registered grids."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY={"raw_total":["delta_g"],"emission_intensity":["D_v"],"leontief_parent":["leontief_production_structure"],"final_demand_parent":["final_demand"]}
LEAVES=["D_v","D_C","D_T_coef","D_w_coef","D_YF","D_YB","D_y"]
RES=["2013","2016","R3"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def episode(frame,previous=2008,current=2009):
    x=frame[(frame.year_previous.between(previous-1,previous+1))&(frame.year_current==frame.year_previous+1)].copy();x["resolution"]=x.resolution.astype(str)
    selected=set(sum(PRIMARY.values(),[])+LEAVES);x=x[x.effect_id.isin(selected)]
    x["net_mt"]=x.contribution/1e9;x["positive_mt"]=x.positive_contribution/1e9;x["negative_mt"]=x.negative_contribution/1e9;x["gross_mt"]=x.gross_contribution/1e9;x["cancellation_ratio"]=np.where(x.gross_mt>0,1-x.net_mt.abs()/x.gross_mt,0.)
    return x[["year_previous","year_current","resolution","units","effect_id","net_mt","positive_mt","negative_mt","gross_mt","cancellation_ratio"]].sort_values(["year_previous","effect_id","resolution"])

def reconciliation(ep,previous=2008,current=2009):
    x=ep[(ep.year_previous==previous)&(ep.year_current==current)].copy();p=x.pivot(index="resolution",columns="effect_id",values="net_mt")
    rows=[]
    for r,row in p.iterrows():
        leontief=row.D_C+row.D_T_coef+row.D_w_coef;final=row.D_YF+row.D_YB+row.D_y;total=row.D_v+leontief+final
        rows.append({"resolution":r,"units":"Mt CO2","raw_total":row.delta_g,"emission_intensity":row.D_v,"leontief_parent_registered":row.leontief_production_structure,"leontief_parent_reconstructed":leontief,"D_C_sourcing":row.D_C,"D_T_companion":row.D_T_coef,"D_w_companion":row.D_w_coef,"final_demand_parent_registered":row.final_demand,"final_demand_parent_reconstructed":final,"D_YF_sourcing":row.D_YF,"D_YB_composition":row.D_YB,"D_y_volume":row.D_y,"total_reconstructed":total,"leontief_residual":row.leontief_production_structure-leontief,"final_demand_residual":row.final_demand-final,"total_residual":row.delta_g-total})
    return pd.DataFrame(rows)

def pairwise(ep,previous=2008,current=2009):
    x=ep[(ep.year_previous==previous)&(ep.year_current==current)&ep.effect_id.isin(["delta_g",*LEAVES,"leontief_production_structure","final_demand"])]
    rows=[]
    for effect,g in x.groupby("effect_id"):
        z=g.set_index("resolution")
        for left,right in combinations(RES,2):
            a,b=z.loc[left],z.loc[right];den=(a.gross_mt+b.gross_mt)/2
            rows.append({"effect_id":effect,"resolution_left":left,"resolution_right":right,"units":"Mt CO2","net_left":a.net_mt,"net_right":b.net_mt,"signed_difference_right_minus_left":b.net_mt-a.net_mt,"absolute_difference":abs(b.net_mt-a.net_mt),"normalized_difference_by_mean_gross":abs(b.net_mt-a.net_mt)/den if den else 0.,"sign_agreement":np.sign(a.net_mt)==np.sign(b.net_mt),"comparison_class":"target-grid-labelled composite difference; database and pure aggregation effects not separately identified"})
    return pd.DataFrame(rows)

def availability():
    return pd.DataFrame([
      {"question":"Observed 2008->2009 total and effect differences across target labels","available":True,"identification":"descriptive","reason":"registered global hierarchy contains matched transition, effect, resolution, net and gross"},
      {"question":"Role-specific D_C versus D_YF allocation","available":True,"identification":"descriptive only","reason":"effect identities are available, but D_C and D_YF answer different sourcing mechanisms and are not summed as one channel"},
      {"question":"Pure resolution/aggregation effect","available":False,"identification":"not identified","reason":"no controlled aggregation treatment holding database construction and all inputs fixed is linked to these panels"},
      {"question":"Pure WIOD-versus-EXIOBASE database effect","available":False,"identification":"not identified","reason":"panel resolution labels do not independently encode a controlled database treatment; local manifest links hashes but not a factorial source comparison"},
      {"question":"Economic or crisis cause of the episode","available":False,"identification":"not causal","reason":"SDA contribution comparison has no causal identification design"},
    ])

def plot(rec,out):
    import matplotlib.pyplot as plt
    effects=["raw_total","emission_intensity","leontief_parent_registered","final_demand_parent_registered","D_C_sourcing","D_T_companion","D_w_companion","D_YF_sourcing","D_YB_composition","D_y_volume"]
    long=rec.melt(id_vars=["resolution"],value_vars=effects,var_name="component",value_name="net_mt");fig,axes=plt.subplots(2,1,figsize=(11,9),constrained_layout=True)
    for ax,subset,title in [(axes[0],effects[:4],"Total and primary effects"),(axes[1],effects[4:],"Sourcing and companion subeffects")]:
        q=long[long.component.isin(subset)];pivot=q.pivot(index="component",columns="resolution",values="net_mt").loc[subset];pivot.plot.bar(ax=ax,color=["#4477AA","#CC6677","#228833"]);ax.axhline(0,color="black",lw=.7);ax.set_ylabel("Net contribution (Mt CO$_2$)");ax.set_title(title);ax.tick_params(axis="x",rotation=25);ax.legend(title="Target-grid label",frameon=False)
    p1=out/"divergence_2008_2009_hierarchy.pdf";p2=out/"divergence_2008_2009_hierarchy.png";fig.savefig(p1,bbox_inches="tight");fig.savefig(p2,dpi=300,bbox_inches="tight");plt.close(fig);return [{"path":p1.name,"sha256":sha(p1)},{"path":p2.name,"sha256":sha(p2)}]

def main():
    p=argparse.ArgumentParser();p.add_argument("--global-hierarchy",type=Path,required=True);p.add_argument("--annual-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(a.global_hierarchy);ep=episode(raw);rec=reconciliation(ep);pairs=pairwise(ep);avail=availability()
    # Local annual manifests establish target labels and validation, but not a controlled database factorial design.
    manifests=[]
    for r in RES:
        q=a.annual_root/f"2008_2009_{r}"/"manifest.v2.json"
        if not q.is_file():raise ValueError(f"missing annual manifest {q}")
        m=json.loads(q.read_text());manifests.append({"resolution":r,"run_id":m["run_id"],"validation_status":m["validation_status"],"units":m["units"],"manifest_path":str(q),"manifest_sha256":sha(q),"source_file_count":len(m["source_files"]),"flow_database_locally_identified":False,"comparison_grid_locally_identified":True})
    manifest_table=pd.DataFrame(manifests)
    if not np.allclose(rec[["leontief_residual","final_demand_residual","total_residual"]],0,rtol=0,atol=.01):raise ValueError("hierarchy reconstruction failed")
    outputs=[]
    for name,x in (("episode_2007_2010_effects.csv",ep),("episode_2008_2009_reconciliation.csv",rec),("pairwise_grid_differences.csv",pairs),("identification_availability.csv",avail),("run_manifest_audit.csv",manifest_table)):
        q=a.output_dir/name;x.to_csv(q,index=False);outputs.append({"path":name,"sha256":sha(q),"rows":len(x)})
    outputs+=plot(rec,a.output_dir)
    r3=rec.set_index("resolution").loc["R3"];w13=rec.set_index("resolution").loc["2013"];w16=rec.set_index("resolution").loc["2016"]
    report=f"""# 2008--2009 hierarchy and target-grid divergence audit\n\nStatus: **accepted provisional** and descriptive. Values are fossil-origin CO2 in Mt (`kg / 1e9`) under `CYP_(t-1) -> PYP_t`.\n\nThe raw total is comparatively similar across the three target labels ({w13.raw_total:.1f}, {w16.raw_total:.1f}, and {r3.raw_total:.1f} Mt for 2013, 2016, and R3), but the internal allocation differs. R3 has a much more negative reconstructed Leontief contribution ({r3.leontief_parent_reconstructed:.1f} Mt) than 2013/2016 ({w13.leontief_parent_reconstructed:.1f}/{w16.leontief_parent_reconstructed:.1f} Mt), principally alongside a more negative `D_T_coef`, while its `D_w_coef` is positive. Conversely, R3 final demand is less negative ({r3.final_demand_parent_reconstructed:.1f} Mt) than 2013/2016 ({w13.final_demand_parent_reconstructed:.1f}/{w16.final_demand_parent_reconstructed:.1f} Mt), principally because `D_YB` is less negative. `D_C` and `D_YF` are positive in all three grids for this transition, but `D_C` is larger on R3 and must not be conflated with final-demand sourcing. The opposing primary-effect differences largely cancel in the raw total.\n\nThese are differences among outputs labelled by target comparison grid. The registered local panels do not supply a controlled factorial experiment that independently varies database and aggregation while holding all other inputs fixed. Therefore neither a pure resolution effect nor a pure WIOD-versus-EXIOBASE database effect is identified here. No economic cause is inferred.\n""";rp=a.output_dir/"audit_report.md";rp.write_text(report,encoding="utf-8");outputs.append({"path":rp.name,"sha256":sha(rp)})
    m={"schema_version":1,"analysis_id":"divergence-2008-2009-audit-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"accepted_provisional","generator":{"script":"scripts/audit_2008_2009_divergence.py","command":sys.argv},"scope":{"transition":"2008->2009","context_transitions":["2007->2008","2009->2010"],"resolutions":RES,"units_output":"Mt CO2","units_source":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t"},"inputs":[{"path":str(a.global_hierarchy),"sha256":sha(a.global_hierarchy)},*[{"path":x["manifest_path"],"sha256":x["manifest_sha256"]} for x in manifests]],"outputs":outputs,"validation":{"passed":True,"hierarchy_reconstruction_tolerance_mt":.01,"manifest_validation_statuses":manifest_table.validation_status.unique().tolist()},"identification":{"descriptive_target_grid_differences":True,"pure_resolution_effect":False,"pure_database_effect":False,"causal_episode_explanation":False},"limitations":["Accepted provisional due to upstream validation.","Target-grid labels are not controlled database treatments.","Effect reallocations can offset in the total."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(m,indent=2)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
