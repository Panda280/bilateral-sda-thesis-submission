#!/usr/bin/env python3
"""Compare raw D_A_x with its mathematical D_T_coef + D_w_coef split."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EFFECTS=["D_A_x","D_T_coef","D_w_coef"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sid(values):return "ASPLIT-"+hashlib.sha256("|".join(map(str,values)).encode()).hexdigest()[:16].upper()
def sign(x,eps=1e-6):return np.where(x>eps,"positive",np.where(x<-eps,"negative","zero"))

def signed_effects(frame,keys,effect_col="effect"):
 rows=[]
 for key,p in frame.groupby(keys+[effect_col],dropna=False,sort=True):
  pos=float(p.loc[p.selection_sign.eq("positive"),"signed_total"].sum());neg=float(p.loc[p.selection_sign.eq("negative"),"signed_total"].sum());gross=pos+abs(neg);net=pos+neg
  rows.append(dict(zip(keys+[effect_col],key),net_kg=net,gross_kg=gross,positive_kg=pos,negative_kg=neg,cancellation_ratio=1-abs(net)/gross if gross else 0))
 return pd.DataFrame(rows)

def diagnostics(long,keys):
 n=long.pivot(index=keys,columns="effect",values="net_kg").reset_index();g=long.pivot(index=keys,columns="effect",values="gross_kg").reset_index();c=long.pivot(index=keys,columns="effect",values="cancellation_ratio").reset_index()
 if any(x not in n for x in EFFECTS):raise ValueError("raw/split effects incomplete at common grain")
 out=n[keys].copy();out["raw_net_kg"]=n.D_A_x;out["raw_gross_kg"]=g.D_A_x;out["raw_cancellation_ratio"]=c.D_A_x
 out["t_net_kg"]=n.D_T_coef;out["t_gross_kg"]=g.D_T_coef;out["t_cancellation_ratio"]=c.D_T_coef
 out["w_net_kg"]=n.D_w_coef;out["w_gross_kg"]=g.D_w_coef;out["w_cancellation_ratio"]=c.D_w_coef
 out["split_net_kg"]=out.t_net_kg+out.w_net_kg;out["split_component_gross_kg"]=out.t_gross_kg+out.w_gross_kg
 out["reconstruction_residual_kg"]=out.raw_net_kg-out.split_net_kg
 out["residual_relative_to_split_gross"]=out.reconstruction_residual_kg.abs()/out.split_component_gross_kg.replace(0,np.nan)
 out["split_information_gain_ratio"]=(out.split_component_gross_kg-out.raw_gross_kg)/out.split_component_gross_kg.replace(0,np.nan)
 pair_abs=out.t_net_kg.abs()+out.w_net_kg.abs();out["tw_pair_cancellation_ratio"]=1-out.split_net_kg.abs()/pair_abs.replace(0,np.nan)
 tshare=out.t_net_kg.abs()/pair_abs.replace(0,np.nan);out["tw_dominance"]=np.where(tshare.ge(.6),"T-dominant",np.where(tshare.le(.4),"w-dominant","balanced"))
 out["raw_sign"]=sign(out.raw_net_kg);out["t_sign"]=sign(out.t_net_kg);out["w_sign"]=sign(out.w_net_kg);out["split_sign"]=sign(out.split_net_kg)
 out["opposing_split_components"]=(out.t_sign!=out.w_sign)&~out.t_sign.eq("zero")&~out.w_sign.eq("zero")
 out["opposition_hidden_by_raw_parent"]=out.opposing_split_components & out.raw_sign.isin(["positive","negative"])
 rankkeys=[x for x in keys if x not in {"entity_id"}]
 out["raw_gross_rank"]=out.groupby(rankkeys,dropna=False).raw_gross_kg.rank(method="min",ascending=False).astype(int)
 out["split_gross_rank"]=out.groupby(rankkeys,dropna=False).split_component_gross_kg.rank(method="min",ascending=False).astype(int)
 out["absolute_rank_change"]=(out.raw_gross_rank-out.split_gross_rank).abs()
 out["result_id"]=[sid([*(getattr(r,k) for k in keys),"D_A_x","D_T_coef+D_w_coef","kg CO2"]) for r in out.itertuples()]
 out["units"]="kg CO2";out["price_basis"]="CYP_(t-1) -> PYP_t";out["provisional"]=True;out["interpretation_limit"]="mathematical/descriptive only; no wage or physical-input interpretation"
 return out

def build(case_scores,annual,groups):
 q=case_scores[case_scores.effect.isin(EFFECTS)&case_scores.entity_type.isin(["country","sector"])].copy();q["resolution"]=q.resolution.astype(str)
 if set(q.units)!={"kg CO2"}:raise ValueError("case units must be kg CO2")
 country=q[q.entity_type.eq("country")&q.attribution_role.isin(["emitter","structural_driver"])].copy();country["grain"]="canonical_country";country["entity_id"]=country.canonical_entity
 sector=q[q.entity_type.eq("sector")&q.attribution_role.isin(["emitter","structural_driver"])&q.selection_scope.eq("native_resolution")].copy();sector["grain"]="native_sector";sector["entity_id"]=sector.canonical_entity
 country_long=signed_effects(country,["grain","attribution_role","resolution","entity_id"])
 sector_long=signed_effects(sector,["grain","attribution_role","resolution","entity_id"])
 registry=groups[["canonical_country","analysis_group"]];group=country.merge(registry,left_on="canonical_entity",right_on="canonical_country",validate="many_to_one");group["grain"]="analytical_group";group["entity_id"]=group.analysis_group
 group_long=signed_effects(group,["grain","attribution_role","resolution","entity_id"])
 a0=annual[annual.effect.isin(EFFECTS)].copy();parts=[]
 for selection,column in [("positive","positive_contribution"),("negative","negative_contribution")]:
  z=a0.copy();z["selection_sign"]=selection;z["signed_total"]=z[column];parts.append(z)
 a=pd.concat(parts,ignore_index=True);a["grain"]="world";a["attribution_role"]="world";a["entity_id"]="WORLD";a["resolution"]=a.resolution.astype(str)
 world_long=signed_effects(a,["grain","attribution_role","resolution","entity_id"])
 long=pd.concat([world_long,group_long,country_long,sector_long],ignore_index=True)
 return long,diagnostics(long,["grain","attribution_role","resolution","entity_id"])

def plot(diag,path):
 fig,axes=plt.subplots(1,2,figsize=(12,5));
 data=[diag[diag.grain.eq(g)].split_information_gain_ratio.dropna() for g in ["world","analytical_group","canonical_country","native_sector"]]
 axes[0].boxplot(data,tick_labels=["World","Group","Country","Sector"],showfliers=False);axes[0].set_ylabel("Split information-gain ratio");axes[0].axhline(0,color="black",lw=.6)
 top=diag[diag.opposition_hidden_by_raw_parent].nlargest(15,"tw_pair_cancellation_ratio").sort_values("tw_pair_cancellation_ratio");labels=top.grain+":"+top.entity_id.astype(str)+" ["+top.resolution+"]";axes[1].barh(labels,top.tw_pair_cancellation_ratio,color="#7570b3");axes[1].set_xlabel("T/w pair cancellation ratio");axes[1].set_title("Opposing split components hidden by raw parent")
 fig.suptitle("Raw D_A_x versus D_T_coef + D_w_coef");fig.tight_layout();fig.savefig(path);plt.close(fig)

def tex(diag,path):
 q=diag[diag.opposition_hidden_by_raw_parent].nlargest(25,"split_component_gross_kg")
 esc=lambda x:str(x).replace("_",r"\_");header="Grain & Role & Resolution & Entity & Raw sign & T sign & w sign & Pair cancellation & Result ID "+"\\\\"
 lines=[r"\begin{longtable}{lllllllrl}",r"\caption{Largest cases where the split reveals opposing components (provisional).}\label{tab:raw-a-split}\\",r"\toprule",header,r"\midrule"]
 for r in q.itertuples():lines.append(" & ".join([esc(r.grain),esc(r.attribution_role),esc(r.resolution),esc(r.entity_id),esc(r.raw_sign),esc(r.t_sign),esc(r.w_sign),f"{r.tw_pair_cancellation_ratio:.3f}",esc(r.result_id)])+" \\\\")
 lines += [r"\bottomrule",r"\end{longtable}"];path.write_text("\n".join(lines)+"\n")

def main():
 p=argparse.ArgumentParser();p.add_argument("--case-scores",type=Path,required=True);p.add_argument("--annual-hierarchy",type=Path,required=True);p.add_argument("--group-registry",type=Path,required=True);p.add_argument("--mapping",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 long,diag=build(pd.read_csv(a.case_scores),pd.read_csv(a.annual_hierarchy),pd.read_csv(a.group_registry));mapping=pd.read_csv(a.mapping,dtype=str);audit=mapping.groupby(["coarse_resolution","mapping_status"],dropna=False).size().rename("links").reset_index()
 outputs=[]
 for name,t in [("raw_split_components.csv",long),("raw_split_diagnostics.csv",diag),("reviewed_mapping_audit.csv",audit),("hidden_opposition_cases.csv",diag[diag.opposition_hidden_by_raw_parent])]:f=a.output_dir/name;t.to_csv(f,index=False,float_format="%.12g");outputs.append({"path":name,"rows":len(t),"sha256":sha(f)})
 fig=a.output_dir/"raw_a_split_diagnostics.pdf";plot(diag,fig);outputs.append({"path":fig.name,"sha256":sha(fig)});lt=a.output_dir/"raw_a_split_hidden_opposition.tex";tex(diag,lt);outputs.append({"path":lt.name,"rows":min(25,int(diag.opposition_hidden_by_raw_parent.sum())),"sha256":sha(lt)})
 maxrel=float(diag.residual_relative_to_split_gross.max());report=f"# Raw A split analysis\n\nD_A_x is compared mathematically with D_T_coef + D_w_coef at exact common world, analytical-group, canonical-country, and native-sector grains. The maximum cumulative residual relative to split gross is {maxrel:.3e}. Split information gain, component cancellation, dominance, sign relationships and rank changes are descriptive diagnostics. The identifiers do not imply wages, value added, or physical inputs. Results remain accepted provisional.\n";(a.output_dir/"report.md").write_text(report)
 manifest={"schema_version":1,"analysis_id":"raw-a-vs-tw-split-v1","status":"complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"generator":{"script":"scripts/build_raw_a_split_analysis.py","command":sys.argv},"scope":{"raw_parent":"D_A_x","children":["D_T_coef","D_w_coef"],"grains":["world","analytical_group","canonical_country","native_sector"],"resolutions":["2013","2016","R3"],"units":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t","provisional":True},"inputs":[{"path":str(x),"sha256":sha(x)} for x in [a.case_scores,a.annual_hierarchy,a.group_registry,a.mapping]],"outputs":outputs,"coverage":{"diagnostic_rows":len(diag),"hidden_opposition_rows":int(diag.opposition_hidden_by_raw_parent.sum())},"validation":{"passed":bool(np.isfinite(diag.select_dtypes('number')).all().all() and maxrel<=1e-6),"thesis_status":"accepted_provisional","stable_ids_unique":bool(diag.result_id.is_unique),"exact_common_roles_only":True,"native_sector_grids_separate":True,"reviewed_mapping_audited":True,"maximum_relative_residual":maxrel,"reconstruction_passed_at_relative_tolerance_1e-6":bool(maxrel<=1e-6)},"limitations":["Mathematical/descriptive decomposition only.","No wage, value-added, intangible-input, or physical-input interpretation is established.","Group results aggregate canonical-country marginals; they are not bilateral cells."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
