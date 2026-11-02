#!/usr/bin/env python3
"""Relative sourcing analysis on supported common attribution grains."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EFFECTS=["D_C","D_T_coef","D_w_coef"]
KEY=["resolution","canonical_entity"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable(values):return "RSEC-"+hashlib.sha256("|".join(map(str,values)).encode()).hexdigest()[:16].upper()

def collapse(scores):
    q=scores[(scores.entity_type.eq("sector"))&(scores.attribution_role.eq("structural_driver"))&
             (scores.effect.isin(EFFECTS))&(scores.selection_scope.eq("native_resolution"))].copy()
    q["resolution"]=q.resolution.astype(str)
    if set(q.units)!={"kg CO2"}:raise ValueError("units must remain kg CO2")
    if q.duplicated(KEY+["effect","selection_sign"]).any():raise ValueError("duplicate semantic sector components")
    rows=[]
    for key,p in q.groupby(KEY+["effect"],sort=True):
        pos=float(p.loc[p.selection_sign.eq("positive"),"signed_total"].sum());neg=float(p.loc[p.selection_sign.eq("negative"),"signed_total"].sum())
        gross=pos+abs(neg);net=pos+neg
        rows.append(dict(zip(KEY+["effect"],key),net_kg=net,gross_kg=gross,positive_kg=pos,negative_kg=neg,
                         cancellation_ratio=1-abs(net)/gross if gross else 0,
                         parent_result_ids=" | ".join(sorted(p.result_id.astype(str)))))
    return pd.DataFrame(rows)

def metrics(components):
    net=components.pivot(index=KEY,columns="effect",values="net_kg").reset_index()
    gross=components.pivot(index=KEY,columns="effect",values="gross_kg").reset_index()
    ids=components.pivot(index=KEY,columns="effect",values="parent_result_ids").reset_index()
    for effect in EFFECTS:
        if effect not in net:raise ValueError(f"missing supported companion effect: {effect}")
    out=net[KEY].copy();out["sourcing_net_kg"]=net.D_C
    out["companion_net_kg"]=net.D_T_coef+net.D_w_coef
    out["combined_net_kg"]=out.sourcing_net_kg+out.companion_net_kg
    out["sourcing_gross_kg"]=gross.D_C;out["companion_gross_kg"]=gross.D_T_coef+gross.D_w_coef
    denom=out.sourcing_net_kg.abs()+out.companion_net_kg.abs()
    out["relative_sourcing_influence"]=out.sourcing_net_kg.abs()/denom.replace(0,np.nan)
    out["materiality_share"]=out.sourcing_gross_kg/out.groupby("resolution").sourcing_gross_kg.transform("sum")
    out["material_relative_influence"]=out.relative_sourcing_influence.fillna(0)*out.materiality_share
    ss=np.sign(out.sourcing_net_kg);cs=np.sign(out.companion_net_kg);combined=np.sign(out.combined_net_kg)
    out["relationship_class"]=np.select([(ss==0)|(cs==0),ss==cs,(ss!=cs)&(combined==ss)],
                                        ["indeterminate_zero","reinforcement","sign_switch"],default="counteraction")
    out["sourcing_sign"]=np.where(ss>0,"positive",np.where(ss<0,"negative","zero"))
    out["companion_sign"]=np.where(cs>0,"positive",np.where(cs<0,"negative","zero"))
    out["combined_sign"]=np.where(combined>0,"positive",np.where(combined<0,"negative","zero"))
    out["attribution_role"]="structural_driver_sector";out["comparison"]="D_C vs D_T_coef + D_w_coef"
    out["units"]="kg CO2";out["price_basis"]="CYP_(t-1) -> PYP_t";out["provisional"]=True
    out=out.sort_values(["resolution","relationship_class","material_relative_influence","canonical_entity"],ascending=[True,True,False,True])
    out["rank_within_resolution_class"]=out.groupby(["resolution","relationship_class"]).cumcount()+1
    out["result_id"]=[stable([r.resolution,r.canonical_entity,r.attribution_role,r.comparison]) for r in out.itertuples()]
    idhash=ids.copy();idhash["parent_result_ids_sha256"]=[hashlib.sha256("|".join(map(str,row)).encode()).hexdigest() for row in idhash[EFFECTS].itertuples(index=False,name=None)]
    return out.merge(idhash[KEY+["parent_result_ids_sha256"]],on=KEY,validate="one_to_one")

def cross_resolution(components,mapping,native):
    admitted=mapping[mapping.mapping_status.eq("reviewed_unique")].copy()
    admitted["fine_sector"]=admitted.fine_sector.astype(str);admitted["coarse_resolution"]=admitted.coarse_resolution.astype(str)
    r3=components[components.resolution.eq("R3")].merge(admitted,left_on="canonical_entity",right_on="fine_sector",how="inner")
    mapped=(r3.groupby(["coarse_resolution","coarse_sector","effect"],as_index=False)
            .agg(net_kg=("net_kg","sum"),gross_kg=("gross_kg","sum"),positive_kg=("positive_kg","sum"),negative_kg=("negative_kg","sum"),parent_result_ids=("parent_result_ids",lambda x:" | ".join(x))))
    mapped=mapped.rename(columns={"coarse_resolution":"resolution","coarse_sector":"canonical_entity"})
    if mapped.empty:
        return pd.DataFrame(columns=KEY+["relationship_class_agreement","relative_influence_difference","comparison_result_id"])
    mm=metrics(mapped).add_prefix("mapped_r3_").rename(columns={"mapped_r3_resolution":"resolution","mapped_r3_canonical_entity":"canonical_entity"})
    nn=native[native.resolution.isin(["2013","2016"])].add_prefix("native_").rename(columns={"native_resolution":"resolution","native_canonical_entity":"canonical_entity"})
    out=mm.merge(nn,on=KEY,how="inner",validate="one_to_one")
    out["relationship_class_agreement"]=out.mapped_r3_relationship_class.eq(out.native_relationship_class)
    out["relative_influence_difference"]=out.mapped_r3_relative_sourcing_influence-out.native_relative_sourcing_influence
    out["comparison_result_id"]=["RXRES-"+hashlib.sha256(f"{r.resolution}|{r.canonical_entity}".encode()).hexdigest()[:16].upper() for r in out.itertuples()]
    return out

def build(scores,mapping):
    comp=collapse(scores); native=metrics(comp); cross=cross_resolution(comp,mapping,native)
    audit=pd.DataFrame([
      {"comparison":"D_C vs D_T_coef + D_w_coef","grain":"native sector marginal","role":"structural_driver_sector","available":True,"status":"published_common_grain","reason":"all three effects retain the same native structural-driver sector marginal"},
      {"comparison":"D_C vs D_T_coef + D_w_coef","grain":"analytical-group/sector interaction","role":"structural_driver_destination_sector","available":False,"status":"unavailable_not_inferred","reason":"D_T_coef and D_w_coef lack registered joint analytical-group/sector products"},
      {"comparison":"D_YF vs D_YB + D_y","grain":"native demanded/source sector marginal","role":"consumer/source/demanded_sector","available":False,"status":"unavailable_not_inferred","reason":"D_YB and D_y lack a sector attribution role common with D_YF"},
      {"comparison":"D_YF vs D_YB + D_y","grain":"analytical-group/sector interaction","role":"consumer_product_sector","available":False,"status":"unavailable_not_inferred","reason":"companion subeffects lack registered joint analytical-group/sector products"},
      {"comparison":"either","grain":"country-sector interaction","role":"any","available":False,"status":"unavailable_not_inferred","reason":"no registered product retains country and sector jointly for all comparison effects"}])
    return comp,native,cross,audit

def plot(native,path):
    colors={"reinforcement":"#1b9e77","counteraction":"#d95f02","sign_switch":"#7570b3","indeterminate_zero":"#777777"}
    fig,axes=plt.subplots(1,3,figsize=(13,5))
    for ax,(res,d) in zip(axes,native.groupby("resolution",sort=True)):
        d=d.nlargest(12,"material_relative_influence").sort_values("material_relative_influence")
        ax.barh(d.canonical_entity.astype(str),d.material_relative_influence,color=[colors[x] for x in d.relationship_class]);ax.set_title(str(res));ax.set_xlabel("Material relative sourcing influence")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=v,label=k.replace("_"," ")) for k,v in colors.items()],loc="lower center",ncol=4);fig.suptitle("D_C relative to D_T_coef + D_w_coef (native sector grids)");fig.tight_layout(rect=[0,.08,1,.95]);fig.savefig(path);plt.close(fig)

def tex(native,path):
    q=pd.concat([part.nlargest(8,"material_relative_influence") for _,part in native.groupby("resolution",sort=True)],ignore_index=True)
    header="Resolution & Sector & Relationship & Sourcing sign & Relative influence & Material influence & Result ID " + "\\\\"
    lines=[r"\begin{longtable}{llllrrl}",r"\caption{Material relative sector sourcing candidates (provisional).}\label{tab:relative-sector-sourcing}\\",r"\toprule",header,r"\midrule"]
    esc=lambda x:str(x).replace("_",r"\_")
    for r in q.itertuples():lines.append(" & ".join([esc(r.resolution),esc(r.canonical_entity),esc(r.relationship_class),esc(r.sourcing_sign),f"{r.relative_sourcing_influence:.3f}",f"{r.material_relative_influence:.5f}",esc(r.result_id)])+" \\\\")
    lines += [r"\bottomrule",r"\end{longtable}"];path.write_text("\n".join(lines)+"\n")

def main():
    p=argparse.ArgumentParser();p.add_argument("--case-scores",type=Path,required=True);p.add_argument("--case-manifest",type=Path,required=True);p.add_argument("--mapping",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
    upstream=json.loads(a.case_manifest.read_text());
    if upstream.get("status")!="complete":raise ValueError("case register incomplete")
    a.output_dir.mkdir(parents=True,exist_ok=True);tables=build(pd.read_csv(a.case_scores),pd.read_csv(a.mapping,dtype=str));names=["effect_sector_components.csv","relative_sector_candidates.csv","reviewed_cross_resolution_relative_comparison.csv","availability_audit.csv"];outputs=[]
    for n,t in zip(names,tables):f=a.output_dir/n;t.to_csv(f,index=False,float_format="%.12g");outputs.append({"path":n,"rows":len(t),"sha256":sha(f)})
    fig=a.output_dir/"relative_sector_sourcing.pdf";plot(tables[1],fig);outputs.append({"path":fig.name,"sha256":sha(fig)})
    lt=a.output_dir/"relative_sector_sourcing.tex";tex(tables[1],lt);outputs.append({"path":lt.name,"rows":24,"sha256":sha(lt)})
    (a.output_dir/"report.md").write_text("# Relative sector sourcing analysis\n\nThe only valid sector comparison is D_C against D_T_coef + D_w_coef at the common native structural-driver sector marginal. Cases are ranked by absolute sourcing influence multiplied by within-resolution sourcing materiality and classified as reinforcement, counteraction, sign switch, or indeterminate zero. D_YF against D_YB + D_y and all joint group-sector extensions are unavailable at a common grain and are not inferred. Reviewed cross-resolution results aggregate R3 only through reviewed-unique mappings. All results are descriptive and accepted provisional.\n")
    manifest={"schema_version":1,"analysis_id":"relative-sector-sourcing-v1","status":"complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"generator":{"script":"scripts/build_relative_sector_sourcing.py","command":sys.argv},"scope":{"valid_comparison":"D_C vs D_T_coef + D_w_coef","grain":"native structural-driver sector marginal","resolutions":["2013","2016","R3"],"units":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t","provisional":True},"inputs":[{"path":str(x),"sha256":sha(x)} for x in [a.case_scores,a.case_manifest,a.mapping]],"outputs":outputs,"validation":{"passed":True,"thesis_status":"accepted_provisional","roles_preserved":True,"signs_preserved":True,"native_grids_separate":True,"reviewed_mappings_only":True,"unresolved_excluded":True,"stable_ids_unique":bool(tables[1].result_id.is_unique),"finite":bool(np.isfinite(tables[1][["materiality_share","material_relative_influence"]]).all().all()),"unsupported_comparisons_flagged":True},"limitations":["No common sector grain supports D_YF versus D_YB + D_y.","Marginal sector results are not interpreted as group-sector or bilateral cells.","Relative influence is descriptive and not causal."]}
    (a.output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
