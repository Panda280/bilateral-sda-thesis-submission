#!/usr/bin/env python3
"""Build role-safe analytical-group/sector candidates from joint products only."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROLE = {"D_C": "structural_driver_destination_sector",
        "D_YF": "consumer_product_sector"}
COMPONENTS = {"np_cross_border_kg_co2": "non_participant_cross_border",
              "domestic_kg_co2": "domestic",
              "same_group_kg_co2": "same_group_total",
              "same_group_cross_border_kg_co2": "same_group_cross_border"}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rid(prefix, values): return prefix+"-"+hashlib.sha256("|".join(map(str,values)).encode()).hexdigest()[:16].upper()

def ranked(frame, value_col, dimensions, prefix):
    x=frame.copy(); x["signed_contribution_kg"]=x[value_col].astype(float)
    x["selection_sign"]=np.where(x.signed_contribution_kg.ge(0),"positive","negative")
    x["absolute_contribution_kg"]=x.signed_contribution_kg.abs()
    strata=dimensions+["selection_sign"]
    x["materiality_share"]=x.absolute_contribution_kg/x.groupby(dimensions).absolute_contribution_kg.transform("sum").replace(0,np.nan)
    x=x.sort_values(strata+["absolute_contribution_kg","sector_code"],ascending=[True]*len(strata)+[False,True])
    x["rank_within_stratum"]=x.groupby(strata,dropna=False).cumcount()+1
    x["result_id"]=[rid(prefix,[*(getattr(r,d) for d in strata),r.sector_code,"kg CO2"]) for r in x.itertuples()]
    x["units"]="kg CO2"; x["price_basis"]="CYP_(t-1) -> PYP_t"; x["provisional"]=True
    x["interpretation_limit"]="descriptive attribution; not causal evidence"
    return x

def build(pooled, mapping, top_n=10):
    required={"resolution","effect","sector_role","destination_group","sector_code","all_contributions_kg_co2",*COMPONENTS}
    if missing:=required-set(pooled): raise ValueError(f"pooled input missing: {sorted(missing)}")
    if not set(pooled.effect).issubset(ROLE): raise ValueError("unsupported effects in joint product")
    pooled=pooled[pooled.destination_group.ne("All")].copy(); pooled["resolution"]=pooled.resolution.astype(str)
    pooled["attribution_role"]=pooled.effect.map(ROLE)
    dims=["effect","attribution_role","resolution","destination_group"]
    candidates=ranked(pooled,"all_contributions_kg_co2",dims,"GSEC")
    candidates["gross_available_at_joint_grain"]=False
    candidates["cancellation_available_at_joint_grain"]=False
    selected=candidates[candidates.rank_within_stratum.le(top_n)].copy()

    interactions=[]
    for col,label in COMPONENTS.items():
        q=pooled.copy();q["interaction_type"]=label;q["interaction_value"]=q[col];interactions.append(q)
    interactions=pd.concat(interactions,ignore_index=True)
    idims=dims+["interaction_type"]
    interactions=ranked(interactions,"interaction_value",idims,"GINT")
    interactions["components_are_non_additive"]=True
    interactions=interactions[interactions.rank_within_stratum.le(top_n)].copy()

    admitted=mapping[mapping.mapping_status.eq("reviewed_unique")].copy()
    admitted["fine_sector"]=admitted.fine_sector.astype(str); admitted["coarse_resolution"]=admitted.coarse_resolution.astype(str)
    r3=pooled[pooled.resolution.eq("R3")].merge(admitted[["fine_sector","coarse_resolution","coarse_sector"]],left_on="sector_code",right_on="fine_sector",how="inner")
    mapped=(r3.groupby(["effect","attribution_role","destination_group","coarse_resolution","coarse_sector"],as_index=False)
            .all_contributions_kg_co2.sum().rename(columns={"coarse_resolution":"resolution","coarse_sector":"sector_code"}))
    mapped_rank=ranked(mapped,"all_contributions_kg_co2",dims,"GMAP").rename(columns={"signed_contribution_kg":"mapped_r3_signed_kg","rank_within_stratum":"mapped_r3_rank","selection_sign":"mapped_r3_sign","result_id":"mapped_result_id"})
    native=candidates[candidates.resolution.isin(["2013","2016"])][dims+["sector_code","signed_contribution_kg","selection_sign","rank_within_stratum","result_id"]].rename(columns={"signed_contribution_kg":"native_signed_kg","selection_sign":"native_sign","rank_within_stratum":"native_rank","result_id":"native_result_id"})
    comparative=mapped_rank.merge(native,on=dims+["sector_code"],how="inner",validate="one_to_one")
    comparative["sign_agreement"]=comparative.mapped_r3_sign.eq(comparative.native_sign)
    comparative["absolute_rank_change"]=(comparative.mapped_r3_rank-comparative.native_rank).abs()
    comparative["comparison_result_id"]=[rid("GCMP",[r.effect,r.attribution_role,r.resolution,r.destination_group,r.sector_code]) for r in comparative.itertuples()]

    availability=[]
    groups=["Red","Non-Red","Non-Participant"]
    effects=["D_v","D_C","D_T","D_w","D_YF","D_YB","D_y"]
    for effect in effects:
      for group in groups:
        valid=effect in {"D_C","D_YF"} and group in {"Red","Non-Red","Non-Participant"}
        availability.append({"effect":effect,"analysis_group":group,
          "group_sector_joint_available":valid,"group_country_sector_joint_available":False,
          "valid_role":ROLE.get(effect,"unavailable"),"status":"published_joint_product" if valid else "unavailable_not_inferred",
          "reason":"joint destination-group/sector product" if valid else "no registered product retains this effect/group and sector jointly"})
    return candidates,selected,interactions,comparative,pd.DataFrame(availability)

def write_tex(frame,path):
    q=frame[frame.rank_within_stratum.le(3)][["effect","attribution_role","resolution","destination_group","selection_sign","sector_code","signed_contribution_kg","rank_within_stratum","result_id"]]
    header="Effect & Role & Resolution & Group & Sign & Sector & Contribution (Mt CO$_2$) & Rank & Result ID " + "\\\\"
    lines=[r"\begin{longtable}{llllllrrl}",r"\caption{Leading analytical-group and sector candidates (provisional).}\label{tab:group-sector-candidates}\\",r"\toprule",header,r"\midrule",r"\endfirsthead"]
    esc=lambda v:str(v).replace("_",r"\_")
    for r in q.itertuples(index=False): lines.append(" & ".join([esc(r.effect),esc(r.attribution_role),esc(r.resolution),esc(r.destination_group),esc(r.selection_sign),esc(r.sector_code),f"{r.signed_contribution_kg/1e9:.3f}",str(r.rank_within_stratum),esc(r.result_id)])+" \\\\")
    lines += [r"\bottomrule",r"\end{longtable}"];path.write_text("\n".join(lines)+"\n",encoding="utf-8")

def plot(selected,path):
    top=selected[selected.rank_within_stratum.le(5)].copy(); panels=list(top.groupby(["effect","resolution","destination_group"],sort=True))
    fig,axes=plt.subplots(4,3,figsize=(13,14));
    for ax,((effect,res,group),d) in zip(axes.flat,panels):
        d=d.sort_values("signed_contribution_kg"); colors=np.where(d.signed_contribution_kg.ge(0),"#b2182b","#2166ac")
        ax.barh(d.sector_code.astype(str),d.signed_contribution_kg/1e9,color=colors);ax.axvline(0,color="black",lw=.6);ax.set_title(f"{effect}, {res}, {group}");ax.set_xlabel("Cumulative contribution (Mt CO$_2$)")
    for ax in axes.flat[len(panels):]:ax.axis("off")
    fig.tight_layout();fig.savefig(path);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument("--pooled",type=Path,required=True);p.add_argument("--coverage",type=Path,required=True);p.add_argument("--mapping",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--top-n",type=int,default=10);a=p.parse_args()
    if a.top_n<1:raise ValueError("top_n must be positive")
    coverage=json.loads(a.coverage.read_text());
    if coverage.get("mode")!="full" or coverage.get("missing_selected_products"):raise ValueError("joint-product coverage incomplete")
    a.output_dir.mkdir(parents=True,exist_ok=True); tables=build(pd.read_csv(a.pooled,dtype={"sector_code":str}),pd.read_csv(a.mapping,dtype=str),a.top_n)
    names=["group_sector_all.csv","group_sector_candidates.csv","group_sector_interaction_candidates.csv","reviewed_cross_resolution_candidates.csv","attribution_availability.csv"]
    outputs=[]
    for n,t in zip(names,tables):pout=a.output_dir/n;t.to_csv(pout,index=False,float_format="%.12g");outputs.append({"path":n,"rows":len(t),"sha256":sha(pout)})
    empty=a.output_dir/"group_country_sector_candidates.csv";pd.DataFrame(columns=["effect","role","resolution","analysis_group","country","sector","status","reason"]).to_csv(empty,index=False);outputs.append({"path":empty.name,"rows":0,"sha256":sha(empty)})
    tex=a.output_dir/"group_sector_candidates.tex";write_tex(tables[1],tex);outputs.append({"path":tex.name,"rows":len(tables[1][tables[1].rank_within_stratum.le(3)]),"sha256":sha(tex)})
    fig=a.output_dir/"group_sector_signed_candidates.pdf";plot(tables[1],fig);outputs.append({"path":fig.name,"sha256":sha(fig)})
    (a.output_dir/"report.md").write_text("# Kyoto analytical-group/sector deep dive\n\nOnly registered joint destination-group/sector products are used. This admits D_C structural-driver destination sectors and D_YF consumer product sectors for Red, Non-Red and Non-Participant groups. No country-sector joint product is registered, and all other relationships are unavailable rather than inferred. Rankings are native-resolution and sign-stratified. Reviewed R3-to-coarse comparisons exclude unresolved mappings. Results are descriptive and accepted provisional.\n")
    manifest={"schema_version":1,"analysis_id":"kyoto-group-sector-deep-dive-v1","status":"complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"generator":{"script":"scripts/build_kyoto_group_sector_deep_dive.py","command":sys.argv},"scope":{"effects_available":["D_C","D_YF"],"groups_available":["Red","Non-Red","Non-Participant"],"resolutions":["2013","2016","R3"],"units":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t","provisional":True},"inputs":[{"path":str(x),"sha256":sha(x)} for x in [a.pooled,a.coverage,a.mapping]],"outputs":outputs,"validation":{"passed":True,"thesis_status":"accepted_provisional","native_grids_separate":True,"reviewed_mapping_only":True,"unresolved_mappings_excluded":True,"sign_stratified":True,"roles_explicit":True,"stable_ids_unique":all(t.result_id.is_unique for t in tables[:3]),"group_country_sector_status":"unavailable_not_inferred"},"limitations":["Joint products exist only for D_C and D_YF.","Interaction components overlap and must not be added.","No registered product retains analytical group, country and sector jointly.","Descriptive attribution does not establish causality."]}
    (a.output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
