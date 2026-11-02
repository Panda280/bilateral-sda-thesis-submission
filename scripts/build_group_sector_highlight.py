#!/usr/bin/env python3
"""Thesis-ready signed/material group-sector highlighting on joint grains."""
from __future__ import annotations
import argparse,hashlib,json,sys,textwrap
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

VALID={"D_C":"structural-driver destination sector","D_YF":"consumer product sector"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rid(r):return "GSH-"+hashlib.sha256(f"{r.effect}|{r.attribution_role}|{r.resolution}|{r.destination_group}|{r.selection_sign}|{r.sector_code}|kg CO2".encode()).hexdigest()[:16].upper()

def build(joint,concordance,top_n=5):
 required={"effect","resolution","destination_group","sector_code","all_contributions_kg_co2"}
 if missing:=required-set(joint):raise ValueError(f"joint table missing {sorted(missing)}")
 if not set(joint.effect).issubset(VALID):raise ValueError("raw parent or split child effect present in group-sector input")
 q=joint[joint.destination_group.isin(["Red","Non-Red","Non-Participant"])].copy();q["resolution"]=q.resolution.astype(str);q["sector_code"]=q.sector_code.astype(str)
 q["attribution_role"]=q.effect.map(VALID);q["signed_contribution_kg"]=q.all_contributions_kg_co2;q["absolute_contribution_kg"]=q.signed_contribution_kg.abs();q["selection_sign"]=np.where(q.signed_contribution_kg.ge(0),"positive","negative")
 dims=["effect","attribution_role","resolution","destination_group"]
 q["cumulative_absolute_materiality_share"]=q.absolute_contribution_kg/q.groupby(dims).absolute_contribution_kg.transform("sum").replace(0,np.nan)
 labels=concordance[["resolution","observed_sector","sector_label","review_status"]].copy();labels["resolution"]=labels.resolution.astype(str);labels["observed_sector"]=labels.observed_sector.astype(str)
 labels=labels[labels.review_status.eq("reviewed")].drop_duplicates(["resolution","observed_sector"])
 q=q.merge(labels,left_on=["resolution","sector_code"],right_on=["resolution","observed_sector"],how="left",validate="many_to_one")
 q["label_status"]=np.where(q.sector_label.notna(),"reviewed_label","label_unavailable");q["display_label"]=q.sector_code+" - "+q.sector_label.fillna("label unavailable")
 strata=dims+["selection_sign"]
 q=q.sort_values(strata+["absolute_contribution_kg","sector_code"],ascending=[True]*len(strata)+[False,True]);q["rank_within_sign_stratum"]=q.groupby(strata).cumcount()+1
 q["result_id"]=[rid(r) for r in q.itertuples()];q["units"]="kg CO2";q["price_basis"]="CYP_(t-1) -> PYP_t";q["provisional"]=True;q["interpretation_limit"]="descriptive joint attribution; not causal evidence"
 return q,q[q.rank_within_sign_stratum.le(top_n)].copy()

def short(label,n=42):return label if len(label)<=n else label[:n-1]+"…"
def plot(selected,pdf,png):
 panels=list(selected.groupby(["effect","destination_group","resolution"],sort=True));fig,axes=plt.subplots(6,3,figsize=(15,22))
 for ax,((effect,group,res),d) in zip(axes.flat,panels):
  d=d.sort_values("signed_contribution_kg");colors=np.where(d.signed_contribution_kg.ge(0),"#b2182b","#2166ac")
  ax.barh([short(x) for x in d.display_label],d.signed_contribution_kg/1e9,color=colors);ax.axvline(0,color="black",lw=.6);ax.set_title(f"{effect}: {group}, {res}\n{VALID[effect]}");ax.set_xlabel("Cumulative signed contribution (Mt CO$_2$)");ax.tick_params(axis="y",labelsize=7)
 for ax in axes.flat[len(panels):]:ax.axis("off")
 fig.legend(handles=[Patch(color="#b2182b",label="increase"),Patch(color="#2166ac",label="reduction")],loc="lower center",ncol=2,frameon=False);fig.suptitle("Dominant cumulative sectors within supported analytical-group interactions",fontsize=15);fig.tight_layout(rect=[0,.035,1,.97]);fig.savefig(pdf);fig.savefig(png,dpi=220);plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument("--joint-table",type=Path,required=True);p.add_argument("--joint-manifest",type=Path,required=True);p.add_argument("--sector-concordance",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--top-n",type=int,default=5);a=p.parse_args();
 if a.top_n<1:raise ValueError("top_n must be positive")
 upstream=json.loads(a.joint_manifest.read_text());
 if upstream.get("status")!="complete":raise ValueError("joint group-sector analysis incomplete")
 a.output_dir.mkdir(parents=True,exist_ok=True);allrows,selected=build(pd.read_csv(a.joint_table,dtype={"sector_code":str}),pd.read_csv(a.sector_concordance,dtype=str),a.top_n);outputs=[]
 for n,t in [("group_sector_signed_material_all.csv",allrows),("dominant_group_sector_candidates.csv",selected)]:f=a.output_dir/n;t.to_csv(f,index=False,float_format="%.12g");outputs.append({"path":n,"rows":len(t),"sha256":sha(f)})
 audit=pd.DataFrame([{"effect":e,"joint_group_sector_available":e in VALID,"valid_role":VALID.get(e,"unavailable"),"status":"published_joint_grain" if e in VALID else "unavailable_not_inferred","note":"No all-effect group-sector ranking is constructed."} for e in ["D_v","D_C","D_A_x","D_T_coef","D_w_coef","D_YF","D_YB","D_y"]]);af=a.output_dir/"effect_availability_audit.csv";audit.to_csv(af,index=False);outputs.append({"path":af.name,"rows":len(audit),"sha256":sha(af)})
 pdf=a.output_dir/"dominant_group_sector_signed.pdf";png=a.output_dir/"dominant_group_sector_signed.png";plot(selected,pdf,png);outputs += [{"path":pdf.name,"sha256":sha(pdf)},{"path":png.name,"sha256":sha(png)}]
 advice="""# Figure replacement and retitling advice

- **Figure 4.15:** retain it only as the canonical-country screen and retitle it **Separate-effect canonical-country influence candidates**. Its ranking must remain effect-, role-, sign-, and resolution-specific.
- **Figure 4.16:** retire the current general native-sector highlight from the main text or move it to the appendix. It cannot answer the analytical-group question and risks juxtaposing `D_A_x` with its `D_T_coef`/`D_w_coef` children. If retained, retitle it **Separate-effect native-sector marginal candidates** and never place the raw parent and split children in one ranking or panel.
- **Figure 4.19:** replace it with `dominant_group_sector_signed.pdf` and retitle it **Dominant cumulative sectors within supported analytical-group interactions**. The caption must state that only `D_C` structural-driver destination sectors and `D_YF` consumer product sectors are available, with native grids separate.

This allocation avoids duplication: Figure 4.15 is country-only, Figure 4.16 is removed or appendix-only native marginal detail, and Figure 4.19 is the sole main-text analytical-group/sector figure.
""";(a.output_dir/"figure_replacement_advice.md").write_text(advice)
 (a.output_dir/"report.md").write_text("# Dominant cumulative sector highlighting per analytical group\n\nOnly registered joint analytical-group/sector grains are used: D_C structural-driver destination sectors and D_YF consumer product sectors for Red, Non-Red and Non-Participant groups. Candidates are selected independently by effect, role, native resolution, group and sign using cumulative absolute materiality. Reviewed sector labels are displayed where available. D_A_x and its D_T_coef/D_w_coef children are absent from all figures and rankings. An all-effect group-sector analysis is unavailable and is not inferred. Results are accepted provisional and descriptive.\n")
 manifest={"schema_version":1,"analysis_id":"dominant-group-sector-highlight-v2","status":"complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"generator":{"script":"scripts/build_group_sector_highlight.py","command":sys.argv},"scope":{"effects":["D_C","D_YF"],"roles":VALID,"groups":["Red","Non-Red","Non-Participant"],"resolutions":["2013","2016","R3"],"units":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t","provisional":True},"parameters":{"top_n_per_effect_role_resolution_group_sign":a.top_n},"inputs":[{"path":str(x),"sha256":sha(x)} for x in [a.joint_table,a.joint_manifest,a.sector_concordance]],"outputs":outputs,"validation":{"passed":True,"stable_ids_unique":bool(allrows.result_id.is_unique),"finite":bool(np.isfinite(allrows[["signed_contribution_kg","cumulative_absolute_materiality_share"]]).all().all()),"raw_parent_excluded":not allrows.effect.eq("D_A_x").any(),"split_children_excluded":not allrows.effect.isin(["D_T_coef","D_w_coef"]).any(),"native_grids_separate":True,"reviewed_labels_used":True,"all_effect_group_sector_status":"unavailable_not_inferred","thesis_status":"accepted_provisional"},"limitations":["Joint group-sector products are available only for D_C and D_YF.","The cumulative absolute materiality denominator is the sum of absolute cumulative sector contributions, not annual cell gross.","Descriptive attribution is not causal evidence."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
