#!/usr/bin/env python3
"""Build hierarchy-safe emitter-country Kyoto-group figures."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

SPLIT=["D_v","D_C","D_T_coef","D_w_coef","D_YF","D_YB","D_y"]
FINAL=["D_YF","D_YB","D_y"]
GROUPS=["Red","Non-Red","Non-Participant"]
COLORS={"D_v":"#4477AA","D_C":"#66CCEE","D_T_coef":"#228833","D_w_coef":"#CCBB44","D_YF":"#AA3377","D_YB":"#EE6677","D_y":"#BBBBBB"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate(frame,resolution):
    required={"effect_id","attribution_role","resolution","units","analysis_group","canonical_country","cumulative_net","cumulative_gross"}
    if not required<=set(frame):raise ValueError(f"missing {sorted(required-set(frame))}")
    x=frame[(frame.attribution_role=="emitter")&(frame.resolution.astype(str)==str(resolution))&frame.effect_id.isin(SPLIT)].copy()
    if set(x.effect_id)!=set(SPLIT):raise ValueError("incomplete split-only effect hierarchy")
    if set(x.analysis_group)!=set(GROUPS):raise ValueError("incomplete analytical groups")
    if x.units.nunique()!=1 or x.units.iloc[0]!="kg CO2":raise ValueError("unit mismatch")
    return x

def selections(x,n_country=8,n_effect=3,n_final=6):
    # Figure 1: top countries by cumulative gross across the complete split-only hierarchy.
    totals=x.groupby(["analysis_group","canonical_country"],as_index=False).agg(selection_gross=("cumulative_gross","sum"))
    chosen=totals.sort_values(["analysis_group","selection_gross"],ascending=[True,False]).groupby("analysis_group",as_index=False).head(n_country)
    highlights=x.merge(chosen,on=["analysis_group","canonical_country"],validate="many_to_one")
    # Figure 2: top effects by global emitter gross; top five countries per group/effect.
    effect_order=x.groupby("effect_id").cumulative_gross.sum().nlargest(n_effect).index.tolist()
    comp=x[x.effect_id.isin(effect_order)].copy();comp["group_effect_gross"]=comp.groupby(["analysis_group","effect_id"]).cumulative_gross.transform("sum")
    comp["emitter_country_gross_share"]=np.where(comp.group_effect_gross>0,comp.cumulative_gross/comp.group_effect_gross,0.)
    comp=comp.sort_values(["analysis_group","effect_id","emitter_country_gross_share"],ascending=[True,True,False]).groupby(["analysis_group","effect_id"],as_index=False).head(5)
    # Figure 3: top countries by absolute reconstructed final-demand net within each group.
    f=x[x.effect_id.isin(FINAL)].copy();ft=f.groupby(["analysis_group","canonical_country"],as_index=False).agg(final_parent_net=("cumulative_net","sum"));ft["absolute_final_parent_net"]=ft.final_parent_net.abs()
    fc=ft.sort_values(["analysis_group","absolute_final_parent_net"],ascending=[True,False]).groupby("analysis_group",as_index=False).head(n_final)
    final=f.merge(fc,on=["analysis_group","canonical_country"],validate="many_to_one")
    return highlights,comp,final,effect_order

def plot(highlights,composition,final,effects,out,resolution):
    import matplotlib.pyplot as plt
    records=[]
    fig,axes=plt.subplots(1,3,figsize=(16,7),constrained_layout=True)
    for ax,group in zip(axes,GROUPS):
        q=highlights[highlights.analysis_group==group].pivot(index="canonical_country",columns="effect_id",values="cumulative_net").fillna(0)/1e9
        order=highlights[highlights.analysis_group==group].drop_duplicates("canonical_country").sort_values("selection_gross").canonical_country;q=q.reindex(order)
        pos=np.zeros(len(q));neg=np.zeros(len(q))
        for effect in SPLIT:
            v=q[effect].to_numpy();bottom=np.where(v>=0,pos,neg);ax.barh(q.index,v,left=bottom,color=COLORS[effect],label=effect);pos+=np.maximum(v,0);neg+=np.minimum(v,0)
        ax.axvline(0,color="black",lw=.7);ax.set_title(group);ax.set_xlabel("Cumulative net (Mt CO$_2$)")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=7,loc="upper center",bbox_to_anchor=(.5,-.01),frameon=False);fig.suptitle(f"Dominant emitter countries by analytical group — split hierarchy, {resolution}")
    for ext in ["pdf","png"]:
        p=out/f"dominant_cumulative_countries_split_{resolution}.{ext}";fig.savefig(p,dpi=300,bbox_inches="tight");records.append({"path":p.name,"sha256":sha(p)})
    plt.close(fig)
    fig,axes=plt.subplots(len(effects),3,figsize=(15,4*len(effects)),constrained_layout=True,squeeze=False)
    for i,effect in enumerate(effects):
        for j,group in enumerate(GROUPS):
            ax=axes[i,j];q=composition[(composition.effect_id==effect)&(composition.analysis_group==group)].sort_values("emitter_country_gross_share")
            ax.barh(q.canonical_country,q.emitter_country_gross_share*100,color=COLORS[effect]);ax.set_title(f"{group} · {effect}");ax.set_xlabel("Share of group-effect gross (%)")
    fig.suptitle(f"Production-location composition: emitter countries, {resolution}\nTop 3 effects by global gross; top 5 countries per group-effect")
    for ext in ["pdf","png"]:
        p=out/f"emitter_country_composition_{resolution}.{ext}";fig.savefig(p,dpi=300,bbox_inches="tight");records.append({"path":p.name,"sha256":sha(p)})
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,7),constrained_layout=True)
    width=.24
    for ax,group in zip(axes,GROUPS):
        q=final[final.analysis_group==group].pivot(index="canonical_country",columns="effect_id",values="cumulative_net").fillna(0)/1e9;q["parent_abs"]=(q[FINAL].sum(axis=1)).abs();q=q.sort_values("parent_abs",ascending=False)
        y=np.arange(len(q));
        for offset,effect in zip((-width,0,width),FINAL):ax.barh(y+offset,q[effect],height=width,color=COLORS[effect],label=effect)
        ax.set_yticks(y,q.index);ax.invert_yaxis();ax.axvline(0,color="black",lw=.7);ax.set_title(group);ax.set_xlabel("Cumulative emitter-role net (Mt CO$_2$)")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.5,-.01),frameon=False);fig.suptitle(f"Leading emitter-side final-demand countries: $D_y$, $D_{{Y_B}}$, and $D_{{Y_F}}$ — {resolution}")
    for ext in ["pdf","png"]:
        p=out/f"emitter_final_demand_country_components_{resolution}.{ext}";fig.savefig(p,dpi=300,bbox_inches="tight");records.append({"path":p.name,"sha256":sha(p)})
    plt.close(fig);return records

def main():
    p=argparse.ArgumentParser();p.add_argument("--country-rankings",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--resolution",default="2016");p.add_argument("--countries-per-group",type=int,default=8);p.add_argument("--composition-effects",type=int,default=3);p.add_argument("--final-countries-per-group",type=int,default=6);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    x=validate(pd.read_csv(a.country_rankings,dtype={"resolution":str}),a.resolution);h,c,f,e=selections(x,a.countries_per_group,a.composition_effects,a.final_countries_per_group)
    outputs=[]
    for name,q in (("dominant_country_highlights.csv",h),("emitter_country_composition.csv",c),("final_demand_country_components.csv",f)):
        path=a.output_dir/name;q.to_csv(path,index=False);outputs.append({"path":name,"sha256":sha(path),"rows":len(q)})
    outputs+=plot(h,c,f,e,a.output_dir,a.resolution)
    report=f"""# Revised Kyoto emitter-country figures\n\nStatus: **accepted provisional** and non-causal. All three figures use exact emitter-country attribution at resolution `{a.resolution}` and retain kg CO2 in the CSVs.\n\nThe dominant-country figure uses the split hierarchy only (`D_v`, `D_C`, `D_T_coef`, `D_w_coef`, `D_YF`, `D_YB`, `D_y`); raw `D_A_x` is excluded to prevent double counting with `D_T_coef + D_w_coef`. Within each analytical group, the {a.countries_per_group} countries with the largest cumulative gross contribution summed across these seven effects are selected before plotting signed effect stacks.\n\nThe production-location figure explicitly reports emitter countries. It selects the {a.composition_effects} effects with the largest global emitter gross at the fixed grid, then the five countries with the largest gross share within each group-effect. Selected effects: {', '.join(e)}.\n\nThe final-demand figure selects {a.final_countries_per_group} countries per group by absolute reconstructed emitter-role `D_YF + D_YB + D_y` net, then displays all three components side by side at the same role and resolution. No consumer-role or source-role values are mixed into these figures.\n""";rp=a.output_dir/"report.md";rp.write_text(report);outputs.append({"path":rp.name,"sha256":sha(rp)})
    m={"schema_version":1,"analysis_id":"revised-kyoto-country-figures-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"accepted_provisional","generator":{"script":"scripts/build_revised_kyoto_country_figures.py","command":sys.argv},"scope":{"resolution":a.resolution,"role":"emitter","units":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t","groups":GROUPS,"split_effects":SPLIT},"selection":{"dominant_countries_per_group":a.countries_per_group,"composition_effect_count":a.composition_effects,"composition_countries_per_group_effect":5,"final_countries_per_group":a.final_countries_per_group,"selected_composition_effects":e},"inputs":[{"path":str(a.country_rankings),"sha256":sha(a.country_rankings)}],"outputs":outputs,"validation":{"passed":True,"raw_D_A_x_excluded":True,"role":"emitter","resolution":a.resolution,"groups_complete":True},"limitations":["Accepted provisional due to upstream validation.","Descriptive and non-causal.","Selection is magnitude-based and fixed by the recorded rule."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(m,indent=2)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
