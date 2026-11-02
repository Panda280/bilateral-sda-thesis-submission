#!/usr/bin/env python3
"""Descriptive Kyoto analytical-group deep dive from registered panels."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

EFFECTS=["D_v","D_C","D_T_coef","D_w_coef","D_YF","D_YB","D_y"]
GROUPS=["Red","Non-Red","Non-Participant"]
SCALE=.6744897501960817
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def robust(s):
    x=pd.to_numeric(s).astype(float);med=float(x.median());mad=float((x-med).abs().median())
    if mad==0:return pd.Series(np.zeros(len(x)),index=x.index),True,med,mad
    return SCALE*(x-med)/mad,False,med,mad

def mappings(country_path,group_path):
    c=pd.read_csv(country_path,dtype=str).fillna("");g=pd.read_csv(group_path,dtype=str).fillna("")
    reqc={"resolution","observed_country","canonical_country"};reqg={"canonical_country","analysis_group"}
    if not reqc<=set(c) or not reqg<=set(g):raise ValueError("normalized country/group registry required")
    if (c.groupby(["resolution","observed_country"]).canonical_country.nunique()>1).any():raise ValueError("ambiguous country mapping")
    if g.canonical_country.duplicated().any() or not set(g.analysis_group)<=set(GROUPS):raise ValueError("invalid group registry")
    cm=c.drop_duplicates(["resolution","observed_country"]).set_index(["resolution","observed_country"]).canonical_country.to_dict()
    gm=g.set_index("canonical_country").analysis_group.to_dict()
    return cm,gm

def prepare(path,role,cm,gm):
    x=pd.read_parquet(path);x=x[x.effect_id.isin(EFFECTS)&x.is_publishable.astype(bool)&x.validation_status.eq("passed")].copy()
    x["resolution"]=x.resolution.astype(str);x["canonical_country"]=[cm.get((r,str(v))) for r,v in zip(x.resolution,x.label)]
    if x.canonical_country.isna().any():raise ValueError(f"unmapped countries: {x.loc[x.canonical_country.isna(),['resolution','label']].drop_duplicates().values.tolist()[:20]}")
    x["analysis_group"]=x.canonical_country.map(gm)
    if x.analysis_group.isna().any():raise ValueError("canonical countries absent from group registry")
    x["attribution_role"]=role
    return x

def calculate(emitter,consumer):
    allrows=pd.concat([emitter,consumer],ignore_index=True)
    k=["year_previous","year_current","effect_id","attribution_role","resolution","units","analysis_group","canonical_country"]
    annual_country=allrows.groupby(k,as_index=False).agg(net=("contribution","sum"),positive=("positive_contribution","sum"),negative=("negative_contribution","sum"),gross=("gross_contribution","sum"))
    annual_country["cancellation_ratio"]=np.where(annual_country.gross>0,1-annual_country.net.abs()/annual_country.gross,0.)
    annual_country["sign"]=np.select([annual_country.net>0,annual_country.net<0],["increase","reduction"],default="zero")
    series=["effect_id","attribution_role","resolution","units","analysis_group","canonical_country"]
    annual_country["annual_anomaly_score"]=0.;annual_country["annual_zero_mad"]=False;annual_country["annual_median"]=0.;annual_country["annual_mad"]=0.
    for _,idx in annual_country.groupby(series).groups.items():
        z,zero,med,mad=robust(annual_country.loc[idx,"net"]);annual_country.loc[idx,"annual_anomaly_score"]=z;annual_country.loc[idx,"annual_zero_mad"]=zero;annual_country.loc[idx,"annual_median"]=med;annual_country.loc[idx,"annual_mad"]=mad
    annual_country["annual_anomaly_flag"]=annual_country.annual_anomaly_score.abs().ge(3.5)&~annual_country.annual_zero_mad
    gkeys=["year_previous","year_current","effect_id","attribution_role","resolution","units","analysis_group"]
    annual_group=annual_country.groupby(gkeys,as_index=False).agg(net=("net","sum"),positive=("positive","sum"),negative=("negative","sum"),gross=("gross","sum"))
    annual_group["cancellation_ratio"]=np.where(annual_group.gross>0,1-annual_group.net.abs()/annual_group.gross,0.)
    ckeys=["effect_id","attribution_role","resolution","units","analysis_group"]
    cumulative_group=annual_group.groupby(ckeys,as_index=False).agg(year_count=("year_current","nunique"),cumulative_net=("net","sum"),cumulative_positive=("positive","sum"),cumulative_negative=("negative","sum"),cumulative_gross=("gross","sum"),mean_annual_net=("net","mean"),annual_net_volatility=("net",lambda x:x.std(ddof=0)))
    cumulative_group["cancellation_ratio"]=np.where(cumulative_group.cumulative_gross>0,1-cumulative_group.cumulative_net.abs()/cumulative_group.cumulative_gross,0.)
    cumulative_group["cumulative_sign"]=np.select([cumulative_group.cumulative_net>0,cumulative_group.cumulative_net<0],["increase","reduction"],default="zero")
    country=annual_country.groupby(series,as_index=False).agg(year_count=("year_current","nunique"),cumulative_net=("net","sum"),cumulative_positive=("positive","sum"),cumulative_negative=("negative","sum"),cumulative_gross=("gross","sum"),annual_anomaly_count=("annual_anomaly_flag","sum"),maximum_absolute_annual_anomaly=("annual_anomaly_score",lambda x:x.abs().max()))
    country["sign"]=np.select([country.cumulative_net>0,country.cumulative_net<0],["increase","reduction"],default="zero")
    totals=country.groupby(ckeys).agg(group_net=("cumulative_net","sum"),group_gross=("cumulative_gross","sum"),group_abs_country_net=("cumulative_net",lambda x:x.abs().sum())).reset_index()
    country=country.merge(totals,on=ckeys,validate="many_to_one");country["gross_share_within_group"]=np.where(country.group_gross>0,country.cumulative_gross/country.group_gross,0.)
    country["absolute_net_share_within_group"]=np.where(country.group_abs_country_net>0,country.cumulative_net.abs()/country.group_abs_country_net,0.)
    country["leave_one_out_group_net"]=country.group_net-country.cumulative_net;country["leave_one_out_group_gross"]=country.group_gross-country.cumulative_gross
    country["leave_one_out_sign_reversal"]=(np.sign(country.group_net)!=np.sign(country.leave_one_out_group_net))&(country.group_net!=0)
    country["influential_country"]=(country.gross_share_within_group>=.25)|country.leave_one_out_sign_reversal
    country["absolute_cumulative_net"]=country.cumulative_net.abs()
    country["rank_within_sign"]=country.groupby(ckeys+["sign"]).absolute_cumulative_net.rank(method="first",ascending=False).astype(int)
    ranked=country.sort_values(ckeys+["sign","rank_within_sign"])
    influential=country[country.influential_country|country.annual_anomaly_count.gt(0)].copy()
    return annual_country,annual_group,cumulative_group,ranked,influential

def figures(group,country,out):
    import matplotlib.pyplot as plt
    records=[]
    emitter=group[group.attribution_role.eq("emitter")]
    pivot=emitter.pivot_table(index=["analysis_group","effect_id"],columns="resolution",values="cumulative_net")/1e9
    fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True);lim=max(abs(pivot.min().min()),abs(pivot.max().max()))
    im=ax.imshow(pivot,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto");ax.set_yticks(range(len(pivot)),[f"{g} · {e}" for g,e in pivot.index]);ax.set_xticks(range(len(pivot.columns)),pivot.columns);fig.colorbar(im,ax=ax,label="Cumulative net (Mt CO$_2$)");ax.set_title("Emitter-side cumulative SDA attribution by analytical group")
    for ext in ("pdf","png"):
        p=out/f"kyoto_group_emitter_effect_heatmap.{ext}";fig.savefig(p,dpi=300,bbox_inches="tight");records.append({"path":p.name,"sha256":sha(p)})
    plt.close(fig)
    top=country[(country.attribution_role=="emitter")&(country.rank_within_sign<=3)].copy().sort_values("gross_share_within_group").tail(30)
    labels=top.analysis_group+" · "+top.effect_id+" · "+top.resolution+" · "+top.canonical_country
    fig,ax=plt.subplots(figsize=(10,9),constrained_layout=True);ax.barh(labels,top.cumulative_net/1e9,color=np.where(top.cumulative_net<0,"#4477AA","#CC6677"));ax.axvline(0,color="black",lw=.7);ax.set_xlabel("Cumulative net (Mt CO$_2$)");ax.set_title("Leading emitter-side country drivers within analytical groups")
    for ext in ("pdf","png"):
        p=out/f"kyoto_group_country_drivers.{ext}";fig.savefig(p,dpi=300,bbox_inches="tight");records.append({"path":p.name,"sha256":sha(p)})
    plt.close(fig);return records

def main():
    p=argparse.ArgumentParser();p.add_argument("--panel-root",type=Path,required=True);p.add_argument("--country-map",type=Path,required=True);p.add_argument("--group-registry",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    cm,gm=mappings(a.country_map,a.group_registry);ep=a.panel_root/"emitter_country.parquet";cp=a.panel_root/"consumer_country.parquet"
    emitter=prepare(ep,"emitter",cm,gm);consumer=prepare(cp,"consumer",cm,gm)
    annual_country,annual_group,cumulative_group,ranked,influential=calculate(emitter,consumer)
    availability=pd.DataFrame([{"accounting_view":"emitter-side SDA attribution","role":"emitter","effect_id":e,"available":True,"is_complete_accounting_total":False,"interpretation":"effect contribution allocated by emitting country; not a territorial inventory total"} for e in EFFECTS]+[{"accounting_view":"consumer-role final-demand attribution","role":"consumer","effect_id":e,"available":e in {"D_YF","D_YB","D_y"},"is_complete_accounting_total":False,"interpretation":"consumer-role attribution component; no true consumption-based total is registered"} for e in EFFECTS])
    outputs=[]
    for name,x in (("accounting_view_availability.csv",availability),("annual_country_effects.csv",annual_country),("annual_group_effects.csv",annual_group),("cumulative_group_effects.csv",cumulative_group),("country_driver_rankings.csv",ranked),("influential_countries_and_anomalies.csv",influential)):
        q=a.output_dir/name;x.to_csv(q,index=False);outputs.append({"path":name,"sha256":sha(q),"rows":len(x)})
    table=ranked[(ranked.attribution_role=="emitter")&(ranked.rank_within_sign<=3)].copy();q=a.output_dir/"thesis_country_driver_table.csv";table.to_csv(q,index=False);outputs.append({"path":q.name,"sha256":sha(q),"rows":len(table)})
    outputs+=figures(cumulative_group,ranked,a.output_dir)
    report=f"""# Kyoto analytical-group deep dive\n\nStatus: **accepted provisional**; descriptive and non-causal. Values remain kg CO2 under `CYP_(t-1) -> PYP_t`.\n\n## Accounting boundary\n\nEmitter-role SDA contributions are available for all seven requested effects and are reported as **emitter-side attribution**, not as observed territorial inventory changes. Consumer-role attribution exists only for `D_YF`, `D_YB`, and `D_y`. No registered `delta_g` consumer-country panel or complete consumption-based emissions total exists. The consumer-role components therefore cannot be treated as a consumption-based total or as equivalent to the emitter-side view.\n\n## Outputs\n\nAll Red, Non-Red, and Non-Participant groups are retained separately by effect, role, resolution, transition, sign, net, positive, negative, gross, and cancellation. Country rankings are separate by group, effect, role, resolution, and cumulative sign. Leave-one-out sign changes and annual robust-MAD anomalies are explicit; zero MAD yields a finite zero score plus a flag. {len(influential):,} country series are influential by 25% within-group gross share, leave-one-out sign reversal, or an annual anomaly flag.\n""";rp=a.output_dir/"report.md";rp.write_text(report,encoding="utf-8");outputs.append({"path":rp.name,"sha256":sha(rp)})
    m={"schema_version":1,"analysis_id":"kyoto-group-deep-dive-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"accepted_provisional","generator":{"script":"scripts/analyze_kyoto_group_deep_dive.py","command":sys.argv},"scope":{"effects":EFFECTS,"groups":GROUPS,"resolutions":["2013","2016","R3"],"units":"kg CO2","price_basis":"CYP_(t-1) -> PYP_t","indicator":"fossil-origin CO2"},"accounting_boundary":{"emitter":"emitter-side effect attribution, not a territorial inventory total","consumer":"only final-demand subeffects available","true_consumption_based_total_available":False},"parameters":{"annual_mad_threshold":3.5,"zero_mad_score":0.0,"influential_gross_share":.25},"inputs":[{"path":str(ep),"sha256":sha(ep)},{"path":str(cp),"sha256":sha(cp)},{"path":str(a.country_map),"sha256":sha(a.country_map)},{"path":str(a.group_registry),"sha256":sha(a.group_registry)}],"outputs":outputs,"validation":{"passed":True,"finite":bool(np.isfinite(cumulative_group.select_dtypes('number')).all().all()),"unmapped_countries":0},"limitations":["Accepted provisional due to unresolved upstream validation.","Analytical groups are not legal Kyoto compliance categories.","Results are descriptive and non-causal.","Emitter and consumer roles are not equivalent accounting totals."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8");return 0
if __name__=="__main__":raise SystemExit(main())
