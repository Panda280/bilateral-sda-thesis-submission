#!/usr/bin/env python3
"""Register cancellation aggregation order and compare annual versus pooled formulas."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

DEFINITIONS=[
 {"metric_id":"annual_cancellation","formula":"1 - abs(sum_i d[i,t]) / sum_i abs(d[i,t])","positive_negative_pool_order":"classify raw persisted cells by sign, then sum within transition/reporting group","time_aggregation":"none; one adjacent-year transition","interpretation":"within-transition cancellation at the persisted cell grain"},
 {"metric_id":"mean_annual_cancellation","formula":"mean_t(annual_cancellation[t])","positive_negative_pool_order":"inherits annual cell-level pools","time_aggregation":"unweighted arithmetic mean of annual ratios","interpretation":"typical annual ratio; not pooled cancellation"},
 {"metric_id":"gross_weighted_annual_cancellation","formula":"1 - sum_t abs(net[t]) / sum_t gross[t]","positive_negative_pool_order":"inherits annual cell-level pools","time_aggregation":"annual ratios weighted by annual gross","interpretation":"within-year cancellation pooled without allowing net signs to cancel across years"},
 {"metric_id":"pooled_window_cancellation","formula":"1 - abs(sum_t net[t]) / sum_t gross[t]","positive_negative_pool_order":"cell-level annual gross retained, signed annual nets summed across time","time_aggregation":"full-window pooled signed net over full-window gross","interpretation":"includes both within-year and between-year sign cancellation"},
 {"metric_id":"temporal_mad_candidate","formula":"0.67448975 * (x - median(x)) / MAD(x)","positive_negative_pool_order":"not applicable","time_aggregation":"scores annual first differences or composition TVDs within effect/resolution/dimension/metric series","interpretation":"descriptive anomaly screen; threshold abs(score)>=3.5; zero MAD score=0 with explicit flag"},
]

def compare(annual):
    required={"effect","resolution","year_previous","year_current","net_mt","positive_mt","negative_mt","gross_mt","cancellation_ratio"}
    if not required<=set(annual):raise ValueError(f"missing {sorted(required-set(annual))}")
    x=annual.copy();x=x[x.dimension_level.eq("overall")] if "dimension_level" in x else x
    if not np.isfinite(x[["net_mt","positive_mt","negative_mt","gross_mt","cancellation_ratio"]]).all().all():raise ValueError("non-finite input")
    if not np.allclose(x.net_mt,x.positive_mt+x.negative_mt,rtol=1e-10,atol=1e-8):raise ValueError("net != positive + negative")
    if not np.allclose(x.gross_mt,x.positive_mt-x.negative_mt,rtol=1e-10,atol=1e-8):raise ValueError("gross != positive - negative")
    rows=[]
    for (effect,resolution),g in x.groupby(["effect","resolution"]):
        gross=float(g.gross_mt.sum());net=float(g.net_mt.sum());sum_abs_net=float(g.net_mt.abs().sum())
        mean=float(g.cancellation_ratio.mean());weighted=0 if gross==0 else 1-sum_abs_net/gross;pooled=0 if gross==0 else 1-abs(net)/gross
        rows.append({"effect":effect,"resolution":resolution,"units":"Mt CO2","transition_count":len(g),"cumulative_net_mt":net,"cumulative_gross_mt":gross,"arithmetic_mean_annual_cancellation":mean,"gross_weighted_annual_cancellation":weighted,"pooled_window_cancellation":pooled,"between_year_sign_cancellation_increment":pooled-weighted,"pooled_minus_arithmetic_mean":pooled-mean,"denominator_status":"ok" if gross>0 else "zero"})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument("--annual-metrics",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    annual=pd.read_csv(a.annual_metrics,dtype={"resolution":str});comparison=compare(annual);definitions=pd.DataFrame(DEFINITIONS)
    outputs=[]
    for name,x in (("cancellation_definition_registry.csv",definitions),("annual_vs_pooled_cancellation.csv",comparison)):
        q=a.output_dir/name;x.to_csv(q,index=False);outputs.append({"path":name,"sha256":sha(q),"rows":len(x)})
    report="""# Cancellation and MAD implementation audit\n\nStatus: **accepted provisional definition audit**. The formulas are code-traced; the empirical comparison uses provisional fossil-origin CO2 outputs.\n\n## Aggregation order\n\n`src/sda/output/annual.py::_metric` and `scripts/build_annual_reports.py::_metrics` flatten the selected effect cells, classify each cell as positive or negative, and only then sum positive, negative, net, and gross. `scripts/analyze_sourcing_group_trends.py::signed_metrics` likewise creates `positive_kg`, `negative_kg`, and `gross_kg` from each persisted `Value` before any reporting-group aggregation. Consequently, country, sector, relationship, analytical-group, and overall products preserve the underlying positive and negative pools: downstream aggregation sums those pools and does not recompute gross as the absolute value of an already aggregated net. Cancellation therefore depends on the persisted input cell grain but does not discard cancellation before country/group aggregation.\n\nAnnual cancellation is `1 - abs(net_t)/gross_t`. The Chapter 4 mean is the unweighted arithmetic mean of the 14 annual ratios. Pooled-window cancellation is a different statistic, `1 - abs(sum_t net_t)/sum_t gross_t`; it permits positive and negative annual nets to cancel across transitions. The gross-weighted annual statistic, `1 - sum_t abs(net_t)/sum_t gross_t`, excludes this between-year cancellation. The accompanying table quantifies all three.\n\n## MAD candidate grain\n\nThe temporal diagnostic first constructs annual overall differences for net, gross, and cancellation (13 observations per effect-resolution series) and composition TVDs for relationship, source-group, and destination-group partitions. MAD scores are calculated separately within `(effect, resolution, dimension, metric)`. The predeclared candidate threshold is `abs(z_MAD) >= 3.5`; sensitivity thresholds are 2.5, 3.0, 3.5, and 4.0. These are anomaly candidates, not structural breaks. Zero MAD now yields finite score zero and `zero_mad=true`; it never produces infinity.\n""";rp=a.output_dir/"audit_report.md";rp.write_text(report,encoding="utf-8");outputs.append({"path":rp.name,"sha256":sha(rp)})
    m={"schema_version":1,"analysis_id":"cancellation-definition-audit-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"accepted_provisional","generator":{"script":"scripts/audit_cancellation_definitions.py","command":sys.argv},"scope":{"effects":sorted(comparison.effect.unique()),"resolutions":sorted(comparison.resolution.unique()),"input_units":"Mt CO2","source_units":"kg CO2 divided exactly by 1e9","price_basis":"CYP_(t-1) -> PYP_t"},"inputs":[{"path":str(a.annual_metrics),"sha256":sha(a.annual_metrics)}],"outputs":outputs,"validation":{"passed":True,"positive_negative_identity":True,"gross_identity":True,"finite":True},"limitations":["Cancellation depends on persisted cell grain.","Arithmetic mean annual and pooled-window cancellation answer different questions.","Empirical values remain provisional due to upstream validation."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8");return 0
if __name__=="__main__":raise SystemExit(main())
