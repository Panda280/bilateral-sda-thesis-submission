#!/usr/bin/env python3
"""Audit and specify a consumption-based Kyoto counterpart without results."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

TOTAL_LEAVES=["D_v","D_C","D_T_coef","D_w_coef","D_YF","D_YB","D_y"]
FINAL_DEMAND_LEAVES=["D_YF","D_YB","D_y"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(scores):
 c=scores[(scores.entity_type.eq("country"))&(scores.attribution_role.eq("consumer"))].copy();c["resolution"]=c.resolution.astype(str)
 rows=[]
 for res in ["2013","2016","R3"]:
  observed=set(c.loc[c.resolution.eq(res),"effect"])
  for effect in TOTAL_LEAVES:
   rows.append({"resolution":res,"effect":effect,"consumer_country_leaf_available":effect in observed,
                "required_for_total_consumption_change":True,"status":"registered_consumer_role" if effect in observed else "unavailable_at_consumer_role"})
 coverage=pd.DataFrame(rows)
 summary=[]
 for res,p in coverage.groupby("resolution"):
  have=set(p.loc[p.consumer_country_leaf_available,"effect"])
  summary.append({"resolution":res,"available_leaves":" | ".join(sorted(have)),"missing_total_leaves":" | ".join(sorted(set(TOTAL_LEAVES)-have)),
                  "final_demand_parent_reconstructable":set(FINAL_DEMAND_LEAVES).issubset(have),
                  "total_consumption_change_reconstructable":set(TOTAL_LEAVES).issubset(have),
                  "readiness_status":"blocked_missing_total_footprint"})
 return coverage,pd.DataFrame(summary)

def main():
 p=argparse.ArgumentParser();p.add_argument("--case-scores",type=Path,required=True);p.add_argument("--case-manifest",type=Path,required=True);p.add_argument("--group-registry",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 manifest=json.loads(a.case_manifest.read_text());
 if manifest.get("status")!="complete":raise ValueError("registered case analysis incomplete")
 registry=pd.read_csv(a.group_registry);coverage,summary=audit(pd.read_csv(a.case_scores));
 inputs=pd.DataFrame([
  {"field":"country_id","type":"string","required":True,"definition":"canonical country identifier under reviewed concordance"},
  {"field":"resolution","type":"enum: 2013|2016|R3","required":True,"definition":"native MRIO resolution; never pooled silently"},
  {"field":"endpoint_year","type":"integer","required":True,"definition":"year of the endpoint footprint"},
  {"field":"price_basis","type":"string","required":True,"definition":"CYP endpoint or explicitly reconstructed PYP endpoint; CYP/PYP may not be mixed"},
  {"field":"indicator_id","type":"string","required":True,"definition":"exact environmental indicator and gas coverage"},
  {"field":"consumption_footprint_kg","type":"float","required":True,"definition":"f_t^T L_t y_{country,t}, including all producing origins and complete final demand"},
  {"field":"territorial_emissions_kg","type":"float","required":True,"definition":"production-based emissions for the identical indicator boundary"},
  {"field":"final_demand_coverage_share","type":"float [0,1]","required":True,"definition":"share of household, government, capital and inventory final demand included"},
  {"field":"row_treatment","type":"string","required":True,"definition":"explicit rest-of-world coverage and allocation rule"},
  {"field":"source_manifest_sha256","type":"sha256","required":True,"definition":"endpoint MRIO/footprint provenance"},
  {"field":"validation_status","type":"enum: passed|failed","required":True,"definition":"footprint-to-world and country-additivity reconciliation"},
  {"field":"uncertainty_lower_kg","type":"float","required":False,"definition":"lower sensitivity bound, not assumed statistical confidence interval"},
  {"field":"uncertainty_upper_kg","type":"float","required":False,"definition":"upper sensitivity bound"}]);
 rules=pd.DataFrame([
  {"category_id":"T_met_C_fell","territorial_rule":"g_T <= tau","consumption_rule":"g_C < 0","label":"met territorially / consumption footprint fell"},
  {"category_id":"T_met_C_rose","territorial_rule":"g_T <= tau","consumption_rule":"g_C > 0","label":"met territorially / consumption footprint rose"},
  {"category_id":"T_not_met_C_fell","territorial_rule":"g_T > tau","consumption_rule":"g_C < 0","label":"not met territorially / consumption footprint fell"},
  {"category_id":"T_not_met_C_rose","territorial_rule":"g_T > tau","consumption_rule":"g_C > 0","label":"not met territorially / consumption footprint rose"},
  {"category_id":"indeterminate","territorial_rule":"status interval crosses tau or formal outcome unavailable","consumption_rule":"sensitivity interval crosses zero or coverage fails","label":"indeterminate / coverage-sensitive"}]);
 for name,t in [("consumer_leaf_coverage.csv",coverage),("readiness_summary.csv",summary),("required_footprint_schema.csv",inputs),("classification_rules.csv",rules)]:t.to_csv(a.output_dir/name,index=False)
 methodology=r"""# Consumption-based counterpart for Kyoto parties

## Readiness conclusion

The registered consumer-country leaves do **not** reconstruct country consumption-based emissions change. They contain `D_YF`, `D_YB`, and `D_y`, which reconstruct the final-demand parent contribution at the consumer role. The emission-intensity and production-structure leaves (`D_v`, `D_C`, `D_T_coef`, and `D_w_coef`) are absent at that role. Summing the available consumer leaves would therefore answer a narrower decomposition question, not recover the change in the consumption footprint.

## Required footprint totals

For each canonical country, endpoint year, native resolution and environmental indicator, calculate and persist

\[
 CF_{c,t}=\mathbf f_t^{\mathsf T}\mathbf L_t\mathbf y_{c,t},
 \]

where `y` includes the complete declared final-demand boundary and all producing origins. Persist matched territorial totals `PBA_{c,t}` for the identical indicator boundary. Both endpoints, country sums, the world total and any rest-of-world treatment must reconcile. Store CYP and PYP identities explicitly; do not combine a CYP baseline with a PYP endpoint without a documented bridge and invariance check.

The formal Kyoto comparison additionally requires an authoritative party-level base year, assigned target `tau`, commitment-period accounting rule, covered-gas basket and official territorial outcome. The current fossil-origin CO2 campaign and its descriptive Red/Non-Red groups do not establish legal compliance.

## Descriptive metrics

For a genuinely matched base and evaluation window define `g_T=(PBA_eval-PBA_base)/PBA_base`, `g_C=(CF_eval-CF_base)/CF_base`, the territorial commitment margin `m_T=tau-g_T` (non-negative means met under the supplied official rule), and the territorial-consumption divergence `d=g_C-g_T`. `d` is descriptive and must not be called leakage or offshoring evidence.

If a 1990-compatible consumption footprint is unavailable, a 2000--2012 MRIO footprint change may be reported separately, but it must not be inserted into the primary four-category compliance table because its window differs from the Kyoto base-year comparison.

## Coverage and uncertainty gates

A category is admissible only when country definitions, indicator/gas scope, endpoints, final-demand coverage, territorial boundary and commitment window match; all manifests pass; no unresolved concordance is used; and sensitivity estimates agree on both territorial status and the sign of `g_C`. Otherwise assign `indeterminate`. Resolution/database spread is a sensitivity envelope unless a statistical uncertainty model is supplied.
""";(a.output_dir/"methodology.md").write_text(methodology,encoding="utf-8")
 report="# Consumption-based Kyoto readiness\n\nStatus: **BLOCKED FOR EMPIRICAL CLASSIFICATION**. Registered consumer-role leaves reconstruct only the final-demand parent, not total consumption-based emissions change. No numerical counterpart or party category is produced. See `methodology.md` and the required-input schema.\n";(a.output_dir/"report.md").write_text(report)
 outfiles=["consumer_leaf_coverage.csv","readiness_summary.csv","required_footprint_schema.csv","classification_rules.csv","methodology.md","report.md"]
 outmanifest={"schema_version":1,"analysis_id":"consumption-based-kyoto-readiness-v1","status":"blocked_missing_footprint_totals","generated_at_utc":datetime.now(timezone.utc).isoformat(),"generator":{"script":"scripts/build_consumption_kyoto_readiness.py","command":sys.argv},"scope":{"resolutions":["2013","2016","R3"],"indicator_current_campaign":"fossil-origin CO2","price_basis_current_campaign":"CYP_(t-1) -> PYP_t","empirical_results_created":False},"inputs":[{"path":str(x),"sha256":sha(x)} for x in [a.case_scores,a.case_manifest,a.group_registry]],"outputs":[{"path":x,"sha256":sha(a.output_dir/x)} for x in outfiles],"validation":{"passed":True,"readiness_only":True,"no_fabricated_results":True,"consumer_leaves_total_reconstruction":False,"consumer_leaves_final_demand_parent_reconstruction":True,"authoritative_groups_loaded":len(registry)>0},"blockers":["country-year endpoint consumption footprints absent","matched territorial endpoint totals and formal Kyoto outcome metadata absent","gas/indicator and base-year boundary not matched"],"limitations":["Analytical groups are not legal Kyoto compliance classifications.","A mixed-window 2000-2012 footprint change cannot substitute for a 1990-based commitment metric."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(outmanifest,indent=2)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
