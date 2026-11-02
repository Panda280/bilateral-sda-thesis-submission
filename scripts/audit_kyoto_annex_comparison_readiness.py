#!/usr/bin/env python3
"""Audit local readiness for a legal Kyoto Annex comparison without inferring compliance."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

SCHEMA=[
 ("identity","canonical_country","Stable thesis country identifier","local concordance","available"),
 ("identity","unfccc_party_name","Official UNFCCC party name for the relevant date","official UNFCCC party registry","missing_official"),
 ("identity","un_m49_or_iso3","Reviewed bridge between official party name and thesis identifier","official code list plus reviewed crosswalk","missing_review"),
 ("treaty_concept","unfccc_annex_i_status","UNFCCC Annex I status with effective dates","official UNFCCC party classification","missing_official"),
 ("treaty_concept","kyoto_annex_a_scope","Applicable Kyoto Annex A gases/source categories; not a country-membership flag","Kyoto Protocol Annex A","missing_structured"),
 ("treaty_concept","kyoto_annex_b_entry","Annex B party entry and quantified commitment text","Kyoto Protocol Annex B / amendment","missing_structured"),
 ("period","commitment_period_id","First or second commitment period","official treaty/amendment source","missing_second_period"),
 ("period","period_start_end","Exact legally applicable start/end dates","official treaty/amendment source","partial_first_only"),
 ("participation","signature_ratification_status","Signature, ratification, acceptance or approval with dates","UN Treaty Collection / UNFCCC","missing_except_canada_withdrawal"),
 ("participation","withdrawal_effective_date","Withdrawal and effective date","UN Treaty Collection / UNFCCC","partial_canada_only"),
 ("participation","amendment_acceptance_status","Second-period amendment acceptance and applicability","official depositary status","missing_official"),
 ("commitment","quantified_commitment_value","Annex B quantified commitment for party/period","official Annex B table","missing_structured"),
 ("commitment","joint_fulfilment_bubble_id","EU or other Article 4 joint-fulfilment arrangement","official agreement and accounting reports","missing_official"),
 ("commitment","bubble_allocation_value","Party allocation inside joint-fulfilment arrangement","official EU/UNFCCC accounting source","missing_official"),
 ("commitment","effective_cp1_target_percent","Party-specific target used for the non-legal consumption analogue; EU joint-fulfilment parties use their individual burden-sharing allocation","official EU burden-sharing instrument reconciled to UNFCCC accounting report","missing_official"),
 ("base","base_year_by_gas","Approved base year/period by gas and applicable exception","UNFCCC accounting reports","missing_official"),
 ("coverage","covered_gases_sectors_lulucf","Legal inventory coverage and LULUCF accounting election/rules","UNFCCC review/accounting records","missing_official"),
 ("territorial","assigned_amount_units","Initial/final assigned amount and adjustments","UNFCCC compilation/accounting database","missing_official"),
 ("territorial","inventory_emissions_commitment_period","Reviewed territorial inventory totals on legal accounting basis","UNFCCC inventory/review data","missing_official"),
 ("territorial","mechanism_units_and_carryover","ITL/registry units, cancellations, retirement and carry-over","UNFCCC accounting/registry data","missing_official"),
 ("territorial","official_compliance_outcome","Official compliance assessment or enforcement status","Kyoto compliance bodies / UNFCCC","missing_official"),
 ("consumption","consumption_footprint_level","Country consumption-based emissions level, not SDA contribution","validated MRIO footprint output","missing_local"),
 ("consumption","footprint_boundary_indicator","Gas scope, characterization, territorial/aviation/shipping boundary","validated footprint metadata","missing_local"),
 ("consumption","production_footprint_bridge","Comparable production/territorial MRIO level for footprint contrast","validated MRIO footprint output","missing_local"),
]

def inventory(groups,countries):
    g=pd.read_csv(groups,dtype=str).fillna("");c=pd.read_csv(countries,dtype=str).fillna("")
    if not {"canonical_country","analysis_group","presentation_code"}<=set(g):raise ValueError("normalized analytical registry required")
    if not {"resolution","observed_country","canonical_country"}<=set(c):raise ValueError("normalized country concordance required")
    coverage=c.groupby("canonical_country").resolution.agg(lambda x:"|".join(sorted(set(x)))).rename("resolution_coverage")
    out=g.merge(coverage,on="canonical_country",how="left");out["local_group_semantics"]="static thesis analytical convention; not treaty membership or compliance"
    for field in ["unfccc_party_name","unfccc_annex_i_status","kyoto_annex_b_cp1_commitment","kyoto_annex_b_cp2_commitment","joint_fulfilment_bubble","effective_cp1_target_percent","legal_base_year","official_compliance_outcome","participation_start_date","withdrawal_effective_date"]:out[field]="[SOURCE NEEDED]"
    out["analytical_group_preserved"]=out["analysis_group"]
    out["canada_rule"]=out["canonical_country"].map(lambda x:"retain Reduction analytical group; evaluate legal status through dated fields" if x=="CAN" else "not_applicable")
    out["legal_crosswalk_status"]="not_ready_do_not_infer"
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--group-registry",type=Path,required=True);p.add_argument("--country-concordance",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    inv=inventory(a.group_registry,a.country_concordance);schema=pd.DataFrame(SCHEMA,columns=["domain","field","definition","required_authority","local_status"])
    concepts=pd.DataFrame([
      {"concept_id":"UNFCCC_ANNEX_I","object_type":"party classification","comparison_use":"official developed/economy-in-transition party scope with effective dates","local_status":"missing structured official roster","source":"[SOURCE NEEDED: official UNFCCC classification]"},
      {"concept_id":"KYOTO_ANNEX_A","object_type":"gas and source-category scope","comparison_use":"coverage metadata only; must not be used as a country list","local_status":"official treaty citation exists; structured rows absent","source":"UNFCCC1998Kyoto, Annex A"},
      {"concept_id":"KYOTO_ANNEX_B_CP1","object_type":"party quantified commitment","comparison_use":"first-period commitment crosswalk","local_status":"official treaty citation exists; structured party table absent","source":"UNFCCC1998Kyoto, Annex B"},
      {"concept_id":"KYOTO_ANNEX_B_CP2","object_type":"party quantified commitment under amendment","comparison_use":"second-period commitment crosswalk","local_status":"no verified structured amendment/status source locally","source":"[SOURCE NEEDED: official amendment and acceptance status]"},
    ])
    periods=pd.DataFrame([
      {"period_id":"THESIS_SOURCE_TABLES","start_year":"2000","end_year":"2014","status":"local decision D-2026-008","comparison_note":"produces transitions 2001-2014"},
      {"period_id":"THESIS_SDA_TRANSITIONS","start_year":"2001","end_year":"2014","status":"local decision D-2026-008","comparison_note":"adjacent-year contributions, not emissions levels"},
      {"period_id":"KYOTO_CP1","start_year":"2008","end_year":"2012","status":"verified local claim C-2026-001","comparison_note":"overlaps thesis transitions, but official compliance fields are absent"},
      {"period_id":"KYOTO_CP2","start_year":"[SOURCE NEEDED]","end_year":"[SOURCE NEEDED]","status":"official amendment/status metadata absent locally","comparison_note":"do not construct membership or overlap until sourced"},
    ])
    tests=pd.DataFrame([
      {"test_id":"KX-01","gate":"one canonical country maps to at most one official party identity per effective interval","failure_action":"block comparison"},
      {"test_id":"KX-02","gate":"Annex I, Annex A scope and Annex B commitment fields cannot share one membership column","failure_action":"block schema"},
      {"test_id":"KX-03","gate":"commitment-period membership and values require effective dates and primary-source identifiers","failure_action":"mark SOURCE NEEDED"},
      {"test_id":"KX-04","gate":"joint-fulfilment bubble and party allocations reconcile to official aggregate","failure_action":"block numerical comparison"},
      {"test_id":"KX-05","gate":"withdrawal/non-ratification/amendment acceptance handled by dated status, not static thesis group","failure_action":"block legal classification"},
      {"test_id":"KX-06","gate":"base year, gases, sectors, LULUCF and units match before target-performance calculation","failure_action":"block compliance metric"},
      {"test_id":"KX-07","gate":"territorial inventory level and consumption footprint level are independently validated and same indicator","failure_action":"block territorial-consumption comparison"},
      {"test_id":"KX-08","gate":"SDA transition contributions are never substituted for inventory or footprint levels","failure_action":"block compliance inference"},
      {"test_id":"KX-09","gate":"consumption analogue uses the reviewed party-specific effective CP1 target; EU parties use individual burden-sharing allocations","failure_action":"block target analogue"},
      {"test_id":"KX-10","gate":"Canada's static Reduction-group assignment remains separate from dated participation and withdrawal fields","failure_action":"block analytical/legal category merge"},
    ])
    outputs=[]
    for name,x in (("local_country_readiness.csv",inv),("legal_crosswalk_schema.csv",schema),("annex_concept_registry.csv",concepts),("period_overlap_registry.csv",periods),("proposed_validation_tests.csv",tests)):
        q=a.output_dir/name;x.to_csv(q,index=False);outputs.append({"path":name,"sha256":sha(q),"rows":len(x)})
    report=f"""# Annex comparison readiness audit\n\nStatus: **blocked for legal or consumption-based performance comparison**; schema preparation is complete.\n\nThe local analytical registry contains {len(inv)} canonical-code rows, including aliases, and a static `Red`/`Non-Red`/`Non-Participant` convention. It does not encode official UNFCCC Annex I status, Annex B commitments by period, second-period amendment acceptance, EU joint-fulfilment bubbles, legal base years, covered gases/sectors, assigned amounts, reviewed territorial inventories, mechanism-unit accounting, or official compliance outcomes. It must therefore not be converted into a legal membership or compliance table.\n\nUNFCCC Annex I is a party classification. Kyoto Annex A is a gas/source-category scope and is not a country list. Kyoto Annex B contains quantified commitments and must be versioned by commitment period and dated participation status. The first-period dates are supported locally by claim C-2026-001; the second-period dates and structured membership/acceptance fields remain `[SOURCE NEEDED]`.\n\nNo validated country consumption-footprint **levels** are registered locally. Existing SDA outputs are adjacent-year decomposition contributions and cannot substitute for territorial inventory levels or consumption footprints. A future comparison requires independently validated production/territorial and consumption-based levels with matching indicator, coverage, year, unit and country boundary.\n""";rp=a.output_dir/"readiness_report.md";rp.write_text(report);outputs.append({"path":rp.name,"sha256":sha(rp)})
    m={"schema_version":1,"analysis_id":"kyoto-annex-comparison-readiness-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"blocked_missing_official_and_footprint_fields","generator":{"script":"scripts/audit_kyoto_annex_comparison_readiness.py","command":sys.argv},"scope":{"thesis_source_years":"2000-2014","sda_transitions":"2001-2014","legal_compliance_inferred":False},"inputs":[{"path":str(a.group_registry),"sha256":sha(a.group_registry)},{"path":str(a.country_concordance),"sha256":sha(a.country_concordance)}],"outputs":outputs,"readiness":{"schema_ready":True,"official_country_crosswalk_ready":False,"territorial_compliance_ready":False,"consumption_footprint_ready":False},"limitations":["No factual treaty membership is inferred from analytical groups.","No compliance is inferred from SDA.","Second-period official sources are missing locally."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(m,indent=2)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
