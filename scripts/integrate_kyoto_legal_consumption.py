#!/usr/bin/env python3
"""Join verified UNFCCC CP1 accounting with the non-legal footprint analogue."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES={"Russia":"Russian Federation","USA":"United States of America"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument("--legal",type=Path,required=True);p.add_argument("--footprint",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 legal=pd.read_csv(a.legal); fp=pd.read_csv(a.footprint)
 fp["unfccc_party_name"]=fp.country_name.replace(ALIASES)
 keep=["unfccc_party_name","entity_id","cp1_protocol_party","participation_note","withdrawal_effective_date","effective_target_percent_of_base_year","joint_fulfilment_id","official_accounting_outcome","outcome_source_document","outcome_source_table","outcome_source_url"]
 out=fp.merge(legal[keep],on="unfccc_party_name",how="outer",validate="one_to_one",indicator=True)
 ratio=100*out.consumption_2008_2012_mean_mtco2/out.consumption_1990_mtco2
 coverage_ok=out.coverage_complete.astype(str).str.lower().eq("true")
 protocol_party=out.cp1_protocol_party.astype(str).str.lower().eq("yes")
 eligible=coverage_ok&out.effective_target_percent_of_base_year.notna()&protocol_party
 out["consumption_evaluation_percent_of_1990"]=ratio
 out["consumption_target_analogue_met"]=pd.Series(pd.NA,index=out.index,dtype="boolean")
 out.loc[eligible,"consumption_target_analogue_met"]=(ratio[eligible]<=out.loc[eligible,"effective_target_percent_of_base_year"])
 out["comparison_scope_note"]="Official outcome uses Kyoto GHG/unit accounting; analogue uses GCB fossil-and-cement CO2 and is non-legal."
 a.output_dir.mkdir(parents=True,exist_ok=True);csv=a.output_dir/"kyoto_legal_and_consumption_comparison.csv";out.to_csv(csv,index=False,float_format="%.12g")
 summary=pd.DataFrame([{"rows":len(out),"official_met":int(out.official_accounting_outcome.eq("met_article_3_1").sum()),"official_excess":int(out.official_accounting_outcome.eq("emissions_exceeded_retired_units").sum()),"analogue_met":int(out.consumption_target_analogue_met.eq(True).sum()),"analogue_not_met":int(out.consumption_target_analogue_met.eq(False).sum()),"analogue_indeterminate":int(out.consumption_target_analogue_met.isna().sum())}]);summary.to_csv(a.output_dir/"summary.csv",index=False)
 manifest={"schema_version":1,"analysis_id":"kyoto-legal-consumption-integration-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"complete","inputs":[{"path":str(a.legal),"sha256":sha(a.legal)},{"path":str(a.footprint),"sha256":sha(a.footprint)}],"outputs":[{"path":csv.name,"sha256":sha(csv)},{"path":"summary.csv","sha256":sha(a.output_dir/'summary.csv')}],"validation":{"legal_footprint_joined":True,"legal_and_analogue_scopes_separate":True,"analogue_requires_complete_1990_and_2008_2012":True},"limitations":["The consumption target analogue is not Kyoto legal compliance.","GCB and Kyoto environmental boundaries differ."]};(a.output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__":main()
