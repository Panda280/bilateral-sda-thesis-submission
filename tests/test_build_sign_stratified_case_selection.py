import json
import pandas as pd
from build_sign_stratified_case_selection import build

def test_sign_stratification_exclusions_units_and_stable_ids(tmp_path):
    rows=[]
    for res,scale in (("2013",1),("2016",1.05),("R3",.95)):
        for year in (2001,2002,2003):
            for entity,value in (("A",10*scale),("B",-8*scale)):
                rows.append(dict(year_current=year,effect="D_C",attribution_role="emitter_source",resolution=res,selection_sign="positive" if value>0 else "negative",entity_type="country",canonical_entity=entity,mapping_status="reviewed_identity",comparison_eligible=True,value=value,units="kg CO2",anomaly_score=float("inf") if entity=="A" else .2,zero_mad=entity=="A",cancellation_ratio=.3,upstream_result_id=f"R-{res}-{year}-{entity}"))
        rows.append(dict(year_current=2004,effect="D_C",attribution_role="emitter_source",resolution=res,selection_sign="positive",entity_type="sector",canonical_entity="52",mapping_status="unresolved_one_to_many",comparison_eligible=False,value=5,units="kg CO2",anomaly_score=0,zero_mad=False,cancellation_ratio=0,upstream_result_id="R-X"))
    # Undefined zero-MAD scores must be normalized by the producer, not passed as infinity.
    for row in rows:
        if row["zero_mad"]: row["anomaly_score"]=0
    table=tmp_path/"cases.csv"; pd.DataFrame(rows).to_csv(table,index=False)
    manifest=tmp_path/"manifest.json"; manifest.write_text(json.dumps({"status":"accepted_provisional"}))
    scores,selected,plot,excluded,blockers,_=build(table,[manifest],min_materiality=0,min_persistence=.6,max_resolution_sensitivity=.5,top_n=1)
    assert not blockers; assert set(selected.selection_sign)=={"positive","negative"}
    assert set(selected.units)=={"kg CO2"}; assert scores.result_id.is_unique
    assert set(excluded.exclusion_reason)=={"unresolved_cross_resolution_mapping"}
    reordered=tmp_path/"reordered.csv"; pd.DataFrame(rows[::-1]).to_csv(reordered,index=False)
    scores2,*_=build(reordered,[manifest],min_materiality=0,min_persistence=.6,max_resolution_sensitivity=.5,top_n=1)
    assert set(scores.result_id)==set(scores2.result_id)
