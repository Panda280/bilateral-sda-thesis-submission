import json
import pandas as pd
from build_integrated_case_selection import build

def test_selection_is_role_safe_and_uses_declared_gates(tmp_path):
    rows=[]
    for resolution,scale in (("2013",1),("2016",1.05),("R3",.95)):
        for year in (2001,2002,2003):
            rows += [{"year":year,"effect":"D_C","role":"emitter_source","resolution":resolution,"entity":"A","value":10*scale,"anomaly_score":.2,"cancellation_ratio":.3},
                     {"year":year,"effect":"D_C","role":"emitter_source","resolution":resolution,"entity":"B","value":(-1 if year%2 else 1),"anomaly_score":.1,"cancellation_ratio":.1}]
    table=tmp_path/"input.csv"; pd.DataFrame(rows).to_csv(table,index=False)
    manifest=tmp_path/"upstream.json"; manifest.write_text(json.dumps({"status":"complete","validation":{"passed":True}}))
    scores,selected,plot,blockers,_=build(table,[manifest],min_materiality=.1,min_persistence=.8,max_resolution_sensitivity=.5,top_n=1)
    assert not blockers; assert set(selected.entity)=={"A"}; assert len(selected)==3; assert set(plot.entity)=={"A"}
    assert set(scores[scores.entity=="B"].meets_predeclared_criteria)=={False}

def test_blocks_missing_or_failed_inputs(tmp_path):
    failed=tmp_path/"failed.json"; failed.write_text(json.dumps({"status":"failed","validation":{"passed":False}}))
    result=build(tmp_path/"missing.csv",[failed])
    assert result[0] is None
    assert any("not complete" in item for item in result[3])
    assert any("missing case-input" in item for item in result[3])

def test_blocks_when_no_upstream_manifest_is_declared(tmp_path):
    result=build(tmp_path/"missing.csv",[])
    assert any("at least one completed upstream" in item for item in result[3])
