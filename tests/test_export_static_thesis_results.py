import json
import pandas as pd
from export_static_thesis_results import run

def test_blocks_all_missing_explicit_views(tmp_path):
    spec=tmp_path/"spec.json"; spec.write_text(json.dumps({"views":[]}))
    blockers,_,outputs=run(spec,tmp_path/"out")
    assert any("required views" in x for x in blockers); assert outputs==[]

def test_deterministic_data_exports_with_complete_views(tmp_path):
    views=[]
    for name in ("annual_hierarchy","cross_resolution","international_domestic","group_trends","selected_cases"):
        table=tmp_path/f"{name}.csv"; manifest=tmp_path/f"{name}.json"
        pd.DataFrame({"year":[2,1],"effect":["D_C","D_C"],"role":["emitter_source"]*2,"resolution":["R3"]*2,"units":["Mt CO2"]*2,"value":[2.,1.]}).to_csv(table,index=False)
        manifest.write_text(json.dumps({"status":"complete","validation":{"passed":True}}))
        views.append({"id":name,"table":str(table),"manifest":str(manifest),"required_columns":["year","effect","role","resolution","units","value"],"semantic_columns":["effect","role","resolution","units"],"numeric_columns":["year","value"],"key_columns":["year","effect","role","resolution"],"plot_columns":["year","effect","role","resolution","units","value"],"x":"year","y":"value","series":["resolution"],"title":name})
    spec=tmp_path/"spec.json"; spec.write_text(json.dumps({"views":views})); out=tmp_path/"out"
    blockers,_,products=run(spec,out)
    assert not blockers; assert (out/"annual_hierarchy.table_data.csv").is_file(); assert len(products)>=10
    assert pd.read_csv(out/"annual_hierarchy.table_data.csv").year.tolist()==[1,2]
