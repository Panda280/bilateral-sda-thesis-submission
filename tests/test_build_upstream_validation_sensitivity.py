import csv,json
from build_upstream_validation_sensitivity import build,sha

def test_missing_inventory_emits_exact_blocker_manifest(tmp_path):
    out=build(tmp_path/"missing.csv",tmp_path/"out",expected_years=range(2000,2001))
    m=json.loads((out/"analysis_manifest.json").read_text())
    assert m["status"]=="blocked" and not m["validation"]["passed"]
    assert "missing inventory" in m["limitations"][0]

def test_complete_pair_preserves_failed_gate(tmp_path):
    rows=[]
    for treatment,error in (("gras",2.0),("non_gras",3.0)):
        consumed=tmp_path/f"{treatment}.bin";consumed.write_bytes(b"x")
        evidence=tmp_path/f"{treatment}.evidence.json"
        checks={n:{"passed":n!="row_balance","max_abs_error":error,"tolerance":1.0} for n in ("gras_convergence","reproduction","row_balance","column_balance","leontief")}
        evidence.write_text(json.dumps({"year":2000,"treatment":treatment,"dimensions":{"rows":2,"columns":2,"final_demand_columns":1},"checks":checks,"consumed_files":[{"path":str(consumed),"sha256":sha(consumed)}]}))
        indicator=tmp_path/f"{treatment}.indicator.json";indicator.write_text(json.dumps({"indicator":"fossil-origin CO2","units":"kg CO2","characterization_method":"direct fossil CO2 satellite account","source_files":[{"path":"x","sha256":"y"}]}))
        canonical=tmp_path/f"{treatment}.canonical.json";canonical.write_text(json.dumps({"effects":{n:{} for n in ("D_v","D_C","D_A_x","D_T_coef","D_w_coef","D_YF","D_YB","D_y")}}))
        rows.append([2000,treatment,evidence.name,indicator.name,canonical.name])
    inv=tmp_path/"inventory.csv"
    with inv.open("w",newline="") as f:w=csv.writer(f);w.writerow(["year","treatment","evidence_json","indicator_manifest","canonical_manifest"]);w.writerows(rows)
    out=build(inv,tmp_path/"out",expected_years=range(2000,2001));m=json.loads((out/"analysis_manifest.json").read_text())
    assert m["status"]=="complete" and not m["validation"]["passed"]
    assert m["validation"]["gates"]["row_balance"] is False
    assert (out/"balancing_sensitivity.csv").is_file()
