import pandas as pd
from scripts.build_consumption_kyoto_readiness import audit

def test_final_demand_parent_is_not_total_consumption_change():
 rows=[]
 for res in ["2013","2016","R3"]:
  for effect in ["D_YF","D_YB","D_y"]:rows.append({"entity_type":"country","attribution_role":"consumer","resolution":res,"effect":effect})
 coverage,summary=audit(pd.DataFrame(rows))
 assert summary.final_demand_parent_reconstructable.all()
 assert not summary.total_consumption_change_reconstructable.any()
 assert set(coverage.loc[~coverage.consumer_country_leaf_available,"effect"])=={"D_v","D_C","D_T_coef","D_w_coef"}
