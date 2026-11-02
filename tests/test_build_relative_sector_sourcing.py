import pandas as pd
from scripts.build_relative_sector_sourcing import build

def test_relative_classification_and_unavailable_audit():
 rows=[]
 values={"A":{"D_C":10,"D_T_coef":-3,"D_w_coef":-2},"B":{"D_C":2,"D_T_coef":3,"D_w_coef":1}}
 for sector,effects in values.items():
  for effect,value in effects.items():
   rows.append({"entity_type":"sector","attribution_role":"structural_driver","effect":effect,"selection_scope":"native_resolution","resolution":"R3","canonical_entity":sector,"units":"kg CO2","selection_sign":"positive" if value>=0 else "negative","signed_total":value,"result_id":f"{sector}-{effect}"})
 mapping=pd.DataFrame(columns=["fine_sector","coarse_resolution","coarse_sector","mapping_status"])
 comp,native,cross,audit=build(pd.DataFrame(rows),mapping)
 by=native.set_index("canonical_entity")
 assert by.loc["A","relationship_class"]=="sign_switch"
 assert by.loc["B","relationship_class"]=="reinforcement"
 assert native.result_id.is_unique
 dyf=audit[audit.comparison.eq("D_YF vs D_YB + D_y")]
 assert not dyf.available.any() and set(dyf.status)=={"unavailable_not_inferred"}
