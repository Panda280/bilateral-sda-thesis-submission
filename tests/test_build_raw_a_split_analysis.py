import pandas as pd
import pytest
from scripts.build_raw_a_split_analysis import diagnostics

def test_reconstruction_hidden_opposition_and_ranks():
 rows=[]
 for entity,vals in {"A":{"D_A_x":5,"D_T_coef":10,"D_w_coef":-5},"B":{"D_A_x":4,"D_T_coef":3,"D_w_coef":1}}.items():
  for effect,value in vals.items():rows.append({"grain":"native_sector","attribution_role":"emitter","resolution":"R3","entity_id":entity,"effect":effect,"net_kg":value,"gross_kg":abs(value),"positive_kg":max(value,0),"negative_kg":min(value,0),"cancellation_ratio":0})
 out=diagnostics(pd.DataFrame(rows),["grain","attribution_role","resolution","entity_id"]).set_index("entity_id")
 assert out.loc["A","reconstruction_residual_kg"]==0
 assert bool(out.loc["A","opposition_hidden_by_raw_parent"])
 assert out.loc["A","tw_pair_cancellation_ratio"]==pytest.approx(2/3)
 assert out.loc["B","tw_dominance"]=="T-dominant"
 assert out.result_id.is_unique
