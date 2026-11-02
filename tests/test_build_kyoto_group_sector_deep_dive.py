import pandas as pd
from scripts.build_kyoto_group_sector_deep_dive import build

def test_rankings_are_sign_role_resolution_safe_and_unavailable_is_explicit():
    base=[]
    for sector,value in [("1",10.),("2",-8.),("3",5.)]:
      base.append({"resolution":"R3","effect":"D_C","sector_role":"destination_sector","destination_group":"Red","sector_code":sector,"all_contributions_kg_co2":value,"np_cross_border_kg_co2":value,"domestic_kg_co2":-value/2,"same_group_kg_co2":value/3,"same_group_cross_border_kg_co2":value/4})
    mapping=pd.DataFrame([{"fine_sector":str(i),"coarse_resolution":"2016","coarse_sector":"A","mapping_status":"reviewed_unique"} for i in range(1,4)])
    allrows,selected,interactions,comparison,audit=build(pd.DataFrame(base),mapping,top_n=1)
    assert set(selected.selection_sign)=={"positive","negative"}
    assert selected.groupby(["effect","attribution_role","resolution","destination_group","selection_sign"]).size().max()==1
    assert allrows.result_id.is_unique and interactions.result_id.is_unique
    row=audit[(audit.effect=="D_C")&(audit.analysis_group=="Red")].iloc[0]
    assert bool(row.group_sector_joint_available) and not bool(row.group_country_sector_joint_available)
    unavailable=audit[(audit.effect=="D_v")&(audit.analysis_group=="Red")].iloc[0]
    assert unavailable.status=="unavailable_not_inferred"
