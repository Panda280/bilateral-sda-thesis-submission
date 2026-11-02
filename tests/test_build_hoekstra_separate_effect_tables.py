import pandas as pd
import pytest

from scripts.build_hoekstra_separate_effect_tables import build


def test_separate_effect_perspectives_and_signed_synthesis():
    rows=[]
    for effect,role in [("D_C","structural_driver"),("D_C","emitter"),("D_v","emitter")]:
        for sign,value in [("positive",10.0),("negative",-6.0)]:
            rows.append({"effect":effect,"attribution_role":role,"resolution":"R3",
                         "selection_sign":sign,"entity_type":"country","canonical_entity":"A",
                         "signed_total":value,"units":"kg CO2","result_id":f"{effect}-{role}-{sign}"})
    groups=pd.DataFrame([{"canonical_country":"A","analysis_group":"Red","review_status":"reviewed"}])
    receiver,driver,synthesis,availability=build(pd.DataFrame(rows),groups)
    assert set(receiver.thesis_effect)=={"D_C","D_v"}
    assert set(driver.thesis_effect)=={"D_C"}
    row=driver.iloc[0]
    assert row.net_contribution_kg==4 and row.gross_contribution_kg==16
    assert row.cancellation_ratio==pytest.approx(.75)
    assert set(synthesis.four_category)=={"receiver_increase","receiver_decrease","driver_increase","driver_decrease"}
    dc=availability[availability.thesis_effect.eq("D_C")].set_index("attribution_role")
    assert bool(dc.loc["structural_driver","bilateral_cells_available"])
    assert not bool(dc.loc["emitter","bilateral_cells_available"])


def test_unreviewed_registry_and_wrong_units_are_rejected():
    base=pd.DataFrame([{"effect":"D_v","attribution_role":"emitter","resolution":"R3",
                       "selection_sign":"positive","entity_type":"country","canonical_entity":"A",
                       "signed_total":1,"units":"kg CO2","result_id":"x"}])
    groups=pd.DataFrame([{"canonical_country":"A","analysis_group":"Red","review_status":"unreviewed"}])
    with pytest.raises(ValueError,match="fully reviewed"):
        build(base,groups)
    groups.review_status="reviewed"; base.units="Mt CO2"
    with pytest.raises(ValueError,match="kg CO2"):
        build(base,groups)
