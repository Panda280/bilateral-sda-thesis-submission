import pandas as pd
import pytest

from scripts.build_integrated_drilldown_cases import STRATUM, build


def fixtures():
    base = pd.DataFrame([
        {"effect":"D_C","attribution_role":"structural_driver","resolution":"2013","selection_sign":"positive","entity_type":"country","selection_scope":"native_resolution","canonical_entity":"A","units":"kg CO2","eligible":True,"descriptive_score":.8,"result_id":"CASE-A","signed_total":8,"absolute_total":8},
        {"effect":"D_C","attribution_role":"structural_driver","resolution":"2013","selection_sign":"positive","entity_type":"country","selection_scope":"native_resolution","canonical_entity":"B","units":"kg CO2","eligible":True,"descriptive_score":.2,"result_id":"CASE-B","signed_total":2,"absolute_total":2},
        {"effect":"D_C","attribution_role":"emitter","resolution":"2013","selection_sign":"negative","entity_type":"country","selection_scope":"native_resolution","canonical_entity":"A","units":"kg CO2","eligible":True,"descriptive_score":.9,"result_id":"CASE-C","signed_total":-9,"absolute_total":9},
    ])
    temporal=pd.DataFrame([{"effect":"D_C","resolution":"2013","year_current":2009,"robust_mad_score":3.0}])
    sourcing=pd.DataFrame([{"resolution":"2013","effect_id":"D_C","focal_role":"structural_driver_country","focal_country":"A","case_score":.7,"result_id":"SRC-A","evidence_persistence":.8,"mean_source_hhi":.2,"mean_source_entropy":2.0}])
    controlled=pd.DataFrame([{"effect_id":"D_C","target_resolution":"2013","cancellation_loss":.2,"sign_reversal_rate":.1,"sparse_edge_loss_rate":.3}])
    harmonized=pd.DataFrame([{"effect_id":"D_C","target_resolution":"2013","sign_disagreement_rate":.1,"source_substitution_total_variation":.2}])
    tw=pd.DataFrame([{"resolution":"2013","split_information_gain_ratio":.2,"paired_total_cancellation":.1,"t_absolute_share_of_signed_totals":.4,"w_absolute_share_of_signed_totals":.6}])
    return base,temporal,sourcing,controlled,harmonized,tw


def test_exact_strata_and_role_safe_sourcing_join():
    scores, selected = build(*fixtures(), top_n=1)
    structural = scores[scores.attribution_role.eq("structural_driver")].set_index("canonical_entity")
    assert structural.loc["A", "sourcing_result_id"] == "SRC-A"
    assert pd.isna(scores.loc[scores.attribution_role.eq("emitter"), "sourcing_result_id"]).all()
    assert len(selected) == scores.groupby(STRATUM, dropna=False).ngroups
    assert scores.integrated_result_id.is_unique


def test_effect_inapplicable_tw_is_missing_and_scores_are_finite():
    scores, _ = build(*fixtures())
    assert scores.tw_component.isna().all()
    assert scores.integrated_score.between(0, 1).all()


def test_units_and_parent_id_gates():
    items=list(fixtures()); items[0].loc[0,"units"]="Mt CO2"
    with pytest.raises(ValueError, match="kg CO2"):
        build(*items)
    items=list(fixtures()); items[0].loc[1,"result_id"]="CASE-A"
    with pytest.raises(ValueError, match="not unique"):
        build(*items)
