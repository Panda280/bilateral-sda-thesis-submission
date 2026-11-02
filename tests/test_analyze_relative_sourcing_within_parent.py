import numpy as np
import pandas as pd
import analyze_relative_sourcing_within_parent as analysis


def test_derive_counteraction_sign_switch_and_ratios():
    x=pd.DataFrame([{"sourcing_net":-8.,"sourcing_gross":10.,"companion_net":5.,"companion_gross":6.}])
    r=analysis.derive(x,eps=1e-9,materiality=.05).iloc[0]
    assert r.reconstructed_parent_net==-3
    assert r.sourcing_share_of_parent_gross_activity==.625
    assert r.absolute_sourcing_share_of_component_absolute_net==8/13
    assert r.offset_ratio==1-3/13
    assert r.signed_alignment=="opposed" and r.interaction_class=="counteract"
    assert r.parent_sign_without_sourcing=="positive" and r.parent_sign_with_sourcing=="negative"
    assert r.sign_switch_flag and r.material


def test_near_zero_denominators_are_explicit_and_finite():
    x=pd.DataFrame([{"sourcing_net":0.,"sourcing_gross":0.,"companion_net":0.,"companion_gross":0.}])
    r=analysis.derive(x,eps=1.,materiality=.05).iloc[0]
    assert r.parent_gross_denominator_status=="near_zero"
    assert r.component_net_denominator_status=="near_zero"
    assert r.sourcing_share_of_parent_gross_activity==0 and r.offset_ratio==0
    assert not r.material and np.isfinite(r.select_dtypes(include=[np.number]) if hasattr(r,'select_dtypes') else [r.offset_ratio]).all()


def test_reinforcement_preserves_sign():
    x=pd.DataFrame([{"sourcing_net":2.,"sourcing_gross":3.,"companion_net":5.,"companion_gross":7.}])
    r=analysis.derive(x,eps=1e-9,materiality=.05).iloc[0]
    assert r.interaction_class=="reinforce" and not r.sign_switch_flag
    assert r.parent_sign_with_sourcing==r.parent_sign_without_sourcing=="positive"
