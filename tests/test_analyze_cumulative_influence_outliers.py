import numpy as np
import pandas as pd
import analyze_cumulative_influence_outliers as analysis


def rows(entity, values, gross=None):
    gross=gross or [abs(v) for v in values]
    return [{"entity_type":"country","effect_id":"D_C","attribution_role":"source","resolution":"2016","units":"kg CO2",
             "canonical_id":entity,"year_previous":2000+i,"year_current":2001+i,"contribution":v,
             "positive_contribution":max(v,0),"negative_contribution":min(v,0),"gross_contribution":g}
            for i,(v,g) in enumerate(zip(values,gross))]


def test_zero_mad_is_explicit_and_finite():
    score,zero,median,mad=analysis.robust(pd.Series([1.,1.,2.]))
    assert zero and mad==0 and median==1
    assert np.isfinite(score).all() and (score==0).all()


def test_cumulative_influence_separate_from_temporal_anomaly_and_leave_one_out_sign():
    frame=pd.DataFrame(rows("A",[10]*10)+rows("B",[-1]*10))
    annual,cumulative,flags=analysis.analyze(frame,min_gross_share=.01,min_years=10,dominance_share=.5)
    assert not annual.temporal_anomaly_flag.any()
    a=cumulative[cumulative.canonical_id.eq("A")].iloc[0]
    assert a.cumulative_net==100 and a.leave_one_out_net==-10
    assert a.leave_one_out_sign_reversal
    assert a.dominant_single_entity
    assert set(cumulative.cumulative_sign)=={"positive","negative"}


def test_materiality_blocks_short_dominant_series():
    frame=pd.DataFrame(rows("A",[10]*2)+rows("B",[1]*2))
    _,cumulative,flags=analysis.analyze(frame,min_gross_share=.01,min_years=10,dominance_share=.5)
    assert not cumulative.material.any()
    assert flags.empty
