import numpy as np
import pandas as pd
import analyze_kyoto_group_deep_dive as analysis


def frame(role,effect,entity,group,values):
    return [{"year_previous":2000+i,"year_current":2001+i,"effect_id":effect,"attribution_role":role,"resolution":"2016","units":"kg CO2","analysis_group":group,"canonical_country":entity,"contribution":v,"positive_contribution":max(v,0),"negative_contribution":min(v,0),"gross_contribution":abs(v)} for i,v in enumerate(values)]


def test_zero_mad_is_finite_and_explicit():
    z,zero,median,mad=analysis.robust(pd.Series([2.,2.,3.]))
    assert zero and median==2 and mad==0
    assert np.isfinite(z).all() and (z==0).all()


def test_group_aggregation_country_ranking_and_leave_one_out():
    emitter=pd.DataFrame(frame("emitter","D_v","A","Red",[10]*14)+frame("emitter","D_v","B","Red",[-2]*14))
    consumer=pd.DataFrame(frame("consumer","D_y","A","Red",[1]*14))
    _,annual_group,cumulative,ranked,influential=analysis.calculate(emitter,consumer)
    row=cumulative[(cumulative.effect_id=="D_v")&(cumulative.attribution_role=="emitter")].iloc[0]
    assert row.cumulative_net==112 and row.cumulative_gross==168
    a=ranked[(ranked.effect_id=="D_v")&(ranked.canonical_country=="A")].iloc[0]
    assert a.leave_one_out_group_net==-28 and a.leave_one_out_sign_reversal
    assert a.influential_country and "A" in set(influential.canonical_country)


def test_emitter_and_consumer_roles_never_pool():
    emitter=pd.DataFrame(frame("emitter","D_y","A","Red",[5]*14))
    consumer=pd.DataFrame(frame("consumer","D_y","A","Red",[-3]*14))
    _,_,cumulative,_,_=analysis.calculate(emitter,consumer)
    assert len(cumulative)==2
    assert dict(zip(cumulative.attribution_role,cumulative.cumulative_net))=={"emitter":70,"consumer":-42}
