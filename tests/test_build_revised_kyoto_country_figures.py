import pandas as pd
import build_revised_kyoto_country_figures as figures


def data():
    rows=[]
    for group in figures.GROUPS:
        for country,scale in [("A",3),("B",2),("C",1)]:
            for i,effect in enumerate(figures.SPLIT,1):
                rows.append({"effect_id":effect,"attribution_role":"emitter","resolution":"2016","units":"kg CO2","analysis_group":group,"canonical_country":country,"cumulative_net":scale*i,"cumulative_gross":scale*(10-i)})
    return pd.DataFrame(rows)


def test_selection_is_emitter_only_split_only_and_group_separate():
    x=figures.validate(data(),"2016");h,c,f,e=figures.selections(x,n_country=2,n_effect=3,n_final=2)
    assert set(h.effect_id)==set(figures.SPLIT) and "D_A_x" not in set(h.effect_id)
    assert h.groupby("analysis_group").canonical_country.nunique().eq(2).all()
    assert c.groupby(["analysis_group","effect_id"]).canonical_country.nunique().le(5).all()
    assert set(f.effect_id)==set(figures.FINAL)
    assert f.attribution_role.eq("emitter").all() and f.resolution.eq("2016").all()


def test_final_country_selection_uses_reconstructed_parent_absolute_net():
    x=figures.validate(data(),"2016");_,_,f,_=figures.selections(x,n_country=1,n_effect=1,n_final=1)
    assert set(f.canonical_country)=={"A"}
    assert f.groupby("analysis_group").effect_id.nunique().eq(3).all()
