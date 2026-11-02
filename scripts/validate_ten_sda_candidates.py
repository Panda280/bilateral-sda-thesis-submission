"""Independent selected-slice identities and annual-influence diagnostics."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/ten_sda_candidates_20260910'
def main():
    s=pd.read_csv(BASE/'ten_candidates.csv',dtype={'source_sector':str})
    a=pd.read_csv(BASE/'annual_country_product_cases.csv',dtype={'source_sector':str})
    q=pd.read_parquet(BASE/'country_product_emitter_sourcing.parquet');q.source_sector=q.source_sector.astype(str);q.value/=1e9
    assert len(s)==10 and not s.duplicated(['effect','destination_country','source_sector']).any()
    parts=[];annual=[];checks=[];influence=[]
    for c in s.itertuples():
      raw=q[(q.effect==c.effect)&(q.destination_country==c.destination_country)&(q.source_sector==c.source_sector)]
      assert len(raw)==5*4*4
      home=raw[raw.emitter_scope=='Home'].value.sum();foreign=raw[raw.emitter_scope!='Home'].value.sum()
      red=raw[raw.emitter_scope.isin(['Home','Foreign Red'])].value.sum();non=raw[~raw.emitter_scope.isin(['Home','Foreign Red'])].value.sum()
      domestic=raw[raw.source_group=='Domestic'].value.sum()
      for label,value,expected in [('Home',home,c.Home),('Foreign',foreign,c.Foreign),('Red',red,c.ReductionEmitter),('Outside',non,c.OutsideReductionEmitter),('Domestic',domestic,c.Domestic)]:
        assert abs(value-expected)<1e-8;checks.append(dict(case=c.case_id,quantity=label,residual_Mt=value-expected))
      parts.append(raw.assign(case_id=c.case_id))
      yr=a[(a.effect==c.effect)&(a.destination_country==c.destination_country)&(a.source_sector==c.source_sector)].copy()
      assert len(yr)==5 and yr.year.nunique()==5;yr['case_id']=c.case_id;annual.append(yr)
      for y in yr.itertuples():
        v=yr[yr.year!=y.year][['Home','Foreign','Domestic','Non-Participant','CrossGroup','SameGroup']].sum()
        positive=(v.Home<0)&(v.Foreign>0)&(v.Domestic<0)&(v['Non-Participant']>0)&(v.CrossGroup>0)&(v.SameGroup<0)
        material=(v.Home<-1)&(v.Foreign>1)&(v.Domestic<-1)&(v['Non-Participant']>1)&(v.CrossGroup>1)&(v.SameGroup<-1)
        influence.append(dict(case_id=c.case_id,omitted_year=y.year,sign_retained=bool(positive),material_1Mt_retained=bool(material),**v.to_dict()))
    combined=pd.concat(parts);assert not combined.duplicated(['year','effect','destination_country','source_sector','source_group','emitter_scope']).any()
    agg=combined.groupby('emitter_scope').value.sum()
    totals=dict(Home=agg['Home'],Other_Red=agg['Foreign Red'],Non_Red=agg['Foreign Non-Red'],Non_Participant=agg['Foreign Non-Participant'])
    totals['Reduction']=totals['Home']+totals['Other_Red'];totals['Outside_Reduction']=totals['Non_Red']+totals['Non_Participant'];totals['World']=sum(agg)
    pd.DataFrame(checks).to_csv(BASE/'selected_slice_checks.csv',index=False)
    pd.concat(annual).to_csv(BASE/'selected_annual_paths.csv',index=False)
    inf=pd.DataFrame(influence);inf.to_csv(BASE/'leave_one_year_case_checks.csv',index=False)
    stability=inf.groupby('case_id').agg(sign_stable=('sign_retained','all'),material_stable=('material_1Mt_retained','all')).reset_index()
    s=s.merge(stability,on='case_id',validate='one_to_one');s.to_csv(BASE/'ten_candidates_validated.csv',index=False)
    (BASE/'selected_totals.json').write_text(json.dumps(totals,indent=2))
    intro=hashlib.sha256((ROOT/'src/chapters/01_introduction.tex').read_bytes()).hexdigest()
    assert intro=='2ea30046ae703d836a7256a5cb993606d4a593befe048807e324d5ce9fe68b43'
    report=dict(passed=True,cases=10,unique_nonoverlapping=True,selected_slice_max_abs_error_Mt=max(abs(c['residual_Mt']) for c in checks),
        sign_stable_cases=int(s.sign_stable.sum()),material_stable_cases=int(s.material_stable.sum()),two_years_at_1Mt=int((s.persistent_1>=2).sum()),introduction_sha256=intro,
        totals=totals,files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(BASE.glob('*.csv'))])
    (BASE/'final_numerical_validation.json').write_text(json.dumps(report,indent=2))
    print(s[['case_id','destination_country','source_sector','persistent_01','persistent_1','sign_stable','material_stable']].to_string(index=False));print(totals)
if __name__=='__main__':main()
