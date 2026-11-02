"""Scope, lag and materiality sensitivities kept separate from primary results."""
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
def main():
    a=pd.read_csv(BASE/'trade/annual_country_changes.csv');a=a[~a.aggregate_region].copy()
    common=set.intersection(*[set(q.canonical_country) for _,q in a.groupby('grid')]);assert 'ROW' not in common
    records=[]
    for grid,q in a.groupby('grid'):
      for design in ['common_countries','one_year_lag']:
        if design=='common_countries':q2=q[q.canonical_country.isin(common)].copy();x='delta_imported_footprint_Mt'
        else:
          q2=q.sort_values(['country','year']).copy();q2['lag_imported_change']=q2.groupby('country').delta_imported_footprint_Mt.shift(1)
          q2=q2.dropna(subset=['lag_imported_change']);x='lag_imported_change'
        model=smf.ols(f'delta_territorial_Mt ~ {x} + C(country) + C(year)',q2).fit(cov_type='cluster',cov_kwds={'groups':q2.country},use_t=True)
        ci=model.conf_int().loc[x]
        records.append({'grid':grid,'design':design,'countries':q2.country.nunique(),'n':len(q2),'slope':model.params[x],'cluster_se':model.bse[x],'ci_low':ci.iloc[0],'ci_high':ci.iloc[1]})
    pd.DataFrame(records).to_csv(BASE/'trade/additional_sensitivity.csv',index=False)
    a=pd.read_csv(BASE/'trade/annual_crop_livestock_changes.csv');a=a[~a.destination_country.isin(['WA','WE','WF','WL','WM','ROW','RoW','WT'])]
    records=[]
    for floor in [.1,1,10]:
      for p,q in a.groupby('period'):
        mask=(q.change_domestic < -floor)&(q.change_cross_border>floor)&(q.change_import_share>0)&(q.total_base>0)&(q.total_t>0)
        records.append({'period':p,'floor_million_EUR':floor,'candidate_count':int(mask.sum()),'total_observations':len(q)})
    pd.DataFrame(records).to_csv(BASE/'trade/input_signature_thresholds.csv',index=False)
    l=pd.read_csv(BASE/'trade_inputs/crop_livestock_links.csv')
    l=l[(l.destination_country=='DE')&(l.using_sector==4)&(l.input_sector==1)&(l.price_basis=='CYP')]
    l=l.groupby(['year','relationship']).million_EUR.sum().unstack().reset_index();l['import_share']=l.cross_border/(l.cross_border+l.domestic)
    l.to_csv(BASE/'trade/germany_cereal_cattle_CYP_sensitivity.csv',index=False)
    print(pd.read_csv(BASE/'trade/additional_sensitivity.csv').to_string(index=False));print(pd.DataFrame(records).to_string(index=False))
if __name__=='__main__':main()
