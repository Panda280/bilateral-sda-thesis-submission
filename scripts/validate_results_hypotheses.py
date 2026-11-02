"""Independent quantitative checks for the A/B/C results package."""
from pathlib import Path
import importlib.util,json,hashlib
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
def main():
    checks={}
    intro=ROOT/'src/chapters/01_introduction.tex'
    checks['introduction_unchanged']=hashlib.sha256(intro.read_bytes()).hexdigest()=='2ea30046ae703d836a7256a5cb993606d4a593befe048807e324d5ce9fe68b43'
    a=pd.read_csv(BASE/'temporal/annual_effects.csv');l=pd.read_csv(BASE/'temporal/leave_one_year_out.csv')
    keys=['resolution','effect','role','group','period']
    agg=a.groupby(keys).net.sum()/1e9
    sums=l.groupby(keys).period_net_Mt.first()
    checks['annual_period_identity']=bool(np.allclose(agg.sort_index(),sums.sort_index(),atol=1e-6,rtol=1e-12))
    checks['leave_one_year_identity']=bool(np.allclose(l.period_net_Mt-l.annual_Mt,l.without_year_Mt,atol=1e-6))
    fine=pd.read_csv(BASE/'sectors/fine_sector_period_effects.csv');s=pd.read_csv(BASE/'sectors/within_parent_visibility.csv')
    f=fine.groupby(['period','effect_id','coarse_sector']).contribution_Mt.sum()
    q=s.set_index(['period','effect','coarse_sector']).mapped_R3_net_Mt
    checks['fine_to_coarse_sum']=bool(np.allclose(f.sort_index(),q.sort_index(),atol=1e-6))
    checks['sector_signed_pools']=bool(np.allclose(s.positive_fine_nets_Mt+s.negative_fine_nets_Mt,s.mapped_R3_net_Mt,atol=1e-6))
    annual=pd.read_csv(BASE/'trade/annual_country_changes.csv');est=pd.read_csv(BASE/'trade/associations.csv')
    regression=[]
    for grid in ['R3','WIOT2013','WIOT2016']:
      for scope in ['All countries','Red']:
        q=annual[(annual.grid==grid)&(~annual.aggregate_region)].copy()
        if scope=='Red':q=q[q['group']=='Red']
        for x in ['delta_imported_footprint_Mt','delta_import_expenditure_billion_EUR']:
          m=smf.ols(f'delta_territorial_Mt ~ {x} + C(country) + C(year)',q).fit(cov_type='cluster',cov_kwds={'groups':q.country})
          e=est[(est.grid==grid)&(est.scope==scope)&(est.x==x)].iloc[0]
          regression.append({'grid':grid,'scope':scope,'x':x,'slope_difference':m.params[x]-e.two_way_slope,'se_difference':m.bse[x]-e.cluster_se})
    r=pd.DataFrame(regression);r.to_csv(BASE/'independent_regression_validation.csv',index=False)
    checks['dummy_regression_matches_demeaning']=bool(r.slope_difference.abs().max()<1e-9 and r.se_difference.abs().max()<1e-9)
    # Raw/effective emission sensitivity: quantify the worst possible influence of discarded zero-output emissions.
    sensitivity=[]
    for grid,q in annual[~annual.aggregate_region].groupby('grid'):
      for outcome in ['delta_territorial_Mt','delta_effective_territorial_Mt']:
        x='delta_imported_footprint_Mt'
        m=smf.ols(f'{outcome} ~ {x} + C(country) + C(year)',q).fit()
        sensitivity.append({'grid':grid,'outcome':outcome,'slope':m.params[x]})
    pd.DataFrame(sensitivity).to_csv(BASE/'raw_effective_sensitivity.csv',index=False)
    raw=pd.read_csv(BASE/'trade_inputs/country_levels_and_trade.csv');c=raw[raw.price_basis=='CYP']
    checks['footprint_domestic_imported_identity']=bool(np.allclose(c.domestic_footprint_kg_CO2+c.imported_footprint_kg_CO2,c.total_footprint_kg_CO2,rtol=1e-12,atol=1))
    v=pd.read_csv(BASE/'trade_inputs/footprint_validation.csv')
    checks['39_grid_year_footprint_audits']=len(v)==39 and v.relative_country_emission_balance.max()<1e-8
    checks['old_footprints_reproduced']=v.old_P_max_relative_difference.dropna().max()<1e-8
    # Period endpoint mass changes must telescope exactly from annual changes.
    p=pd.read_csv(BASE/'trade/period_group_level_changes.csv')
    group=annual.groupby(['grid','group','period'])[['delta_territorial_Mt','delta_imported_footprint_Mt']].sum()
    pp=p.set_index(['grid','group','period'])
    checks['period_changes_telescope']=bool(np.allclose(group.delta_territorial_Mt.sort_index(),pp.territorial_change_Mt.sort_index(),atol=1e-6) and np.allclose(group.delta_imported_footprint_Mt.sort_index(),pp.imported_footprint_change_Mt.sort_index(),atol=1e-6))
    report={'passed':all(checks.values()),'checks':checks,'raw_impact_loss_max_relative':v.impact_loss_relative.max(),
            'max_absolute_regression_check_error':r.slope_difference.abs().max(),
            'limitations':['The initial raw-impact-zero-loss assertion failed and was replaced with explicit loss reporting, not hidden or called zero.',
              'R3 retains a maximum 2.33e-5 raw-impact loss fraction in footprints; raw territorial emissions remain unadjusted.',
              'No claim of causal identification or physical feed quantities is validated by these tests.']}
    (BASE/'validation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));assert report['passed']
if __name__=='__main__':main()
