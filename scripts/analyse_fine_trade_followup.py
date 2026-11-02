"""Sector-matched descriptive displacement screens; no causal estimates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from build_results_trade_associations import estimate
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
OUT=BASE/'followup'
REGIONS={'WA','WE','WF','WL','WM','ROW','RoW','WT'}

def main():
    raw=pd.read_csv(BASE/'fine_followup/sector_levels.csv',dtype={'sector':str})
    assert not raw.duplicated(['year','price_basis','country','sector']).any()
    assert np.isfinite(raw[['territorial_kg_CO2','direct_product_imports_million_EUR']]).all().all()
    q=raw[raw.price_basis.eq('CYP')].sort_values(['country','sector','year']).copy()
    for old,new in [('territorial_kg_CO2','delta_territorial_Mt'),('effective_territorial_kg_CO2','delta_effective_territorial_Mt'),('imported_emitting_sector_kg_CO2','delta_imported_footprint_Mt')]:q[new]=q.groupby(['country','sector'])[old].diff()/1e9
    prev=raw[raw.price_basis.eq('CYP')][['year','country','sector','direct_product_imports_million_EUR']].copy();prev.year+=1
    exp=raw[raw.price_basis.eq('PYP')].merge(prev,on=['year','country','sector'],suffixes=('_t','_base'),validate='one_to_one')
    exp['delta_import_expenditure_billion_EUR']=(exp.direct_product_imports_million_EUR_t-exp.direct_product_imports_million_EUR_base)/1000
    q=q[q.year.ge(2001)].merge(exp[['year','country','sector','delta_import_expenditure_billion_EUR']],on=['year','country','sector'],validate='one_to_one')
    q['period']=q.year.map(lambda y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3')
    q.to_csv(OUT/'annual_sector_changes.csv',index=False)
    # Reconcile detailed levels and changes to the prior country audit.
    old=pd.read_csv(BASE/'trade/annual_country_changes.csv');old=old[old.grid.eq('R3')]
    sums=q.groupby(['year','country'])[['delta_territorial_Mt','delta_imported_footprint_Mt','delta_import_expenditure_billion_EUR']].sum()
    joined=sums.join(old.set_index(['year','country'])[sums.columns],rsuffix='_old')
    residuals={c:float(abs(joined[c]-joined[c+'_old']).max()) for c in sums}
    assert max(residuals.values())<1e-5
    nat=q[~q.country.isin(REGIONS)].copy()
    labels=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/sector_concordance_by_resolution.csv',dtype=str)
    labels=labels[labels.resolution.eq('R3')].set_index('observed_sector').sector_label.to_dict()
    results=[]
    for s,sec in nat.groupby('sector'):
      for scope,d in [('All countries',sec),('Reduction',sec[sec.group.eq('Red')])]:
       for x in ['delta_imported_footprint_Mt','delta_import_expenditure_billion_EUR']:
        if d[x].std()<1e-12 or d.delta_territorial_Mt.std()<1e-12:continue
        e=estimate(d,x)
        e['p_value']=2*stats.t.sf(abs(e['two_way_slope']/e['cluster_se']),e['countries']-1)
        results.append(dict(sector=s,label=labels[s],scope=scope,x=x,**e))
    r=pd.DataFrame(results)
    for _,ii in r.groupby(['scope','x']).groups.items():r.loc[ii,'q_BH']=multipletests(r.loc[ii,'p_value'],method='fdr_bh')[1]
    r.to_csv(OUT/'sector_associations.csv',index=False)
    # Report the full threshold inventory, not only successful examples.
    counts=[]
    for scope,d in [('All countries',nat),('Reduction',nat[nat.group.eq('Red')])]:
      for floor in [.01,.1,1.]:
        for period,g in d.groupby('period'):
          event=(g.delta_territorial_Mt<-floor)&(g.delta_imported_footprint_Mt>floor)
          counts.append(dict(scope=scope,period=period,floor_Mt=floor,observations=len(g),events=int(event.sum()),also_real_product_import_increase=int((event&(g.delta_import_expenditure_billion_EUR>.01)).sum())))
    pd.DataFrame(counts).to_csv(OUT/'sector_event_counts.csv',index=False)
    events=nat[(nat.delta_territorial_Mt<-.1)&(nat.delta_imported_footprint_Mt>.1)].copy()
    events['matched_offset_Mt']=np.minimum(-events.delta_territorial_Mt,events.delta_imported_footprint_Mt)
    events['label']=events.sector.map(labels)
    events.sort_values('matched_offset_Mt',ascending=False).to_csv(OUT/'sector_episode_candidates.csv',index=False)
    # Group-period aggregation of the SAME fine calculation isolates hidden
    # country/sector differences without changing the source grid.
    # Do not sum adjacent-year monetary changes across different price years.
    gp=q.groupby(['group','period','sector'])[['delta_territorial_Mt','delta_imported_footprint_Mt']].sum().reset_index()
    gp['label']=gp.sector.map(labels);gp.to_csv(OUT/'group_period_sector_changes.csv',index=False)
    origins=pd.read_csv(BASE/'fine_followup/sector_import_origins.csv',dtype={'sector':str}).sort_values(['country','sector','source_group','year'])
    origins['delta_imported_Mt']=origins.groupby(['country','sector','source_group']).imported_kg_CO2.diff()/1e9
    origins[origins.year.ge(2001)].to_csv(OUT/'annual_sector_import_origins.csv',index=False)
    (OUT/'fine_validation.json').write_text(json.dumps(dict(country_reconciliation_max_residuals=residuals,passed=True),indent=2))
    print('NEGATIVE SECTOR SLOPES, BH q<.05:')
    print(r[(r.two_way_slope<0)&(r.q_BH<.05)].to_string(index=False))
    print('REDUCTION GROUP PERIOD SECTOR EVENTS:')
    print(gp[(gp.group=='Red')&(gp.delta_territorial_Mt<-.1)&(gp.delta_imported_footprint_Mt>.1)].to_string(index=False))
    print('TOP REDUCTION COUNTRY EPISODES:')
    print(events[events.group.eq('Red')].sort_values('matched_offset_Mt',ascending=False)[['year','country','sector','label','delta_territorial_Mt','delta_imported_footprint_Mt','delta_import_expenditure_billion_EUR']].head(20).to_string(index=False))
if __name__=='__main__':main()
