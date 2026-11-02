"""Descriptive grid test using existing D_v and matched country baseline levels.

Country-average intensity is not sector intensity. No numerator/denominator
subeffects or causal offshoring estimates are introduced.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
OUT=AN/'intensity_grid_hypothesis_20260910'
OUT.mkdir(parents=True,exist_ok=True)
period=lambda y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3'
levels=AN/'results_hypotheses_20260909/trade_inputs/country_levels_and_trade.csv'
panel=ROOT/'data/processed/provisional_panel_v2/cross_year_panel/emitter_country.parquet'
groups=AN/'kyoto_period_group_products_20260908/kyoto_period_group_effects.csv'
l=pd.read_csv(levels); l=l[(l.price_basis=='CYP')&l.year.between(2000,2011)].copy()
l['resolution']=l.grid.replace({'WIOT2013':'2013','WIOT2016':'2016'})
l['year_current']=l.year+1;l=l.rename(columns={'country':'label'})
p=pd.read_parquet(panel);p=p[(p.effect_id=='D_v')&p.year_current.between(2001,2012)].copy()
assert p.is_publishable.all() and p.validation_status.eq('passed').all() and p.units.eq('kg CO2').all()
d=p.merge(l,on=['resolution','year_current','label'],validate='one_to_one',how='left')
assert d.output_million_EUR.notna().all()
d['period']=d.year_current.map(period)
d['baseline_intensity']=d.territorial_kg_CO2/d.output_million_EUR
d['reduction_Mt']=-d.contribution/1e9
d['reduction_relative_baseline']=-d.contribution/d.territorial_kg_CO2
d['valid']=np.isfinite(d.baseline_intensity)&(d.output_million_EUR>0)&(d.territorial_kg_CO2>0)
# Reconcile every grid-period-group to the independently registered emitter product.
r=pd.read_csv(groups,dtype={'resolution':str});r=r[(r.role=='emitter')&(r.effect=='D_v')]
q=d.groupby(['resolution','period','group']).contribution.sum().rename('reconstructed')
check=r.set_index(['resolution','period','group'])[['net']].join(q)
assert check.reconstructed.notna().all()
check['relative_error']=abs(check.net-check.reconstructed)/np.maximum(abs(check.net),1e9)
assert check.relative_error.max()<1e-9
check.to_csv(OUT/'group_reconciliation.csv')
# Annual within-group ranks avoid comparing nominal intensity levels across years.
d['intensity_rank']=d.groupby(['resolution','year_current','group']).baseline_intensity.rank(pct=True)
d['high']=d.intensity_rank>.75
annual=[]
for k,z in d[d.valid].groupby(['resolution','period','year_current','group']):
    rho_abs=z.baseline_intensity.rank().corr(z.reduction_Mt.rank())
    rho_rel=z.baseline_intensity.rank().corr(z.reduction_relative_baseline.rank())
    hi=z[z.high];lo=z[~z.high]
    annual.append(dict(zip(['resolution','period','year','group'],k),n=len(z),
        rho_absolute=rho_abs,rho_relative=rho_rel,
        high_mean_relative=hi.reduction_relative_baseline.mean(),
        lower_mean_relative=lo.reduction_relative_baseline.mean(),
        high_net_Dint_Mt=hi.contribution.sum()/1e9,lower_net_Dint_Mt=lo.contribution.sum()/1e9))
a=pd.DataFrame(annual);a.to_csv(OUT/'annual_rank_tests.csv',index=False)
# Common country coverage and fixed period-start intensity ranks are separate
# sensitivity checks, not replacements selected for a stronger association.
coverage=d.groupby('canonical_country').resolution.nunique()
common=set(coverage[coverage==3].index)
sens=[]
for scope,dd in [('all',d),('common_countries',d[d.canonical_country.isin(common)])]:
 for k,z in dd.groupby(['resolution','period','group']):
    start=z.year_current.min();b=z[z.year_current==start].set_index('label').baseline_intensity
    high=set(b[b.rank(pct=True)>.75].index)
    yy=[]
    for year,v in z.groupby('year_current'):
        initial=v.label.map(b)
        yy.append(dict(rho=initial.rank().corr(v.reduction_relative_baseline.rank()),
          gap=v.loc[v.label.isin(high),'reduction_relative_baseline'].mean()-v.loc[~v.label.isin(high),'reduction_relative_baseline'].mean()))
    sens.append(dict(zip(['resolution','period','group'],k),scope=scope,countries=len(b),
       fixed_start_mean_rho=float(np.mean([v['rho'] for v in yy])),
       fixed_start_high_minus_lower=float(np.mean([v['gap'] for v in yy]))))
pd.DataFrame(sens).to_csv(OUT/'fixed_start_common_country_sensitivity.csv',index=False)
s=a.groupby(['resolution','period','group']).agg(years=('year','size'),
    mean_rho_absolute=('rho_absolute','mean'),mean_rho_relative=('rho_relative','mean'),
    positive_relative_years=('rho_relative',lambda x:int((x>0).sum())),
    high_mean_relative=('high_mean_relative','mean'),lower_mean_relative=('lower_mean_relative','mean'),
    high_net_Dint_Mt=('high_net_Dint_Mt','sum'),lower_net_Dint_Mt=('lower_net_Dint_Mt','sum')).reset_index()
s['high_minus_lower_relative']=s.high_mean_relative-s.lower_mean_relative
s.to_csv(OUT/'period_rank_tests.csv',index=False)
d.to_csv(OUT/'matched_country_observations.csv',index=False)
g=check.reset_index();g['Dint_Mt']=g.net/1e9
grid=g.pivot(index=['group','period'],columns='resolution',values='Dint_Mt')
grid['R3_minus_2016_Mt']=grid.R3-grid['2016'];grid['R3_minus_2013_Mt']=grid.R3-grid['2013']
grid.to_csv(OUT/'group_grid_comparison.csv')
manifest=dict(passed=True,matched_rows=len(d),invalid_rows=int((~d.valid).sum()),
    group_reconciliation_max_relative_error=float(check.relative_error.max()),
    definition='Baseline country-average direct emissions/output in CYP(t-1); D_v for CYP(t-1)->PYP(t). Positive reduction variables mean more negative D_v.',
    specification='Within each year/grid/group, compare upper quartile of country baseline intensity with remaining countries; equal-country means and Spearman rank correlations; annual diagnostics averaged within period. No p-values or causal interpretation.',
    limitations=['Country-average intensity incorporates sector composition; not a high/low sector-intensity test.',
      'Baseline-emission normalization mitigates scale but creates a shared numerator/denominator relationship; association is descriptive.',
      'Grid changes country and sector resolution together; matched aggregate effects cannot identify offshoring.',
      'Sector output and initial intensity are not retained locally; remote host unreachable during this test.',
      'Unresolved balanced-versus-consumed input provenance applies.'],
    sources=[dict(path=str(x.relative_to(ROOT)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in [levels,panel,groups]])
(OUT/'validation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(grid.to_string());print(s[s.group=='Red'].to_string(index=False));print(json.dumps({k:v for k,v in manifest.items() if k in ['passed','matched_rows','invalid_rows','group_reconciliation_max_relative_error']}))
