"""Diagnostic allocation within reported D_int; never a causal offshoring share."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
BASE=AN/'exhaustive_sourcing_20260910';B=AN/'sourcing_Dint_share_20260910';B.mkdir(exist_ok=True)
h=pd.read_csv(BASE/'all_hierarchy_annual.csv',dtype={'grid':str,'sector':str,'parent':str})
h=h[h.effect=='D_C'].copy()
hp=pd.read_csv(BASE/'all_hierarchy_period.csv',dtype={'grid':str,'sector':str,'parent':str})
hp=hp[hp.effect=='D_C']
e=pd.read_parquet(AN/'hierarchical_intensity_20260910/sector_Dint.parquet')
e.sector=e.sector.astype(str)
lookup=pd.read_csv(AN/'results_hypotheses_20260909/trade_inputs/country_levels_and_trade.csv')
lookup=lookup[(lookup.year==2000)&(lookup.price_basis=='CYP')][['grid','country','group']].copy()
lookup['resolution']=lookup.grid.replace({'WIOT2013':'2013','WIOT2016':'2016'})
e=e.merge(lookup[['resolution','country','group']],on=['resolution','country'],validate='many_to_one')
e['period']=e.year.map(lambda y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3')
e['Dint_Mt']=e.D_v/1e9
den=e[e.group=='Red'].groupby(['resolution','period']).agg(net_Dint_Mt=('Dint_Mt','sum'),
    gross_negative_Dint_Mt=('Dint_Mt',lambda x:-x[x<0].sum()),positive_Dint_Mt=('Dint_Mt',lambda x:x[x>0].sum())).reset_index()
# Independently reconcile net denominator to the registered group panel.
old=pd.read_csv(AN/'kyoto_period_group_products_20260908/kyoto_period_group_effects.csv',dtype={'resolution':str})
old=old[(old.group=='Red')&(old.role=='emitter')&(old.effect=='D_v')&old.configuration.str.startswith('exio')]
check=den.merge(old[['resolution','period','net']],on=['resolution','period'],validate='one_to_one')
assert len(check)==9
den_error=float((abs(check.net_Dint_Mt-check.net/1e9)/np.maximum(abs(check.net/1e9),1)).max());assert den_error<1e-10
den.to_csv(B/'group_Dint_denominators.csv',index=False)
# Use actual coarse bundle delta, not the approximately reconstructing fine delta.
cm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/fine_to_coarse_country_sector_review.csv',dtype=str)
country=cm[['coarse_resolution','fine_country','coarse_country']].drop_duplicates()
h=h.merge(country,left_on=['grid','country'],right_on=['coarse_resolution','fine_country'],validate='many_to_one')
ce=e[e.resolution!='R3'][['resolution','year','country','sector','bundle_v0','bundle_v1','Dint_Mt']].rename(columns={'country':'coarse_country','sector':'parent','bundle_v0':'actual_v0','bundle_v1':'actual_v1','Dint_Mt':'actual_Dint_Mt'})
h=h.merge(ce,left_on=['grid','year','coarse_country','parent'],right_on=['resolution','year','coarse_country','parent'],validate='many_to_one')
h['actual_delta_v']=h.actual_v1-h.actual_v0
h['stable_divisor']=abs(h.actual_delta_v)>1e-8*np.maximum(abs(h.actual_v0),1)
h['K_Mt_per_intensity']=np.where(h.stable_divisor,h.actual_Dint_Mt/h.actual_delta_v,np.nan)
h['selected_composition_Mt']=h['K_Mt_per_intensity']*h.centered_product_composition
h['parent_composition_Mt']=h['K_Mt_per_intensity']*h.composition
h['parent_within_Mt']=h['K_Mt_per_intensity']*h.within
h['mapping_residual_Mt']=h['K_Mt_per_intensity']*(h.actual_delta_v-h.delta_v)
keys=['grid','country','sector','parent','period']
h=h.merge(hp[keys+['repeated_ABCD','complete','comparable','period_AB','period_C','period_composition_negative','annual_ABCD_outside','outside_Mt']],on=keys,validate='many_to_one',suffixes=('','_period'))
h['repeated_outside']=h.repeated_ABCD & (h.annual_ABCD_outside>=2) & (h.outside_Mt>.1)
rows=[];details=[]
for scope in ['repeated','all_annual']:
 for foreign in ['all_foreign','outside_Reduction']:
  gate=(h.ABCD if foreign=='all_foreign' else h.ABCD_outside).copy()
  if scope=='repeated':gate &= h.repeated_ABCD if foreign=='all_foreign' else h.repeated_outside
  q=h[gate].copy();assert q.stable_divisor.all() and (q['K_Mt_per_intensity']>=0).all()
  q['scope']=scope;q['foreign_definition']=foreign
  details.append(q)
  for grid in ['2013','2016']:
   for period in ['P1','P2','P3']:
    z=q[(q.grid==grid)&(q.period==period)]
    # Centered product terms are additive within a parent. Count parent effects once.
    parents=z.drop_duplicates(['year','country','parent'])
    dv=den[(den.resolution==grid)&(den.period==period)].iloc[0]
    amount=-z.selected_composition_Mt.sum()
    rows.append(dict(scope=scope,foreign_definition=foreign,grid=grid,period=period,
        qualifying_product_years=len(z),qualifying_parent_years=len(parents),
        selected_composition_reduction_Mt=amount,group_net_Dint_Mt=dv.net_Dint_Mt,
        net_Dint_share_pct=100*amount/(-dv.net_Dint_Mt) if dv.net_Dint_Mt<0 else np.nan,
        gross_negative_Dint_share_pct=100*amount/dv.gross_negative_Dint_Mt,
        associated_parent_Dint_reduction_Mt=-parents.actual_Dint_Mt.sum()))
summary=pd.DataFrame(rows);summary.to_csv(B/'period_shares.csv',index=False)
detail=pd.concat(details,ignore_index=True);detail.to_csv(B/'allocated_product_years.csv',index=False)
assert (detail.selected_composition_Mt<0).all()
for foreign in ['all_foreign','outside_Reduction']:
 assert len(detail[(detail.scope=='all_annual')&(detail.foreign_definition==foreign)])>len(detail[(detail.scope=='repeated')&(detail.foreign_definition==foreign)])
case=detail.groupby(['scope','foreign_definition','grid','country','sector','parent','period']).agg(
    selected_composition_Mt=('selected_composition_Mt','sum'),qualifying_years=('year',lambda x:', '.join(map(str,sorted(x))))).reset_index()
case.to_csv(B/'case_allocations.csv',index=False)
reconstruction=float(abs(detail.actual_Dint_Mt-detail.parent_composition_Mt-detail.parent_within_Mt-detail.mapping_residual_Mt).max())
assert reconstruction<1e-9
# Check linear splitting against the actual implementation on nonuniform synthetic inputs.
path=ROOT/'implementation/bilateral_sda_attribution/src/sda/effects/intensity.py'
spec=importlib.util.spec_from_file_location('intensity',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(57);n=5;v0=rng.uniform(1,3,n);within=rng.normal(0,.1,n);comp=rng.normal(0,.2,n)
L0=np.eye(n)+rng.uniform(0,.2,(n,n));L1=np.eye(n)+rng.uniform(0,.2,(n,n));y0=rng.uniform(1,10,n);y1=rng.uniform(1,10,n)
effect=m.refined_laspeyres_intensity_effect
err=float(abs(effect(v0,v0+within+comp,L0,L1,y0,y1)-effect(v0,v0+within,L0,L1,y0,y1)-effect(v0,v0+comp,L0,L1,y0,y1)).max())
assert err<1e-12
v=dict(passed=True,indicator='GWP100 conditional',denominator_net_reconciliation_scaled_error=den_error,
    decomposition_reconstruction_max_error_Mt=reconstruction,synthetic_linearity_max_error=err,
    allocated_divisors_stable=True,allocated_exposures_nonnegative=True,
    max_mapping_residual_Mt=float(abs(detail.mapping_residual_Mt).max()),
    interpretation='Sourcing-associated centered composition allocation within reported D_int; not causal offshoring or a statistical bound',
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [path,BASE/'all_hierarchy_annual.csv',BASE/'all_hierarchy_period.csv',AN/'hierarchical_intensity_20260910/sector_Dint.parquet']])
(B/'validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
report=['# Percentage of Reduction-group D_int associated with sourcing/composition candidates','',
'Conditional GWP100 accounting diagnostic, R-2026-058. This is not a causal percentage of offshoring, a confidence interval or a share of legal target achievement.','',
'## Main result: repeated A-D cases','',
'The numerator includes only qualifying annual observations within the repeated, complete-period A-D candidates. It allocates the centered composition contribution of the selected high-intensity product, rather than attributing the entire parent intensity reduction to sourcing. The denominator is the negative of the net Reduction-group D_int in the same grid and period.','',
'| Grid | Period | Associated composition reduction (Mt CO2-eq) | Group net D_int reduction (Mt CO2-eq) | Share | Share excluding landfill |',
'| --- | --- | ---: | ---: | ---: | ---: |']
for r in summary[(summary.scope=='repeated')&(summary.foreign_definition=='all_foreign')].itertuples():
 landfill=-detail[(detail.scope=='repeated')&(detail.foreign_definition=='all_foreign')&(detail.grid==r.grid)&(detail.period==r.period)&(detail.sector=='94')].selected_composition_Mt.sum()
 share_no=100*(r.selected_composition_reduction_Mt-landfill)/(-r.group_net_Dint_Mt)
 report.append(f'| {r.grid}-compatible | {r.period} | {r.selected_composition_reduction_Mt:.3f} | {-r.group_net_Dint_Mt:.3f} | {r.net_Dint_share_pct:.3f}% | {share_no:.3f}% |')
report += ['',
'P1 is 2001-2004; P2 is 2005-2007; P3 is 2008-2012. Foreign includes other Reduction countries. The P3 numerator is UK plastics: approximately 1.736 Mt across four qualifying years. The 2016-compatible denominator is 765.118 Mt, giving 0.227%. This is a group-level share, not the percentage of the UK chemical-parent decline.','',
'The 2016-compatible P2 amount includes 7.731 Mt allocated to German landfill composition. Removing landfill leaves 1.522 Mt and a 0.312% share. Landfill services do not establish physical waste exports. The 2013-compatible result has different admissible mappings and group coverage; the two grids must not be averaged or presented as statistical bounds.','',
'## Foreign-source and persistence sensitivity','',
'| Grid | Period | Repeated, outside Reduction | All qualifying annual cases, all foreign | All annual, outside Reduction |',
'| --- | --- | ---: | ---: | ---: |']
for grid in ['2013','2016']:
 for per in ['P1','P2','P3']:
  vals=[]
  for scope,foreign in [('repeated','outside_Reduction'),('all_annual','all_foreign'),('all_annual','outside_Reduction')]:
   val=summary[(summary.grid==grid)&(summary.period==per)&(summary.scope==scope)&(summary.foreign_definition==foreign)].net_Dint_share_pct.iloc[0]
   vals.append(f'{abs(val):.3f}%')
  report.append('| '+ ' | '.join([grid,per]+vals)+' |')
report += ['',
'The broad annual screen is not an upper bound. It admits isolated or reversing episodes, unlike the repeated-period screen. German coke in 2005 alone contributes 83.129 Mt to the 2016-compatible P2 diagnostic; the recorded product output falls 95.75%. That episode requires input-data scrutiny and does not establish a stable relocation. A zero selected share means no admitted candidate, not proof of no offshoring.','',
'## Calculation and validation','',
'For emitter parent p and year t, the implemented refined-Laspeyres effect is D_int[p,t] = K[p,t] * delta_v[p,t]. K contains the fixed Leontief/final-demand terms. For stable nonzero delta_v, it is recovered from the reported effect divided by the actual bundle intensity change. The selected centered product composition term is c[i,p,t] = (mean(v_i) - mean(v_p)) * delta_w_i. Its diagnostic allocation is K[p,t] * c[i,p,t]. The reported percentage is minus the sum of selected allocations divided by minus the group net D_int, times 100.','',
'The ratio uses the actual coarse bundle delta, with a separately retained residual for the approximate fine-to-coarse reconstruction. Within + composition + this mapping residual reconstructs D_int. Distinct selected product terms are added within a parent; the parent effect is never counted twice. No D_C contribution is added to D_int: sourcing effects select observations, while the numerator is an allocation within D_int. No chained percentage is multiplied by a period effect.','',
'Implementation source: implementation/bilateral_sda_attribution/src/sda/effects/intensity.py, refined_laspeyres_intensity_effect. The denominator reconciles to the registered group panel within 4.23e-16 scaled error. Allocation reconstruction error is below 1e-12 Mt; selected divisors are stable and inferred K values nonnegative. A synthetic split test against the actual implementation confirms linearity.','',
'period_shares.csv also supplies a denominator based on gross negative emitter-sector-year D_int contributions, which avoids netting away positive offsets. R3 is retained in group_Dint_denominators.csv for context, but coarse composition allocations must not be divided by an R3 denominator.','',
'Interpretation remains conditional: 58 products have comparable parent diagnostics, and allocation among products uses the stated centered convention. The indicator mismatch and unresolved consumed-input provenance remain. These estimates quantify an accounting pattern under explicit selection rules; they do not identify how much intensity reduction would have occurred without offshoring.','',
'Reproduce with quantify_sourcing_Dint_share.py. All allocated product-years, case totals, denominators and validation hashes are retained in this directory.']
(B/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
print(summary.to_string(index=False));print(json.dumps({k:v for k,v in v.items() if k!='sources'},indent=2))
