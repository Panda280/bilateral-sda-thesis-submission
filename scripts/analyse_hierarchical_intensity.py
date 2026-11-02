"""Conditional fine/coarse composition diagnostic for the stored SDA indicator.

The coefficient decomposition is not an additive SDA subeffect. Source and
emitter partitions of D_C are never added to one another or to this diagnostic.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
B=AN/'hierarchical_intensity_20260910'
l=pd.read_parquet(B/'sector_levels.parquet');e=pd.read_parquet(B/'sector_Dint.parquet')
keys=['resolution','year','country','sector']
d=e.copy()
for side,basis in [(0,'CYP'),(1,'PYP')]:
 q=l[l.basis==basis].copy();q.year+=1-side
 q=q[keys+['x','u']].rename(columns={'x':f'x{side}','u':f'fossil_u{side}'})
 d=d.merge(q,on=keys,validate='one_to_one')
d['period']=d.year.map(lambda y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3')
lookup=pd.read_csv(AN/'results_hypotheses_20260909/trade_inputs/country_levels_and_trade.csv')
lookup=lookup[(lookup.year==2000)&(lookup.price_basis=='CYP')][['grid','country','group']]
lookup['resolution']=lookup.grid.replace({'WIOT2013':'2013','WIOT2016':'2016'})
d=d.merge(lookup[['resolution','country','group']],on=['resolution','country'],validate='many_to_one')
assert len(d)==len(e)
p=pd.read_parquet(ROOT/'data/processed/provisional_panel_v2/cross_year_panel/emitter_country.parquet')
p=p[p.effect_id.eq('D_v')&p.year_current.between(2001,2012)]
chk=d.groupby(['resolution','year','country']).D_v.sum().to_frame('extracted').join(p.rename(columns={'year_current':'year','label':'country'}).set_index(['resolution','year','country']).contribution)
chk['error']=abs(chk.extracted-chk.contribution)/np.maximum(abs(chk.contribution),1e9)
assert chk.contribution.notna().all() and chk.error.max()<1e-5
chk.to_csv(B/'effect_reconciliation.csv')
sm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/granularity_sector_mapping_review.csv',dtype=str)
cm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/fine_to_coarse_country_sector_review.csv',dtype=str)
coarse=d[d.resolution!='R3'].set_index(keys)
records=[];cells=[]
for grid in ['2013','2016']:
 m=sm[sm.coarse_resolution==grid]
 excluded=set(t.strip() for s in m.loc[m.mapping_status!='reviewed_unique','candidate_coarse_sectors'] for t in s.split('|'))
 c=cm[(cm.coarse_resolution==grid)&cm.mapping_status.eq('reviewed_unique')&~cm.coarse_sector.isin(excluded)]
 f=d[d.resolution=='R3'].merge(c,left_on=['country','sector'],right_on=['fine_country','fine_sector'],validate='many_to_one')
 for key,q in f.groupby(['year','period','coarse_country','coarse_sector']):
  year,period,cc,cs=key
  if q.country.nunique()!=1 or q.sector.nunique()<2:continue  # isolate sector aggregation
  ccrow=coarse.loc[(grid,year,cc,cs)]
  if (q[['x0','x1']]<=0).any().any():continue  # zero/negative-output weights not inferred
  x0=q.x0.sum();x1=q.x1.sum();w0=q.x0/x0;w1=q.x1/x1
  v0=q.bundle_v0;v1=q.bundle_v1
  gv0=float((w0*v0).sum());gv1=float((w1*v1).sum())
  within=float(((w0+w1)/2*(v1-v0)).sum());composition=float(((v0+v1)/2*(w1-w0)).sum())
  identity=abs(gv1-gv0-within-composition)/max(abs(gv0),abs(gv1),1)
  assert identity<1e-10
  xe=max(abs(x0-ccrow.x0)/max(abs(ccrow.x0),1),abs(x1-ccrow.x1)/max(abs(ccrow.x1),1))
  ve=max(abs(gv0-ccrow.bundle_v0)/max(abs(ccrow.bundle_v0),1),abs(gv1-ccrow.bundle_v1)/max(abs(ccrow.bundle_v1),1))
  complete=xe<1e-5 and ve<1e-5
  record=dict(grid=grid,year=year,period=period,country=q.country.iloc[0],coarse_country=cc,group=q.group.iloc[0],parent=cs,n_fine=len(q),
    v0=gv0,v1=gv1,delta_v=gv1-gv0,within=within,composition=composition,composition_relative=composition/gv0 if gv0 else np.nan,
    within_relative=within/gv0 if gv0 else np.nan,coarse_Dint_Mt=ccrow.D_v/1e9,fine_Dint_Mt=q.D_v.sum()/1e9,
    coarse_minus_fine_Dint_Mt=(ccrow.D_v-q.D_v.sum())/1e9,x_error=xe,v_error=ve,identity_error=identity,comparable=complete)
  records.append(record)
  for i,r in q.iterrows():
   cells.append(dict(grid=grid,year=year,period=period,country=r.country,group=r.group,parent=cs,sector=r.sector,
    initial_v=r.bundle_v0,parent_v=gv0,above_parent=r.bundle_v0>gv0,share0=w0.loc[i],share1=w1.loc[i],share_change=w1.loc[i]-w0.loc[i],
    Dint_Mt=r.D_v/1e9,composition_cell=(r.bundle_v0+r.bundle_v1)/2*(w1.loc[i]-w0.loc[i]),comparable=complete))
r=pd.DataFrame(records);f=pd.DataFrame(cells)
r.to_csv(B/'parent_annual_diagnostics.csv',index=False);f.to_csv(B/'fine_cell_diagnostics.csv',index=False)
summ=[]
for k,q in r[r.comparable].groupby(['grid','period','group']):
 summ.append(dict(zip(['grid','period','group'],k),parent_years=len(q),
   declining_intensity=int((q.delta_v<0).sum()),composition_lowering=int((q.composition<0).sum()),
   both=int(((q.delta_v<0)&(q.composition<0)).sum()),
   median_composition_relative=q.composition_relative.median(),coarse_Dint_Mt=q.coarse_Dint_Mt.sum(),fine_Dint_Mt=q.fine_Dint_Mt.sum()))
pd.DataFrame(summ).to_csv(B/'parent_summary.csv',index=False)
# P3 joint sourcing slices already validated for R-2026-051. Their product
# axis is a source product, not the using sector of a coke/steel transaction.
t=pd.read_parquet(AN/'ten_sda_candidates_20260910/country_product_emitter_sourcing.parquet')
t=t[t.effect=='D_C'].copy();t.source_sector=t.source_sector.astype(str)
src=t.groupby(['year','destination_country','source_sector','source_group']).value.sum().unstack('source_group').reset_index()
em=t.groupby(['year','destination_country','source_sector','emitter_scope']).value.sum().unstack('emitter_scope').reset_index()
t=src.merge(em,on=['year','destination_country','source_sector'],validate='one_to_one')
t['outside_source']=t['Non-Red']+t['Non-Participant'];t['outside_emitter']=t['Foreign Non-Red']+t['Foreign Non-Participant']
f=f[(f.group=='Red')&(f.period=='P3')&f.comparable]
j=f.merge(t,left_on=['year','country','sector'],right_on=['year','destination_country','source_sector'],validate='many_to_one')
j=j.merge(r[['grid','year','country','parent','delta_v','composition','coarse_Dint_Mt']],on=['grid','year','country','parent'],validate='many_to_one')
j['signature']=(j.above_parent&(j.share_change<0)&(j.delta_v<0)&(j.composition<0)&(j.coarse_Dint_Mt<0)&(j.Domestic<-1e8)&(j.outside_source>1e8)&(j.Home<-1e8)&(j.outside_emitter>1e8))
j.to_csv(B/'P3_joint_sourcing_tests.csv',index=False)
cases=j.groupby(['grid','country','parent','sector']).agg(years=('year','size'),passes=('signature','sum'),
    domestic_Mt=('Domestic',lambda x:x.sum()/1e9),outside_source_Mt=('outside_source',lambda x:x.sum()/1e9),
    home_Mt=('Home',lambda x:x.sum()/1e9),outside_emitter_Mt=('outside_emitter',lambda x:x.sum()/1e9)).reset_index()
cases.to_csv(B/'P3_case_persistence.csv',index=False)
validation=dict(passed=True,matched_effect_rows=len(d),effect_max_scaled_error=float(chk.error.max()),parent_years=len(r),
    comparable_parent_years=int(r.comparable.sum()),not_comparable_parent_years=int((~r.comparable).sum()),
    max_identity_error=float(r.identity_error.max()),P3_signature_rows=int(j.signature.sum()),P3_cases_two_years=int((cases.passes>=2).sum()),
    limits=['Stored indicator; fossil-CO2 labels are under audit.','Composition identity is a coefficient diagnostic, not a reported SDA effect.',
      'Only one-country, multiple-fine-sector parents with positive fine outputs and matched aggregate outputs/intensities are admitted.',
      'Existing P3 source-product D_C slices sum over using sectors; no feed-to-livestock or coke-to-steel bilateral using-sector attribution is inferred.',
      '0.1 Mt sourcing gates are materiality screens, not statistical significance.','No inference of low production cost from intensity per monetary output.'])
(B/'hierarchy_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps(validation,indent=2));print(cases[cases.passes>=2].sort_values('passes',ascending=False).to_string(index=False))
