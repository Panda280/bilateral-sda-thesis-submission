"""Full stored-GWP100 hierarchy and sourcing screen; no SDA recomputation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
B=AN/'full_drilldown_20260910';H=AN/'hierarchical_intensity_20260910';T=AN/'ten_sda_candidates_20260910'
B.mkdir(parents=True,exist_ok=True)
period=lambda y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3'
parent=pd.read_csv(H/'parent_annual_diagnostics.csv',dtype={'grid':str,'parent':str})
fine=pd.read_csv(H/'fine_cell_diagnostics.csv',dtype={'grid':str,'sector':str,'parent':str})
sm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/granularity_sector_mapping_review.csv',dtype=str)
cm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/fine_to_coarse_country_sector_review.csv',dtype=str)
# Determine nested edges from reviewed links. Ambiguous R3 links are not split.
wide=sm[sm.mapping_status=='reviewed_unique'].pivot(index='fine_sector',columns='coarse_resolution',values='coarse_sector').dropna()
edges=wide[['2016','2013']].drop_duplicates()
bad16=set(edges.loc[edges['2016'].duplicated(False),'2016'])
bad13=set(edges.loc[edges['2016'].isin(bad16),'2013'])
# A parent potentially containing an unallocated R3 composite is excluded.
for g in ['2013','2016']:
 ambiguous=set(t.strip() for s in sm.loc[(sm.coarse_resolution==g)&(sm.mapping_status!='reviewed_unique'),'candidate_coarse_sectors'] for t in s.split('|'))
 if g=='2013':bad13|=ambiguous
 else:bad16|=ambiguous
bad13|=set(edges.loc[edges['2016'].isin(bad16),'2013'])
edges['admitted']=~edges['2016'].isin(bad16)&~edges['2013'].isin(bad13)
edges.to_csv(B/'sector_edges_2016_2013.csv',index=False)
edge=edges[edges.admitted].rename(columns={'2016':'sector','2013':'parent'})
countries=cm[['coarse_resolution','fine_country','coarse_country']].drop_duplicates().pivot(index='fine_country',columns='coarse_resolution',values='coarse_country').dropna()
cmap=countries[['2016','2013']].drop_duplicates();cmap=cmap[~cmap['2016'].duplicated(False)]
# Pure sector comparison: no merging of countries at the 2013 endpoint.
cmap=cmap[~cmap['2013'].duplicated(False)].rename(columns={'2016':'country','2013':'coarse_country'})
l=pd.read_parquet(H/'sector_levels.parquet');e=pd.read_parquet(H/'sector_Dint.parquet');d=e.copy();keys=['resolution','year','country','sector']
for side,basis in [(0,'CYP'),(1,'PYP')]:
 q=l[l.basis==basis].copy();q.year+=1-side
 d=d.merge(q[keys+['x']].rename(columns={'x':f'x{side}'}),on=keys,validate='one_to_one')
c13=d[d.resolution=='2013'].set_index(['year','country','sector'])
n=d[d.resolution=='2016'].merge(edge,on='sector',validate='many_to_one').merge(cmap,on='country',validate='many_to_one')
native=[]
for (year,cc,par),q in n.groupby(['year','coarse_country','parent']):
 c=c13.loc[(year,cc,par)]
 valid_output=bool((q[['x0','x1']]>0).all().all())
 x0=q.x0.sum();x1=q.x1.sum()
 if x0<=0 or x1<=0:continue
 w0=q.x0/x0;w1=q.x1/x1;v0=float((w0*q.bundle_v0).sum());v1=float((w1*q.bundle_v1).sum())
 within=float(((w0+w1)/2*(q.bundle_v1-q.bundle_v0)).sum());comp=float(((q.bundle_v0+q.bundle_v1)/2*(w1-w0)).sum())
 err=abs(v1-v0-within-comp)/max(abs(v0),abs(v1),1);assert err<1e-10
 xe=max(abs(x0-c.x0)/max(abs(c.x0),1),abs(x1-c.x1)/max(abs(c.x1),1))
 ve=max(abs(v0-c.bundle_v0)/max(abs(c.bundle_v0),1),abs(v1-c.bundle_v1)/max(abs(c.bundle_v1),1))
 native.append(dict(year=year,period=period(year),country16=q.country.iloc[0],country13=cc,parent13=par,children16=' | '.join(sorted(q.sector)),n_children16=len(q),
    within=within,composition=comp,composition_relative=comp/v0 if v0 else np.nan,
    Dint16_Mt=q.D_v.sum()/1e9,Dint13_Mt=c.D_v/1e9,x_error=xe,v_error=ve,identity_error=err,
    comparable=valid_output and xe<1e-5 and ve<1e-5))
native=pd.DataFrame(native);native.to_csv(B/'native_2016_to_2013.csv',index=False)
triple=parent[parent.grid=='2013'].merge(native,left_on=['year','coarse_country','parent'],right_on=['year','country13','parent13'],suffixes=('_R3','_16'),validate='one_to_one')
triple['full_chain']=triple.comparable_R3&triple.comparable_16
# Require each intermediate child itself to reproduce its R3 components.
good=set(parent.loc[(parent.grid=='2016')&parent.comparable,['year','coarse_country','parent']].itertuples(index=False,name=None))
# Audit single-child intermediate parents too: they have no within-parent
# sector composition, but may belong to a multi-child 2013 parent.
mc=cm[(cm.coarse_resolution=='2016')&cm.mapping_status.eq('reviewed_unique')]
cnt=mc.groupby(['coarse_country','coarse_sector']).size()
single=set(cnt[cnt==1].index)
ss=d[d.resolution=='R3'].merge(mc,left_on=['country','sector'],right_on=['fine_country','fine_sector'],validate='many_to_one')
ss=ss[[(c,s) in single for c,s in zip(ss.coarse_country,ss.coarse_sector)]]
c16=d[d.resolution=='2016'].rename(columns={'country':'coarse_country','sector':'coarse_sector'})
ss=ss.merge(c16[['year','coarse_country','coarse_sector','x0','x1','bundle_v0','bundle_v1']],on=['year','coarse_country','coarse_sector'],suffixes=('_f','_c'),validate='one_to_one')
ok=(ss.x0_f>0)&(ss.x1_f>0)
for col in ['x0','x1','bundle_v0','bundle_v1']:
 ok&=abs(ss[col+'_f']-ss[col+'_c'])/np.maximum(abs(ss[col+'_c']),1)<1e-5
good.update(ss.loc[ok,['year','coarse_country','coarse_sector']].itertuples(index=False,name=None))
triple['all_intermediate_audited']=[all((r.year,r.country16,c) in good for c in r.children16.split(' | ')) for r in triple.itertuples()]
triple['full_chain']&=triple.all_intermediate_audited
triple['R3_to16_flip']=(triple.fine_Dint_Mt*triple.Dint16_Mt<0)
triple['16_to13_flip']=(triple.Dint16_Mt*triple.Dint13_Mt<0)
triple.to_csv(B/'three_grid_parent_comparison.csv',index=False)
# Merge previously audited P3 with newly extracted P1/P2, and validate to
# existing country marginals and compact source/destination products.
t=pd.concat([pd.read_parquet(B/'country_product_emitter_sourcing.parquet'),pd.read_parquet(T/'country_product_emitter_sourcing.parquet')],ignore_index=True)
t.source_sector=t.source_sector.astype(str)
assert not t.duplicated(['year','effect','destination_country','source_group','source_sector','emitter_scope']).any()
src=t.groupby(['year','effect','destination_country','source_sector','source_group']).value.sum().unstack('source_group').reset_index()
em=t.groupby(['year','effect','destination_country','source_sector','emitter_scope']).value.sum().unstack('emitter_scope').reset_index()
s=src.merge(em,on=['year','effect','destination_country','source_sector'],validate='one_to_one')
s['outside_source']=s['Non-Red']+s['Non-Participant'];s['outside_emitter']=s['Foreign Non-Red']+s['Foreign Non-Participant']
s.to_csv(B/'annual_sourcing_slices.csv',index=False)
control=pd.concat([pd.read_csv(B/'emitter_country_control.csv'),pd.read_csv(T/'emitter_country_control.csv')])
p=pd.read_parquet(ROOT/'data/processed/provisional_panel_v2/cross_year_panel/emitter_country.parquet')
p=p[(p.resolution=='R3')&p.effect_id.isin(['D_C','D_YF'])&p.year_current.between(2001,2012)]
v=control.merge(p,left_on=['year','effect','country'],right_on=['year_current','effect_id','label'],validate='one_to_one')
v['scaled_error']=abs(v.value-v.contribution)/np.maximum(abs(v.contribution),1e9)
assert len(v)==len(p) and v.scaled_error.max()<1e-5
v[['year','effect','country','value','contribution','scaled_error']].to_csv(B/'sourcing_reconciliation.csv',index=False)
# Retain only numerically comparable direct parent cells in joint screen.
j=pd.concat([fine[(fine.group=='Red')&fine.comparable].merge(s[s.effect==eff],left_on=['year','country','sector'],right_on=['year','destination_country','source_sector'],validate='many_to_one') for eff in ['D_C','D_YF']],ignore_index=True)
j=j.merge(parent[['grid','year','country','parent','delta_v','composition','coarse_Dint_Mt','fine_Dint_Mt','coarse_minus_fine_Dint_Mt','composition_relative','within_relative']],on=['grid','year','country','parent'],validate='many_to_one')
struct=j.above_parent&(j.share_change<0)&(j.delta_v<0)&(j.composition<0)&(j.coarse_Dint_Mt<0)
for floor,name in [(1e8,'signature'),(1e9,'signature_1Mt')]:
 j[name]=struct&(j.Domestic<-floor)&(j.outside_source>floor)&(j.Home<-floor)&(j.outside_emitter>floor)
j.to_csv(B/'all_annual_joint_tests.csv',index=False)
cases=j.groupby(['grid','period','effect','country','parent','sector']).agg(years=('year','size'),passes=('signature','sum'),strict_passes=('signature_1Mt','sum'),
    domestic_Mt=('Domestic',lambda x:x.sum()/1e9),outside_source_Mt=('outside_source',lambda x:x.sum()/1e9),home_Mt=('Home',lambda x:x.sum()/1e9),outside_emitter_Mt=('outside_emitter',lambda x:x.sum()/1e9),
    product_Dint_Mt=('Dint_Mt','sum'),parent_Dint_Mt=('coarse_Dint_Mt','sum'),fine_parent_Dint_Mt=('fine_Dint_Mt','sum'),
    mean_composition_relative=('composition_relative','mean')).reset_index()
cases['complete_period']=cases.years==cases.period.map({'P1':4,'P2':3,'P3':5})
cases['period_sourcing_signs']=(cases.domestic_Mt<0)&(cases.outside_source_Mt>0)&(cases.home_Mt<0)&(cases.outside_emitter_Mt>0)
cases['repeated']=cases.passes>=2
cases.to_csv(B/'all_case_persistence.csv',index=False)
rep=cases[cases.repeated].copy();rep['case_key']=rep.period+'|'+rep.effect+'|'+rep.country+'|'+rep.sector
rep['both_mappings']=rep.case_key.map(rep.groupby('case_key').grid.nunique())==2
rep.to_csv(B/'repeated_candidates.csv',index=False)
# All annual high-intensity contractions, including near misses, are retained
# so agriculture is examined under the same gates as manufacturing.
ag=j[j.parent.isin(['A01','AtB'])].copy()
ag['failed_gates']=['; '.join(name for name,passed in [('high',r.above_parent),('share',r.share_change<0),('intensity',r.delta_v<0),('composition',r.composition<0),('Dint',r.coarse_Dint_Mt<0),('domestic',r.Domestic<-1e8),('outside_source',r.outside_source>1e8),('home',r.Home<-1e8),('outside_emitter',r.outside_emitter>1e8)] if not passed) for r in ag.itertuples()]
ag.to_csv(B/'agriculture_gate_audit.csv',index=False)
coverage=parent.groupby(['grid','period','group']).agg(screened=('year','size'),comparable=('comparable','sum')).reset_index();coverage.to_csv(B/'coverage.csv',index=False)
summary=rep.groupby(['period','effect']).agg(mapping_cases=('case_key','size'),distinct_cases=('case_key','nunique'),both_grid_cases=('both_mappings',lambda x:int(x.sum()/2)),complete_mapping_cases=('complete_period','sum')).reset_index()
summary.to_csv(B/'period_summary.csv',index=False)
validation=dict(passed=True,indicator='GWP100; no fossil-CO2 inference',annual_joint_rows=len(j),repeated_mapping_cases=len(rep),distinct_repeated_cases=rep.case_key.nunique(),
    source_reconciliation_max_scaled_error=float(v.scaled_error.max()),native_edge_rows=len(native),native_edge_comparable=int(native.comparable.sum()),full_three_grid_parent_years=int(triple.full_chain.sum()),
    source_slice_unique=True,excluded_non_nested_2016=sorted(bad16),excluded_2013_parents=sorted(bad13),
    source_files=[dict(path=str(x.relative_to(ROOT)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in [H/'sector_levels.parquet',H/'sector_Dint.parquet',B/'country_product_emitter_sourcing.parquet',T/'country_product_emitter_sourcing.parquet']])
(B/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in validation.items() if k!='source_files'},indent=2));print(summary.to_string(index=False));print(rep[['grid','period','effect','country','parent','sector','passes','strict_passes','complete_period','period_sourcing_signs','both_mappings']].to_string(index=False))
