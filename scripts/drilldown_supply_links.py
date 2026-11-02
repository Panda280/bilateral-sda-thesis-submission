"""Directed reported D_C links into Reduction using industries; all emitters summed."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
B=AN/'full_drilldown_20260910'
d=pd.read_parquet(AN/'ten_sda_candidates_20260910/D_C_red_interactions.parquet')
d=d[d.resolution=='R3'].copy()
d.Source_Sector=d.Source_Sector.astype(str);d.Destination_Sector=d.Destination_Sector.astype(str)
assert d.Destination_Group.eq('Red').all()
d['component']=d.Source_Group.where(d.Relationship!='Domestic','Domestic')
pairs=[('Coal/peat to steel','9','43'),('Coke to steel','31','43'),
 ('Cereals to cattle/milk','1','4'),('Vegetables/oil seeds to cattle/milk','2','4'),
 ('Other crops to cattle/milk','3','4'),('Cereals to pigs/poultry','1','5')]
rows=[]
for label,source,dest in pairs:
 q=d[(d.Source_Sector==source)&(d.Destination_Sector==dest)]
 q=q.groupby(['year','component']).Value.sum().unstack('component').reset_index()
 assert len(q)==12
 q['link']=label;q['source']=source;q['using_sector']=dest
 q['outside']=q['Non-Red']+q['Non-Participant'];q['same_group']=q.Domestic+q.Red
 q['period']=q.year.map(lambda y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3')
 q['opposing_01']=(q.Domestic<-1e8)&(q.outside>1e8)
 rows.append(q)
a=pd.concat(rows,ignore_index=True);a.to_csv(B/'directed_supply_links_annual.csv',index=False)
p=a.groupby(['link','source','using_sector','period']).agg(years=('year','size'),opposing_years=('opposing_01','sum'),
 domestic_Mt=('Domestic',lambda x:x.sum()/1e9),foreign_Red_Mt=('Red',lambda x:x.sum()/1e9),non_Red_Mt=('Non-Red',lambda x:x.sum()/1e9),
 non_Participant_Mt=('Non-Participant',lambda x:x.sum()/1e9),outside_Mt=('outside',lambda x:x.sum()/1e9)).reset_index()
p['period_opposing']=(p.domestic_Mt<-.1)&(p.outside_Mt>.1)
p.to_csv(B/'directed_supply_links_period.csv',index=False)
# Follow the same labelled link into increasingly broad source/using parents.
# The native coarse cell contains additional products; it is not the original
# fine link, and its value must not be substituted for that link's value.
allgrids=pd.read_parquet(AN/'ten_sda_candidates_20260910/D_C_red_interactions.parquet')
sm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/granularity_sector_mapping_review.csv',dtype=str)
links=[]
for label,source,dest in pairs:
 for grid in ['R3','2016','2013']:
  if grid=='R3':ss,ds=source,dest
  else:
   mm=sm[(sm.coarse_resolution==grid)&sm.mapping_status.eq('reviewed_unique')].set_index('fine_sector').coarse_sector
   ss,ds=mm.loc[source],mm.loc[dest]
  q=allgrids[(allgrids.resolution==grid)&(allgrids.Source_Sector.astype(str)==ss)&(allgrids.Destination_Sector.astype(str)==ds)].copy()
  q['period']=q.year.map(lambda y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3')
  q['component']=q.Source_Group.where(q.Relationship!='Domestic','Domestic')
  for period,z in q.groupby('period'):
   v=z.groupby('component').Value.sum()/1e9
   links.append(dict(link=label,grid=grid,source_parent=ss,using_parent=ds,period=period,
     domestic_Mt=v['Domestic'],foreign_Red_Mt=v['Red'],outside_Mt=v['Non-Red']+v['Non-Participant']))
pd.DataFrame(links).to_csv(B/'directed_link_coarse_context.csv',index=False)
print(p.to_string(index=False))
