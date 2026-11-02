"""Reported-effect-only screens and role-matched reconciliation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
BASE=AN/'ten_sda_candidates_20260910'
PERIOD={y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3' for y in range(2001,2013)}
def main():
    rows=[]
    for effect in ['D_C','D_YF']:
      d=pd.read_parquet(BASE/f'{effect}_red_interactions.parquet')
      d=d.rename(columns={'Source_Sector':'product','Product_Sector':'product'})
      d['source_component']=np.where(d.Relationship.eq('Domestic'),'Domestic',d.Source_Group)
      keys=['resolution','year','effect','product']
      q=d.groupby(keys+['source_component']).Value.sum().unstack('source_component').reset_index()
      q['period']=q.year.map(PERIOD);rows.append(q)
    a=pd.concat(rows,ignore_index=True);a['same_group']=a.Domestic+a.Red
    a['cross_group']=a['Non-Red']+a['Non-Participant']
    a['net']=a.Domestic+a.Red+a.cross_group
    a['signature']=(a.Domestic<-1e9)&(a['Non-Participant']>1e9)&(a.same_group<-1e9)&(a.cross_group>1e9)
    a.to_csv(BASE/'annual_product_sourcing.csv',index=False)
    q=a.groupby(['resolution','effect','product','period']).agg(**{c:(c,'sum') for c in ['Domestic','Red','Non-Red','Non-Participant','same_group','cross_group','net']},years=('year','size'),material_years=('signature','sum')).reset_index()
    q['rank_magnitude']=np.minimum(-q.Domestic,q['Non-Participant'])
    q['eligible']=(q.Domestic<-1e10)&(q['Non-Participant']>1e10)&(q.same_group<-1e10)&(q.cross_group>1e10)&(q.material_years>=3)
    q.to_csv(BASE/'period_product_sourcing.csv',index=False)
    print('GROUP PRODUCT SCREEN:')
    print(q[(q.resolution=='R3')&(q.period=='P3')&q.eligible].sort_values('rank_magnitude',ascending=False).head(30).to_string(index=False))
    # Independent world/destination-role reconciliation to R038/R039.
    g=pd.read_csv(BASE/'annual_group_interactions.csv',dtype={'resolution':str});g['period']=g.year.map(PERIOD)
    got=g.groupby(['resolution','period','effect','Relationship','Source_Group','Destination_Group']).Value.sum()
    reg=pd.read_csv(AN/'kyoto_period_group_products_20260908/kyoto_period_group_interactions.csv',dtype={'resolution':str})
    reg=reg.rename(columns={'relationship':'Relationship','source_group':'Source_Group','destination_group':'Destination_Group'})
    expected=reg.set_index(got.index.names).net
    joined=expected.to_frame('registered').join(got.rename('extracted'));assert joined.extracted.notna().all()
    err=float((abs(joined.registered-joined.extracted)/np.maximum(abs(joined.registered),1e9)).max());assert err<1e-9
    gr=pd.read_csv(AN/'kyoto_period_group_products_20260908/kyoto_period_group_effects.csv',dtype={'resolution':str})
    effects=['D_v','D_A_x','D_C','D_YF','D_YB','D_y']
    emitter=gr[(gr.role=='emitter')&gr.group.eq('Red')&gr.effect.isin(effects)].pivot(index=['resolution','period'],columns='effect',values='net')
    emitter['total']=emitter[effects].sum(1);emitter['sourcing']=emitter.D_C+emitter.D_YF
    emitter['sourcing_share_of_net_change']=emitter.sourcing/emitter.total
    emitter.to_csv(BASE/'reduction_emitter_decomposition.csv')
    (BASE/'compact_validation.json').write_text(json.dumps(dict(passed=True,registered_interaction_max_relative_error=err),indent=2))
if __name__=='__main__':main()
