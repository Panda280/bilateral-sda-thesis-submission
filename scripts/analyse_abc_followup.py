"""Follow-up screens: geographic concentration and cross-group crop inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
OUT=BASE/'followup';OUT.mkdir(exist_ok=True)
CON=ROOT/'references/current_codebase/concordance/normalized_2026-08-31'
REGIONS={'WA','WE','WF','WL','WM','ROW','RoW','WT'}
def main():
    cmap=pd.read_csv(CON/'country_concordance_by_resolution.csv',dtype=str)
    registry=pd.read_csv(CON/'analytical_group_registry.csv').set_index('canonical_country').analysis_group.to_dict()
    panels=[]
    for role in ['driver_country','emitter_country','consumer_country']:
      d=pd.read_parquet(ROOT/f'data/processed/provisional_panel_v2/cross_year_panel/{role}.parquet')
      d=d[d.year_current.between(2001,2004)].copy()
      assert d.is_publishable.all() and d.validation_status.eq('passed').all()
      d=d.merge(cmap[['resolution','observed_country','canonical_country']],left_on=['resolution','label'],right_on=['resolution','observed_country'],validate='many_to_one')
      d['group']=d.canonical_country.map(registry);assert d.group.notna().all()
      d['role']=role;d['Mt']=d.contribution/1e9
      d['red_plus_USA']=d.group.eq('Red')|d.canonical_country.eq('USA')
      panels.append(d)
    d=pd.concat(panels);d.to_csv(OUT/'annual_country_attribution.csv',index=False)
    rows=[]
    for key,q in d.groupby(['resolution','year_current','role','effect_id']):
      for name,sel in [('Red plus USA',q.red_plus_USA),('Others',~q.red_plus_USA)]+[(g,q.group.eq(g)) for g in sorted(q.group.unique())]:
        x=q[sel];neg=-q.Mt.clip(upper=0).sum();absolute=q.Mt.abs().sum()
        rows.append(dict(zip(['resolution','year','role','effect'],key))|dict(scope=name,net_Mt=x.Mt.sum(),negative_country_net_mass_Mt=-x.Mt.clip(upper=0).sum(),share_negative_country_net_mass=-x.Mt.clip(upper=0).sum()/neg if neg else np.nan,share_absolute_country_nets=x.Mt.abs().sum()/absolute if absolute else np.nan))
    pd.DataFrame(rows).to_csv(OUT/'geographic_concentration.csv',index=False)
    # Joint inputs: select country, supplying crop and livestock user; keep
    # cross-border within-group and cross-group origins separate.
    links=pd.read_csv(BASE/'trade_inputs/crop_livestock_links.csv',dtype={'input_sector':str,'using_sector':str})
    keys=['year','destination_country','destination_group','using_sector','input_sector']
    links['origin_relation']=np.where(links.source_country.eq(links.destination_country),'domestic',np.where(links.source_group.eq(links.destination_group),'within_group_foreign','cross_group'))
    q=links.groupby(keys+['price_basis','origin_relation']).million_EUR.sum().unstack('origin_relation').reset_index()
    prev=q[q.price_basis.eq('CYP')].copy();prev.year+=1
    q=q[q.price_basis.eq('PYP')].merge(prev,on=keys,suffixes=('_t','_base'),validate='one_to_one')
    for c in ['domestic','within_group_foreign','cross_group']:q[c+'_change']=q[c+'_t']-q[c+'_base']
    q['imports_base']=q.within_group_foreign_base+q.cross_group_base;q['imports_t']=q.within_group_foreign_t+q.cross_group_t
    q['imports_change']=q.imports_t-q.imports_base
    q['total_base']=q.domestic_base+q.imports_base;q['total_t']=q.domestic_t+q.imports_t
    q['share_base']=q.imports_base/q.total_base;q['share_t']=q.imports_t/q.total_t
    q['cross_group_share_of_import_increment']=q.cross_group_change/q.imports_change
    q['candidate']=(q.domestic_change<-10)&(q.imports_change>10)&(q.cross_group_change>10)&(q.share_t>q.share_base)&(q.domestic_t>1)&(q.domestic_base>1)&(q.imports_base>1)&(~q.destination_country.isin(REGIONS))
    q['matched_cross_group_offset']=np.minimum(-q.domestic_change,q.cross_group_change).clip(lower=0)
    q.to_csv(OUT/'cross_group_input_screen.csv',index=False)
    ranked=q[q.candidate].sort_values('matched_cross_group_offset',ascending=False)
    ranked.to_csv(OUT/'cross_group_candidates.csv',index=False)
    # Retain country-origin detail for all candidates, avoiding hand-picked
    # counterparties after looking at the desired group relationship.
    o=links[links.price_basis.eq('PYP')].merge(links[links.price_basis.eq('CYP')].assign(year=lambda x:x.year+1),on=keys+['source_country','source_group','origin_relation','relationship'],suffixes=('_t','_base'),validate='one_to_one')
    o['change_million_EUR']=o.million_EUR_t-o.million_EUR_base
    o=o.merge(ranked[keys],on=keys,validate='many_to_one');o.to_csv(OUT/'cross_group_candidate_origins.csv',index=False)
    print(pd.DataFrame(rows).query("year==2003 and effect=='D_T_coef' and scope=='Red plus USA'").to_string(index=False))
    print(ranked.head(15)[keys+['domestic_change','imports_change','cross_group_change','share_base','share_t','matched_cross_group_offset']].to_string(index=False))
if __name__=='__main__':main()
