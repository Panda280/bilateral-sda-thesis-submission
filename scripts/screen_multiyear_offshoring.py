"""Descriptive consecutive-year diagnostics; no new SDA reporting periods."""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
OUT=BASE/'multiyear';OUT.mkdir(exist_ok=True)
REGIONS={'WA','WE','WF','WL','WM','ROW','RoW','WT'}

def runs(frame,keys,flag):
    results=[]
    for key,q in frame.groupby(keys):
      q=q.sort_values('year');streak=[]
      for row in q.to_dict('records')+[{'year':9999,flag:False}]:
        if not row[flag] or (streak and row['year']!=streak[-1]['year']+1):
          if len(streak)>=2:
            r={k:v for k,v in zip(keys,key if isinstance(key,tuple) else [key])}
            r.update(start=streak[0]['year'],end=streak[-1]['year'],years=len(streak))
            for c in ['delta_territorial_Mt','delta_imported_footprint_Mt','cross_group_Mt']:
              if c in streak[0]:r[c]=sum(x[c] for x in streak)
            results.append(r)
          streak=[]
        if row[flag]:streak.append(row)
    return pd.DataFrame(results)

def main():
    q=pd.read_csv(BASE/'followup/annual_sector_changes.csv')
    o=pd.read_csv(BASE/'followup/annual_sector_import_origins.csv')
    groups=q[['country','group']].drop_duplicates();o=o.merge(groups,on='country',validate='many_to_one')
    o=o[o.source_group.ne(o.group)].groupby(['country','sector','year']).delta_imported_Mt.sum().rename('cross_group_Mt')
    q=q.join(o,on=['country','sector','year']);assert q.cross_group_Mt.notna().all()
    q=q[~q.country.isin(REGIONS)]
    q['signature']=(q.delta_territorial_Mt<-.1)&(q.delta_imported_footprint_Mt>.1)&(q.delta_import_expenditure_billion_EUR>.01)&(q.cross_group_Mt>.1)
    q.to_csv(OUT/'annual_matched_sector_screen.csv',index=False)
    r=runs(q,['country','group','sector'],'signature')
    r['matched_Mt']=np.minimum(-r.delta_territorial_Mt,r.delta_imported_footprint_Mt)
    r=r.sort_values(['years','matched_Mt'],ascending=False)
    r.to_csv(OUT/'consecutive_sector_cases.csv',index=False)
    print('MATCHED TERRITORY / EMBODIED IMPORT / REAL PRODUCT IMPORT / CROSS-GROUP INCREASE:')
    print(r.head(24).to_string(index=False))
    c=pd.read_csv(BASE/'followup/cross_group_input_screen.csv')
    c=c[~c.destination_country.isin(REGIONS)]
    c['signature']=(c.domestic_change<-1)&(c.imports_change>1)&(c.cross_group_change>1)&(c.share_t>c.share_base)&(c.domestic_t>1)&(c.imports_base>1)
    cr=runs(c,['destination_country','destination_group','using_sector','input_sector'],'signature')
    cr.to_csv(OUT/'consecutive_input_cases.csv',index=False)
    print('CROSS-GROUP INPUT RUNS:');print(cr.to_string(index=False))
    # Registered periods, not newly optimised endpoint windows.
    pp=q.groupby(['country','group','sector','period']).agg(
      territorial_Mt=('delta_territorial_Mt','sum'),imported_Mt=('delta_imported_footprint_Mt','sum'),cross_group_Mt=('cross_group_Mt','sum'),
      n=('year','size'),territory_decline_years=('delta_territorial_Mt',lambda x:(x<-.1).sum()),
      import_increase_years=('delta_imported_footprint_Mt',lambda x:(x>.1).sum()),
      real_import_increase_years=('delta_import_expenditure_billion_EUR',lambda x:(x>.01).sum()),
      all_four_years=('signature','sum')).reset_index()
    pp.to_csv(OUT/'registered_period_sector_screen.csv',index=False)
    pp=pp[(pp.group=='Red')&(pp.territorial_Mt<-1)&(pp.imported_Mt>1)&(pp.cross_group_Mt>1)&(pp.all_four_years>=2)]
    print('REDUCTION PERIODS:');print(pp.sort_values('all_four_years',ascending=False).head(25).to_string(index=False))
if __name__=='__main__':main()
