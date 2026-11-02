"""Read existing SDA interactions, validate, and export compact Red slices."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    specs={'D_C':('source_group_sector_interaction','D_C.source_destination_group_sector_interaction.parquet','source_group_sector_interaction_report.json'),
       'D_YF':('yf_group_product_interaction','D_YF.source_destination_group_product_interaction.parquet','yf_group_product_interaction_report.json')}
    groups=[];checks=[];sources=[];red={k:[] for k in specs}
    for grid in ['2013','2016','R3']:
      for year in range(2001,2013):
       for effect,(folder,filename,report) in specs.items():
        path=a.campaign/'offshoring'/f'{year-1}_{year}_{grid}'/folder/filename
        rp=path.with_name(report);meta=json.loads(rp.read_text());q=pd.read_parquet(path)
        assert meta['units']=='kg CO2' and meta['effect']==effect
        assert meta['reconciliation']['failing_index'] is None
        assert tuple(meta['columns'])==tuple(q.columns) and int(np.prod(meta['shape']))==len(q)
        assert np.isfinite(q.Value).all() and not q.duplicated(list(q.columns[:-1])).any()
        impossible=(q.Relationship=='Domestic')&(q.Source_Group!=q.Destination_Group)
        assert q.loc[impossible,'Value'].eq(0).all()
        residual=abs(q.Value.sum()-meta['total'])/max(abs(meta['total']),1e9);assert residual<1e-10
        checks.append(dict(grid=grid,year=year,effect=effect,rows=len(q),relative_report_total_error=residual,reconciliation=meta['reconciliation']))
        q['resolution']=grid;q['year']=year;q['effect']=effect
        groups.append(q.groupby(['resolution','year','effect','Relationship','Source_Group','Destination_Group'],as_index=False).Value.sum())
        red[effect].append(q[q.Destination_Group.eq('Red')].copy())
        sources.extend(dict(path=str(p),sha256=sha(p)) for p in [path,rp])
       print(grid,year,'checked',flush=True)
    for effect,frames in red.items():pd.concat(frames,ignore_index=True).to_parquet(a.output/f'{effect}_red_interactions.parquet',index=False)
    pd.concat(groups,ignore_index=True).to_csv(a.output/'annual_group_interactions.csv',index=False)
    (a.output/'extraction_manifest.json').write_text(json.dumps(dict(completed=True,sources=sources,checks=checks,source_scope='existing registered SDA products; no recalculation or matrix changes'),indent=2))
if __name__=='__main__':main()
