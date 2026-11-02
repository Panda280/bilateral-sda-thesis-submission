"""Stream existing P1/P2 sourcing cubes; preserve country, product and emitter scope."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

GROUPS=['Red','Non-Red','Non-Participant']
SOURCE=['Domestic']+GROUPS
SCOPES=['Home','Foreign Red','Foreign Non-Red','Foreign Non-Participant']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
      for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def codes(column,labels):
    out=pc.index_in(pc.cast(column,pa.string()),value_set=pa.array(labels))
    assert out.null_count==0
    return out.to_numpy().astype(np.int64)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--lookup',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    lookup=pd.read_csv(a.lookup);lookup=lookup[(lookup.grid=='R3')&(lookup.year==2000)&(lookup.price_basis=='CYP')]
    countries=lookup.country.tolist();assert len(countries)==49
    groupcodes=np.array([GROUPS.index(g) for g in lookup.group]);sectors=[str(i) for i in range(1,99)]
    outputs=[];checks=[];metadata=[];emitter_rows=[]
    for year in range(2001,2008):
      for effect,folder,dest in [('D_C','source_group_sector','Driver_Country'),('D_YF','yf_source_group_sector','Consumer_Country')]:
        b=a.campaign/'offshoring'/f'{year-1}_{year}_R3'/folder
        path=b/f'{effect}.source_group_sector_cube.parquet'
        report=b/('source_group_sector_cube_report.json' if effect=='D_C' else 'yf_source_group_sector_cube_report.json')
        meta=json.loads(report.read_text());assert meta['units']=='kg CO2'
        pf=pq.ParquetFile(path)
        acc=np.zeros(4*49*4*98);emit=np.zeros(49);count=0;total=0.
        for batch in pf.iter_batches(batch_size=262144,columns=['Emitter_Country',dest,'Source_Group','Source_Sector','Value']):
          ec=codes(batch['Emitter_Country'],countries);dc=codes(batch[dest],countries)
          sg=codes(batch['Source_Group'],SOURCE);ss=codes(batch['Source_Sector'],sectors)
          v=batch['Value'].to_numpy().astype(np.float64);assert np.isfinite(v).all()
          scope=np.where(ec==dc,0,groupcodes[ec]+1)
          index=((scope*49+dc)*4+sg)*98+ss
          acc+=np.bincount(index,weights=v,minlength=len(acc));emit+=np.bincount(ec,weights=v,minlength=49)
          count+=len(v);total+=v.sum()
        assert count==pf.metadata.num_rows
        err=abs(acc.sum()-meta['total'])/max(abs(meta['total']),1e9);assert err<1e-9,(effect,year,err)
        for i,c in enumerate(countries):emitter_rows.append(dict(year=year,effect=effect,country=c,value=emit[i]))
        ix=pd.MultiIndex.from_product([SCOPES,countries,SOURCE,sectors],names=['emitter_scope','destination_country','source_group','source_sector'])
        q=pd.DataFrame({'value':acc},index=ix).reset_index();q['year']=year;q['effect']=effect
        q['destination_group']=q.destination_country.map(lookup.set_index('country').group)
        outputs.append(q)
        checks.append(dict(year=year,effect=effect,rows=count,relative_total_error=err))
        metadata.extend(dict(path=str(p),sha256=sha(p)) for p in [path,report])
        pd.concat(outputs).to_parquet(a.output/'country_product_emitter_sourcing.parquet',index=False)
        print(year,effect,'emitter-resolved extraction complete',flush=True)
    pd.DataFrame(emitter_rows).to_csv(a.output/'emitter_country_control.csv',index=False)
    (a.output/'emitter_extraction_manifest.json').write_text(json.dumps(dict(completed=True,checks=checks,sources=metadata,
      scope='P1/P2 R3 existing baseline D_C and D_YF; GWP100 per R-2026-053 despite legacy kg CO2 metadata, country-product by relative source and emitter location. No new effects calculated.'),indent=2))
if __name__=='__main__':main()
