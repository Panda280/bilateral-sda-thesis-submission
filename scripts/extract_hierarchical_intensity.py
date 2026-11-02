"""Read-only compact sector extraction from existing campaign inputs and D_v."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);rows=[];effects=[];sources=[];checks=[]
 for grid,tag in [('R3','R3'),('2016','WIOT2016'),('2013','WIOT2013')]:
  for year in range(2000,2013):
   for basis in ['CYP','PYP']:
    if (year==2000 and basis=='PYP') or (year==2012 and basis=='CYP'):continue
    stamp=str(year) if basis=='CYP' else f'{year}_in_{year-1}_prices'
    b=a.campaign/'build'/str(year)
    zp=b/f'EXIOBASE_{basis}_{stamp}_{tag}_Z.parquet';yp=b/f'EXIOBASE_{basis}_{stamp}_{tag}_Y.parquet';up=b/f'EXIOBASE_CO2_{year}_{tag}.csv'
    z=pd.read_parquet(zp);y=pd.read_parquet(yp)
    assert z.index.equals(z.columns) and z.index.equals(y.index)
    u=pd.read_csv(up,dtype={'region':str,'sector':str}).set_index(['region','sector']).CO2
    ix=pd.MultiIndex.from_tuples([(str(c),str(s)) for c,s in z.index],names=['country','sector'])
    u=u.reindex(ix);assert u.notna().all()
    x=z.sum(axis=1).to_numpy(dtype=float)+y.sum(axis=1).to_numpy(dtype=float)
    q=pd.DataFrame({'u':u.to_numpy(),'x':x},index=ix).reset_index()
    q['v']=np.divide(q.u,q.x,out=np.zeros(len(q)),where=q.x!=0)
    q['year']=year;q['basis']=basis;q['resolution']=grid;rows.append(q)
    checks.append(dict(grid=grid,year=year,basis=basis,rows=len(q),nonpositive_output=int((x<=0).sum()),nonfinite=int((~np.isfinite(q[['u','x','v']])).to_numpy().sum())))
    sources.extend(dict(path=str(p),sha256=sha(p)) for p in [zp,yp,up])
    del z,y
   if year>=2001:
    bundle=a.campaign/'bundles'/f'{year-1}_{year}_{grid}'
    config=json.loads((bundle/'config.json').read_text());labels=config['semantics']['emitter_labels']
    path=a.campaign/'sda'/f'sda_{year-1}_{year}_{grid}'/'arrays/effects/D_v/baseline.npy'
    arr=np.load(path,mmap_mode='r');assert arr.shape[0]==len(labels)
    vals=np.asarray(arr).sum(axis=tuple(range(1,arr.ndim)),dtype=np.float64) if arr.ndim>1 else np.asarray(arr,dtype=float)
    e=pd.DataFrame([v.split(':',1) for v in labels],columns=['country','sector']);e['D_v']=vals;e['year']=year;e['resolution']=grid
    # The bundle intensity is what the stored SDA actually used. Retain it
    # to verify extracted build levels before interpreting the mechanism.
    with np.load(bundle/'state.npz') as state:
     e['bundle_v0']=state['v0'].ravel();e['bundle_v1']=state['v1'].ravel()
    effects.append(e);sources.append(dict(path=str(path),sha256=sha(path)))
   pd.concat(rows,ignore_index=True).to_parquet(a.output/'sector_levels.parquet',index=False)
   if effects:pd.concat(effects,ignore_index=True).to_parquet(a.output/'sector_Dint.parquet',index=False)
   print(grid,year,'complete',flush=True)
 (a.output/'extraction_manifest.json').write_text(json.dumps(dict(completed=True,checks=checks,sources=sources,scope='Existing build levels and baseline intensity effect; no SDA recomputation. CYP(t-1), PYP(t).'),indent=2))
if __name__=='__main__':main()
