"""Identify the stored campaign indicator without modifying inputs or effects."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('/local/s2053268/sda-campaign-2000-2014-v2')
OUT=ROOT/'analysis/hierarchical_intensity_20260910'
os.environ['SDA_WIOT_VERSION']='R3'
os.environ['SDA_EXIOBASE_RAW_DIR']='/local/s2053268/sda-campaign-2000-2014/cache/exiobase'
os.environ['SDA_NUM_SAMPLES']='1'
sys.path.insert(0,'/local/s2053268/ie-thesis/IE_Thesis_EXIO_WIOT/SDA')
import data_loader as dl
import config
l=pd.read_parquet(OUT/'sector_levels.parquet');e=pd.read_parquet(OUT/'sector_Dint.parquet')
records=[];checks=[];cache={}
for (grid,year),e0 in e.groupby(['resolution','year'],sort=False):
 for side,basis,stamp in [(0,'CYP',year-1),(1,'PYP',year)]:
  q=l[(l.resolution==grid)&(l.year==stamp)&(l.basis==basis)].copy()
  q=q.merge(e0[['country','sector',f'bundle_v{side}']],on=['country','sector'],validate='one_to_one')
  ix=pd.MultiIndex.from_frame(q[['country','sector']]);ix.names=['region','sector']
  key=(grid,stamp)
  if key not in cache: cache[key]=dl._wiod_totals_at_resolution(int(stamp),'R3' if grid=='R3' else int(grid))
  g=cache[key].reindex(ix);assert g.notna().all();q['gwp_u']=g.to_numpy()
  v=np.asarray(np.divide(q.gwp_u,q.x,out=np.zeros(len(q)),where=q.x!=0)).astype(np.float32)
  v[abs(v)<config.eps]=0
  nc=q.country.nunique();ns=q.sector.nunique();mat=v.reshape(nc,ns);med=np.median(mat,axis=0,keepdims=True)
  mask=abs(mat)>config.V_SECTOR_OUTLIER_MULTIPLE*abs(med)
  vv=np.where(mask,med,mat).ravel();vv=np.clip(vv,-config.V_INTENSITY_MAX,config.V_INTENSITY_MAX)
  q['gwp_policy_v']=vv;q['policy_replaced']=mask.ravel();q['side']=side;q['transition']=year
  q['gwp_scaled_error']=abs(q[f'bundle_v{side}']-vv)/np.maximum(abs(q[f'bundle_v{side}']),1)
  q['fossil_scaled_error']=abs(q[f'bundle_v{side}']-q.v)/np.maximum(abs(q[f'bundle_v{side}']),1)
  checks.append(dict(year=int(year),side=side,resolution=grid,rows=len(q),gwp_max_scaled_error=float(q.gwp_scaled_error.max()),gwp_failing_1e5=int((q.gwp_scaled_error>1e-5).sum()),fossil_failing_1e5=int((q.fossil_scaled_error>1e-5).sum()),replaced=int(mask.sum())))
  records.append(q)
 pd.DataFrame(checks).to_csv(OUT/'indicator_checks.csv',index=False)
 pd.concat(records,ignore_index=True).to_parquet(OUT/'indicator_checked_levels.parquet',index=False)
 print(grid,year,checks[-1]['gwp_max_scaled_error'],flush=True)
r=pd.DataFrame(checks)
result=dict(completed=True,observations=int(r.rows.sum()),gwp_max_scaled_error=float(r.gwp_max_scaled_error.max()),gwp_failing_1e5=int(r.gwp_failing_1e5.sum()),fossil_failing_1e5=int(r.fossil_failing_1e5.sum()),outlier_multiple=float(config.V_SECTOR_OUTLIER_MULTIPLE),intensity_max=float(config.V_INTENSITY_MAX),loader_sha256=hashlib.sha256(Path(dl.__file__).read_bytes()).hexdigest(),note='All saved coefficients compared with current GWP100 stressor mapping and recorded intensity policy. No input/effect changes.')
(OUT/'indicator_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
