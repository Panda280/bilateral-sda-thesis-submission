"""Read-only R3 sector audit for the author's A/B/C follow-up.

Emitting-sector imported footprints and product-sector direct imports are
different measurements and are deliberately stored in separate columns.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(a.previous/'country_levels_and_trade.csv')
    lookup=old[(old.grid=='R3')&(old.year==2000)&(old.price_basis=='CYP')].set_index('country').group.to_dict()
    rows=[];origins=[];checks=[];sources=[]
    for year in range(2000,2013):
      for basis in ['CYP','PYP']:
        if year==2000 and basis=='PYP':continue
        stamp=str(year) if basis=='CYP' else f'{year}_in_{year-1}_prices'
        base=a.campaign/'build'/str(year)
        paths=[base/f'EXIOBASE_{basis}_{stamp}_R3_{k}.parquet' for k in ['Z','Y']]
        up=base/f'EXIOBASE_CO2_{year}_R3.csv'
        Z,Y=[pd.read_parquet(p) for p in paths]
        assert Z.index.equals(Z.columns) and Z.index.equals(Y.index) and Z.shape==(4802,4802)
        countries=list(dict.fromkeys(Z.index.get_level_values(0).astype(str)))
        sectors=list(dict.fromkeys(Z.index.get_level_values(1).astype(str)))
        assert len(countries)==49 and len(sectors)==98
        assert list(Z.index)==[(c,s) for c in countries for s in sectors]
        z=Z.to_numpy();y=Y.to_numpy();x=z.sum(1)+y.sum(1)
        yc=np.column_stack([y[:,Y.columns.get_level_values(0).astype(str)==c].sum(1) for c in countries])
        direct=z.reshape(4802,49,98).sum(2)+yc
        direct=direct.reshape(49,98,49)
        imports=direct.sum(0).T-np.array([direct[i,:,i] for i in range(49)])
        u=pd.read_csv(up,dtype={'region':str,'sector':str}).set_index(['region','sector']).CO2.reindex(Z.index)
        assert u.notna().all();u=u.to_numpy()
        if basis=='CYP':
            positive=x>0;xs=np.where(positive,x,1);A=z/xs[None,:];A[:,~positive]=0
            m=np.where(positive,u/xs,0)
            F=(m[:,None]*np.linalg.solve(np.eye(4802)-A,yc)).reshape(49,98,49)
            imported=F.sum(0).T-np.array([F[i,:,i] for i in range(49)])
            effective=(m*x).reshape(49,98)
            q=old[(old.grid=='R3')&(old.year==year)&(old.price_basis=='CYP')].set_index('country').loc[countries]
            terr_error=float(np.abs(u.reshape(49,98).sum(1)-q.territorial_kg_CO2).max())
            fp_error=float(np.abs(imported.sum(1)-q.imported_footprint_kg_CO2).max())
            balance=float(np.abs(F.sum(2)-effective).max())
            assert terr_error<1 and fp_error<1 and balance<1
            checks.append(dict(year=year,territory_error_kg=terr_error,imported_error_kg=fp_error,sector_balance_error_kg=balance))
            for j,c in enumerate(countries):
              for g in sorted(set(lookup.values())):
                ii=[i for i,sc in enumerate(countries) if lookup[sc]==g and i!=j]
                values=F[ii,:,j].sum(0)
                for k,s in enumerate(sectors):origins.append(dict(year=year,country=c,sector=s,source_group=g,imported_kg_CO2=values[k]))
        for j,c in enumerate(countries):
          for k,s in enumerate(sectors):
            rows.append(dict(year=year,price_basis=basis,price_year=year if basis=='CYP' else year-1,country=c,group=lookup[c],sector=s,
                territorial_kg_CO2=u.reshape(49,98)[j,k],effective_territorial_kg_CO2=effective[j,k] if basis=='CYP' else np.nan,
                imported_emitting_sector_kg_CO2=imported[j,k] if basis=='CYP' else np.nan,direct_product_imports_million_EUR=imports[j,k]))
        sources.extend(dict(path=str(p),sha256=sha(p)) for p in paths+[up])
        print(year,basis,'complete',flush=True)
    pd.DataFrame(rows).to_csv(a.output/'sector_levels.csv',index=False)
    pd.DataFrame(origins).to_csv(a.output/'sector_import_origins.csv',index=False)
    pd.DataFrame(checks).to_csv(a.output/'validation.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps(dict(completed=True,sources=sources,price_basis='CYP levels; PYP(t) minus CYP(t-1) direct import changes',
      limitation='Imported footprint sector identifies emitting activity, not the final product. Direct product imports include intermediate and final purchases. Neither identifies causal displacement.'),indent=2))
if __name__=='__main__':main()
