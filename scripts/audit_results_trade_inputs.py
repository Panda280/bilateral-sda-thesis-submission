"""Read-only raw-input audit and compact trade/footprint extracts for hypotheses B/C.

Runs beside campaign inputs. Writes only to a new explicitly selected output directory.
No source tables, concordances, intensities, or SDA outputs are modified.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--country-concordance',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    registry=pd.read_csv(a.registry,dtype=str).set_index('country_code').analysis_group.to_dict()
    concordance=pd.read_csv(a.country_concordance,dtype=str)
    source_meta=[]; country_rows=[]; links=[]; balances=[]; pairs=[]; impact_checks=[]
    for grid,res,M,N in [('WIOT2016','2016',44,50),('WIOT2013','2013',41,35),('R3','R3',49,98)]:
      cmap=concordance[concordance.resolution.eq(res)].set_index('observed_country').canonical_country.to_dict()
      for year in range(2000,2013):
        for basis in ['CYP','PYP']:
          if year==2000 and basis=='PYP':continue
          stamp=f'{year}' if basis=='CYP' else f'{year}_in_{year-1}_prices'
          b=a.campaign/'build'/str(year)
          zp=b/f'EXIOBASE_{basis}_{stamp}_{grid}_Z.parquet'; yp=b/f'EXIOBASE_{basis}_{stamp}_{grid}_Y.parquet'
          up=b/f'EXIOBASE_CO2_{year}_{grid}.csv'
          Z=pd.read_parquet(zp); Y=pd.read_parquet(yp)
          assert Z.shape==(M*N,M*N) and Z.index.equals(Z.columns) and Z.index.equals(Y.index)
          assert isinstance(Z.index,pd.MultiIndex) and isinstance(Y.columns,pd.MultiIndex)
          countries=list(dict.fromkeys(Z.index.get_level_values(0).astype(str)))
          assert len(countries)==M
          cpos={c:i for i,c in enumerate(countries)}
          cc=np.array([cpos[str(c)] for c in Z.index.get_level_values(0)])
          sector=np.array(Z.index.get_level_values(1).astype(str))
          yg=np.array([cpos[str(c)] for c in Y.columns.get_level_values(0)])
          groups=[registry[cmap[c]] for c in countries]
          assert set(groups)<= {'Red','Non-Red','Non-Participant'}
          z=Z.to_numpy(dtype=np.float64); y=Y.to_numpy(dtype=np.float64)
          assert np.isfinite(z).all() and np.isfinite(y).all()
          x=z.sum(axis=1)+y.sum(axis=1)
          raw=pd.read_csv(up,dtype={'region':str,'sector':str}).set_index(['region','sector'])
          assert raw.index.is_unique
          index=pd.MultiIndex.from_tuples([(str(c),str(s)) for c,s in Z.index])
          emissions=raw.CO2.reindex(index)
          assert not emissions.isna().any(), 'unmatched emissions'
          u=emissions.to_numpy(dtype=float)
          source_meta.extend([{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in [zp,yp,up]])
          yc=np.zeros((M*N,M))
          for j,d in enumerate(yg): yc[:,d]+=y[:,j]
          # Country purchases at a common within-transition monetary price basis.
          foreign_z=np.zeros(M); domestic_z=np.zeros(M); foreign_y=np.zeros(M); domestic_y=np.zeros(M)
          for i in range(M):
            source=cc==i; dest=cc==i
            domestic_z[i]=z[np.ix_(source,dest)].sum(); foreign_z[i]=z[:,dest].sum()-domestic_z[i]
            domestic_y[i]=yc[source,i].sum(); foreign_y[i]=yc[:,i].sum()-domestic_y[i]
          # Independently reconstruct footprints in CYP for every source year.
          # PYP import-expenditure changes are kept separate from these mass levels.
          if basis=='CYP':
            positive=x>0; xs=np.where(positive,x,1)
            A=z/xs[None,:]; A[:,~positive]=0
            m=np.where(positive,u/xs,0)
            induced=np.linalg.solve(np.eye(M*N)-A,yc)
            F=m[:,None]*induced
            P=np.zeros((M,M)); np.add.at(P,cc,F)
            terr=np.bincount(cc,weights=u,minlength=M)
            effective=np.bincount(cc,weights=m*x,minlength=M)
            dom=np.diag(P); total=P.sum(axis=0); imp=total-dom
            rowsum=P.sum(axis=1)
            residual=np.max(np.abs(rowsum-effective))/max(np.abs(effective).sum(),1)
            retained=np.abs((u-m*x)).sum()/max(np.abs(u).sum(),1)
            assert residual<1e-8, ('country mass balance',grid,year,residual)
            # Preserve this loss as a reported limitation, not a hidden zero-fill.
            # It is evaluated for admissibility downstream against the observed bound.
            old=a.campaign/'analysis/footprint_levels_20260908'/f'P_{grid}_{year}.npy'
            old_rel=None
            if old.exists():
              oldP=np.load(old)
              oldcountries=json.loads((old.parent/f'countries_{grid}.json').read_text())
              assert oldcountries==countries
              old_rel=float(np.abs(oldP-P).max()/max(np.abs(P).max(),1))
              assert old_rel<1e-8, ('existing footprint mismatch',grid,year,old_rel)
            balances.append({'grid':grid,'year':year,'relative_country_emission_balance':residual,'impact_loss_relative':retained,'old_P_max_relative_difference':old_rel,
                             'world_territorial_kg':terr.sum(),'world_footprint_kg':P.sum(),'nonpositive_output_cells':int((~positive).sum())})
            np.save(a.output_dir/f'P_{grid}_{year}.npy',P)
            for i,c in enumerate(countries):
              for j,d in enumerate(countries):
                pairs.append({'grid':grid,'year':year,'origin':c,'destination':d,'origin_group':groups[i],'destination_group':groups[j],
                              'relationship':'domestic' if i==j else 'cross_border','embodied_kg_CO2':P[i,j]})
          else:
            terr=np.bincount(cc,weights=u,minlength=M); dom=imp=total=np.full(M,np.nan)
          for i,c in enumerate(countries):
            country_rows.append({'grid':grid,'year':year,'price_basis':basis,'price_year':year if basis=='CYP' else year-1,
              'country':c,'canonical_country':cmap[c],'group':groups[i],'territorial_kg_CO2':terr[i],
              'effective_territorial_kg_CO2':effective[i] if basis=='CYP' else np.nan,
              'domestic_footprint_kg_CO2':dom[i],'imported_footprint_kg_CO2':imp[i],'total_footprint_kg_CO2':total[i],
              'intermediate_imports_million_EUR':foreign_z[i],'intermediate_domestic_million_EUR':domestic_z[i],
              'final_imports_million_EUR':foreign_y[i],'final_domestic_million_EUR':domestic_y[i],
              'output_million_EUR':x[cc==i].sum()})
          # Exact input links lost when crop and livestock sectors are merged into A01.
          if grid=='R3':
            for dest_country in countries:
              j=cpos[dest_country]
              for ds in ['4','5','6']:
                cols=(cc==j)&(sector==ds)
                assert cols.sum()==1
                for ss in ['1','2','3']:
                  for source_country in countries:
                    i=cpos[source_country]; row=(cc==i)&(sector==ss)
                    assert row.sum()==1
                    links.append({'year':year,'price_basis':basis,'price_year':year if basis=='CYP' else year-1,
                        'source_country':source_country,'source_group':groups[i],'input_sector':ss,
                        'destination_country':dest_country,'destination_group':groups[j],'using_sector':ds,
                        'relationship':'domestic' if i==j else 'cross_border','million_EUR':float(z[np.ix_(row,cols)].sum())})
          print(f'{grid} {year} {basis}: extracted',flush=True)
          # Checkpoint so progress survives interruption without overwriting upstream data.
          pd.DataFrame(country_rows).to_csv(a.output_dir/'country_levels_and_trade.csv',index=False)
          pd.DataFrame(balances).to_csv(a.output_dir/'footprint_validation.csv',index=False)
    pd.DataFrame(links).to_csv(a.output_dir/'crop_livestock_links.csv',index=False)
    pd.DataFrame(pairs).to_csv(a.output_dir/'embodied_country_pairs.csv',index=False)
    inputs={x['path']:x for x in source_meta}
    manifest={'analysis_id':'trade-input-audit-20260909','completed':True,'input_sources':list(inputs.values()),
      'registry_sha256':sha(a.registry),'country_concordance_sha256':sha(a.country_concordance),
      'years':list(range(2000,2013)),'grids':['WIOT2013','WIOT2016','R3'],
      'units':{'emissions':'kg fossil-origin CO2','flows':'million EUR, labelled CYP or PYP'},
      'validation':{'passed':True,'footprint_grid_years':len(balances),'index_matching_exact':True,'no_silent_impact_fill':True},
      'limitations':['PYP changes are adjacent-year constant-price monetary flows, not physical quantities.',
                    'Footprint levels do not identify causal displacement.','Crop-to-livestock flows are monetary inputs, not necessarily feed alone.']}
    (a.output_dir/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
