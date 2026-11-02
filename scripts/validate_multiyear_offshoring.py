"""Validate selected consecutive-year examples against independent levels."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
OUT=BASE/'multiyear'
def main():
    annual=pd.read_csv(OUT/'annual_matched_sector_screen.csv')
    raw=pd.read_csv(BASE/'fine_followup/sector_levels.csv')
    selected=[];res=[]
    for country,sector,start,end in [('IT',43,2005,2007),('JP',32,2006,2008)]:
      q=annual[(annual.country==country)&(annual.sector==sector)&annual.year.between(start,end)]
      assert len(q)==end-start+1 and q.signature.all()
      r=raw[(raw.country==country)&(raw.sector==sector)&(raw.price_basis=='CYP')].set_index('year')
      for delta,level in [('delta_territorial_Mt','territorial_kg_CO2'),('delta_imported_footprint_Mt','imported_emitting_sector_kg_CO2')]:
        error=float(q[delta].sum()-(r.loc[end,level]-r.loc[start-1,level])/1e9)
        assert abs(error)<1e-8;res.append(error)
      selected.append(q)
    pd.concat(selected).to_csv(OUT/'selected_industry_annual_paths.csv',index=False)
    links=pd.read_csv(BASE/'trade_inputs/crop_livestock_links.csv')
    m=links[(links.destination_country=='MX')&(links.using_sector==5)&(links.input_sector==1)]
    rows=[]
    for y in [2006,2007,2008]:
      before=m[(m.year==y-1)&(m.price_basis=='CYP')].set_index('source_country').million_EUR
      after=m[(m.year==y)&(m.price_basis=='PYP')].set_index('source_country').million_EUR
      dom=after['MX']-before['MX'];imp=after.drop('MX').sum()-before.drop('MX').sum();us=after['US']-before['US']
      assert dom<-1 and imp>1 and us>1
      rows.append(dict(year=y,price_year=y-1,domestic_change_million_EUR=dom,import_change_million_EUR=imp,US_change_million_EUR=us))
    pd.DataFrame(rows).to_csv(OUT/'mexico_input_annual_checks.csv',index=False)
    report=dict(passed=True,max_endpoint_identity_residual_Mt=max(abs(x) for x in res),
      selected_industrial_years=6,selected_input_years=3,
      limitations=['Consecutive-year diagnostics do not replace P1/P2/P3 reporting periods.',
        'Different preceding-year monetary changes are not added together.',
        'Emitting-sector imported footprints are not product import emissions.',
        'Italy crude steel output rose in external physical statistics; do not infer broad production relocation.',
        'Japan pattern reverses in 2009. Mexico purchases are monetary livestock inputs, not a physical feed balance.'],
      sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [BASE/'fine_followup/sector_levels.csv',BASE/'followup/annual_sector_changes.csv',BASE/'followup/annual_sector_import_origins.csv',BASE/'followup/cross_group_input_screen.csv',BASE/'trade_inputs/crop_livestock_links.csv']],
      outputs=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(OUT.glob('*.csv'))])
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['passed','max_endpoint_identity_residual_Mt','selected_industrial_years','selected_input_years']},indent=2))
if __name__=='__main__':main()
