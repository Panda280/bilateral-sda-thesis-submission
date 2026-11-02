"""Independent identities, influence checks and case extracts for follow-up."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from build_results_trade_associations import estimate
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
OUT=BASE/'followup';REGIONS={'WA','WE','WF','WL','WM','ROW','RoW','WT'}

def main():
    a=pd.read_csv(OUT/'annual_country_attribution.csv',dtype={'resolution':str})
    world=pd.read_parquet(ROOT/'data/processed/provisional_panel_v2/cross_year_panel/global_hierarchy.parquet')
    w=world[world.year_current.between(2001,2004)].set_index(['resolution','year_current','effect_id']).contribution
    sums=a.groupby(['resolution','year_current','effect_id','role']).contribution.sum().reset_index()
    sums['world']=sums.set_index(['resolution','year_current','effect_id']).index.map(w)
    assert sums.world.notna().all()
    err=float((abs(sums.contribution-sums.world)/np.maximum(abs(sums.world),1e9)).max());assert err<1e-5
    q=pd.read_csv(OUT/'annual_sector_changes.csv',dtype={'sector':str});q=q[~q.country.isin(REGIONS)]
    r=pd.read_csv(OUT/'sector_associations.csv',dtype={'sector':str})
    candidates=r[(r.two_way_slope<0)&(r.q_BH<.05)]
    rows=[];independent=[]
    for c in candidates.itertuples():
      d=q[q.sector.eq(c.sector)].copy()
      if c.scope=='Reduction':d=d[d.group.eq('Red')]
      fit=smf.ols('delta_territorial_Mt ~ '+c.x+' + C(country) + C(year)',d).fit(cov_type='cluster',cov_kwds={'groups':d.country})
      assert abs(fit.params[c.x]-c.two_way_slope)<1e-8
      assert abs(fit.bse[c.x]-c.cluster_se)<1e-8
      independent.append(dict(sector=c.sector,scope=c.scope,x=c.x,coefficient_error=fit.params[c.x]-c.two_way_slope,se_error=fit.bse[c.x]-c.cluster_se))
      for kind in ['country','year']:
        for omitted in d[kind].unique():
          e=estimate(d[d[kind].ne(omitted)],c.x)
          rows.append(dict(sector=c.sector,scope=c.scope,x=c.x,omission_kind=kind,omitted=omitted,**e))
    inf=pd.DataFrame(rows);inf.to_csv(OUT/'negative_sector_influence.csv',index=False)
    pd.DataFrame(independent).to_csv(OUT/'independent_sector_regression_checks.csv',index=False)
    summaries=inf.groupby(['sector','scope','x','omission_kind']).two_way_slope.agg(['min','max']);summaries.to_csv(OUT/'negative_sector_influence_summary.csv')
    print(summaries.to_string())
    links=pd.read_csv(BASE/'trade_inputs/crop_livestock_links.csv')
    ca=links[(links.destination_country=='CA')&(links.input_sector==1)&(links.using_sector==4)].copy()
    ca['origin']=ca.source_country.where(ca.source_country.isin(['CA','US']),'Other')
    ca=ca.groupby(['year','price_basis','origin']).million_EUR.sum().unstack('origin').reset_index()
    ca['import_share']=(ca.US+ca.Other)/(ca.CA+ca.US+ca.Other)
    ca.to_csv(OUT/'canada_cereal_cattle_path.csv',index=False)
    case=pd.read_csv(OUT/'cross_group_candidates.csv')
    case=case[(case.destination_country=='CA')&(case.year==2001)&(case.input_sector==1)&(case.using_sector==4)].iloc[0]
    origin=pd.read_csv(OUT/'cross_group_candidate_origins.csv')
    origin=origin[(origin.destination_country=='CA')&(origin.year==2001)&(origin.input_sector==1)&(origin.using_sector==4)]
    assert np.isclose(origin[origin.source_country.ne('CA')].change_million_EUR.sum(),case.imports_change)
    assert np.isclose(origin[origin.source_group.ne('Red')].change_million_EUR.sum(),case.cross_group_change)
    intro=hashlib.sha256((ROOT/'src/chapters/01_introduction.tex').read_bytes()).hexdigest()
    assert intro=='2ea30046ae703d836a7256a5cb993606d4a593befe048807e324d5ce9fe68b43'
    inputs=list((BASE/'fine_followup').glob('*'))+[BASE/'trade_inputs/crop_livestock_links.csv',BASE/'trade/annual_country_changes.csv']
    inputs += list((ROOT/'data/processed/provisional_panel_v2/cross_year_panel').glob('*country.parquet'))
    inputs += [ROOT/'data/processed/provisional_panel_v2/cross_year_panel/global_hierarchy.parquet']
    inputs += [ROOT/'references/current_codebase/concordance/normalized_2026-08-31'/f for f in ['country_concordance_by_resolution.csv','analytical_group_registry.csv','sector_concordance_by_resolution.csv']]
    manifest=dict(passed=True,world_attribution_max_relative_residual=err,introduction_sha256=intro,
        independent_sector_models=len(independent),country_and_year_omissions=len(inf),canada_origin_reconciliation=True,
        inputs=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(inputs)],
        outputs=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(OUT.glob('*.csv'))])
    (OUT/'validation_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
