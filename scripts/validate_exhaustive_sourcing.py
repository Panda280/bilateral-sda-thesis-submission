"""Independent coverage, margin and coefficient-identity checks for delivery."""
from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
B=AN/'exhaustive_sourcing_20260910'
p=pd.read_csv(B/'all_pairs_annual.csv',dtype={'source':str,'user':str})
s=pd.read_csv(B/'all_country_product_annual.csv',dtype={'sector':str})
h=pd.read_csv(B/'all_hierarchy_annual.csv',dtype={'grid':str,'sector':str,'parent':str})
hp=pd.read_csv(B/'all_hierarchy_period.csv')
assert len(p)==98*98*12 and not p.duplicated(['source','user','year']).any()
assert p.groupby(['source','user']).year.nunique().eq(12).all()
assert len(s)==98*22*12*2 and not s.duplicated(['country','sector','year','effect']).any()
a=p.groupby(['year','source'])[['Domestic','foreign','outside']].sum()
c=s[s.effect=='D_C'].groupby(['year','sector'])[['Domestic','foreign','outside_source']].sum().rename(columns={'outside_source':'outside'})
a.index.names=c.index.names
margin_error=float((abs(a-c)/np.maximum(abs(c),1)).max().max())
assert margin_error<1e-5
# Each centered product allocation must sum to the exact parent composition.
f=pd.read_csv(AN/'hierarchical_intensity_20260910/fine_cell_diagnostics.csv',dtype={'grid':str,'sector':str,'parent':str})
pa=pd.read_csv(AN/'hierarchical_intensity_20260910/parent_annual_diagnostics.csv',dtype={'grid':str,'parent':str})
e=pd.read_parquet(AN/'hierarchical_intensity_20260910/sector_Dint.parquet')
e=e[e.resolution=='R3'].copy();e.sector=e.sector.astype(str)
f=f.merge(e[['year','country','sector','bundle_v0','bundle_v1']],on=['year','country','sector'],validate='many_to_one')
f=f.merge(pa[['grid','year','country','parent','v0','v1','composition']],on=['grid','year','country','parent'],validate='many_to_one')
f['centered']=(f.bundle_v0+f.bundle_v1-f.v0-f.v1)/2*f.share_change
g=f.groupby(['grid','year','country','parent']).agg(centered=('centered','sum'),composition=('composition','first'),v0=('v0','first'))
centered_error=float((abs(g.centered-g.composition)/np.maximum(abs(g.v0),1)).max())
assert centered_error<1e-10
chain_error=float(abs(hp.chained_intensity_change_pct-hp.chained_composition_pp-hp.chained_within_pp).max())
assert chain_error<1e-8
# Coverage lists all products even when no hierarchical test is available.
sm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/granularity_sector_mapping_review.csv',dtype=str)
coverage=[]
for grid in ['2013','2016']:
    for sector in map(str,range(1,99)):
        q=h[(h.grid==grid)&(h.sector==sector)&(h.effect=='D_C')]
        m=sm[(sm.coarse_resolution==grid)&(sm.fine_sector==sector)]
        coverage.append(dict(grid=grid,sector=sector,product_name=m.fine_sector_label.iloc[0],
            mapping_status=m.mapping_status.iloc[0],possible_country_years=22*12,
            available_parent_tests=len(q),comparable_parent_tests=int(q.comparable.sum()),
            unavailable_or_noncomparable=22*12-int(q.comparable.sum())))
pd.DataFrame(coverage).to_csv(B/'product_hierarchy_coverage.csv',index=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
intro=ROOT/'src/chapters/01_introduction.tex'
assert sha(intro)=='2ea30046ae703d836a7256a5cb993606d4a593befe048807e324d5ce9fe68b43'
v=dict(passed=True,all_pairs_complete=True,all_country_products_complete=True,
       margin_max_scaled_error=margin_error,centered_composition_max_scaled_error=centered_error,
       chain_reconstruction_max_error_pp=chain_error,introduction_unchanged=True,
       artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [ROOT/'out/exhaustive_sourcing_intensity.pdf',B/'REPORT.md',B/'report_cases.csv']])
(B/'delivery_validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
print(json.dumps(v,indent=2))
