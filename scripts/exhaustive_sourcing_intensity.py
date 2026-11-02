"""Exhaustive retained-data screen; effects are conditional GWP100, not trade flows.

No product or using-industry shortlist is used. Keep A/B independent of C/D,
and retain all failed and unavailable tests. Coefficient diagnostics are not SDA
effects or causal estimates. Source and emitter margins must never be added.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / 'implementation/bilateral_sda_attribution/docs/analysis'
B = AN / 'exhaustive_sourcing_20260910'
B.mkdir(exist_ok=True)
H = AN / 'hierarchical_intensity_20260910'
F = AN / 'full_drilldown_20260910'
period = lambda y: 'P1' if y <= 2004 else 'P2' if y <= 2007 else 'P3'
length = {'P1': 4, 'P2': 3, 'P3': 5}
sm = pd.read_csv(ROOT / 'references/current_codebase/concordance/normalized_2026-08-31/granularity_sector_mapping_review.csv', dtype=str)
names = sm.drop_duplicates('fine_sector').set_index('fine_sector').fine_sector_label.to_dict()

# All 98 x 98 available R3 supplier-user pairs, all 12 transitions.
d = pd.read_parquet(AN / 'ten_sda_candidates_20260910/D_C_red_interactions.parquet')
d = d[d.resolution.eq('R3')].copy()
assert d.Destination_Group.eq('Red').all() and d.effect.eq('D_C').all()
d['component'] = d.Source_Group.where(d.Relationship.ne('Domestic'), 'Domestic')
pair = d.groupby(['Source_Sector', 'Destination_Sector', 'year', 'component']).Value.sum().unstack('component').reset_index()
pair = pair.rename(columns={'Source_Sector':'source', 'Destination_Sector':'user'})
pair['source'] = pair.source.astype(str); pair['user'] = pair.user.astype(str)
pair['period'] = pair.year.map(period)
pair['foreign'] = pair.Red + pair['Non-Red'] + pair['Non-Participant']
pair['outside'] = pair['Non-Red'] + pair['Non-Participant']
assert len(pair) == 98*98*12
for c in ['Domestic', 'Red', 'Non-Red', 'Non-Participant', 'foreign', 'outside']:
    pair[c] /= 1e9
pair['AB_foreign'] = (pair.Domestic < -.1) & (pair.foreign > .1)
pair['AB_outside'] = (pair.Domestic < -.1) & (pair.outside > .1)
pair.to_csv(B/'all_pairs_annual.csv', index=False)
pp = pair.groupby(['source','user','period']).agg(
    years=('year','size'), domestic_Mt=('Domestic','sum'), foreign_Red_Mt=('Red','sum'),
    non_Red_Mt=('Non-Red','sum'), non_Participant_Mt=('Non-Participant','sum'),
    foreign_Mt=('foreign','sum'), outside_Mt=('outside','sum'),
    annual_AB_foreign=('AB_foreign','sum'), annual_AB_outside=('AB_outside','sum')).reset_index()
pp['period_AB_foreign'] = (pp.domestic_Mt < -.1) & (pp.foreign_Mt > .1)
pp['period_AB_outside'] = (pp.domestic_Mt < -.1) & (pp.outside_Mt > .1)
pp['repeated_foreign'] = pp.period_AB_foreign & (pp.annual_AB_foreign >= 2)
pp['repeated_outside'] = pp.period_AB_outside & (pp.annual_AB_outside >= 2)
pp['ranking_scale_Mt'] = np.minimum(-pp.domestic_Mt, pp.foreign_Mt)
pp['source_name'] = pp.source.map(names); pp['user_name'] = pp.user.map(names)
pp.sort_values('ranking_scale_Mt',ascending=False).to_csv(B/'all_pairs_period.csv', index=False)

# All Reduction-country products, whether or not a hierarchical test is possible.
groups = pd.read_csv(AN/'results_hypotheses_20260909/trade_inputs/country_levels_and_trade.csv')
red = set(groups.loc[(groups.grid=='R3') & (groups.group=='Red'), 'country'])
s = pd.read_csv(F/'annual_sourcing_slices.csv', dtype={'source_sector':str})
s = s[s.destination_country.isin(red)].rename(columns={'destination_country':'country','source_sector':'sector'})
s['period'] = s.year.map(period)
s['foreign'] = s.Red + s.outside_source
for c in ['Domestic','Red','outside_source','foreign','Home','outside_emitter']:
    s[c] /= 1e9
e = pd.read_parquet(H/'sector_Dint.parquet')
e = e[e.resolution=='R3'].copy(); e.sector = e.sector.astype(str)
l = pd.read_parquet(H/'sector_levels.parquet')
l = l[l.resolution=='R3'].copy(); l.sector = l.sector.astype(str)
for side,basis in [(0,'CYP'),(1,'PYP')]:
    z = l[l.basis==basis].copy(); z.year += 1-side
    e = e.merge(z[['year','country','sector','x']].rename(columns={'x':f'x{side}'}), on=['year','country','sector'], validate='one_to_one')
e['direct_intensity_falls'] = e.bundle_v1 < e.bundle_v0
s = s.merge(e[['year','country','sector','D_v','bundle_v0','bundle_v1','x0','x1','direct_intensity_falls']], on=['year','country','sector'], validate='many_to_one')
s['AB'] = (s.Domestic < -.1) & (s.foreign > .1)
s['AB_outside'] = (s.Domestic < -.1) & (s.outside_source > .1)
s['ABC_fine'] = s.AB & s.direct_intensity_falls
s.to_csv(B/'all_country_product_annual.csv',index=False)
cp = s.groupby(['effect','country','sector','period']).agg(years=('year','size'),domestic_Mt=('Domestic','sum'),
    foreign_Mt=('foreign','sum'),outside_Mt=('outside_source','sum'),annual_AB=('AB','sum'),
    annual_ABC_fine=('ABC_fine','sum'),Dint_kg=('D_v','sum')).reset_index()
cp['period_AB'] = (cp.domestic_Mt < -.1) & (cp.foreign_Mt > .1)
cp['repeated_AB'] = cp.period_AB & (cp.annual_AB >= 2)
cp['product_name'] = cp.sector.map(names)
cp.to_csv(B/'all_country_product_period.csv',index=False)

# C/D: same-country parent intensity, positive output, exact reviewed mapping.
fine = pd.read_csv(H/'fine_cell_diagnostics.csv',dtype={'grid':str,'sector':str,'parent':str})
parent = pd.read_csv(H/'parent_annual_diagnostics.csv',dtype={'grid':str,'parent':str})
fine = fine[fine.group.eq('Red')].copy()
h = fine.merge(parent[['grid','year','country','parent','v0','v1','delta_v','within','composition','composition_relative','within_relative','coarse_Dint_Mt','fine_Dint_Mt']], on=['grid','year','country','parent'],validate='many_to_one')
h = pd.concat([h.merge(s[s.effect==eff],on=['year','period','country','sector'],validate='many_to_one',suffixes=('','_s')) for eff in ['D_C','D_YF']],ignore_index=True)
h['product_intensity_ratio'] = h.initial_v / h.parent_v
h['output_change_pct'] = 100*(h.x1/h.x0-1)
h['parent_change_pct'] = 100*h.delta_v/h.v0
h['centered_product_composition'] = ((h.bundle_v0+h.bundle_v1-h.v0-h.v1)/2)*h.share_change
h['C'] = h.delta_v < 0
h['D'] = h.above_parent & (h.share_change < 0) & (h.x1 < h.x0) & (h.composition < 0) & (h.centered_product_composition < 0)
h['ABCD'] = h.comparable & h.AB & h.C & h.D
h['ABCD_outside'] = h.comparable & h.AB_outside & h.C & h.D
h['composition_necessary'] = h.ABCD & (h.within >= 0)
h['negative_parent_Dint'] = h.coarse_Dint_Mt < 0
h.to_csv(B/'all_hierarchy_annual.csv',index=False)
rows = []
for key,q in h.groupby(['grid','effect','country','sector','parent','period']):
    q=q.sort_values('year'); r=dict(zip(['grid','effect','country','sector','parent','period'],key))
    r.update(years=len(q),complete=len(q)==length[r['period']], comparable=bool(q.comparable.all()),
        annual_AB=int(q.AB.sum()), annual_ABC=int((q.AB&q.C&q.comparable).sum()),
        annual_ABCD=int(q.ABCD.sum()), annual_ABCD_outside=int(q.ABCD_outside.sum()),
        necessary_years=int(q.composition_necessary.sum()), domestic_Mt=q.Domestic.sum(),
        foreign_Mt=q.foreign.sum(), outside_Mt=q.outside_source.sum(),
        home_Mt=q.Home.sum(), outside_emitter_Mt=q.outside_emitter.sum(),
        parent_Dint_Mt=q.coarse_Dint_Mt.sum(), fine_parent_Dint_Mt=q.fine_Dint_Mt.sum(),
        product_Dint_Mt=q.Dint_Mt.sum(), qualifying_years=', '.join(map(str,q.loc[q.ABCD,'year'])))
    # Chain the annual CYP(t-1)->PYP(t) ratios; do not subtract levels in different price bases.
    index=1.; comp=0.; within=0.; product_comp=0.
    for z in q.itertuples():
        comp += index*z.composition/z.v0
        within += index*z.within/z.v0
        product_comp += index*z.centered_product_composition/z.v0
        index *= z.v1/z.v0
    assert abs(index-1-comp-within)<1e-10
    r.update(chained_intensity_change_pct=100*(index-1),chained_composition_pp=100*comp,
             chained_within_pp=100*within,chained_selected_product_composition_pp=100*product_comp)
    r['period_AB'] = r['domestic_Mt'] < -.1 and r['foreign_Mt'] > .1
    r['period_C'] = index < 1
    r['period_composition_negative'] = comp < 0
    r['repeated_ABCD'] = r['complete'] and r['comparable'] and r['period_AB'] and r['period_C'] and comp<0 and r['annual_ABCD']>=2
    r['ranking_scale_Mt'] = min(-r['domestic_Mt'],r['foreign_Mt'])
    r['source_sign_stable_omit_year'] = all(q.Domestic.sum()-z.Domestic < 0 and q.foreign.sum()-z.foreign>0 for z in q.itertuples())
    r['product_name'] = names.get(r['sector'],r['sector'])
    rows.append(r)
hp=pd.DataFrame(rows).sort_values('ranking_scale_Mt',ascending=False)
hp.to_csv(B/'all_hierarchy_period.csv',index=False)
selected=hp[hp.repeated_ABCD].copy()
selected['both_mappings']=selected.groupby(['effect','country','sector','period']).grid.transform('nunique')==2
selected.to_csv(B/'repeated_ABCD_candidates.csv',index=False)

# Threshold sensitivity repeats the entire joint screen, not just retained cases.
sens=[]
for threshold in [0,.01,.1,.5,1]:
    for foreign in ['foreign','outside_source']:
        z=h.copy();z['pass_now']=z.comparable & (z.Domestic < -threshold) & (z[foreign] > threshold) & z.C & z.D
        z=z.groupby(['grid','effect','country','sector','parent','period']).agg(passes=('pass_now','sum'),
            domestic=('Domestic','sum'),foreign=(foreign,'sum')).reset_index()
        z=z.merge(hp[['grid','effect','country','sector','parent','period','complete','comparable','period_C','period_composition_negative']],on=['grid','effect','country','sector','parent','period'],validate='one_to_one')
        z=z[(z.passes>=2)&(z.domestic < -threshold)&(z.foreign>threshold)&z.complete&z.comparable&z.period_C&z.period_composition_negative]
        sens.append(dict(threshold_Mt=threshold,foreign_definition=foreign,mapping_cases=len(z),distinct_cases=len(z[['effect','country','sector','period']].drop_duplicates())))
pd.DataFrame(sens).to_csv(B/'threshold_sensitivity.csv',index=False)
summary=dict(indicator='Conditional GWP100; not fossil CO2',pair_count=98*98,pair_annual_rows=len(pair),pair_period_rows=len(pp),
    repeated_pair_foreign=int(pp.repeated_foreign.sum()),repeated_pair_outside=int(pp.repeated_outside.sum()),
    reduction_countries=len(red),country_product_rows=len(s),country_products=s.sector.nunique(),
    repeated_country_product_AB=int(cp.repeated_AB.sum()),hierarchy_annual_rows=len(h),
    hierarchy_products=h.sector.nunique(),comparable_hierarchy_products=h.loc[h.comparable,'sector'].nunique(),
    repeated_ABCD_mapping_cases=len(selected),repeated_ABCD_distinct=len(selected[['effect','country','sector','period']].drop_duplicates()),
    source_files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [F/'annual_sourcing_slices.csv',H/'fine_cell_diagnostics.csv',H/'parent_annual_diagnostics.csv',AN/'ten_sda_candidates_20260910/D_C_red_interactions.parquet']],
    limitations=['Pair data have no individual destination-country axis; no country-specific link is inferred.',
    'D_C and D_YF contributions are not observed import quantities or a causal offshoring amount.',
    'C/D unavailable for unverified mappings, nonpositive output or country-merging parents.',
    'Exhaustive over retained 2001-2012 campaign data, not every possible causal hypothesis.'])
(B/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='source_files'},indent=2))
print(selected[['grid','effect','country','sector','period','annual_ABCD','necessary_years','domestic_Mt','foreign_Mt','chained_intensity_change_pct','chained_composition_pp','chained_within_pp','parent_Dint_Mt']].head(30).to_string(index=False))
