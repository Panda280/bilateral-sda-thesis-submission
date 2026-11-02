"""Reproduce the report's case selection and complete three-grid periods."""
from pathlib import Path
import pandas as pd

B = Path(__file__).resolve().parents[1] / 'docs/analysis/full_drilldown_20260910'
rep = pd.read_csv(B / 'repeated_candidates.csv', dtype={'grid': str, 'sector': str})
j = pd.read_csv(B / 'all_annual_joint_tests.csv', dtype={'grid': str, 'sector': str})
rows = []
for _, group in rep.groupby(['period', 'effect', 'country', 'sector'], sort=True):
    row = group.sort_values('grid', ascending=False).iloc[0].copy()
    annual = j[(j.grid == row.grid) & (j.period == row.period) &
               (j.effect == row.effect) & (j.country == row.country) &
               (j.sector == row.sector)]
    row['case_id'] = f'H{len(rows)+1:02d}'
    row['qualifying_years'] = ', '.join(map(str, sorted(annual.loc[annual.signature, 'year'])))
    stable = True
    for year in annual.year:
        rest = annual[annual.year != year]
        stable &= (rest.Domestic.sum() < 0 and rest.outside_source.sum() > 0
                   and rest.Home.sum() < 0 and rest.outside_emitter.sum() > 0)
    row['source_sign_stable_omit_year'] = stable
    row['period_parent_negative'] = row.parent_Dint_Mt < 0
    row['period_source_material_01'] = (row.domestic_Mt < -.1 and
        row.outside_source_Mt > .1 and row.home_Mt < -.1 and row.outside_emitter_Mt > .1)
    rows.append(row)
pd.DataFrame(rows).to_csv(B / 'highlighted_cases.csv', index=False)
t = pd.read_csv(B / 'three_grid_parent_comparison.csv')
t = t[t.full_chain].copy()
t['period'] = t.year.map(lambda y: 'P1' if y <= 2004 else 'P2' if y <= 2007 else 'P3')
p = t.groupby(['country', 'group', 'parent', 'period']).agg(
    years=('year', 'size'), R3_Mt=('fine_Dint_Mt', 'sum'),
    WIOT2016_Mt=('Dint16_Mt', 'sum'), WIOT2013_Mt=('Dint13_Mt', 'sum')).reset_index()
p['complete'] = p.years == p.period.map({'P1': 4, 'P2': 3, 'P3': 5})
p.to_csv(B / 'complete_three_grid_periods.csv', index=False)
print(f'{len(rows)} distinct cases; {int(p.complete.sum())} complete three-grid periods')
