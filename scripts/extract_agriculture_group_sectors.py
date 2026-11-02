"""Read saved canonical SDA arrays; emit compact joint driver-country/sector data.

Run on the campaign host with Python and NumPy. No campaign files are changed.
Country groups are assigned locally from the versioned concordances.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path('/local/s2053268/sda-campaign-2000-2014-v2/sda')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    rows, controls, sources, hierarchy_checks = [], [], [], []
    for resolution in ('R3', '2016'):
        selected = {'1', '2', '3', '4', '5', '6', '97'} if resolution == 'R3' else {'A01'}
        for year in range(2001, 2013):
            run = ROOT / f'sda_{year-1}_{year}_{resolution}'
            manifest = json.loads((run / 'manifest.json').read_text())
            sources.append({'path': str(run / 'manifest.json'), 'sha256': sha(run / 'manifest.json')})
            vectors = {}
            for effect in ('D_C', 'D_A_x', 'D_T_coef', 'D_w_coef'):
                record = manifest['effects'][effect]
                path = run / record['baseline']['file']
                digest = sha(path)
                assert digest == record['baseline']['sha256'], str(path)
                sources.append({'path': str(path), 'sha256': digest})
                dim = record['dimensions']['column']
                labels = dim['labels']
                if isinstance(labels, dict):
                    labels = labels['__tuple__']
                pairs = [s.split(':', 1) for s in labels]
                a = np.load(path, mmap_mode='r', allow_pickle=False)
                assert a.shape[1] == len(pairs)
                v = a.sum(axis=0)
                assert np.isfinite(v).all()
                vectors[effect] = v
                for (country, sector), value in zip(pairs, v):
                    if sector in selected:
                        rows.append([resolution, year, effect, country, sector, float(value)])
                for role, index in [('driver_country', 0), ('driver_sector', 1)]:
                    totals = {}
                    for pair, value in zip(pairs, v):
                        totals[pair[index]] = totals.get(pair[index], 0.0) + float(value)
                    controls.extend([resolution, year, effect, role, k, v] for k, v in totals.items())
            residual = vectors['D_A_x'] - vectors['D_T_coef'] - vectors['D_w_coef']
            scale = np.maximum(1, np.abs(vectors['D_A_x']) + np.abs(vectors['D_T_coef']) + np.abs(vectors['D_w_coef']))
            hierarchy_checks.append({'run': run.name, 'max_abs_residual_kg': float(abs(residual).max()),
                                     'max_scaled_residual': float((abs(residual)/scale).max())})
            print(f'Checked {run.name}', file=sys.stderr, flush=True)
    json.dump({'columns': ['resolution', 'year_current', 'effect_id', 'country', 'sector', 'contribution'],
               'rows': rows, 'control_columns': ['resolution', 'year_current', 'effect_id', 'role', 'label', 'contribution'],
               'controls': controls, 'sources': sources, 'hierarchy_checks': hierarchy_checks, 'indicator': 'GWP100; corrected interpretation of legacy kg CO2 labels (R-2026-053)',
               'price_basis': 'CYP_(t-1) -> PYP_t', 'role': 'structural_driver'}, sys.stdout)

if __name__ == '__main__':
    main()
