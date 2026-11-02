from pathlib import Path

import pandas as pd

from scripts.plot_detailed_hierarchical_stack import LEAVES, build


def test_detailed_stack_preserves_complete_leaf_hierarchy(tmp_path: Path):
    rows = []
    for resolution in ("2013", "2016", "R3"):
        for year in range(2001, 2015):
            values = {effect: float(index - 3) for index, effect in enumerate(LEAVES)}
            for effect, value in values.items():
                rows.append({"year_current": year, "effect": effect,
                             "resolution": resolution, "units": "kg CO2", "value": value})
            rows.append({"year_current": year, "effect": "delta_g",
                         "resolution": resolution, "units": "kg CO2", "value": sum(values.values())})
    source = tmp_path / "annual.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    manifest = build(source, tmp_path / "out")
    assert manifest["validation"]["leaf_sum_reconstructs_total"]
    assert (tmp_path / "out" / "detailed_hierarchical_signed_stack.pdf").is_file()
