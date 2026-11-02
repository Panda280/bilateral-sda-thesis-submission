import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "native_wiod_impacts.py"
SPEC = importlib.util.spec_from_file_location("native_wiod_impacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_jrc_loader_keeps_households_separate_and_sums_components(tmp_path):
    countries = [f"C{i:02d}" for i in range(44)]
    sectors = [f"S{i:02d}" for i in range(56)]
    rows = [
        {"year": 2004, "country": country, "sector": sector, "role": "industry", "CO2en_Gg": 2, "CO2nen_Gg": 3}
        for country in countries for sector in sectors
    ]
    rows.append({"year": 2004, "country": countries[0], "sector": "FC_HH", "role": "household_direct", "CO2en_Gg": 7, "CO2nen_Gg": 11})
    path = tmp_path / "jrc.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = MODULE.load_wiod2016_jrc_canonical(path, 2004)
    assert len(result.values_kg_co2) == 44 * 56
    assert result.values_kg_co2.iloc[0] == 5_000_000
    assert result.household_direct_kg_co2.iloc[0] == 18_000_000


def test_unreviewed_sector_mapping_blocks_aggregation(tmp_path):
    values = pd.Series(
        [1.0], index=pd.MultiIndex.from_tuples([("C00", "S00")], names=["country", "sector"])
    )
    impact = MODULE.NativeImpact(values, pd.Series(dtype=float), "db", "release", 2004, "CO2", "boundary", "grid", "Gg", 1, "p", "h")
    path = tmp_path / "map.csv"
    pd.DataFrame([{"native_sector": "S00", "target_sector": "T00", "weight": 1, "review_status": "manual_review_required"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not fully reviewed"):
        MODULE.aggregate_2016_sectors(impact, path)
