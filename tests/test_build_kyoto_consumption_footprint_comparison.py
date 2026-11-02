from pathlib import Path
import importlib.util
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_kyoto_consumption_footprint_comparison.py"
spec = importlib.util.spec_from_file_location("kyoto_footprint", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sheet_reader_uses_documented_rows(tmp_path):
    path = tmp_path / "fixture.xlsx"
    raw = pd.DataFrame([[None, "title"], [None, "Code"], [None, "Country"], [1990, 2.0], [2008, 3.0]])
    with pd.ExcelWriter(path) as writer:
        raw.to_excel(writer, index=False, header=False, sheet_name="Sheet")
    got = module.read_sheet(path, "Sheet", 2, 3)
    assert got.country_name.tolist() == ["Country", "Country"]
    assert got.year.tolist() == [1990, 2008]


def test_conversion_and_change_identity():
    base_mtc, endpoint_mtc = 10.0, 8.0
    base_co2, endpoint_co2 = base_mtc * 3.664, endpoint_mtc * 3.664
    assert round((endpoint_co2 / base_co2 - 1) * 100, 10) == -20.0
