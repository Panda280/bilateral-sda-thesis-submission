import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "campaign_results_overview.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_dashboard_notebook_is_valid_and_all_code_cells_compile():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_dashboard_covers_annual_drilldowns_panels_sources_and_validation():
    source = "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])
    required = {
        "annual_hierarchy.parquet", "annual_drilldown.parquet", "cross_year_panel",
        "D_C.source_country_cube.parquet", "D_C.source_group_sector_cube.parquet",
        "D_YF.source_country_cube.parquet", "D_YF.source_group_sector_cube.parquet",
        "net_pct_of_parent_gross", "denominator_status", "source_sector",
        "consumer_country", "demanded_sector", "driver_country",
    }
    assert required <= {item for item in required if item in source}
    assert "accounting-attribution" in source
    assert "causal responsibility" not in source.lower()
    assert "REGISTRY['panels']" in source
    assert "is_publishable" in source
