import json

import pyarrow.parquet as pq
import pytest

from sda.output.annual import REGISTRY_PATH, load_dashboard_registry, publish_annual_report, publish_panel
from test_annual_reporting import _result
from test_production_bridge import _build_state, _consumed_files
from build_annual_reports import _technical_source_sector_rows
from sda.integration import write_state_bundle
from sda.decomposition import run_canonical_sda_state


def test_registry_is_versioned_complete_and_has_no_year_panel_partitions():
    registry = load_dashboard_registry()
    assert registry["contract_version"] == 2
    assert len(registry["panels"]) == 11
    assert set(registry["products"]) == {"annual_hierarchy", "annual_drilldown", "cross_year_panel"}
    panel = registry["products"]["cross_year_panel"]
    assert "year_previous" not in panel["partition_keys"]
    assert "year_current" not in panel["partition_keys"]
    assert {"year_previous", "year_current"} == set(panel["forbidden_partition_keys"])
    assert json.loads(REGISTRY_PATH.read_text())["language_policy"] == "accounting_attribution"


def test_annual_columns_metrics_and_publishable_filter_follow_registry(tmp_path):
    root = publish_annual_report(_result(), tmp_path, run_id="r", year_prev=2000, year=2001,
        resolution="test", units="kg", source_files=[{"path": "x"}],
        supplementary_drilldown=[{"effect_id": "D_C", "grain_id": "source_country",
            "source_country": "A", "contribution": 9.0, "positive_contribution": 9.0,
            "negative_contribution": 0.0, "gross_contribution": 9.0,
            "denominator_value": 9.0, "denominator_status": "ok",
            "net_pct_of_parent_gross": 100.0, "is_publishable": True,
            "validation_status": "passed"},
            {"effect_id": "D_C", "grain_id": "source_country", "source_country": "hidden",
             "is_publishable": False}])
    registry = load_dashboard_registry()
    hierarchy = pq.read_table(root / "annual_hierarchy.parquet")
    drilldown = pq.read_table(root / "annual_drilldown.parquet")
    assert hierarchy.column_names == registry["products"]["annual_hierarchy"]["columns"]
    assert drilldown.column_names == registry["products"]["annual_drilldown"]["columns"]
    assert "hidden" not in drilldown.column("source_country").to_pylist()
    for row in hierarchy.to_pylist() + drilldown.to_pylist():
        assert row["positive_contribution"] >= 0
        assert row["negative_contribution"] <= 0
        assert row["gross_contribution"] == row["positive_contribution"] - row["negative_contribution"]


def test_panel_writes_exactly_eleven_flat_files(tmp_path):
    annual = publish_annual_report(_result(), tmp_path / "annual", run_id="r", year_prev=2000,
        year=2001, resolution="test", units="kg", source_files=[{"path": "x"}])
    panel = publish_panel([annual], tmp_path / "panel", run_id="p")
    files = sorted((panel / "cross_year_panel").glob("*.parquet"))
    assert len(files) == 11
    assert not any(path.is_dir() for path in (panel / "cross_year_panel").iterdir())


def test_nested_technical_source_sector_split_reconciles(tmp_path):
    state = _build_state()
    bundle = tmp_path / "bundle"
    write_state_bundle(state, bundle, consumed_files=_consumed_files(tmp_path))
    result = run_canonical_sda_state(state)
    rows = _technical_source_sector_rows(bundle, result, {})
    for effect in ("D_T_coef", "D_w_coef"):
        sector_rows = [row for row in rows if row["effect_id"] == effect and row["grain_id"] == "source_sector"]
        assert sector_rows
        assert sum(row["contribution"] for row in sector_rows) == pytest.approx(result.effects[effect].baseline.sum())
