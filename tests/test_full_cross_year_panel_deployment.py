import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_full_cross_year_panel.py"
SPEC = importlib.util.spec_from_file_location("panel_deploy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)
REGISTRY = json.loads((Path(__file__).parents[1] / "src/sda/output/dashboard_reporting_contract_v2.json").read_text())


def _record(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _annual(root):
    source = root / "source.bin"; root.mkdir(parents=True); source.write_bytes(b"source")
    for run_id, (previous, current, resolution) in MODULE.EXPECTED.items():
        directory = root / "annual" / run_id; directory.mkdir(parents=True)
        context = {"run_id": run_id, "year_previous": previous, "year_current": current,
                   "resolution": resolution, "units": "kg CO2"}
        hierarchy = [{**context, "node_id": "delta_g", "parent_node_id": None, "depth": 0,
                      "contribution": 0., "positive_contribution": 0., "negative_contribution": 0.,
                      "gross_contribution": 0., "denominator_value": None, "denominator_status": "not_applicable",
                      "net_pct_of_parent_gross": None, "is_publishable": True, "validation_status": "passed"}]
        drilldown = []
        artifacts = {}
        for name, rows, columns in (("annual_hierarchy.parquet", hierarchy, REGISTRY["products"]["annual_hierarchy"]["columns"]),
                                    ("annual_drilldown.parquet", drilldown, REGISTRY["products"]["annual_drilldown"]["columns"])):
            path = directory / name; pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([(c, pa.string()) for c in columns])) if not rows else pa.Table.from_pylist(rows), path); artifacts[name] = _record(path)
        for name in ("annual_overview.md", "annual_summary.v2.json"):
            path = directory / name; path.write_text("{}" if name.endswith("json") else "# test\n"); artifacts[name] = _record(path)
        manifest = {"report_schema_version": 2, **context, "year_pair": [previous, current], "validation_status": "passed",
                    "source_files": [{"path": str(source), **_record(source)}], "artifacts": artifacts}
        (directory / "manifest.v2.json").write_text(json.dumps(manifest))
    return root / "annual"


def test_blocked_deployment_writes_evidence_without_panel(tmp_path):
    annual = tmp_path / "empty"; annual.mkdir()
    metadata = tmp_path / "metadata.json"; metadata.write_text(json.dumps({"price_basis": "CYP_(t-1) -> PYP_t", "indicator": "fossil-origin CO2", "units": "kg CO2"}))
    output, complete = MODULE.deploy(annual, tmp_path / "out", campaign_metadata=metadata, analysis_id="blocked", command=["test"])
    assert not complete
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["status"] == "blocked" and manifest["coverage"]["accepted"] == 0
    assert not (output / "panel_product").exists()


def test_complete_deployment_has_exact_panel_inventory(tmp_path):
    annual = _annual(tmp_path / "campaign")
    metadata = tmp_path / "metadata.json"; metadata.write_text(json.dumps({"price_basis": "CYP_(t-1) -> PYP_t", "indicator": "fossil-origin CO2", "units": "kg CO2"}))
    output, complete = MODULE.deploy(annual, tmp_path / "out", campaign_metadata=metadata, analysis_id="complete", command=["test"])
    assert complete
    summary = json.loads((output / "panel_product/panel_summary.v2.json").read_text())
    assert summary["annual_runs"] == 42 and summary["panel_count"] == 11
    assert len(list((output / "panel_product/cross_year_panel").glob("*.parquet"))) == 11
