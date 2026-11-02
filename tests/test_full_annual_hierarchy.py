import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_full_annual_hierarchy.py"
SPEC = importlib.util.spec_from_file_location("full_hierarchy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def _digest(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _campaign(root, *, units="kg CO2"):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "canonical.bin"; source.write_bytes(b"canonical")
    source_record = {"path": str(source), **_digest(source)}
    for run_id, (previous, current, resolution) in MODULE._expected().items():
        directory = root / "annual" / run_id; directory.mkdir(parents=True)
        rows = [{"run_id": run_id, "year_previous": previous, "year_current": current,
                 "resolution": resolution, "units": units, "node_id": node,
                 "parent_node_id": None, "depth": 0, "contribution": 0.0,
                 "positive_contribution": 0.0, "negative_contribution": 0.0,
                 "gross_contribution": 0.0, "denominator_value": None,
                 "denominator_status": "not_applicable", "net_pct_of_parent_gross": None,
                 "is_publishable": True, "validation_status": "passed"} for node in MODULE.NODES]
        hierarchy = directory / "annual_hierarchy.parquet"
        pq.write_table(pa.Table.from_pylist(rows), hierarchy)
        manifest = {"report_schema_version": 2, "run_id": run_id, "year_pair": [previous, current],
                    "resolution": resolution, "units": units, "validation_status": "passed",
                    "source_files": [source_record], "artifacts": {hierarchy.name: _digest(hierarchy)}}
        (directory / "manifest.v2.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "annual"


def test_builds_exact_full_window_and_discloses_upstream_failure(tmp_path):
    annual = _campaign(tmp_path)
    audit = tmp_path / "audit.json"; audit.write_text(json.dumps({"status": "failed", "passed_years": 0, "expected_years": 15}))
    config = tmp_path / "config.json"; config.write_text(json.dumps({"price_basis": "CYP_(t-1) -> PYP_t", "indicator": "fossil-origin CO2", "units_input": "kg CO2", "units_output": "Gt CO2", "unit_conversion_factor": 1e-12}))
    output = MODULE.build_report(annual, tmp_path / "analysis", upstream_audit=audit, analysis_config=config, run_id="test")
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["coverage"] == {"expected": 42, "accepted": 42, "missing": [], "excluded": []}
    assert manifest["status"] == "complete"
    assert manifest["validation"]["upstream_input_status"] == "failed"
    assert len((output / "annual_hierarchy_full.csv").read_text().splitlines()) == 42 * 12 + 1
    assert len((output / "cumulative_hierarchy_full.csv").read_text().splitlines()) == 3 * 12 + 1


def test_rejects_missing_run_and_changed_source(tmp_path):
    annual = _campaign(tmp_path)
    missing = annual / "2000_2001_2013" / "manifest.v2.json"; missing.unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        MODULE.load_full_hierarchy(annual)
    missing.parent.rmdir() if not any(missing.parent.iterdir()) else None
    # Recreate separately because the missing directory still contains its parquet.
    other = tmp_path / "other"; annual = _campaign(other)
    (other / "canonical.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source checksum mismatch"):
        MODULE.load_full_hierarchy(annual)
