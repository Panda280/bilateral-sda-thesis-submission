import hashlib
import json

from audit_reporting_deployment import (
    ANNUAL_ARTIFACTS,
    _annual_status,
    audit_reporting,
    expected_runs,
)


def _record(path):
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _annual(root, source):
    run_id = "2000_2001_2013"
    destination = root / run_id
    destination.mkdir(parents=True)
    artifacts = {}
    for name in ANNUAL_ARTIFACTS:
        path = destination / name
        path.write_text(name, encoding="utf-8")
        artifacts[name] = _record(path)
    manifest = {
        "run_id": run_id,
        "year_pair": [2000, 2001],
        "resolution": "2013",
        "validation_status": "passed",
        "artifacts": artifacts,
        "source_files": [_record(source)],
    }
    (destination / "manifest.v2.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_id


def test_annual_status_detects_current_then_changed_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("one", encoding="utf-8")
    annual = tmp_path / "annual"
    run_id = _annual(annual, source)

    assert _annual_status(annual, run_id)["status"] == "current"
    source.write_text("two", encoding="utf-8")
    status = _annual_status(annual, run_id)
    assert status["status"] == "stale_source"
    assert status["stale_sources"] == [str(source.resolve())]


def test_campaign_audit_requires_exact_42_current_runs(tmp_path):
    result = audit_reporting(tmp_path)
    assert len(expected_runs()) == 42
    assert result["annual_counts"] == {
        "current": 0,
        "stale_source": 0,
        "missing": 42,
        "invalid": 0,
    }
    assert len(result["missing_prerequisite_runs"]) == 42
    assert not result["panel_build_ready"]
    assert result["panel"]["status"] == "missing"
