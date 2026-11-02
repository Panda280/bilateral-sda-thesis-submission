import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sda.output import (
    build_group_sector_reports,
    export_canonical_reports,
    load_result,
    publish_result,
)

from _group_report_fixtures import GROUPS, build_synthetic_result

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_sda_reports.py"


def test_group_report_reconciles_to_an_independently_reloaded_canonical_store(tmp_path):
    """Build -> publish -> *independently* reload -> report -> reconcile.

    This is the round trip the acceptance gate names explicitly: every
    report subtotal must reconcile to the canonical detailed store (not to
    the in-memory object that produced it) and show validation status and
    run ID.
    """
    result = build_synthetic_result()
    published = publish_result(result, tmp_path / "runs", "rt-run")

    # Reload independently: a fresh object read back from disk through
    # checksum-verified storage, not the `result` that produced it.
    reloaded = load_result(published)
    assert reloaded is not result

    root = build_group_sector_reports(
        reloaded, tmp_path / "group", run_id="rt-run", units="kg CO2e", country_group=GROUPS,
    )
    manifest = json.loads((root / "group_report.json").read_text())

    assert manifest["run_id"] == "rt-run"
    assert manifest["validation"]["passed"] is True
    assert manifest["validation"]["checks"] == dict(reloaded.validation.checks)

    # Reload the canonical store a second time, independently of the report
    # build above, and compare every effect total by hand.
    independent = load_result(published)
    assert manifest["reconciliation"]
    for name, totals in manifest["reconciliation"].items():
        canonical_total = float(np.asarray(independent.effects[name].baseline).sum())
        np.testing.assert_allclose(totals["detailed_total"], canonical_total, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(totals["group_total"], canonical_total, rtol=1e-10, atol=1e-12)
        assert totals["abs_error"] < 1e-9

    # The 4-way split reconciles too: every category sums back to the
    # unsplit group-sector total for D_C/D_YF.
    for name in ("D_C", "D_YF"):
        category_total = 0.0
        for filename in manifest["source_category_files"][name].values():
            rows = (root / filename).read_text().splitlines()[1:]
            category_total += sum(float(row.rsplit(",", 1)[1]) for row in rows if row)
        np.testing.assert_allclose(
            category_total, manifest["effect_totals"][name], rtol=1e-10, atol=1e-12
        )


def test_detailed_and_group_reports_agree_on_every_headline_total(tmp_path):
    result = build_synthetic_result()
    published = publish_result(result, tmp_path / "runs", "rt-run-2")
    reloaded = load_result(published)

    detailed_root = export_canonical_reports(
        reloaded, tmp_path / "detailed", run_id="rt-run-2", units="kg CO2e",
    )
    group_root = build_group_sector_reports(
        reloaded, tmp_path / "group", run_id="rt-run-2", units="kg CO2e", country_group=GROUPS,
    )
    detailed = json.loads((detailed_root / "report.json").read_text())
    grouped = json.loads((group_root / "group_report.json").read_text())

    np.testing.assert_allclose(
        detailed["headline_total"], grouped["headline_total"], rtol=1e-10, atol=1e-12
    )
    for name in detailed["effect_totals"]:
        np.testing.assert_allclose(
            detailed["effect_totals"][name], grouped["effect_totals"][name], rtol=1e-10, atol=1e-12
        )


def test_build_sda_reports_script_reads_a_published_run_directory(tmp_path):
    """Exercise scripts/build_sda_reports.py exactly as an operator would: from disk."""
    result = build_synthetic_result()
    published = publish_result(result, tmp_path / "runs", "script-run")

    mapping_path = tmp_path / "country_groups.json"
    mapping_path.write_text(json.dumps(GROUPS), encoding="utf-8")

    output_dir = tmp_path / "reports"
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--run-dir", str(published),
            "--output-dir", str(output_dir),
            "--units", "kg CO2e",
            "--country-group-map", str(mapping_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    top_manifest = json.loads((output_dir / "run_report.json").read_text())
    assert top_manifest["run_id"] == "script-run"
    assert top_manifest["validation"]["passed"] is True
    assert (output_dir / "detailed" / "report.json").exists()
    assert (output_dir / "group" / "group_report.json").exists()
    assert (output_dir / "group" / "narrative.txt").exists()


def test_build_sda_reports_script_defaults_run_id_to_directory_name(tmp_path):
    result = build_synthetic_result()
    published = publish_result(result, tmp_path / "runs", "default-run-id")
    mapping_path = tmp_path / "country_groups.json"
    mapping_path.write_text(json.dumps(GROUPS), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--run-dir", str(published),
            "--output-dir", str(tmp_path / "reports2"),
            "--units", "kg CO2e",
            "--country-group-map", str(mapping_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((tmp_path / "reports2" / "run_report.json").read_text())
    assert manifest["run_id"] == "default-run-id"


def test_build_sda_reports_script_rejects_a_malformed_country_group_map(tmp_path):
    result = build_synthetic_result()
    published = publish_result(result, tmp_path / "runs", "bad-map-run")
    mapping_path = tmp_path / "bad.json"
    mapping_path.write_text(json.dumps({"A": 1}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--run-dir", str(published),
            "--output-dir", str(tmp_path / "reports3"),
            "--units", "kg CO2e",
            "--country-group-map", str(mapping_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode != 0
    assert "country-group mapping" in completed.stderr
