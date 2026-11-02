import json

import pytest

from sda.hierarchy import CANONICAL_EFFECTS
from sda.output.coverage import CoverageManifest, RunCoverage, RunSpec

HASH = "a" * 64


def spec(impact="co2"):
    return RunSpec("EXIOBASE", 2019, 2020, "emissions", impact)


def start(run):
    return run.transition(
        "running", code_hash=HASH, input_hash="b" * 64, group_hash="c" * 64,
        execution_plan={"chunk_size": 2, "estimated_peak_bytes": 4096},
        updated_at="2026-08-27T00:00:00+00:00",
    )


def test_run_spec_validates_years_and_has_stable_identity():
    assert spec().run_id == spec().run_id
    with pytest.raises(ValueError, match="later"):
        RunSpec("EXIOBASE", 2020, 2020, "emissions", "co2")


def test_transitions_are_immutable_and_validated():
    pending = RunCoverage(spec())
    running = start(pending)
    assert pending.status == "pending"
    assert running.status == "running"
    with pytest.raises(ValueError, match="invalid status transition"):
        running.transition("pending")
    with pytest.raises(TypeError):
        running.execution_plan["chunk_size"] = 4


def test_valid_requires_complete_expected_effect_inventory():
    running = start(RunCoverage(spec()))
    with pytest.raises(ValueError, match="every expected effect"):
        running.transition("valid", validated_effects=("D_v",))
    valid = running.transition("valid", validated_effects=CANONICAL_EFFECTS)
    assert valid.status == "valid"


def test_failed_run_requires_reason_and_can_be_retried():
    running = start(RunCoverage(spec()))
    with pytest.raises(ValueError, match="failure message"):
        running.transition("failed")
    failed = running.transition("failed", failure="reconciliation gate failed")
    assert failed.transition("running").failure is None


def test_manifest_complete_gate_and_json_summary():
    first = start(RunCoverage(spec("co2"))).transition("valid", validated_effects=CANONICAL_EFFECTS)
    second = RunCoverage(spec("ch4"))
    manifest = CoverageManifest((first, second))
    assert not manifest.complete
    summary = json.loads(manifest.to_json())
    assert summary["complete"] is False
    assert summary["counts"] == {"failed": 0, "pending": 1, "running": 0, "valid": 1}
    second = start(second).transition("valid", validated_effects=CANONICAL_EFFECTS)
    assert manifest.replace(second).complete


def test_hashes_and_duplicate_runs_are_rejected():
    with pytest.raises(ValueError, match="SHA-256"):
        start(RunCoverage(spec())).transition("failed", code_hash="bad", failure="x")
    run = RunCoverage(spec())
    with pytest.raises(ValueError, match="duplicate"):
        CoverageManifest((run, run))


def test_execution_plan_must_be_json_metadata():
    with pytest.raises(ValueError, match="JSON-compatible"):
        RunCoverage(spec(), execution_plan={"bad": object()})
