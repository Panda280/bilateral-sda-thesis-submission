import importlib.util
from pathlib import Path

from sda.output import publish_result

from _group_report_fixtures import build_synthetic_result


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_published_run.py"
SPEC = importlib.util.spec_from_file_location("verify_published_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_verify_run_loads_checksums_and_returns_semantic_summary(tmp_path):
    result = build_synthetic_result()
    run_dir = publish_result(result, tmp_path, "verified")
    summary = MODULE.verify_run(run_dir)

    assert summary["passed"] is True
    assert set(summary["effects"]) == set(result.effects)
    assert summary["effect_shapes"]["D_y"] == list(result.effects["D_y"].baseline.shape)
    assert all(summary["checks"].values())
