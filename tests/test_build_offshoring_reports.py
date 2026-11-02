import importlib.util
import json
import sys
from pathlib import Path

from sda.integration import write_state_bundle
from sda.output import load_result
from test_production_bridge import _build_state, _consumed_files
from build_production_run import build_production_run
from build_annual_reports import _sourcing_rows, _validate_sourcing_reconciliation


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_offshoring_reports.py"
SPEC = importlib.util.spec_from_file_location("build_offshoring_reports", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_command_builds_intermediate_and_final_demand_products_from_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    write_state_bundle(
        _build_state(), bundle, consumed_files=_consumed_files(tmp_path)
    )
    output = tmp_path / "reports"
    assert MODULE.main([
        "--bundle-dir", str(bundle), "--output-dir", str(output),
        "--run-id", "tiny", "--file-format", "csv", "--chunk-rows", "11",
    ]) == 0
    assert (output / "source_country" / "D_C.source_country_cube.csv").is_file()
    assert (output / "source_group_sector" / "D_C.source_group_sector_cube.csv").is_file()
    assert (output / "yf_source_country" / "D_YF.source_country_cube.csv").is_file()
    assert (output / "yf_source_group_sector" / "D_YF.source_group_sector_cube.csv").is_file()
    assert (output / "source_group_sector_interaction" /
            "D_C.source_destination_group_sector_interaction.csv").is_file()
    assert (output / "yf_group_product_interaction" /
            "D_YF.source_destination_group_product_interaction.csv").is_file()
    profile = json.loads((output / "wall_time_profile.json").read_text(encoding="utf-8"))
    assert profile["status"] == "completed"
    assert profile["configuration"]["backend"] == "numpy"
    assert profile["total_seconds"] > 0
    expected_stages = {
        "load_inputs", "source_country.build", "source_country.export",
        "source_group_sector.build", "source_group_sector.export",
        "yf_source_country.build", "yf_source_country.export",
        "yf_source_group_sector.build",
        "yf_source_group_sector.export",
        "source_group_sector_interaction.build",
        "source_group_sector_interaction.export",
        "yf_group_product_interaction.build",
        "yf_group_product_interaction.export",
    }
    assert expected_stages <= profile["stages_seconds"].keys()
    assert all(profile["stages_seconds"][name] >= 0 for name in expected_stages)


def test_dashboard_ingests_d_c_and_d_yf_sourcing_without_role_aliases(tmp_path):
    bundle = tmp_path / "bundle"
    write_state_bundle(_build_state(), bundle, consumed_files=_consumed_files(tmp_path))
    output = tmp_path / "reports"
    assert MODULE.main(["--bundle-dir", str(bundle), "--output-dir", str(output),
        "--run-id", "tiny", "--file-format", "parquet", "--chunk-rows", "11"]) == 0
    canonical = build_production_run(bundle, tmp_path / "canonical", "tiny")
    rows = _sourcing_rows(output, load_result(canonical), {})
    assert {row["effect_id"] for row in rows} == {"D_C", "D_YF"}
    yf = [row for row in rows if row["effect_id"] == "D_YF"]
    assert {row["grain_id"] for row in yf} >= {"source_country", "source_group", "source_sector"}
    assert all("driver_country" not in row for row in yf)


def test_sourcing_reconciliation_is_scale_aware_at_emitter_grain():
    # A global-net comparison spuriously fails here because the two material
    # emitter effects cancel.  Both emitter errors are within the established
    # float32-derived scale-aware tolerance.
    parent = [[1.0e11], [-1.0e11]]
    reconstructed = [1.0e11 + 1.0e5, -1.0e11 + 1.0e5]
    _validate_sourcing_reconciliation(
        reconstructed, parent, effect="D_C", grain="source_country"
    )


def test_sourcing_reconciliation_rejects_material_emitter_error():
    import pytest

    parent = [[1.0e11], [-1.0e11]]
    reconstructed = [1.0e11 + 1.0e8, -1.0e11]
    with pytest.raises(ValueError, match="does not reconcile at emitter grain"):
        _validate_sourcing_reconciliation(
            reconstructed, parent, effect="D_C", grain="source_country"
        )
