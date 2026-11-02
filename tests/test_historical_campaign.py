import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_historical_campaign.py"
SPEC = importlib.util.spec_from_file_location("build_historical_campaign", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_full_range_plan_builds_each_year_once_and_all_transitions(tmp_path):
    args = argparse.Namespace(
        start_year=2000, end_year=2014, run_root=tmp_path / "run",
        exiodeflator_repo=tmp_path / "exio", sda_repo=tmp_path / "sda",
        python=Path("python"), system="pxp", gras_backend="numpy",
        download_dir=tmp_path / "cache-exio", wiod_dir=tmp_path / "cache-wiod",
        deflator_file=Path("deflators.xlsx"), deflator_concordance=Path("deflator.csv"),
        wiod_concordance=Path("wiod.xlsx"), wiod_concordance_sheet="Product_to_WIOD_concordance",
        r3_concordance=Path("r3.xlsx"), country_concordance=Path("countries.csv"),
        weights_file=None, exio_doi=None, export_script=Path("export.py"), impact="CO2", num_samples=1,
        cuda_device="0",
        include_sourcing=True,
    )
    plan = MODULE.campaign_plan(args)
    assert plan["matrix_builds"] == 15
    assert plan["transitions"] == 14
    assert plan["canonical_sda_runs"] == 42
    assert len(plan["commands"]["builds"]) == 15
    assert len(plan["commands"]["bundles"]) == 42
    assert len(plan["commands"]["publications"]) == 42
    assert len(plan["commands"]["sourcing"]) == 42
    assert plan["sourcing_products"] == 168
    assert plan["sourcing_products_by_effect"] == {"D_C": 84, "D_YF": 84}
    command = next(iter(plan["commands"]["sourcing"].values()))
    assert "yf-source-country" in command
    assert "yf-source-group-sector" in command
    assert plan["resolutions"] == ["2013", "2016", "R3"]
    assert plan["execution"] == {
        "cuda_visible_devices": "0",
        "gpu_concurrency": 1,
        "matrix_build_concurrency": 1,
        "sda_concurrency": 1,
    }
    command = plan["commands"]["sourcing"]["2000_2001_2013"]
    assert command[command.index("--products") + 1:command.index("--backend")] == [
        "source-country", "source-group-sector",
        "yf-source-country", "yf-source-group-sector",
    ]
    assert command[command.index("--backend") + 1] == "cupy"


def test_campaign_reuses_one_year_directory_between_adjacent_pairs(tmp_path):
    first = MODULE._campaign_layout(tmp_path, 2000, 2001)
    second = MODULE._campaign_layout(tmp_path, 2001, 2002)
    assert first.current == second.previous == tmp_path / "build" / "2001"


def test_yf_completion_requires_consumer_country_resolution_manifest(tmp_path):
    parquet = tmp_path / "D_YF.parquet"
    report = tmp_path / "report.json"
    parquet.touch()
    report.write_text('{"columns":["Demanded_Sector"]}', encoding="utf-8")
    assert not MODULE._sourcing_product_complete("yf-source-country", (parquet, report))

    report.write_text(
        '{"sourcing_resolution":"consumer_country","columns":["Consumer_Country","Source_Country","Value"]}',
        encoding="utf-8",
    )
    assert MODULE._sourcing_product_complete("yf-source-country", (parquet, report))


def test_campaign_inventory_distinguishes_every_completion_layer(tmp_path):
    run = tmp_path / "run"
    for relative in (
        "sda/sda_2000_2001_2013/manifest.json",
        "offshoring/2000_2001_2013/source_country/D_C.source_country_cube.parquet",
        "offshoring/2000_2001_2013/source_group_sector/D_C.source_group_sector_cube.parquet",
        "offshoring/2000_2001_2013/yf_source_country/D_YF.source_country_cube.parquet",
        "annual/2000_2001_2013/manifest.v2.json",
    ):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    inventory = MODULE.campaign_inventory(run, expected_runs=1, sourcing_expected=True)

    assert inventory["observed"] == {
        "canonical": 1,
        "D_C_source_country": 1,
        "D_C_source_group_sector": 1,
        "D_YF_source_country": 1,
        "D_YF_source_group_sector": 0,
        "annual": 1,
        "panel": 0,
    }
    assert inventory["sourcing_totals"] == {
        "expected": 4, "observed": 3,
        "D_C_expected": 2, "D_C_observed": 2,
        "D_YF_expected": 2, "D_YF_observed": 1,
    }
    assert inventory["complete"]["canonical"] is True
    assert inventory["complete"]["D_YF_source_group_sector"] is False
    assert inventory["all_complete"] is False


def test_input_build_audit_surfaces_known_upstream_failures(tmp_path):
    log = tmp_path / "build.log"
    log.write_text(
        "GRAS did not converge after 500 iterations\n"
        "x = L*y reproduction failed\n"
        "Row balance failed\n"
        "Column balance failed\n",
        encoding="utf-8",
    )
    report = MODULE.assess_input_build(2000, tmp_path / "missing.json", log)
    assert report["status"] == "failed"
    assert report["thesis_eligible"] is False
    assert set(report["failed_gates"]) == {
        "gras_convergence", "leontief_reproduction", "row_balance", "column_balance",
    }


def test_input_build_audit_requires_explicit_gate_evidence_for_pass(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"completed": true}', encoding="utf-8")
    report = MODULE.assess_input_build(2001, summary, tmp_path / "missing.log")
    assert report["status"] == "unverified"
    assert report["thesis_eligible"] is False

    summary.write_text(
        '{"input_validation":{"checks":{"gras_convergence":true,'
        '"leontief_reproduction":true,"row_balance":true,"column_balance":true}}}',
        encoding="utf-8",
    )
    report = MODULE.assess_input_build(2001, summary, tmp_path / "missing.log")
    assert report["status"] == "passed"
    assert report["thesis_eligible"] is True


def test_failed_input_build_blocks_downstream_acceptance():
    inventory = {"builds": [{"year": 2000, "status": "failed"}]}
    try:
        MODULE.require_no_failed_input_builds(inventory)
    except MODULE.CampaignError as exc:
        assert "2000" in str(exc)
    else:
        raise AssertionError("failed upstream input build was accepted")


def test_failed_input_build_can_be_explicitly_overridden():
    inventory = {"builds": [{"year": 2000, "status": "failed"}]}

    MODULE.require_no_failed_input_builds(inventory, allow_failed=True)
