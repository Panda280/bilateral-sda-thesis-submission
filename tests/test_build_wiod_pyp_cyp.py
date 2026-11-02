from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_wiod_pyp_cyp.py"
SPEC = importlib.util.spec_from_file_location("build_wiod_pyp_cyp", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_complete_contract_contains_all_three_sda_resolutions_and_benchmarks():
    names = set(MODULE.expected_build_artifacts(2004, 2003))
    assert "EXIOBASE_CYP_2004_R3_Z.parquet" in names
    for version in (2013, 2016):
        assert f"EXIOBASE_CYP_2004_WIOT{version}_Z.parquet" in names
        assert f"EXIOBASE_PYP_2004_in_2003_prices_WIOT{version}_Y.parquet" in names
        assert f"WIOD_CYP_2004_WIOT{version}_Z.parquet" in names
        assert f"WIOD_PYP_2004_in_2003_prices_WIOT{version}_Y.parquet" in names
        assert f"EXIOBASE_CO2_2004_WIOT{version}.csv" in names


def test_unavailable_official_benchmarks_are_not_mandatory():
    year_2000 = set(MODULE.expected_build_artifacts(2000, 1999))
    assert "WIOD_PYP_2000_in_1999_prices_WIOT2016_Z.parquet" not in year_2000
    year_2014 = set(MODULE.expected_build_artifacts(2014, 2013))
    assert "WIOD_PYP_2014_in_2013_prices_WIOT2013_Z.parquet" not in year_2014
    assert "EXIOBASE_PYP_2014_in_2013_prices_WIOT2013_Z.parquet" in year_2014
    assert "WIOD_PYP_2014_in_2013_prices_WIOT2016_Z.parquet" in year_2014


def test_build_command_enables_both_wiod_versions_and_r3(tmp_path):
    repo = tmp_path / "exio"
    args = argparse.Namespace(
        python=Path("python"), exiodeflator_repo=repo, system="pxp",
        download_dir=tmp_path / "exio-cache", wiod_dir=tmp_path / "wiod-cache",
        deflator_file=Path("deflators.xlsx"), deflator_concordance=Path("deflator.csv"),
        wiod_concordance=Path("wiod.xlsx"), wiod_concordance_sheet="sheet",
        r3_concordance=Path("r3.xlsx"), country_concordance=Path("countries.csv"),
        gras_backend="numpy", weights_file=None, exio_doi=None,
    )
    command = MODULE.build_command(args, 2004, 2003, tmp_path / "out")
    assert command[command.index("--wiod-versions") + 1:] == [
        "2013", "2016", "--gras-backend", "numpy",
        "--gras-max-iter", "500", "--gras-rtol", "0.0001",
        "--gras-small-col-frac", "0.0", "--gras-check-every", "5",
        "--gras-relaxation", "1.5",
        "--gras-infeasible-target-materiality", "0.0001",
    ]
    assert "--r3-concordance" in command
    assert "--skip-r3" not in command


def test_bundle_command_covers_each_canonical_database(tmp_path):
    layout = MODULE.Layout(tmp_path / "run", 2004, 2003)
    args = argparse.Namespace(
        python=Path("python"), export_script=Path("export.py"),
        sda_repo=tmp_path / "sda", impact="CO2", num_samples=1,
        download_dir=tmp_path / "cache" / "exiobase",
    )
    for database in MODULE.DATABASES:
        command = MODULE.bundle_command(args, layout, database)
        assert command[command.index("--database") + 1] == database
        assert command[command.index("--prev-out-dir") + 1] == str(layout.previous)
        assert command[command.index("--exiobase-raw-dir") + 1] == str(args.download_dir)
