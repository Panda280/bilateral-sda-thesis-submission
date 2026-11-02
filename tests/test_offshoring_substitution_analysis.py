import json
from pathlib import Path

import pandas as pd

import analyze_offshoring_substitution as analysis


def _write_product(
    root: Path,
    *,
    start_year: int,
    resolution: str,
    effect: str,
    np_value: float = 3_000_000_000.0,
    domestic_value: float = -2_000_000_000.0,
    same_group_cross_border_value: float = 500_000_000.0,
) -> None:
    spec = analysis.PRODUCTS[effect]
    product_dir = root / f"{start_year}_{start_year + 1}_{resolution}" / spec.directory
    product_dir.mkdir(parents=True, exist_ok=True)
    sector_column = spec.sector_column
    rows = [
        {
            "Relationship": "Cross-border",
            "Source_Group": "Non-Participant",
            "Destination_Group": "Red",
            sector_column: "s0",
            "Value": np_value,
        },
        {
            "Relationship": "Domestic",
            "Source_Group": "Red",
            "Destination_Group": "Red",
            sector_column: "s0",
            "Value": domestic_value,
        },
        {
            "Relationship": "Cross-border",
            "Source_Group": "Red",
            "Destination_Group": "Red",
            sector_column: "s0",
            "Value": same_group_cross_border_value,
        },
        {
            "Relationship": "Cross-border",
            "Source_Group": "Non-Participant",
            "Destination_Group": "Non-Red",
            sector_column: "s0",
            "Value": 200.0,
        },
        {
            "Relationship": "Domestic",
            "Source_Group": "Non-Red",
            "Destination_Group": "Non-Red",
            sector_column: "s0",
            "Value": 100.0,
        },
    ]
    if effect == "D_C":
        for row in rows:
            row["Source_Sector"] = "source0"
    pd.DataFrame(rows).to_parquet(product_dir / spec.filename, index=False)
    (product_dir / spec.report_filename).write_text(
        json.dumps({
            "effect": effect,
            "units": "kg CO2",
            "reconciliation": {"failing_index": None, "max_abs_error": 0.0},
        }),
        encoding="utf-8",
    )


def _write_transition(root: Path, start_year: int, resolution: str) -> None:
    _write_product(root, start_year=start_year, resolution=resolution, effect="D_C")
    _write_product(root, start_year=start_year, resolution=resolution, effect="D_YF")


def test_balanced_mode_outputs_strict_signatures_and_correct_mt_units(tmp_path):
    run_root = tmp_path / "offshoring"
    _write_transition(run_root, 2000, "2013")
    _write_transition(run_root, 2001, "2013")
    _write_transition(run_root, 2000, "R3")
    output = tmp_path / "analysis"

    result = analysis.main([
        "--run-root", str(run_root),
        "--output-dir", str(output),
        "--mode", "balanced",
        "--formats", "csv", "parquet",
        "--destination-groups", "Red", "Non-Red",
    ])

    assert result == 0
    assert (output / "cell_year_signatures.parquet").is_file()
    assert (output / "annual_summary.parquet").is_file()
    coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["selected_transitions"] == ["2000_2001"]
    details = pd.read_csv(output / "cell_year_signatures.csv")
    red = details.loc[
        (details["resolution"] == "2013")
        & (details["effect"] == "D_C")
        & (details["destination_group"] == "Red")
    ].iloc[0]
    assert bool(red["strict_signature"])
    assert red["np_cross_border_kg_co2"] == 3_000_000_000.0
    assert red["np_cross_border_mt_co2"] == 3.0
    assert red["same_group_kg_co2"] == -1_500_000_000.0
    assert red["matched_overlap_mt_co2"] == 1.5
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["units"]["kg_per_mt"] == 1_000_000_000.0
    assert summary["units"]["derived_display_unit"] == "Mt CO2"


def test_full_mode_reports_and_fails_on_missing_products(tmp_path):
    run_root = tmp_path / "offshoring"
    _write_transition(run_root, 2000, "2013")
    _write_transition(run_root, 2001, "2013")
    _write_transition(run_root, 2000, "R3")
    output = tmp_path / "analysis"

    result = analysis.main([
        "--run-root", str(run_root),
        "--output-dir", str(output),
        "--mode", "full",
        "--formats", "csv",
    ])

    assert result == 2
    coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
    missing = coverage["missing_selected_products"]
    assert {(row["resolution"], row["effect"], row["start_year"]) for row in missing} == {
        ("R3", "D_C", 2001),
        ("R3", "D_YF", 2001),
    }
    assert not (output / "summary.json").exists()


def test_zero_tolerance_suppresses_immaterial_signs(tmp_path):
    run_root = tmp_path / "offshoring"
    _write_product(
        run_root,
        start_year=2000,
        resolution="2013",
        effect="D_YF",
        np_value=5.0,
        domestic_value=-5.0,
        same_group_cross_border_value=0.0,
    )
    inventory, _, _ = analysis.discover_products(run_root)
    record = inventory[("2013", "D_YF", (2000, 2001))]

    exact = analysis.analyze_product(
        Path(record["product_path"]),
        resolution="2013",
        effect="D_YF",
        transition=(2000, 2001),
        destination_groups=("Red", "Non-Red"),
        zero_tolerance_kg=0.0,
    )
    tolerant = analysis.analyze_product(
        Path(record["product_path"]),
        resolution="2013",
        effect="D_YF",
        transition=(2000, 2001),
        destination_groups=("Red", "Non-Red"),
        zero_tolerance_kg=10.0,
    )

    assert bool(exact.loc[exact["destination_group"] == "Red", "strict_signature"].iloc[0])
    assert not bool(
        tolerant.loc[tolerant["destination_group"] == "Red", "strict_signature"].iloc[0]
    )
