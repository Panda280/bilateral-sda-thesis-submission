import csv
import json

from sda import build_source_group_sector_offshoring_cube
from sda.output import export_source_group_sector_cube
from test_source_country_cube import _inputs
from test_source_group_sector_cube import GROUPS, SECTORS


def test_csv_export_is_complete_and_records_both_gates(tmp_path):
    cube = build_source_group_sector_offshoring_cube(
        **_inputs(), source_sectors=SECTORS, source_group_by_country=GROUPS
    )
    root = export_source_group_sector_cube(
        cube, tmp_path, run_id="tiny", units="kg CO2", file_format="csv", chunk_rows=7
    )
    with (root / "D_C.source_group_sector_cube.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == cube.baseline.size
    assert set(rows[0]) == {
        "Emitter_Country", "Emitter_Sector", "Driver_Country",
        "Source_Group", "Source_Sector", "Value",
    }
    manifest = json.loads((root / "source_group_sector_cube_report.json").read_text())
    assert manifest["reconciliation"]["parent"]["failing_index"] is None
    assert all(
        report["failing_index"] is None
        for report in manifest["reconciliation"]["by_source_group"].values()
    )
    assert manifest["cube_metadata"]["backend"] == "numpy"
