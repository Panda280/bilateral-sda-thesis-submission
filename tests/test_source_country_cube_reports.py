import json

import numpy as np
import pandas as pd
import pytest

from sda import build_source_country_offshoring_cube
from sda.output import export_source_country_cube
from test_source_country_cube import _inputs


@pytest.fixture
def cube():
    return build_source_country_offshoring_cube(**_inputs(), metadata={"bridge": "tiny"})


def test_csv_export_has_exact_schema_values_and_provenance(tmp_path, cube):
    root = export_source_country_cube(
        cube,
        tmp_path / "csv",
        run_id="run-tiny",
        units="kg CO2",
        run_metadata={"year": 2004},
        provenance={"input_bundle": "sha256:abc"},
        chunk_rows=3,
    )
    frame = pd.read_csv(root / "D_C.source_country_cube.csv")

    assert list(frame.columns) == [
        "Emitter_Country", "Emitter_Sector", "Driver_Country",
        "Source_Country", "Value",
    ]
    assert len(frame) == cube.baseline.size
    np.testing.assert_allclose(frame["Value"], cube.baseline.ravel())
    assert frame["Driver_Country"].tolist()[:4] == ["B", "B", "A", "A"]
    assert frame["Source_Country"].tolist()[:4] == ["B", "A", "B", "A"]

    manifest = json.loads((root / "source_country_cube_report.json").read_text())
    assert manifest["run_id"] == "run-tiny"
    assert manifest["units"] == "kg CO2"
    assert manifest["run_metadata"] == {"year": 2004}
    assert manifest["provenance"] == {"input_bundle": "sha256:abc"}
    assert manifest["cube_metadata"]["bridge"] == "tiny"
    assert manifest["reconciliation"]["passed"] is True


def test_chunked_parquet_matches_csv_order_and_values(tmp_path, cube):
    csv_root = export_source_country_cube(
        cube, tmp_path / "csv", run_id="r", units="kg", chunk_rows=5
    )
    parquet_root = export_source_country_cube(
        cube, tmp_path / "parquet", run_id="r", units="kg",
        file_format="parquet", chunk_rows=5,
    )
    csv_frame = pd.read_csv(csv_root / "D_C.source_country_cube.csv")
    parquet_frame = pd.read_parquet(
        parquet_root / "D_C.source_country_cube.parquet"
    )
    assert csv_frame.columns.tolist() == parquet_frame.columns.tolist()
    assert csv_frame.iloc[:, :-1].astype(str).equals(parquet_frame.iloc[:, :-1].astype(str))
    np.testing.assert_allclose(csv_frame["Value"], parquet_frame["Value"])


def test_rejects_unknown_format_and_invalid_chunk_size(tmp_path, cube):
    with pytest.raises(ValueError, match="file_format"):
        export_source_country_cube(
            cube, tmp_path / "bad", run_id="r", units="kg", file_format="xlsx"
        )
    with pytest.raises(ValueError, match="chunk_rows"):
        export_source_country_cube(
            cube, tmp_path / "bad-chunk", run_id="r", units="kg", chunk_rows=0
        )
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "bad-chunk").exists()
