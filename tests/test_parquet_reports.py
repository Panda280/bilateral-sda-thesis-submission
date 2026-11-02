"""Correctness tests for the Parquet report-export path.

Reuses tests/_group_report_fixtures.py's synthetic, fully-validated
SDAResult (real D_C/D_YF source splits, real Leontief inverses) so every
canonical effect shape (D_v's plain vector, D_y's country-only column,
the general structural/demand pair-column shape) is exercised against
real, non-degenerate data rather than a hand-picked toy case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sda.hierarchy import CANONICAL_EFFECTS
from sda.output.reports import (
    EFFECT_REGISTRY,
    GROUP_EFFECT_REGISTRY,
    _flatten_columns,
    _rows,
    aggregate_to_group_sector,
    build_group_sector_reports,
    export_canonical_reports,
    export_effect_csv,
    export_effect_parquet,
    export_group_effect_parquet,
)

from _group_report_fixtures import GROUPS, build_synthetic_result


@pytest.fixture(scope="module")
def result():
    return build_synthetic_result()


@pytest.mark.parametrize("name", CANONICAL_EFFECTS)
def test_flatten_columns_matches_rows_exactly(result, name):
    effect = result.effects[name]
    expected_rows = list(_rows(effect))
    columns = _flatten_columns(effect)
    assert len(set(len(c) for c in columns)) == 1
    actual_rows = list(zip(*columns))
    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows):
        assert tuple(actual) == expected


@pytest.mark.parametrize("name", CANONICAL_EFFECTS)
def test_export_effect_parquet_matches_csv_values(tmp_path, result, name):
    effect = result.effects[name]
    csv_path = export_effect_csv(effect, tmp_path / f"{name}.csv")
    parquet_path = export_effect_parquet(effect, tmp_path / f"{name}.parquet")

    csv_frame = pd.read_csv(csv_path)
    parquet_frame = pd.read_parquet(parquet_path)

    assert list(csv_frame.columns) == list(parquet_frame.columns) == list(EFFECT_REGISTRY[name].columns)
    assert len(csv_frame) == len(parquet_frame)
    # CSV round-trips floats through text (str(float) -> pandas' text parser),
    # which can lose a few ULPs versus Parquet's exact binary encoding -- not
    # a Parquet bug, the opposite: Parquet preserves the source float64
    # exactly, CSV doesn't. assert_allclose at ~float64-eps rtol confirms
    # both represent the same value; it deliberately isn't assert_array_equal.
    np.testing.assert_allclose(
        csv_frame["Value"].to_numpy(), parquet_frame["Value"].to_numpy(), rtol=1e-12, atol=1e-12
    )
    for column in csv_frame.columns:
        if column == "Value":
            continue
        assert csv_frame[column].astype(str).tolist() == parquet_frame[column].astype(str).tolist()


def test_export_canonical_reports_parquet_format(tmp_path, result):
    root = export_canonical_reports(
        result, tmp_path / "run", run_id="test-run", units="kg CO2", file_format="parquet"
    )
    import json

    manifest = json.loads((root / "report.json").read_text())
    for name in CANONICAL_EFFECTS:
        filename = manifest["files"][name]
        assert filename == f"{name}.parquet"
        frame = pd.read_parquet(root / filename)
        assert len(frame) == np.asarray(result.effects[name].baseline).size
        assert abs(frame["Value"].sum() - manifest["effect_totals"][name]) < 1e-6


def test_export_canonical_reports_rejects_unknown_format(tmp_path, result):
    with pytest.raises(ValueError, match="file_format"):
        export_canonical_reports(
            result, tmp_path / "run", run_id="test-run", units="kg CO2", file_format="xlsx"
        )


def test_group_effect_parquet_matches_group_csv_values(result):
    from sda.output.reports import export_group_effect_csv

    grouped = aggregate_to_group_sector(result.effects["D_C"], GROUPS)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv_path = export_group_effect_csv(grouped, tmp / "D_C.group.csv")
        parquet_path = export_group_effect_parquet(grouped, tmp / "D_C.group.parquet")
        csv_frame = pd.read_csv(csv_path)
        parquet_frame = pd.read_parquet(parquet_path)
        assert list(csv_frame.columns) == list(GROUP_EFFECT_REGISTRY["D_C"].columns)
        np.testing.assert_allclose(
            csv_frame["Value"].to_numpy(), parquet_frame["Value"].to_numpy(), rtol=1e-12, atol=1e-12
        )


def test_build_group_sector_reports_parquet_format(tmp_path, result):
    import json

    root = build_group_sector_reports(
        result, tmp_path / "group-run", run_id="test-run", units="kg CO2",
        country_group=GROUPS, file_format="parquet",
    )
    manifest = json.loads((root / "group_report.json").read_text())
    for name in CANONICAL_EFFECTS:
        filename = manifest["files"][name]
        assert filename.endswith(".parquet")
        pd.read_parquet(root / filename)
    for name, categories in manifest["source_category_files"].items():
        for category, filename in categories.items():
            assert filename.endswith(".parquet")
            pd.read_parquet(root / filename)


def test_pair_accepts_real_bundle_flat_string_convention():
    """sda.integration.bundle_export._label ("country:sector") must round-trip
    through the same _pair() every export function relies on -- this is the
    real EXIODeflator-sourced bundle's label shape, not just tuples."""
    from sda.output.reports import _pair

    assert _pair("AUS:AtB", "emitter") == ("AUS", "AtB")
    assert _pair(("AUS", "AtB"), "emitter") == ("AUS", "AtB")
    with pytest.raises(ValueError, match="country, sector"):
        _pair("AUS", "emitter")
    with pytest.raises(ValueError, match="country, sector"):
        _pair("AUS:AtB:extra", "emitter")
    with pytest.raises(ValueError, match="country, sector"):
        _pair(123, "emitter")
