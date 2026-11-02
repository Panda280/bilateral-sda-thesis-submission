import json

import numpy as np
import pytest

from sda.regression import (
    compare_legacy_aggregates,
    load_legacy_manifest,
    load_legacy_parquet_profile,
)


def test_alias_comparison_reduces_only_to_emitter_and_global_totals(tmp_path):
    canonical = {"D_C": np.array([[1.0, 2.0], [3.0, 4.0]])}
    legacy = {"technology": np.array([[0.5, 2.5], [2.0, 5.0]])}
    report = compare_legacy_aggregates(
        canonical, legacy, ["D_C"], aliases={"D_C": "technology"}
    )

    assert report.passed
    assert report.detail_comparison_refused
    assert report.comparisons[0].max_emitter_abs_error == 0.0
    output = report.write_json(tmp_path / "regression.json")
    assert json.loads(output.read_text())["comparison_scope"] == "per-emitter and global totals only"


def test_missing_effect_fails_even_when_difference_is_declared():
    report = compare_legacy_aggregates(
        {}, {}, ["D_YF"], intentional_differences={"D_YF": "known methodology change"}
    )
    assert not report.passed
    assert report.missing_effects == ("D_YF",)


def test_mismatch_fails_unless_intentionally_documented():
    canonical = {"D_v": np.array([1.0, 2.0])}
    legacy = {"D_v": np.array([1.0, 2.1])}
    failed = compare_legacy_aggregates(canonical, legacy, ["D_v"], atol=1e-12, rtol=0)
    accepted = compare_legacy_aggregates(
        canonical,
        legacy,
        ["D_v"],
        intentional_differences={"D_v": "corrected legacy rounding"},
        atol=1e-12,
        rtol=0,
    )
    assert not failed.passed
    assert accepted.passed
    assert accepted.comparisons[0].intentional_difference == "corrected legacy rounding"


def test_detailed_comparison_is_explicitly_refused():
    with pytest.raises(ValueError, match="legacy columns are endpoints"):
        compare_legacy_aggregates({}, {}, [], detail=True)


def test_manifest_loads_npy_without_pickle(tmp_path):
    np.save(tmp_path / "legacy.npy", np.array([[1.0, 2.0]]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"effects": {"old_C": "legacy.npy"}}))
    loaded = load_legacy_manifest(manifest)
    np.testing.assert_array_equal(loaded["old_C"], [[1.0, 2.0]])


def test_parquet_profile_preserves_labels_aliases_and_provenance(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"Country": ["DEU", "NLD"], "Sector": ["A", "B"], "Dint": [1.5, -0.5]}
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "provenance": {"source": "EXIODeflator", "year": 2004},
        "aliases": {"D_v": "Dint"},
        "effects": {"Dint": {"path": "2004_Dint_tot_baseline.parquet"}},
    }))

    bundle = load_legacy_parquet_profile(profile)

    np.testing.assert_array_equal(bundle["Dint"], [1.5, -0.5])
    assert bundle.effects["Dint"].countries == ("DEU", "NLD")
    assert bundle.effects["Dint"].sectors == ("A", "B")
    assert bundle.effects["Dint"].value_column == "Dint"
    assert bundle.aliases == {"D_v": "Dint"}
    assert bundle.provenance["year"] == 2004


def test_parquet_profile_refuses_detailed_path_before_reader_import(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "effects": {"Dint": {"path": "2004_Dint_detailed_baseline.parquet"}}
    }))
    with pytest.raises(ValueError, match="aggregate"):
        load_legacy_parquet_profile(profile)


def test_exact_approval_does_not_override_detailed_artifact_ban(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "approved_aggregate_filenames": ["old_matrix.parquet"],
        "effects": {"Dint": {"path": "old_matrix.parquet"}},
    }))
    with pytest.raises(ValueError, match="detailed legacy artifact"):
        load_legacy_parquet_profile(profile)
