"""Unit tests for sda.integration.bundle_export -- the pure-NumPy half of
the data-pipeline bridge (see tests/test_production_bridge.py for the full
chain, including sda.cli.execute_config and Gate C reconciliation)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sda.attribution.groups import SOURCE_ORIGIN_GROUPS
from sda.cli import ARRAY_INPUTS, SCHEMA_VERSION, SEMANTIC_INPUTS
from sda.integration.bundle_export import source_file_record, write_state_bundle
from sda.state import MappingStateAdapter, SDAState, SourceGroupInputs, StateLabels


def _tiny_state() -> SDAState:
    eye = np.eye(2)
    ones2 = np.ones((2, 2), dtype=bool)
    zeros2 = np.zeros((2, 2), dtype=bool)
    zeros_y = np.zeros((2, 1), dtype=bool)
    ones_y = np.ones((2, 1), dtype=bool)
    arrays = {
        "v0": np.array([1.0, 2.0]), "v1": np.array([1.5, 2.5]),
        "L0": eye, "L1": eye, "C0": eye, "C1": eye,
        "A_x0": eye, "A_x1": eye, "T_coef0": eye, "T_coef1": eye,
        "w_coef0": np.ones(2), "w_coef1": np.ones(2),
        "F0": np.ones((2, 1)), "F1": np.ones((2, 1)),
        "B0": np.ones((2, 1)), "B1": np.ones((2, 1)),
        "levels0": np.array([1.0]), "levels1": np.array([2.0]),
        "supplier_sector": np.array([0, 1]),
    }
    labels = StateLabels(
        emitters=(("A", "s0"), ("A", "s1")),
        structural_drivers=(("A", "s0"), ("A", "s1")),
        consumers=("A",),
        demanded_sectors=("s0", "s1"),
    )
    source_groups = SourceGroupInputs(
        C={"Domestic": ones2, "Red": zeros2, "Non-Red": zeros2, "Non-Participant": zeros2},
        YF={"Domestic": ones_y, "Red": zeros_y, "Non-Red": zeros_y, "Non-Participant": zeros_y},
    )
    return MappingStateAdapter(
        arrays=arrays, labels=labels, source_groups=source_groups, sectors_per_country=2
    ).adapt()


def test_write_state_bundle_produces_a_cli_consumable_config(tmp_path):
    state = _tiny_state()
    source = tmp_path / "source.csv"
    source.write_text("stand-in\n", encoding="utf-8")
    consumed = [{"role": "Z_source", **source_file_record(source)}]

    config_path = write_state_bundle(state, tmp_path / "bundle", consumed_files=consumed)

    assert config_path.name == "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["schema_version"] == SCHEMA_VERSION
    assert set(config["arrays"]) == set(ARRAY_INPUTS)
    assert set(config["semantics"]) == set(SEMANTIC_INPUTS)
    assert config["semantics"]["emitter_labels"] == ["A:s0", "A:s1"]
    assert config["semantics"]["consumer_labels"] == ["A"]
    assert config["semantics"]["demanded_sector_labels"] == ["s0", "s1"]
    for family in ("c_source_masks", "yf_source_masks"):
        assert set(config["semantics"][family]) == set(SOURCE_ORIGIN_GROUPS)

    npz_path = config_path.parent / config["input_npz"]
    with np.load(npz_path) as archive:
        assert set(config["arrays"].values()) <= set(archive.files)
        for family in ("c_source_masks", "yf_source_masks"):
            assert set(config["semantics"][family].values()) <= set(archive.files)

    manifest = json.loads((tmp_path / "bundle" / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["consumed_files"][0]["role"] == "Z_source"
    assert manifest["consumed_files"][0]["sha256"] == consumed[0]["sha256"]


def test_write_state_bundle_config_actually_executes(tmp_path):
    """The exported config.json must round-trip through the real CLI path,
    not just satisfy its own schema."""
    from sda.cli import execute_config

    state = _tiny_state()
    source = tmp_path / "source.csv"
    source.write_text("stand-in\n", encoding="utf-8")
    config_path = write_state_bundle(
        state, tmp_path / "bundle",
        consumed_files=[{"role": "Z_source", **source_file_record(source)}],
    )

    published = execute_config(config_path, tmp_path / "results", "run")
    assert published.exists()


def test_write_state_bundle_rejects_non_state():
    with pytest.raises(TypeError, match="SDAState"):
        write_state_bundle(object(), "unused", consumed_files=[{"role": "x", "path": "x", "sha256": "x"}])


def test_source_file_record_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        source_file_record(tmp_path / "does-not-exist.csv")


def test_source_file_record_checksum_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "a.csv"
    path.write_bytes(b"hello world")
    record = source_file_record(path)
    assert record["sha256"] == hashlib.sha256(b"hello world").hexdigest()
    assert record["bytes"] == len(b"hello world")
