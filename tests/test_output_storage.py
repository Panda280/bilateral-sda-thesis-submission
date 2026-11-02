import json

import numpy as np
import pytest

from sda import CANONICAL_EFFECTS, EffectDimensions, EffectResult, LabelSet, SDAResult, ValidationReport
from sda.output import StorageError, load_result, publish_result


def _result(*, passed=True):
    emitters = LabelSet("emitter_country_sector", (("A", "x"), ("B", "x")))
    drivers = LabelSet("structural_driver_country_sector", (("A", "x"), ("B", "x")))
    effects = {}
    for index, name in enumerate(CANONICAL_EFFECTS):
        dimensions = EffectDimensions(emitters) if name == "D_v" else EffectDimensions(emitters, drivers)
        shape = (2,) if name == "D_v" else (2, 2)
        effects[name] = EffectResult(
            name,
            np.full(shape, index, dtype=np.float64),
            dimensions,
            mean=np.full(shape, index + 0.5),
            metadata={"iteration": np.int64(index), "kind": ("canonical", name)},
        )
    parent = effects["D_C"]
    domestic = EffectResult(
        "D_C",
        parent.baseline.copy(),
        parent.dimensions,
        metadata={"source_origin": "Domestic"},
    )
    effects["D_C"] = EffectResult(
        parent.name,
        parent.baseline,
        parent.dimensions,
        mean=parent.mean,
        metadata=parent.metadata,
        source_categories={"Domestic": domestic},
    )
    validation = ValidationReport(passed, {"reconciles": passed}, details={"note": "synthetic"})
    if passed:
        return SDAResult(effects, np.array([1.0, 2.0]), validation)
    return effects, validation


def test_round_trip_preserves_arrays_roles_labels_metadata_and_validation(tmp_path):
    original = _result()
    published = publish_result(original, tmp_path, "run-001")
    loaded = load_result(published)

    assert published == tmp_path / "run-001"
    assert not list(tmp_path.glob(".staging-*"))
    assert loaded.validation.details == original.validation.details
    for name, effect in loaded.effects.items():
        np.testing.assert_array_equal(effect.baseline, original.effects[name].baseline)
        np.testing.assert_array_equal(effect.mean, original.effects[name].mean)
        assert effect.dimensions == original.effects[name].dimensions
        assert effect.metadata == original.effects[name].metadata
    assert loaded.effects["D_C"].source_categories["Domestic"].metadata == {
        "source_origin": "Domestic"
    }


def test_failed_validation_is_refused_without_staging_or_published_run(tmp_path):
    effects, validation = _result(passed=False)
    invalid = object.__new__(SDAResult)
    object.__setattr__(invalid, "effects", effects)
    object.__setattr__(invalid, "total_change", np.zeros(2))
    object.__setattr__(invalid, "validation", validation)

    with pytest.raises(ValueError, match="validation failed"):
        publish_result(invalid, tmp_path, "bad-run")
    assert list(tmp_path.iterdir()) == []


def test_loader_rejects_array_checksum_corruption(tmp_path):
    published = publish_result(_result(), tmp_path, "run-001")
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    array_path = published / manifest["effects"]["D_C"]["baseline"]["file"]
    array_path.write_bytes(array_path.read_bytes() + b"corrupt")

    with pytest.raises(StorageError, match="checksum mismatch"):
        load_result(published)


def test_loader_rejects_declared_role_or_shape_tampering(tmp_path):
    published = publish_result(_result(), tmp_path, "run-001")
    manifest_path = published / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["effects"]["D_C"]["dimensions"]["column"]["labels"] = {
        "__tuple__": [{"__tuple__": ["A", "x"]}]
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StorageError, match="declared roles and shape"):
        load_result(published)


def test_publish_never_overwrites_existing_run(tmp_path):
    publish_result(_result(), tmp_path, "run-001")
    with pytest.raises(FileExistsError):
        publish_result(_result(), tmp_path, "run-001")
