"""Atomic, self-verifying storage for canonical SDA results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..model import EffectDimensions, EffectResult, LabelSet, SDAResult, ValidationReport

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


class StorageError(ValueError):
    """The stored run violates its manifest or the canonical result contract."""


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _from_json_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_from_json_value(item) for item in value["__tuple__"])
    if isinstance(value, dict):
        return {key: _from_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_json_value(item) for item in value]
    return value


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_array(root: Path, relative: str, value: Any) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    np.save(path, array, allow_pickle=False)
    return {
        "file": relative,
        "sha256": _checksum(path),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _labels(label_set: LabelSet) -> dict[str, Any]:
    return {"role": label_set.role, "labels": _json_value(label_set.labels)}


def _write_effect(root: Path, effect: EffectResult, key: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": effect.name,
        "dimensions": {
            "row": _labels(effect.dimensions.row),
            "column": _labels(effect.dimensions.column) if effect.dimensions.column else None,
        },
        "baseline": _write_array(root, f"arrays/{key}/baseline.npy", effect.baseline),
        "mean": None,
        "metadata": _json_value(effect.metadata),
        "source_categories": {},
    }
    if effect.mean is not None:
        entry["mean"] = _write_array(root, f"arrays/{key}/mean.npy", effect.mean)
    for category, child in (effect.source_categories or {}).items():
        safe = hashlib.sha256(category.encode("utf-8")).hexdigest()[:16]
        entry["source_categories"][category] = _write_effect(root, child, f"{key}/source-{safe}")
    return entry


def publish_result(result: SDAResult, output_root: str | Path, run_id: str) -> Path:
    """Publish one validated run atomically and return its final directory."""
    result.validation.raise_if_invalid()
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be one non-empty path component")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    if destination.exists():
        raise FileExistsError(f"run already exists: {destination}")
    staging = root / f".staging-{run_id}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "effects": {
                name: _write_effect(staging, effect, f"effects/{name}")
                for name, effect in result.effects.items()
            },
            "total_change": _write_array(staging, "arrays/total_change.npy", result.total_change),
            "validation": {
                "passed": result.validation.passed,
                "checks": _json_value(result.validation.checks),
                "max_abs_error": result.validation.max_abs_error,
                "max_rel_error": result.validation.max_rel_error,
                "details": _json_value(result.validation.details),
            },
        }
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        load_result(staging)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _read_array(root: Path, entry: Mapping[str, Any]) -> np.ndarray:
    relative = Path(str(entry["file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise StorageError(f"unsafe array path: {relative}")
    path = root / relative
    try:
        if _checksum(path) != entry["sha256"]:
            raise StorageError(f"checksum mismatch: {entry['file']}")
        array = np.load(path, allow_pickle=False)
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"cannot read array: {entry.get('file')}") from exc
    if list(array.shape) != entry["shape"]:
        raise StorageError(f"shape mismatch: {entry['file']}")
    if array.dtype.str != entry["dtype"]:
        raise StorageError(f"dtype mismatch: {entry['file']}")
    return array


def _label_set(entry: Mapping[str, Any]) -> LabelSet:
    labels = _from_json_value(entry["labels"])
    if not isinstance(labels, tuple):
        raise StorageError("labels must use the versioned tuple encoding")
    return LabelSet(str(entry["role"]), labels)


def _read_effect(root: Path, entry: Mapping[str, Any]) -> EffectResult:
    dimensions = entry["dimensions"]
    row = _label_set(dimensions["row"])
    column = _label_set(dimensions["column"]) if dimensions["column"] else None
    try:
        return EffectResult(
            name=str(entry["name"]),
            baseline=_read_array(root, entry["baseline"]),
            mean=_read_array(root, entry["mean"]) if entry["mean"] else None,
            dimensions=EffectDimensions(row, column),
            metadata=_from_json_value(entry["metadata"]),
            source_categories={
                category: _read_effect(root, child)
                for category, child in entry["source_categories"].items()
            } or None,
        )
    except StorageError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StorageError("effect does not satisfy its declared roles and shape") from exc


def load_result(run_directory: str | Path) -> SDAResult:
    """Load a run only after checksum, shape, role, and contract verification."""
    root = Path(run_directory)
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise StorageError(f"unsupported schema version: {manifest['schema_version']}")
        validation_data = manifest["validation"]
        validation = ValidationReport(
            passed=validation_data["passed"],
            checks=_from_json_value(validation_data["checks"]),
            max_abs_error=validation_data["max_abs_error"],
            max_rel_error=validation_data["max_rel_error"],
            details=_from_json_value(validation_data["details"]),
        )
        effects = {
            name: _read_effect(root, entry) for name, entry in manifest["effects"].items()
        }
        for key, effect in effects.items():
            if key != effect.name:
                raise StorageError(f"effect key/name mismatch: {key}/{effect.name}")
        return SDAResult(
            effects=effects,
            total_change=_read_array(root, manifest["total_change"]),
            validation=validation,
        )
    except StorageError:
        raise
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"invalid SDA manifest in {root}") from exc
