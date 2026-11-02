"""Aggregate-only regression checks against trusted legacy SDA outputs.

Legacy matrices use endpoint columns, not canonical driver columns.  This module
therefore compares only per-emitter and global totals and refuses cell-level
comparisons by construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LegacyAggregateEffect:
    """One labelled, emitter-level legacy aggregate."""

    values: np.ndarray
    countries: tuple[str, ...]
    sectors: tuple[str, ...]
    value_column: str
    source_path: str


@dataclass(frozen=True)
class LegacyParquetBundle(Mapping[str, np.ndarray]):
    """Aggregate arrays plus the labels and provenance needed to audit them."""

    effects: Mapping[str, LegacyAggregateEffect]
    aliases: Mapping[str, str]
    provenance: Mapping[str, object]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.effects[key].values

    def __iter__(self) -> Iterator[str]:
        return iter(self.effects)

    def __len__(self) -> int:
        return len(self.effects)


@dataclass(frozen=True)
class EffectComparison:
    effect: str
    legacy_effect: str
    passed: bool
    intentional_difference: str | None
    max_emitter_abs_error: float
    max_emitter_rel_error: float
    global_abs_error: float
    global_rel_error: float


@dataclass(frozen=True)
class RegressionReport:
    passed: bool
    comparison_scope: str
    detail_comparison_refused: bool
    atol: float
    rtol: float
    comparisons: tuple[EffectComparison, ...]
    missing_effects: tuple[str, ...]

    def write_json(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path


def _emitter_totals(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("effect arrays require an emitter axis")
    if not np.isfinite(array).all():
        raise ValueError("effect arrays must contain only finite values")
    return array if array.ndim == 1 else array.sum(axis=tuple(range(1, array.ndim)))


def _relative_error(actual: np.ndarray, expected: np.ndarray, atol: float) -> np.ndarray:
    return np.abs(actual - expected) / np.maximum(np.abs(expected), atol)


def compare_legacy_aggregates(
    canonical: Mapping[str, np.ndarray],
    legacy: Mapping[str, np.ndarray],
    effects: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
    intentional_differences: Mapping[str, str] | None = None,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    detail: bool = False,
) -> RegressionReport:
    """Compare per-emitter/global totals, never legacy endpoint cells.

    An intentional-difference record documents an expected mismatch and prevents
    that mismatch from failing the report. Missing inputs always fail, including
    effects listed as intentional differences.
    """
    if detail:
        raise ValueError(
            "detailed-cell comparison is invalid: legacy columns are endpoints, "
            "not canonical attribution drivers"
        )
    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be non-negative")
    alias_map = dict(aliases or {})
    difference_map = dict(intentional_differences or {})
    comparisons: list[EffectComparison] = []
    missing: list[str] = []

    for effect in effects:
        legacy_effect = alias_map.get(effect, effect)
        if effect not in canonical or legacy_effect not in legacy:
            missing.append(effect)
            continue
        current = _emitter_totals(canonical[effect])
        reference = _emitter_totals(legacy[legacy_effect])
        if current.shape != reference.shape:
            raise ValueError(
                f"{effect} emitter count differs: canonical={current.shape}, legacy={reference.shape}"
            )
        emitter_abs = np.abs(current - reference)
        emitter_rel = _relative_error(current, reference, atol)
        current_global = float(current.sum())
        reference_global = float(reference.sum())
        global_abs = abs(current_global - reference_global)
        global_rel = global_abs / max(abs(reference_global), atol)
        equal = bool(np.allclose(current, reference, atol=atol, rtol=rtol)) and bool(
            np.isclose(current_global, reference_global, atol=atol, rtol=rtol)
        )
        reason = difference_map.get(effect)
        comparisons.append(
            EffectComparison(
                effect=effect,
                legacy_effect=legacy_effect,
                passed=equal or reason is not None,
                intentional_difference=reason,
                max_emitter_abs_error=float(emitter_abs.max(initial=0.0)),
                max_emitter_rel_error=float(emitter_rel.max(initial=0.0)),
                global_abs_error=global_abs,
                global_rel_error=global_rel,
            )
        )

    passed = not missing and all(item.passed for item in comparisons)
    return RegressionReport(
        passed=passed,
        comparison_scope="per-emitter and global totals only",
        detail_comparison_refused=True,
        atol=atol,
        rtol=rtol,
        comparisons=tuple(comparisons),
        missing_effects=tuple(missing),
    )


def load_legacy_manifest(path: str | Path) -> dict[str, np.ndarray]:
    """Load trusted ``.npy`` arrays declared by a small JSON manifest.

    Format: ``{"effects": {"legacy_name": "relative/path.npy"}}``.
    Paths are resolved beside the manifest and arrays are loaded without pickle.
    """
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    effects = payload.get("effects")
    if not isinstance(effects, dict) or not effects:
        raise ValueError("legacy manifest requires a non-empty 'effects' mapping")
    loaded: dict[str, np.ndarray] = {}
    for name, relative_path in effects.items():
        if not isinstance(name, str) or not isinstance(relative_path, str):
            raise ValueError("legacy effect names and paths must be strings")
        loaded[name] = np.load(manifest_path.parent / relative_path, allow_pickle=False)
    return loaded


def _require_aggregate_path(path: Path, approved_names: set[str]) -> None:
    """Refuse endpoint/detailed matrices before any optional reader is imported."""
    name = path.name
    if "_tot_" not in name and name not in approved_names:
        raise ValueError(
            f"legacy parquet must be an aggregate '*_tot_*' artifact or explicitly "
            f"approved by exact filename; refused {name!r}"
        )
    lowered = name.lower()
    if any(token in lowered for token in ("detailed", "flows", "matrix")):
        raise ValueError(f"detailed legacy artifact is not safe for comparison: {name!r}")


def _label_values(frame, label: str) -> tuple[str, ...]:
    if label in frame.columns:
        return tuple(str(value) for value in frame[label].tolist())
    index = frame.index
    if getattr(index, "nlevels", 1) > 1 and label in tuple(index.names):
        return tuple(str(value) for value in index.get_level_values(label).tolist())
    if getattr(index, "name", None) == label:
        return tuple(str(value) for value in index.tolist())
    raise ValueError(f"legacy aggregate is missing required {label!r} labels")


def load_legacy_parquet_profile(
    path: str | Path, *, artifact_root: str | Path | None = None
) -> LegacyParquetBundle:
    """Load labelled aggregate-only Parquet artifacts from a JSON profile.

    Pandas and a Parquet engine are optional and imported only after every path
    has passed the aggregate filename gate. The profile's ``effects`` entries
    contain ``path`` and optionally ``value_column`` (default: effect name).
    """
    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    effects = payload.get("effects")
    if not isinstance(effects, dict) or not effects:
        raise ValueError("legacy parquet profile requires a non-empty 'effects' mapping")
    approved = payload.get("approved_aggregate_filenames", [])
    if not isinstance(approved, list) or not all(isinstance(item, str) for item in approved):
        raise ValueError("approved_aggregate_filenames must be a list of exact filenames")

    configured_root = payload.get("artifact_root")
    if artifact_root is None and configured_root is not None:
        if not isinstance(configured_root, str):
            raise ValueError("artifact_root must be a string path")
        artifact_root = configured_root
    base = Path(artifact_root) if artifact_root is not None else profile_path.parent
    resolved: dict[str, tuple[Path, str]] = {}
    for effect, specification in effects.items():
        if not isinstance(effect, str) or not isinstance(specification, dict):
            raise ValueError("effect names must map to configuration objects")
        relative = specification.get("path")
        value_column = specification.get("value_column", effect)
        if not isinstance(relative, str) or not isinstance(value_column, str):
            raise ValueError("each effect requires string path and value_column fields")
        artifact = base / relative
        _require_aggregate_path(artifact, set(approved))
        resolved[effect] = (artifact, value_column)

    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "legacy Parquet ingestion requires the 'legacy-parquet' optional extra"
        ) from error

    loaded: dict[str, LegacyAggregateEffect] = {}
    for effect, (artifact, value_column) in resolved.items():
        frame = pd.read_parquet(artifact)
        if value_column not in frame.columns:
            raise ValueError(f"{artifact.name!r} is missing value column {value_column!r}")
        countries = _label_values(frame, "Country")
        sectors = _label_values(frame, "Sector")
        values = frame[value_column].to_numpy(dtype=float, copy=True)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"{artifact.name!r} must contain one finite aggregate value per row")
        loaded[effect] = LegacyAggregateEffect(
            values=values,
            countries=countries,
            sectors=sectors,
            value_column=value_column,
            source_path=str(artifact),
        )

    aliases = payload.get("aliases", {})
    provenance = payload.get("provenance", {})
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in aliases.items()
    ):
        raise ValueError("aliases must map canonical names to legacy names")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    return LegacyParquetBundle(loaded, dict(aliases), dict(provenance))
