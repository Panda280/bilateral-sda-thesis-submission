"""Dimension-aware data contracts for canonical SDA results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .hierarchy import CANONICAL_EFFECTS

Array = NDArray[np.floating]


@dataclass(frozen=True)
class LabelSet:
    """Ordered labels for one semantic axis."""

    role: str
    labels: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("label role must be non-empty")
        if not self.labels:
            raise ValueError(f"{self.role} labels must be non-empty")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"{self.role} labels must be unique")


@dataclass(frozen=True)
class EffectDimensions:
    row: LabelSet
    column: LabelSet | None = None


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    checks: Mapping[str, bool]
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        missing_or_failed = [name for name, ok in self.checks.items() if not ok]
        if not self.passed or missing_or_failed:
            raise ValueError(f"SDA validation failed: {missing_or_failed or 'unspecified failure'}")


@dataclass(frozen=True)
class EffectResult:
    name: str
    baseline: Array
    dimensions: EffectDimensions
    mean: Array | None = None
    source_categories: Mapping[str, "EffectResult"] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in CANONICAL_EFFECTS:
            raise ValueError(f"non-canonical effect name: {self.name}")
        data = np.asarray(self.baseline)
        expected = (len(self.dimensions.row.labels),)
        if self.dimensions.column is not None:
            expected += (len(self.dimensions.column.labels),)
        if data.shape != expected:
            raise ValueError(f"{self.name} shape {data.shape} does not match dimensions {expected}")
        if not np.isfinite(data).all():
            raise ValueError(f"{self.name} contains non-finite values")
        if self.mean is not None and np.asarray(self.mean).shape != expected:
            raise ValueError(f"{self.name} mean shape does not match dimensions {expected}")


@dataclass(frozen=True)
class SDAResult:
    effects: Mapping[str, EffectResult]
    total_change: Array
    validation: ValidationReport

    def __post_init__(self) -> None:
        missing = set(CANONICAL_EFFECTS) - set(self.effects)
        unexpected = set(self.effects) - set(CANONICAL_EFFECTS)
        if missing or unexpected:
            raise ValueError(f"effect inventory mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        self.validation.raise_if_invalid()
