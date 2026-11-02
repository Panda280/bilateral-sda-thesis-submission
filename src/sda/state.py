"""Validated input state and narrow adapters for external calculation results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attribution.groups import SOURCE_ORIGIN_GROUPS

FloatArray = NDArray[np.float64]

# Tolerance for the "A_x = T_coef * w_coef" identity below. Looser than the
# rtol=1e-10/atol=1e-12 used elsewhere in this package for values that are
# genuinely computed in float64 throughout (e.g. synthetic test fixtures, or
# arrays this package itself derives). A_x/T_coef/w_coef, when sourced from
# EXIODeflatorStateAdapter, come from EXIODeflator's own GPU pipeline, which
# computes entirely in float32 (SDA.config.DTYPE = cupy.float32) -- so the
# identity can only ever hold to float32 rounding precision, not float64's.
# Measured on a real 2003 WIOT2013 CYP period (2026-08-27): 39647/2059225
# cells differ, max abs diff 2.98e-08 (== 2**-25, one float32 ULP at
# magnitude ~1) -- confined to normal (w >= 1e-6) columns, zero diff at the
# clipped (w < 1e-6) columns. That is plain float32 division/multiplication
# roundoff, not a systematic divergence, so this tolerance stays many orders
# of magnitude tighter than would be needed to hide a real one (e.g. the
# deliberately-mismatched fixture in test_state.py uses a magnitude-~1 to
# ~2 gap, far above this).
A_X_IDENTITY_RTOL = 1e-5
A_X_IDENTITY_ATOL = 1e-6


@dataclass(frozen=True)
class StateLabels:
    """Ordered labels that define every semantic state axis."""

    emitters: tuple[object, ...]
    structural_drivers: tuple[object, ...]
    consumers: tuple[object, ...]
    demanded_sectors: tuple[object, ...]

    def __post_init__(self) -> None:
        for field in fields(self):
            labels = getattr(self, field.name)
            if not labels or len(set(labels)) != len(labels):
                raise ValueError(f"{field.name} labels must be non-empty and unique")
        if self.structural_drivers != self.emitters:
            raise ValueError("structural driver labels must use the emitter country-sector order")


@dataclass(frozen=True)
class SourceGroupInputs:
    """Canonical, exhaustive source-origin partitions for C and YF."""

    C: Mapping[str, ArrayLike]
    YF: Mapping[str, ArrayLike]

    def validated(self, *, structural_shape: tuple[int, int], final_demand_shape: tuple[int, int]) -> "SourceGroupInputs":
        return SourceGroupInputs(
            C=_validate_partition(self.C, structural_shape, "C"),
            YF=_validate_partition(self.YF, final_demand_shape, "YF"),
        )


def _validate_partition(
    masks: Mapping[str, ArrayLike], shape: tuple[int, int], name: str
) -> dict[str, NDArray[np.bool_]]:
    if set(masks) != set(SOURCE_ORIGIN_GROUPS):
        raise ValueError(f"{name} groups must be exactly {SOURCE_ORIGIN_GROUPS}")
    result = {group: np.asarray(masks[group], dtype=bool) for group in SOURCE_ORIGIN_GROUPS}
    if any(mask.shape != shape for mask in result.values()):
        raise ValueError(f"{name} source masks must have shape {shape}")
    total = np.sum(np.stack(tuple(result.values())), axis=0)
    if not np.array_equal(total, np.ones(shape, dtype=np.int64)):
        raise ValueError(f"{name} source masks must be mutually exclusive and exhaustive")
    return result


@dataclass(frozen=True)
class SDAState:
    """Complete, dimension-checked inputs for one two-period SDA run."""

    v0: ArrayLike
    v1: ArrayLike
    L0: ArrayLike
    L1: ArrayLike
    C0: ArrayLike
    C1: ArrayLike
    A_x0: ArrayLike
    A_x1: ArrayLike
    T_coef0: ArrayLike
    T_coef1: ArrayLike
    w_coef0: ArrayLike
    w_coef1: ArrayLike
    F0: ArrayLike
    F1: ArrayLike
    B0: ArrayLike
    B1: ArrayLike
    levels0: ArrayLike
    levels1: ArrayLike
    supplier_sector: ArrayLike
    sectors_per_country: int
    labels: StateLabels
    source_groups: SourceGroupInputs

    def __post_init__(self) -> None:
        n = len(self.labels.emitters)
        c = len(self.labels.consumers)
        s = len(self.labels.demanded_sectors)
        if self.sectors_per_country != s or n != c * s:
            raise ValueError("label counts must describe a complete country-sector ordering")

        vectors = ("v0", "v1", "w_coef0", "w_coef1")
        matrices = ("L0", "L1", "C0", "C1", "A_x0", "A_x1", "T_coef0", "T_coef1")
        for name in vectors:
            self._set_float(name, (n,))
        for name in matrices:
            self._set_float(name, (n, n))
        for name in ("F0", "F1"):
            self._set_float(name, (n, c))
        for name in ("B0", "B1"):
            self._set_float(name, (s, c))
        for name in ("levels0", "levels1"):
            self._set_float(name, (c,))

        sector = np.asarray(self.supplier_sector)
        expected_sector = np.tile(np.arange(s, dtype=np.int64), c)
        if sector.shape != (n,) or not np.issubdtype(sector.dtype, np.integer):
            raise ValueError("supplier_sector must be an integer vector matching emitters")
        if not np.array_equal(sector, expected_sector):
            raise ValueError("supplier_sector must follow consumer-country then demanded-sector order")
        object.__setattr__(self, "supplier_sector", sector.astype(np.int64, copy=True))

        for period in (0, 1):
            actual = getattr(self, f"A_x{period}")
            expected = getattr(self, f"T_coef{period}") * getattr(self, f"w_coef{period}")[None, :]
            if not np.allclose(actual, expected, rtol=A_X_IDENTITY_RTOL, atol=A_X_IDENTITY_ATOL):
                raise ValueError(f"A_x{period} must equal T_coef{period} * w_coef{period} cellwise")

        groups = self.source_groups.validated(
            structural_shape=(n, n), final_demand_shape=(n, c)
        )
        object.__setattr__(self, "source_groups", groups)

    def _set_float(self, name: str, shape: tuple[int, ...]) -> None:
        value = np.asarray(getattr(self, name), dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        object.__setattr__(self, name, value)

    def coordinator_arguments(self) -> dict[str, Any]:
        """Return the legacy keyword surface consumed by the coordinator."""

        names = (
            "v0", "v1", "L0", "L1", "C0", "C1", "A_x0", "A_x1",
            "T_coef0", "T_coef1", "w_coef0", "w_coef1", "F0", "F1",
            "B0", "B1", "levels0", "levels1", "supplier_sector",
            "sectors_per_country",
        )
        result = {name: getattr(self, name) for name in names}
        result.update(
            emitter_labels=self.labels.emitters,
            structural_driver_labels=self.labels.structural_drivers,
            consumer_labels=self.labels.consumers,
            demanded_sector_labels=self.labels.demanded_sectors,
            c_source_masks=self.source_groups.C,
            yf_source_masks=self.source_groups.YF,
        )
        return result


class SDAStateAdapter(Protocol):
    """Boundary for adapters that produce a validated state."""

    def adapt(self) -> SDAState: ...


@dataclass(frozen=True)
class MappingStateAdapter:
    """Adapt already-computed arrays without importing their source project."""

    arrays: Mapping[str, Any]
    labels: StateLabels
    source_groups: SourceGroupInputs
    sectors_per_country: int
    keys: Mapping[str, str] | None = None

    def adapt(self) -> SDAState:
        array_names = tuple(
            field.name for field in fields(SDAState)
            if field.name not in {"labels", "source_groups", "sectors_per_country"}
        )
        key_map = self.keys or {}
        missing = [key_map.get(name, name) for name in array_names if key_map.get(name, name) not in self.arrays]
        if missing:
            raise ValueError(f"reference mapping is missing required arrays: {missing}")
        values = {name: self.arrays[key_map.get(name, name)] for name in array_names}
        return SDAState(
            **values,
            labels=self.labels,
            source_groups=self.source_groups,
            sectors_per_country=self.sectors_per_country,
        )
