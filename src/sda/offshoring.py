"""Derived source-country attribution products built from canonical ``D_C`` math."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attribution import (
    SOURCE_ORIGIN_GROUPS,
    attribute_structural_components,
    build_group_sector_source_masks,
    build_per_country_source_masks,
    build_source_origin_masks,
)
from .effects import split_c_component_by_source, two_factor_structural_components
from .model import EffectDimensions, EffectResult, LabelSet
from .output.aggregation import collapse_driver_to_country
from .validation import ScaleAwareAdditivityReport, assert_scaled_additive

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SourceCountryCube:
    """Emitter × driver-country × source-country derived ``D_C`` attribution."""

    baseline: FloatArray
    emitter: LabelSet
    driver_country: LabelSet
    source_country: LabelSet
    reconciliation: ScaleAwareAdditivityReport
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.baseline, dtype=np.float64)
        expected = (
            len(self.emitter.labels),
            len(self.driver_country.labels),
            len(self.source_country.labels),
        )
        if values.shape != expected:
            raise ValueError(f"source-country cube shape {values.shape} does not match {expected}")
        if not np.isfinite(values).all():
            raise ValueError("source-country cube must contain only finite values")
        if self.emitter.role != "emitter_country_sector":
            raise ValueError("source-country cube emitter role is invalid")
        if self.driver_country.role != "structural_driver_country":
            raise ValueError("source-country cube driver role is invalid")
        if self.source_country.role != "source_country":
            raise ValueError("source-country cube source role is invalid")
        if not self.reconciliation.passed:
            raise ValueError("source-country cube reconciliation must pass")
        object.__setattr__(self, "baseline", values)


@dataclass(frozen=True)
class SourceGroupSectorCube:
    """Emitter × driver-country × source-group × source-sector ``D_C``."""

    baseline: FloatArray
    emitter: LabelSet
    driver_country: LabelSet
    source_group: LabelSet
    source_sector: LabelSet
    reconciliation: ScaleAwareAdditivityReport
    group_reconciliation: Mapping[str, ScaleAwareAdditivityReport]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.baseline, dtype=np.float64)
        expected = (
            len(self.emitter.labels), len(self.driver_country.labels),
            len(self.source_group.labels), len(self.source_sector.labels),
        )
        if values.shape != expected:
            raise ValueError(f"source-group-sector cube shape {values.shape} does not match {expected}")
        if not np.isfinite(values).all():
            raise ValueError("source-group-sector cube must contain only finite values")
        if self.source_group.role != "source_group" or self.source_sector.role != "source_sector":
            raise ValueError("source-group-sector cube source roles are invalid")
        if not self.reconciliation.passed or any(
            not report.passed for report in self.group_reconciliation.values()
        ):
            raise ValueError("source-group-sector reconciliation must pass")
        object.__setattr__(self, "baseline", values)


def build_source_country_offshoring_cube(
    C0: ArrayLike,
    C1: ArrayLike,
    A_x0: ArrayLike,
    A_x1: ArrayLike,
    *,
    v0: ArrayLike,
    delta_v: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    y0: ArrayLike,
    delta_y: ArrayLike,
    source_countries: Sequence[str],
    destination_countries: Sequence[str],
    emitter_labels: Sequence[Any],
    structural_driver_labels: Sequence[Any],
    country_order: Sequence[str] | None = None,
    backend: str = "numpy",
    batch_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SourceCountryCube:
    """Build the CPU reference source-country split without long-form rows.

    ``delta C`` is partitioned before propagation. Each source component is
    attributed independently with the canonical structural kernel and its
    driver-sector axis is then collapsed to driver country. The source axis
    is retained as the final (dense) cube axis.
    """

    masks = build_per_country_source_masks(
        source_countries, destination_countries, country_order=country_order
    )
    components = split_c_component_by_source(C0, C1, A_x0, A_x1, masks)
    emitter = LabelSet("emitter_country_sector", tuple(emitter_labels))
    driver = LabelSet(
        "structural_driver_country_sector", tuple(structural_driver_labels)
    )
    dimensions = EffectDimensions(emitter, driver)
    propagation = {
        "v0": v0,
        "delta_v": delta_v,
        "L0": L0,
        "L1": L1,
        "y0": y0,
        "delta_y": delta_y,
    }

    collapsed_labels: LabelSet | None = None
    collapse_rtol = 1e-6 if backend == "cupy" else 1e-12

    def collapse(_key: Any, attributed: FloatArray) -> FloatArray:
        nonlocal collapsed_labels
        collapsed = collapse_driver_to_country(
            EffectResult("D_C", attributed, dimensions), rtol=collapse_rtol,
        )
        if collapsed_labels is None:
            collapsed_labels = collapsed.dimensions.column
        elif collapsed.dimensions.column != collapsed_labels:
            raise ValueError("source-country components produced inconsistent driver axes")
        return np.asarray(collapsed.baseline, dtype=np.float64)

    collapsed_components = attribute_structural_components(
        components, **propagation, backend=backend, batch_size=batch_size,
        result_transform=collapse,
    )
    collapsed_parts = list(collapsed_components.values())

    unsplit_component = two_factor_structural_components(C0, C1, A_x0, A_x1).C
    unsplit = attribute_structural_components(
        {"D_C": unsplit_component}, **propagation, backend=backend,
        batch_size=batch_size, result_transform=collapse,
    )["D_C"]
    reconciliation = assert_scaled_additive(collapsed_parts, unsplit)
    if collapsed_labels is None:  # masks are validated as non-empty; defensive only
        raise ValueError("source-country cube requires at least one source country")

    return SourceCountryCube(
        baseline=np.stack(collapsed_parts, axis=-1),
        emitter=emitter,
        driver_country=collapsed_labels,
        source_country=LabelSet("source_country", tuple(components)),
        reconciliation=reconciliation,
        metadata={
            "effect": "D_C",
            "method": "pre-propagation source-country split",
            "backend": backend,
            **dict(metadata or {}),
        },
    )


def build_source_group_sector_offshoring_cube(
    C0: ArrayLike,
    C1: ArrayLike,
    A_x0: ArrayLike,
    A_x1: ArrayLike,
    *,
    v0: ArrayLike,
    delta_v: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    y0: ArrayLike,
    delta_y: ArrayLike,
    source_countries: Sequence[str],
    source_sectors: Sequence[str],
    destination_countries: Sequence[str],
    source_group_by_country: Mapping[str, str],
    emitter_labels: Sequence[Any],
    structural_driver_labels: Sequence[Any],
    sector_order: Sequence[str] | None = None,
    backend: str = "numpy",
    batch_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SourceGroupSectorCube:
    """Build the source-group×sector cube and both mandatory split gates."""

    masks = build_group_sector_source_masks(
        source_countries, source_sectors, destination_countries,
        source_group_by_country, sector_order=sector_order,
    )
    components = split_c_component_by_source(C0, C1, A_x0, A_x1, masks)
    propagation = {
        "v0": v0, "delta_v": delta_v, "L0": L0, "L1": L1,
        "y0": y0, "delta_y": delta_y,
    }
    emitter = LabelSet("emitter_country_sector", tuple(emitter_labels))
    driver = LabelSet("structural_driver_country_sector", tuple(structural_driver_labels))
    dimensions = EffectDimensions(emitter, driver)
    sectors = tuple(dict.fromkeys(source_sectors)) if sector_order is None else tuple(sector_order)

    driver_country: LabelSet | None = None
    collapse_rtol = 1e-6 if backend == "cupy" else 1e-12

    def collapse(_key: Any, values: FloatArray) -> FloatArray:
        nonlocal driver_country
        effect = collapse_driver_to_country(
            EffectResult("D_C", values, dimensions), rtol=collapse_rtol,
        )
        driver_country = driver_country or effect.dimensions.column
        return np.asarray(effect.baseline, dtype=np.float64)

    collapsed = attribute_structural_components(
        components, **propagation, backend=backend, batch_size=batch_size,
        result_transform=collapse,
    )

    unsplit = two_factor_structural_components(C0, C1, A_x0, A_x1).C
    parent = attribute_structural_components(
        {"D_C": unsplit}, **propagation, backend=backend, batch_size=batch_size,
        result_transform=collapse,
    )["D_C"]
    reconciliation = assert_scaled_additive(collapsed.values(), parent)

    canonical_group_masks = build_source_origin_masks(
        source_countries, destination_countries, source_group_by_country
    )
    canonical_group_components = split_c_component_by_source(
        C0, C1, A_x0, A_x1, canonical_group_masks
    )
    canonical_group_values = attribute_structural_components(
        canonical_group_components, **propagation, backend=backend, batch_size=batch_size,
        result_transform=collapse,
    )
    group_reports: dict[str, ScaleAwareAdditivityReport] = {}
    for group in SOURCE_ORIGIN_GROUPS:
        expected = canonical_group_values[group]
        group_reports[group] = assert_scaled_additive(
            [collapsed[(group, sector)] for sector in sectors], expected
        )
    if driver_country is None:  # defensive; validated masks are non-empty
        raise ValueError("source-group-sector cube requires at least one component")
    values = np.stack(
        [collapsed[(group, sector)] for group in SOURCE_ORIGIN_GROUPS for sector in sectors],
        axis=-1,
    ).reshape(len(emitter.labels), len(driver_country.labels), len(SOURCE_ORIGIN_GROUPS), len(sectors))
    return SourceGroupSectorCube(
        baseline=values,
        emitter=emitter,
        driver_country=driver_country,
        source_group=LabelSet("source_group", SOURCE_ORIGIN_GROUPS),
        source_sector=LabelSet("source_sector", sectors),
        reconciliation=reconciliation,
        group_reconciliation=group_reports,
        metadata={"effect": "D_C", "method": "pre-propagation source-group-sector split", "backend": backend, **dict(metadata or {})},
    )
