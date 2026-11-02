"""Compact source-to-destination group/sector interaction products."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attribution.groups import EXTERNAL_SOURCE_GROUPS, SOURCE_ORIGIN_GROUPS
from .effects.structural import two_factor_structural_components
from .final_demand_sourcing import YFSourceGroupSectorCube
from .model import LabelSet
from .validation import ScaleAwareAdditivityReport, assert_scaled_additive

FloatArray = NDArray[np.float64]
SOURCING_RELATIONSHIPS = ("Domestic", "Cross-border")


@dataclass(frozen=True)
class SourceGroupSectorInteraction:
    """D_C relationship × source group-sector × destination group-sector."""

    baseline: FloatArray
    relationship: LabelSet
    source_group: LabelSet
    source_sector: LabelSet
    destination_group: LabelSet
    destination_sector: LabelSet
    reconciliation: ScaleAwareAdditivityReport
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = tuple(len(axis.labels) for axis in (
            self.relationship, self.source_group, self.source_sector,
            self.destination_group, self.destination_sector,
        ))
        values = np.asarray(self.baseline, dtype=np.float64)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"D_C interaction must be finite with shape {expected}")
        if not self.reconciliation.passed:
            raise ValueError("D_C interaction reconciliation must pass")
        object.__setattr__(self, "baseline", values)


@dataclass(frozen=True)
class YFGroupProductInteraction:
    """D_YF relationship × source group × destination group × product sector."""

    baseline: FloatArray
    relationship: LabelSet
    source_group: LabelSet
    destination_group: LabelSet
    product_sector: LabelSet
    reconciliation: ScaleAwareAdditivityReport
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = tuple(len(axis.labels) for axis in (
            self.relationship, self.source_group, self.destination_group,
            self.product_sector,
        ))
        values = np.asarray(self.baseline, dtype=np.float64)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"D_YF interaction must be finite with shape {expected}")
        if not self.reconciliation.passed:
            raise ValueError("D_YF interaction reconciliation must pass")
        object.__setattr__(self, "baseline", values)


def _labels(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{name} labels must be non-empty strings")
    return result


def _groups(countries: Sequence[str], mapping: Mapping[str, str]) -> np.ndarray:
    missing = sorted(set(countries) - set(mapping))
    if missing:
        raise ValueError(f"missing country-group classifications: {missing}")
    groups = np.asarray([mapping[country] for country in countries], dtype=object)
    invalid = sorted(set(groups) - set(EXTERNAL_SOURCE_GROUPS))
    if invalid:
        raise ValueError(f"invalid absolute country groups: {invalid}")
    return groups


def build_source_group_sector_interaction(
    C0: ArrayLike, C1: ArrayLike, A_x0: ArrayLike, A_x1: ArrayLike, *,
    v0: ArrayLike, delta_v: ArrayLike, L0: ArrayLike, L1: ArrayLike,
    y0: ArrayLike, delta_y: ArrayLike,
    source_countries: Sequence[str], source_sectors: Sequence[str],
    destination_countries: Sequence[str], destination_sectors: Sequence[str],
    country_group_by_country: Mapping[str, str],
    sector_order: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SourceGroupSectorInteraction:
    """Build an emitter-summed D_C source/destination interaction matrix."""

    source_country = np.asarray(_labels(source_countries, "source-country"), dtype=object)
    source_sector = np.asarray(_labels(source_sectors, "source-sector"), dtype=object)
    destination_country = np.asarray(_labels(destination_countries, "destination-country"), dtype=object)
    destination_sector = np.asarray(_labels(destination_sectors, "destination-sector"), dtype=object)
    n = len(source_country)
    if any(len(axis) != n for axis in (source_sector, destination_country, destination_sector)):
        raise ValueError("source and destination country-sector axes must share one length")
    sectors = tuple(dict.fromkeys(source_sector)) if sector_order is None else tuple(sector_order)
    if set(sectors) != set(source_sector):
        raise ValueError("sector_order must contain every observed sector exactly once")
    source_groups = _groups(source_country, country_group_by_country)
    destination_groups = _groups(destination_country, country_group_by_country)

    component = two_factor_structural_components(C0, C1, A_x0, A_x1).C
    baseline_v = np.asarray(v0, dtype=np.float64)
    change_v = np.asarray(delta_v, dtype=np.float64)
    baseline_L = np.asarray(L0, dtype=np.float64)
    current_L = np.asarray(L1, dtype=np.float64)
    baseline_y = np.asarray(y0, dtype=np.float64)
    change_y = np.asarray(delta_y, dtype=np.float64)
    if component.shape != (n, n) or baseline_L.shape != (n, n) or current_L.shape != (n, n):
        raise ValueError("D_C interaction matrices must match the country-sector axes")
    if any(value.shape != (n,) for value in (baseline_v, change_v, baseline_y, change_y)):
        raise ValueError("D_C interaction vectors must match the country-sector axes")

    baseline_left = baseline_v @ baseline_L
    change_left = change_v @ baseline_L
    baseline_requirement = current_L @ baseline_y
    change_requirement = current_L @ change_y

    def finish(base_projection: FloatArray, change_projection: FloatArray) -> FloatArray:
        return (
            base_projection * baseline_requirement
            + 0.5 * base_projection * change_requirement
            + 0.5 * change_projection * baseline_requirement
            + (1.0 / 3.0) * change_projection * change_requirement
        )

    group_labels = tuple(EXTERNAL_SOURCE_GROUPS)
    values = np.zeros((len(SOURCING_RELATIONSHIPS), len(group_labels), len(sectors),
                       len(group_labels), len(sectors)), dtype=np.float64)
    for source_group_index, source_group in enumerate(group_labels):
        countries_in_group = tuple(dict.fromkeys(source_country[source_groups == source_group]))
        for source_sector_index, sector in enumerate(sectors):
            rows = np.flatnonzero((source_groups == source_group) & (source_sector == sector))
            base_total = baseline_left[rows] @ component[rows, :]
            change_total = change_left[rows] @ component[rows, :]
            base_domestic = np.zeros(n, dtype=np.float64)
            change_domestic = np.zeros(n, dtype=np.float64)
            for country in countries_in_group:
                country_rows = np.flatnonzero((source_country == country) & (source_sector == sector))
                columns = np.flatnonzero(destination_country == country)
                if len(country_rows) != 1:
                    raise ValueError(f"country-sector labels must be unique for {country}/{sector}")
                row = int(country_rows[0])
                base_domestic[columns] = baseline_left[row] * component[row, columns]
                change_domestic[columns] = change_left[row] * component[row, columns]
            relationship_vectors = (
                finish(base_domestic, change_domestic),
                finish(base_total - base_domestic, change_total - change_domestic),
            )
            for relationship_index, vector in enumerate(relationship_vectors):
                for destination_group_index, destination_group in enumerate(group_labels):
                    for destination_sector_index, destination_sector_label in enumerate(sectors):
                        columns = ((destination_groups == destination_group)
                                   & (destination_sector == destination_sector_label))
                        values[relationship_index, source_group_index, source_sector_index,
                               destination_group_index, destination_sector_index] = vector[columns].sum()

    parent_vector = finish(baseline_left @ component, change_left @ component)
    parent = np.zeros((len(group_labels), len(sectors)), dtype=np.float64)
    for group_index, group in enumerate(group_labels):
        for sector_index, sector in enumerate(sectors):
            parent[group_index, sector_index] = parent_vector[
                (destination_groups == group) & (destination_sector == sector)
            ].sum()
    parts = (values[relationship, group, sector]
             for relationship in range(len(SOURCING_RELATIONSHIPS))
             for group in range(len(group_labels)) for sector in range(len(sectors)))
    reconciliation = assert_scaled_additive(parts, parent)
    return SourceGroupSectorInteraction(
        values, LabelSet("sourcing_relationship", SOURCING_RELATIONSHIPS),
        LabelSet("source_country_group", group_labels), LabelSet("source_sector", sectors),
        LabelSet("destination_country_group", group_labels), LabelSet("destination_sector", sectors),
        reconciliation, {"effect": "D_C", "emitter_axis": "summed", **dict(metadata or {})},
    )


def build_yf_group_product_interaction(
    cube: YFSourceGroupSectorCube, *,
    consumer_group_by_country: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
) -> YFGroupProductInteraction:
    """Aggregate a validated D_YF cube to relationship/group/product resolution."""

    consumers = tuple(str(value) for value in cube.consumer_country.labels)
    destination_groups = _groups(consumers, consumer_group_by_country)
    group_labels = tuple(EXTERNAL_SOURCE_GROUPS)
    sectors = tuple(str(value) for value in cube.source_sector.labels)
    relative_groups = tuple(str(value) for value in cube.source_group.labels)
    if relative_groups != SOURCE_ORIGIN_GROUPS:
        raise ValueError("D_YF cube must use the canonical relative source groups")
    by_consumer = np.asarray(cube.baseline, dtype=np.float64).sum(axis=0)
    values = np.zeros((len(SOURCING_RELATIONSHIPS), len(group_labels),
                       len(group_labels), len(sectors)), dtype=np.float64)
    for consumer_index, destination_group in enumerate(destination_groups):
        destination_group_index = group_labels.index(str(destination_group))
        for relative_group_index, relative_group in enumerate(relative_groups):
            if relative_group == "Domestic":
                relationship_index = 0
                source_group_index = destination_group_index
            else:
                relationship_index = 1
                source_group_index = group_labels.index(relative_group)
            values[relationship_index, source_group_index, destination_group_index] += (
                by_consumer[consumer_index, relative_group_index]
            )
    parent = np.zeros((len(group_labels), len(sectors)), dtype=np.float64)
    consumer_parent = by_consumer.sum(axis=1)
    for group_index, group in enumerate(group_labels):
        parent[group_index] = consumer_parent[destination_groups == group].sum(axis=0)
    parts = (values[relationship, source_group]
             for relationship in range(len(SOURCING_RELATIONSHIPS))
             for source_group in range(len(group_labels)))
    reconciliation = assert_scaled_additive(parts, parent)
    return YFGroupProductInteraction(
        values, LabelSet("sourcing_relationship", SOURCING_RELATIONSHIPS),
        LabelSet("source_country_group", group_labels),
        LabelSet("destination_country_group", group_labels),
        LabelSet("demanded_sector", sectors), reconciliation,
        {"effect": "D_YF", "emitter_axis": "summed", **dict(metadata or {})},
    )
