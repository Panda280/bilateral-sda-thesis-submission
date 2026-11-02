"""Detailed source-origin products for canonical final-demand sourcing ``D_YF``."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attribution.final_demand import (
    attribute_final_demand_composition,
    attribute_final_demand_compositions,
)
from .attribution.groups import (
    SOURCE_ORIGIN_GROUPS,
    build_group_sector_source_masks,
    build_per_country_source_masks,
    build_source_origin_masks,
)
from .effects.final_demand import split_yf_component_by_source
from .model import LabelSet
from .validation import ScaleAwareAdditivityReport, assert_scaled_additive

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class YFSourceCountryCube:
    """Emitter × consumer-country × source-country ``D_YF`` sourcing."""

    baseline: FloatArray
    emitter: LabelSet
    consumer_country: LabelSet
    source_country: LabelSet
    reconciliation: ScaleAwareAdditivityReport
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (len(self.emitter.labels), len(self.consumer_country.labels),
                    len(self.source_country.labels))
        values = np.asarray(self.baseline, dtype=np.float64)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"D_YF source-country cube must be finite with shape {expected}")
        if self.consumer_country.role != "consumer_country":
            raise ValueError("D_YF consumer role is invalid")
        if self.source_country.role != "source_country" or not self.reconciliation.passed:
            raise ValueError("D_YF source-country role or reconciliation is invalid")
        object.__setattr__(self, "baseline", values)


@dataclass(frozen=True)
class YFSourceGroupSectorCube:
    """Emitter × consumer × source-group × source-sector ``D_YF`` sourcing."""

    baseline: FloatArray
    emitter: LabelSet
    consumer_country: LabelSet
    source_group: LabelSet
    source_sector: LabelSet
    reconciliation: ScaleAwareAdditivityReport
    group_reconciliation: Mapping[str, ScaleAwareAdditivityReport]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (len(self.emitter.labels), len(self.consumer_country.labels),
                    len(self.source_group.labels),
                    len(self.source_sector.labels))
        values = np.asarray(self.baseline, dtype=np.float64)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"D_YF source-group-sector cube must be finite with shape {expected}")
        if self.source_group.role != "source_group" or self.source_sector.role != "source_sector":
            raise ValueError("D_YF source roles are invalid")
        if not self.reconciliation.passed or any(not report.passed for report in self.group_reconciliation.values()):
            raise ValueError("D_YF source-group-sector reconciliation is invalid")
        object.__setattr__(self, "baseline", values)


def _attribute(component: ArrayLike, *, v0: ArrayLike, delta_v: ArrayLike, L0: ArrayLike,
               L1: ArrayLike, sectors_per_country: int) -> FloatArray:
    return attribute_final_demand_composition(component, v0=v0, delta_v=delta_v,
        L0=L0, L1=L1, sectors_per_country=sectors_per_country)


def build_yf_source_country_cube(yf_component: ArrayLike, *, v0: ArrayLike, delta_v: ArrayLike,
        L0: ArrayLike, L1: ArrayLike, sectors_per_country: int,
        source_countries: Sequence[str], consumer_countries: Sequence[str],
        emitter_labels: Sequence[Any],
        country_order: Sequence[str] | None = None,
        backend: str = "numpy", batch_size: int | None = None,
        metadata: Mapping[str, Any] | None = None) -> YFSourceCountryCube:
    masks = build_per_country_source_masks(source_countries, consumer_countries, country_order)
    components = split_yf_component_by_source(yf_component, masks)
    kwargs = dict(v0=v0, delta_v=delta_v, L0=L0, L1=L1, sectors_per_country=sectors_per_country)
    n = len(emitter_labels)
    consumers = len(consumer_countries)

    def collapse(_key: Any, values: FloatArray) -> FloatArray:
        return values.reshape(n, consumers, sectors_per_country).sum(axis=2)

    attributed = attribute_final_demand_compositions(
        components, **kwargs, backend=backend, batch_size=batch_size,
        result_transform=collapse,
    )
    parent = attribute_final_demand_compositions(
        {"parent": yf_component}, **kwargs, backend=backend, batch_size=1,
        result_transform=collapse,
    )["parent"]
    report = assert_scaled_additive(attributed.values(), parent)
    return YFSourceCountryCube(np.stack(list(attributed.values()), axis=-1),
        LabelSet("emitter_country_sector", tuple(emitter_labels)),
        LabelSet("consumer_country", tuple(consumer_countries)),
        LabelSet("source_country", tuple(attributed)), report,
        {"effect": "D_YF", "method": "pre-propagation source-country split; demanded sector collapsed",
         "backend": backend, **dict(metadata or {})})


def build_yf_source_group_sector_cube(yf_component: ArrayLike, *, v0: ArrayLike, delta_v: ArrayLike,
        L0: ArrayLike, L1: ArrayLike, sectors_per_country: int,
        source_countries: Sequence[str], source_sectors: Sequence[str],
        consumer_countries: Sequence[str], source_group_by_country: Mapping[str, str],
        emitter_labels: Sequence[Any],
        sector_order: Sequence[str] | None = None,
        backend: str = "numpy", batch_size: int | None = None,
        metadata: Mapping[str, Any] | None = None) -> YFSourceGroupSectorCube:
    masks = build_group_sector_source_masks(source_countries, source_sectors, consumer_countries,
                                             source_group_by_country, sector_order=sector_order)
    components = split_yf_component_by_source(yf_component, masks)
    kwargs = dict(v0=v0, delta_v=delta_v, L0=L0, L1=L1, sectors_per_country=sectors_per_country)
    sectors = tuple(dict.fromkeys(source_sectors)) if sector_order is None else tuple(sector_order)
    ordered_components = {
        (group, sector): components[(group, sector)]
        for group in SOURCE_ORIGIN_GROUPS for sector in sectors
    }
    n = len(emitter_labels)
    consumers = len(consumer_countries)

    def collapse(_key: Any, values: FloatArray) -> FloatArray:
        return values.reshape(n, consumers, sectors_per_country).sum(axis=2)

    attributed = attribute_final_demand_compositions(
        ordered_components, **kwargs, backend=backend, batch_size=batch_size,
        result_transform=collapse,
    )
    parent = attribute_final_demand_compositions(
        {"parent": yf_component}, **kwargs, backend=backend, batch_size=1,
        result_transform=collapse,
    )["parent"]
    report = assert_scaled_additive(attributed.values(), parent)
    group_masks = build_source_origin_masks(source_countries, consumer_countries, source_group_by_country)
    group_components = split_yf_component_by_source(yf_component, group_masks)
    group_values = attribute_final_demand_compositions(
        group_components, **kwargs, backend=backend, batch_size=batch_size,
        result_transform=collapse,
    )
    group_reports = {group: assert_scaled_additive(
        [attributed[(group, sector)] for sector in sectors], group_values[group]) for group in SOURCE_ORIGIN_GROUPS}
    values = np.stack([attributed[(group, sector)]
                       for group in SOURCE_ORIGIN_GROUPS for sector in sectors], axis=-1)
    values = values.reshape(n, consumers, len(SOURCE_ORIGIN_GROUPS), len(sectors))
    return YFSourceGroupSectorCube(values, LabelSet("emitter_country_sector", tuple(emitter_labels)),
        LabelSet("consumer_country", tuple(consumer_countries)),
        LabelSet("source_group", SOURCE_ORIGIN_GROUPS), LabelSet("source_sector", sectors), report, group_reports,
        {"effect": "D_YF", "method": "pre-propagation source-group-sector split; demanded sector collapsed",
         "backend": backend, **dict(metadata or {})})
