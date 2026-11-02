"""Aggregate effect axes without guessing their semantics from array shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..model import EffectDimensions, EffectResult, LabelSet


@dataclass(frozen=True)
class AxisGrouping:
    """An ordered, exhaustive partition used to aggregate one labelled axis."""

    role: str
    groups: Mapping[Any, tuple[Any, ...]]

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("grouped axis role must be non-empty")
        if not self.groups:
            raise ValueError("axis grouping must contain at least one group")
        if any(not members for members in self.groups.values()):
            raise ValueError("axis grouping may not contain empty groups")

    @property
    def labels(self) -> tuple[Any, ...]:
        return tuple(self.groups)

    def indices_for(self, source: LabelSet) -> tuple[tuple[int, ...], ...]:
        """Validate this grouping against ``source`` and return source indices."""

        source_positions = {label: index for index, label in enumerate(source.labels)}
        members = tuple(member for group in self.groups.values() for member in group)

        duplicates = _duplicates(members)
        if duplicates:
            raise ValueError(f"axis grouping contains duplicate source labels: {duplicates!r}")

        missing = tuple(label for label in source.labels if label not in members)
        extra = tuple(label for label in members if label not in source_positions)
        if missing or extra:
            raise ValueError(
                f"axis grouping is not an exact partition of {source.role}; "
                f"missing={missing!r}, extra={extra!r}"
            )

        return tuple(
            tuple(source_positions[member] for member in group)
            for group in self.groups.values()
        )


def aggregate_effect(
    effect: EffectResult,
    *,
    rows: AxisGrouping,
    columns: AxisGrouping | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> EffectResult:
    """Aggregate an effect using independent mappings for its declared axes.

    A vector accepts only a row grouping. A two-dimensional effect requires both
    row and column groupings, even when its two axes happen to have equal sizes.
    """

    row_indices = rows.indices_for(effect.dimensions.row)
    column_labels = effect.dimensions.column
    if column_labels is None:
        if columns is not None:
            raise ValueError("one-dimensional effect has no column axis to aggregate")
        baseline = _aggregate_vector(np.asarray(effect.baseline), row_indices)
        mean = (
            None
            if effect.mean is None
            else _aggregate_vector(np.asarray(effect.mean), row_indices)
        )
        dimensions = EffectDimensions(LabelSet(rows.role, rows.labels))
    else:
        if columns is None:
            raise ValueError("two-dimensional effect requires a column grouping")
        column_indices = columns.indices_for(column_labels)
        baseline = _aggregate_matrix(
            np.asarray(effect.baseline), row_indices, column_indices
        )
        mean = (
            None
            if effect.mean is None
            else _aggregate_matrix(np.asarray(effect.mean), row_indices, column_indices)
        )
        dimensions = EffectDimensions(
            LabelSet(rows.role, rows.labels),
            LabelSet(columns.role, columns.labels),
        )

    _assert_conserved(effect.baseline, baseline, atol=atol, rtol=rtol)
    if effect.mean is not None:
        _assert_conserved(effect.mean, mean, atol=atol, rtol=rtol)

    metadata = dict(effect.metadata)
    metadata["aggregated_from_roles"] = tuple(
        role
        for role in (
            effect.dimensions.row.role,
            None if column_labels is None else column_labels.role,
        )
        if role is not None
    )
    source_categories = (
        None
        if effect.source_categories is None
        else {
            category: aggregate_effect(
                component, rows=rows, columns=columns, atol=atol, rtol=rtol
            )
            for category, component in effect.source_categories.items()
        }
    )
    return EffectResult(
        name=effect.name,
        baseline=baseline,
        mean=mean,
        dimensions=dimensions,
        source_categories=source_categories,
        metadata=metadata,
    )


def collapse_driver_to_country(
    effect: EffectResult,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> EffectResult:
    """Collapse structural-driver sectors to countries without changing emitters.

    Driver labels may be ``(country, sector)`` tuples or ``"country:sector"``
    strings. Countries retain their first-appearance order on the declared
    driver axis; no alphabetical sorting or shape-based inference is used.
    """

    columns = effect.dimensions.column
    if columns is None or columns.role != "structural_driver_country_sector":
        raise ValueError(
            "driver-country collapse requires a structural_driver_country_sector column axis"
        )
    if effect.source_categories is not None:
        raise ValueError("driver-country collapse does not support source-category effects")

    countries: list[str] = []
    country_positions: dict[str, int] = {}
    column_groups: list[list[int]] = []
    for index, label in enumerate(columns.labels):
        country = _driver_country(label)
        if country not in country_positions:
            country_positions[country] = len(countries)
            countries.append(country)
            column_groups.append([])
        column_groups[country_positions[country]].append(index)

    baseline = _sum_columns(np.asarray(effect.baseline), column_groups)
    mean = None if effect.mean is None else _sum_columns(np.asarray(effect.mean), column_groups)
    _assert_conserved(effect.baseline, baseline, atol=atol, rtol=rtol)
    if effect.mean is not None:
        _assert_conserved(effect.mean, mean, atol=atol, rtol=rtol)

    return EffectResult(
        name=effect.name,
        baseline=baseline,
        dimensions=EffectDimensions(
            effect.dimensions.row,
            LabelSet("structural_driver_country", tuple(countries)),
        ),
        mean=mean,
        metadata=effect.metadata,
    )


def _driver_country(label: Any) -> str:
    if isinstance(label, tuple):
        if len(label) != 2 or not all(isinstance(part, str) and part for part in label):
            raise ValueError(
                "structural driver tuple labels must be non-empty (country, sector) strings"
            )
        return label[0]
    if isinstance(label, str):
        parts = label.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "flat structural driver labels must use non-empty 'country:sector' form"
            )
        return parts[0]
    raise ValueError(
        "structural driver labels must be (country, sector) tuples or 'country:sector' strings"
    )


def _sum_columns(
    values: NDArray[np.floating], groups: list[list[int]]
) -> NDArray[np.floating]:
    return np.column_stack([values[:, indices].sum(axis=1) for indices in groups])


def _aggregate_vector(
    values: NDArray[np.floating], groups: tuple[tuple[int, ...], ...]
) -> NDArray[np.floating]:
    return np.asarray([values[list(indices)].sum() for indices in groups])


def _aggregate_matrix(
    values: NDArray[np.floating],
    row_groups: tuple[tuple[int, ...], ...],
    column_groups: tuple[tuple[int, ...], ...],
) -> NDArray[np.floating]:
    return np.asarray(
        [
            [values[np.ix_(row_indices, column_indices)].sum() for column_indices in column_groups]
            for row_indices in row_groups
        ]
    )


def _assert_conserved(
    source: NDArray[np.floating],
    aggregated: NDArray[np.floating] | None,
    *,
    atol: float,
    rtol: float,
) -> None:
    if aggregated is None:
        raise ValueError("aggregation does not conserve the effect total: aggregated value is missing")
    source_values = np.asarray(source, dtype=np.float64)
    aggregated_values = np.asarray(aggregated, dtype=np.float64)
    source_total = float(source_values.sum(dtype=np.float64))
    aggregated_total = float(aggregated_values.sum(dtype=np.float64))
    residual = abs(source_total - aggregated_total)
    # Aggregation changes summation order. For signed effects with strong
    # cancellation, net-relative tolerance is not a meaningful error bound;
    # use the gross magnitude while retaining the explicit absolute floor.
    gross_scale = float(np.abs(source_values).sum(dtype=np.float64))
    tolerance = float(atol + rtol * gross_scale)
    if not np.isfinite(residual) or residual > tolerance:
        raise ValueError(
            "aggregation does not conserve the effect total: "
            f"source_total={source_total:.17g}, aggregated_total={aggregated_total:.17g}, "
            f"abs_residual={residual:.6g}, tolerance={tolerance:.6g}, "
            f"gross_scale={gross_scale:.6g}"
        )


def _duplicates(labels: tuple[Any, ...]) -> tuple[Any, ...]:
    seen: set[Any] = set()
    duplicates: list[Any] = []
    for label in labels:
        if label in seen and label not in duplicates:
            duplicates.append(label)
        seen.add(label)
    return tuple(duplicates)
