"""Source-origin classifications applied before emissions propagation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

SOURCE_ORIGIN_GROUPS = ("Domestic", "Red", "Non-Red", "Non-Participant")
EXTERNAL_SOURCE_GROUPS = SOURCE_ORIGIN_GROUPS[1:]


def _validated_country_labels(labels: Sequence[str], *, axis: str) -> tuple[str, ...]:
    values = tuple(labels)
    if not values:
        raise ValueError(f"{axis} country labels must be non-empty")
    invalid = [
        label
        for label in values
        if not isinstance(label, str) or not label.strip() or label != label.strip()
    ]
    if invalid:
        raise ValueError(f"{axis} country labels must be non-empty strings")
    return values


def build_per_country_source_masks(
    source_countries: Sequence[str],
    destination_countries: Sequence[str],
    country_order: Sequence[str] | None = None,
) -> dict[str, NDArray[np.bool_]]:
    """Partition source-by-destination cells by their source country.

    Repeated source labels represent sectors within a country.  Every cell
    in those rows belongs to that source country, including cells whose
    destination has the same country label; domestic sourcing is therefore
    folded into the country's own mask rather than emitted as a separate
    category.

    By default keys follow first appearance on the source axis.  An explicit
    ``country_order`` may reorder them, but must contain every observed source
    country exactly once and no unknown labels.
    """

    sources = _validated_country_labels(source_countries, axis="source")
    destinations = _validated_country_labels(destination_countries, axis="destination")

    observed = tuple(dict.fromkeys(sources))
    observed_set = set(observed)
    unknown_destinations = sorted(set(destinations) - observed_set)
    if unknown_destinations:
        raise ValueError(f"unknown destination country labels: {unknown_destinations}")

    if country_order is None:
        ordered = observed
    else:
        ordered = _validated_country_labels(country_order, axis="country-order")
        if len(set(ordered)) != len(ordered):
            raise ValueError("country-order labels must be unique")
        unknown = sorted(set(ordered) - observed_set)
        missing = sorted(observed_set - set(ordered))
        if unknown or missing:
            raise ValueError(
                "country-order labels must match observed source countries; "
                f"unknown={unknown}, missing={missing}"
            )

    source_array = np.asarray(sources, dtype=object)
    masks = {
        country: np.broadcast_to(
            source_array[:, None] == country,
            (len(sources), len(destinations)),
        ).copy()
        for country in ordered
    }
    total = np.sum(np.stack(tuple(masks.values())), axis=0)
    if not np.array_equal(total, np.ones_like(total)):
        raise ValueError("per-country source masks must be mutually exclusive and exhaustive")
    return masks


def build_source_origin_masks(
    source_countries: Sequence[str],
    destination_countries: Sequence[str],
    source_group_by_country: Mapping[str, str],
) -> dict[str, NDArray[np.bool_]]:
    """Partition source-by-destination cells into canonical source groups."""

    sources = np.asarray(tuple(source_countries), dtype=object)
    destinations = np.asarray(tuple(destination_countries), dtype=object)
    if sources.ndim != 1 or destinations.ndim != 1 or not len(sources) or not len(destinations):
        raise ValueError("source and destination country labels must be non-empty one-dimensional sequences")

    missing = sorted({str(country) for country in sources if country not in source_group_by_country})
    if missing:
        raise ValueError(f"missing source-origin classification for countries: {missing}")
    invalid = sorted(
        {str(source_group_by_country[country]) for country in sources
         if source_group_by_country[country] not in EXTERNAL_SOURCE_GROUPS}
    )
    if invalid:
        raise ValueError(f"invalid external source-origin groups: {invalid}")

    domestic = sources[:, None] == destinations[None, :]
    masks: dict[str, NDArray[np.bool_]] = {"Domestic": domestic}
    for group in EXTERNAL_SOURCE_GROUPS:
        source_in_group = np.asarray(
            [source_group_by_country[country] == group for country in sources], dtype=bool
        )
        masks[group] = source_in_group[:, None] & ~domestic

    total = np.sum(np.stack([masks[name] for name in SOURCE_ORIGIN_GROUPS]), axis=0)
    if not np.array_equal(total, np.ones_like(total)):
        raise ValueError("source-origin masks must be mutually exclusive and exhaustive")
    return masks


def build_group_sector_source_masks(
    source_countries: Sequence[str],
    source_sectors: Sequence[str],
    destination_countries: Sequence[str],
    source_group_by_country: Mapping[str, str],
    *,
    sector_order: Sequence[str] | None = None,
) -> dict[tuple[str, str], NDArray[np.bool_]]:
    """Partition structural cells by canonical source group and source sector.

    ``Domestic`` is destination-relative, exactly as in
    :func:`build_source_origin_masks`. External political groups exclude those
    domestic cells. Summing masks over sector within each group therefore
    reproduces the existing four source-origin masks cell for cell.
    """

    sources = _validated_country_labels(source_countries, axis="source")
    destinations = _validated_country_labels(destination_countries, axis="destination")
    sectors = _validated_country_labels(source_sectors, axis="source-sector")
    if len(sectors) != len(sources):
        raise ValueError("source country and sector axes must have equal length")

    observed_sectors = tuple(dict.fromkeys(sectors))
    if sector_order is None:
        ordered_sectors = observed_sectors
    else:
        ordered_sectors = _validated_country_labels(sector_order, axis="sector-order")
        if len(set(ordered_sectors)) != len(ordered_sectors):
            raise ValueError("sector-order labels must be unique")
        if set(ordered_sectors) != set(observed_sectors):
            raise ValueError("sector-order labels must match observed source sectors")

    group_masks = build_source_origin_masks(sources, destinations, source_group_by_country)
    sector_values = np.asarray(sectors, dtype=object)
    masks = {
        (group, sector): group_masks[group] & (sector_values[:, None] == sector)
        for group in SOURCE_ORIGIN_GROUPS
        for sector in ordered_sectors
    }
    total = np.sum(np.stack(tuple(masks.values())), axis=0)
    if not np.array_equal(total, np.ones_like(total)):
        raise ValueError("group-sector masks must be mutually exclusive and exhaustive")
    for group in SOURCE_ORIGIN_GROUPS:
        reconstructed = np.sum(
            np.stack([masks[(group, sector)] for sector in ordered_sectors]), axis=0
        )
        if not np.array_equal(reconstructed, group_masks[group]):
            raise ValueError(f"group-sector masks do not reconstruct {group}")
    return masks
