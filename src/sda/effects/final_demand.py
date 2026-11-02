"""Algebraic final-demand components before emissions propagation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FinalDemandComponents:
    """Three-factor contributions retaining supplier and consumer axes."""

    YF: FloatArray
    YB: FloatArray
    y: FloatArray


def three_factor_final_demand_components(
    F0: ArrayLike,
    F1: ArrayLike,
    B0: ArrayLike,
    B1: ArrayLike,
    y0: ArrayLike,
    y1: ArrayLike,
    supplier_sector: ArrayLike,
) -> FinalDemandComponents:
    """Split ``Y[p,c] = F[p,c] * B[j(p),c] * y[c]`` by Shapley averaging.

    ``p`` is a supplier country-sector, ``c`` is a consumer country, and the
    explicit integer mapping ``supplier_sector[p]`` identifies demanded sector
    ``j``.  All returned arrays have shape supplier-by-consumer and reconstruct
    ``Y1 - Y0`` cellwise.  No supplier or consumer dimension is contracted.
    """

    F0_array = np.asarray(F0, dtype=np.float64)
    F1_array = np.asarray(F1, dtype=np.float64)
    B0_array = np.asarray(B0, dtype=np.float64)
    B1_array = np.asarray(B1, dtype=np.float64)
    y0_array = np.asarray(y0, dtype=np.float64)
    y1_array = np.asarray(y1, dtype=np.float64)
    sector = np.asarray(supplier_sector)

    if F0_array.ndim != 2 or F1_array.shape != F0_array.shape:
        raise ValueError("F0 and F1 must share one supplier-by-consumer shape")
    supplier_count, consumer_count = F0_array.shape
    if B0_array.ndim != 2 or B1_array.shape != B0_array.shape:
        raise ValueError("B0 and B1 must share one sector-by-consumer shape")
    if B0_array.shape[1] != consumer_count:
        raise ValueError("F and B must have the same consumer count")
    if y0_array.shape != (consumer_count,) or y1_array.shape != (consumer_count,):
        raise ValueError("y0 and y1 must be vectors matching the consumer count")
    if sector.shape != (supplier_count,) or not np.issubdtype(sector.dtype, np.integer):
        raise ValueError("supplier_sector must be an integer vector matching suppliers")
    if np.any(sector < 0) or np.any(sector >= B0_array.shape[0]):
        raise ValueError("supplier_sector contains an index outside the B sector axis")
    if any(
        not np.isfinite(value).all()
        for value in (F0_array, F1_array, B0_array, B1_array, y0_array, y1_array)
    ):
        raise ValueError("final-demand inputs must contain only finite values")

    states = (
        (F0_array, F1_array),
        (B0_array[sector, :], B1_array[sector, :]),
        (y0_array[None, :], y1_array[None, :]),
    )
    contributions = [np.zeros_like(F0_array) for _ in range(3)]
    for order in permutations(range(3)):
        current = [states[index][0] for index in range(3)]
        for changed in order:
            before = current[0] * current[1] * current[2]
            current[changed] = states[changed][1]
            after = current[0] * current[1] * current[2]
            contributions[changed] += (after - before) / 6.0

    return FinalDemandComponents(
        YF=contributions[0],
        YB=contributions[1],
        y=contributions[2],
    )


def split_yf_component_by_source(
    yf_component: ArrayLike,
    masks: Mapping[str, ArrayLike],
) -> dict[str, FloatArray]:
    """Partition a pre-propagation ``YF`` component by source origin.

    Rows identify source country-sectors and columns identify consumer-country
    demanded sectors. The masks must describe the same cells and form a strict
    mutually exclusive, exhaustive partition. No emissions propagation or
    aggregation is performed here.
    """

    component = np.asarray(yf_component, dtype=np.float64)
    if component.ndim != 2 or not all(component.shape):
        raise ValueError("yf_component must be a non-empty two-dimensional matrix")
    if not np.isfinite(component).all():
        raise ValueError("yf_component must contain only finite values")
    if not masks:
        raise ValueError("at least one source-origin mask is required")

    mask_arrays = {name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}
    if any(mask.shape != component.shape for mask in mask_arrays.values()):
        raise ValueError("every source-origin mask must match the YF component shape")
    mask_total = np.sum(np.stack(tuple(mask_arrays.values())), axis=0)
    if not np.array_equal(mask_total, np.ones(component.shape, dtype=np.int64)):
        raise ValueError("source-origin masks must be mutually exclusive and exhaustive")

    return {name: component * mask for name, mask in mask_arrays.items()}
