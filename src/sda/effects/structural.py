"""Algebraic structural components before emissions propagation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StructuralComponents:
    """Canonical two-factor contributions to the change in A."""

    C: FloatArray
    A_x: FloatArray


@dataclass(frozen=True)
class NestedTechnicalComponents:
    """Nested contributions that reconstruct the canonical A_x component."""

    T_coef: FloatArray
    w_coef: FloatArray


def two_factor_structural_components(
    C0: ArrayLike,
    C1: ArrayLike,
    A_x0: ArrayLike,
    A_x1: ArrayLike,
) -> StructuralComponents:
    """Split ``A = C * A_x`` with the symmetric two-factor convention.

    Multiplication is elementwise. The returned matrices satisfy
    ``C_component + A_x_component == C1*A_x1 - C0*A_x0`` up to floating-point
    tolerance. Inputs are converted to float64 and must share one square shape.
    """

    arrays = [np.asarray(value, dtype=np.float64) for value in (C0, C1, A_x0, A_x1)]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or any(value.shape != shape for value in arrays):
        raise ValueError("C0, C1, A_x0 and A_x1 must share one square two-dimensional shape")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("structural inputs must contain only finite values")

    C0_array, C1_array, A_x0_array, A_x1_array = arrays
    delta_C = C1_array - C0_array
    delta_A_x = A_x1_array - A_x0_array
    interaction = 0.5 * delta_C * delta_A_x
    return StructuralComponents(
        C=delta_C * A_x0_array + interaction,
        A_x=C0_array * delta_A_x + interaction,
    )


def split_c_component_by_source(
    C0: ArrayLike,
    C1: ArrayLike,
    A_x0: ArrayLike,
    A_x1: ArrayLike,
    masks: Mapping[str, ArrayLike],
) -> dict[str, FloatArray]:
    """Apply source-origin masks to ``delta C`` before the two-factor split."""

    arrays = [np.asarray(value, dtype=np.float64) for value in (C0, C1, A_x0, A_x1)]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or any(value.shape != shape for value in arrays):
        raise ValueError("C0, C1, A_x0 and A_x1 must share one square two-dimensional shape")
    if not masks:
        raise ValueError("at least one source-origin mask is required")

    mask_arrays = {name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}
    if any(mask.shape != shape for mask in mask_arrays.values()):
        raise ValueError("every source-origin mask must match the structural matrix shape")
    mask_total = np.sum(np.stack(tuple(mask_arrays.values())), axis=0)
    if not np.array_equal(mask_total, np.ones(shape, dtype=np.int64)):
        raise ValueError("source-origin masks must be mutually exclusive and exhaustive")

    C0_array, C1_array, A_x0_array, A_x1_array = arrays
    delta_C = C1_array - C0_array
    delta_A_x = A_x1_array - A_x0_array
    return {
        name: (delta_C * mask) * A_x0_array + 0.5 * (delta_C * mask) * delta_A_x
        for name, mask in mask_arrays.items()
    }


def nested_technical_components(
    C0: ArrayLike,
    C1: ArrayLike,
    T0: ArrayLike,
    T1: ArrayLike,
    w0: ArrayLike,
    w1: ArrayLike,
) -> NestedTechnicalComponents:
    """Split canonical ``D_A_x`` using symmetric averaging within ``A_x=T*w``.

    The top-level ``C/A_x`` decomposition is fixed first. Inside its ``A_x``
    parent, the two possible T/w replacement orders are averaged, exactly as
    the final-demand implementation averages factor replacement orders. This
    allocates the triple interaction as 1/2 to C and 1/4 to each nested child.
    """

    C0_array = np.asarray(C0, dtype=np.float64)
    C1_array = np.asarray(C1, dtype=np.float64)
    T0_array = np.asarray(T0, dtype=np.float64)
    T1_array = np.asarray(T1, dtype=np.float64)
    w0_array = np.asarray(w0, dtype=np.float64)
    w1_array = np.asarray(w1, dtype=np.float64)
    shape = C0_array.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("C0 must be a square two-dimensional matrix")
    if any(value.shape != shape for value in (C1_array, T0_array, T1_array)):
        raise ValueError("C0, C1, T0 and T1 must share one square shape")
    if w0_array.shape != (shape[1],) or w1_array.shape != (shape[1],):
        raise ValueError("w0 and w1 must be vectors matching the structural columns")
    if any(not np.isfinite(value).all() for value in (C0_array, C1_array, T0_array, T1_array, w0_array, w1_array)):
        raise ValueError("nested structural inputs must contain only finite values")

    delta_C = C1_array - C0_array
    delta_T = T1_array - T0_array
    delta_w = w1_array - w0_array
    average_C = C0_array + 0.5 * delta_C
    interaction = 0.5 * delta_T * delta_w[None, :]
    return NestedTechnicalComponents(
        T_coef=average_C * (delta_T * w0_array[None, :] + interaction),
        w_coef=average_C * (T0_array * delta_w[None, :] + interaction),
    )
