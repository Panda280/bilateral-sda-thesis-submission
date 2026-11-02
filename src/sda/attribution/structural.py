"""Driver-preserving propagation for structural contribution matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def attribute_structural_driver(
    structural_component: ArrayLike,
    *,
    v0: ArrayLike,
    delta_v: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    y0: ArrayLike,
    delta_y: ArrayLike,
) -> FloatArray:
    """Return emitter-by-structural-driver attribution without per-driver runs.

    This is the driver-preserving equivalent of the reference implementation's
    refined-Laspeyres four-term ``v/L/y`` sandwich. The column ``k`` of the
    structural component remains column ``k`` of the result. Summing result
    columns reproduces the emitter total obtained by summing the reference
    emitter-by-final-demand-endpoint matrix.
    """

    component = np.asarray(structural_component, dtype=np.float64)
    left_L = np.asarray(L0, dtype=np.float64)
    right_L = np.asarray(L1, dtype=np.float64)
    baseline_v = np.asarray(v0, dtype=np.float64)
    change_v = np.asarray(delta_v, dtype=np.float64)
    baseline_y = np.asarray(y0, dtype=np.float64)
    change_y = np.asarray(delta_y, dtype=np.float64)

    if component.ndim not in (2, 3) or component.shape[-2] != component.shape[-1]:
        raise ValueError("structural_component must be square with an optional sample axis")
    batched = component.ndim == 3
    samples = component.shape[0] if batched else None
    n = component.shape[-1]
    matrix_shape = (samples, n, n) if batched else (n, n)
    vector_shape = (samples, n) if batched else (n,)
    if left_L.shape != matrix_shape or right_L.shape != matrix_shape:
        raise ValueError("L0 and L1 must match the structural component shape")
    if any(vector.shape != vector_shape for vector in (baseline_v, change_v, baseline_y, change_y)):
        raise ValueError("v0, delta_v, y0 and delta_y must be vectors matching the matrix size")
    if any(
        not value.all()
        for value in (
            np.isfinite(component), np.isfinite(left_L), np.isfinite(right_L),
            np.isfinite(baseline_v), np.isfinite(change_v),
            np.isfinite(baseline_y), np.isfinite(change_y),
        )
    ):
        raise ValueError("attribution inputs must contain only finite values")

    baseline_left = baseline_v[..., :, None] * (left_L @ component)
    change_left = change_v[..., :, None] * (left_L @ component)
    baseline_requirement = (right_L @ baseline_y[..., None])[..., 0]
    change_requirement = (right_L @ change_y[..., None])[..., 0]

    return (
        baseline_left * baseline_requirement[..., None, :]
        + 0.5 * baseline_left * change_requirement[..., None, :]
        + 0.5 * change_left * baseline_requirement[..., None, :]
        + (1.0 / 3.0) * change_left * change_requirement[..., None, :]
    )
