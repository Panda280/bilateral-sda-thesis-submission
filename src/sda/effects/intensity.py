"""Emitter-specific emissions-intensity effect."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def refined_laspeyres_intensity_effect(
    v0: ArrayLike,
    v1: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    y0: ArrayLike,
    y1: ArrayLike,
) -> FloatArray:
    """Return ``D_v`` as one value per emitter country-sector.

    This is the intensity-attributable term of the three-factor product
    ``diag(v) @ L @ y`` under the refined-Laspeyres convention. The result has
    no driver axis: a change in intensity belongs to the emitter whose
    intensity changed.
    """

    baseline_v = np.asarray(v0, dtype=np.float64)
    current_v = np.asarray(v1, dtype=np.float64)
    baseline_L = np.asarray(L0, dtype=np.float64)
    current_L = np.asarray(L1, dtype=np.float64)
    baseline_y = np.asarray(y0, dtype=np.float64)
    current_y = np.asarray(y1, dtype=np.float64)

    if baseline_L.ndim not in (2, 3) or baseline_L.shape[-2] != baseline_L.shape[-1]:
        raise ValueError("L0 must be square with an optional sample axis")
    batched = baseline_L.ndim == 3
    samples = baseline_L.shape[0] if batched else None
    n = baseline_L.shape[-1]
    matrix_shape = (samples, n, n) if batched else (n, n)
    vector_shape = (samples, n) if batched else (n,)
    if current_L.shape != matrix_shape:
        raise ValueError("L1 must match the L0 shape")
    if any(vector.shape != vector_shape for vector in (baseline_v, current_v, baseline_y, current_y)):
        raise ValueError("v0, v1, y0 and y1 must be vectors matching the matrix size")
    if any(
        not np.isfinite(value).all()
        for value in (baseline_v, current_v, baseline_L, current_L, baseline_y, current_y)
    ):
        raise ValueError("intensity-effect inputs must contain only finite values")

    delta_v = current_v - baseline_v
    delta_L = current_L - baseline_L
    delta_y = current_y - baseline_y
    requirement_0 = (baseline_L @ baseline_y[..., None])[..., 0]
    requirement_delta_L = (delta_L @ baseline_y[..., None])[..., 0]
    requirement_delta_y = (baseline_L @ delta_y[..., None])[..., 0]
    requirement_joint = (delta_L @ delta_y[..., None])[..., 0]

    return delta_v * (
        requirement_0
        + 0.5 * requirement_delta_L
        + 0.5 * requirement_delta_y
        + (1.0 / 3.0) * requirement_joint
    )
