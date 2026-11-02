"""Optional batched acceleration for repeated structural attribution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on operator environment
        raise RuntimeError(
            "GPU attribution requires a CuPy build matching the installed CUDA runtime"
        ) from exc
    try:
        cp.cuda.runtime.getDevice()
    except Exception as exc:  # pragma: no cover - depends on operator environment
        raise RuntimeError("GPU attribution requires an accessible CUDA device") from exc
    return cp


def _validate(
    components: Mapping[Any, ArrayLike],
    v0: ArrayLike,
    delta_v: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    y0: ArrayLike,
    delta_y: ArrayLike,
) -> tuple[list[Any], list[np.ndarray], tuple[np.ndarray, ...]]:
    if not components:
        raise ValueError("at least one structural component is required")
    keys = list(components)
    matrices = [np.asarray(components[key]) for key in keys]
    n = matrices[0].shape[0] if matrices[0].ndim == 2 else -1
    if n < 1 or any(matrix.shape != (n, n) for matrix in matrices):
        raise ValueError("every structural component must share one square shape")
    arrays = tuple(np.asarray(value) for value in (v0, delta_v, L0, L1, y0, delta_y))
    if arrays[2].shape != (n, n) or arrays[3].shape != (n, n):
        raise ValueError("L0 and L1 must match the component shape")
    if any(value.shape != (n,) for value in (arrays[0], arrays[1], arrays[4], arrays[5])):
        raise ValueError("v0, delta_v, y0 and delta_y must match the component size")
    if any(not np.isfinite(value).all() for value in (*matrices, *arrays)):
        raise ValueError("attribution inputs must contain only finite values")
    return keys, matrices, arrays


def _automatic_gpu_batch_size(cp: Any, n: int, requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("batch_size must be positive")
        return requested
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    # Component, propagated component, and two weighted output work arrays,
    # plus headroom for allocator/workspace overhead. Never consume over 50%.
    conservative_bytes_per_component = 5 * n * n * np.dtype(np.float32).itemsize
    return max(1, int((0.5 * free_bytes) // conservative_bytes_per_component))


def attribute_structural_components(
    components: Mapping[Any, ArrayLike],
    *,
    v0: ArrayLike,
    delta_v: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    y0: ArrayLike,
    delta_y: ArrayLike,
    backend: str = "numpy",
    batch_size: int | None = None,
    result_transform: Callable[[Any, FloatArray], ArrayLike] | None = None,
) -> dict[Any, FloatArray]:
    """Attribute many independent structural components in bounded batches.

    The NumPy backend is the float64 reference. The opt-in CuPy backend uses
    float32, matching the precision of the upstream EXIODeflator matrices, and
    converts each completed batch back to NumPy float64 at the public boundary.
    """

    keys, matrices, arrays = _validate(components, v0, delta_v, L0, L1, y0, delta_y)
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be 'numpy' or 'cupy'")
    xp = np if backend == "numpy" else _cupy()
    dtype = np.float64 if backend == "numpy" else xp.float32
    base_v, change_v, left_l, right_l, base_y, change_y = (
        xp.asarray(value, dtype=dtype) for value in arrays
    )
    base_requirement = (right_l @ base_y[:, None])[:, 0]
    change_requirement = (right_l @ change_y[:, None])[:, 0]
    # Sequential is the conservative CPU default: a full R3 stack can exceed
    # host memory. GPU batches are sized from currently free device memory.
    size = 1 if backend == "numpy" else _automatic_gpu_batch_size(xp, len(base_v), batch_size)
    if backend == "numpy" and batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        size = batch_size

    result: dict[Any, FloatArray] = {}
    for start in range(0, len(keys), size):
        stop = min(start + size, len(keys))
        batch = xp.stack([xp.asarray(value, dtype=dtype) for value in matrices[start:stop]])
        propagated = xp.matmul(left_l[None, :, :], batch)
        baseline_left = base_v[None, :, None] * propagated
        change_left = change_v[None, :, None] * propagated
        attributed = (
            baseline_left * base_requirement[None, None, :]
            + 0.5 * baseline_left * change_requirement[None, None, :]
            + 0.5 * change_left * base_requirement[None, None, :]
            + (1.0 / 3.0) * change_left * change_requirement[None, None, :]
        )
        host = np.asarray(attributed if backend == "numpy" else xp.asnumpy(attributed), dtype=np.float64)
        for offset, key in enumerate(keys[start:stop]):
            value = host[offset]
            if result_transform is not None:
                value = np.asarray(result_transform(key, value), dtype=np.float64)
            result[key] = value
    return result
