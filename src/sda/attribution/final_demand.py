"""Consumer-preserving propagation for final-demand contribution matrices."""

from __future__ import annotations

import numpy as np
from collections.abc import Callable, Mapping
from typing import Any
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validated_inputs(
    component: ArrayLike,
    *,
    v0: ArrayLike,
    delta_v: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    sectors_per_country: int,
) -> tuple[FloatArray, FloatArray]:
    """Validate inputs and return the component and refined ``v/L`` kernel."""

    demand = np.asarray(component, dtype=np.float64)
    baseline_v = np.asarray(v0, dtype=np.float64)
    change_v = np.asarray(delta_v, dtype=np.float64)
    baseline_L = np.asarray(L0, dtype=np.float64)
    current_L = np.asarray(L1, dtype=np.float64)

    if demand.ndim not in (2, 3):
        raise ValueError("final_demand_component must have an optional leading sample axis")
    batched = demand.ndim == 3
    samples = demand.shape[0] if batched else None
    n, consumers = demand.shape[-2:]
    if sectors_per_country <= 0 or n % sectors_per_country:
        raise ValueError("sectors_per_country must be positive and divide the supplier axis")
    if consumers != n // sectors_per_country:
        raise ValueError("consumer count must equal the number of supplier countries")
    matrix_shape = (samples, n, n) if batched else (n, n)
    vector_shape = (samples, n) if batched else (n,)
    if baseline_L.shape != matrix_shape or current_L.shape != matrix_shape:
        raise ValueError("L0 and L1 must match the supplier axis")
    if baseline_v.shape != vector_shape or change_v.shape != vector_shape:
        raise ValueError("v0 and delta_v must match the supplier axis")
    if not all(
        np.isfinite(value).all()
        for value in (demand, baseline_v, change_v, baseline_L, current_L)
    ):
        raise ValueError("attribution inputs must contain only finite values")

    delta_L = current_L - baseline_L
    kernel = (
        baseline_v[..., :, None] * baseline_L
        + 0.5 * baseline_v[..., :, None] * delta_L
        + 0.5 * change_v[..., :, None] * baseline_L
        + (1.0 / 3.0) * change_v[..., :, None] * delta_L
    )
    return demand, kernel


def attribute_final_demand_composition(
    final_demand_component: ArrayLike,
    *,
    v0: ArrayLike,
    delta_v: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    sectors_per_country: int,
) -> FloatArray:
    """Attribute ``D_YF`` or ``D_YB`` to consumer-country/demanded-sector.

    ``final_demand_component[i, c]`` is the already decomposed ``YF`` or
    ``YB`` contribution for supplier country-sector ``i`` and consumer
    country ``c``. Supplier rows are country-major and sector-minor. The
    returned columns are consumer-major and demanded-sector-minor; source
    countries are summed, but their propagation paths to emitters are not.

    This replaces the legacy early sum over consumer countries. It implements
    the same refined-Laspeyres propagation kernel

    ``V0 L0 + 1/2 V0 dL + 1/2 dV L0 + 1/3 dV dL``

    while retaining the determinant's canonical driver axis.
    """

    demand, kernel = _validated_inputs(
        final_demand_component,
        v0=v0,
        delta_v=delta_v,
        L0=L0,
        L1=L1,
        sectors_per_country=sectors_per_country,
    )
    n, consumers = demand.shape[-2:]
    source_countries = n // sectors_per_country
    by_sector_consumer = np.einsum(
        "...eqs,...qsc->...esc",
        kernel.reshape(*kernel.shape[:-2], n, source_countries, sectors_per_country),
        demand.reshape(*demand.shape[:-2], source_countries, sectors_per_country, consumers),
        optimize=True,
    )
    return by_sector_consumer.swapaxes(-2, -1).reshape(
        *demand.shape[:-2], n, consumers * sectors_per_country
    )


def _cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("GPU final-demand attribution requires CuPy") from exc
    try:
        cp.cuda.runtime.getDevice()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("GPU final-demand attribution requires an accessible CUDA device") from exc
    return cp


def attribute_final_demand_compositions(
    components: Mapping[Any, ArrayLike], *, v0: ArrayLike, delta_v: ArrayLike,
    L0: ArrayLike, L1: ArrayLike, sectors_per_country: int,
    backend: str = "numpy", batch_size: int | None = None,
    result_transform: Callable[[Any, FloatArray], ArrayLike] | None = None,
) -> dict[Any, FloatArray]:
    """Attribute many final-demand components in bounded CPU/GPU batches.

    The CuPy path uses FP32, matching the upstream campaign matrices, and
    transfers each completed batch to NumPy FP64 at the public boundary.
    """

    if not components:
        raise ValueError("at least one final-demand component is required")
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be 'numpy' or 'cupy'")
    keys = list(components)
    first, kernel = _validated_inputs(
        components[keys[0]], v0=v0, delta_v=delta_v, L0=L0, L1=L1,
        sectors_per_country=sectors_per_country,
    )
    if first.ndim != 2:
        raise ValueError("batched source components are not supported")
    demands = [np.asarray(components[key], dtype=np.float64) for key in keys]
    if any(value.shape != first.shape or not np.isfinite(value).all() for value in demands):
        raise ValueError("all final-demand components must be finite and share one shape")

    n, consumers = first.shape
    countries = n // sectors_per_country
    xp = np if backend == "numpy" else _cupy()
    dtype = np.float64 if backend == "numpy" else xp.float32
    device_kernel = xp.asarray(kernel, dtype=dtype).reshape(n, countries, sectors_per_country)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_size is None:
        if backend == "numpy":
            size = 1
        else:
            free_bytes, _ = xp.cuda.runtime.memGetInfo()
            bytes_per_component = 4 * n * n * np.dtype(np.float32).itemsize
            size = max(1, int((0.35 * free_bytes) // bytes_per_component))
    else:
        size = batch_size

    result: dict[Any, FloatArray] = {}
    for start in range(0, len(keys), size):
        stop = min(start + size, len(keys))
        batch = xp.stack([xp.asarray(value, dtype=dtype) for value in demands[start:stop]])
        batch = batch.reshape(stop - start, countries, sectors_per_country, consumers)
        attributed = xp.einsum("eqs,bqsc->becs", device_kernel, batch, optimize=True)
        attributed = attributed.reshape(
            stop - start, n, consumers * sectors_per_country
        )
        host = np.asarray(
            attributed if backend == "numpy" else xp.asnumpy(attributed), dtype=np.float64
        )
        for offset, key in enumerate(keys[start:stop]):
            value = host[offset]
            if result_transform is not None:
                value = np.asarray(result_transform(key, value), dtype=np.float64)
            result[key] = value
        del batch, attributed, host
    return result


def attribute_final_demand_level(
    final_demand_component: ArrayLike,
    *,
    v0: ArrayLike,
    delta_v: ArrayLike,
    L0: ArrayLike,
    L1: ArrayLike,
    sectors_per_country: int,
) -> FloatArray:
    """Attribute ``D_y`` to consumer country, with no artificial sector axis.

    The input has the same supplier-by-consumer layout as the composition
    components because sourcing and composition weights distribute each
    country's level change before propagation. Those supplier rows are
    summed only after their distinct paths through the Leontief system have
    been retained. The result is emitter-by-consumer-country.
    """

    demand, kernel = _validated_inputs(
        final_demand_component,
        v0=v0,
        delta_v=delta_v,
        L0=L0,
        L1=L1,
        sectors_per_country=sectors_per_country,
    )
    return np.einsum("...ei,...ic->...ec", kernel, demand)
