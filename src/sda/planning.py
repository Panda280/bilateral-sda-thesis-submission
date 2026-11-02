"""Pure, conservative memory planning for sampled SDA attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionPlan:
    """Serializable plan for processing a sample axis in bounded chunks."""

    emitters: int
    consumers: int
    samples: int
    dtype: str
    budget_bytes: int
    bytes_per_sample: int
    chunk_size: int
    chunks: int
    estimated_peak_bytes: int

    @property
    def metadata(self) -> dict[str, int | str]:
        """Return plain metadata suitable for a run manifest."""

        return asdict(self)


def plan_sample_execution(
    *,
    emitters: int,
    consumers: int,
    samples: int,
    dtype: np.dtype | str | type = np.float64,
    budget_bytes: int,
) -> ExecutionPlan:
    """Estimate a conservative sample chunk size under ``budget_bytes``.

    The estimate covers the largest current attribution path per sample. It
    reserves space for eight emitter-square arrays (inputs, products and
    output), four emitter vectors, and two emitter-by-consumer arrays. This is
    deliberately a planning bound, not a measurement of allocator behaviour.
    """

    for name, value in (("emitters", emitters), ("consumers", consumers), ("samples", samples)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, (int, np.integer)) or budget_bytes <= 0:
        raise ValueError("budget_bytes must be a positive integer")

    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.kind != "f":
        raise ValueError("dtype must be a floating-point dtype")

    elements_per_sample = 8 * emitters * emitters + 4 * emitters + 2 * emitters * consumers
    bytes_per_sample = elements_per_sample * resolved_dtype.itemsize
    if budget_bytes < bytes_per_sample:
        raise ValueError(
            f"memory budget cannot fit one sample: requires {bytes_per_sample} bytes, "
            f"budget is {budget_bytes} bytes"
        )

    chunk_size = min(samples, budget_bytes // bytes_per_sample)
    chunks = (samples + chunk_size - 1) // chunk_size
    return ExecutionPlan(
        emitters=int(emitters),
        consumers=int(consumers),
        samples=int(samples),
        dtype=resolved_dtype.name,
        budget_bytes=int(budget_bytes),
        bytes_per_sample=int(bytes_per_sample),
        chunk_size=int(chunk_size),
        chunks=int(chunks),
        estimated_peak_bytes=int(chunk_size * bytes_per_sample),
    )
