"""Bounded-memory coordination for sampled canonical SDA runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .hierarchy import CANONICAL_EFFECTS
from .model import EffectDimensions, SDAResult
from .output.statistics import StreamingStatistics
from .planning import ExecutionPlan


SampleChunkCallback = Callable[[int, int], Iterable[SDAResult]]


@dataclass(frozen=True)
class SampledSDAStatistics:
    """Statistics-only output from a complete sampled execution plan."""

    effects: Mapping[str, StreamingStatistics]
    total_change: StreamingStatistics
    dimensions: Mapping[str, EffectDimensions]
    plan: ExecutionPlan

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "samples": self.total_change.count,
            "plan": self.plan.metadata,
            "effects": {name: stats.metadata for name, stats in self.effects.items()},
            "total_change": self.total_change.metadata,
        }


def run_chunked_samples(plan: ExecutionPlan, compute_chunk: SampleChunkCallback) -> SampledSDAStatistics:
    """Execute canonical samples in planned chunks and retain statistics only.

    ``compute_chunk(start, stop)`` may compute each sample through
    :func:`run_canonical_sda` or another boundary that returns fully validated
    :class:`SDAResult` objects. The coordinator never changes the single-run
    implementation and never retains prior sample results.
    """

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    if not callable(compute_chunk):
        raise TypeError("compute_chunk must be callable")

    effect_stats: dict[str, StreamingStatistics] = {}
    dimensions: dict[str, EffectDimensions] = {}
    total_stats: StreamingStatistics | None = None
    processed = 0

    for start in range(0, plan.samples, plan.chunk_size):
        stop = min(start + plan.chunk_size, plan.samples)
        chunk_count = 0
        try:
            results = compute_chunk(start, stop)
            for result in results:
                if chunk_count >= stop - start:
                    raise ValueError(f"chunk [{start}, {stop}) returned too many samples")
                _validate_and_accumulate(
                    result, plan=plan, effect_stats=effect_stats,
                    dimensions=dimensions, total_stats=total_stats,
                )
                if total_stats is None:
                    total_stats = StreamingStatistics.empty(result.total_change.shape)
                    total_stats.update(result.total_change)
                else:
                    total_stats.update(result.total_change)
                chunk_count += 1
                processed += 1
        except Exception as exc:
            if isinstance(exc, (TypeError, ValueError)) and str(exc).startswith("chunk"):
                raise
            raise ValueError(f"chunk [{start}, {stop}) failed validation or computation") from exc
        if chunk_count != stop - start:
            raise ValueError(
                f"chunk [{start}, {stop}) returned {chunk_count} samples; expected {stop - start}"
            )

    if processed != plan.samples or total_stats is None:
        raise ValueError(f"sample execution incomplete: processed {processed} of {plan.samples}")
    return SampledSDAStatistics(effect_stats, total_stats, dimensions, plan)


def _validate_and_accumulate(
    result: SDAResult,
    *,
    plan: ExecutionPlan,
    effect_stats: dict[str, StreamingStatistics],
    dimensions: dict[str, EffectDimensions],
    total_stats: StreamingStatistics | None,
) -> None:
    if not isinstance(result, SDAResult):
        raise TypeError("chunk entries must be SDAResult objects")
    result.validation.raise_if_invalid()
    if len(result.total_change) != plan.emitters:
        raise ValueError("result emitter count is inconsistent with execution plan")
    if len(result.effects["D_y"].dimensions.column.labels) != plan.consumers:
        raise ValueError("result consumer count is inconsistent with execution plan")
    if total_stats is not None and result.total_change.shape != total_stats.mean.shape:
        raise ValueError("total-change shape changed between samples")

    for name in CANONICAL_EFFECTS:
        effect = result.effects[name]
        if name not in effect_stats:
            dimensions[name] = effect.dimensions
            effect_stats[name] = StreamingStatistics.empty(effect.baseline.shape)
        elif effect.dimensions != dimensions[name]:
            raise ValueError(f"{name} labels or dimensions changed between samples")
        if effect.baseline.shape != effect_stats[name].mean.shape:
            raise ValueError(f"{name} shape changed between samples")
        effect_stats[name].update(effect.baseline)
