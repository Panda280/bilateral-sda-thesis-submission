"""Descriptive summaries for the nested ``D_T_coef``/``D_w_coef`` effects.

The functions in this module do not assign an economic or causal meaning to
the two implementation-defined factors. They operate on two reconciled,
same-shaped effect matrices and preserve their identity as children of
``D_A_x``.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TWSummary:
    """One signed, descriptive summary of the nested pair."""

    t_value: float
    w_value: float
    parent_value: float
    t_absolute_share: float
    w_absolute_share: float
    cancellation_ratio: float
    dominance: str


def _arrays(d_t: ArrayLike, d_w: ArrayLike) -> tuple[FloatArray, FloatArray]:
    t = np.asarray(d_t, dtype=np.float64)
    w = np.asarray(d_w, dtype=np.float64)
    if t.shape != w.shape or t.ndim != 2:
        raise ValueError("D_T_coef and D_w_coef must share one two-dimensional shape")
    if not np.isfinite(t).all() or not np.isfinite(w).all():
        raise ValueError("D_T_coef and D_w_coef must contain only finite values")
    return t, w


def summarize_tw(d_t: ArrayLike, d_w: ArrayLike, *, dominance_threshold: float = 0.6) -> TWSummary:
    """Summarize signed totals, absolute shares, and cancellation.

    ``cancellation_ratio`` is ``1 - |T + w| / (|T| + |w|)`` after summing
    each child over the supplied selection. It is zero when both totals are
    zero. Dominance is based on absolute shares and is ``balanced`` unless a
    child reaches ``dominance_threshold``.
    """

    if not 0.5 < dominance_threshold <= 1.0:
        raise ValueError("dominance_threshold must be in (0.5, 1]")
    t, w = _arrays(d_t, d_w)
    t_value, w_value = float(t.sum()), float(w.sum())
    denominator = abs(t_value) + abs(w_value)
    if denominator == 0.0:
        t_share = w_share = cancellation = 0.0
        dominance = "zero"
    else:
        t_share = abs(t_value) / denominator
        w_share = abs(w_value) / denominator
        cancellation = 1.0 - abs(t_value + w_value) / denominator
        if t_share >= dominance_threshold:
            dominance = "T-dominant"
        elif w_share >= dominance_threshold:
            dominance = "w-dominant"
        else:
            dominance = "balanced"
    return TWSummary(
        t_value=t_value,
        w_value=w_value,
        parent_value=t_value + w_value,
        t_absolute_share=t_share,
        w_absolute_share=w_share,
        cancellation_ratio=cancellation,
        dominance=dominance,
    )


def aggregate_tw(
    d_t: ArrayLike,
    d_w: ArrayLike,
    labels: Sequence[Hashable],
    *,
    axis: int,
    dominance_threshold: float = 0.6,
) -> dict[Hashable, TWSummary]:
    """Aggregate the pair by labels on the emitter (0) or driver (1) axis."""

    t, w = _arrays(d_t, d_w)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (emitter) or 1 (structural driver)")
    if len(labels) != t.shape[axis]:
        raise ValueError("labels must match the selected effect axis")
    groups: dict[Hashable, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return {
        label: summarize_tw(
            np.take(t, indexes, axis=axis),
            np.take(w, indexes, axis=axis),
            dominance_threshold=dominance_threshold,
        )
        for label, indexes in groups.items()
    }


def combine_transitions(pairs: Iterable[tuple[ArrayLike, ArrayLike]]) -> tuple[FloatArray, FloatArray]:
    """Add same-shaped transition effects for a cumulative descriptive view."""

    iterator = iter(pairs)
    try:
        first_t, first_w = _arrays(*next(iterator))
    except StopIteration as exc:
        raise ValueError("at least one transition pair is required") from exc
    total_t, total_w = first_t.copy(), first_w.copy()
    for d_t, d_w in iterator:
        t, w = _arrays(d_t, d_w)
        if t.shape != total_t.shape:
            raise ValueError("all transition pairs must share one shape")
        total_t += t
        total_w += w
    return total_t, total_w
