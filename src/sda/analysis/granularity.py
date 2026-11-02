"""Controlled aggregation and descriptive granularity diagnostics.

All concentration and ranking diagnostics use absolute magnitudes explicitly;
signed totals are retained separately.  The functions make no empirical or
causal claims and do not infer concordances.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AggregatedMatrix:
    values: FloatArray
    row_labels: tuple[Hashable, ...]
    column_labels: tuple[Hashable, ...]


@dataclass(frozen=True)
class GranularityDiagnostics:
    fine_signed_total: float
    coarse_signed_total: float
    signed_conservation_error: float
    fine_absolute_total: float
    coarse_absolute_total: float
    cancellation_loss: float
    sign_reversal_rate: float
    magnitude_weighted_sign_reversal_rate: float
    fine_hhi: float
    coarse_hhi: float
    hhi_change: float
    fine_entropy: float
    coarse_entropy: float
    entropy_change: float
    sparse_edge_loss_rate: float


def _matrix(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    if not np.isfinite(result).all():
        raise ValueError("values must contain only finite numbers")
    return result


def _codes(labels: Sequence[Hashable], expected: int, name: str) -> tuple[tuple[Hashable, ...], NDArray[np.int64]]:
    if len(labels) != expected:
        raise ValueError(f"{name} must match the corresponding matrix dimension")
    unique: list[Hashable] = []
    lookup: dict[Hashable, int] = {}
    codes = np.empty(expected, dtype=np.int64)
    missing = object()
    for index, label in enumerate(labels):
        try:
            code = lookup.get(label, missing)
        except TypeError as exc:
            raise ValueError(f"{name} must contain hashable labels") from exc
        if code is missing:
            code = len(unique)
            try:
                lookup[label] = code
            except TypeError as exc:
                raise ValueError(f"{name} must contain hashable labels") from exc
            unique.append(label)
        codes[index] = int(code)
    return tuple(unique), codes


def controlled_aggregate(
    values: ArrayLike,
    row_groups: Sequence[Hashable],
    column_groups: Sequence[Hashable],
) -> AggregatedMatrix:
    """Sum a signed matrix using explicit row and column concordances."""

    fine = _matrix(values)
    rows, row_codes = _codes(row_groups, fine.shape[0], "row_groups")
    columns, column_codes = _codes(column_groups, fine.shape[1], "column_groups")
    coarse = np.zeros((len(rows), len(columns)), dtype=np.float64)
    np.add.at(coarse, (row_codes[:, None], column_codes[None, :]), fine)
    return AggregatedMatrix(coarse, rows, columns)


def concentration(values: ArrayLike) -> tuple[float, float]:
    """Return HHI and Shannon entropy of absolute cell magnitudes."""

    magnitudes = np.abs(np.asarray(values, dtype=np.float64)).ravel()
    if not np.isfinite(magnitudes).all():
        raise ValueError("values must contain only finite numbers")
    total = float(magnitudes.sum())
    if total == 0.0:
        return 0.0, 0.0
    shares = magnitudes[magnitudes > 0.0] / total
    return float(np.square(shares).sum()), float(-(shares * np.log(shares)).sum())


def _average_ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_rank_correlation(left: ArrayLike, right: ArrayLike) -> float:
    """Spearman correlation with average ranks; return 0 for constants."""

    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape or x.size == 0:
        raise ValueError("left and right must be non-empty and equally sized")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("rank inputs must contain only finite numbers")
    rx, ry = _average_ranks(x), _average_ranks(y)
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def top_k_overlap(left: ArrayLike, right: ArrayLike, k: int) -> float:
    """Return set overlap among the top-k absolute magnitudes."""

    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape or x.size == 0:
        raise ValueError("left and right must be non-empty and equally sized")
    if not 1 <= k <= x.size:
        raise ValueError("k must be between one and the number of entries")
    left_top = set(np.argsort(np.abs(x), kind="stable")[-k:])
    right_top = set(np.argsort(np.abs(y), kind="stable")[-k:])
    return len(left_top & right_top) / k


def diagnose_aggregation(
    values: ArrayLike,
    row_groups: Sequence[Hashable],
    column_groups: Sequence[Hashable],
) -> GranularityDiagnostics:
    """Compare a fine signed matrix with its controlled aggregation.

    A sign reversal is a non-zero fine cell whose sign opposes its containing
    non-zero coarse cell. Sparse edge loss is the share of non-zero fine cells
    assigned to a coarse cell that cancels exactly to zero.
    """

    fine = _matrix(values)
    rows, row_codes = _codes(row_groups, fine.shape[0], "row_groups")
    columns, column_codes = _codes(column_groups, fine.shape[1], "column_groups")
    aggregated = controlled_aggregate(fine, row_groups, column_groups).values
    expanded = aggregated[row_codes[:, None], column_codes[None, :]]

    nonzero = fine != 0.0
    reversals = nonzero & (expanded != 0.0) & (np.sign(fine) != np.sign(expanded))
    lost = nonzero & (expanded == 0.0)
    count = int(nonzero.sum())
    absolute = np.abs(fine)
    absolute_total = float(absolute.sum())
    fine_hhi, fine_entropy = concentration(fine)
    coarse_hhi, coarse_entropy = concentration(aggregated)
    coarse_absolute = float(np.abs(aggregated).sum())
    cancellation_loss = 0.0 if absolute_total == 0.0 else 1.0 - coarse_absolute / absolute_total
    return GranularityDiagnostics(
        fine_signed_total=float(fine.sum()),
        coarse_signed_total=float(aggregated.sum()),
        signed_conservation_error=float(aggregated.sum() - fine.sum()),
        fine_absolute_total=absolute_total,
        coarse_absolute_total=coarse_absolute,
        cancellation_loss=cancellation_loss,
        sign_reversal_rate=0.0 if count == 0 else float(reversals.sum() / count),
        magnitude_weighted_sign_reversal_rate=(
            0.0 if absolute_total == 0.0 else float(absolute[reversals].sum() / absolute_total)
        ),
        fine_hhi=fine_hhi,
        coarse_hhi=coarse_hhi,
        hhi_change=coarse_hhi - fine_hhi,
        fine_entropy=fine_entropy,
        coarse_entropy=coarse_entropy,
        entropy_change=coarse_entropy - fine_entropy,
        sparse_edge_loss_rate=0.0 if count == 0 else float(lost.sum() / count),
    )
