"""Pure, CPU-only temporal diagnostics for signed SDA contributions.

The functions operate on already validated, long-form observations.  They do
not aggregate across units or effect levels implicitly.  Callers must supply a
stable ``series_id`` for the entity being followed through time (for example,
an emitter country-sector pair).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from statistics import median
from typing import Iterable, Sequence

from ...hierarchy import CANONICAL_EFFECTS

EFFECT_PARENT = {
    "D_v": "emission_intensity",
    "D_C": "leontief_production_structure",
    "D_A_x": "leontief_production_structure",
    "D_T_coef": "D_A_x",
    "D_w_coef": "D_A_x",
    "D_YF": "final_demand",
    "D_YB": "final_demand",
    "D_y": "final_demand",
}


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """One signed effect contribution for one transition and analytical unit."""

    transition: str
    year_previous: int
    year_current: int
    effect_id: str
    series_id: str
    value: float
    units: str
    absolute_activity_share: float | None = None
    robust_anomaly_score: float | None = None
    is_anomaly: bool | None = None

    def __post_init__(self) -> None:
        if self.effect_id not in CANONICAL_EFFECTS:
            raise ValueError(f"unknown canonical effect: {self.effect_id}")
        if self.year_current <= self.year_previous:
            raise ValueError("year_current must be later than year_previous")
        if not self.transition or not self.series_id or not self.units:
            raise ValueError("transition, series_id, and units must be non-empty")
        if not isfinite(self.value):
            raise ValueError("value must be finite")

    @property
    def hierarchy_parent(self) -> str:
        """Canonical immediate parent; no aggregation is performed."""

        return EFFECT_PARENT[self.effect_id]

    @property
    def hierarchy_level(self) -> int:
        """Canonical report depth (top-level effect 2; nested subeffect 3)."""

        return 3 if self.effect_id in {"D_T_coef", "D_w_coef"} else 2


def _group_key(row: TemporalObservation) -> tuple[str, str, str]:
    return row.effect_id, row.series_id, row.units


def _ensure_comparable(rows: Sequence[TemporalObservation]) -> None:
    labels: dict[tuple[int, int], str] = {}
    for row in rows:
        pair = (row.year_previous, row.year_current)
        previous = labels.setdefault(pair, row.transition)
        if previous != row.transition:
            raise ValueError(f"inconsistent transition labels for {pair}")


def with_activity_shares(
    observations: Iterable[TemporalObservation],
) -> list[TemporalObservation]:
    """Attach ``abs(value) / sum(abs(value))`` within effect-transition-unit groups.

    Shares are calculated separately by effect, transition, and units.  A group
    with zero gross activity receives a share of zero for every observation.
    """

    rows = list(observations)
    _ensure_comparable(rows)
    gross: dict[tuple[str, int, int, str], float] = {}
    for row in rows:
        key = (row.effect_id, row.year_previous, row.year_current, row.units)
        gross[key] = gross.get(key, 0.0) + abs(row.value)
    return [
        replace(
            row,
            absolute_activity_share=(
                abs(row.value) / gross[key] if gross[key] else 0.0
            ),
        )
        for row in rows
        for key in [(row.effect_id, row.year_previous, row.year_current, row.units)]
    ]


def anomaly_scores(
    observations: Iterable[TemporalObservation], *, threshold: float = 3.5
) -> list[TemporalObservation]:
    """Attach robust median/MAD scores within each effect-series-unit time series.

    The score is ``0.67448975 * (value - median) / MAD``.  If MAD is zero,
    observations equal to the median receive zero and unequal observations
    receive signed infinity, making isolated departures explicit.
    """

    if threshold <= 0 or not isfinite(threshold):
        raise ValueError("threshold must be finite and positive")
    rows = list(observations)
    _ensure_comparable(rows)
    grouped: dict[tuple[str, str, str], list[TemporalObservation]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    scored: dict[int, TemporalObservation] = {}
    for group in grouped.values():
        centre = median(row.value for row in group)
        mad = median(abs(row.value - centre) for row in group)
        for row in group:
            delta = row.value - centre
            score = 0.67448975 * delta / mad if mad else (0.0 if delta == 0 else (float("inf") if delta > 0 else float("-inf")))
            scored[id(row)] = replace(row, robust_anomaly_score=score, is_anomaly=abs(score) >= threshold)
    return [scored[id(row)] for row in rows]


def materiality_filter(
    observations: Iterable[TemporalObservation], *, min_absolute: float = 0.0,
    min_activity_share: float = 0.0,
) -> list[TemporalObservation]:
    """Retain observations meeting both absolute and gross-activity thresholds."""

    if min_absolute < 0 or min_activity_share < 0 or min_activity_share > 1:
        raise ValueError("materiality thresholds must be non-negative; share must be <= 1")
    rows = list(observations)
    if min_activity_share and any(row.absolute_activity_share is None for row in rows):
        rows = with_activity_shares(rows)
    return [row for row in rows if abs(row.value) >= min_absolute and (row.absolute_activity_share or 0.0) >= min_activity_share]


def find_sign_reversals(
    observations: Iterable[TemporalObservation], *, min_absolute: float = 0.0,
) -> list[tuple[TemporalObservation, TemporalObservation]]:
    """Return adjacent material observations whose non-zero signs reverse."""

    if min_absolute < 0:
        raise ValueError("min_absolute must be non-negative")
    rows = list(observations)
    _ensure_comparable(rows)
    grouped: dict[tuple[str, str, str], list[TemporalObservation]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    reversals = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: (row.year_previous, row.year_current))
        for left, right in zip(ordered, ordered[1:]):
            if right.year_previous != left.year_current:
                continue
            if abs(left.value) < min_absolute or abs(right.value) < min_absolute:
                continue
            if left.value * right.value < 0:
                reversals.append((left, right))
    return reversals


def cancellation_ratio(observations: Iterable[TemporalObservation]) -> float:
    """Return ``1 - abs(net)/gross`` for comparable signed contributions.

    The result lies in [0, 1], where zero denotes no offsetting signs and one
    denotes complete cancellation.  An all-zero collection returns zero.
    """

    rows = list(observations)
    if not rows:
        raise ValueError("at least one observation is required")
    effects = {row.effect_id for row in rows}
    units = {row.units for row in rows}
    if len(effects) != 1 or len(units) != 1:
        raise ValueError("cancellation requires one effect and one unit")
    gross = sum(abs(row.value) for row in rows)
    return 0.0 if gross == 0 else 1.0 - abs(sum(row.value for row in rows)) / gross
