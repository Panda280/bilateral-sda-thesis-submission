"""Descriptive period scorecards for annual, signed SDA results.

This module deliberately does not assess Kyoto Protocol compliance.  It
summarises already validated SDA contributions and compares user-supplied
country groups without introducing emissions targets, inventories, or legal
status.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class AnnualEffect:
    """One tidy annual-transition contribution at a fixed decomposition level."""

    year_previous: int
    year_current: int
    entity: str
    effect_id: str
    value: float
    units: str

    def __post_init__(self) -> None:
        if self.year_current <= self.year_previous:
            raise ValueError("year_current must be later than year_previous")
        if not self.entity or not self.effect_id or not self.units:
            raise ValueError("entity, effect_id, and units must be non-empty")
        if not isfinite(self.value):
            raise ValueError("value must be finite")


@dataclass(frozen=True, slots=True)
class PeriodScorecard:
    """Descriptive summary for one entity/group, effect, unit, and period."""

    entity: str
    effect_id: str
    units: str
    period_start: int
    period_end: int
    observation_count: int
    cumulative_signed: float
    cumulative_absolute: float
    annual_average: float
    volatility: float
    positive_count: int
    negative_count: int
    zero_count: int
    sign_persistence: float
    sign_reversals: int


def _score(rows: list[AnnualEffect], *, entity: str, start_year: int, end_year: int) -> PeriodScorecard:
    if not rows:
        raise ValueError("cannot score an empty series")
    effects = {row.effect_id for row in rows}
    units = {row.units for row in rows}
    if len(effects) != 1 or len(units) != 1:
        raise ValueError("a scorecard requires one effect and one unit")
    ordered = sorted(rows, key=lambda row: (row.year_previous, row.year_current))
    pairs = [(row.year_previous, row.year_current) for row in ordered]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate transitions in one scorecard series")
    values = [row.value for row in ordered]
    mean = sum(values) / len(values)
    volatility = sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = len(values) - positive - negative
    nonzero = positive + negative
    reversals = sum(
        left.value * right.value < 0 and left.year_current == right.year_previous
        for left, right in zip(ordered, ordered[1:])
    )
    return PeriodScorecard(
        entity=entity,
        effect_id=ordered[0].effect_id,
        units=ordered[0].units,
        period_start=start_year,
        period_end=end_year,
        observation_count=len(values),
        cumulative_signed=sum(values),
        cumulative_absolute=sum(abs(value) for value in values),
        annual_average=mean,
        volatility=volatility,
        positive_count=positive,
        negative_count=negative,
        zero_count=zero,
        sign_persistence=(max(positive, negative) / nonzero if nonzero else 0.0),
        sign_reversals=reversals,
    )


def period_scorecards(
    observations: Iterable[AnnualEffect], *, start_year: int, end_year: int
) -> list[PeriodScorecard]:
    """Score entity-effect series for transitions wholly inside a period.

    A transition is included when ``year_previous >= start_year`` and
    ``year_current <= end_year``. Rows are never combined across effects or
    units.
    """

    if end_year <= start_year:
        raise ValueError("end_year must be later than start_year")
    groups: dict[tuple[str, str, str], list[AnnualEffect]] = {}
    for row in observations:
        if row.year_previous >= start_year and row.year_current <= end_year:
            groups.setdefault((row.entity, row.effect_id, row.units), []).append(row)
    return [
        _score(rows, entity=entity, start_year=start_year, end_year=end_year)
        for (entity, _effect, _units), rows in sorted(groups.items())
    ]


def group_scorecards(
    observations: Iterable[AnnualEffect],
    country_groups: Mapping[str, str],
    *,
    start_year: int,
    end_year: int,
) -> list[PeriodScorecard]:
    """Return descriptive group totals using an explicit country-group map.

    Every observed entity must be mapped. Contributions are first summed by
    group, transition, effect, and unit, then passed through the same period
    scorecard calculation. The result is not a compliance assessment.
    """

    aggregated: dict[tuple[str, int, int, str, str], float] = {}
    for row in observations:
        if row.entity not in country_groups:
            raise ValueError(f"missing country-group mapping for {row.entity}")
        group = country_groups[row.entity]
        if not group:
            raise ValueError(f"empty country-group mapping for {row.entity}")
        key = (group, row.year_previous, row.year_current, row.effect_id, row.units)
        aggregated[key] = aggregated.get(key, 0.0) + row.value
    rows = [
        AnnualEffect(previous, current, group, effect, value, units)
        for (group, previous, current, effect, units), value in aggregated.items()
    ]
    return period_scorecards(rows, start_year=start_year, end_year=end_year)


def rank_scorecards(
    scorecards: Iterable[PeriodScorecard], *, metric: str = "cumulative_absolute", descending: bool = True
) -> list[PeriodScorecard]:
    """Rank comparable scorecards by an explicitly selected numeric metric."""

    allowed = {"cumulative_signed", "cumulative_absolute", "annual_average", "volatility", "sign_persistence", "sign_reversals"}
    if metric not in allowed:
        raise ValueError(f"unsupported ranking metric: {metric}")
    rows = list(scorecards)
    if len({(row.effect_id, row.units, row.period_start, row.period_end) for row in rows}) > 1:
        raise ValueError("rankings require the same effect, unit, and period")
    return sorted(rows, key=lambda row: (getattr(row, metric), row.entity), reverse=descending)
