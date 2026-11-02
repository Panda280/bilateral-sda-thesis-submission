"""Role-safe descriptive integration and case selection for SDA panel rows.

The functions in this module rank observations within an effect/role stratum.
They deliberately do not combine emitter sourcing (``D_C``) with final-demand
sourcing (``D_YF``), because those dimensions answer different questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping, Sequence


EFFECT_ROLES = {
    "D_C": "emitter_source",
    "D_YF": "final_demand_source",
}


@dataclass(frozen=True)
class ScoreConfig:
    """Weights for a descriptive, non-causal case-selection score."""

    materiality: float = 0.4
    anomaly: float = 0.25
    persistence: float = 0.2
    cancellation: float = 0.15

    def __post_init__(self) -> None:
        weights = (self.materiality, self.anomaly, self.persistence, self.cancellation)
        if any(not isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("score weights must be finite and non-negative")
        if sum(weights) == 0:
            raise ValueError("at least one score weight must be positive")


def _number(row: Mapping[str, object], name: str, default: float | None = None) -> float:
    value = row.get(name, default)
    if value is None:
        raise ValueError(f"missing required numeric field: {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_row(row: Mapping[str, object], dimensions: Sequence[str]) -> None:
    for field in ("year", "effect", "role", "value", *dimensions):
        if field not in row:
            raise ValueError(f"missing required field: {field}")
    effect = str(row["effect"])
    role = str(row["role"])
    expected = EFFECT_ROLES.get(effect)
    if expected is not None and role != expected:
        raise ValueError(f"{effect} requires role={expected!r}, got {role!r}")
    if effect not in EFFECT_ROLES:
        raise ValueError(f"unsupported sourcing effect: {effect}")
    _number(row, "year")
    _number(row, "value")
    for metric in ("anomaly_score", "cancellation_ratio"):
        value = _number(row, metric, 0.0)
        if value < 0:
            raise ValueError(f"{metric} must be non-negative")


def rank_descriptive_cases(
    rows: Iterable[Mapping[str, object]],
    *,
    case_dimensions: Sequence[str],
    config: ScoreConfig = ScoreConfig(),
) -> list[dict[str, object]]:
    """Rank cases separately within each effect/role stratum.

    Annual materiality is the absolute contribution divided by the stratum's
    annual absolute total. Persistence is directional consistency across years,
    ``abs(sum(value)) / sum(abs(value))``. Input anomaly and cancellation scores
    are expected to be non-negative descriptive diagnostics.
    """

    materialized = [dict(row) for row in rows]
    if not case_dimensions:
        raise ValueError("case_dimensions must not be empty")
    for row in materialized:
        _validate_row(row, case_dimensions)

    denominators: dict[tuple[object, str, str], float] = {}
    for row in materialized:
        key = (row["year"], str(row["effect"]), str(row["role"]))
        denominators[key] = denominators.get(key, 0.0) + abs(_number(row, "value"))

    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in materialized:
        key = (str(row["effect"]), str(row["role"]), *(row[d] for d in case_dimensions))
        groups.setdefault(key, []).append(row)

    total_weight = sum((config.materiality, config.anomaly, config.persistence, config.cancellation))
    results: list[dict[str, object]] = []
    for key, observations in groups.items():
        effect, role, *case_values = key
        annual_materiality = []
        values = []
        for row in observations:
            value = _number(row, "value")
            values.append(value)
            denominator = denominators[(row["year"], effect, role)]
            annual_materiality.append(abs(value) / denominator if denominator else 0.0)
        materiality = sum(annual_materiality) / len(annual_materiality)
        anomaly = sum(_number(row, "anomaly_score", 0.0) for row in observations) / len(observations)
        cancellation = sum(_number(row, "cancellation_ratio", 0.0) for row in observations) / len(observations)
        absolute_total = sum(abs(value) for value in values)
        persistence = abs(sum(values)) / absolute_total if absolute_total else 0.0
        score = (
            config.materiality * materiality
            + config.anomaly * anomaly
            + config.persistence * persistence
            + config.cancellation * cancellation
        ) / total_weight
        result: dict[str, object] = {
            "effect": effect,
            "role": role,
            **dict(zip(case_dimensions, case_values, strict=True)),
            "year_count": len({row["year"] for row in observations}),
            "signed_total": sum(values),
            "absolute_total": absolute_total,
            "materiality": materiality,
            "anomaly": anomaly,
            "persistence": persistence,
            "cancellation": cancellation,
            "descriptive_score": score,
            "interpretation": "descriptive candidate; no causal inference",
        }
        results.append(result)

    strata = sorted({(str(row["effect"]), str(row["role"])) for row in results})
    ranked: list[dict[str, object]] = []
    for effect, role in strata:
        subset = [row for row in results if row["effect"] == effect and row["role"] == role]
        subset.sort(
            key=lambda row: (
                -float(row["descriptive_score"]),
                tuple(str(row[d]) for d in case_dimensions),
            )
        )
        for rank, row in enumerate(subset, start=1):
            ranked.append({**row, "rank_within_effect_role": rank})
    return ranked


def build_drilldown(
    rows: Iterable[Mapping[str, object]],
    candidate: Mapping[str, object],
    *,
    case_dimensions: Sequence[str],
) -> list[dict[str, object]]:
    """Return deterministic annual detail for one explicitly role-bound case."""

    required = ("effect", "role", *case_dimensions)
    if any(field not in candidate for field in required):
        raise ValueError("candidate lacks effect, role, or case dimensions")
    selected = []
    for original in rows:
        row = dict(original)
        _validate_row(row, case_dimensions)
        if all(row[field] == candidate[field] for field in required):
            selected.append(row)
    selected.sort(key=lambda row: (float(row["year"]), tuple(str(row[d]) for d in case_dimensions)))
    return selected
