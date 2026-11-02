"""Role-safe descriptive indicators for detailed ``D_C`` and ``D_YF`` sourcing.

The indicators describe attribution results; they do not identify causal offshoring.
Concentration measures use absolute contributions because signed shares can be
undefined or misleading under cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Literal

import numpy as np

from .final_demand_sourcing import YFSourceCountryCube
from .offshoring import SourceCountryCube

Evidence = Literal["NO", "WEAK", "MODERATE", "STRONG"]


@dataclass(frozen=True)
class SourceShare:
    """Persistence-ready contribution and shares for one source and focal role."""

    source_country: str
    signed_contribution: float
    signed_share_of_net: float | None
    absolute_share: float


@dataclass(frozen=True)
class SourcingIndicators:
    """Indicators for one structural driver (D_C) or consumer (D_YF)."""

    effect_id: Literal["D_C", "D_YF"]
    focal_role: Literal["structural_driver_country", "consumer_country"]
    focal_country: str
    signed_total: float
    signed_domestic_contribution: float
    signed_foreign_contribution: float
    gross_absolute_contribution: float
    absolute_foreign_sourcing_intensity: float | None
    source_hhi: float | None
    source_entropy: float | None
    effective_source_count: float | None
    cancellation_ratio: float | None
    evidence: Evidence
    source_shares: tuple[SourceShare, ...]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _evidence(domestic: float, foreign: float, intensity: float | None) -> Evidence:
    """Classify an offshoring-consistent pattern without making a causal claim.

    Evidence requires a negative domestic and positive foreign contribution.
    Strength is the foreign share of gross absolute sourcing: below 0.50 is weak,
    0.50--0.75 moderate, and at least 0.75 strong.
    """
    if not (domestic < 0.0 and foreign > 0.0) or intensity is None:
        return "NO"
    if intensity >= 0.75:
        return "STRONG"
    if intensity >= 0.50:
        return "MODERATE"
    return "WEAK"


def analyse_source_country_cube(
    cube: SourceCountryCube | YFSourceCountryCube,
) -> tuple[SourcingIndicators, ...]:
    """Collapse a reconciled source-country cube into role-safe indicators.

    Emitters (and D_YF demanded sectors) are summed. D_C retains structural
    driver country as its focal role; D_YF retains consumer country. Domestic
    means source country equals that focal country.
    """
    values = np.asarray(cube.baseline, dtype=np.float64)
    if isinstance(cube, SourceCountryCube):
        effect_id: Literal["D_C", "D_YF"] = "D_C"
        focal_role: Literal["structural_driver_country", "consumer_country"] = (
            "structural_driver_country"
        )
        focal_labels = cube.driver_country.labels
        by_focal_source = values.sum(axis=0)
    elif isinstance(cube, YFSourceCountryCube):
        effect_id = "D_YF"
        focal_role = "consumer_country"
        focal_labels = cube.consumer_country.labels
        by_focal_source = values.sum(axis=0)
    else:
        raise TypeError("cube must be SourceCountryCube or YFSourceCountryCube")

    sources = cube.source_country.labels
    rows: list[SourcingIndicators] = []
    for focal_index, focal in enumerate(focal_labels):
        contributions = np.asarray(by_focal_source[focal_index], dtype=np.float64)
        domestic = float(sum(v for s, v in zip(sources, contributions) if s == focal))
        total = float(contributions.sum())
        foreign = total - domestic
        absolute = np.abs(contributions)
        gross = float(absolute.sum())
        foreign_gross = float(sum(v for s, v in zip(sources, absolute) if s != focal))
        intensity = _safe_ratio(foreign_gross, gross)
        shares = absolute / gross if gross else np.zeros_like(absolute)
        hhi = float(np.square(shares).sum()) if gross else None
        entropy = float(-sum(p * log(p) for p in shares if p > 0.0)) if gross else None
        effective = exp(entropy) if entropy is not None else None
        cancellation = _safe_ratio(gross - abs(total), gross)
        persistence = tuple(
            SourceShare(
                source_country=str(source),
                signed_contribution=float(value),
                signed_share_of_net=_safe_ratio(float(value), total),
                absolute_share=float(share),
            )
            for source, value, share in zip(sources, contributions, shares)
        )
        rows.append(SourcingIndicators(
            effect_id=effect_id, focal_role=focal_role, focal_country=str(focal),
            signed_total=total, signed_domestic_contribution=domestic,
            signed_foreign_contribution=foreign, gross_absolute_contribution=gross,
            absolute_foreign_sourcing_intensity=intensity, source_hhi=hhi,
            source_entropy=entropy, effective_source_count=effective,
            cancellation_ratio=cancellation,
            evidence=_evidence(domestic, foreign, intensity), source_shares=persistence,
        ))
    return tuple(rows)
