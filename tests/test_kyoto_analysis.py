import math

import pytest

from sda.analysis.kyoto import AnnualEffect, group_scorecards, period_scorecards, rank_scorecards


def row(previous, current, entity, value, effect="D_C", units="kg CO2"):
    return AnnualEffect(previous, current, entity, effect, value, units)


def test_period_scorecard_signed_absolute_volatility_and_sign_diagnostics():
    result = period_scorecards(
        [row(2004, 2005, "A", 2), row(2005, 2006, "A", -1), row(2006, 2007, "A", -3), row(2007, 2008, "A", 0)],
        start_year=2004,
        end_year=2008,
    )[0]
    assert result.cumulative_signed == -2
    assert result.cumulative_absolute == 6
    assert result.annual_average == -0.5
    assert result.volatility == pytest.approx(math.sqrt(13 / 4))
    assert (result.positive_count, result.negative_count, result.zero_count) == (1, 2, 1)
    assert result.sign_persistence == pytest.approx(2 / 3)
    assert result.sign_reversals == 1


def test_period_boundary_excludes_crossing_transitions_and_preserves_effects():
    results = period_scorecards(
        [row(2003, 2004, "A", 99), row(2004, 2005, "A", 2), row(2004, 2005, "A", 3, effect="D_YF"), row(2008, 2009, "A", 99)],
        start_year=2004,
        end_year=2008,
    )
    assert [(result.effect_id, result.cumulative_signed) for result in results] == [("D_C", 2), ("D_YF", 3)]


def test_group_comparison_sums_countries_before_temporal_diagnostics():
    results = group_scorecards(
        [row(2004, 2005, "A", 4), row(2004, 2005, "B", -1), row(2005, 2006, "A", -2), row(2005, 2006, "B", 1), row(2004, 2005, "C", 5)],
        {"A": "Kyoto-listed", "B": "Kyoto-listed", "C": "comparison"},
        start_year=2004,
        end_year=2006,
    )
    by_group = {result.entity: result for result in results}
    assert by_group["Kyoto-listed"].cumulative_signed == 2
    assert by_group["Kyoto-listed"].cumulative_absolute == 4
    assert by_group["Kyoto-listed"].sign_reversals == 1
    assert by_group["comparison"].cumulative_signed == 5


def test_missing_group_mapping_is_reported():
    with pytest.raises(ValueError, match="missing country-group mapping"):
        group_scorecards([row(2004, 2005, "A", 1)], {}, start_year=2004, end_year=2008)


def test_rankings_require_comparable_rows():
    scores = period_scorecards([row(2004, 2005, "A", -3), row(2004, 2005, "B", 2)], start_year=2004, end_year=2008)
    assert [score.entity for score in rank_scorecards(scores)] == ["A", "B"]
    incompatible = scores + period_scorecards([row(2004, 2005, "C", 4, effect="D_YF")], start_year=2004, end_year=2008)
    with pytest.raises(ValueError, match="same effect"):
        rank_scorecards(incompatible)


def test_duplicate_transition_and_invalid_observation_are_rejected():
    with pytest.raises(ValueError, match="duplicate transitions"):
        period_scorecards([row(2004, 2005, "A", 1), row(2004, 2005, "A", 2)], start_year=2004, end_year=2008)
    with pytest.raises(ValueError, match="finite"):
        row(2004, 2005, "A", float("nan"))
