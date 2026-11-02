import math

import pytest

from sda.analysis.temporal import (
    TemporalObservation,
    anomaly_scores,
    cancellation_ratio,
    find_sign_reversals,
    materiality_filter,
    with_activity_shares,
)


def obs(year, value, series="DEU:A", effect="D_C", units="kg CO2"):
    return TemporalObservation(f"{year-1}_{year}", year - 1, year, effect, series, value, units)


def test_activity_shares_are_absolute_and_preserve_signed_values_and_metadata():
    rows = with_activity_shares([obs(2002, 3, "A"), obs(2002, -1, "B")])
    assert [row.value for row in rows] == [3, -1]
    assert [row.absolute_activity_share for row in rows] == pytest.approx([0.75, 0.25])
    assert rows[0].transition == "2001_2002"
    assert rows[0].units == "kg CO2"
    assert rows[0].hierarchy_parent == "leontief_production_structure"
    assert obs(2002, 1, effect="D_T_coef").hierarchy_level == 3


def test_robust_scores_flag_isolated_departure_when_mad_is_zero():
    rows = anomaly_scores([obs(2002, 1), obs(2003, 1), obs(2004, 1), obs(2005, 20)])
    assert [row.is_anomaly for row in rows] == [False, False, False, True]
    assert math.isinf(rows[-1].robust_anomaly_score)


def test_robust_scores_use_configurable_threshold_and_separate_effects():
    rows = [obs(2002, 0), obs(2003, 1), obs(2004, 10), obs(2002, 100, effect="D_v")]
    scored = anomaly_scores(rows, threshold=2)
    assert scored[2].is_anomaly
    assert scored[3].robust_anomaly_score == 0


def test_sign_reversals_require_adjacent_transitions_and_material_values():
    rows = [obs(2002, 2), obs(2003, -3), obs(2005, 4), obs(2006, -0.1)]
    reversals = find_sign_reversals(rows, min_absolute=1)
    assert [(a.transition, b.transition) for a, b in reversals] == [("2001_2002", "2002_2003")]


def test_cancellation_ratio_and_materiality_filter():
    rows = with_activity_shares([obs(2002, 8, "A"), obs(2002, -2, "B")])
    assert cancellation_ratio(rows) == pytest.approx(0.4)
    assert [row.series_id for row in materiality_filter(rows, min_absolute=1, min_activity_share=0.5)] == ["A"]


def test_rejects_unknown_effect_mixed_units_and_inconsistent_labels():
    with pytest.raises(ValueError, match="unknown canonical effect"):
        obs(2002, 1, effect="other")
    with pytest.raises(ValueError, match="one effect and one unit"):
        cancellation_ratio([obs(2002, 1), obs(2003, -1, units="Mt CO2")])
    bad = TemporalObservation("wrong", 2001, 2002, "D_C", "B", 1, "kg CO2")
    with pytest.raises(ValueError, match="inconsistent transition labels"):
        with_activity_shares([obs(2002, 1), bad])
