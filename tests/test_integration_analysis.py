import pytest

from sda.analysis.integration import ScoreConfig, build_drilldown, rank_descriptive_cases


ROWS = [
    {"year": 2000, "effect": "D_C", "role": "emitter_source", "country": "A", "value": 8, "anomaly_score": 0.2, "cancellation_ratio": 0.1},
    {"year": 2000, "effect": "D_C", "role": "emitter_source", "country": "B", "value": 2, "anomaly_score": 0.1, "cancellation_ratio": 0.0},
    {"year": 2001, "effect": "D_C", "role": "emitter_source", "country": "A", "value": 6, "anomaly_score": 0.3, "cancellation_ratio": 0.2},
    {"year": 2001, "effect": "D_C", "role": "emitter_source", "country": "B", "value": -4, "anomaly_score": 0.1, "cancellation_ratio": 0.0},
    {"year": 2000, "effect": "D_YF", "role": "final_demand_source", "country": "A", "value": 100, "anomaly_score": 0.0, "cancellation_ratio": 0.0},
]


def test_ranking_is_role_separated_and_deterministic():
    ranked = rank_descriptive_cases(ROWS, case_dimensions=("country",))
    dc = [row for row in ranked if row["effect"] == "D_C"]
    dyf = [row for row in ranked if row["effect"] == "D_YF"]
    assert [row["country"] for row in dc] == ["A", "B"]
    assert [row["rank_within_effect_role"] for row in dc] == [1, 2]
    assert dyf[0]["rank_within_effect_role"] == 1
    assert all(row["interpretation"].startswith("descriptive") for row in ranked)


def test_persistence_uses_signed_consistency_over_time():
    ranked = rank_descriptive_cases(ROWS, case_dimensions=("country",))
    dc = {row["country"]: row for row in ranked if row["effect"] == "D_C"}
    assert dc["A"]["persistence"] == 1.0
    assert dc["B"]["persistence"] == pytest.approx(1 / 3)


def test_role_mismatch_is_rejected_instead_of_semantically_merged():
    bad = [{**ROWS[0], "role": "final_demand_source"}]
    with pytest.raises(ValueError, match="D_C requires role"):
        rank_descriptive_cases(bad, case_dimensions=("country",))


def test_drilldown_preserves_effect_role_and_orders_years():
    candidate = {"effect": "D_C", "role": "emitter_source", "country": "A"}
    detail = build_drilldown(reversed(ROWS), candidate, case_dimensions=("country",))
    assert [row["year"] for row in detail] == [2000, 2001]
    assert {row["role"] for row in detail} == {"emitter_source"}


def test_configuration_requires_nonnegative_nonzero_weights():
    with pytest.raises(ValueError):
        ScoreConfig(materiality=-1)
    with pytest.raises(ValueError):
        ScoreConfig(0, 0, 0, 0)
