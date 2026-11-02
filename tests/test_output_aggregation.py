from collections import OrderedDict

import numpy as np
import pytest

from sda import EffectDimensions, EffectResult, LabelSet
from sda.output import AxisGrouping, aggregate_effect


EMITTERS = LabelSet(
    "emitter_country_sector", (("A", "x"), ("A", "y"), ("B", "x"), ("B", "y"))
)
STRUCTURAL = LabelSet(
    "structural_driver_country_sector", (("A", "x"), ("A", "y"), ("B", "x"), ("B", "y"))
)
DEMAND = LabelSet(
    "consumer_demanded_sector", (("A", "x"), ("A", "y"), ("B", "x"), ("B", "y"))
)
CONSUMERS = LabelSet("consumer_country", ("A", "B"))

BY_EMITTER_COUNTRY = AxisGrouping(
    "emitter_country", OrderedDict((country, ((country, "x"), (country, "y"))) for country in ("A", "B"))
)
BY_STRUCTURAL_SECTOR = AxisGrouping(
    "structural_driver_sector",
    OrderedDict((sector, (("A", sector), ("B", sector))) for sector in ("x", "y")),
)
BY_CONSUMER = AxisGrouping(
    "consumer_country",
    OrderedDict((country, ((country, "x"), (country, "y"))) for country in ("A", "B")),
)
CONSUMER_IDENTITY = AxisGrouping(
    "consumer_country", OrderedDict((country, (country,)) for country in ("A", "B"))
)


def test_aggregates_dv_vector_without_creating_a_driver_axis():
    effect = EffectResult("D_v", np.array([1.0, 2.0, 3.0, 4.0]), EffectDimensions(EMITTERS))

    result = aggregate_effect(effect, rows=BY_EMITTER_COUNTRY)

    np.testing.assert_array_equal(result.baseline, [3.0, 7.0])
    assert result.dimensions.column is None
    assert result.dimensions.row.role == "emitter_country"


@pytest.mark.parametrize(
    ("name", "columns", "grouping", "expected_role"),
    [
        ("D_C", STRUCTURAL, BY_STRUCTURAL_SECTOR, "structural_driver_sector"),
        ("D_YF", DEMAND, BY_CONSUMER, "consumer_country"),
    ],
)
def test_equal_shaped_effects_use_declared_column_semantics(name, columns, grouping, expected_role):
    values = np.arange(1.0, 17.0).reshape(4, 4)
    effect = EffectResult(name, values, EffectDimensions(EMITTERS, columns))

    result = aggregate_effect(effect, rows=BY_EMITTER_COUNTRY, columns=grouping)

    assert result.baseline.shape == (2, 2)
    assert result.dimensions.column.role == expected_role
    assert result.baseline.sum() == values.sum()


def test_aggregates_dy_with_distinct_mn_and_consumer_axes_and_mean():
    baseline = np.arange(1.0, 9.0).reshape(4, 2)
    effect = EffectResult(
        "D_y", baseline, EffectDimensions(EMITTERS, CONSUMERS), mean=baseline / 2
    )

    result = aggregate_effect(
        effect, rows=BY_EMITTER_COUNTRY, columns=CONSUMER_IDENTITY
    )

    np.testing.assert_array_equal(result.baseline, [[4.0, 6.0], [12.0, 14.0]])
    np.testing.assert_array_equal(result.mean, result.baseline / 2)
    assert result.baseline.sum() == baseline.sum()


def test_rejects_non_partition_duplicate_and_wrong_axis_count():
    effect = EffectResult("D_v", np.ones(4), EffectDimensions(EMITTERS))
    missing = AxisGrouping("country", {"A": (("A", "x"),), "B": (("B", "x"), ("B", "y"))})
    duplicate = AxisGrouping(
        "country",
        {"A": (("A", "x"), ("A", "y")), "B": (("A", "y"), ("B", "x"), ("B", "y"))},
    )

    with pytest.raises(ValueError, match="exact partition"):
        aggregate_effect(effect, rows=missing)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_effect(effect, rows=duplicate)
    with pytest.raises(ValueError, match="no column axis"):
        aggregate_effect(effect, rows=BY_EMITTER_COUNTRY, columns=BY_CONSUMER)


def test_matrix_requires_explicit_column_grouping():
    effect = EffectResult("D_C", np.ones((4, 4)), EffectDimensions(EMITTERS, STRUCTURAL))

    with pytest.raises(ValueError, match="requires a column grouping"):
        aggregate_effect(effect, rows=BY_EMITTER_COUNTRY)


def test_source_category_effects_are_aggregated_with_the_same_explicit_axes():
    values = np.arange(1.0, 17.0).reshape(4, 4)
    domestic = EffectResult("D_C", values / 4, EffectDimensions(EMITTERS, STRUCTURAL))
    effect = EffectResult(
        "D_C",
        values,
        EffectDimensions(EMITTERS, STRUCTURAL),
        source_categories={"Domestic": domestic},
    )

    result = aggregate_effect(
        effect, rows=BY_EMITTER_COUNTRY, columns=BY_STRUCTURAL_SECTOR
    )

    assert result.source_categories is not None
    assert result.source_categories["Domestic"].baseline.sum() == domestic.baseline.sum()
    assert result.source_categories["Domestic"].dimensions == result.dimensions
