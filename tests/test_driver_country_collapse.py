import numpy as np
import pytest

from sda import EffectDimensions, EffectResult, LabelSet
from sda.output import collapse_driver_to_country


EMITTERS = LabelSet("emitter_country_sector", (("A", "x"), ("B", "y")))


@pytest.mark.parametrize(
    "driver_labels",
    [
        (("B", "x"), ("A", "x"), ("B", "y"), ("A", "y")),
        ("B:x", "A:x", "B:y", "A:y"),
    ],
)
def test_collapses_driver_sectors_in_first_country_appearance_order(driver_labels):
    baseline = np.arange(1.0, 9.0).reshape(2, 4)
    mean = baseline / 2
    metadata = {"unit": "kg", "method": "bilateral"}
    effect = EffectResult(
        "D_A_x",
        baseline,
        EffectDimensions(
            EMITTERS, LabelSet("structural_driver_country_sector", driver_labels)
        ),
        mean=mean,
        metadata=metadata,
    )

    result = collapse_driver_to_country(effect)

    np.testing.assert_array_equal(result.baseline, [[4.0, 6.0], [12.0, 14.0]])
    np.testing.assert_array_equal(result.mean, result.baseline / 2)
    assert result.dimensions.row is EMITTERS
    assert result.dimensions.column == LabelSet("structural_driver_country", ("B", "A"))
    assert result.metadata == metadata
    assert result.baseline.sum() == baseline.sum()
    assert result.mean.sum() == mean.sum()


@pytest.mark.parametrize(
    "labels",
    [
        (("A",), ("B", "x")),
        (("A", ""), ("B", "x")),
        ("A-x", "B:x"),
        ("A:x:extra", "B:x"),
    ],
)
def test_rejects_malformed_driver_labels(labels):
    effect = EffectResult(
        "D_C",
        np.ones((2, 2)),
        EffectDimensions(
            EMITTERS, LabelSet("structural_driver_country_sector", labels)
        ),
    )

    with pytest.raises(ValueError, match="structural driver"):
        collapse_driver_to_country(effect)


def test_rejects_non_structural_column_semantics_and_source_categories():
    demand = EffectResult(
        "D_YF",
        np.ones((2, 2)),
        EffectDimensions(EMITTERS, LabelSet("consumer_country", ("A", "B"))),
    )
    with pytest.raises(ValueError, match="requires a structural"):
        collapse_driver_to_country(demand)

    structural_dimensions = EffectDimensions(
        EMITTERS,
        LabelSet("structural_driver_country_sector", (("A", "x"), ("B", "x"))),
    )
    component = EffectResult("D_C", np.ones((2, 2)), structural_dimensions)
    grouped = EffectResult(
        "D_C",
        np.ones((2, 2)),
        structural_dimensions,
        source_categories={"Domestic": component},
    )
    with pytest.raises(ValueError, match="does not support source-category"):
        collapse_driver_to_country(grouped)


def test_signed_cancellation_is_checked_against_gross_scale():
    emitter = LabelSet("emitter_country_sector", (("E", "s0"),))
    drivers = LabelSet(
        "structural_driver_country_sector",
        (("A", "s0"), ("A", "s1"), ("B", "s0"), ("B", "s1")),
    )
    values = np.array([[1.0e12, -1.0e12 + 1.0, 1.0e12, -1.0e12 - 1.0]])
    result = collapse_driver_to_country(
        EffectResult("D_C", values, EffectDimensions(emitter, drivers)),
        rtol=1e-12,
    )
    assert result.baseline.shape == (1, 2)
    np.testing.assert_allclose(result.baseline.sum(), values.sum(), atol=4.0, rtol=0.0)
