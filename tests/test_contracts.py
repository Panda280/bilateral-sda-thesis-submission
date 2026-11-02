import numpy as np
import pytest

from sda import CANONICAL_EFFECTS, EffectDimensions, EffectResult, LabelSet
from sda.hierarchy import TOP_LEVEL_EFFECTS


def test_canonical_effect_inventory_and_top_level_do_not_double_count_nested_effects():
    assert TOP_LEVEL_EFFECTS == ("D_v", "D_C", "D_A_x", "D_YF", "D_YB", "D_y")
    assert set(CANONICAL_EFFECTS) == set(TOP_LEVEL_EFFECTS) | {"D_T_coef", "D_w_coef"}


def test_effect_result_checks_dimension_cardinality():
    emitters = LabelSet("emitter_country_sector", (("A", "x"), ("B", "x")))
    drivers = LabelSet("structural_driver_country_sector", (("A", "x"), ("B", "x")))
    dimensions = EffectDimensions(emitters, drivers)

    result = EffectResult("D_C", np.zeros((2, 2)), dimensions)
    assert result.baseline.shape == (2, 2)

    with pytest.raises(ValueError, match="does not match dimensions"):
        EffectResult("D_C", np.zeros((2, 1)), dimensions)


def test_dv_is_a_vector_without_a_fake_driver_axis():
    emitters = LabelSet("emitter_country_sector", (("A", "x"), ("B", "x")))
    result = EffectResult("D_v", np.zeros(2), EffectDimensions(emitters))
    assert result.dimensions.column is None


def test_labels_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        LabelSet("consumer_country", ("A", "A"))
