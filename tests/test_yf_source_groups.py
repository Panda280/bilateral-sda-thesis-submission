import numpy as np
import pytest

from sda.attribution import SOURCE_ORIGIN_GROUPS, build_source_origin_masks
from sda.effects import split_yf_component_by_source
from sda.validation import assert_additive


def source_masks():
    source_countries = ("A", "A", "B", "C")
    consumer_countries = ("A", "A", "B", "C")
    groups = {"A": "Red", "B": "Non-Red", "C": "Non-Participant"}
    return build_source_origin_masks(source_countries, consumer_countries, groups)


def test_yf_source_groups_reconstruct_component_cellwise():
    component = np.arange(-8.0, 8.0).reshape(4, 4) / 10.0
    grouped = split_yf_component_by_source(component, source_masks())

    assert tuple(grouped) == SOURCE_ORIGIN_GROUPS
    assert_additive(grouped.values(), component)
    for name, mask in source_masks().items():
        np.testing.assert_array_equal(grouped[name][~mask], 0.0)


def test_yf_source_groups_require_matching_exhaustive_masks():
    component = np.ones((4, 4))
    masks = source_masks()
    with pytest.raises(ValueError, match="match the YF"):
        split_yf_component_by_source(component, {"Domestic": np.ones((3, 3))})

    overlapping = dict(masks)
    overlapping["Red"] = overlapping["Red"] | overlapping["Domestic"]
    with pytest.raises(ValueError, match="mutually exclusive and exhaustive"):
        split_yf_component_by_source(component, overlapping)


def test_yf_source_groups_reject_invalid_component_data():
    with pytest.raises(ValueError, match="two-dimensional"):
        split_yf_component_by_source(np.ones(4), source_masks())
    invalid = np.ones((4, 4))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        split_yf_component_by_source(invalid, source_masks())
