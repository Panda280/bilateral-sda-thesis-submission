import numpy as np
import pytest

from sda.attribution import SOURCE_ORIGIN_GROUPS, attribute_structural_driver, build_source_origin_masks
from sda.effects import split_c_component_by_source, two_factor_structural_components
from sda.validation import assert_additive


def labels():
    countries = ("A", "A", "B", "C")
    groups = {"A": "Red", "B": "Non-Red", "C": "Non-Participant"}
    return countries, groups


def test_source_masks_are_exhaustive_and_domestic_overrides_group():
    countries, groups = labels()
    masks = build_source_origin_masks(countries, countries, groups)

    assert tuple(masks) == SOURCE_ORIGIN_GROUPS
    np.testing.assert_array_equal(sum(masks.values()), np.ones((4, 4), dtype=int))
    assert masks["Domestic"][0, 1]
    assert not masks["Red"][0, 1]
    assert masks["Red"][0, 2]
    assert masks["Non-Red"][2, 0]
    assert masks["Non-Participant"][3, 0]


def test_missing_or_invalid_group_classification_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        build_source_origin_masks(("A", "B"), ("A", "B"), {"A": "Red"})
    with pytest.raises(ValueError, match="invalid"):
        build_source_origin_masks(("A",), ("B",), {"A": "Unknown"})


def test_group_restricted_c_components_reconstruct_total_c_cellwise():
    countries, groups = labels()
    masks = build_source_origin_masks(countries, countries, groups)
    C0 = np.array([
        [0.7, 0.6, 0.1, 0.2], [0.3, 0.4, 0.2, 0.1],
        [0.0, 0.0, 0.6, 0.2], [0.0, 0.0, 0.1, 0.5],
    ])
    C1 = C0 + np.array([
        [0.01, -0.02, 0.03, 0.00], [-0.01, 0.02, -0.01, 0.02],
        [0.00, 0.01, -0.02, -0.01], [0.00, -0.01, 0.00, -0.01],
    ])
    A_x0 = np.full((4, 4), 0.2)
    A_x1 = A_x0 + np.arange(16, dtype=float).reshape(4, 4) / 1000.0

    total = two_factor_structural_components(C0, C1, A_x0, A_x1).C
    grouped = split_c_component_by_source(C0, C1, A_x0, A_x1, masks)
    assert_additive(grouped.values(), total)


def test_group_attribution_reconstructs_bilateral_d_c_cellwise():
    countries, groups = labels()
    masks = build_source_origin_masks(countries, countries, groups)
    C0 = np.full((4, 4), 0.25)
    C1 = C0 + np.array([
        [0.02, 0.00, -0.01, 0.01], [0.00, -0.01, 0.02, 0.00],
        [-0.01, 0.01, 0.00, 0.02], [0.01, 0.00, -0.01, -0.02],
    ])
    A_x0 = np.full((4, 4), 0.15)
    A_x1 = A_x0 + 0.01
    grouped = split_c_component_by_source(C0, C1, A_x0, A_x1, masks)
    total_component = two_factor_structural_components(C0, C1, A_x0, A_x1).C
    kwargs = dict(
        v0=np.array([0.7, 0.9, 1.1, 1.3]), delta_v=np.array([0.01, -0.02, 0.0, 0.03]),
        L0=np.eye(4) + 0.02, L1=np.eye(4) + 0.03,
        y0=np.array([2.0, 3.0, 4.0, 5.0]), delta_y=np.array([0.2, -0.1, 0.0, 0.3]),
    )
    expected = attribute_structural_driver(total_component, **kwargs)
    attributed_groups = [attribute_structural_driver(part, **kwargs) for part in grouped.values()]
    assert_additive(attributed_groups, expected)
