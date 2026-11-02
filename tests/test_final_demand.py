import numpy as np
import pytest

from sda.effects import three_factor_final_demand_components
from sda.validation import assert_additive


def _fixture():
    F0 = np.array([[0.7, 0.2], [0.3, 0.8], [0.6, 0.4], [0.4, 0.6]])
    F1 = np.array([[0.6, 0.25], [0.4, 0.75], [0.55, 0.5], [0.45, 0.5]])
    B0 = np.array([[0.65, 0.4], [0.35, 0.6]])
    B1 = np.array([[0.6, 0.45], [0.4, 0.55]])
    y0 = np.array([10.0, 8.0])
    y1 = np.array([11.0, 7.5])
    sectors = np.array([0, 1, 0, 1])
    return F0, F1, B0, B1, y0, y1, sectors


def test_final_demand_components_reconstruct_delta_y_cellwise():
    F0, F1, B0, B1, y0, y1, sectors = _fixture()
    components = three_factor_final_demand_components(F0, F1, B0, B1, y0, y1, sectors)

    Y0 = F0 * B0[sectors, :] * y0[None, :]
    Y1 = F1 * B1[sectors, :] * y1[None, :]
    assert_additive([components.YF, components.YB, components.y], Y1 - Y0)
    assert components.YF.shape == (4, 2)
    assert components.YB.shape == (4, 2)
    assert components.y.shape == (4, 2)


@pytest.mark.parametrize("changed, expected", [("F", "YF"), ("B", "YB"), ("y", "y")])
def test_one_factor_change_is_confined_to_its_component(changed, expected):
    F0, _, B0, _, y0, _, sectors = _fixture()
    F1, B1, y1 = F0.copy(), B0.copy(), y0.copy()
    if changed == "F":
        F1[2, 1] += 0.1
    elif changed == "B":
        B1[1, 0] += 0.1
    else:
        y1[1] += 1.0

    components = three_factor_final_demand_components(F0, F1, B0, B1, y0, y1, sectors)
    for name in ("YF", "YB", "y"):
        values = getattr(components, name)
        if name == expected:
            assert np.count_nonzero(values) > 0
        else:
            np.testing.assert_array_equal(values, np.zeros_like(values))


def test_single_consumer_change_does_not_leak_to_another_consumer():
    F0, _, B0, _, y0, _, sectors = _fixture()
    F1 = F0.copy()
    F1[2, 1] += 0.1
    components = three_factor_final_demand_components(F0, F1, B0, B0, y0, y0, sectors)

    np.testing.assert_array_equal(components.YF[:, 0], np.zeros(4))
    assert components.YF[2, 1] != 0.0
    assert np.count_nonzero(components.YF) == 1


@pytest.mark.parametrize(
    "replacement, message",
    [
        ((np.ones((4, 3)), 0), "share one sector-by-consumer shape"),
        ((np.ones(3), 1), "matching the consumer count"),
        ((np.array([0, 1, 2, 0]), 2), "outside the B sector axis"),
    ],
)
def test_final_demand_algebra_rejects_invalid_dimensions(replacement, message):
    F0, F1, B0, B1, y0, y1, sectors = _fixture()
    values = [B1, y1, sectors]
    values[replacement[1]] = replacement[0]
    with pytest.raises(ValueError, match=message):
        three_factor_final_demand_components(F0, F1, B0, values[0], y0, values[1], values[2])
from sda.attribution import (
    attribute_final_demand_composition,
    attribute_final_demand_compositions,
    attribute_final_demand_level,
)


def test_batched_compositions_match_independent_reference():
    components = {
        "a": np.array([[0.2, -0.1], [0.3, 0.4], [-0.05, 0.2], [0.1, 0.25]]),
        "b": np.array([[0.1, 0.3], [-0.2, 0.1], [0.4, -0.1], [0.2, 0.05]]),
    }
    actual = attribute_final_demand_compositions(
        components, **_inputs(), backend="numpy", batch_size=2
    )
    for key, component in components.items():
        expected = attribute_final_demand_composition(component, **_inputs())
        np.testing.assert_allclose(actual[key], expected)


def test_batched_compositions_apply_result_transform_before_retention():
    components = {
        "a": np.array([[0.2, -0.1], [0.3, 0.4], [-0.05, 0.2], [0.1, 0.25]]),
        "b": np.array([[0.1, 0.3], [-0.2, 0.1], [0.4, -0.1], [0.2, 0.05]]),
    }
    actual = attribute_final_demand_compositions(
        components, **_inputs(), backend="numpy", batch_size=1,
        result_transform=lambda _key, values: values.reshape(4, 2, 2).sum(axis=2),
    )

    for key, component in components.items():
        expected = attribute_final_demand_composition(component, **_inputs())
        np.testing.assert_allclose(actual[key], expected.reshape(4, 2, 2).sum(axis=2))


def test_batched_compositions_reject_invalid_backend_and_batch_size():
    component = {"a": np.ones((4, 2))}
    with pytest.raises(ValueError, match="backend"):
        attribute_final_demand_compositions(component, **_inputs(), backend="invalid")
    with pytest.raises(ValueError, match="batch_size"):
        attribute_final_demand_compositions(component, **_inputs(), batch_size=0)


def _inputs():
    return {
        "v0": np.array([0.7, 1.1, 0.5, 0.9]),
        "delta_v": np.array([0.05, -0.02, 0.01, 0.03]),
        "L0": np.array(
            [
                [1.2, 0.1, 0.05, 0.02],
                [0.2, 1.3, 0.04, 0.01],
                [0.03, 0.05, 1.1, 0.2],
                [0.01, 0.02, 0.15, 1.25],
            ]
        ),
        "L1": np.array(
            [
                [1.25, 0.12, 0.04, 0.03],
                [0.18, 1.28, 0.06, 0.02],
                [0.04, 0.04, 1.15, 0.18],
                [0.02, 0.03, 0.13, 1.3],
            ]
        ),
        "sectors_per_country": 2,
    }


def _reference_endpoint(component, inputs):
    v0 = inputs["v0"]
    dv = inputs["delta_v"]
    L0 = inputs["L0"]
    dL = inputs["L1"] - L0
    kernel = (
        v0[:, None] * L0
        + 0.5 * v0[:, None] * dL
        + 0.5 * dv[:, None] * L0
        + (1.0 / 3.0) * dv[:, None] * dL
    )
    return np.einsum("ei,ic->eic", kernel, component)


@pytest.mark.parametrize("effect", ["YF", "YB"])
def test_composition_retains_consumer_sector_and_reconciles_emitter_total(effect):
    component = np.array(
        [[0.2, -0.1], [0.3, 0.4], [-0.05, 0.2], [0.1, 0.25]]
    )
    inputs = _inputs()

    result = attribute_final_demand_composition(component, **inputs)
    reference = _reference_endpoint(component, inputs)

    assert result.shape == (4, 4)
    np.testing.assert_allclose(result.sum(axis=1), reference.sum(axis=(1, 2)))
    # Column order is (consumer 0, sectors 0/1), then (consumer 1, sectors 0/1).
    np.testing.assert_allclose(result[:, 2], reference[:, [0, 2], 1].sum(axis=1))


def test_composition_change_stays_in_its_consumer_sector_column():
    component = np.zeros((4, 2))
    component[3, 0] = 2.0  # source country 1, sector 1; consumer country 0
    result = attribute_final_demand_composition(component, **_inputs())

    assert np.any(result[:, 1] != 0.0)
    np.testing.assert_array_equal(result[:, [0, 2, 3]], np.zeros((4, 3)))


def test_level_has_only_consumer_axis_and_reconciles_emitter_total():
    component = np.array(
        [[0.5, 0.0], [0.2, 0.1], [0.3, -0.2], [0.4, 0.6]]
    )
    inputs = _inputs()

    result = attribute_final_demand_level(component, **inputs)
    reference = _reference_endpoint(component, inputs)

    assert result.shape == (4, 2)
    np.testing.assert_allclose(result, reference.sum(axis=1))
    np.testing.assert_allclose(result.sum(axis=1), reference.sum(axis=(1, 2)))


def test_final_demand_attribution_rejects_nonconformable_country_count():
    with pytest.raises(ValueError, match="consumer count"):
        attribute_final_demand_level(
            np.ones((4, 3)), **_inputs()
        )
