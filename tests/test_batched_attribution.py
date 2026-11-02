import numpy as np
import pytest

from sda.attribution import attribute_final_demand_composition, attribute_final_demand_level
from sda.attribution import attribute_structural_driver
from sda.effects import refined_laspeyres_intensity_effect


def _stack(value, scale=1.1):
    return np.stack((value, value * scale))


def test_structural_batch_matches_independent_samples():
    component = np.array([[0.02, -0.01], [0.03, 0.04]])
    kwargs = dict(v0=np.array([.7, 1.1]), delta_v=np.array([.05, -.02]),
                  L0=np.array([[1.2, .1], [.2, 1.3]]), L1=np.array([[1.25, .12], [.18, 1.28]]),
                  y0=np.array([5., 3.]), delta_y=np.array([.4, -.2]))
    actual = attribute_structural_driver(_stack(component), **{k: _stack(v) for k, v in kwargs.items()})
    expected = np.stack([attribute_structural_driver(_stack(component)[i], **{k: _stack(v)[i] for k, v in kwargs.items()}) for i in range(2)])
    np.testing.assert_allclose(actual, expected)


def test_final_demand_batch_matches_independent_samples():
    component = np.array([[.2, -.1], [.3, .4], [-.05, .2], [.1, .25]])
    kwargs = dict(v0=np.array([.7, 1.1, .5, .9]), delta_v=np.array([.05, -.02, .01, .03]),
                  L0=np.eye(4), L1=np.eye(4) * 1.05, sectors_per_country=2)
    batch_kwargs = {k: (_stack(v) if k != "sectors_per_country" else v) for k, v in kwargs.items()}
    for function in (attribute_final_demand_composition, attribute_final_demand_level):
        actual = function(_stack(component), **batch_kwargs)
        expected = np.stack([function(_stack(component)[i], **{k: (_stack(v)[i] if k != "sectors_per_country" else v) for k, v in kwargs.items()}) for i in range(2)])
        np.testing.assert_allclose(actual, expected)


def test_intensity_batch_matches_independent_samples():
    args = [np.array([.7, 1.1]), np.array([.8, 1.0]), np.eye(2), np.eye(2) * 1.1,
            np.array([2., 3.]), np.array([2.2, 2.8])]
    batched = [_stack(v) for v in args]
    actual = refined_laspeyres_intensity_effect(*batched)
    expected = np.stack([refined_laspeyres_intensity_effect(*(v[i] for v in batched)) for i in range(2)])
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("function,component,kwargs", [
    (attribute_structural_driver, np.ones((2, 2, 2)), dict(v0=np.ones((3, 2)), delta_v=np.ones((2, 2)), L0=np.ones((2, 2, 2)), L1=np.ones((2, 2, 2)), y0=np.ones((2, 2)), delta_y=np.ones((2, 2)))),
    (attribute_final_demand_level, np.ones((2, 4, 2)), dict(v0=np.ones((3, 4)), delta_v=np.ones((2, 4)), L0=np.ones((2, 4, 4)), L1=np.ones((2, 4, 4)), sectors_per_country=2)),
])
def test_batch_rejects_mismatched_sample_count(function, component, kwargs):
    with pytest.raises(ValueError):
        function(component, **kwargs)
