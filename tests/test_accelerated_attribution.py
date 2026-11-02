import numpy as np
import pytest

from sda.attribution import attribute_structural_components, attribute_structural_driver


def _inputs():
    n = 4
    return {
        "v0": np.linspace(0.8, 1.1, n),
        "delta_v": np.linspace(-0.02, 0.01, n),
        "L0": np.eye(n) + 0.02,
        "L1": np.eye(n) + 0.03,
        "y0": np.arange(2.0, 6.0),
        "delta_y": np.linspace(-0.1, 0.2, n),
    }


def test_numpy_batches_match_canonical_kernel_exactly():
    components = {"a": np.eye(4) * 0.1, "b": np.arange(16).reshape(4, 4) / 100}
    inputs = _inputs()
    actual = attribute_structural_components(components, **inputs, batch_size=2)
    for key, component in components.items():
        expected = attribute_structural_driver(component, **inputs)
        np.testing.assert_allclose(actual[key], expected, rtol=1e-12, atol=1e-12)


def test_invalid_backend_and_batch_are_rejected():
    components = {"a": np.eye(4)}
    with pytest.raises(ValueError, match="backend"):
        attribute_structural_components(components, **_inputs(), backend="other")
    with pytest.raises(ValueError, match="batch_size"):
        attribute_structural_components(components, **_inputs(), batch_size=0)


def test_cupy_matches_reference_when_cuda_is_available():
    pytest.importorskip("cupy")
    components = {"a": np.eye(4) * 0.1, "b": np.arange(16).reshape(4, 4) / 100}
    inputs = _inputs()
    try:
        actual = attribute_structural_components(components, **inputs, backend="cupy", batch_size=2)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    for key, component in components.items():
        expected = attribute_structural_driver(component, **inputs)
        np.testing.assert_allclose(actual[key], expected, rtol=2e-5, atol=2e-6)
