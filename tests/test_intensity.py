import numpy as np
import pytest

from sda.effects import refined_laspeyres_intensity_effect


def synthetic_inputs():
    v0 = np.array([0.7, 1.1, 0.4])
    v1 = np.array([0.6, 1.3, 0.45])
    L0 = np.array([[1.10, 0.12, 0.03], [0.04, 1.20, 0.08], [0.02, 0.05, 1.08]])
    L1 = np.array([[1.12, 0.10, 0.04], [0.05, 1.18, 0.09], [0.03, 0.04, 1.10]])
    y0 = np.array([2.0, 3.0, 4.0])
    y1 = np.array([2.2, 2.8, 4.5])
    return v0, v1, L0, L1, y0, y1


def test_intensity_effect_is_emitter_only_vector():
    result = refined_laspeyres_intensity_effect(*synthetic_inputs())

    assert result.shape == (3,)
    assert result.ndim == 1


def test_intensity_effect_reconciles_three_factor_change():
    v0, v1, L0, L1, y0, y1 = synthetic_inputs()
    dv, dL, dy = v1 - v0, L1 - L0, y1 - y0
    intensity = refined_laspeyres_intensity_effect(v0, v1, L0, L1, y0, y1)

    # Independent refined-Laspeyres contributions of L and y. Together with
    # D_v they must reconstruct the change in emitter-specific emissions.
    leontief = (
        v0 * (dL @ y0)
        + 0.5 * dv * (dL @ y0)
        + 0.5 * v0 * (dL @ dy)
        + (1.0 / 3.0) * dv * (dL @ dy)
    )
    demand = (
        v0 * (L0 @ dy)
        + 0.5 * dv * (L0 @ dy)
        + 0.5 * v0 * (dL @ dy)
        + (1.0 / 3.0) * dv * (dL @ dy)
    )
    total_change = v1 * (L1 @ y1) - v0 * (L0 @ y0)

    np.testing.assert_allclose(intensity + leontief + demand, total_change, atol=1e-12)


def test_intensity_effect_is_zero_without_intensity_change():
    v0, _, L0, L1, y0, y1 = synthetic_inputs()
    np.testing.assert_array_equal(
        refined_laspeyres_intensity_effect(v0, v0, L0, L1, y0, y1),
        np.zeros(3),
    )


def test_intensity_effect_rejects_shape_and_nonfinite_inputs():
    v0, v1, L0, L1, y0, y1 = synthetic_inputs()
    with pytest.raises(ValueError, match="vectors matching"):
        refined_laspeyres_intensity_effect(v0[:, None], v1, L0, L1, y0, y1)
    invalid_L = L1.copy()
    invalid_L[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        refined_laspeyres_intensity_effect(v0, v1, L0, invalid_L, y0, y1)
