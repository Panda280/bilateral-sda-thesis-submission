import numpy as np
import pytest

from sda.validation import assert_scaled_additive, check_scaled_additivity


def test_real_float32_noise_passes_at_large_scale_with_diagnostics():
    parent = np.array([[1.0e11, -8.0e10], [2.0e10, 0.0]])
    first = parent * 0.25
    second = parent - first
    second[0, 0] += 1.0e6

    report = assert_scaled_additive((first, second), parent)

    assert report.passed
    assert report.failing_index is None
    assert report.max_abs_error == pytest.approx(1.0e6)
    assert report.atol > 1.0e6


def test_material_bug_is_rejected_and_reports_location():
    parent = np.full((2, 3), 1.0e11)
    first = parent * 0.4
    second = parent * 0.6
    second[1, 2] += 1.0e8

    report = check_scaled_additivity((first, second), parent)

    assert not report.passed
    assert report.failing_index == (1, 2)
    assert report.max_abs_error == pytest.approx(1.0e8)
    with pytest.raises(AssertionError, match=r"first failing index \(1, 2\)"):
        report.raise_if_invalid()


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must all match expected shape"):
        check_scaled_additivity((np.zeros((2, 2)), np.zeros((2, 1))), np.zeros((2, 2)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_rejected(bad):
    component = np.array([1.0, bad])
    with pytest.raises(ValueError, match="only finite numbers"):
        check_scaled_additivity((component,), np.ones(2))


def test_empty_component_inventory_is_rejected():
    with pytest.raises(ValueError, match="at least one component"):
        check_scaled_additivity((), np.ones(2))
