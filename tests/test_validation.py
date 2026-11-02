import numpy as np
import pytest

from sda import EffectDimensions, EffectResult, LabelSet
from sda.hierarchy import CANONICAL_EFFECTS
from sda.validation import MANDATORY_CHECKS, assert_additive, assert_driver_sum, build_mandatory_validation_report


def test_nested_structural_identity_is_checked_cellwise():
    technology = np.array([[1.0, 2.0], [3.0, 4.0]])
    intensity = np.array([[0.5, -0.5], [1.0, 0.0]])
    assert_additive([technology, intensity], technology + intensity)


def test_driver_sum_reconstructs_emitter_total():
    attributed = np.array([[1.0, 2.0], [-0.5, 0.25]])
    assert_driver_sum(attributed, np.array([3.0, -0.25]))


def test_driver_sum_rejects_wrong_axis_or_total():
    with pytest.raises(AssertionError):
        assert_driver_sum(np.eye(2), np.array([2.0, 1.0]))
    with pytest.raises(ValueError, match="two-dimensional"):
        assert_driver_sum(np.zeros((1, 2, 2)), np.zeros(1))


def _effect(name, value, *, sources=False):
    dimensions = EffectDimensions(LabelSet("emitter", ("e",)), None if name == "D_v" else LabelSet("driver", ("d",)))
    categories = None
    if sources:
        categories = {
            group: EffectResult(name, np.array([[value if group == "Domestic" else 0.0]]), dimensions)
            for group in ("Domestic", "Red", "Non-Red", "Non-Participant")
        }
    data = np.array([value]) if name == "D_v" else np.array([[value]])
    return EffectResult(name, data, dimensions, source_categories=categories)


def test_mandatory_report_passes_only_complete_reconciliation():
    values = {name: 1.0 for name in CANONICAL_EFFECTS}
    values["D_A_x"] = 2.0
    effects = {name: _effect(name, value, sources=name in {"D_C", "D_YF"}) for name, value in values.items()}
    report = build_mandatory_validation_report(
        effects=effects,
        structural_parts=(np.array([[1.0]]), np.array([[2.0]])),
        structural_change=np.array([[3.0]]),
        final_demand_parts=(np.array([[1.0]]),) * 3,
        final_demand_change=np.array([[3.0]]),
        emitter_totals={name: np.array([value]) for name, value in values.items()},
        total_change=np.array([7.0]),
    )
    assert report.passed
    assert set(report.checks) == set(MANDATORY_CHECKS)


def test_mandatory_report_marks_missing_checks_as_failures():
    report = build_mandatory_validation_report(
        effects={}, structural_parts=None, structural_change=None,
        final_demand_parts=None, final_demand_change=None,
        emitter_totals=None, total_change=None,
    )
    assert not report.passed
    assert not any(report.checks.values())
    with pytest.raises(ValueError, match="SDA validation failed"):
        report.raise_if_invalid()
