import numpy as np
import pytest

from sda.analysis.tw import aggregate_tw, combine_transitions, summarize_tw


def test_summary_preserves_signed_parent_and_reports_cancellation():
    summary = summarize_tw([[6.0, 0.0]], [[-4.0, 0.0]])

    assert summary.parent_value == 2.0
    assert summary.t_absolute_share == pytest.approx(0.6)
    assert summary.w_absolute_share == pytest.approx(0.4)
    assert summary.cancellation_ratio == pytest.approx(0.8)
    assert summary.dominance == "T-dominant"


def test_zero_pair_has_defined_descriptive_metrics():
    summary = summarize_tw(np.zeros((2, 2)), np.zeros((2, 2)))
    assert summary.t_absolute_share == summary.w_absolute_share == 0.0
    assert summary.cancellation_ratio == 0.0
    assert summary.dominance == "zero"


def test_aggregation_supports_country_or_sector_labels_on_either_axis():
    t = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([[1.0, -1.0], [1.0, -2.0]])

    emitter_country = aggregate_tw(t, w, ["A", "B"], axis=0)
    driver_sector = aggregate_tw(t, w, ["goods", "services"], axis=1)

    assert emitter_country["A"].parent_value == 3.0
    assert emitter_country["B"].parent_value == 6.0
    assert driver_sector["goods"].parent_value == 6.0
    assert driver_sector["services"].parent_value == 3.0


def test_cumulative_transition_combination_is_cellwise_and_shape_checked():
    t, w = combine_transitions([
        (np.ones((2, 2)), np.full((2, 2), 2.0)),
        (np.full((2, 2), 3.0), np.full((2, 2), -1.0)),
    ])
    np.testing.assert_array_equal(t, np.full((2, 2), 4.0))
    np.testing.assert_array_equal(w, np.ones((2, 2)))

    with pytest.raises(ValueError, match="share one shape"):
        combine_transitions([(np.ones((2, 2)), np.ones((2, 2))),
                             (np.ones((1, 1)), np.ones((1, 1)))])


@pytest.mark.parametrize("threshold", [0.5, 1.01])
def test_invalid_dominance_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="dominance_threshold"):
        summarize_tw([[1.0]], [[1.0]], dominance_threshold=threshold)
