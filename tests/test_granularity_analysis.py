import numpy as np
import pytest

from sda.analysis.granularity import (
    concentration,
    controlled_aggregate,
    diagnose_aggregation,
    spearman_rank_correlation,
    top_k_overlap,
)


def test_controlled_aggregation_preserves_signed_total_and_label_order():
    fine = np.array([[3.0, -1.0, 2.0], [-2.0, 4.0, -6.0], [1.0, 2.0, 3.0]])
    result = controlled_aggregate(fine, ["A", "A", "B"], ["x", "y", "y"])

    assert result.row_labels == ("A", "B")
    assert result.column_labels == ("x", "y")
    np.testing.assert_array_equal(result.values, [[1.0, -1.0], [1.0, 5.0]])
    assert result.values.sum() == fine.sum()


def test_diagnostics_distinguish_cancellation_reversal_and_edge_loss():
    fine = np.array([[3.0, -1.0], [-3.0, 1.0], [1.0, -4.0]])
    diagnostics = diagnose_aggregation(fine, ["A", "A", "B"], ["all", "all"])

    assert diagnostics.signed_conservation_error == 0.0
    assert diagnostics.fine_absolute_total == 13.0
    assert diagnostics.coarse_absolute_total == 3.0
    assert diagnostics.cancellation_loss == pytest.approx(10 / 13)
    assert diagnostics.sign_reversal_rate == pytest.approx(1 / 6)
    assert diagnostics.magnitude_weighted_sign_reversal_rate == pytest.approx(1 / 13)
    assert diagnostics.sparse_edge_loss_rate == pytest.approx(4 / 6)


def test_concentration_is_based_on_absolute_magnitude():
    hhi, entropy = concentration([[-1.0, 1.0]])
    assert hhi == pytest.approx(0.5)
    assert entropy == pytest.approx(np.log(2.0))


def test_rank_and_top_k_diagnostics_are_dependency_free_and_tie_aware():
    assert spearman_rank_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_rank_correlation([1, 1, 1], [3, 2, 1]) == 0.0
    assert top_k_overlap([9, 8, 1, 0], [9, 0, 8, 1], 2) == pytest.approx(0.5)


def test_invalid_dimensions_and_k_are_rejected():
    with pytest.raises(ValueError, match="row_groups"):
        controlled_aggregate(np.ones((2, 2)), ["A"], ["x", "y"])
    with pytest.raises(ValueError, match="k must"):
        top_k_overlap([1, 2], [1, 2], 0)
