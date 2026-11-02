import numpy as np

from build_p2_granularity_tw_analysis import aggregate_and_diagnose, comparison_metrics, scalar_tw


def test_memory_bounded_aggregation_preserves_signed_total_and_detects_cancellation():
    fine = np.array([[3.0, -2.0], [-1.0, 1.0]])
    coarse, rows, columns, metrics = aggregate_and_diagnose(
        fine, [("A", "all"), ("A", "all")], [("A", "all"), ("A", "all")]
    )
    assert rows == (("A", "all"),) and columns == (("A", "all"),)
    assert coarse.shape == (1, 1)
    assert np.isclose(coarse.sum(), fine.sum())
    assert np.isclose(metrics["signed_conservation_error"], 0.0)
    assert metrics["cancellation_loss"] > 0
    assert metrics["sign_reversal_rate"] > 0


def test_harmonized_comparison_reports_rank_and_source_substitution():
    left = np.array([[2.0, 0.0], [0.0, -1.0]])
    same = comparison_metrics(left, left.copy())
    assert same["normalized_l1_native"] == 0.0
    assert same["source_substitution_total_variation"] == 0.0
    assert same["top_100_overlap"] == 1.0


def test_scalar_tw_zero_and_opposition_are_finite():
    assert scalar_tw(0.0, 0.0) == (0.0, 0.0, 0.0, "zero")
    t_share, w_share, cancellation, dominance = scalar_tw(1.0, -1.0)
    assert (t_share, w_share, cancellation, dominance) == (0.5, 0.5, 1.0, "balanced")
