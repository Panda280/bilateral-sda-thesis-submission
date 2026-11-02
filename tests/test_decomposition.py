import numpy as np

from sda import run_canonical_sda


def _inputs():
    eye = np.eye(2)
    # One country, two sectors; changes only final-demand level. Structural
    # nested components are explicit zero matrices pending the ADR decision.
    return dict(
        v0=np.array([1.0, 2.0]), v1=np.array([1.0, 2.0]),
        L0=eye, L1=eye, C0=eye, C1=eye, A_x0=eye, A_x1=eye,
        T_coef0=eye, T_coef1=eye, w_coef0=np.ones(2), w_coef1=np.ones(2),
        F0=np.ones((2, 1)), F1=np.ones((2, 1)),
        B0=np.ones((2, 1)), B1=np.ones((2, 1)),
        levels0=np.array([1.0]), levels1=np.array([2.0]),
        supplier_sector=np.array([0, 1]), sectors_per_country=2,
        emitter_labels=("e0", "e1"), structural_driver_labels=("d0", "d1"),
        consumer_labels=("c0",), demanded_sector_labels=("s0", "s1"),
        c_source_masks={
            "Domestic": np.ones((2, 2), dtype=bool),
            "Red": np.zeros((2, 2), dtype=bool),
            "Non-Red": np.zeros((2, 2), dtype=bool),
            "Non-Participant": np.zeros((2, 2), dtype=bool),
        },
        yf_source_masks={
            "Domestic": np.ones((2, 1), dtype=bool),
            "Red": np.zeros((2, 1), dtype=bool),
            "Non-Red": np.zeros((2, 1), dtype=bool),
            "Non-Participant": np.zeros((2, 1), dtype=bool),
        },
    )


def test_coordinator_builds_valid_complete_canonical_result():
    result = run_canonical_sda(**_inputs())
    assert result.validation.passed
    assert set(result.effects) == {"D_v", "D_C", "D_A_x", "D_T_coef", "D_w_coef", "D_YF", "D_YB", "D_y"}
    np.testing.assert_allclose(result.total_change, np.array([1.0, 2.0]))
