import numpy as np
import pytest

from sda.attribution import attribute_structural_driver
from sda.effects import two_factor_structural_components
from sda.effects import nested_technical_components
from sda.validation import assert_additive


def test_two_factor_components_reconstruct_delta_A_cellwise():
    C0 = np.array([[0.8, 0.2], [0.3, 0.7]])
    C1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    A_x0 = np.array([[0.4, 0.2], [0.1, 0.5]])
    A_x1 = np.array([[0.5, 0.25], [0.2, 0.45]])

    components = two_factor_structural_components(C0, C1, A_x0, A_x1)
    assert_additive([components.C, components.A_x], C1 * A_x1 - C0 * A_x0)


def test_driver_preserving_kernel_matches_reference_endpoint_total():
    component = np.array([[0.02, -0.01], [0.03, 0.04]])
    v0 = np.array([0.7, 1.1])
    dv = np.array([0.05, -0.02])
    L0 = np.array([[1.2, 0.1], [0.2, 1.3]])
    L1 = np.array([[1.25, 0.12], [0.18, 1.28]])
    y0 = np.array([5.0, 3.0])
    dy = np.array([0.4, -0.2])

    bilateral = attribute_structural_driver(
        component, v0=v0, delta_v=dv, L0=L0, L1=L1, y0=y0, delta_y=dy
    )

    V0 = np.diag(v0)
    dV = np.diag(dv)
    base_endpoint = V0 @ L0 @ component @ L1
    change_endpoint = dV @ L0 @ component @ L1
    reference_endpoint = (
        base_endpoint * y0[None, :]
        + 0.5 * base_endpoint * dy[None, :]
        + 0.5 * change_endpoint * y0[None, :]
        + (1.0 / 3.0) * change_endpoint * dy[None, :]
    )

    np.testing.assert_allclose(bilateral.sum(axis=1), reference_endpoint.sum(axis=1))


def test_single_changed_structural_column_stays_in_that_driver_column():
    component = np.array([[0.0, 0.2], [0.0, -0.1]])
    result = attribute_structural_driver(
        component,
        v0=np.array([1.0, 2.0]),
        delta_v=np.zeros(2),
        L0=np.eye(2),
        L1=np.eye(2),
        y0=np.ones(2),
        delta_y=np.zeros(2),
    )
    np.testing.assert_array_equal(result[:, 0], np.zeros(2))
    assert np.any(result[:, 1] != 0.0)


def test_structural_kernel_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching the matrix size"):
        attribute_structural_driver(
            np.eye(2), v0=np.ones(3), delta_v=np.ones(2), L0=np.eye(2), L1=np.eye(2),
            y0=np.ones(2), delta_y=np.ones(2)
        )


def test_nested_technical_components_reconstruct_canonical_a_x_parent():
    C0 = np.array([[0.7, 0.2], [0.3, 0.8]])
    C1 = np.array([[0.6, 0.3], [0.4, 0.7]])
    T0 = np.array([[0.4, 0.2], [0.1, 0.5]])
    T1 = np.array([[0.5, 0.25], [0.2, 0.45]])
    w0 = np.array([0.8, 0.6])
    w1 = np.array([0.9, 0.7])
    A_x0 = T0 * w0[None, :]
    A_x1 = T1 * w1[None, :]

    parent = two_factor_structural_components(C0, C1, A_x0, A_x1).A_x
    nested = nested_technical_components(C0, C1, T0, T1, w0, w1)
    assert_additive([nested.T_coef, nested.w_coef], parent)


def test_nested_triple_interaction_is_half_c_and_quarter_each_child():
    zero_matrix = np.zeros((1, 1))
    one_matrix = np.ones((1, 1))
    nested = nested_technical_components(
        zero_matrix, one_matrix, zero_matrix, one_matrix, np.zeros(1), np.ones(1)
    )
    top = two_factor_structural_components(zero_matrix, one_matrix, zero_matrix, one_matrix)
    np.testing.assert_allclose(top.C, [[0.5]])
    np.testing.assert_allclose(nested.T_coef, [[0.25]])
    np.testing.assert_allclose(nested.w_coef, [[0.25]])
    assert_additive([nested.T_coef, nested.w_coef], top.A_x)
