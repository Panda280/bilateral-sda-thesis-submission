"""Algebraic checks for the documented coefficient-parametrized T/w meaning."""

import numpy as np


def test_w_is_intermediate_share_and_remainder_is_va_share():
    z = np.array([[20.0, 10.0], [30.0, 50.0]])
    value_added = np.array([50.0, 40.0])
    output = z.sum(axis=0) + value_added

    a = z / output[None, :]
    w = a.sum(axis=0)

    np.testing.assert_allclose(w, z.sum(axis=0) / output)
    np.testing.assert_allclose(1.0 - w, value_added / output)


def test_t_normalizes_input_composition_and_reconstructs_ax():
    # Rows are input sectors after source-country aggregation; columns are buyers.
    a_x = np.array([[0.20, 0.00, 0.15], [0.30, 0.00, 0.45]])
    w = a_x.sum(axis=0)
    t = np.divide(a_x, w[None, :], out=np.zeros_like(a_x), where=w[None, :] != 0)

    np.testing.assert_allclose(t[:, w != 0].sum(axis=0), 1.0)
    np.testing.assert_allclose(t * w[None, :], a_x)
    np.testing.assert_array_equal(t[:, w == 0], 0.0)


def test_va_complement_requires_the_column_identity():
    z = np.array([[20.0], [30.0]])
    output = np.array([100.0])
    reported_value_added = np.array([45.0])  # Five units leave the column unbalanced.
    w = (z / output[None, :]).sum(axis=0)

    assert not np.allclose(1.0 - w, reported_value_added / output)
