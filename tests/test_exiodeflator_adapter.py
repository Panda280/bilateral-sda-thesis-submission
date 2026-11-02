from types import SimpleNamespace

import numpy as np
import pytest

from sda import EXIODeflatorStateAdapter


INDEX = (("AA", "s0"), ("AA", "s1"), ("BB", "s0"), ("BB", "s1"))
GROUPS = {"AA": "Red", "BB": "Non-Participant"}


def _period(offset=0.0, samples=2):
    n, c, s = 4, 2, 2
    eye = np.eye(n)
    w = np.ones(n) + offset
    t = eye * (1.0 + offset)
    ax = t * w[None, :]

    def sampled(value):
        return np.stack([value, value + 0.1], axis=-1)

    f = np.full((n, c), 0.5 + offset)
    b_unique = np.array([[0.6, 0.4], [0.4, 0.6]]) + offset
    return SimpleNamespace(
        indexes_out=INDEX,
        v=np.stack([
            np.stack([np.arange(n) + 1.0 + offset, np.arange(n) + 11.0 + offset], axis=1),
            np.stack([np.arange(n) + 1.1 + offset, np.arange(n) + 11.1 + offset], axis=1),
        ], axis=-1),
        L=sampled(eye), C=sampled(eye),
        A_x=np.stack([ax, (t + 0.1) * (w + 0.1)[None, :]], axis=-1),
        T_coefficient=sampled(t),
        w_coefficient=np.stack([w, w + 0.1], axis=-1)[None, ...],
        F=sampled(f), B=sampled(np.tile(b_unique, (c, 1))),
        y_b=np.stack([np.array([10.0, 20.0]) + offset,
                      np.array([10.1, 20.1]) + offset], axis=-1)[:, None, :],
    )


def test_adapts_exact_ty_layout_and_period_orientation():
    previous, current = _period(0.0), _period(1.0)
    state = EXIODeflatorStateAdapter(
        previous, current, GROUPS, sample_index=0, impact_index=1
    ).adapt()

    np.testing.assert_allclose(state.v0, [11, 12, 13, 14])
    np.testing.assert_allclose(state.v1, [12, 13, 14, 15])
    np.testing.assert_allclose(state.B0, [[0.6, 0.4], [0.4, 0.6]])
    np.testing.assert_allclose(state.levels1, [11, 21])
    assert state.labels.emitters == INDEX
    assert state.labels.consumers == ("AA", "BB")
    assert state.labels.demanded_sectors == ("s0", "s1")
    assert state.source_groups.C["Domestic"][0, 1]
    assert state.source_groups.C["Red"][0, 2]
    assert state.source_groups.YF["Non-Participant"][2, 0]


def test_selects_trailing_sample_consistently():
    state = EXIODeflatorStateAdapter(
        _period(), _period(), GROUPS, sample_index=1
    ).adapt()
    np.testing.assert_allclose(state.L0, np.eye(4) + 0.1)
    np.testing.assert_allclose(state.w_coef0, np.full(4, 1.1))
    np.testing.assert_allclose(state.levels0, [10.1, 20.1])


def test_rejects_misaligned_periods_and_non_repeated_b():
    previous, current = _period(), _period()
    current.indexes_out = tuple(reversed(INDEX))
    with pytest.raises(ValueError, match="order must match"):
        EXIODeflatorStateAdapter(previous, current, GROUPS).adapt()

    previous, current = _period(), _period()
    current.B[2, 0, 0] += 1
    with pytest.raises(ValueError, match="blocks must be identical"):
        EXIODeflatorStateAdapter(previous, current, GROUPS).adapt()


def test_requires_complete_source_classification():
    with pytest.raises(ValueError, match="missing source-origin classification"):
        EXIODeflatorStateAdapter(_period(), _period(), {"AA": "Red"}).adapt()
