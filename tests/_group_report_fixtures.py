"""Shared synthetic multi-country fixture for group-sector report tests.

Reuses the same construction pattern as ``tests/test_exiodeflator_adapter.py``
(a ``SimpleNamespace`` "TY period" object fed through
:class:`sda.EXIODeflatorStateAdapter`) rather than hand-rolling a bespoke
``SDAResult``. Going through the real adapter and
:func:`sda.run_canonical_sda_state` means every effect -- including the
``D_C``/``D_YF`` source-origin split -- is genuinely computed and validated,
not asserted into existence, and ``L0``/``L1`` are built as true Leontief
inverses of ``A0 = C0 * A_x0`` and ``A1 = C1 * A_x1`` so the mandatory
``emitter_reconciliation`` gate is satisfiable exactly rather than by luck.

Three countries (``A``: Red, ``B``: Non-Red, ``C``: Non-Participant), two
sectors each, so the classification used for group-sector resolution
(:data:`GROUPS`) exercises all three external source-origin groups, and a
small off-diagonal term on the technical coefficients gives every one of the
four Domestic/Red/Non-Red/Non-Participant ``D_C``/``D_YF`` splits a nonzero
total -- not just Domestic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sda import EXIODeflatorStateAdapter, SDAResult, run_canonical_sda_state

INDEX = (
    ("A", "s0"), ("A", "s1"),
    ("B", "s0"), ("B", "s1"),
    ("C", "s0"), ("C", "s1"),
)
GROUPS = {"A": "Red", "B": "Non-Red", "C": "Non-Participant"}

_N, _C, _S = 6, 3, 2
_EYE = np.eye(_N)
_OFF_MASK = np.ones((_N, _N)) - _EYE
_OFF = 0.02


def _period(delta: float) -> SimpleNamespace:
    """Build one EXIODeflator-shaped ``TY`` period, offset by ``delta``.

    ``L`` is derived as the true Leontief inverse of ``A = C * A_x`` so the
    standard perturbation identity ``L1 - L0 == L0 @ (A1 - A0) @ L1`` holds
    exactly -- required for the coordinator's ``emitter_reconciliation``
    gate, which is otherwise unsatisfiable by an arbitrarily chosen ``L``.
    """
    t = _EYE * (1.0 + delta) + _OFF * (1.0 + 0.5 * delta) * _OFF_MASK
    w = np.full(_N, 1.0 + delta)
    a_x = t * w[None, :]
    c_mat = _EYE * (0.5 + 0.05 * delta) + _OFF * (1.0 + 0.3 * delta) * _OFF_MASK
    a = c_mat * a_x
    L = np.linalg.inv(_EYE - a)
    v = np.arange(1.0, _N + 1.0) + delta
    f = np.full((_N, _C), 0.5 + 0.1 * delta)
    b_unique = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]) + delta * 0.02
    b_full = np.tile(b_unique, (_C, 1))
    levels = np.array([10.0, 20.0, 30.0]) + delta
    return SimpleNamespace(
        indexes_out=INDEX, v=v, L=L, C=c_mat, A_x=a_x,
        T_coefficient=t, w_coefficient=w, F=f, B=b_full, y_b=levels,
    )


def build_synthetic_result() -> SDAResult:
    """Build one fully validated, multi-country, source-split synthetic run."""
    previous, current = _period(0.0), _period(1.0)
    state = EXIODeflatorStateAdapter(
        previous, current, GROUPS, sample_index=0, impact_index=0
    ).adapt()
    return run_canonical_sda_state(state)
