"""CPU-only adapter for already-computed EXIODeflator ``TY`` period objects.

This module deliberately does not import EXIODeflator or CuPy. Objects are
accepted at the completed ``TY`` boundary and copied to NumPy through ``get``
when their arrays are device-backed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..attribution.groups import build_source_origin_masks
from ..state import SDAState, SourceGroupInputs, StateLabels


def _cpu(value: Any) -> np.ndarray:
    getter = getattr(value, "get", None)
    return np.asarray(getter() if callable(getter) else value)


def _sample(value: Any, base_shape: tuple[int, ...], sample_index: int, name: str) -> np.ndarray:
    array = _cpu(value)
    if array.shape == base_shape:
        if sample_index:
            raise ValueError(f"{name} has no sample axis; sample_index must be 0")
        return array
    if array.ndim == len(base_shape) + 1 and array.shape[:-1] == base_shape:
        if not 0 <= sample_index < array.shape[-1]:
            raise ValueError(f"sample_index {sample_index} is out of range for {name}")
        return array[..., sample_index]
    raise ValueError(f"{name} must have shape {base_shape} or {base_shape}+(samples,), got {array.shape}")


def _index(period: Any) -> tuple[tuple[object, object], ...]:
    raw = getattr(period, "indexes_out", None)
    if raw is None:
        raise ValueError("EXIODeflator period is missing indexes_out")
    values = tuple(tuple(item) for item in raw)
    if not values or any(len(item) < 2 for item in values):
        raise ValueError("indexes_out must contain ordered (country, sector) pairs")
    return tuple((item[0], item[1]) for item in values)


@dataclass(frozen=True)
class EXIODeflatorStateAdapter:
    """Convert previous CYP and current PYP ``TY`` objects into one state.

    ``sample_index`` selects the reference object's trailing Monte Carlo axis;
    ``impact_index`` selects its middle intensity axis. Source membership must
    use canonical external names: ``Red``, ``Non-Red``, or
    ``Non-Participant``. Domestic cells override that membership.
    """

    previous: Any
    current: Any
    source_group_by_country: Mapping[str, str]
    sample_index: int = 0
    impact_index: int = 0

    def adapt(self) -> SDAState:
        previous_index = _index(self.previous)
        current_index = _index(self.current)
        if previous_index != current_index:
            raise ValueError("previous and current indexes_out order must match exactly")

        countries = tuple(dict.fromkeys(item[0] for item in current_index))
        sectors = tuple(dict.fromkeys(item[1] for item in current_index))
        expected = tuple((country, sector) for country in countries for sector in sectors)
        if current_index != expected:
            raise ValueError("indexes_out must be a complete country-major country-sector grid")
        n, c, s = len(current_index), len(countries), len(sectors)

        labels = StateLabels(
            emitters=current_index,
            structural_drivers=current_index,
            consumers=countries,
            demanded_sectors=sectors,
        )
        source_countries = tuple(item[0] for item in current_index)
        c_groups = build_source_origin_masks(
            source_countries, source_countries, self.source_group_by_country
        )
        yf_groups = build_source_origin_masks(
            source_countries, countries, self.source_group_by_country
        )

        def matrix(period: Any, name: str) -> np.ndarray:
            return _sample(getattr(period, name), (n, n), self.sample_index, name)

        def final(period: Any, name: str) -> np.ndarray:
            return _sample(getattr(period, name), (n, c), self.sample_index, name)

        def vector(period: Any, name: str) -> np.ndarray:
            raw = _cpu(getattr(period, name))
            # EXIODeflator stores w as (1, n, samples).
            if raw.ndim >= 2 and raw.shape[0] == 1:
                raw = raw[0]
            return _sample(raw, (n,), self.sample_index, name)

        def intensity(period: Any) -> np.ndarray:
            raw = _cpu(getattr(period, "v"))
            if raw.ndim == 1:
                if self.impact_index or self.sample_index:
                    raise ValueError("v has neither impact nor sample axes; both indexes must be 0")
                return raw
            if raw.ndim == 2:  # (emitter, impact), no sample axis
                if self.sample_index or not 0 <= self.impact_index < raw.shape[1]:
                    raise ValueError("requested impact/sample index is out of range for v")
                return raw[:, self.impact_index]
            if raw.ndim == 3 and raw.shape[0] == n:  # (emitter, impact, sample)
                if not (0 <= self.impact_index < raw.shape[1] and 0 <= self.sample_index < raw.shape[2]):
                    raise ValueError("requested impact/sample index is out of range for v")
                return raw[:, self.impact_index, self.sample_index]
            raise ValueError(f"v must be (emitter[, impact[, sample]]), got {raw.shape}")

        def demand_b(period: Any) -> np.ndarray:
            # B is repeated over source-country row blocks; only its first
            # demanded-sector block is semantically unique.
            full = final(period, "B")
            first = full[:s]
            if not np.allclose(full, np.tile(first, (c, 1)), rtol=1e-10, atol=1e-12):
                raise ValueError("B source-country blocks must be identical")
            return first

        def levels(period: Any) -> np.ndarray:
            raw = _cpu(getattr(period, "y_b"))
            if raw.ndim >= 2 and raw.shape[1] == 1:
                raw = raw[:, 0, ...]
            return _sample(raw, (c,), self.sample_index, "y_b")

        p0, p1 = self.previous, self.current
        return SDAState(
            v0=intensity(p0), v1=intensity(p1),
            L0=matrix(p0, "L"), L1=matrix(p1, "L"),
            C0=matrix(p0, "C"), C1=matrix(p1, "C"),
            A_x0=matrix(p0, "A_x"), A_x1=matrix(p1, "A_x"),
            T_coef0=matrix(p0, "T_coefficient"), T_coef1=matrix(p1, "T_coefficient"),
            w_coef0=vector(p0, "w_coefficient"), w_coef1=vector(p1, "w_coefficient"),
            F0=final(p0, "F"), F1=final(p1, "F"),
            B0=demand_b(p0), B1=demand_b(p1),
            levels0=levels(p0), levels1=levels(p1),
            supplier_sector=np.tile(np.arange(s, dtype=np.int64), c),
            sectors_per_country=s,
            labels=labels,
            source_groups=SourceGroupInputs(C=c_groups, YF=yf_groups),
        )
