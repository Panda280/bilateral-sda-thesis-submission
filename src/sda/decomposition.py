"""Canonical, validation-gated bilateral SDA orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .attribution.final_demand import (
    attribute_final_demand_composition,
    attribute_final_demand_level,
)
from .attribution.structural import attribute_structural_driver
from .effects.final_demand import split_yf_component_by_source, three_factor_final_demand_components
from .effects.intensity import refined_laspeyres_intensity_effect
from .effects.structural import (
    nested_technical_components,
    split_c_component_by_source,
    two_factor_structural_components,
)
from .model import EffectDimensions, EffectResult, LabelSet, SDAResult
from .state import A_X_IDENTITY_ATOL, A_X_IDENTITY_RTOL, SDAState
from .validation import build_mandatory_validation_report


def run_canonical_sda(
    *,
    v0: ArrayLike, v1: ArrayLike, L0: ArrayLike, L1: ArrayLike,
    C0: ArrayLike, C1: ArrayLike, A_x0: ArrayLike, A_x1: ArrayLike,
    T_coef0: ArrayLike, T_coef1: ArrayLike, w_coef0: ArrayLike, w_coef1: ArrayLike,
    F0: ArrayLike, F1: ArrayLike, B0: ArrayLike, B1: ArrayLike,
    levels0: ArrayLike, levels1: ArrayLike, supplier_sector: ArrayLike,
    sectors_per_country: int,
    emitter_labels: Sequence[object], structural_driver_labels: Sequence[object],
    consumer_labels: Sequence[object], demanded_sector_labels: Sequence[object],
    c_source_masks: Mapping[str, ArrayLike], yf_source_masks: Mapping[str, ArrayLike],
) -> SDAResult:
    """Assemble the canonical result from existing kernels and enforce every gate."""

    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in {
        "v0": v0, "v1": v1, "L0": L0, "L1": L1,
        "F0": F0, "F1": F1, "B0": B0, "B1": B1,
        "levels0": levels0, "levels1": levels1,
    }.items()}
    sector = np.asarray(supplier_sector)
    y0_matrix = arrays["F0"] * arrays["B0"][sector, :] * arrays["levels0"][None, :]
    y1_matrix = arrays["F1"] * arrays["B1"][sector, :] * arrays["levels1"][None, :]
    y0 = y0_matrix.sum(axis=1)
    y1 = y1_matrix.sum(axis=1)
    delta_v = arrays["v1"] - arrays["v0"]
    delta_y = y1 - y0

    structural = two_factor_structural_components(C0, C1, A_x0, A_x1)
    nested = nested_technical_components(C0, C1, T_coef0, T_coef1, w_coef0, w_coef1)
    # Same "A_x = T_coef * w_coef" identity SDAState.__post_init__ already
    # checked at construction time -- re-checked here in case a caller builds
    # arrays directly rather than through SDAState. See A_X_IDENTITY_RTOL's
    # definition in state.py for why this tolerance is looser than the
    # rtol=1e-10/atol=1e-12 used elsewhere: real (float32-sourced) A_x/T_coef/
    # w_coef only satisfy this identity to float32 precision, not float64's.
    np.testing.assert_allclose(
        np.asarray(A_x0, dtype=np.float64),
        np.asarray(T_coef0, dtype=np.float64) * np.asarray(w_coef0, dtype=np.float64)[None, :],
        rtol=A_X_IDENTITY_RTOL, atol=A_X_IDENTITY_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(A_x1, dtype=np.float64),
        np.asarray(T_coef1, dtype=np.float64) * np.asarray(w_coef1, dtype=np.float64)[None, :],
        rtol=A_X_IDENTITY_RTOL, atol=A_X_IDENTITY_ATOL,
    )
    # Derived quantities (delta-based averaging over both periods), so the
    # same float32-precision gap can compound slightly further here than in
    # the two single-period checks above -- same tolerance, empirically still
    # comfortably wide enough (see state.py); if it ever isn't, this is
    # exactly the kind of thing that should fail loudly rather than pass
    # silently.
    np.testing.assert_allclose(
        nested.T_coef + nested.w_coef, structural.A_x,
        rtol=A_X_IDENTITY_RTOL, atol=A_X_IDENTITY_ATOL,
    )
    final_demand = three_factor_final_demand_components(
        arrays["F0"], arrays["F1"], arrays["B0"], arrays["B1"],
        arrays["levels0"], arrays["levels1"], sector,
    )
    propagation = dict(v0=arrays["v0"], delta_v=delta_v, L0=arrays["L0"], L1=arrays["L1"])
    structural_propagation = dict(**propagation, y0=y0, delta_y=delta_y)

    d_c = attribute_structural_driver(structural.C, **structural_propagation)
    d_ax = attribute_structural_driver(structural.A_x, **structural_propagation)
    d_t = attribute_structural_driver(nested.T_coef, **structural_propagation)
    d_w = attribute_structural_driver(nested.w_coef, **structural_propagation)
    d_yf = attribute_final_demand_composition(final_demand.YF, **propagation, sectors_per_country=sectors_per_country)
    d_yb = attribute_final_demand_composition(final_demand.YB, **propagation, sectors_per_country=sectors_per_country)
    d_y = attribute_final_demand_level(final_demand.y, **propagation, sectors_per_country=sectors_per_country)

    emitter = LabelSet("emitter country-sector", tuple(emitter_labels))
    structural_driver = LabelSet("structural driver country-sector", tuple(structural_driver_labels))
    composition_driver = LabelSet(
        "consumer country and demanded sector",
        tuple((consumer, sector_label) for consumer in consumer_labels for sector_label in demanded_sector_labels),
    )
    consumer = LabelSet("consumer country", tuple(consumer_labels))
    structural_dimensions = EffectDimensions(emitter, structural_driver)
    composition_dimensions = EffectDimensions(emitter, composition_driver)

    c_groups = {
        name: EffectResult("D_C", attribute_structural_driver(component, **structural_propagation), structural_dimensions)
        for name, component in split_c_component_by_source(C0, C1, A_x0, A_x1, c_source_masks).items()
    }
    yf_groups = {
        name: EffectResult(
            "D_YF",
            attribute_final_demand_composition(component, **propagation, sectors_per_country=sectors_per_country),
            composition_dimensions,
        )
        for name, component in split_yf_component_by_source(final_demand.YF, yf_source_masks).items()
    }
    effects = {
        "D_v": EffectResult("D_v", refined_laspeyres_intensity_effect(arrays["v0"], arrays["v1"], arrays["L0"], arrays["L1"], y0, y1), EffectDimensions(emitter)),
        "D_C": EffectResult("D_C", d_c, structural_dimensions, source_categories=c_groups),
        "D_A_x": EffectResult("D_A_x", d_ax, structural_dimensions),
        "D_T_coef": EffectResult("D_T_coef", d_t, structural_dimensions),
        "D_w_coef": EffectResult("D_w_coef", d_w, structural_dimensions),
        "D_YF": EffectResult("D_YF", d_yf, composition_dimensions, source_categories=yf_groups),
        "D_YB": EffectResult("D_YB", d_yb, composition_dimensions),
        "D_y": EffectResult("D_y", d_y, EffectDimensions(emitter, consumer)),
    }
    emitter_totals = {
        name: effect.baseline if effect.dimensions.column is None else effect.baseline.sum(axis=1)
        for name, effect in effects.items()
    }
    total_change = arrays["v1"] * (arrays["L1"] @ y1) - arrays["v0"] * (arrays["L0"] @ y0)
    report = build_mandatory_validation_report(
        effects=effects,
        structural_parts=(structural.C, structural.A_x),
        structural_change=np.asarray(C1) * np.asarray(A_x1) - np.asarray(C0) * np.asarray(A_x0),
        final_demand_parts=(final_demand.YF, final_demand.YB, final_demand.y),
        final_demand_change=y1_matrix - y0_matrix,
        emitter_totals=emitter_totals,
        total_change=total_change,
    )
    return SDAResult(effects=effects, total_change=total_change, validation=report)


def run_canonical_sda_state(state: SDAState) -> SDAResult:
    """Run the canonical coordinator from its validated input contract."""

    if not isinstance(state, SDAState):
        raise TypeError("state must be an SDAState")
    return run_canonical_sda(**state.coordinator_arguments())
