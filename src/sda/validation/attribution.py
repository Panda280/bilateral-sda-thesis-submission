"""CPU-first numerical validation primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..hierarchy import CANONICAL_EFFECTS
from ..model import EffectResult, ValidationReport
from ..attribution.groups import SOURCE_ORIGIN_GROUPS

MANDATORY_CHECKS = (
    "effect_inventory",
    "structural_algebra",
    "final_demand_algebra",
    "nested_structural_additivity",
    "c_source_additivity",
    "yf_source_additivity",
    "driver_sums",
    "emitter_reconciliation",
)

# Tolerance for build_mandatory_validation_report's gates. Looser than
# assert_additive/assert_driver_sum's own rtol=1e-10/atol=1e-12 defaults
# (kept as-is below -- both are exercised only against synthetic, genuinely
# float64 fixtures in tests/, where that tight tolerance is correct and
# worth keeping). The mandatory gates below run against whatever
# run_canonical_sda is actually given, which for a real EXIODeflator-sourced
# run means values built from float32 GPU arithmetic (see state.py's
# A_X_IDENTITY_RTOL/ATOL for the same root cause one stage earlier in the
# pipeline) then summed/differenced across ~2e6-cell arrays with real-world
# magnitudes reaching 1e10-1e11 (CO2 mass units).
#
# A single flat atol does not work across these gates: some (e.g.
# emitter_reconciliation) subtract large, similarly-sized float32-quantized
# quantities, so their absolute noise floor scales with the array's own
# magnitude even at cells whose true expected value is small or zero
# (catastrophic cancellation) -- an atol sized for one gate's magnitude
# under- or over-shoots another's. _scaled_atol below follows the same
# pattern EXIODeflator's own real pipeline already uses for exactly this
# problem (see its "materiality_tol=max(eps, 32 * dtype_float_eps *
# max(1, max(abs(Y))))" diagnostic in matrix_calculations.py): an absolute
# floor proportional to one float32 ULP times the array's own peak
# magnitude, with a safety-factor multiplier for headroom.
#
# Measured on the first real published run (2003-2004, WIOT2013,
# 2026-08-27) before this change, with a flat atol=1e-4: nested_structural_
# additivity failed with abs error 43276 against a ~6.45e10-magnitude D_A_x
# (relative ~6.7e-7); c_source_additivity abs error 7.6e-6 at a near-zero-
# expected cell (passed under a flat 1e-4 atol, but would not have scaled
# safely to a smaller-magnitude run); emitter_reconciliation abs error
# 944558 against a ~2.09e11-magnitude total_change (relative ~4.5e-6) --
# all consistent with a handful of accumulated float32 ULPs relative to
# each check's own scale, not a systematic divergence (which would show up
# as a much larger, non-ULP-shaped relative error). With _scaled_atol below
# (safety factor 256), the scaled floor for those two failing gates is
# ~1.97e6 and ~6.37e6 respectively -- ~45x and ~6.7x headroom over what was
# actually observed, while remaining a tiny fraction (~3e-5) of each gate's
# own magnitude, far tighter than would be needed to hide a real
# reconciliation bug.
MANDATORY_CHECK_RTOL = 1e-5
_FLOAT32_EPS = 1.1920929e-7
_MANDATORY_CHECK_ATOL_SAFETY_FACTOR = 256


def _scaled_atol(expected: ArrayLike) -> float:
    """Absolute-error floor proportional to one float32 ULP at ``expected``'s
    own peak magnitude -- see MANDATORY_CHECK_RTOL's definition above."""

    array = np.asarray(expected, dtype=np.float64)
    magnitude = float(np.max(np.abs(array), initial=0.0)) if array.size else 0.0
    return _MANDATORY_CHECK_ATOL_SAFETY_FACTOR * _FLOAT32_EPS * max(1.0, magnitude)


@dataclass(frozen=True)
class ScaleAwareAdditivityReport:
    """Diagnostics for an elementwise split-to-parent reconciliation."""

    max_abs_error: float
    failing_index: tuple[int, ...] | None
    rtol: float
    atol: float

    @property
    def passed(self) -> bool:
        return self.failing_index is None

    def raise_if_invalid(self) -> None:
        if not self.passed:
            raise AssertionError(
                "scale-aware additivity failed: "
                f"max absolute error {self.max_abs_error:.12g}; "
                f"first failing index {self.failing_index}; "
                f"rtol={self.rtol:.12g}, atol={self.atol:.12g}"
            )


def check_scaled_additivity(
    parts: Iterable[ArrayLike],
    expected: ArrayLike,
    *,
    rtol: float = MANDATORY_CHECK_RTOL,
) -> ScaleAwareAdditivityReport:
    """Check an elementwise split using the real-float32 tolerance policy.

    This is intended for derived cubes whose categories must sum back to a
    canonical parent. Shape and finiteness failures are rejected as invalid
    inputs rather than being reported as numerical reconciliation failures.
    """

    parent = np.asarray(expected, dtype=np.float64)
    if not np.isfinite(parent).all():
        raise ValueError("components and expected values must contain only finite numbers")

    total = np.zeros_like(parent)
    count = 0
    shapes = []
    for part in parts:
        value = np.asarray(part, dtype=np.float64)
        count += 1
        shapes.append(value.shape)
        if value.shape != parent.shape:
            raise ValueError(f"component shapes {shapes} must all match expected shape {parent.shape}")
        if not np.isfinite(value).all():
            raise ValueError("components and expected values must contain only finite numbers")
        np.add(total, value, out=total)
    if count == 0:
        raise ValueError("at least one component is required")
    absolute_error = np.abs(total - parent)
    max_error = float(np.max(absolute_error, initial=0.0))
    atol = _scaled_atol(parent)
    failures = absolute_error > (atol + rtol * np.abs(parent))
    first_failure = np.argwhere(failures)
    failing_index = tuple(int(item) for item in first_failure[0]) if first_failure.size else None
    return ScaleAwareAdditivityReport(max_error, failing_index, rtol, atol)


def assert_scaled_additive(
    parts: Iterable[ArrayLike],
    expected: ArrayLike,
    *,
    rtol: float = MANDATORY_CHECK_RTOL,
) -> ScaleAwareAdditivityReport:
    """Return reconciliation diagnostics, raising on any failing element."""

    report = check_scaled_additivity(parts, expected, rtol=rtol)
    report.raise_if_invalid()
    return report


def assert_additive(
    parts: Iterable[ArrayLike],
    expected: ArrayLike,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    arrays = [np.asarray(part, dtype=np.float64) for part in parts]
    if not arrays:
        raise ValueError("at least one component is required")
    total = np.sum(np.stack(arrays), axis=0)
    np.testing.assert_allclose(total, np.asarray(expected, dtype=np.float64), rtol=rtol, atol=atol)


def assert_driver_sum(
    attributed: ArrayLike,
    emitter_total: ArrayLike,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    matrix = np.asarray(attributed, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("driver-attributed effect must be two-dimensional")
    np.testing.assert_allclose(
        matrix.sum(axis=1), np.asarray(emitter_total, dtype=np.float64), rtol=rtol, atol=atol
    )


def _close(actual: ArrayLike, expected: ArrayLike, *, rtol: float, atol: float) -> tuple[bool, float]:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return False, float("inf")
    error = float(np.max(np.abs(left - right), initial=0.0))
    return bool(np.allclose(left, right, rtol=rtol, atol=atol)), error


def build_mandatory_validation_report(
    *,
    effects: Mapping[str, EffectResult],
    structural_parts: Iterable[ArrayLike] | None,
    structural_change: ArrayLike | None,
    final_demand_parts: Iterable[ArrayLike] | None,
    final_demand_change: ArrayLike | None,
    emitter_totals: Mapping[str, ArrayLike] | None,
    total_change: ArrayLike | None,
    rtol: float = MANDATORY_CHECK_RTOL,
) -> ValidationReport:
    """Run every canonical gate; unavailable inputs are failures, never skips."""

    checks = {name: False for name in MANDATORY_CHECKS}
    errors: dict[str, float] = {}
    details: dict[str, str] = {}

    checks["effect_inventory"] = set(effects) == set(CANONICAL_EFFECTS)

    def additive_gate(name: str, parts: Iterable[ArrayLike] | None, expected: ArrayLike | None) -> None:
        if parts is None or expected is None:
            details[name] = "required validation input missing"
            return
        values = [np.asarray(part, dtype=np.float64) for part in parts]
        if not values:
            details[name] = "required components missing"
            return
        checks[name], errors[name] = _close(
            np.sum(np.stack(values), axis=0), expected, rtol=rtol, atol=_scaled_atol(expected)
        )

    additive_gate("structural_algebra", structural_parts, structural_change)
    additive_gate("final_demand_algebra", final_demand_parts, final_demand_change)

    if "D_A_x" in effects and all(name in effects for name in ("D_T_coef", "D_w_coef")):
        additive_gate(
            "nested_structural_additivity",
            (effects["D_T_coef"].baseline, effects["D_w_coef"].baseline),
            effects["D_A_x"].baseline,
        )

    for check_name, effect_name in (("c_source_additivity", "D_C"), ("yf_source_additivity", "D_YF")):
        effect = effects.get(effect_name)
        if effect is not None and effect.source_categories and set(effect.source_categories) == set(SOURCE_ORIGIN_GROUPS):
            additive_gate(check_name, (item.baseline for item in effect.source_categories.values()), effect.baseline)
        else:
            details[check_name] = "canonical source-category inventory missing"

    if emitter_totals is not None and set(emitter_totals) == set(CANONICAL_EFFECTS):
        driver_ok = True
        for name, effect in effects.items():
            calculated = effect.baseline if effect.dimensions.column is None else effect.baseline.sum(axis=1)
            ok, error = _close(calculated, emitter_totals[name], rtol=rtol, atol=_scaled_atol(emitter_totals[name]))
            driver_ok &= ok
            errors[f"driver_sum:{name}"] = error
        checks["driver_sums"] = driver_ok
    else:
        details["driver_sums"] = "complete per-effect emitter totals missing"

    if total_change is not None:
        top_level = ("D_v", "D_C", "D_A_x", "D_YF", "D_YB", "D_y")
        if all(name in effects for name in top_level):
            reconstructed = sum(
                (effects[name].baseline if effects[name].dimensions.column is None else effects[name].baseline.sum(axis=1))
                for name in top_level
            )
            checks["emitter_reconciliation"], errors["emitter_reconciliation"] = _close(
                reconstructed, total_change, rtol=rtol, atol=_scaled_atol(total_change)
            )
    else:
        details["emitter_reconciliation"] = "emitter total change missing"

    max_error = max(errors.values(), default=0.0)
    return ValidationReport(
        passed=all(checks.values()),
        checks=checks,
        max_abs_error=max_error,
        details=details,
    )
