"""End-to-end test of the production data-pipeline bridge.

Exercises the exact chain W1 commits to: an EXIODeflator-shaped ``TY`` pair
-> ``EXIODeflatorStateAdapter`` -> ``sda.integration.bundle_export`` ->
``scripts/build_production_run.py`` -> a published canonical run that passes
Gate C (``emitter_reconciliation``) end to end.

The real ``SDA.matrix_calculations.TY`` requires CuPy and a CUDA GPU (see
``docs/decisions/0004-bridge-execution-environment.md``), unavailable in this
CPU-only test environment, so the ``TY`` pair here is a synthetic
``SimpleNamespace`` built to the exact attribute/shape contract
``EXIODeflatorStateAdapter`` and ``docs/provenance.md`` already document and
``tests/test_exiodeflator_adapter.py`` already pins -- larger than that
adapter unit test (3 countries x 2 sectors, 2 Monte Carlo samples, and a real
3-D ``v[emitter, impact, sample]`` axis) so the country-major grid,
multi-sample selection, and all four source-origin groups are genuinely
exercised together, the way a small real WIOT run would be.

One thing the (adapter-only) unit test fixture does not have to get right,
and this one does: ``L`` must be the genuine Leontief inverse of the
technical-coefficient matrix ``C ⊙ A_x`` implies (``SDA.matrix_calculations
.TY.get_C``/``get_A_x`` document ``C ⊙ A_x == A``, and ``get_L`` computes
``L = inv(I - A)`` from that same ``A`` -- see
docs/decisions/0005-bundle-export-boundary.md). Gate C
(``emitter_reconciliation``) checks ``total_change`` -- computed directly
from ``v1·(L1·y1) - v0·(L0·y0)`` -- against the sum of the top-level
effects, whose structural terms are propagated *through* ``L0``/``L1``
(``sda.attribution.structural.attribute_structural_driver``). An
inconsistent ``L`` (e.g. a fixed identity matrix while ``A_x`` varies, as
the smaller adapter-only fixture uses, since it never runs the full
decomposition) breaks that identity outright -- this was caught by this
test during development, not assumed going in. Everything downstream of the
synthetic ``TY`` pair -- the adapter, the bundle writer, the bridge script,
the CLI, storage, and every validation gate -- is the real, unmodified
production code path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from build_production_run import BridgeError, build_production_run
from sda.integration import EXIODeflatorStateAdapter, source_file_record, write_state_bundle
from sda.output import load_result

COUNTRIES = ("AA", "BB", "CC")
SECTORS = ("s0", "s1")
INDEX = tuple((country, sector) for country in COUNTRIES for sector in SECTORS)
GROUPS = {"AA": "Red", "BB": "Non-Red", "CC": "Non-Participant"}
N, C, S = len(INDEX), len(COUNTRIES), len(SECTORS)
NUM_SAMPLES = 2


def _sampled(baseline: np.ndarray) -> np.ndarray:
    """Stack a baseline array with a perturbed twin along a trailing Monte
    Carlo sample axis -- sample 0 stays the noiseless baseline, matching
    real SDA.matrix_calculations.convert_to_gpu_arrays (noise fixed to 1 at
    sample index 0)."""
    return np.stack([baseline, baseline * 1.01], axis=-1)


def _leontief_inverse(c_arr: np.ndarray, ax_arr: np.ndarray) -> np.ndarray:
    """L = inv(I - C ⊙ A_x), per sample -- see module docstring."""
    n = c_arr.shape[0]
    out = np.zeros_like(c_arr)
    for sample in range(c_arr.shape[-1]):
        technical = c_arr[:, :, sample] * ax_arr[:, :, sample]
        out[:, :, sample] = np.linalg.inv(np.eye(n) - technical)
    return out


def _period(offset: float) -> SimpleNamespace:
    """Build one synthetic period matching SDA.matrix_calculations.TY's real
    attribute/shape/sample-axis contract (docs/provenance.md), with a
    technical-coefficient spectral radius kept well under 1 so L stays a
    well-conditioned Leontief inverse."""
    eye = np.eye(N)
    w = np.full(N, 0.10) + 0.01 * offset
    t = eye * (0.9 + 0.02 * offset)
    ax = t * w[None, :]
    c_mat = eye * (1.0 + 0.05 * offset)

    f = np.full((N, C), 0.4 + offset)
    b_unique = np.linspace(0.2, 0.8, S * C).reshape(S, C) + offset
    b_unique = b_unique / b_unique.sum(axis=1, keepdims=True)

    v_baseline = (np.arange(N, dtype=float) + 1.0 + offset)[:, None]  # (n, impact=1)
    c_arr = _sampled(c_mat)
    ax_arr = _sampled(ax)

    return SimpleNamespace(
        indexes_out=INDEX,
        v=_sampled(v_baseline),  # (n, impact, sample)
        L=_leontief_inverse(c_arr, ax_arr),
        C=c_arr,
        A_x=ax_arr,
        T_coefficient=_sampled(t),
        w_coefficient=_sampled(w)[None, ...],
        F=_sampled(f),
        B=_sampled(np.tile(b_unique, (C, 1))),
        y_b=_sampled(np.full(C, 30.0) + offset)[:, None, :],
    )


def _build_state():
    previous, current = _period(0.0), _period(1.0)
    return EXIODeflatorStateAdapter(
        previous, current, GROUPS, sample_index=0, impact_index=0
    ).adapt()


def _consumed_files(tmp_path):
    """Stand-ins for the four build_wiod_cyp_pyp.py output files a real
    bridge run consumes (CYP Z/Y for the previous year, PYP Z/Y for the
    current year) -- real files so source_file_record's checksum path is
    genuinely exercised, not mocked."""
    records = []
    for role in ("CYP_Z_2003", "CYP_Y_2003", "PYP_Z_2004", "PYP_Y_2004"):
        path = tmp_path / f"{role}.csv"
        path.write_text(f"synthetic stand-in for {role}\n", encoding="utf-8")
        records.append({"role": role, **source_file_record(path)})
    return records


def test_bridge_publishes_a_run_that_passes_gate_c(tmp_path):
    state = _build_state()
    bundle_dir = tmp_path / "bundle"
    consumed = _consumed_files(tmp_path)
    config_path = write_state_bundle(
        state,
        bundle_dir,
        consumed_files=consumed,
        provenance={"year": 2004, "year_prev": 2003, "database": "2013"},
    )
    assert config_path == bundle_dir / "config.json"

    published = build_production_run(bundle_dir, tmp_path / "results", "bridge-test-run")

    result = load_result(published)
    assert result.validation.passed
    assert all(result.validation.checks.values())
    assert result.validation.checks["emitter_reconciliation"]

    provenance_path = published / "source_provenance.json"
    assert provenance_path.exists()
    manifest = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert [entry["role"] for entry in manifest["consumed_files"]] == [
        "CYP_Z_2003", "CYP_Y_2003", "PYP_Z_2004", "PYP_Y_2004",
    ]
    for entry, expected in zip(manifest["consumed_files"], consumed):
        assert entry["sha256"] == expected["sha256"]
    assert manifest["provenance"]["database"] == "2013"


def test_bridge_reconstructs_total_change_from_top_level_effects(tmp_path):
    """Sanity-checks the bridge's own arithmetic, not just that gates pass:
    the six canonical top-level effects should reconstruct total_change
    exactly, independent of build_mandatory_validation_report's own
    internal check of the same identity."""
    state = _build_state()
    bundle_dir = tmp_path / "bundle"
    config_path = write_state_bundle(
        state, bundle_dir, consumed_files=_consumed_files(tmp_path)
    )
    published = build_production_run(config_path.parent, tmp_path / "results", "run")
    result = load_result(published)

    reconstructed = sum(
        effect.baseline if effect.dimensions.column is None else effect.baseline.sum(axis=1)
        for name, effect in result.effects.items()
        if name in ("D_v", "D_C", "D_A_x", "D_YF", "D_YB", "D_y")
    )
    np.testing.assert_allclose(reconstructed, result.total_change, rtol=1e-8, atol=1e-9)


def test_bridge_refuses_bundle_without_manifest(tmp_path):
    state = _build_state()
    bundle_dir = tmp_path / "bundle"
    write_state_bundle(state, bundle_dir, consumed_files=_consumed_files(tmp_path))
    (bundle_dir / "bundle_manifest.json").unlink()

    with pytest.raises(BridgeError, match="bundle_manifest.json"):
        build_production_run(bundle_dir, tmp_path / "results", "run")


def test_bridge_refuses_bundle_with_empty_manifest(tmp_path):
    state = _build_state()
    bundle_dir = tmp_path / "bundle"
    write_state_bundle(state, bundle_dir, consumed_files=_consumed_files(tmp_path))
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps({"schema_version": 1, "consumed_files": [], "provenance": {}}),
        encoding="utf-8",
    )

    with pytest.raises(BridgeError, match="at least one consumed source file"):
        build_production_run(bundle_dir, tmp_path / "results", "run")


def test_write_state_bundle_requires_at_least_one_consumed_file(tmp_path):
    state = _build_state()
    with pytest.raises(ValueError, match="at least one source file"):
        write_state_bundle(state, tmp_path / "bundle", consumed_files=[])


def test_bridge_missing_bundle_config_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(BridgeError, match="config.json"):
        build_production_run(empty_dir, tmp_path / "results", "run")
