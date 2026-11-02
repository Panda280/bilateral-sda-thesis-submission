import numpy as np
import pytest

from sda import (
    MappingStateAdapter,
    SDAState,
    SourceGroupInputs,
    StateLabels,
    run_canonical_sda_state,
)


def _groups():
    names = ("Domestic", "Red", "Non-Red", "Non-Participant")
    c_masks = {name: np.zeros((2, 2), dtype=bool) for name in names}
    yf_masks = {name: np.zeros((2, 1), dtype=bool) for name in names}
    c_masks["Domestic"][:] = True
    yf_masks["Domestic"][:] = True
    return SourceGroupInputs(C=c_masks, YF=yf_masks)


def _arrays():
    eye = np.eye(2)
    return dict(
        v0=np.array([1.0, 2.0]), v1=np.array([1.0, 2.0]),
        L0=eye, L1=eye, C0=eye, C1=eye, A_x0=eye, A_x1=eye,
        T_coef0=eye, T_coef1=eye, w_coef0=np.ones(2), w_coef1=np.ones(2),
        F0=np.ones((2, 1)), F1=np.ones((2, 1)),
        B0=np.ones((2, 1)), B1=np.ones((2, 1)),
        levels0=np.array([1.0]), levels1=np.array([2.0]),
        supplier_sector=np.array([0, 1]),
    )


def _labels():
    return StateLabels(
        emitters=("c0-s0", "c0-s1"),
        structural_drivers=("c0-s0", "c0-s1"),
        consumers=("c0",),
        demanded_sectors=("s0", "s1"),
    )


def test_mapping_adapter_builds_state_and_runs_coordinator():
    adapter = MappingStateAdapter(_arrays(), _labels(), _groups(), sectors_per_country=2)
    state = adapter.adapt()

    result = run_canonical_sda_state(state)

    assert result.validation.passed
    np.testing.assert_allclose(result.total_change, [1.0, 2.0])


def test_mapping_adapter_supports_explicit_reference_names():
    arrays = _arrays()
    arrays["emissions_intensity_t0"] = arrays.pop("v0")
    state = MappingStateAdapter(
        arrays, _labels(), _groups(), 2, keys={"v0": "emissions_intensity_t0"}
    ).adapt()
    np.testing.assert_array_equal(state.v0, [1.0, 2.0])


def test_state_rejects_inconsistent_technical_identity():
    arrays = _arrays()
    arrays["A_x1"] = np.full((2, 2), 2.0)
    with pytest.raises(ValueError, match="A_x1 must equal"):
        SDAState(**arrays, sectors_per_country=2, labels=_labels(), source_groups=_groups())


def test_labels_reject_different_structural_order():
    with pytest.raises(ValueError, match="emitter country-sector order"):
        StateLabels(
            emitters=("a", "b"), structural_drivers=("b", "a"),
            consumers=("c",), demanded_sectors=("s0", "s1"),
        )


def test_state_rejects_non_finite_values_and_bad_partitions():
    arrays = _arrays()
    arrays["v1"] = np.array([np.nan, 1.0])
    with pytest.raises(ValueError, match="finite"):
        SDAState(**arrays, sectors_per_country=2, labels=_labels(), source_groups=_groups())

    groups = _groups()
    invalid = dict(groups.C)
    invalid["Red"] = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="mutually exclusive"):
        SDAState(
            **_arrays(), sectors_per_country=2, labels=_labels(),
            source_groups=SourceGroupInputs(invalid, groups.YF),
        )
