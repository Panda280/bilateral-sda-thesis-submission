import numpy as np

from scripts.audit_tw_value_added_identity import audit_table


def _save(tmp_path, name, value):
    path = tmp_path / name
    np.save(path, np.asarray(value, dtype=float))
    return str(path)


def test_exact_identity_passes_and_detects_no_rescaling(tmp_path):
    z = [[20, 10], [30, 50]]; x = [100, 100]; va = [50, 40]; w = [.5, .6]
    row = {"table_id": "balanced", "price_basis": "PYP", "va_authority": "balanced_build_target",
           "z_path": _save(tmp_path, "z.npy", z), "x_path": _save(tmp_path, "x.npy", x),
           "va_path": _save(tmp_path, "va.npy", va), "w_path": _save(tmp_path, "w.npy", w)}
    summary, _ = audit_table(row, 1e-8, 1e-10)
    assert summary["status"] == "PASS"
    assert summary["post_rescaled_column_count"] == 0


def test_post_rescaling_fails_value_added_identity(tmp_path):
    row = {"table_id": "scaled", "price_basis": "PYP", "va_authority": "balanced_build_target",
           "z_path": _save(tmp_path, "z.npy", [[80], [30]]), "x_path": _save(tmp_path, "x.npy", [100]),
           "va_path": _save(tmp_path, "va.npy", [-10]), "w_path": _save(tmp_path, "w.npy", [.999999])}
    summary, _ = audit_table(row, 1e-8, 1e-10)
    assert summary["status"] == "FAIL"
    assert summary["raw_identity_pass"]
    assert summary["post_rescaled_column_count"] == 1


def test_residual_va_is_not_accepted_as_authoritative(tmp_path):
    row = {"table_id": "diagnostic", "price_basis": "PYP", "va_authority": "residual",
           "z_path": _save(tmp_path, "z.npy", [[50]]), "x_path": _save(tmp_path, "x.npy", [100]),
           "va_path": _save(tmp_path, "va.npy", [50]), "w_path": ""}
    summary, _ = audit_table(row, 1e-8, 1e-10)
    assert summary["status"] == "UNRESOLVED_NON_AUTHORITATIVE_VA"
