#!/usr/bin/env python3
"""Audit the exact-table identity ``VA/x = 1-w`` before interpreting w.

The inventory is a CSV with one row per exact table and these columns:
``table_id,price_basis,z_path,x_path,va_path,w_path,va_authority``.
``w_path`` is optional. Arrays may be CSV, NPY, or NPZ (use ``path::key``).
VA is authoritative only when ``va_authority`` is non-empty and not one of
``unknown``, ``diagnostic`` or ``residual``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


NON_AUTHORITATIVE = {"", "unknown", "diagnostic", "residual", "none"}


def _split(spec: str) -> tuple[Path, str | None]:
    path, sep, key = spec.partition("::")
    return Path(path), key if sep else None


def load_array(spec: str) -> np.ndarray:
    path, key = _split(spec)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key is None:
                if len(data.files) != 1:
                    raise ValueError(f"NPZ requires ::key: {path}")
                key = data.files[0]
            value = data[key]
    elif suffix in {".csv", ".txt"}:
        value = np.loadtxt(path, delimiter=",", ndmin=1)
    else:
        raise ValueError(f"unsupported array file: {path}")
    # Preserve matrix rank: a one-column Z table must remain two-dimensional.
    # Vector callers flatten explicitly after loading.
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite values in {spec}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_table(row: dict[str, str], rtol: float, atol: float) -> tuple[dict, list[dict]]:
    z = load_array(row["z_path"])
    x = load_array(row["x_path"]).reshape(-1)
    va = load_array(row["va_path"]).reshape(-1)
    if z.ndim != 2 or z.shape[1] != x.size or va.size != x.size:
        raise ValueError(f"{row['table_id']}: incompatible Z/x/VA dimensions")
    if np.any(x == 0):
        raise ValueError(f"{row['table_id']}: zero output prevents coefficient audit")

    intermediate = z.sum(axis=0)
    w_raw = intermediate / x
    va_share = va / x
    raw_remainder = 1.0 - w_raw
    column_residual = x - intermediate - va
    raw_error = raw_remainder - va_share

    stored_w = None
    if row.get("w_path", "").strip():
        stored_w = load_array(row["w_path"]).reshape(-1)
        if stored_w.size != x.size:
            raise ValueError(f"{row['table_id']}: stored w dimension mismatch")
    post_remainder = 1.0 - stored_w if stored_w is not None else np.full_like(x, np.nan)
    post_error = post_remainder - va_share
    scale = np.divide(stored_w, w_raw, out=np.full_like(w_raw, np.nan), where=w_raw != 0) if stored_w is not None else np.full_like(w_raw, np.nan)
    rescaled = np.isfinite(scale) & ~np.isclose(scale, 1.0, rtol=rtol, atol=atol)
    authoritative = row.get("va_authority", "").strip().lower() not in NON_AUTHORITATIVE
    raw_pass = bool(np.allclose(raw_remainder, va_share, rtol=rtol, atol=atol))
    post_pass = bool(stored_w is not None and np.allclose(post_remainder, va_share, rtol=rtol, atol=atol))

    details = []
    for j in range(x.size):
        details.append({
            "table_id": row["table_id"], "column_index": j, "output": x[j],
            "intermediate_inputs": intermediate[j], "value_added": va[j],
            "w_raw": w_raw[j], "one_minus_w_raw": raw_remainder[j],
            "va_share": va_share[j], "raw_identity_error": raw_error[j],
            "stored_w": stored_w[j] if stored_w is not None else "",
            "one_minus_stored_w": post_remainder[j] if stored_w is not None else "",
            "post_identity_error": post_error[j] if stored_w is not None else "",
            "stored_to_raw_w_ratio": scale[j] if stored_w is not None else "",
            "post_rescaling_detected": bool(rescaled[j]),
            "column_accounting_residual": column_residual[j],
        })
    status = "PASS" if authoritative and raw_pass and (stored_w is None or post_pass) else "FAIL"
    if not authoritative:
        status = "UNRESOLVED_NON_AUTHORITATIVE_VA"
    return {
        "table_id": row["table_id"], "price_basis": row.get("price_basis", ""),
        "va_authority": row.get("va_authority", ""), "va_authoritative": authoritative,
        "columns": x.size, "stored_w_available": stored_w is not None,
        "raw_identity_pass": raw_pass, "post_identity_pass": post_pass if stored_w is not None else "",
        "post_rescaled_column_count": int(rescaled.sum()),
        "max_abs_column_accounting_residual": float(np.max(np.abs(column_residual))),
        "max_abs_raw_identity_error": float(np.max(np.abs(raw_error))),
        "max_abs_post_identity_error": float(np.max(np.abs(post_error))) if stored_w is not None else "",
        "status": status,
    }, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--fail-on-identity", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.inventory.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"table_id", "price_basis", "z_path", "x_path", "va_path", "w_path", "va_authority"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"inventory must contain {sorted(required)}")
    summaries, details = [], []
    errors = []
    for row in rows:
        try:
            summary, table_details = audit_table(row, args.rtol, args.atol)
            summaries.append(summary); details.extend(table_details)
        except Exception as exc:
            errors.append({"table_id": row.get("table_id", ""), "status": "ERROR", "error": str(exc)})
            summaries.append(errors[-1])
    for name, values in (("table_summary.csv", summaries), ("column_diagnostics.csv", details)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            fields = sorted({key for value in values for key in value}) if values else ["table_id"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    passed = bool(summaries) and all(row.get("status") == "PASS" for row in summaries)
    inputs = []
    for row in rows:
        for field in ("z_path", "x_path", "va_path", "w_path"):
            if row.get(field, "").strip():
                path, _ = _split(row[field]); inputs.append({"role": field, "path": str(path), "sha256": sha256(path)})
    manifest = {
        "schema_version": 1, "analysis_id": "tw-va-exact-table-identity-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed_or_unresolved", "validation_passed": passed,
        "definitions": {"w_raw": "sum_rows(Z) / x", "va_share": "authoritative VA / x", "required_identity": "1 - w = VA / x"},
        "tolerances": {"rtol": args.rtol, "atol": args.atol}, "inputs": inputs,
        "coverage": {"tables": len(rows), "errors": len(errors), "passed": sum(r.get("status") == "PASS" for r in summaries)},
        "limitations": ["Post-rescaling w can differ from the raw table identity.", "Residual or diagnostic VA is not accepted as authoritative.", "A passed algebraic identity does not identify wages within value added."],
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 2 if args.fail_on_identity and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
