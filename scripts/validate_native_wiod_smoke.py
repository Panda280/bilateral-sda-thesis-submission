#!/usr/bin/env python3
"""Validate native-WIOD smoke outputs before a production campaign is launched."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITIES = {
    "primary": (["Dint", "Leontief_total", "FD_total"], "Total"),
    "leontief_split": (["D_C_coef", "D_T_coef", "D_w_coef"], "Leontief_total"),
    "raw_leontief_split": (["D_C2", "D_Ax2"], "Leontief_total"),
    "final_demand_split": (["YF", "YB", "DYy"], "FD_total"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def total(path: Path) -> float:
    frame = pd.read_parquet(path)
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError(f"No numeric values in {path}")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {path}")
    return float(values.sum())


def validate(run: Path, absolute_tolerance: float, relative_tolerance: float) -> dict:
    effect_dir = run / "CO2_7"
    files = sorted(effect_dir.glob("*_tot_baseline.parquet"))
    if not files:
        raise FileNotFoundError(f"No total baseline outputs under {effect_dir}")
    years = {path.name.split("_", 1)[0] for path in files}
    if len(years) != 1:
        raise ValueError(f"Expected one result year under {effect_dir}, got {sorted(years)}")
    year = years.pop()
    paths = {path.name[len(year) + 1 : -len("_tot_baseline.parquet")]: path for path in files}
    values = {effect: total(path) for effect, path in paths.items()}
    checks = []
    for identity, (children, parent) in IDENTITIES.items():
        missing = [effect for effect in [*children, parent] if effect not in values]
        if missing:
            checks.append({"identity": identity, "passed": False, "missing": missing})
            continue
        child_sum = sum(values[effect] for effect in children)
        residual = child_sum - values[parent]
        scale = max(abs(child_sum), abs(values[parent]), 1.0)
        passed = abs(residual) <= absolute_tolerance + relative_tolerance * scale
        checks.append({
            "identity": identity,
            "children": children,
            "parent": parent,
            "child_sum_kg_co2": child_sum,
            "parent_kg_co2": values[parent],
            "residual_kg_co2": residual,
            "relative_residual": abs(residual) / scale,
            "passed": passed,
        })
    return {
        "run": str(run),
        "result_year": int(year),
        "effect_directory": str(effect_dir),
        "files": [{"path": str(path), "sha256": sha256(path)} for path in files],
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--absolute-tolerance-kg", type=float, default=1.0)
    parser.add_argument("--relative-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    reports = [validate(run, args.absolute_tolerance_kg, args.relative_tolerance) for run in args.runs]
    manifest = {
        "schema_version": 1,
        "validation_id": "native-wiod-smoke-hierarchy-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "units": "kg CO2",
        "tolerances": {
            "absolute_kg": args.absolute_tolerance_kg,
            "relative": args.relative_tolerance,
        },
        "runs": reports,
        "passed": all(report["passed"] for report in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": manifest["passed"]}))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
