#!/usr/bin/env python3
"""Post-run validation for the completed native-WIOD CO2 sensitivity campaign.

This wraps the smoke-stage checks in ``validate_native_wiod_smoke.py`` and
iterates over every production run directory recorded in ``campaign_manifest.json``.
It writes a campaign-level validation manifest and a Markdown report. It never
edits ``campaign_manifest.json``; that file is the execution record.

Per-run checks (handover docs/HANDOVER_CURRENT_STATE_20260902.md, section 6.A):

1.  all expected effect products exist (12 effects x 18 file variants);
2.  every array contains only finite values;
3.  ``Dint + Leontief_total + FD_total == Total``;
4.  ``D_C_coef + D_T_coef + D_w_coef == Leontief_total``;
5.  ``D_C2 + D_Ax2 == Leontief_total``;
6.  ``YF + YB + DYy == FD_total``;
7.  output aggregation conserves each effect (every re-aggregated matrix
    variant sums to the same total as the effect's ``_tot`` file);
8.  the run log ends with ``All tasks are done!`` and records no non-finite
    Leontief fallback, and the manifest exit code is 0;
9.  configuration, units, years and price basis match the campaign manifest.

Identities 3-6 and the conservation check are evaluated for both the
``baseline`` (deterministic sample) and ``mean`` (Monte-Carlo mean) result
families.

Usage::

    .venv/bin/python scripts/validate_native_wiod_campaign.py \
        --campaign-root /local/.../native-wiod-co2-2001-2009-20260902 \
        --output-dir    /local/.../native-wiod-co2-2001-2009-20260902/validation_YYYYMMDD
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EFFECTS = (
    "Dint",
    "Leontief_total",
    "FD_total",
    "Total",
    "D_C_coef",
    "D_T_coef",
    "D_w_coef",
    "D_C2",
    "D_Ax2",
    "YF",
    "YB",
    "DYy",
)

# Matrix re-aggregations of the effect; each must conserve the effect total
# (up to float error).
MATRIX_VARIANTS = (
    "country_matrix",
    "sector_matrix",
    "group_matrix",
    "country_to_group_sector_matrix",
    "group_sector_to_country_matrix",
    "sector_group_matrix",
)
# ``flows`` is an edge-list view; it is checked at a looser tolerance and never
# hard-fails the run on its own.
FLOWS_VARIANT = "flows"
# The plain ``{year}_{effect}_{family}.parquet`` detail matrix, keyed as "".
DETAIL_VARIANT = ""
# Variants whose sum must match the effect's ``_tot`` file exactly.
REAGGREGATION_VARIANTS = (DETAIL_VARIANT,) + MATRIX_VARIANTS
ALL_VARIANTS = REAGGREGATION_VARIANTS + (FLOWS_VARIANT,)
# Per effect per family: detail + tot + 6 matrix + flows == 9 files;
# x {baseline, mean} == 18 files/effect.
MID_TOKENS = ("tot",) + MATRIX_VARIANTS + (FLOWS_VARIANT,)

IDENTITIES = {
    "primary": (["Dint", "Leontief_total", "FD_total"], "Total"),
    "leontief_split": (["D_C_coef", "D_T_coef", "D_w_coef"], "Leontief_total"),
    "raw_leontief_split": (["D_C2", "D_Ax2"], "Leontief_total"),
    "final_demand_split": (["YF", "YB", "DYy"], "FD_total"),
}

EXPECTED_UNITS = "kg CO2"
EXPECTED_PRICE_BASIS = "CYP_(t-1) to PYP_t"
FALLBACK_LOG_MARKERS = (
    "non-finite entries after the full",
    "zeroing per SDA_ALLOW_NONFINITE_L_FALLBACK",
)
DONE_MARKER = "All tasks are done!"
IA_L_RESIDUAL_RE = re.compile(
    r"\(I-A\)@L residual vs identity - Max:\s*([0-9.eE+-]+)\s*\(sample\s*(\d+),"
    r"\s*gate threshold\s*([0-9.eE+-]+)\)"
)
ZERO_OUTPUT_INTENSITY_RE = re.compile(
    r"(\d+)\s+zero-output \(sector, sample\) cell\(s\) have undefined direct intensity"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_path(effect_dir: Path, year: int, effect: str, variant: str, family: str) -> Path:
    if variant == DETAIL_VARIANT:
        return effect_dir / f"{year}_{effect}_{family}.parquet"
    return effect_dir / f"{year}_{effect}_{variant}_{family}.parquet"


def expected_filenames(year: int) -> list[str]:
    names: list[str] = []
    for effect in EFFECTS:
        for family in ("baseline", "mean"):
            names.append(f"{year}_{effect}_{family}.parquet")
            for token in MID_TOKENS:
                names.append(f"{year}_{effect}_{token}_{family}.parquet")
    return names


def numeric_array(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError(f"no numeric columns in {path}")
    return numeric.to_numpy(dtype=float)


def array_stats(path: Path) -> tuple[float, bool]:
    values = numeric_array(path)
    finite = bool(np.isfinite(values).all())
    total = float(np.nansum(values)) if not finite else float(values.sum())
    return total, finite


def close(child_sum: float, parent: float, abs_tol: float, rel_tol: float) -> tuple[bool, float, float]:
    residual = child_sum - parent
    scale = max(abs(child_sum), abs(parent), 1.0)
    passed = abs(residual) <= abs_tol + rel_tol * scale
    return passed, residual, abs(residual) / scale


def check_files_present(effect_dir: Path, year: int) -> dict:
    expected = expected_filenames(year)
    present = {p.name for p in effect_dir.glob("*.parquet")}
    missing = sorted(name for name in expected if name not in present)
    unexpected = sorted(name for name in present if name not in set(expected))
    return {
        "passed": not missing,
        "expected": len(expected),
        "found": len(present),
        "missing": missing,
        "unexpected": unexpected,
    }


def read_family(effect_dir: Path, year: int, family: str) -> tuple[dict, dict, list[dict]]:
    """Return (effect_total, per-variant sums, non-finite file records) for a family."""
    totals: dict[str, float] = {}
    variant_sums: dict[str, dict[str, float]] = {}
    nonfinite: list[dict] = []
    for effect in EFFECTS:
        tot_path = effect_dir / f"{year}_{effect}_tot_{family}.parquet"
        if tot_path.exists():
            total, finite = array_stats(tot_path)
            totals[effect] = total
            if not finite:
                nonfinite.append({"file": tot_path.name})
        variant_sums[effect] = {}
        for variant in ALL_VARIANTS:
            path = variant_path(effect_dir, year, effect, variant, family)
            if not path.exists():
                continue
            total, finite = array_stats(path)
            variant_sums[effect][variant or "detail"] = total
            if not finite:
                nonfinite.append({"file": path.name})
    return totals, variant_sums, nonfinite


def check_finite_all(effect_dir: Path, year: int) -> dict:
    nonfinite: list[str] = []
    checked = 0
    for name in expected_filenames(year):
        path = effect_dir / name
        if not path.exists():
            continue
        checked += 1
        try:
            values = numeric_array(path)
        except ValueError:
            nonfinite.append(name)
            continue
        if not np.isfinite(values).all():
            nonfinite.append(name)
    return {"passed": not nonfinite, "files_checked": checked, "nonfinite_files": nonfinite}


def check_hierarchy(totals: dict, abs_tol: float, rel_tol: float) -> dict:
    identities = []
    for name, (children, parent) in IDENTITIES.items():
        missing = [e for e in [*children, parent] if e not in totals]
        if missing:
            identities.append({"identity": name, "passed": False, "missing": missing})
            continue
        child_sum = sum(totals[e] for e in children)
        passed, residual, rel = close(child_sum, totals[parent], abs_tol, rel_tol)
        identities.append({
            "identity": name,
            "children": children,
            "parent": parent,
            "child_sum_kg_co2": child_sum,
            "parent_kg_co2": totals[parent],
            "residual_kg_co2": residual,
            "relative_residual": rel,
            "passed": passed,
        })
    return {"passed": all(i["passed"] for i in identities), "identities": identities}


def check_conservation(
    totals: dict,
    variant_sums: dict,
    abs_tol: float,
    rel_tol: float,
    flows_rel_tol: float,
) -> dict:
    effects_out = []
    hard_pass = True
    for effect in EFFECTS:
        if effect not in totals:
            effects_out.append({"effect": effect, "passed": False, "reason": "missing _tot file"})
            hard_pass = False
            continue
        canonical = totals[effect]
        variants = []
        effect_pass = True
        for variant, value in variant_sums.get(effect, {}).items():
            is_flows = variant == FLOWS_VARIANT
            tol = flows_rel_tol if is_flows else rel_tol
            passed, residual, rel = close(value, canonical, abs_tol, tol)
            variants.append({
                "variant": variant,
                "sum_kg_co2": value,
                "residual_kg_co2": residual,
                "relative_residual": rel,
                "tolerance_relative": tol,
                "passed": passed,
                "hard": not is_flows,
            })
            if not passed and not is_flows:
                effect_pass = False
        effects_out.append({
            "effect": effect,
            "canonical_total_kg_co2": canonical,
            "variants": variants,
            "passed": effect_pass,
        })
        hard_pass = hard_pass and effect_pass
    return {"passed": hard_pass, "effects": effects_out}


def check_log(log_path: Path, exit_code, samples: int) -> dict:
    result: dict = {
        "log": str(log_path),
        "exists": log_path.exists(),
        "manifest_exit_code": exit_code,
        "all_tasks_done": False,
        "nonfinite_leontief_fallback": False,
        "ia_l_residuals": [],
        "zero_output_intensity_notes": [],
        "declared_num_samples": None,
    }
    if not log_path.exists():
        result["passed"] = False
        return result
    text = log_path.read_text(errors="replace")
    result["all_tasks_done"] = DONE_MARKER in text
    result["nonfinite_leontief_fallback"] = any(m in text for m in FALLBACK_LOG_MARKERS)
    for match in IA_L_RESIDUAL_RE.finditer(text):
        max_res, sample, threshold = float(match.group(1)), int(match.group(2)), float(match.group(3))
        result["ia_l_residuals"].append({
            "max_residual": max_res,
            "sample": sample,
            "gate_threshold": threshold,
            "passed": max_res <= threshold,
        })
    for match in ZERO_OUTPUT_INTENSITY_RE.finditer(text):
        result["zero_output_intensity_notes"].append(int(match.group(1)))
    sample_match = re.search(r"num_samples=(\d+)", text)
    if sample_match:
        result["declared_num_samples"] = int(sample_match.group(1))
    result["passed"] = (
        result["all_tasks_done"]
        and not result["nonfinite_leontief_fallback"]
        and exit_code == 0
        and all(r["passed"] for r in result["ia_l_residuals"])
        and (result["declared_num_samples"] in (None, samples))
    )
    return result


def check_config_match(run_record: dict, scope: dict, effect_dir: Path, run_id: str) -> dict:
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    parts = run_id.split("_")
    config_id = "_".join(parts[2:])
    prev_year, target_year = int(parts[0]), int(parts[1])

    add("run_id_matches_manifest_configuration",
        run_record.get("configuration", {}).get("id") == config_id,
        {"run_id_config": config_id, "manifest_config": run_record.get("configuration", {}).get("id")})
    add("transition_matches_manifest",
        run_record.get("transition") == f"{prev_year}->{target_year}",
        {"expected": f"{prev_year}->{target_year}", "manifest": run_record.get("transition")})
    add("transition_definition_is_target_minus_one",
        target_year - prev_year == 1,
        {"previous": prev_year, "target": target_year})
    add("target_year_in_campaign_scope",
        target_year in scope.get("target_years", []),
        {"target_year": target_year, "scope": scope.get("target_years")})
    add("units_are_kg_co2",
        scope.get("units") == EXPECTED_UNITS,
        {"manifest_units": scope.get("units")})
    add("price_basis_matches",
        scope.get("price_basis") == EXPECTED_PRICE_BASIS,
        {"manifest_price_basis": scope.get("price_basis")})

    result_years = sorted({p.name.split("_", 1)[0] for p in effect_dir.glob("*_tot_baseline.parquet")})
    add("single_result_year_equals_target",
        result_years == [str(target_year)],
        {"result_years_on_disk": result_years, "target_year": target_year})

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def validate_run(run_record: dict, scope: dict, args) -> dict:
    run_id = run_record["run_id"]
    output_dir = Path(run_record["output"])
    log_path = Path(run_record["log"])
    parts = run_id.split("_")
    prev_year, target_year = int(parts[0]), int(parts[1])
    effect_dir = output_dir / "CO2_7"

    report: dict = {
        "run_id": run_id,
        "configuration": run_record.get("configuration"),
        "transition": run_record.get("transition"),
        "previous_year": prev_year,
        "target_year": target_year,
        "output_dir": str(output_dir),
        "effect_dir": str(effect_dir),
        "checks": {},
    }

    if not effect_dir.is_dir():
        report["checks"]["files_present"] = {"passed": False, "error": f"missing {effect_dir}"}
        report["passed"] = False
        return report

    files_present = check_files_present(effect_dir, target_year)
    report["checks"]["files_present"] = files_present

    report["checks"]["finite"] = check_finite_all(effect_dir, target_year)

    baseline_totals, baseline_variants, baseline_nf = read_family(effect_dir, target_year, "baseline")
    mean_totals, mean_variants, mean_nf = read_family(effect_dir, target_year, "mean")

    report["checks"]["hierarchy_baseline"] = check_hierarchy(
        baseline_totals, args.hierarchy_absolute_kg, args.hierarchy_relative)
    report["checks"]["hierarchy_mean"] = check_hierarchy(
        mean_totals, args.hierarchy_absolute_kg, args.hierarchy_relative)

    report["checks"]["conservation_baseline"] = check_conservation(
        baseline_totals, baseline_variants,
        args.conservation_absolute_kg, args.conservation_relative, args.flows_conservation_relative)
    report["checks"]["conservation_mean"] = check_conservation(
        mean_totals, mean_variants,
        args.conservation_absolute_kg, args.conservation_relative, args.flows_conservation_relative)

    report["checks"]["log"] = check_log(log_path, run_record.get("exit_code"), scope.get("samples", 8))
    report["checks"]["config_match"] = check_config_match(run_record, scope, effect_dir, run_id)

    tot_hashes = []
    for effect in EFFECTS:
        p = effect_dir / f"{target_year}_{effect}_tot_baseline.parquet"
        if p.exists():
            tot_hashes.append({"path": str(p), "sha256": sha256(p)})
    report["tot_baseline_hashes"] = tot_hashes

    # Gating checks. The smoke gate and the thesis result set are defined on the
    # deterministic ``baseline`` sample, so the ``mean`` (Monte-Carlo) family is
    # advisory: its residuals accumulate extra floating-point error and are
    # reported, not gated.
    gating = (
        "files_present",
        "finite",
        "hierarchy_baseline",
        "conservation_baseline",
        "log",
        "config_match",
    )
    advisory = ("hierarchy_mean", "conservation_mean")
    report["gating_checks"] = list(gating)
    report["advisory_checks"] = list(advisory)
    report["passed"] = all(report["checks"][name]["passed"] for name in gating)
    report["advisory_passed"] = all(report["checks"][name]["passed"] for name in advisory)
    return report


def build_markdown(manifest: dict) -> str:
    lines = []
    a = lines.append
    a(f"# Native-WIOD campaign post-run validation")
    a("")
    a(f"- Generated: {manifest['generated_at_utc']}")
    a(f"- Campaign: `{manifest['campaign']['campaign_id']}` "
      f"(manifest status `{manifest['campaign']['status']}`)")
    a(f"- Campaign manifest sha256: `{manifest['campaign']['manifest_sha256']}`")
    a(f"- Units: {manifest['campaign']['units']}; price basis: "
      f"{manifest['campaign']['price_basis']}")
    a(f"- Generator commit: `{manifest['generator']['git_commit']}`")
    a("")
    a(f"**Overall: {'PASS' if manifest['passed'] else 'FAIL'}** "
      f"({manifest['summary']['runs_passed']}/{manifest['expected_runs']} runs passed)")
    a("")
    if manifest["summary"]["failed_run_ids"]:
        a("Gating failures: " + ", ".join(f"`{r}`" for r in manifest["summary"]["failed_run_ids"]))
        a("")
    adv = manifest["summary"].get("advisory_failed_run_ids", [])
    if adv:
        a(f"Advisory-only findings (not gating) on {len(adv)} run(s): "
          + ", ".join(f"`{r}`" for r in adv))
        a("")
    a("## Tolerances")
    a("")
    for k, v in manifest["tolerances"].items():
        a(f"- {k}: {v}")
    a("")
    a("## Per-run summary")
    a("")
    a("Gating columns: files, finite, hier(base), cons(base), log, config. "
      "Advisory columns (not gating): hier(mean), cons(mean).")
    a("")
    a("| run | files | finite | hier(base) | cons(base) | log | config | **pass** | hier(mean) | cons(mean) |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for run in manifest["runs"]:
        c = run["checks"]

        def mark(key):
            return "ok" if c.get(key, {}).get("passed") else "FAIL"

        def adv_mark(key):
            return "ok" if c.get(key, {}).get("passed") else "advisory"

        a(f"| `{run['run_id']}` | {mark('files_present')} | {mark('finite')} | "
          f"{mark('hierarchy_baseline')} | {mark('conservation_baseline')} | "
          f"{mark('log')} | {mark('config_match')} | "
          f"**{'PASS' if run['passed'] else 'FAIL'}** | "
          f"{adv_mark('hierarchy_mean')} | {adv_mark('conservation_mean')} |")
    a("")
    a("## Worst residuals observed")
    a("")
    worst = {"hier_base": (0.0, ""), "hier_mean": (0.0, ""),
             "cons_base": (0.0, ""), "cons_mean": (0.0, ""), "flows": (0.0, "")}

    def bump(key, value, where):
        if value > worst[key][0]:
            worst[key] = (value, where)

    for run in manifest["runs"]:
        rid = run["run_id"]
        for fam, key in (("hierarchy_baseline", "hier_base"), ("hierarchy_mean", "hier_mean")):
            for ident in run["checks"].get(fam, {}).get("identities", []):
                bump(key, ident.get("relative_residual", 0.0), f"{rid}/{ident['identity']}")
        for fam, key in (("conservation_baseline", "cons_base"), ("conservation_mean", "cons_mean")):
            for eff in run["checks"].get(fam, {}).get("effects", []):
                for v in eff.get("variants", []):
                    if v["variant"] == "flows":
                        bump("flows", v["relative_residual"], f"{rid}/{eff['effect']}")
                    else:
                        bump(key, v["relative_residual"], f"{rid}/{eff['effect']}/{v['variant']}")

    t = manifest["tolerances"]
    a(f"- hierarchy, baseline: max {worst['hier_base'][0]:.3e} at {worst['hier_base'][1]} "
      f"(tolerance {t['hierarchy_relative']}, gating)")
    a(f"- hierarchy, mean: max {worst['hier_mean'][0]:.3e} at {worst['hier_mean'][1]} "
      f"(tolerance {t['hierarchy_relative']}, advisory)")
    a(f"- re-aggregation conservation, baseline: max {worst['cons_base'][0]:.3e} at "
      f"{worst['cons_base'][1]} (tolerance {t['conservation_relative']}, gating)")
    a(f"- re-aggregation conservation, mean: max {worst['cons_mean'][0]:.3e} at "
      f"{worst['cons_mean'][1]} (tolerance {t['conservation_relative']}, advisory)")
    a(f"- flows-view conservation: max {worst['flows'][0]:.3e} at {worst['flows'][1]} "
      f"(tolerance {t['flows_conservation_relative']}, informational)")
    a("")
    a("## Notes")
    a("")
    a("- `baseline` = deterministic midpoint sample; `mean` = mean over the "
      f"{manifest['campaign']['samples']} Monte-Carlo orderings.")
    a("- Gating is defined on the `baseline` family only, matching the smoke "
      "hierarchy gate (`validate_native_wiod_smoke.py`, which globs "
      "`*_tot_baseline.parquet`). The `mean` family accumulates additional "
      "floating-point error and is reported as advisory, not gated.")
    a("- The `flows` edge-list view is checked at a looser tolerance and never "
      "hard-fails a run on its own; residuals are recorded.")
    a("- Zero-output cells with undefined direct intensity are set to 0 by "
      "pipeline policy; per-run counts from the logs are recorded in the JSON "
      "manifest under `checks.log.zero_output_intensity_notes`.")
    a("- This validation is an execution-integrity gate. It does not by itself "
      "confer publication acceptance; results still require annual reporting "
      "(task B) and registration (task D).")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-relative", type=float, default=2e-5)
    parser.add_argument("--hierarchy-absolute-kg", type=float, default=1.0)
    parser.add_argument("--conservation-relative", type=float, default=2e-5)
    parser.add_argument("--conservation-absolute-kg", type=float, default=1.0)
    parser.add_argument("--flows-conservation-relative", type=float, default=1e-3)
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args()

    manifest_path = args.campaign_root / "campaign_manifest.json"
    campaign = json.loads(manifest_path.read_text())
    scope = campaign.get("scope", {})
    runs = campaign.get("runs", [])

    run_reports = []
    for record in runs:
        print(f"[validate] {record['run_id']}", flush=True)
        run_reports.append(validate_run(record, scope, args))

    passed_ids = [r["run_id"] for r in run_reports if r["passed"]]
    failed_ids = [r["run_id"] for r in run_reports if not r["passed"]]
    advisory_failed_ids = [
        r["run_id"] for r in run_reports if not r.get("advisory_passed", True)
    ]

    manifest = {
        "schema_version": 1,
        "validation_id": "native-wiod-campaign-postrun-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": "implementation/bilateral_sda_attribution/scripts/validate_native_wiod_campaign.py",
            "git_commit": args.git_commit,
            "host": platform.node(),
            "python": sys.version.split()[0],
            "command": sys.argv,
        },
        "campaign": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "campaign_id": campaign.get("campaign_id"),
            "status": campaign.get("status"),
            "started_at_utc": campaign.get("started_at_utc"),
            "finished_at_utc": campaign.get("finished_at_utc"),
            "units": scope.get("units"),
            "price_basis": scope.get("price_basis"),
            "indicator_boundary": scope.get("indicator_boundary"),
            "samples": scope.get("samples"),
        },
        "tolerances": {
            "hierarchy_relative": args.hierarchy_relative,
            "hierarchy_absolute_kg": args.hierarchy_absolute_kg,
            "conservation_relative": args.conservation_relative,
            "conservation_absolute_kg": args.conservation_absolute_kg,
            "flows_conservation_relative": args.flows_conservation_relative,
        },
        "effects": list(EFFECTS),
        "file_variants_per_effect": 18,
        "expected_files_per_run": 18 * len(EFFECTS),
        "expected_runs": len(runs),
        "identities": {k: {"children": v[0], "parent": v[1]} for k, v in IDENTITIES.items()},
        "runs": run_reports,
        "summary": {
            "runs_passed": len(passed_ids),
            "runs_failed": len(failed_ids),
            "failed_run_ids": failed_ids,
            "advisory_failed_run_ids": advisory_failed_ids,
        },
        "passed": bool(not failed_ids and run_reports and len(run_reports) == len(runs)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "validation_manifest.json"
    out_md = args.output_dir / "VALIDATION_REPORT.md"
    out_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(build_markdown(manifest), encoding="utf-8")

    print(json.dumps({
        "output": str(out_json),
        "passed": manifest["passed"],
        "runs_passed": len(passed_ids),
        "runs_failed": len(failed_ids),
        "failed_run_ids": failed_ids,
    }, indent=2))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
