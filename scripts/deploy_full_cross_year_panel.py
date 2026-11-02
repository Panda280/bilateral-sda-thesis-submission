#!/usr/bin/env python
"""Gate, build, and audit the immutable 42-run cross-year panel deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sda.output.annual import load_dashboard_registry, publish_panel


RESOLUTIONS = ("2013", "2016", "R3")
EXPECTED = {f"{year}_{year + 1}_{resolution}": (year, year + 1, resolution)
            for year in range(2000, 2014) for resolution in RESOLUTIONS}
ANNUAL_ARTIFACTS = ("annual_hierarchy.parquet", "annual_drilldown.parquet",
                    "annual_overview.md", "annual_summary.v2.json")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, record: dict) -> bool:
    return (path.is_file() and path.stat().st_size == record.get("bytes")
            and _sha(path) == record.get("sha256"))


def assess_annual_inputs(annual_root: Path) -> tuple[dict, list[Path]]:
    observed = {path.parent.name for path in annual_root.glob("*/manifest.v2.json")}
    issues: dict[str, list[str]] = {}
    accepted, manifest_paths, units = [], [], set()
    for run_id, (previous, current, resolution) in EXPECTED.items():
        root, reasons = annual_root / run_id, []
        path = root / "manifest.v2.json"
        if not path.is_file():
            issues[run_id] = ["missing manifest.v2.json"]
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues[run_id] = [f"unreadable manifest: {exc}"]
            continue
        if manifest.get("report_schema_version") != 2: reasons.append("report schema is not v2")
        if (manifest.get("run_id"), manifest.get("year_pair"), manifest.get("resolution")) != (run_id, [previous, current], resolution):
            reasons.append("run identity mismatch")
        if manifest.get("validation_status") != "passed": reasons.append("annual validation did not pass")
        if not manifest.get("units"): reasons.append("units missing")
        else: units.add(manifest["units"])
        for name in ANNUAL_ARTIFACTS:
            if not _matches(root / name, manifest.get("artifacts", {}).get(name, {})):
                reasons.append(f"artifact missing or checksum mismatch: {name}")
        sources = manifest.get("source_files")
        if not isinstance(sources, list) or not sources:
            reasons.append("source-file inventory missing")
        else:
            for record in sources:
                source = Path(record.get("path", ""))
                if not _matches(source, record): reasons.append(f"source missing or checksum mismatch: {source}")
        if reasons: issues[run_id] = reasons
        else: accepted.append(run_id); manifest_paths.append(path)
    unexpected = sorted(observed - set(EXPECTED))
    if unexpected: issues["unexpected_runs"] = unexpected
    if len(units) > 1: issues["common_units"] = sorted(units)
    report = {"expected": 42, "accepted": len(accepted), "missing": sorted(set(EXPECTED) - observed),
              "excluded": sorted(set(EXPECTED) - set(accepted)), "unexpected": unexpected,
              "issues": issues, "units": next(iter(units)) if len(units) == 1 else None,
              "ready": len(accepted) == 42 and not unexpected and len(units) == 1}
    return report, manifest_paths


def _audit_panel(panel: Path) -> dict:
    registry = load_dashboard_registry()
    summary_path = panel / "panel_summary.v2.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_inventory = {tuple(value) for value in EXPECTED.values()}
    reasons = []
    if summary.get("validation_status") != "passed": reasons.append("panel validation status")
    if summary.get("annual_runs") != 42: reasons.append("annual run count")
    if summary.get("panel_count") != 11: reasons.append("panel count")
    if {tuple(item) for item in summary.get("inventory", [])} != expected_inventory: reasons.append("panel inventory")
    expected_names = {f"cross_year_panel/{item['panel_id']}.parquet" for item in registry["panels"]}
    records = {name.replace("\\", "/"): record for name, record in summary.get("artifacts", {}).items()}
    if set(records) != expected_names: reasons.append("panel artifact inventory")
    for name in expected_names:
        if not _matches(panel / name, records.get(name, {})): reasons.append(f"panel checksum: {name}")
    return {"passed": not reasons, "reasons": reasons, "summary": summary}


def _git_state(repository: Path) -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repository, check=True,
                           capture_output=True, text=True).stdout.strip()
    return f"{commit} (dirty)" if dirty else commit


def deploy(annual_root: Path, output_root: Path, *, campaign_metadata: Path,
           analysis_id: str, command: list[str]) -> tuple[Path, bool]:
    coverage, manifests = assess_annual_inputs(annual_root)
    metadata = json.loads(campaign_metadata.read_text(encoding="utf-8"))
    required = {"price_basis", "indicator", "units"}
    if set(metadata) < required: raise ValueError(f"campaign metadata missing: {sorted(required - set(metadata))}")
    if coverage["units"] is not None and metadata["units"] != coverage["units"]:
        coverage["issues"]["campaign_metadata"] = ["units differ from annual manifests"]
        coverage["ready"] = False
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / analysis_id
    if destination.exists(): raise FileExistsError(destination)
    staging = output_root / f".staging-{analysis_id}-{uuid.uuid4().hex}"; staging.mkdir()
    complete = False
    try:
        panel_audit = None
        if coverage["ready"]:
            panel = publish_panel([annual_root / run_id for run_id in EXPECTED], staging,
                                  run_id="panel_product")
            panel_audit = _audit_panel(panel)
            if not panel_audit["passed"]: raise ValueError(f"built panel failed audit: {panel_audit['reasons']}")
            complete = True
        report = ["# Full cross-year panel deployment", "",
                  f"Status: **{'complete' if complete else 'blocked'}**.", "",
                  f"Accepted annual reports: {coverage['accepted']} of 42.", ""]
        if not complete:
            report += ["No panel or numerical result table was emitted. Resolve the exact issues in `readiness.json` and rerun with a new immutable analysis ID.", ""]
        else:
            report += ["All 42 annual reports and all 11 cross-year panel files passed identity, inventory, validation, and checksum gates.",
                       "This deployment is technically complete but remains accepted provisional because upstream CYP/PYP validation is outside this downstream gate.", ""]
        (staging / "deployment_report.md").write_text("\n".join(report), encoding="utf-8")
        (staging / "readiness.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "analysis_manifest.json":
                outputs.append({"role": "panel-artifact" if "panel_product" in path.parts else "deployment-evidence",
                                "path": str(path.relative_to(staging)).replace("\\", "/"), "sha256": _sha(path), "rows": 0})
        repository = Path(__file__).resolve().parents[1]
        manifest = {"schema_version": 1, "analysis_id": analysis_id,
                    "status": "complete" if complete else "blocked",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "generator": {"script": "scripts/deploy_full_cross_year_panel.py",
                                  "git_commit": _git_state(repository), "command": command},
                    "scope": {"price_basis": metadata["price_basis"],
                              "indicator": metadata["indicator"], "units_input": metadata["units"],
                              "units_output": "unchanged", "effects": [], "resolutions": list(RESOLUTIONS),
                              "year_previous_min": 2000, "year_current_max": 2014},
                    "coverage": {key: coverage[key] for key in ("expected", "accepted", "missing", "excluded")},
                    "inputs": ([{"role": "campaign-metadata", "path": str(campaign_metadata.resolve()), "sha256": _sha(campaign_metadata)}]
                               + [{"role": "annual-manifest", "path": str(path.resolve()), "sha256": _sha(path)} for path in manifests]),
                    "outputs": outputs,
                    "validation": {"passed": complete, "gates": {"annual_readiness": coverage["ready"],
                                   "exact_42_inventory": coverage["ready"], "eleven_panel_files": bool(panel_audit and panel_audit["passed"]),
                                   "panel_summary_validation": bool(panel_audit and panel_audit["passed"])},
                                   "upstream_input_status": "accepted_provisional", "thesis_status": "accepted_provisional" if complete else "ineligible"},
                    "limitations": ["Downstream panel validation does not resolve upstream CYP/PYP failures.",
                                    "Effect and role semantics remain those declared by the annual reporting contract."]}
        (staging / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    return destination, complete


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--campaign-metadata", required=True, type=Path)
    parser.add_argument("--analysis-id", default="cross_year_panel_2001_2014")
    args = parser.parse_args(argv)
    path, complete = deploy(args.annual_root, args.output_root, campaign_metadata=args.campaign_metadata,
                            analysis_id=args.analysis_id, command=sys.argv)
    print(path)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
