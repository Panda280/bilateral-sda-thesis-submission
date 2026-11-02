#!/usr/bin/env python
"""Audit annual-report provenance and gate cross-year panel deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RESOLUTIONS = ("2013", "2016", "R3")
ANNUAL_ARTIFACTS = (
    "annual_hierarchy.parquet",
    "annual_drilldown.parquet",
    "annual_overview.md",
    "annual_summary.v2.json",
)


def expected_runs() -> tuple[str, ...]:
    return tuple(
        f"{year}_{year + 1}_{resolution}"
        for year in range(2000, 2014)
        for resolution in RESOLUTIONS
    )


def _record_matches(path: Path, record: dict) -> bool:
    if not path.is_file() or path.stat().st_size != record.get("bytes"):
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == record.get("sha256")


def _prerequisites(run_root: Path, run_id: str) -> dict:
    required = {
        "bundle_config": run_root / "bundles" / run_id / "config.json",
        "bundle_manifest": run_root / "bundles" / run_id / "bundle_manifest.json",
        "canonical_manifest": run_root / "sda" / f"sda_{run_id}" / "manifest.json",
        "D_C_source_country": run_root / "offshoring" / run_id / "source_country" / "D_C.source_country_cube.parquet",
        "D_C_source_group_sector": run_root / "offshoring" / run_id / "source_group_sector" / "D_C.source_group_sector_cube.parquet",
        "D_YF_source_country": run_root / "offshoring" / run_id / "yf_source_country" / "D_YF.source_country_cube.parquet",
        "D_YF_source_group_sector": run_root / "offshoring" / run_id / "yf_source_group_sector" / "D_YF.source_group_sector_cube.parquet",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    return {"passed": not missing, "missing": missing}


def _annual_status(annual_root: Path, run_id: str) -> dict:
    root = annual_root / run_id
    manifest_path = root / "manifest.v2.json"
    if not manifest_path.is_file():
        return {"status": "missing", "reasons": ["manifest.v2.json missing"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "reasons": [f"manifest unreadable: {exc}"]}
    reasons = []
    previous, current, resolution = run_id.split("_", 2)
    if manifest.get("validation_status") != "passed":
        reasons.append("validation_status is not passed")
    if manifest.get("year_pair") != [int(previous), int(current)]:
        reasons.append("year_pair differs")
    if manifest.get("resolution") != resolution or manifest.get("run_id") != run_id:
        reasons.append("run identity differs")
    for name in ANNUAL_ARTIFACTS:
        record = manifest.get("artifacts", {}).get(name)
        if not isinstance(record, dict) or not _record_matches(root / name, record):
            reasons.append(f"artifact checksum differs: {name}")
    stale_sources = []
    for record in manifest.get("source_files", []):
        source = Path(record.get("path", ""))
        if not _record_matches(source, record):
            stale_sources.append(str(source))
    if stale_sources:
        return {"status": "stale_source", "reasons": reasons, "stale_sources": stale_sources}
    return {"status": "invalid" if reasons else "current", "reasons": reasons}


def _panel_status(panel_directory: Path) -> dict:
    summary_path = panel_directory / "panel_summary.v2.json"
    if not summary_path.is_file():
        return {"status": "missing", "reasons": ["panel_summary.v2.json missing"]}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_inventory = {
        (year, year + 1, resolution)
        for year in range(2000, 2014) for resolution in RESOLUTIONS
    }
    observed = {tuple(item) for item in summary.get("inventory", [])}
    reasons = []
    if summary.get("validation_status") != "passed": reasons.append("validation_status")
    if summary.get("annual_runs") != 42: reasons.append("annual_runs")
    if summary.get("panel_count") != 11: reasons.append("panel_count")
    if observed != expected_inventory: reasons.append("inventory")
    for name, record in summary.get("artifacts", {}).items():
        if not _record_matches(panel_directory / name, record):
            reasons.append(f"artifact checksum: {name}")
    if len(summary.get("artifacts", {})) != 11: reasons.append("artifact count")
    return {"status": "invalid" if reasons else "current", "reasons": reasons}


def audit_reporting(
    run_root: Path,
    annual_root: Path | None = None,
    panel_directory: Path | None = None,
) -> dict:
    annual = annual_root or run_root / "annual"
    runs = {}
    for run_id in expected_runs():
        runs[run_id] = {
            "prerequisites": _prerequisites(run_root, run_id),
            "annual": _annual_status(annual, run_id),
        }
    counts = {
        status: sum(item["annual"]["status"] == status for item in runs.values())
        for status in ("current", "stale_source", "missing", "invalid")
    }
    missing_prerequisites = [
        run_id for run_id, item in runs.items() if not item["prerequisites"]["passed"]
    ]
    panel_build_ready = counts == {
        "current": 42, "stale_source": 0, "missing": 0, "invalid": 0
    } and not missing_prerequisites
    panel = _panel_status(panel_directory or run_root / "panel" / "full_panel")
    return {
        "schema_version": 1,
        "run_root": str(run_root.resolve()),
        "annual_root": str(annual.resolve()),
        "expected_runs": 42,
        "annual_counts": counts,
        "missing_prerequisite_runs": missing_prerequisites,
        "panel_build_ready": panel_build_ready,
        "panel": panel,
        "runs": runs,
        "gate": (
            "Build a panel only when all 42 annual reports are current, validated, "
            "checksum-matching, and based on current source files."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--annual-root", type=Path)
    parser.add_argument("--panel-directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_reporting(args.run_root, args.annual_root, args.panel_directory)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(text, end="")
    return 0 if result["panel_build_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
