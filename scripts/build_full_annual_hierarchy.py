#!/usr/bin/env python
"""Build a reproducible, provisional 2001--2014 annual hierarchy report.

This analysis consumes the published annual-report interface.  It deliberately
does not reinterpret or reconstruct effects from lower-level sourcing products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


RESOLUTIONS = ("2013", "2016", "R3")
YEARS = range(2000, 2014)
NODES = (
    "delta_g", "emission_intensity", "D_v",
    "leontief_production_structure", "D_C", "D_A_x", "D_T_coef", "D_w_coef",
    "final_demand", "D_YF", "D_YB", "D_y",
)
METRICS = ("contribution", "positive_contribution", "negative_contribution", "gross_contribution")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, record: dict) -> bool:
    return (path.is_file() and path.stat().st_size == record.get("bytes")
            and _sha256(path) == record.get("sha256"))


def _expected() -> dict[str, tuple[int, int, str]]:
    return {f"{year}_{year + 1}_{resolution}": (year, year + 1, resolution)
            for year in YEARS for resolution in RESOLUTIONS}


def load_full_hierarchy(annual_root: Path) -> tuple[list[dict], dict]:
    """Validate and load exactly 42 current annual hierarchy products."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("full hierarchy analysis requires pyarrow") from exc

    expected = _expected()
    observed = {path.parent.name for path in annual_root.glob("*/manifest.v2.json")}
    missing, unexpected = sorted(set(expected) - observed), sorted(observed - set(expected))
    if missing or unexpected:
        raise ValueError(f"annual inventory mismatch; missing={missing}, unexpected={unexpected}")

    rows, manifests, units = [], [], set()
    for run_id, (previous, current, resolution) in expected.items():
        root = annual_root / run_id
        manifest_path = root / "manifest.v2.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("report_schema_version") != 2:
            raise ValueError(f"{run_id}: unsupported report schema")
        identity = (manifest.get("run_id"), manifest.get("year_pair"), manifest.get("resolution"))
        if identity != (run_id, [previous, current], resolution):
            raise ValueError(f"{run_id}: manifest identity mismatch")
        if manifest.get("validation_status") != "passed":
            raise ValueError(f"{run_id}: downstream validation did not pass")
        artifact = manifest.get("artifacts", {}).get("annual_hierarchy.parquet", {})
        hierarchy_path = root / "annual_hierarchy.parquet"
        if not _matches(hierarchy_path, artifact):
            raise ValueError(f"{run_id}: annual hierarchy checksum mismatch")
        for source in manifest.get("source_files", []):
            source_path = Path(source.get("path", ""))
            if not _matches(source_path, source):
                raise ValueError(f"{run_id}: source checksum mismatch: {source_path}")
        units.add(manifest.get("units"))
        annual = pq.read_table(hierarchy_path).to_pylist()
        annual_nodes = [row.get("node_id") for row in annual]
        if len(annual) != len(NODES) or set(annual_nodes) != set(NODES):
            raise ValueError(f"{run_id}: hierarchy node inventory mismatch")
        if any(row.get("validation_status") != "passed" or row.get("is_publishable") is not True for row in annual):
            raise ValueError(f"{run_id}: rejected hierarchy row")
        if any(not math.isfinite(float(row[metric])) for row in annual for metric in METRICS):
            raise ValueError(f"{run_id}: non-finite hierarchy metric")
        by_node = {row["node_id"]: row for row in annual}
        for row in annual:
            if not math.isclose(float(row["positive_contribution"]) + float(row["negative_contribution"]),
                                float(row["contribution"]), rel_tol=1e-10, abs_tol=1e-6):
                raise ValueError(f"{run_id}: signed metrics do not reconcile for {row['node_id']}")
            if not math.isclose(float(row["positive_contribution"]) - float(row["negative_contribution"]),
                                float(row["gross_contribution"]), rel_tol=1e-10, abs_tol=1e-6):
                raise ValueError(f"{run_id}: gross metric does not reconcile for {row['node_id']}")
        identities = {
            "delta_g": ("emission_intensity", "leontief_production_structure", "final_demand"),
            "emission_intensity": ("D_v",),
            "leontief_production_structure": ("D_C", "D_A_x"),
            "D_A_x": ("D_T_coef", "D_w_coef"),
            "final_demand": ("D_YF", "D_YB", "D_y"),
        }
        for parent, children in identities.items():
            if not math.isclose(float(by_node[parent]["contribution"]),
                                sum(float(by_node[child]["contribution"]) for child in children),
                                rel_tol=1e-10, abs_tol=1e-6):
                raise ValueError(f"{run_id}: hierarchy contribution does not reconstruct {parent}")
        if any((row.get("run_id"), row.get("year_previous"), row.get("year_current"), row.get("resolution"), row.get("units"))
               != (run_id, previous, current, resolution, manifest.get("units")) for row in annual):
            raise ValueError(f"{run_id}: hierarchy row identity mismatch")
        rows.extend(annual)
        manifests.append({"run_id": run_id, "path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)})
    if len(units) != 1 or None in units or "" in units:
        raise ValueError(f"annual reports do not have one explicit common unit: {sorted(map(str, units))}")
    return rows, {"annual_manifests": manifests, "units": next(iter(units))}


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _git_state(repository: Path) -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, text=True,
                            capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repository, text=True,
                           capture_output=True, check=True).stdout.strip()
    return f"{commit} (dirty)" if dirty else commit


def build_report(annual_root: Path, output_root: Path, *, upstream_audit: Path,
                 analysis_config: Path, run_id: str, command: list[str] | None = None) -> Path:
    rows, provenance = load_full_hierarchy(annual_root)
    upstream = json.loads(upstream_audit.read_text(encoding="utf-8"))
    config = json.loads(analysis_config.read_text(encoding="utf-8"))
    required = {"price_basis", "indicator", "units_input", "units_output", "unit_conversion_factor"}
    if set(config) < required:
        raise ValueError(f"analysis config missing fields: {sorted(required - set(config))}")
    if config["units_input"] != provenance["units"]:
        raise ValueError("analysis config input units differ from annual manifests")
    factor = float(config["unit_conversion_factor"])
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("unit conversion factor must be finite and positive")
    # Upstream status is disclosed, never promoted by downstream acceptance.
    upstream_status = upstream.get("status", upstream.get("validation_status", "unknown"))
    cumulative = []
    for resolution in RESOLUTIONS:
        for node in NODES:
            selected = [row for row in rows if row["resolution"] == resolution and row["node_id"] == node]
            cumulative.append({"resolution": resolution, "node_id": node,
                               **{metric: factor * sum(float(row[metric]) for row in selected) for metric in METRICS}})

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(destination)
    staging = output_root / f".staging-{run_id}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        annual_fields = ("run_id", "year_previous", "year_current", "resolution", "units", "node_id",
                         "parent_node_id", "depth", *METRICS, "denominator_value", "denominator_status",
                         "net_pct_of_parent_gross", "is_publishable", "validation_status")
        cumulative_fields = ("resolution", "node_id", *METRICS)
        ordered = sorted(rows, key=lambda row: (row["year_current"], RESOLUTIONS.index(row["resolution"]), row["depth"], row["node_id"]))
        ordered = [{**row, "units": config["units_output"],
                    **{metric: factor * float(row[metric]) for metric in METRICS},
                    "denominator_value": (None if row.get("denominator_value") is None else factor * float(row["denominator_value"]))}
                   for row in ordered]
        _write_csv(staging / "annual_hierarchy_full.csv", ordered, annual_fields)
        _write_csv(staging / "cumulative_hierarchy_full.csv", cumulative, cumulative_fields)
        report = ["# Full-window annual hierarchy (provisional)", "",
                  "This report covers all 14 adjacent transitions from 2000--2001 through 2013--2014 at the 2013, 2016, and R3 target resolutions.", "",
                  f"Indicator: `{config['indicator']}`. Price basis: `{config['price_basis']}`. Output units: `{config['units_output']}` (input values multiplied by `{factor}`).", "",
                  f"Upstream validation status: `{upstream_status}`. Downstream report validation passed, but it does not resolve upstream CYP/PYP failures. Results remain accepted provisional and must not be described as fully validated.", "",
                  "Machine-readable outputs contain signed annual values and cumulative sums. Target resolution is not treated as database identity.", ""]
        (staging / "full_annual_hierarchy.md").write_text("\n".join(report), encoding="utf-8")
        artifacts = {}
        for name in ("annual_hierarchy_full.csv", "cumulative_hierarchy_full.csv", "full_annual_hierarchy.md"):
            path = staging / name; artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        repository = Path(__file__).resolve().parents[1]
        manifest = {"schema_version": 1, "analysis_id": run_id, "status": "complete",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "generator": {"script": "scripts/build_full_annual_hierarchy.py", "git_commit": _git_state(repository),
                                  "command": command or []},
                    "scope": {"price_basis": config["price_basis"], "indicator": config["indicator"],
                              "units_input": provenance["units"], "units_output": config["units_output"],
                              "effects": list(NODES), "resolutions": list(RESOLUTIONS),
                              "year_previous_min": 2000, "year_current_max": 2014},
                    "coverage": {"expected": 42, "accepted": 42, "missing": [], "excluded": []},
                    "inputs": ([{"role": "upstream-validation-audit", "path": str(upstream_audit.resolve()), "sha256": _sha256(upstream_audit)},
                                {"role": "analysis-config", "path": str(analysis_config.resolve()), "sha256": _sha256(analysis_config)}]
                               + [{"role": "annual-manifest", **item} for item in provenance["annual_manifests"]]),
                    "outputs": [{"role": name.removesuffix(".csv").removesuffix(".md"), "path": name,
                                 "sha256": record["sha256"], "rows": (42 * 12 if name.startswith("annual_") else 3 * 12 if name.startswith("cumulative_") else 0)}
                                for name, record in artifacts.items()],
                    "validation": {"passed": True, "gates": {"exact_inventory": True, "identity": True,
                                   "common_units": True, "artifact_checksums": True, "source_checksums": True,
                                   "finite_metrics": True, "signed_metrics_reconcile": True,
                                   "hierarchy_reconstruction": True, "annual_downstream_validation": True},
                                   "upstream_input_status": upstream_status, "thesis_status": "accepted_provisional"},
                    "limitations": ["Upstream CYP/PYP validation is not repaired by downstream reconciliation.",
                                    "Target resolution is not database identity."],
                    "unit_conversion_factor": factor}
        (staging / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", required=True, type=Path)
    parser.add_argument("--upstream-audit", required=True, type=Path)
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default="full_annual_hierarchy_2001_2014")
    args = parser.parse_args(argv)
    import sys
    print(build_report(args.annual_root, args.output_root, upstream_audit=args.upstream_audit,
                       analysis_config=args.analysis_config, run_id=args.run_id, command=sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
