#!/usr/bin/env python3
"""Audit empirical inputs for country/group and sector SDA reports.

This command never infers a country group or sector mapping.  It writes a
stable JSON manifest and Markdown handoff whether readiness passes or fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GROUPS = {"Red", "Non-Red", "Non-Participant"}
ROLES = {
    "D_C": ("Emitter_Country", "Driver_Country", "Source_Country"),
    "D_YF": ("Emitter_Country", "Consumer_Country", "Source_Country"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(path: Path | None, required: set[str], name: str) -> tuple[pd.DataFrame | None, list[str]]:
    if path is None or not path.is_file():
        return None, [f"missing authoritative {name}: {path or '<not supplied>'}"]
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = required - set(frame.columns)
    if missing:
        return None, [f"{name} lacks columns: {sorted(missing)}"]
    return frame, []


def audit(campaign_root: Path, groups_csv: Path | None, countries_csv: Path | None,
          sectors_csv: Path | None) -> dict:
    blockers: list[str] = []
    groups, errors = _mapping(groups_csv, {"canonical_country", "analysis_group"}, "group registry")
    blockers += errors
    countries, errors = _mapping(
        countries_csv, {"resolution", "observed_country", "canonical_country"}, "country concordance"
    )
    blockers += errors
    sectors, errors = _mapping(
        sectors_csv, {"resolution", "observed_sector", "canonical_sector"}, "sector concordance"
    )
    blockers += errors
    if groups is not None:
        if groups["canonical_country"].duplicated().any():
            blockers.append("group registry has duplicate canonical_country values")
        invalid = sorted(set(groups["analysis_group"]) - GROUPS)
        if invalid:
            blockers.append(f"group registry has invalid analysis_group values: {invalid}")
    for label, frame, keys in (
        ("country concordance", countries, ["resolution", "observed_country"]),
        ("sector concordance", sectors, ["resolution", "observed_sector"]),
    ):
        if frame is not None and frame.duplicated(keys, keep=False).any():
            blockers.append(f"{label} has ambiguous duplicate source keys")

    inventory = []
    observed_countries: set[tuple[str, str]] = set()
    observed_sectors: set[tuple[str, str]] = set()
    for effect, relative, filename in (
        ("D_C", "source_country", "D_C.source_country_cube.parquet"),
        ("D_YF", "yf_source_country", "D_YF.source_country_cube.parquet"),
    ):
        for path in sorted((campaign_root / "offshoring").glob(f"*_{relative}/{filename}")):
            run_id = path.parents[1].name
            parts = run_id.split("_")
            if len(parts) != 3:
                blockers.append(f"unparseable run directory: {run_id}")
                continue
            previous, current, resolution = parts
            columns = list(ROLES[effect]) + ["Emitter_Sector", "Value"]
            frame = pd.read_parquet(path, columns=columns)
            numeric = pd.to_numeric(frame["Value"], errors="coerce")
            if not numeric.notna().all() or not (numeric.abs() != float("inf")).all():
                blockers.append(f"non-numeric or non-finite Value: {path}")
            report = path.parent / ("source_country_cube_report.json" if effect == "D_C" else "yf_source_country_cube_report.json")
            if not report.is_file():
                blockers.append(f"missing published provenance/reconciliation report: {report}")
            for role in ROLES[effect]:
                observed_countries |= {(resolution, str(code)) for code in frame[role].unique()}
            observed_sectors |= {(resolution, str(code)) for code in frame["Emitter_Sector"].unique()}
            inventory.append({"effect": effect, "run_id": run_id, "year_previous": int(previous),
                              "year_current": int(current), "resolution": resolution,
                              "path": str(path), "sha256": sha256(path), "rows": len(frame)})
    keys = {(row["run_id"], row["effect"]) for row in inventory}
    expected = {(f"{p}_{p+1}_{r}", e) for p in range(2000, 2014)
                for r in ("2013", "2016", "R3") for e in ROLES}
    missing_products = sorted(expected - keys)
    if missing_products:
        blockers.append(f"missing {len(missing_products)} of 84 source-country products")

    if countries is not None:
        country_keys = set(map(tuple, countries[["resolution", "observed_country"]].itertuples(index=False, name=None)))
        absent = sorted(observed_countries - country_keys)
        if absent:
            blockers.append(f"country concordance lacks {len(absent)} observed resolution/code pairs")
        if groups is not None:
            canon = set(groups["canonical_country"])
            unknown = sorted(set(countries["canonical_country"]) - canon)
            if unknown:
                blockers.append(f"country concordance targets {len(unknown)} countries absent from group registry")
    if sectors is not None:
        sector_keys = set(map(tuple, sectors[["resolution", "observed_sector"]].itertuples(index=False, name=None)))
        absent = sorted(observed_sectors - sector_keys)
        if absent:
            blockers.append(f"sector concordance lacks {len(absent)} observed resolution/code pairs")

    files = {}
    for label, path in (("groups", groups_csv), ("countries", countries_csv), ("sectors", sectors_csv)):
        if path is not None and path.is_file():
            files[label] = {"path": str(path), "sha256": sha256(path)}
    return {"schema_version": 1, "analysis_id": "country-sector-readiness-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if not blockers else "blocked", "blockers": blockers,
            "generator": {"script": "scripts/audit_country_sector_readiness.py",
                          "git_commit": "dirty-state-not-resolved-by-readiness-audit", "command": sys.argv},
            "scope": {"price_basis": "CYP_(t-1) -> PYP_t", "indicator": "fossil-origin CO2",
                      "units_input": "kg CO2", "units_output": None, "effects": ["D_C", "D_YF"],
                      "resolutions": ["2013", "2016", "R3"], "year_previous_min": 2000,
                      "year_current_max": 2014},
            "coverage": {"expected": 84, "accepted": len(inventory),
                         "missing": [list(item) for item in missing_products], "excluded": []},
            "inputs": [{"role": key, **value} for key, value in files.items()], "outputs": [],
            "validation": {"passed": not blockers, "gates": {"blockers": blockers},
                           "upstream_input_status": "accepted_provisional",
                           "thesis_status": "accepted_provisional" if not blockers else "ineligible"},
            "limitations": ["Upstream validation failures remain unresolved.",
                            "Analytical groups do not represent legal Kyoto compliance.",
                            "Sector and country mappings must be supplied, never inferred."],
            "provisional_due_to_upstream_validation": True,
            "interpretation": "descriptive analytical groups; not legal Kyoto compliance",
            "mapping_inputs": files, "inventory_count": len(inventory),
            "expected_inventory_count": 84, "inventory": inventory}


def markdown(result: dict) -> str:
    lines = ["# Country/group and sector report readiness", "", f"Status: **{result['status'].upper()}**", "",
             "All future outputs are provisional because upstream input validation remains unresolved. "
             "Groups are descriptive analytical groups, not legal Kyoto compliance classifications.", ""]
    if result["blockers"]:
        lines += ["## Exact blockers", ""] + [f"- {item}" for item in result["blockers"]] + [""]
    lines += ["## Product inventory", "", f"Found {result['inventory_count']} of {result['expected_inventory_count']} required effect products.", "",
              "## Reproducible next command", "", "```bash",
              "python scripts/audit_country_sector_readiness.py --campaign-root /path/to/campaign \\",
              "  --groups-csv /authoritative/kyoto_country_groups.csv \\",
              "  --countries-csv /authoritative/country_concordance.csv \\",
              "  --sectors-csv /authoritative/sector_concordance.csv --output-dir /path/to/audit", "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--countries-csv", type=Path)
    parser.add_argument("--sectors-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign_root, args.groups_csv, args.countries_csv, args.sectors_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "readiness_report.md").write_text(markdown(result), encoding="utf-8")
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
