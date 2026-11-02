#!/usr/bin/env python3
"""Run the gated 2001--2009 native-CO2 sensitivity campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# The four 2013-grid (41x35) configurations run first -- they form the
# environmental-extension x flow-table 2x2 and are ~2 GB GPU each. The
# 2016-grid (44x50) jrc2019_to_2016 is ~5 GB and last, so a GPU-OOM under
# contention (use --keep-going) does not block the rest.
CONFIGURATIONS = (
    {"id": "jrc2019_to_2013", "impact": "JRC2019", "target": "2013", "flows": "2016"},
    {"id": "jrc2019_flows2013", "impact": "JRC2019", "target": "2013", "flows": "2013"},
    {"id": "wiod2013_flows2016", "impact": "WIOD2013", "target": "2013", "flows": "2016"},
    {"id": "wiod2013_native", "impact": "WIOD2013", "target": "2013", "flows": "2013"},
    # Mixed cells above separate the environmental-extension effect from the
    # input-output (flow-table) effect:
    #   jrc2019_to_2013 vs jrc2019_flows2013  -> flow-table effect (same JRC EA)
    #   wiod2013_flows2016 vs wiod2013_native -> flow-table effect (same WIOD2013 EA)
    #   jrc2019_to_2013 vs wiod2013_flows2016 -> extension effect (same 2016 flows)
    #   jrc2019_flows2013 vs wiod2013_native  -> extension effect (same 2013 flows)
    {"id": "jrc2019_to_2016", "impact": "JRC2019", "target": "2016", "flows": "2016"},
    # --- Phase 1: genuine WIOD2013-published flow tables (SDA_FLOW_TABLE_SOURCE=WIOD),
    #     the 41x35 grid only. Completes the WIOD2013-flows column of the
    #     3 emissions x 3 flow-tables matrix (FULL_FLOW_EMISSION_MATRIX_PLAN.md).
    {"id": "wiod2013_true_native", "impact": "WIOD2013", "target": "2013", "flows": "2013", "flow_table": "WIOD"},
    {"id": "jrc2019_wiodflows2013", "impact": "JRC2019", "target": "2013", "flows": "2013", "flow_table": "WIOD"},
    {"id": "exio_wiodflows2013", "impact": "EXIOBASE", "target": "2013", "flows": "2013", "flow_table": "WIOD"},
    # --- Phase 2 (D-2026-037): genuine WIOD2016-PUBLISHED flow tables (44x56),
    #     aggregated 56->50 for the WIOD2016 grid and 56->35 for the WIOD2013
    #     grid. Completes the flow x emission x aggregation matrix
    #     (R3 1x1 / WIOD2016 2x2 / WIOD2013 3x3).
    {"id": "exio_wiod2016flows_2016", "impact": "EXIOBASE", "target": "2016", "flows": "2016", "flow_table": "WIOD"},
    {"id": "jrc2019_wiod2016flows_2016", "impact": "JRC2019", "target": "2016", "flows": "2016", "flow_table": "WIOD"},
    {"id": "exio_wiod2016flows_2013", "impact": "EXIOBASE", "target": "2013", "flows": "2016", "flow_table": "WIOD"},
    {"id": "jrc2019_wiod2016flows_2013", "impact": "JRC2019", "target": "2013", "flows": "2016", "flow_table": "WIOD"},
    {"id": "wiod2013_wiod2016flows_2013", "impact": "WIOD2013", "target": "2013", "flows": "2016", "flow_table": "WIOD"},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_passed(path: Path) -> dict:
    payload = json.loads(path.read_text())
    passed = payload.get("passed", payload.get("gate", {}).get("passed"))
    if passed is not True:
        raise RuntimeError(f"Required gate has not passed: {path}")
    return payload


def write_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hierarchy-gate", type=Path, required=True)
    parser.add_argument("--impact-gate", action="append", type=Path, required=True)
    parser.add_argument("--year-first", type=int, default=2001)
    parser.add_argument("--year-last", type=int, default=2009)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--campaign-id", default="native-wiod-co2-2001-2009-v1")
    # Direct-intensity sanitization policy. Defaults reproduce v1 (native modes
    # disable both). Set to the EXIOBASE values (100 / 1e9) to apply the
    # sector-median replacement and absolute cap to the native accounts too.
    parser.add_argument("--v-sector-outlier-multiple", default="inf")
    parser.add_argument("--v-intensity-max", default="inf")
    parser.add_argument("--keep-going", action="store_true",
                        help="record a failed run and continue instead of stopping "
                             "the campaign (e.g. a transient GPU OOM)")
    parser.add_argument("--only", action="append",
                        help="restrict to these configuration ids (repeatable)")
    args = parser.parse_args()

    read_passed(args.hierarchy_gate)
    for gate in args.impact_gate:
        read_passed(gate)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "campaign_manifest.json"
    main_script = args.repo / "SDA" / "main.py"
    tracked = [
        main_script, args.repo / "SDA" / "config.py", args.repo / "SDA" / "data_loader.py",
        args.repo / "scripts" / "native_wiod_impacts.py",
        args.repo / "native_wiod_config" / "native_wiod_sources.json",
        args.repo / "native_wiod_config" / "native_wiod_2016_56_to_thesis.csv",
        args.hierarchy_gate, *args.impact_gate,
    ]
    manifest = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "target_years": list(range(args.year_first, args.year_last + 1)),
            "transition_definition": "target_year-1 to target_year",
            "units": "kg CO2",
            "indicator_boundary": "industry CO2 only; household direct emissions excluded",
            "samples": args.samples,
            "price_basis": "CYP_(t-1) to PYP_t",
            "configurations": list(CONFIGURATIONS),
            "intensity_policy": {
                "v_sector_outlier_multiple": args.v_sector_outlier_multiple,
                "v_intensity_max": args.v_intensity_max,
            },
        },
        "comparability": {
            "main_campaign_indicator": "IPCC AR4 GWP100 multi-gas basket, kg CO2-equivalent",
            "this_campaign_indicator": "CO2 only, kg CO2",
            "like_for_like_with_main_campaign": False,
            "wiod2016_to_2013_non_nested_assumption": "C13-C15 is allocated equally to WIOD2013 17t18 and 19 by the reviewed existing crosswalk; report as harmonization assumption, not pure aggregation.",
        },
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in tracked],
        "runs": [],
    }

    configurations = [c for c in CONFIGURATIONS if not args.only or c["id"] in args.only]
    this_run_ids = {
        f"{y-1}_{y}_{c['id']}"
        for c in configurations
        for y in range(args.year_first, args.year_last + 1)
    }
    if manifest_path.exists():
        # resume/extend: keep prior run records for configs not in this invocation
        prior = json.loads(manifest_path.read_text())
        manifest["runs"] = [r for r in prior.get("runs", []) if r["run_id"] not in this_run_ids]
        if manifest["runs"]:
            manifest["started_at_utc"] = prior.get("started_at_utc", manifest["started_at_utc"])
            # An extension invocation normally covers only the newly required
            # years.  Preserve the complete campaign window in the replacement
            # manifest instead of narrowing ``scope.target_years`` to that
            # invocation's range.
            prior_years = prior.get("scope", {}).get("target_years", [])
            manifest["scope"]["target_years"] = sorted(
                {int(year) for year in prior_years}
                | set(manifest["scope"]["target_years"])
            )
            manifest["extension_provenance"] = {
                "prior_campaign_id": prior.get("campaign_id"),
                "prior_manifest_sha256": sha256(manifest_path),
                "retained_prior_run_count": len(manifest["runs"]),
            }
    write_atomic(manifest_path, manifest)

    failed = False
    for configuration in configurations:
        for year in range(args.year_first, args.year_last + 1):
            run_id = f"{year-1}_{year}_{configuration['id']}"
            output = args.output_root / run_id
            log = args.output_root / f"{run_id}.log"
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(args.repo / "SDA"),
                "SDA_RESULTS_DIR": str(args.input_root),
                "SDA_OUTPUT_DIR": str(output),
                "SDA_IMPACT_DATABASE": configuration["impact"],
                "SDA_WIOT_VERSION": configuration["target"],
                "SDA_FLOW_SOURCE_WIOT_VERSION": configuration["flows"],
                "SDA_NUM_SAMPLES": str(args.samples),
                "SDA_GPU_DEVICE": str(args.gpu),
                "SDA_V_SECTOR_OUTLIER_MULTIPLE": args.v_sector_outlier_multiple,
                "SDA_V_INTENSITY_MAX": args.v_intensity_max,
                "SDA_FLOW_TABLE_SOURCE": configuration.get("flow_table", "EXIOBASE"),
            })
            command = [sys.executable, str(main_script), str(year)]
            record = {
                "run_id": run_id,
                "configuration": configuration,
                "transition": f"{year-1}->{year}",
                "command": command,
                "output": str(output),
                "log": str(log),
                "status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            manifest["runs"].append(record)
            write_atomic(manifest_path, manifest)
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=args.repo / "SDA", env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
            record["exit_code"] = completed.returncode
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            record["status"] = "complete" if completed.returncode == 0 else "failed"
            write_atomic(manifest_path, manifest)
            if completed.returncode != 0:
                failed = True
                if not args.keep_going:
                    break
        if failed and not args.keep_going:
            break
    manifest["status"] = "failed" if failed else "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_atomic(manifest_path, manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
