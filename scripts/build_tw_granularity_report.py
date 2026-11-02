"""Build reproducible D_T/D_w and controlled-granularity diagnostics.

The command reads canonical SDA publications through ``load_result`` (which
verifies stored checksums), requires an explicit exhaustive concordance, and
publishes only descriptive accounting diagnostics. It does not attach an
economic meaning to T or w.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sda.analysis.granularity import diagnose_aggregation
from sda.analysis.tw import summarize_tw
from sda.output.storage import load_result

EFFECTS = ("D_A_x", "D_T_coef", "D_w_coef")
CONCORDANCE_COLUMNS = (
    "fine_country", "fine_sector", "coarse_country", "coarse_sector"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_concordance(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(CONCORDANCE_COLUMNS) <= set(reader.fieldnames):
            raise ValueError(f"concordance must contain {CONCORDANCE_COLUMNS}")
        mapping: dict[tuple[str, str], tuple[str, str]] = {}
        for line, row in enumerate(reader, 2):
            fine = (row["fine_country"].strip(), row["fine_sector"].strip())
            coarse = (row["coarse_country"].strip(), row["coarse_sector"].strip())
            if not all((*fine, *coarse)):
                raise ValueError(f"blank concordance value at line {line}")
            if fine in mapping and mapping[fine] != coarse:
                raise ValueError(f"ambiguous concordance for {fine}")
            mapping[fine] = coarse
    if not mapping:
        raise ValueError("concordance is empty")
    return mapping


def _groups(labels: tuple[object, ...], mapping: dict[tuple[str, str], tuple[str, str]], role: str):
    normalized: list[tuple[str, str]] = []
    for label in labels:
        if not isinstance(label, tuple) or len(label) != 2:
            raise ValueError(f"{role} labels must be (country, sector) tuples")
        normalized.append((str(label[0]), str(label[1])))
    missing = sorted(set(normalized) - set(mapping))
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"concordance misses {len(missing)} {role} labels: {preview}")
    return [mapping[label] for label in normalized]


def _parse_run_id(run_id: str) -> tuple[int, int, str]:
    value = run_id.removeprefix("sda_")
    parts = value.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"cannot parse run id {run_id!r}; expected YEAR_YEAR_RESOLUTION")
    return int(parts[0]), int(parts[1]), parts[2]


def build(run_dirs: list[Path], concordance: Path, output: Path, *, units: str,
          price_basis: str, tolerance: float, provisional: bool,
          indicator: str = "fossil-origin CO2", expected_runs: int | None = None,
          command: list[str] | None = None) -> Path:
    if expected_runs is not None and len(run_dirs) != expected_runs:
        raise ValueError(f"required inventory incomplete: expected {expected_runs}, found {len(run_dirs)}")
    mapping = _read_concordance(concordance)
    staging = output.parent / f".staging-{output.name}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    try:
        for run_dir in sorted(run_dirs):
            result = load_result(run_dir)
            run_id = run_dir.name
            previous, current, resolution = _parse_run_id(run_id)
            t = result.effects["D_T_coef"]
            w = result.effects["D_w_coef"]
            parent = result.effects["D_A_x"]
            if t.dimensions != w.dimensions or t.dimensions != parent.dimensions:
                raise ValueError(f"{run_id}: nested effects do not share dimensions")
            residual = np.asarray(t.baseline) + np.asarray(w.baseline) - np.asarray(parent.baseline)
            scale = max(1.0, float(np.max(np.abs(parent.baseline))))
            max_error = float(np.max(np.abs(residual)))
            if max_error > tolerance * scale:
                raise ValueError(f"{run_id}: D_T_coef + D_w_coef reconstruction failed: {max_error}")
            dims = parent.dimensions
            if dims.column is None:
                raise ValueError(f"{run_id}: structural effects must be matrices")
            row_groups = _groups(dims.row.labels, mapping, "emitter")
            column_groups = _groups(dims.column.labels, mapping, "structural-driver")
            summary = asdict(summarize_tw(t.baseline, w.baseline))
            base = {
                "run_id": run_id, "year_previous": previous, "year_current": current,
                "resolution": resolution, "units": units, "price_basis": price_basis,
                "provisional": provisional,
            }
            rows.append({**base, "record_type": "tw_pair", "effect_id": "D_T_coef+D_w_coef",
                         "max_reconstruction_error": max_error, **summary})
            for effect_id in EFFECTS:
                diagnostic = asdict(diagnose_aggregation(
                    result.effects[effect_id].baseline, row_groups, column_groups
                ))
                if abs(diagnostic["signed_conservation_error"]) > tolerance * max(
                    1.0, abs(diagnostic["fine_signed_total"])
                ):
                    raise ValueError(f"{run_id}: aggregation conservation failed for {effect_id}")
                rows.append({**base, "record_type": "granularity", "effect_id": effect_id,
                             "max_reconstruction_error": max_error, **diagnostic})
            manifest_path = run_dir / "manifest.json"
            sources.append({"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)})

        fields = sorted({key for row in rows for key in row})
        csv_path = staging / "tw_granularity_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        report = staging / "tw_granularity_report.md"
        report.write_text(
            "# Nested T/w and controlled-granularity diagnostics\n\n"
            f"Status: **{'provisional' if provisional else 'validated'}**. "
            f"Runs: {len(run_dirs)}. Units: `{units}`. Price basis: `{price_basis}`.\n\n"
            "The machine-readable CSV reports the nested reconstruction, signed totals, "
            "gross cancellation, sign reversals, concentration changes and sparse-edge loss. "
            "T and w retain their implementation-defined names; no causal or standalone "
            "economic interpretation is asserted. D_T_coef and D_w_coef are children of "
            "D_A_x and are not additional top-level effects.\n",
            encoding="utf-8",
        )
        artifacts = {p.name: _sha256(p) for p in (csv_path, report)}
        try:
            git_state = subprocess.run(
                ["git", "describe", "--always", "--dirty"], cwd=Path(__file__).resolve().parents[1],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_state = "unavailable"
        manifest = {
            "schema_version": 1, "analysis_id": "nested-tw-controlled-granularity-v1",
            "status": "complete", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": {"script": "scripts/build_tw_granularity_report.py",
                          "git_commit": git_state, "command": command or []},
            "scope": {"price_basis": price_basis, "indicator": indicator,
                      "units_input": units, "units_output": units,
                      "effects": list(EFFECTS),
                      "resolutions": sorted({_parse_run_id(p.name)[2] for p in run_dirs}),
                      "year_previous_min": min(_parse_run_id(p.name)[0] for p in run_dirs),
                      "year_current_max": max(_parse_run_id(p.name)[1] for p in run_dirs)},
            "coverage": {"expected": expected_runs or len(run_dirs), "accepted": len(run_dirs),
                         "missing": [], "excluded": []},
            "run_count": len(run_dirs), "row_count": len(rows), "tolerance": tolerance,
            "concordance": {"path": str(concordance.resolve()), "sha256": _sha256(concordance),
                            "required_columns": list(CONCORDANCE_COLUMNS), "mapped_labels": len(mapping)},
            "inputs": [{"role": "canonical-manifest", **source} for source in sources] +
                      [{"role": "concordance", "path": str(concordance.resolve()),
                        "sha256": _sha256(concordance)}],
            "outputs": [{"role": "result-table", "path": csv_path.name,
                         "sha256": artifacts[csv_path.name], "rows": len(rows)},
                        {"role": "report", "path": report.name,
                         "sha256": artifacts[report.name], "rows": None}],
            "validation": {"passed": True,
                           "upstream_input_status": "accepted_provisional" if provisional else "passed",
                           "thesis_status": "accepted_provisional" if provisional else "eligible",
                           "gates": {
                "canonical_checksums_verified": True, "nested_reconstruction_passed": True,
                "concordance_exhaustive_for_used_labels": True,
                "controlled_aggregation_conserved_signed_totals": True,
                "required_inventory_complete": expected_runs is None or len(run_dirs) == expected_runs,
            }},
            "limitations": [
                "T_coef and w_coef have no sourced standalone economic definition in this repository.",
                "Provisional runs retain all upstream CYP/PYP validation limitations.",
                "Aggregation diagnostics are descriptive and do not identify database quality or causality.",
            ],
        }
        (staging / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        staging.replace(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--run-glob", default="sda_*")
    parser.add_argument("--concordance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--price-basis", required=True)
    parser.add_argument("--indicator", default="fossil-origin CO2")
    parser.add_argument("--expected-runs", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args(argv)
    runs = [p for p in args.canonical_root.glob(args.run_glob) if (p / "manifest.json").is_file()]
    if not runs:
        parser.error("no canonical runs matched")
    published = build(runs, args.concordance, args.output, units=args.units,
                      price_basis=args.price_basis, tolerance=args.tolerance,
                      provisional=args.provisional, indicator=args.indicator,
                      expected_runs=args.expected_runs, command=sys.argv)
    print(published)
    return 0


if __name__ == "__main__":
    sys.exit(main())
