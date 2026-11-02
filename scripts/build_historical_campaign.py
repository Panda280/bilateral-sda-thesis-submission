#!/usr/bin/env python
"""Run a resumable 2000-2014 matrix, bundle, and canonical-SDA campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from build_wiod_pyp_cyp import (
    DATABASES,
    Layout,
    build_command,
    bundle_artifacts,
    bundle_command,
    expected_build_artifacts,
)


class CampaignError(RuntimeError):
    pass


SOURCING_PRODUCT_FILES = {
    "D_C_source_country": Path("source_country/D_C.source_country_cube.parquet"),
    "D_C_source_group_sector": Path("source_group_sector/D_C.source_group_sector_cube.parquet"),
    "D_YF_source_country": Path("yf_source_country/D_YF.source_country_cube.parquet"),
    "D_YF_source_group_sector": Path("yf_source_group_sector/D_YF.source_group_sector_cube.parquet"),
}


_INPUT_FAILURE_PATTERNS = {
    "gras_convergence": re.compile(
        r"(?:GRAS\s+(?:did\s+not|failed\s+to)\s+converge|GRAS\s+non[- ]convergence|"
        r"GRAS.{0,30}(?:converged|convergence)[\s:=]+false)", re.I
    ),
    "leontief_reproduction": re.compile(
        r"(?:(?:Leontief|reproduction|x\s*=\s*L\s*[·*]\s*y).{0,80}(?:failed|failure)|"
        r"(?:failed|failure).{0,80}(?:Leontief|reproduction|x\s*=\s*L\s*[·*]\s*y))", re.I
    ),
    "row_balance": re.compile(r"row[- ]balance(?:\s+(?:check|test))?\s+(?:failed|failure)", re.I),
    "column_balance": re.compile(r"column[- ]balance(?:\s+(?:check|test))?\s+(?:failed|failure)", re.I),
}


def assess_input_build(year: int, summary_path: Path, log_path: Path) -> dict:
    """Classify upstream MRIO validity without altering its numerical policy.

    Artifact presence and downstream SDA reconciliation are deliberately not
    evidence that the input system is balanced. Known fatal diagnostics are
    read from the preserved build log. A build with no known failure remains
    unverified unless its pipeline summary explicitly records all four gates.
    """
    evidence_files: list[str] = []
    text_parts: list[str] = []
    if log_path.is_file():
        evidence_files.append(str(log_path.resolve()))
        text_parts.append(log_path.read_text(encoding="utf-8", errors="replace"))
    summary = None
    if summary_path.is_file():
        evidence_files.append(str(summary_path.resolve()))
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            text_parts.append(json.dumps(summary, sort_keys=True))
        except (OSError, json.JSONDecodeError):
            summary = None
    evidence = "\n".join(text_parts)
    required = ("gras_convergence", "leontief_reproduction", "row_balance", "column_balance")
    explicit = summary.get("input_validation") if isinstance(summary, dict) else None
    explicit_checks = explicit.get("checks", {}) if isinstance(explicit, dict) else {}
    failed_gates = [
        name for name, pattern in _INPUT_FAILURE_PATTERNS.items()
        if pattern.search(evidence) or explicit_checks.get(name) is False
    ]
    passed_gates = [name for name in required if explicit_checks.get(name) is True]
    if failed_gates:
        status = "failed"
    elif len(passed_gates) == len(required):
        status = "passed"
    else:
        status = "unverified"
    return {
        "schema_version": 1,
        "year": year,
        "status": status,
        "thesis_eligible": status == "passed",
        "required_gates": list(required),
        "passed_gates": passed_gates,
        "failed_gates": failed_gates,
        "evidence_files": evidence_files,
        "note": (
            "This status concerns upstream MRIO balance and reproduction; it is distinct "
            "from canonical SDA reconciliation and artifact completeness."
        ),
    }


def write_input_validation_inventory(run_root: Path, years: range) -> dict:
    reports = []
    for year in years:
        build_dir = run_root / "build" / str(year)
        report = assess_input_build(
            year,
            build_dir / f"pipeline_summary_{year}_in_{year - 1}.json",
            run_root / "logs" / f"build_{year}.log",
        )
        _write_json(build_dir / "input_validation_status.json", report)
        reports.append(report)
    counts = {status: sum(item["status"] == status for item in reports) for status in ("passed", "failed", "unverified")}
    inventory = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if counts["failed"] else ("passed" if counts["passed"] == len(reports) else "unverified"),
        "thesis_eligible": counts["passed"] == len(reports),
        "counts": counts,
        "builds": reports,
    }
    _write_json(run_root / "input_validation_inventory.json", inventory)
    return inventory


def require_no_failed_input_builds(inventory: dict, *, allow_failed: bool = False) -> None:
    failed = [str(item["year"]) for item in inventory["builds"] if item["status"] == "failed"]
    if failed and not allow_failed:
        raise CampaignError(
            "upstream input validation failed for build year(s) " + ", ".join(failed)
            + "; downstream artifacts are not accepted"
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise CampaignError(f"command failed ({completed.returncode}); see {log_path}")


def _complete(directory: Path, names: tuple[str, ...]) -> bool:
    return all((directory / name).is_file() for name in names)


def _sourcing_product_complete(product: str, paths: tuple[Path, ...]) -> bool:
    if not all(path.is_file() for path in paths):
        return False
    if not product.startswith("yf-"):
        return True
    try:
        report = json.loads(paths[1].read_text(encoding="utf-8"))
    except (OSError, ValueError, IndexError):
        return False
    return (
        report.get("sourcing_resolution") == "consumer_country"
        and "Demanded_Sector" not in report.get("columns", ())
    )


def _campaign_layout(root: Path, year_prev: int, year: int) -> Layout:
    # Layout's build properties are exactly build/<year>; using the campaign
    # root lets every adjacent pair share the same one-time yearly builds.
    return Layout(root, year, year_prev)


def _publication_command(args: argparse.Namespace, bundle: Path, run_id: str) -> list[str]:
    return [
        str(args.python), str(args.sda_repo / "scripts" / "build_production_run.py"),
        "--bundle-dir", str(bundle),
        "--output-root", str(args.run_root / "sda"),
        "--run-id", run_id,
    ]


def _sourcing_command(
    args: argparse.Namespace, bundle: Path, output: Path, run_id: str,
    products: tuple[str, ...] = (
        "source-country", "source-group-sector",
        "yf-source-country", "yf-source-group-sector",
    ),
) -> list[str]:
    return [
        str(args.python), str(args.sda_repo / "scripts" / "build_offshoring_reports.py"),
        "--bundle-dir", str(bundle),
        "--output-dir", str(output),
        "--run-id", run_id,
        "--products", *products,
        "--backend", getattr(args, "sourcing_backend", "cupy"),
        "--file-format", "parquet",
        "--chunk-rows", str(getattr(args, "sourcing_chunk_rows", 100_000)),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--exiodeflator-repo", required=True, type=Path)
    parser.add_argument("--sda-repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--stage", choices=("plan", "audit", "build", "bundle", "sda", "sourcing", "report", "panel", "all"),
        default="plan",
    )
    parser.add_argument(
        "--include-sourcing", action="store_true",
        help="Include D_C intermediate-input and D_YF final-demand source-country/group-sector products in plan/all.",
    )
    parser.add_argument(
        "--allow-failed-input-validation", action="store_true",
        help=(
            "Permit downstream computational stages despite recorded upstream input failures. "
            "The failures remain recorded and outputs remain ineligible for thesis use."
        ),
    )
    parser.add_argument("--system", choices=("pxp", "ixi"), default="pxp")
    parser.add_argument("--gras-backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--sourcing-backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--sourcing-workers", type=int, default=1)
    parser.add_argument("--sourcing-chunk-rows", type=int, default=100_000)
    parser.add_argument("--gras-max-iter", type=int, default=500)
    parser.add_argument("--gras-rtol", type=float, default=1e-4)
    parser.add_argument("--gras-atol", type=float)
    parser.add_argument(
        "--gras-small-col-frac", type=float, default=0.0,
        help="Use 0 for the full-matrix GRAS solve; nonzero values enable the legacy thin-column shortcut.",
    )
    parser.add_argument("--gras-check-every", type=int, default=5)
    parser.add_argument(
        "--gras-relaxation", type=float, default=1.5,
        help="Log-space GRAS over-relaxation; 1.5 accelerates slow row/column alternation.",
    )
    parser.add_argument(
        "--gras-infeasible-target-materiality", type=float, default=1e-4,
        help="Maximum sign-infeasible residual share (default 0.0001 = 0.01%% of system margins).",
    )
    parser.add_argument(
        "--cuda-device", default="0",
        help="Single physical GPU exposed to all campaign subprocesses (default: 0).",
    )
    parser.add_argument("--impact", default="CO2")
    parser.add_argument("--units", default="kg CO2")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--wiod-dir", type=Path)
    parser.add_argument("--exio-doi")
    args = parser.parse_args(argv)
    if not args.cuda_device.isdigit():
        parser.error("--cuda-device must identify one non-negative GPU index")
    if args.start_year < 2000 or args.end_year > 2014 or args.start_year >= args.end_year:
        parser.error("campaign range must satisfy 2000 <= start-year < end-year <= 2014")
    args.run_root = args.run_root.resolve()
    args.exiodeflator_repo = args.exiodeflator_repo.resolve()
    args.sda_repo = args.sda_repo.resolve()
    args.download_dir = (args.download_dir or args.run_root / "cache" / "exiobase").resolve()
    args.wiod_dir = (args.wiod_dir or args.run_root / "cache" / "wiod").resolve()
    concordance = args.exiodeflator_repo / "concordance"
    args.deflator_file = concordance / "price_deflators_ultimate.xlsx"
    args.deflator_concordance = concordance / "deflator_exio_concordance.csv"
    args.wiod_concordance = concordance / "EXIOBASE3_product_to_WIOD_concordance.xlsx"
    args.wiod_concordance_sheet = "Product_to_WIOD_concordance"
    args.r3_concordance = concordance / "EXIOBASE3_R3_aggregation.xlsx"
    args.country_concordance = concordance / "countries_concordance.csv"
    args.weights_file = None
    args.export_script = args.exiodeflator_repo / "scripts" / "export_sda_state_bundle.py"
    return args


def campaign_plan(args: argparse.Namespace) -> dict:
    years = list(range(args.start_year, args.end_year + 1))
    transitions = list(zip(years[:-1], years[1:]))
    builds = {
        str(year): build_command(args, year, year - 1, args.run_root / "build" / str(year))
        for year in years
    }
    bundles = {}
    publications = {}
    sourcing = {}
    reports = {}
    for previous, year in transitions:
        layout = _campaign_layout(args.run_root, previous, year)
        for database in DATABASES:
            key = f"{previous}_{year}_{database}"
            bundles[key] = bundle_command(args, layout, database)
            publications[key] = _publication_command(args, layout.bundle(database), f"sda_{key}")
            sourcing[key] = _sourcing_command(
                args, layout.bundle(database), args.run_root / "offshoring" / key,
                f"sourcing_{key}",
            )
            reports[key] = [
                str(args.python), str(args.sda_repo / "scripts" / "build_annual_reports.py"), "annual",
                "--canonical-run", str(args.run_root / "sda" / f"sda_{key}"),
                "--bundle-dir", str(layout.bundle(database)),
                "--sourcing-dir", str(args.run_root / "offshoring" / key),
                "--output-root", str(args.run_root / "annual"), "--run-id", key,
                "--year-prev", str(previous), "--year", str(year),
                "--resolution", database, "--units", getattr(args, "units", "kg CO2"),
            ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "range": [args.start_year, args.end_year],
        "matrix_builds": len(builds),
        "transitions": len(transitions),
        "resolutions": list(DATABASES),
        "canonical_sda_runs": len(publications),
        "sourcing_enabled": args.include_sourcing,
        "sourcing_products": len(sourcing) * 4 if args.include_sourcing else 0,
        "sourcing_products_by_effect": {
            "D_C": len(sourcing) * 2 if args.include_sourcing else 0,
            "D_YF": len(sourcing) * 2 if args.include_sourcing else 0,
        },
        "annual_reports": len(reports),
        "execution": {
            "cuda_visible_devices": args.cuda_device,
            "gpu_concurrency": 1,
            "matrix_build_concurrency": 1,
            "sda_concurrency": 1,
        },
        "official_benchmark_coverage": {
            "WIOT2013_PYP": [args.start_year, min(args.end_year, 2009)],
            "WIOT2016_PYP": [max(args.start_year, 2001), args.end_year],
            "note": "benchmark availability does not limit EXIOBASE-derived SDA resolutions",
        },
        "commands": {
            "builds": builds,
            "bundles": bundles,
            "publications": publications,
            "sourcing": sourcing if args.include_sourcing else {},
            "reports": reports,
            "panel": [str(args.python), str(args.sda_repo / "scripts" / "build_annual_reports.py"), "panel",
                      "--annual-root", str(args.run_root / "annual"), "--output-root", str(args.run_root / "panel"),
                      "--run-id", "full_panel"],
        },
    }


def campaign_inventory(run_root: Path, expected_runs: int, sourcing_expected: bool) -> dict:
    """Return explicit observed/expected completion for every publication layer."""
    canonical = len(list((run_root / "sda").glob("*/manifest.json")))
    annual = len(list((run_root / "annual").glob("*/manifest.v2.json")))
    sourcing = {
        name: len(list((run_root / "offshoring").glob(f"*/{relative.as_posix()}")))
        for name, relative in SOURCING_PRODUCT_FILES.items()
    }
    panel = int((run_root / "panel" / "full_panel" / "panel_summary.v2.json").is_file())
    expected_sourcing = expected_runs if sourcing_expected else 0
    observed = {
        "canonical": canonical,
        "D_C_source_country": sourcing["D_C_source_country"],
        "D_C_source_group_sector": sourcing["D_C_source_group_sector"],
        "D_YF_source_country": sourcing["D_YF_source_country"],
        "D_YF_source_group_sector": sourcing["D_YF_source_group_sector"],
        "annual": annual,
        "panel": panel,
    }
    expected = {
        "canonical": expected_runs,
        "D_C_source_country": expected_sourcing,
        "D_C_source_group_sector": expected_sourcing,
        "D_YF_source_country": expected_sourcing,
        "D_YF_source_group_sector": expected_sourcing,
        "annual": expected_runs,
        "panel": 1,
    }
    complete = {name: observed[name] == count for name, count in expected.items()}
    return {
        "expected": expected,
        "observed": observed,
        "complete": complete,
        "all_complete": all(complete.values()),
        "sourcing_totals": {
            "expected": expected_sourcing * 4,
            "observed": sum(sourcing.values()),
            "D_C_expected": expected_sourcing * 2,
            "D_C_observed": sourcing["D_C_source_country"] + sourcing["D_C_source_group_sector"],
            "D_YF_expected": expected_sourcing * 2,
            "D_YF_observed": sourcing["D_YF_source_country"] + sourcing["D_YF_source_group_sector"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Deliberately expose exactly one GPU. CuPy sees it as logical device 0;
    # the second physical RTX 3090 remains unavailable to this campaign.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    args.run_root.mkdir(parents=True, exist_ok=True)
    required = [
        args.python, args.exiodeflator_repo / "build_wiod_cyp_pyp.py", args.export_script,
        args.sda_repo / "scripts" / "build_production_run.py", args.deflator_file,
        args.deflator_concordance, args.wiod_concordance, args.r3_concordance,
        args.country_concordance,
    ]
    if args.include_sourcing or args.stage == "sourcing":
        required.append(args.sda_repo / "scripts" / "build_offshoring_reports.py")
    if args.stage in {"report", "panel", "all"}:
        required.append(args.sda_repo / "scripts" / "build_annual_reports.py")
    missing = [path for path in required if not path.is_file()]
    if missing:
        print("error: missing campaign dependencies: " + ", ".join(map(str, missing)), file=sys.stderr)
        return 1
    plan = campaign_plan(args)
    _write_json(args.run_root / "campaign_plan.json", plan)
    if args.stage == "plan":
        print(args.run_root / "campaign_plan.json")
        return 0

    try:
        years = range(args.start_year, args.end_year + 1)
        if args.stage in {"build", "all"}:
            for year in years:
                directory = args.run_root / "build" / str(year)
                expected = expected_build_artifacts(year, year - 1)
                if not _complete(directory, expected):
                    directory.mkdir(parents=True, exist_ok=True)
                    _run(build_command(args, year, year - 1, directory), args.exiodeflator_repo, args.run_root / "logs" / f"build_{year}.log")
                if not _complete(directory, expected):
                    raise CampaignError(f"year {year} build did not satisfy its artifact contract")

        input_inventory = write_input_validation_inventory(args.run_root, range(args.start_year, args.end_year + 1))
        if args.stage == "audit":
            print(args.run_root / "input_validation_inventory.json")
            return 0
        if args.stage in {"bundle", "sda", "sourcing", "report", "panel", "all"}:
            require_no_failed_input_builds(
                input_inventory, allow_failed=args.allow_failed_input_validation
            )

        transitions = zip(range(args.start_year, args.end_year), range(args.start_year + 1, args.end_year + 1))
        if args.stage in {"bundle", "all"}:
            for previous, year in transitions:
                layout = _campaign_layout(args.run_root, previous, year)
                for database in DATABASES:
                    directory = layout.bundle(database)
                    if not _complete(directory, bundle_artifacts()):
                        directory.mkdir(parents=True, exist_ok=True)
                        _run(bundle_command(args, layout, database), args.exiodeflator_repo, args.run_root / "logs" / f"bundle_{previous}_{year}_{database}.log")
                    if not _complete(directory, bundle_artifacts()):
                        raise CampaignError(f"bundle {previous}_{year}_{database} is incomplete")

        transitions = zip(range(args.start_year, args.end_year), range(args.start_year + 1, args.end_year + 1))
        if args.stage in {"sda", "all"}:
            for previous, year in transitions:
                layout = _campaign_layout(args.run_root, previous, year)
                for database in DATABASES:
                    run_id = f"sda_{previous}_{year}_{database}"
                    published = args.run_root / "sda" / run_id
                    if not (published / "manifest.json").is_file():
                        _run(_publication_command(args, layout.bundle(database), run_id), args.sda_repo, args.run_root / "logs" / f"{run_id}.log")
                    if not (published / "manifest.json").is_file():
                        raise CampaignError(f"canonical publication {run_id} is incomplete")

        run_sourcing = args.stage == "sourcing" or (
            args.stage == "all" and args.include_sourcing
        )
        transitions = zip(range(args.start_year, args.end_year), range(args.start_year + 1, args.end_year + 1))
        if run_sourcing:
            sourcing_jobs = []
            for previous, year in transitions:
                layout = _campaign_layout(args.run_root, previous, year)
                for database in DATABASES:
                    key = f"{previous}_{year}_{database}"
                    output = args.run_root / "offshoring" / key
                    expected = (
                        output / "source_country" / "D_C.source_country_cube.parquet",
                        output / "source_country" / "source_country_cube_report.json",
                        output / "source_group_sector" / "D_C.source_group_sector_cube.parquet",
                        output / "source_group_sector" / "source_group_sector_cube_report.json",
                        output / "yf_source_country" / "D_YF.source_country_cube.parquet",
                        output / "yf_source_country" / "yf_source_country_cube_report.json",
                        output / "yf_source_group_sector" / "D_YF.source_group_sector_cube.parquet",
                        output / "yf_source_group_sector" / "yf_source_group_sector_cube_report.json",
                        output / "source_group_sector_interaction" / "D_C.source_destination_group_sector_interaction.parquet",
                        output / "source_group_sector_interaction" / "source_group_sector_interaction_report.json",
                        output / "yf_group_product_interaction" / "D_YF.source_destination_group_product_interaction.parquet",
                        output / "yf_group_product_interaction" / "yf_group_product_interaction_report.json",
                    )
                    product_outputs = {
                        "source-country": expected[0:2],
                        "source-group-sector": (*expected[2:4], *expected[8:10]),
                        "yf-source-country": expected[4:6],
                        "yf-source-group-sector": (*expected[6:8], *expected[10:12]),
                    }
                    missing_products = tuple(
                        product for product, paths in product_outputs.items()
                        if not _sourcing_product_complete(product, paths)
                    )
                    if missing_products:
                        sourcing_jobs.append((key, expected, _sourcing_command(
                            args, layout.bundle(database), output, f"sourcing_{key}",
                            missing_products,
                        ), args.run_root / "logs" / f"sourcing_{key}.log"))

            workers = getattr(args, "sourcing_workers", 1)
            if workers <= 0:
                raise CampaignError("--sourcing-workers must be positive")

            def run_sourcing_job(job):
                key, expected, command, log = job
                _run(command, args.sda_repo, log)
                if not all(path.is_file() for path in expected):
                    raise CampaignError(f"sourcing products {key} are incomplete")

            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(run_sourcing_job, sourcing_jobs))

        transitions = zip(range(args.start_year, args.end_year), range(args.start_year + 1, args.end_year + 1))
        if args.stage in {"report", "all"}:
            for previous, year in transitions:
                layout = _campaign_layout(args.run_root, previous, year)
                for database in DATABASES:
                    key = f"{previous}_{year}_{database}"
                    output = args.run_root / "annual" / key
                    if not (output / "manifest.v2.json").is_file():
                        command = [str(args.python), str(args.sda_repo / "scripts" / "build_annual_reports.py"), "annual",
                            "--canonical-run", str(args.run_root / "sda" / f"sda_{key}"),
                            "--bundle-dir", str(layout.bundle(database)),
                            "--sourcing-dir", str(args.run_root / "offshoring" / key),
                            "--output-root", str(args.run_root / "annual"), "--run-id", key,
                            "--year-prev", str(previous), "--year", str(year),
                            "--resolution", database, "--units", args.units]
                        _run(command, args.sda_repo, args.run_root / "logs" / f"report_{key}.log")
                    if not (output / "manifest.v2.json").is_file():
                        raise CampaignError(f"annual report {key} is incomplete")

        if args.stage in {"panel", "all"}:
            output = args.run_root / "panel" / "full_panel"
            if not (output / "panel_summary.v2.json").is_file():
                _run([str(args.python), str(args.sda_repo / "scripts" / "build_annual_reports.py"), "panel",
                    "--annual-root", str(args.run_root / "annual"), "--output-root", str(args.run_root / "panel"),
                    "--run-id", "full_panel"], args.sda_repo, args.run_root / "logs" / "panel.log")
            if not (output / "panel_summary.v2.json").is_file():
                raise CampaignError("full panel is incomplete")

        expected_runs = (args.end_year - args.start_year) * len(DATABASES)
        inventory = campaign_inventory(args.run_root, expected_runs, args.include_sourcing)
        complete = {
            "schema_version": 1,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "range": [args.start_year, args.end_year],
            "resolutions": list(DATABASES),
            "matrix_builds": len(list((args.run_root / "build").glob("*/pipeline_summary_*.json"))),
            "canonical_sda_runs": len(list((args.run_root / "sda").glob("*/manifest.json"))),
            "sourcing_products": len(list((args.run_root / "offshoring").glob("*/*/*cube_report.json"))),
            "computational_status": "complete",
            "thesis_validation_status": input_inventory["status"],
            "thesis_eligible": input_inventory["thesis_eligible"],
            "failed_input_validation_override": args.allow_failed_input_validation,
            "input_validation_inventory": str((args.run_root / "input_validation_inventory.json").resolve()),
            "validation_policy": (
                "canonical publication reconciliation and upstream MRIO input validation are separate; "
                "campaign completion does not imply thesis eligibility"
            ),
            "annual_reports": len(list((args.run_root / "annual").glob("*/manifest.v2.json"))),
            "inventory": inventory,
            "status": "COMPLETE" if inventory["all_complete"] else "PARTIAL",
        }
        _write_json(args.run_root / "campaign_completion.json", complete)
        print(args.run_root / "campaign_completion.json")
        return 0
    except (OSError, CampaignError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
