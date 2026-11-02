#!/usr/bin/env python
"""Prepare a complete, reproducible CYP/PYP input set for bilateral SDA.

This is the canonical orchestration boundary around EXIODeflator's validated
``build_wiod_cyp_pyp.py`` engine.  It deliberately does not duplicate the
deflation, GRAS, concordance, aggregation, or WIOD parsing equations.

For a transition ``year_prev -> year`` it:

1. builds ``year_prev`` so its own-price CYP matrices exist;
2. builds ``year`` and its PYP matrices in ``year_prev`` prices;
3. verifies the complete artifact contract for EXIOBASE native, R3,
   WIOT2013, WIOT2016, and the two official WIOD benchmark datasets;
4. exports canonical bilateral-SDA bundles for WIOT2013, WIOT2016, and R3;
5. writes a checksummed manifest covering inputs, outputs, commands, and
   provenance.

The default mode is ``plan``: it validates configuration and writes the run
plan without starting a costly data build. Use ``--stage all`` to execute.
Existing complete stages are reused unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


DATABASES = ("2013", "2016", "R3")
WIOD_VERSIONS = (2013, 2016)


class PreparationError(RuntimeError):
    """The requested full-run input set cannot be prepared safely."""


@dataclass(frozen=True)
class Layout:
    root: Path
    year: int
    year_prev: int

    @property
    def previous(self) -> Path:
        return self.root / "build" / str(self.year_prev)

    @property
    def current(self) -> Path:
        return self.root / "build" / str(self.year)

    def bundle(self, database: str) -> Path:
        return self.root / "bundles" / f"{self.year_prev}_{self.year}_{database}"

    @property
    def manifest(self) -> Path:
        return self.root / "full_run_manifest.json"

    @property
    def plan(self) -> Path:
        return self.root / "run_plan.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, role: str) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256(resolved),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def expected_build_artifacts(year: int, year_prev: int) -> tuple[str, ...]:
    """Return the complete output contract for the current-period build."""
    names = [
        f"pipeline_summary_{year}_in_{year_prev}.json",
        f"deflator_ratio_{year}_to_{year_prev}.csv",
        f"Z_pyp_{year}_in_{year_prev}_prices.parquet",
        f"Y_pyp_{year}_in_{year_prev}_prices.parquet",
        f"X_pyp_{year}_in_{year_prev}_prices.csv",
        f"Z_pyp_{year}_in_{year_prev}_prices_balanced.parquet",
        f"A_pyp_{year}_in_{year_prev}_prices_balanced.parquet",
        f"L_pyp_{year}_in_{year_prev}_prices_balanced.parquet",
        f"EXIOBASE_CYP_{year}_R3_Z.parquet",
        f"EXIOBASE_CYP_{year}_R3_Y.parquet",
        f"EXIOBASE_PYP_{year}_in_{year_prev}_prices_R3_Z.parquet",
        f"EXIOBASE_PYP_{year}_in_{year_prev}_prices_R3_Y.parquet",
        f"EXIOBASE_CO2_{year}_R3.csv",
    ]
    for version in WIOD_VERSIONS:
        names.extend(
            (
                f"EXIOBASE_CYP_{year}_WIOT{version}_Z.parquet",
                f"EXIOBASE_CYP_{year}_WIOT{version}_Y.parquet",
                f"EXIOBASE_PYP_{year}_in_{year_prev}_prices_WIOT{version}_Z.parquet",
                f"EXIOBASE_PYP_{year}_in_{year_prev}_prices_WIOT{version}_Y.parquet",
                f"EXIOBASE_CO2_{year}_WIOT{version}.csv",
            )
        )
        # Official tables are benchmark-only. WIOD2013 PYP ends in 2009;
        # WIOD2016 has no 2000 PYP because there is no 1999 price basis.
        benchmark_available = (version == 2013 and year <= 2009) or (
            version == 2016 and year >= 2001
        )
        if benchmark_available:
            names.extend(
                (
                    f"WIOD_CYP_{year}_WIOT{version}_Z.parquet",
                    f"WIOD_CYP_{year}_WIOT{version}_Y.parquet",
                    f"WIOD_PYP_{year}_in_{year_prev}_prices_WIOT{version}_Z.parquet",
                    f"WIOD_PYP_{year}_in_{year_prev}_prices_WIOT{version}_Y.parquet",
                )
            )
    return tuple(names)


def bundle_artifacts() -> tuple[str, ...]:
    return ("config.json", "state.npz", "bundle_manifest.json")


def _missing(directory: Path, names: Iterable[str]) -> list[Path]:
    return [directory / name for name in names if not (directory / name).is_file()]


def _require_files(paths: Iterable[Path], label: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = "\n  ".join(str(path) for path in missing)
        raise PreparationError(f"missing {label}:\n  {rendered}")


def build_command(args: argparse.Namespace, build_year: int, build_prev: int, out_dir: Path) -> list[str]:
    cmd = [
        str(args.python), str(args.exiodeflator_repo / "build_wiod_cyp_pyp.py"),
        "--year", str(build_year), "--year-prev", str(build_prev),
        "--system", args.system,
        "--download-dir", str(args.download_dir),
        "--wiod-dir", str(args.wiod_dir),
        "--out-dir", str(out_dir),
        "--deflator-file", str(args.deflator_file),
        "--concordance", str(args.deflator_concordance),
        "--wiod-concordance", str(args.wiod_concordance),
        "--wiod-concordance-sheet", args.wiod_concordance_sheet,
        "--r3-concordance", str(args.r3_concordance),
        "--country-concordance", str(args.country_concordance),
        "--wiod-versions", "2013", "2016",
        "--gras-backend", args.gras_backend,
        "--gras-max-iter", str(getattr(args, "gras_max_iter", 500)),
        "--gras-rtol", str(getattr(args, "gras_rtol", 1e-4)),
        "--gras-small-col-frac", str(getattr(args, "gras_small_col_frac", 0.0)),
        "--gras-check-every", str(getattr(args, "gras_check_every", 5)),
        "--gras-relaxation", str(getattr(args, "gras_relaxation", 1.5)),
        "--gras-infeasible-target-materiality", str(
            getattr(args, "gras_infeasible_target_materiality", 1e-4)
        ),
    ]
    if getattr(args, "gras_atol", None) is not None:
        cmd.extend(("--gras-atol", str(args.gras_atol)))
    if args.weights_file:
        cmd.extend(("--weights-file", str(args.weights_file)))
    if args.exio_doi:
        cmd.extend(("--exio-doi", args.exio_doi))
    return cmd


def bundle_command(args: argparse.Namespace, layout: Layout, database: str) -> list[str]:
    return [
        str(args.python), str(args.export_script),
        "--out-dir", str(layout.current),
        "--prev-out-dir", str(layout.previous),
        "--year", str(layout.year), "--year-prev", str(layout.year_prev),
        "--database", database,
        "--bundle-dir", str(layout.bundle(database)),
        "--exiobase-raw-dir", str(args.download_dir),
        "--sda-package-src", str(args.sda_repo / "src"),
        "--impact", args.impact,
        "--num-samples", str(args.num_samples),
    ]


def _run(command: Sequence[str], cwd: Path) -> None:
    completed = subprocess.run(list(command), cwd=cwd, check=False)
    if completed.returncode:
        raise PreparationError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--year-prev", type=int)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--stage", choices=("plan", "build", "bundle", "all"), default="plan")
    parser.add_argument("--force", action="store_true", help="Re-run complete stages; never deletes existing files.")
    parser.add_argument("--exiodeflator-repo", required=True, type=Path)
    parser.add_argument("--sda-repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--python", type=Path, help="Python executable with EXIODeflator, CuPy, and CUDA support.")
    parser.add_argument("--system", choices=("pxp", "ixi"), default="pxp")
    parser.add_argument("--gras-backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--gras-max-iter", type=int, default=500)
    parser.add_argument("--gras-rtol", type=float, default=1e-4)
    parser.add_argument("--gras-atol", type=float)
    parser.add_argument("--gras-small-col-frac", type=float, default=0.0)
    parser.add_argument("--gras-check-every", type=int, default=5)
    parser.add_argument("--gras-relaxation", type=float, default=1.5)
    parser.add_argument("--gras-infeasible-target-materiality", type=float, default=1e-4)
    parser.add_argument("--impact", default="CO2")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--exio-doi")
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--wiod-dir", type=Path)
    parser.add_argument("--deflator-file", type=Path)
    parser.add_argument("--deflator-concordance", type=Path)
    parser.add_argument("--wiod-concordance", type=Path)
    parser.add_argument("--wiod-concordance-sheet", default="Product_to_WIOD_concordance")
    parser.add_argument("--r3-concordance", type=Path)
    parser.add_argument("--country-concordance", type=Path)
    parser.add_argument("--weights-file", type=Path)
    args = parser.parse_args(argv)

    args.year_prev = args.year_prev or args.year - 1
    if args.year_prev >= args.year:
        parser.error("--year-prev must be earlier than --year")
    concordance = args.exiodeflator_repo / "concordance"
    args.python = args.python or args.exiodeflator_repo / ".venv" / "Scripts" / "python.exe"
    args.download_dir = args.download_dir or args.run_root / "cache" / "exiobase"
    args.wiod_dir = args.wiod_dir or args.run_root / "cache" / "wiod"
    args.deflator_file = args.deflator_file or concordance / "price_deflators_ultimate.xlsx"
    args.deflator_concordance = args.deflator_concordance or concordance / "deflator_exio_concordance.csv"
    args.wiod_concordance = args.wiod_concordance or concordance / "EXIOBASE3_product_to_WIOD_concordance.xlsx"
    args.r3_concordance = args.r3_concordance or concordance / "EXIOBASE3_R3_aggregation.xlsx"
    args.country_concordance = args.country_concordance or concordance / "countries_concordance.csv"
    args.kyoto_groups = concordance / "kyoto_country_groups.csv"
    args.export_script = args.exiodeflator_repo / "scripts" / "export_sda_state_bundle.py"
    return args


def _validate_configuration(args: argparse.Namespace) -> list[Path]:
    inputs = [
        args.exiodeflator_repo / "build_wiod_cyp_pyp.py", args.export_script,
        args.python, args.deflator_file, args.deflator_concordance,
        args.wiod_concordance, args.r3_concordance, args.country_concordance,
        args.kyoto_groups,
        args.sda_repo / "src" / "sda" / "__init__.py",
    ]
    if args.weights_file:
        inputs.append(args.weights_file)
    _require_files(inputs, "required programs or concordances")
    return inputs


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        inputs = _validate_configuration(args)
        layout = Layout(args.run_root.resolve(), args.year, args.year_prev)
        layout.root.mkdir(parents=True, exist_ok=True)
        commands = {
            "build_previous": build_command(args, args.year_prev, args.year_prev - 1, layout.previous),
            "build_current": build_command(args, args.year, args.year_prev, layout.current),
            "bundles": {database: bundle_command(args, layout, database) for database in DATABASES},
        }
        _write_json(layout.plan, {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "year": args.year, "year_prev": args.year_prev,
            "datasets": ["EXIOBASE", "WIOT2013", "WIOT2016"],
            "aggregated_variants": ["R3", "WIOT2013", "WIOT2016"],
            "commands": commands,
        })

        if args.stage in {"build", "all"}:
            for label, build_year, build_prev, directory in (
                ("previous", args.year_prev, args.year_prev - 1, layout.previous),
                ("current", args.year, args.year_prev, layout.current),
            ):
                directory.mkdir(parents=True, exist_ok=True)
                expected = expected_build_artifacts(build_year, build_prev)
                if args.force or _missing(directory, expected):
                    _run(build_command(args, build_year, build_prev, directory), args.exiodeflator_repo)
                remaining = _missing(directory, expected)
                if remaining:
                    raise PreparationError(f"{label} build incomplete; missing: " + ", ".join(p.name for p in remaining))

        if args.stage in {"bundle", "all"}:
            _require_files(
                [layout.previous / f"EXIOBASE_CYP_{args.year_prev}_{'R3' if d == 'R3' else 'WIOT' + d}_{m}.parquet"
                 for d in DATABASES for m in ("Z", "Y")],
                "previous-period CYP matrices",
            )
            for database in DATABASES:
                bundle_dir = layout.bundle(database)
                if args.force or _missing(bundle_dir, bundle_artifacts()):
                    bundle_dir.mkdir(parents=True, exist_ok=True)
                    _run(bundle_command(args, layout, database), args.exiodeflator_repo)
                remaining = _missing(bundle_dir, bundle_artifacts())
                if remaining:
                    raise PreparationError(f"{database} bundle incomplete: " + ", ".join(p.name for p in remaining))

        outputs = []
        for directory in (layout.previous, layout.current):
            if directory.is_dir():
                outputs.extend(path for path in directory.iterdir() if path.is_file())
        for database in DATABASES:
            directory = layout.bundle(database)
            if directory.is_dir():
                outputs.extend(path for path in directory.iterdir() if path.is_file())
        _write_json(layout.manifest, {
            "schema_version": 1,
            "status": "complete" if all(not _missing(layout.bundle(d), bundle_artifacts()) for d in DATABASES) else "planned_or_partial",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "year": args.year, "year_prev": args.year_prev,
            "inputs": [_record(path, path.name) for path in inputs],
            "outputs": [_record(path, path.name) for path in sorted(outputs)],
            "plan": str(layout.plan.resolve()),
        })
        print(layout.manifest)
        return 0
    except (OSError, PreparationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
