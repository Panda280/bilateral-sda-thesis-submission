#!/usr/bin/env python
"""Build opt-in source-country and source-group×sector reports from a bundle."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sda import (  # noqa: E402
    build_source_country_offshoring_cube,
    build_source_group_sector_offshoring_cube,
    build_yf_source_country_cube,
    build_yf_source_group_sector_cube,
    build_source_group_sector_interaction,
    build_yf_group_product_interaction,
)
from sda.attribution import EXTERNAL_SOURCE_GROUPS  # noqa: E402
from sda.effects.final_demand import three_factor_final_demand_components  # noqa: E402
from sda.output import (export_source_country_cube, export_source_group_sector_cube,
                        export_yf_source_country_cube, export_yf_source_group_sector_cube,
                        export_source_group_sector_interaction,
                        export_yf_group_product_interaction)  # noqa: E402


def _write_timing_profile(path: Path, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _profile_stage(profile: dict, path: Path, name: str):
    started = perf_counter()
    try:
        yield
    finally:
        profile["stages_seconds"][name] = perf_counter() - started
        _write_timing_profile(path, profile)


def _pair(label: object) -> tuple[str, str]:
    if isinstance(label, list) and len(label) == 2:
        return str(label[0]), str(label[1])
    if isinstance(label, str) and ":" in label:
        country, sector = label.split(":", 1)
        if country and sector:
            return country, sector
    raise ValueError(f"invalid country-sector label: {label!r}")


def _infer_external_groups(
    countries: tuple[str, ...], masks: dict[str, np.ndarray]
) -> dict[str, str]:
    """Recover the exact bundle classification from its authoritative masks."""

    unique = tuple(dict.fromkeys(countries))
    if len(unique) < 2:
        raise ValueError("at least two countries are required to infer external source groups")
    result = {}
    country_array = np.asarray(countries, dtype=object)
    for country in unique:
        row = int(np.flatnonzero(country_array == country)[0])
        external_columns = np.flatnonzero(country_array != country)
        matches = [
            group for group in EXTERNAL_SOURCE_GROUPS
            if masks[group][row, external_columns].all()
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot infer one external source group for {country}: {matches}")
        result[country] = matches[0]
    return result


def load_inputs(bundle_dir: Path) -> tuple[dict, dict]:
    config_path = bundle_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    semantics = config["semantics"]
    labels = tuple(_pair(label) for label in semantics["structural_driver_labels"])
    archive_path = bundle_dir / config["input_npz"]
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[member], copy=True) for name, member in config["arrays"].items()}
        masks = {
            group: np.array(archive[member], dtype=bool, copy=True)
            for group, member in semantics["c_source_masks"].items()
        }
    supplier_sector = arrays["supplier_sector"]
    y0 = (arrays["F0"] * arrays["B0"][supplier_sector, :] * arrays["levels0"][None, :]).sum(axis=1)
    y1 = (arrays["F1"] * arrays["B1"][supplier_sector, :] * arrays["levels1"][None, :]).sum(axis=1)
    countries = tuple(country for country, _ in labels)
    sectors = tuple(sector for _, sector in labels)
    common = {
        "C0": arrays["C0"], "C1": arrays["C1"],
        "A_x0": arrays["A_x0"], "A_x1": arrays["A_x1"],
        "v0": arrays["v0"], "delta_v": arrays["v1"] - arrays["v0"],
        "L0": arrays["L0"], "L1": arrays["L1"],
        "y0": y0, "delta_y": y1 - y0,
        "source_countries": countries, "destination_countries": countries,
        "emitter_labels": labels, "structural_driver_labels": labels,
    }
    extra = {
        "source_sectors": sectors,
        "source_group_by_country": _infer_external_groups(countries, masks),
        "manifest": bundle_dir / "bundle_manifest.json",
        "consumer_countries": tuple(semantics["consumer_labels"]),
        "demanded_sector_labels": tuple(semantics["demanded_sector_labels"]),
        "sectors_per_country": int(semantics["sectors_per_country"]),
        "yf_component": three_factor_final_demand_components(
            arrays["F0"], arrays["F1"], arrays["B0"], arrays["B1"],
            arrays["levels0"], arrays["levels1"], supplier_sector,
        ).YF,
        "final_demand_propagation": {
            "v0": arrays["v0"], "delta_v": arrays["v1"] - arrays["v0"],
            "L0": arrays["L0"], "L1": arrays["L1"],
        },
    }
    return common, extra


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--units", default="kg CO2")
    parser.add_argument("--products", nargs="+", choices=("source-country", "source-group-sector", "yf-source-country", "yf-source-group-sector"),
                        default=("source-country", "source-group-sector", "yf-source-country", "yf-source-group-sector"))
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--file-format", choices=("csv", "parquet"), default="parquet")
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--timing-profile", type=Path,
                        help="JSON timing profile (default: OUTPUT_DIR/wall_time_profile.json)")
    args = parser.parse_args(argv)

    profile_path = args.timing_profile or args.output_dir / "wall_time_profile.json"
    total_started = perf_counter()
    profile = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "backend": args.backend, "batch_size": args.batch_size,
            "file_format": args.file_format, "chunk_rows": args.chunk_rows,
            "products": list(args.products), "bundle_dir": str(args.bundle_dir.resolve()),
        },
        "stages_seconds": {},
    }
    _write_timing_profile(profile_path, profile)
    with _profile_stage(profile, profile_path, "load_inputs"):
        common, extra = load_inputs(args.bundle_dir.resolve())
    provenance = {"bundle_dir": str(args.bundle_dir.resolve()), "bundle_manifest": str(extra["manifest"].resolve())}
    options = {"backend": args.backend, "batch_size": args.batch_size, "metadata": provenance}
    if "source-country" in args.products:
        with _profile_stage(profile, profile_path, "source_country.build"):
            cube = build_source_country_offshoring_cube(**common, **options)
        with _profile_stage(profile, profile_path, "source_country.export"):
            export_source_country_cube(
                cube, args.output_dir / "source_country", run_id=args.run_id,
                units=args.units, provenance=provenance, file_format=args.file_format,
                chunk_rows=args.chunk_rows,
            )
    if "source-group-sector" in args.products:
        with _profile_stage(profile, profile_path, "source_group_sector.build"):
            cube = build_source_group_sector_offshoring_cube(
                **common,
                source_sectors=extra["source_sectors"],
                source_group_by_country=extra["source_group_by_country"],
                **options,
            )
        with _profile_stage(profile, profile_path, "source_group_sector.export"):
            export_source_group_sector_cube(
                cube, args.output_dir / "source_group_sector", run_id=args.run_id,
                units=args.units, provenance=provenance, file_format=args.file_format,
                chunk_rows=args.chunk_rows,
            )
        with _profile_stage(profile, profile_path, "source_group_sector_interaction.build"):
            interaction = build_source_group_sector_interaction(
                common["C0"], common["C1"], common["A_x0"], common["A_x1"],
                v0=common["v0"], delta_v=common["delta_v"],
                L0=common["L0"], L1=common["L1"], y0=common["y0"],
                delta_y=common["delta_y"], source_countries=common["source_countries"],
                source_sectors=extra["source_sectors"],
                destination_countries=common["destination_countries"],
                destination_sectors=extra["source_sectors"],
                country_group_by_country=extra["source_group_by_country"],
                metadata=provenance,
            )
        with _profile_stage(profile, profile_path, "source_group_sector_interaction.export"):
            export_source_group_sector_interaction(
                interaction, args.output_dir / "source_group_sector_interaction",
                run_id=args.run_id, units=args.units, provenance=provenance,
                file_format=args.file_format,
            )
    yf_common = {
        "yf_component": extra["yf_component"], **extra["final_demand_propagation"],
        "sectors_per_country": extra["sectors_per_country"],
        "source_countries": common["source_countries"],
        "consumer_countries": extra["consumer_countries"],
        "emitter_labels": common["emitter_labels"],
        "metadata": provenance,
        "backend": args.backend,
        "batch_size": args.batch_size,
    }
    if "yf-source-country" in args.products:
        with _profile_stage(profile, profile_path, "yf_source_country.build"):
            cube = build_yf_source_country_cube(**yf_common)
        with _profile_stage(profile, profile_path, "yf_source_country.export"):
            export_yf_source_country_cube(cube, args.output_dir / "yf_source_country",
                run_id=args.run_id, units=args.units, provenance=provenance,
                file_format=args.file_format, chunk_rows=args.chunk_rows)
    if "yf-source-group-sector" in args.products:
        with _profile_stage(profile, profile_path, "yf_source_group_sector.build"):
            cube = build_yf_source_group_sector_cube(**yf_common,
                source_sectors=extra["source_sectors"],
                source_group_by_country=extra["source_group_by_country"])
        with _profile_stage(profile, profile_path, "yf_source_group_sector.export"):
            export_yf_source_group_sector_cube(cube, args.output_dir / "yf_source_group_sector",
                run_id=args.run_id, units=args.units, provenance=provenance,
                file_format=args.file_format, chunk_rows=args.chunk_rows)
        with _profile_stage(profile, profile_path, "yf_group_product_interaction.build"):
            interaction = build_yf_group_product_interaction(
                cube, consumer_group_by_country=extra["source_group_by_country"],
                metadata=provenance,
            )
        with _profile_stage(profile, profile_path, "yf_group_product_interaction.export"):
            export_yf_group_product_interaction(
                interaction, args.output_dir / "yf_group_product_interaction",
                run_id=args.run_id, units=args.units, provenance=provenance,
                file_format=args.file_format,
            )
    profile["total_seconds"] = perf_counter() - total_started
    profile["finished_at"] = datetime.now(timezone.utc).isoformat()
    profile["status"] = "completed"
    _write_timing_profile(profile_path, profile)
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
