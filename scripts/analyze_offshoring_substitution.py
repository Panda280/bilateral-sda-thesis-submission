#!/usr/bin/env python
"""Screen sourcing-interaction cubes for offshoring-substitution signatures.

The analysis is descriptive accounting, not a causal offshoring estimate.  For
each transition, resolution, effect, destination group ``d``, and destination
sector/product ``s``, it constructs (all quantities remain in kg CO2):

``NP_XB[d,s]``
    Cross-border contribution sourced from the Non-Participant group.
``Domestic[d,s]``
    Domestic contribution whose source group equals destination group ``d``.
``SameGroup[d,s]``
    Domestic plus cross-border contributions whose source group equals ``d``.

The strict signature is

``NP_XB > tolerance AND Domestic < -tolerance AND SameGroup < -tolerance``.

Two looser flags retain the domestic-only and same-group-only comparisons.
``Matched_Overlap = min(NP_XB, -SameGroup)`` for strict cells and zero
otherwise.  It is a descriptive magnitude, not an additive decomposition or a
causal estimate.  D_C is collapsed over Source_Sector to destination
group-sector cells; D_YF already has one Product_Sector axis.

Balanced mode analyzes the intersection of completed transitions across every
selected resolution/effect.  Full mode analyzes the requested range, or the
union of discovered transitions, and fails after writing ``coverage.json`` if
any selected product is missing or invalid.  A completed product requires both
its Parquet file and its JSON report, kg CO2 units, the expected effect, and a
passing reconciliation record.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KG_PER_MT = 1_000_000_000.0
RUN_DIRECTORY = re.compile(
    r"^(?P<start_year>\d{4})_(?P<end_year>\d{4})_(?P<resolution>.+)$"
)


@dataclass(frozen=True)
class ProductSpec:
    effect: str
    directory: str
    filename: str
    report_filename: str
    sector_column: str
    sector_role: str


PRODUCTS = {
    "D_C": ProductSpec(
        effect="D_C",
        directory="source_group_sector_interaction",
        filename="D_C.source_destination_group_sector_interaction.parquet",
        report_filename="source_group_sector_interaction_report.json",
        sector_column="Destination_Sector",
        sector_role="destination_sector",
    ),
    "D_YF": ProductSpec(
        effect="D_YF",
        directory="yf_group_product_interaction",
        filename="D_YF.source_destination_group_product_interaction.parquet",
        report_filename="yf_group_product_interaction_report.json",
        sector_column="Product_Sector",
        sector_role="product_sector",
    ),
}

REQUIRED_COMMON_COLUMNS = {
    "Relationship", "Source_Group", "Destination_Group", "Value"
}


def _transition_label(transition: tuple[int, int]) -> str:
    return f"{transition[0]}_{transition[1]}"


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _records(frame: pd.DataFrame) -> list[dict]:
    """Return JSON-safe records without leaking numpy scalar types."""

    return json.loads(frame.to_json(orient="records"))


def _product_status(path: Path, report_path: Path, effect: str) -> tuple[str, str]:
    if not path.is_file():
        return "missing", f"missing product: {path}"
    if not report_path.is_file():
        return "invalid", f"missing report: {report_path}"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", f"unreadable report {report_path}: {exc}"
    if report.get("effect") != effect:
        return "invalid", f"report effect is {report.get('effect')!r}, expected {effect!r}"
    if report.get("units") != "kg CO2":
        return "invalid", f"report units are {report.get('units')!r}, expected 'kg CO2'"
    reconciliation = report.get("reconciliation")
    if not isinstance(reconciliation, dict):
        return "invalid", "report has no reconciliation object"
    if reconciliation.get("failing_index") is not None:
        return "invalid", f"report reconciliation failed at {reconciliation['failing_index']!r}"
    return "complete", ""


def discover_products(run_root: Path) -> tuple[dict, set[str], set[tuple[int, int]]]:
    """Inventory all run directories and validate known interaction products."""

    inventory: dict[tuple[str, str, tuple[int, int]], dict] = {}
    resolutions: set[str] = set()
    transitions: set[tuple[int, int]] = set()
    if not run_root.is_dir():
        raise ValueError(f"run root does not exist or is not a directory: {run_root}")
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        match = RUN_DIRECTORY.match(run_dir.name)
        if match is None:
            continue
        start_year = int(match.group("start_year"))
        end_year = int(match.group("end_year"))
        if end_year != start_year + 1:
            continue
        resolution = match.group("resolution")
        transition = (start_year, end_year)
        resolutions.add(resolution)
        transitions.add(transition)
        for effect, spec in PRODUCTS.items():
            product_path = run_dir / spec.directory / spec.filename
            report_path = run_dir / spec.directory / spec.report_filename
            status, reason = _product_status(product_path, report_path, effect)
            inventory[(resolution, effect, transition)] = {
                "resolution": resolution,
                "effect": effect,
                "start_year": start_year,
                "end_year": end_year,
                "status": status,
                "reason": reason,
                "product_path": str(product_path.resolve()),
                "report_path": str(report_path.resolve()),
            }
    return inventory, resolutions, transitions


def select_transitions(
    inventory: dict,
    *,
    resolutions: tuple[str, ...],
    effects: tuple[str, ...],
    mode: str,
    discovered_transitions: set[tuple[int, int]],
    start_year: int | None,
    end_year: int | None,
) -> tuple[tuple[tuple[int, int], ...], list[dict], tuple[tuple[int, int], ...]]:
    """Select transitions and return (selected, missing, candidate universe)."""

    if (start_year is None) != (end_year is None):
        raise ValueError("--start-year and --end-year must be supplied together")
    if start_year is not None:
        if end_year <= start_year:
            raise ValueError("--end-year must be greater than --start-year")
        candidates = tuple((year, year + 1) for year in range(start_year, end_year))
    else:
        candidates = tuple(sorted(discovered_transitions))
    if not candidates:
        raise ValueError("no annual transitions were discovered or requested")

    complete_sets = []
    for resolution in resolutions:
        for effect in effects:
            complete_sets.append({
                transition for transition in candidates
                if inventory.get((resolution, effect, transition), {}).get("status") == "complete"
            })
    if mode == "balanced":
        selected_set = set.intersection(*complete_sets) if complete_sets else set()
        selected = tuple(sorted(selected_set))
        if not selected:
            raise ValueError("no balanced completed transition exists across selected products")
    elif mode == "full":
        selected = candidates
    else:  # pragma: no cover - argparse and callers constrain this
        raise ValueError(f"unsupported mode: {mode}")

    missing = []
    for transition in selected:
        for resolution in resolutions:
            for effect in effects:
                record = inventory.get((resolution, effect, transition))
                if record is None:
                    spec = PRODUCTS[effect]
                    expected_root = f"{transition[0]}_{transition[1]}_{resolution}"
                    missing.append({
                        "resolution": resolution,
                        "effect": effect,
                        "start_year": transition[0],
                        "end_year": transition[1],
                        "status": "missing",
                        "reason": f"run directory or product not discovered: {expected_root}/{spec.directory}/{spec.filename}",
                    })
                elif record["status"] != "complete":
                    missing.append(record)
    return selected, missing, candidates


def _component(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    sector_column: str,
    index: pd.MultiIndex,
    name: str,
) -> pd.Series:
    return (
        frame.loc[mask]
        .groupby(["Destination_Group", sector_column], observed=False)["Value"]
        .sum()
        .reindex(index, fill_value=0.0)
        .rename(name)
    )


def analyze_product(
    path: Path,
    *,
    resolution: str,
    effect: str,
    transition: tuple[int, int],
    destination_groups: tuple[str, ...],
    zero_tolerance_kg: float,
) -> pd.DataFrame:
    """Build one complete destination-cell table in kg CO2."""

    spec = PRODUCTS[effect]
    frame = pd.read_parquet(path)
    required = REQUIRED_COMMON_COLUMNS | {spec.sector_column}
    if effect == "D_C":
        required.add("Source_Sector")
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    if not pd.api.types.is_numeric_dtype(frame["Value"]):
        raise ValueError(f"{path} Value column is not numeric")
    if not np.isfinite(frame["Value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} Value column contains non-finite values")
    available_groups = set(frame["Destination_Group"].astype(str).unique())
    absent_groups = sorted(set(destination_groups) - available_groups)
    if absent_groups:
        raise ValueError(f"{path} lacks requested destination groups: {absent_groups}")

    domestic_mismatch = (
        (frame["Relationship"] == "Domestic")
        & (frame["Source_Group"] != frame["Destination_Group"])
        & (frame["Value"].abs() > zero_tolerance_kg)
    )
    if domestic_mismatch.any():
        raise ValueError(
            f"{path} contains {int(domestic_mismatch.sum())} material Domestic rows "
            "whose source and destination groups differ"
        )

    frame = frame.copy()
    frame[spec.sector_column] = frame[spec.sector_column].astype(str)
    sectors = tuple(sorted(frame[spec.sector_column].unique()))
    index = pd.MultiIndex.from_product(
        [destination_groups, sectors],
        names=["Destination_Group", spec.sector_column],
    )
    components = pd.concat(
        [
            _component(
                frame,
                (frame["Relationship"] == "Cross-border")
                & (frame["Source_Group"] == "Non-Participant"),
                sector_column=spec.sector_column,
                index=index,
                name="np_cross_border_kg_co2",
            ),
            _component(
                frame,
                (frame["Relationship"] == "Domestic")
                & (frame["Source_Group"] == frame["Destination_Group"]),
                sector_column=spec.sector_column,
                index=index,
                name="domestic_kg_co2",
            ),
            _component(
                frame,
                frame["Source_Group"] == frame["Destination_Group"],
                sector_column=spec.sector_column,
                index=index,
                name="same_group_kg_co2",
            ),
            _component(
                frame,
                (frame["Relationship"] == "Cross-border")
                & (frame["Source_Group"] == frame["Destination_Group"]),
                sector_column=spec.sector_column,
                index=index,
                name="same_group_cross_border_kg_co2",
            ),
            _component(
                frame,
                pd.Series(True, index=frame.index),
                sector_column=spec.sector_column,
                index=index,
                name="all_contributions_kg_co2",
            ),
        ],
        axis=1,
    ).reset_index()
    components = components.rename(columns={spec.sector_column: "sector_code"})
    components.insert(0, "resolution", resolution)
    components.insert(1, "effect", effect)
    components.insert(2, "sector_role", spec.sector_role)
    components.insert(3, "start_year", transition[0])
    components.insert(4, "end_year", transition[1])
    components = components.rename(columns={"Destination_Group": "destination_group"})

    positive = components["np_cross_border_kg_co2"] > zero_tolerance_kg
    domestic_negative = components["domestic_kg_co2"] < -zero_tolerance_kg
    same_group_negative = components["same_group_kg_co2"] < -zero_tolerance_kg
    components["np_cross_border_positive"] = positive
    components["domestic_signature"] = positive & domestic_negative
    components["same_group_signature"] = positive & same_group_negative
    components["strict_signature"] = positive & domestic_negative & same_group_negative
    components["matched_overlap_kg_co2"] = np.where(
        components["strict_signature"],
        np.minimum(
            components["np_cross_border_kg_co2"],
            -components["same_group_kg_co2"],
        ),
        0.0,
    )
    kg_columns = [column for column in components if column.endswith("_kg_co2")]
    for column in kg_columns:
        components[column.removesuffix("_kg_co2") + "_mt_co2"] = (
            components[column] / KG_PER_MT
        )
    return components


def build_annual_summary(cell_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["resolution", "effect", "start_year", "end_year"]
    for key, full_group in cell_year.groupby(keys, sort=True):
        for destination_group, group in [
            ("All", full_group),
            *list(full_group.groupby("destination_group", sort=True)),
        ]:
            strict = group.loc[group["strict_signature"]]
            positive_total = group["np_cross_border_kg_co2"].clip(lower=0).sum()
            strict_np = strict["np_cross_border_kg_co2"].sum()
            rows.append({
                **dict(zip(keys, key, strict=True)),
                "destination_group": destination_group,
                "eligible_cells": len(group),
                "np_positive_cells": int(group["np_cross_border_positive"].sum()),
                "domestic_signature_cells": int(group["domestic_signature"].sum()),
                "same_group_signature_cells": int(group["same_group_signature"].sum()),
                "strict_signature_cells": len(strict),
                "strict_cell_share": len(strict) / len(group) if len(group) else None,
                "positive_np_cross_border_kg_co2": positive_total,
                "strict_np_cross_border_kg_co2": strict_np,
                "strict_domestic_kg_co2": strict["domestic_kg_co2"].sum(),
                "strict_same_group_kg_co2": strict["same_group_kg_co2"].sum(),
                "strict_matched_overlap_kg_co2": strict["matched_overlap_kg_co2"].sum(),
                "positive_np_capture_share": strict_np / positive_total if positive_total else None,
            })
    result = pd.DataFrame(rows)
    for column in [name for name in result if name.endswith("_kg_co2")]:
        result[column.removesuffix("_kg_co2") + "_mt_co2"] = result[column] / KG_PER_MT
    return result


def build_pooled_cell_summary(cell_year: pd.DataFrame, zero_tolerance_kg: float) -> pd.DataFrame:
    keys = ["resolution", "effect", "sector_role", "destination_group", "sector_code"]
    value_columns = [
        "np_cross_border_kg_co2", "domestic_kg_co2", "same_group_kg_co2",
        "same_group_cross_border_kg_co2", "all_contributions_kg_co2",
    ]
    pooled = cell_year.groupby(keys, as_index=False, sort=True)[value_columns].sum()
    years = cell_year.groupby(keys, as_index=False).agg(
        transitions_analyzed=("strict_signature", "size"),
        strict_years=("strict_signature", "sum"),
    )
    annual_hits = (
        cell_year.loc[cell_year["strict_signature"]]
        .groupby(keys, as_index=False)[
            ["np_cross_border_kg_co2", "domestic_kg_co2", "same_group_kg_co2",
             "matched_overlap_kg_co2"]
        ]
        .sum()
        .rename(columns={
            "np_cross_border_kg_co2": "strict_year_np_cross_border_kg_co2",
            "domestic_kg_co2": "strict_year_domestic_kg_co2",
            "same_group_kg_co2": "strict_year_same_group_kg_co2",
            "matched_overlap_kg_co2": "strict_year_matched_overlap_kg_co2",
        })
    )
    pooled = pooled.merge(years, on=keys, validate="one_to_one")
    pooled = pooled.merge(annual_hits, on=keys, how="left", validate="one_to_one").fillna(0)
    positive = pooled["np_cross_border_kg_co2"] > zero_tolerance_kg
    domestic_negative = pooled["domestic_kg_co2"] < -zero_tolerance_kg
    same_negative = pooled["same_group_kg_co2"] < -zero_tolerance_kg
    pooled["pooled_domestic_signature"] = positive & domestic_negative
    pooled["pooled_same_group_signature"] = positive & same_negative
    pooled["pooled_strict_signature"] = positive & domestic_negative & same_negative
    pooled["pooled_matched_overlap_kg_co2"] = np.where(
        pooled["pooled_strict_signature"],
        np.minimum(pooled["np_cross_border_kg_co2"], -pooled["same_group_kg_co2"]),
        0.0,
    )
    for column in [name for name in pooled if name.endswith("_kg_co2")]:
        pooled[column.removesuffix("_kg_co2") + "_mt_co2"] = pooled[column] / KG_PER_MT
    return pooled


def build_consistency_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    return (
        pooled.groupby(["resolution", "effect", "transitions_analyzed", "strict_years"], as_index=False)
        .size()
        .rename(columns={"size": "cell_count"})
        .sort_values(["resolution", "effect", "strict_years"], ascending=[True, True, False])
    )


def _write_frame(frame: pd.DataFrame, stem: Path, formats: tuple[str, ...]) -> list[str]:
    written = []
    if "csv" in formats:
        path = stem.with_suffix(".csv")
        frame.to_csv(path, index=False)
        written.append(str(path.resolve()))
    if "parquet" in formats:
        path = stem.with_suffix(".parquet")
        try:
            frame.to_parquet(path, index=False)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Parquet output requires pandas with pyarrow; install the project's parquet extra"
            ) from exc
        written.append(str(path.resolve()))
    return written


def _coverage_payload(
    *,
    run_root: Path,
    mode: str,
    resolutions: tuple[str, ...],
    effects: tuple[str, ...],
    candidates: Iterable[tuple[int, int]],
    selected: Iterable[tuple[int, int]],
    missing: list[dict],
    inventory: dict,
) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root.resolve()),
        "mode": mode,
        "resolutions": list(resolutions),
        "effects": list(effects),
        "candidate_transitions": [_transition_label(item) for item in candidates],
        "selected_transitions": [_transition_label(item) for item in selected],
        "missing_selected_products": missing,
        "inventory": sorted(
            inventory.values(),
            key=lambda row: (row["start_year"], row["resolution"], row["effect"]),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-root", required=True, type=Path,
                        help="Campaign offshoring directory containing YEAR_YEAR_RESOLUTION runs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("balanced", "full"), default="balanced")
    parser.add_argument("--resolutions", nargs="+",
                        help="Resolution labels; default discovers all labels")
    parser.add_argument("--effects", nargs="+", choices=tuple(PRODUCTS),
                        default=tuple(PRODUCTS))
    parser.add_argument("--start-year", type=int,
                        help="First transition start year (requires --end-year)")
    parser.add_argument("--end-year", type=int,
                        help="Final data year; transitions end at END_YEAR")
    parser.add_argument("--destination-groups", nargs="+", default=("Red", "Non-Red", "Non-Participant"),
                        help="Destinations to screen; defaults include all three analytical groups")
    parser.add_argument("--zero-tolerance-kg", type=float, default=0.0,
                        help="Absolute sign threshold in kg CO2 (default: exact zero)")
    parser.add_argument("--formats", nargs="+", choices=("csv", "parquet"),
                        default=("csv", "parquet"))
    args = parser.parse_args(argv)
    if args.zero_tolerance_kg < 0:
        parser.error("--zero-tolerance-kg must be non-negative")

    try:
        inventory, discovered_resolutions, discovered_transitions = discover_products(
            args.run_root.resolve()
        )
        resolutions = tuple(args.resolutions or sorted(discovered_resolutions))
        if not resolutions:
            raise ValueError("no resolutions were discovered or selected")
        effects = tuple(dict.fromkeys(args.effects))
        selected, missing, candidates = select_transitions(
            inventory,
            resolutions=resolutions,
            effects=effects,
            mode=args.mode,
            discovered_transitions=discovered_transitions,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = _coverage_payload(
        run_root=args.run_root,
        mode=args.mode,
        resolutions=resolutions,
        effects=effects,
        candidates=candidates,
        selected=selected,
        missing=missing,
        inventory=inventory,
    )
    _json_write(args.output_dir / "coverage.json", coverage)
    if missing:
        print(
            f"error: {len(missing)} selected products are missing or invalid; "
            f"see {args.output_dir / 'coverage.json'}",
            file=sys.stderr,
        )
        return 2

    frames = []
    try:
        for transition in selected:
            for resolution in resolutions:
                for effect in effects:
                    record = inventory[(resolution, effect, transition)]
                    frames.append(analyze_product(
                        Path(record["product_path"]),
                        resolution=resolution,
                        effect=effect,
                        transition=transition,
                        destination_groups=tuple(args.destination_groups),
                        zero_tolerance_kg=args.zero_tolerance_kg,
                    ))
        cell_year = pd.concat(frames, ignore_index=True)
        annual = build_annual_summary(cell_year)
        pooled = build_pooled_cell_summary(cell_year, args.zero_tolerance_kg)
        consistency = build_consistency_summary(pooled)
        output_files = []
        for name, frame in [
            ("cell_year_signatures", cell_year),
            ("annual_summary", annual),
            ("pooled_cell_summary", pooled),
            ("consistency_summary", consistency),
        ]:
            output_files.extend(_write_frame(frame, args.output_dir / name, tuple(args.formats)))
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "mode": args.mode,
        "units": {
            "internal_and_primary_output": "kg CO2",
            "derived_display_unit": "Mt CO2",
            "kg_per_mt": KG_PER_MT,
        },
        "zero_tolerance_kg": args.zero_tolerance_kg,
        "definitions": {
            "np_cross_border": "sum(Value) where Relationship=Cross-border and Source_Group=Non-Participant",
            "domestic": "sum(Value) where Relationship=Domestic and Source_Group=Destination_Group",
            "same_group": "sum(Value) where Source_Group=Destination_Group across both relationships",
            "domestic_signature": "NP_XB > tolerance and Domestic < -tolerance",
            "same_group_signature": "NP_XB > tolerance and SameGroup < -tolerance",
            "strict_signature": "domestic_signature and same_group_signature",
            "matched_overlap": "min(NP_XB, -SameGroup) for strict cells; zero otherwise",
            "interpretation": "descriptive accounting sign pattern; not causal evidence of offshoring substitution",
        },
        "resolutions": list(resolutions),
        "effects": list(effects),
        "destination_groups": list(args.destination_groups),
        "transitions": [_transition_label(item) for item in selected],
        "output_files": output_files,
        "annual_summary": _records(annual),
        "consistency_summary": _records(consistency),
    }
    _json_write(args.output_dir / "summary.json", summary)
    print(
        f"Analyzed {len(selected)} transitions, {len(resolutions)} resolutions, "
        f"and {len(effects)} effects; outputs: {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
