#!/usr/bin/env python
"""Validate strict source-sector-preserving offshoring substitution screens."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


GROUPS = ("Red", "Non-Red", "Non-Participant")
RELATIONSHIPS = ("Domestic", "Cross-border")
FLOAT32_EPS = 1.1920929e-7
RUN = re.compile(r"^(\d{4})_(\d{4})_(2013|2016|R3)$")
SCREENS = ("exact", "tolerance", "material", "strong")


@dataclass(frozen=True)
class Product:
    effect: str
    path: Path
    report: str
    columns: tuple[str, ...]
    pair_key: tuple[str, ...]
    gross_key: tuple[str, ...]


PRODUCTS = {
    "D_C": Product(
        "D_C",
        Path("source_group_sector_interaction")
        / "D_C.source_destination_group_sector_interaction.parquet",
        "source_group_sector_interaction_report.json",
        (
            "Relationship", "Source_Group", "Source_Sector",
            "Destination_Group", "Destination_Sector", "Value",
        ),
        ("Source_Sector", "Destination_Group", "Destination_Sector"),
        ("Destination_Group", "Destination_Sector"),
    ),
    "D_YF": Product(
        "D_YF",
        Path("yf_group_product_interaction")
        / "D_YF.source_destination_group_product_interaction.parquet",
        "yf_group_product_interaction_report.json",
        (
            "Relationship", "Source_Group", "Destination_Group",
            "Product_Sector", "Value",
        ),
        ("Destination_Group", "Product_Sector"),
        ("Destination_Group", "Product_Sector"),
    ),
}


class InteractionAuditError(ValueError):
    pass


def numerical_floor(values: Any, multiplier: float = 1.0) -> float:
    array = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(array), initial=0.0))
    return multiplier * 256.0 * FLOAT32_EPS * max(1.0, scale)


def validate_frame(frame: pd.DataFrame, product_spec: Product) -> dict[str, Any]:
    if tuple(frame.columns) != product_spec.columns:
        raise InteractionAuditError(f"{product_spec.effect} column schema differs")
    if not np.isfinite(frame.Value.to_numpy(dtype=np.float64)).all():
        raise InteractionAuditError(f"{product_spec.effect} contains non-finite values")
    if set(frame.Relationship) != set(RELATIONSHIPS):
        raise InteractionAuditError("relationship labels differ")
    if set(frame.Source_Group) != set(GROUPS) or set(frame.Destination_Group) != set(GROUPS):
        raise InteractionAuditError("absolute country-group labels differ")
    if frame.duplicated(list(product_spec.columns[:-1])).any():
        raise InteractionAuditError("duplicate interaction cells")
    if product_spec.effect == "D_C":
        source_sectors = set(frame.Source_Sector)
        destination_sectors = set(frame.Destination_Sector)
        if source_sectors != destination_sectors:
            raise InteractionAuditError("D_C source/destination sector sets differ")
        sectors = len(source_sectors)
        expected = 2 * 3 * sectors * 3 * sectors
    else:
        sectors = frame.Product_Sector.nunique()
        expected = 2 * 3 * 3 * sectors
    if len(frame) != expected:
        raise InteractionAuditError(
            f"incomplete Cartesian interaction table: {len(frame)} != {expected}"
        )
    invalid_domestic = frame[
        (frame.Relationship == "Domestic")
        & (frame.Source_Group != frame.Destination_Group)
    ]
    if np.any(invalid_domestic.Value.to_numpy(dtype=np.float64) != 0.0):
        raise InteractionAuditError("nonzero domestic value crosses absolute groups")
    return {"rows": len(frame), "sectors": sectors}


def validate_manifest(path: Path, frame: pd.DataFrame, product_spec: Product) -> dict[str, Any]:
    report_path = path.parent / product_spec.report
    if not report_path.is_file():
        raise InteractionAuditError(f"missing report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reconciliation = report.get("reconciliation", {})
    checks = {
        "effect": report.get("effect") == product_spec.effect,
        "file": report.get("file") == path.name,
        "columns": tuple(report.get("columns", ())) == product_spec.columns,
        "units": report.get("units") == "kg CO2",
        "shape": int(np.prod(report.get("shape", ()), dtype=np.int64)) == len(frame),
        "total": np.isclose(
            float(report.get("total", np.nan)), float(frame.Value.sum()), rtol=1e-12
        ),
        "reconciliation": reconciliation.get("failing_index") is None,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise InteractionAuditError(f"invalid {product_spec.effect} manifest checks: {failed}")
    return {
        "units": report["units"],
        "shape": report["shape"],
        "reconciliation_max_abs_error": reconciliation.get("max_abs_error"),
    }


def screen_pairs(
    frame: pd.DataFrame,
    product_spec: Product,
    *,
    destination_group: str = "Red",
    materiality_share: float = 0.01,
    balance_min: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare exact cells without collapsing D_C source sector."""

    if destination_group not in GROUPS:
        raise InteractionAuditError("unknown destination group")
    selected = frame[frame.Destination_Group == destination_group]
    keys = list(product_spec.pair_key)
    gross_keys = list(product_spec.gross_key)
    gross = (
        selected.assign(gross_kg=selected.Value.abs())
        .groupby(gross_keys, as_index=False).gross_kg.sum()
    )
    foreign = selected[
        (selected.Relationship == "Cross-border")
        & (selected.Source_Group == "Non-Participant")
    ][keys + ["Value"]].rename(columns={"Value": "foreign_kg"})
    domestic = selected[
        (selected.Relationship == "Domestic")
        & (selected.Source_Group == selected.Destination_Group)
    ][keys + ["Value"]].rename(columns={"Value": "domestic_kg"})
    pairs = foreign.merge(domestic, on=keys, validate="one_to_one")
    pairs = pairs.merge(gross, on=gross_keys, validate="many_to_one")
    pairs["positive_kg"] = pairs.foreign_kg.clip(lower=0.0)
    pairs["decline_kg"] = (-pairs.domestic_kg).clip(lower=0.0)
    pairs["matched_kg"] = np.minimum(pairs.positive_kg, pairs.decline_kg)
    largest = np.maximum(pairs.positive_kg, pairs.decline_kg)
    pairs["balance"] = np.divide(
        pairs.matched_kg, largest,
        out=np.zeros(len(pairs)), where=largest > 0.0,
    )
    pairs["gross_share"] = np.divide(
        pairs.matched_kg, pairs.gross_kg,
        out=np.zeros(len(pairs)), where=pairs.gross_kg > 0.0,
    )
    floor = numerical_floor(frame.Value)
    pairs["exact"] = (pairs.positive_kg > 0.0) & (pairs.decline_kg > 0.0)
    pairs["tolerance"] = (pairs.positive_kg > floor) & (pairs.decline_kg > floor)
    pairs["material"] = pairs.tolerance & (pairs.gross_share >= materiality_share)
    pairs["strong"] = pairs.material & (pairs.balance >= balance_min)
    summary: dict[str, Any] = {
        "numerical_floor_kg": floor,
        "screens": {},
        "floor_sensitivity": {},
    }
    for screen in SCREENS:
        subset = pairs[pairs[screen]]
        summary["screens"][screen] = {
            "count": len(subset),
            "matched_kg": float(subset.matched_kg.sum()),
            "matched_mt": float(subset.matched_kg.sum() / 1.0e9),
        }
    for multiplier in (0.0, 0.25, 1.0, 4.0):
        threshold = numerical_floor(frame.Value, multiplier)
        subset = pairs[
            (pairs.positive_kg > threshold) & (pairs.decline_kg > threshold)
        ]
        summary["floor_sensitivity"][str(multiplier)] = {
            "threshold_kg": threshold,
            "count": len(subset),
            "matched_kg": float(subset.matched_kg.sum()),
        }
    summary["false_small_sign"] = {
        "removed_count": (
            summary["screens"]["exact"]["count"]
            - summary["screens"]["tolerance"]["count"]
        ),
        "removed_matched_kg": (
            summary["screens"]["exact"]["matched_kg"]
            - summary["screens"]["tolerance"]["matched_kg"]
        ),
    }
    return pairs, summary


def persistence(records: list[dict[str, Any]], screen: str) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if record[screen]:
            key = (record["effect"], record["resolution"], *record["cell_key"])
            grouped.setdefault(key, []).append(record)
    streaks = []
    for key, values in grouped.items():
        current = []
        for value in sorted(values, key=lambda item: item["previous_year"]):
            if current and value["previous_year"] != current[-1]["current_year"]:
                if len(current) >= 2:
                    streaks.append(_streak(key, current))
                current = []
            current.append(value)
        if len(current) >= 2:
            streaks.append(_streak(key, current))
    streaks.sort(key=lambda item: (-item["length"], -item["matched_kg"]))
    return {
        "streak_count": len(streaks),
        "cell_count": len({tuple(item["cell_key"]) for item in streaks}),
        "max_streak": max((item["length"] for item in streaks), default=0),
        "top": streaks[:20],
    }


def _streak(key: tuple[Any, ...], values: list[dict[str, Any]]) -> dict[str, Any]:
    effect, resolution, *cell = key
    matched = sum(item["matched_kg"] for item in values)
    return {
        "effect": effect,
        "resolution": resolution,
        "cell_key": list(cell),
        "start_year": values[0]["previous_year"],
        "end_year": values[-1]["current_year"],
        "length": len(values),
        "matched_kg": matched,
        "matched_mt": matched / 1.0e9,
        "minimum_gross_share": min(item["gross_share"] for item in values),
        "minimum_balance": min(item["balance"] for item in values),
    }


def audit(
    offshoring_root: Path,
    validation_inventory: Path,
    *,
    destination_group: str = "Red",
    materiality_share: float = 0.01,
    balance_min: float = 0.5,
) -> dict[str, Any]:
    inventory = (
        json.loads(validation_inventory.read_text(encoding="utf-8"))
        if validation_inventory.is_file() else {}
    )
    thesis_eligible = bool(inventory.get("thesis_eligible", False))
    expected = {
        f"{year}_{year + 1}_{resolution}"
        for year, resolution in product(range(2000, 2014), ("2013", "2016", "R3"))
    }
    observed = {effect: set() for effect in PRODUCTS}
    runs = []
    records = []
    aggregate = {
        effect: {
            resolution: {
                "transition_count": 0,
                "screens": {name: {"count": 0, "matched_mt": 0.0} for name in SCREENS},
                "false_small_sign": {"removed_count": 0, "removed_matched_mt": 0.0},
            }
            for resolution in ("2013", "2016", "R3")
        }
        for effect in PRODUCTS
    }
    for effect, spec in PRODUCTS.items():
        for path in sorted(offshoring_root.glob(f"*/{spec.path.as_posix()}")):
            run_id = path.parents[1].name
            match = RUN.fullmatch(run_id)
            if match is None or int(match.group(2)) != int(match.group(1)) + 1:
                raise InteractionAuditError(f"invalid annual run id: {run_id}")
            previous, current, resolution = int(match.group(1)), int(match.group(2)), match.group(3)
            frame = pd.read_parquet(path)
            schema = validate_frame(frame, spec)
            persisted = validate_manifest(path, frame, spec)
            pairs, screened = screen_pairs(
                frame, spec,
                destination_group=destination_group,
                materiality_share=materiality_share,
                balance_min=balance_min,
            )
            observed[effect].add(run_id)
            runs.append({
                "effect": effect, "run_id": run_id, "resolution": resolution,
                "previous_year": previous, "current_year": current,
                "schema": schema, "persistence": persisted, "screens": screened,
            })
            item = aggregate[effect][resolution]
            item["transition_count"] += 1
            for name in SCREENS:
                item["screens"][name]["count"] += screened["screens"][name]["count"]
                item["screens"][name]["matched_mt"] += screened["screens"][name]["matched_mt"]
            item["false_small_sign"]["removed_count"] += screened["false_small_sign"]["removed_count"]
            item["false_small_sign"]["removed_matched_mt"] += (
                screened["false_small_sign"]["removed_matched_kg"] / 1.0e9
            )
            for _, pair in pairs.iterrows():
                records.append({
                    "effect": effect, "resolution": resolution,
                    "previous_year": previous, "current_year": current,
                    "cell_key": tuple(pair[name] for name in spec.pair_key),
                    "matched_kg": float(pair.matched_kg),
                    "gross_share": float(pair.gross_share),
                    "balance": float(pair.balance),
                    **{name: bool(pair[name]) for name in SCREENS},
                })
    completeness = {
        effect: {
            "observed": len(values), "expected": 42,
            "complete": values == expected, "missing": sorted(expected - values),
        }
        for effect, values in observed.items()
    }
    persistence_output = {
        effect: {
            resolution: {
                screen: persistence([
                    item for item in records
                    if item["effect"] == effect and item["resolution"] == resolution
                ], screen)
                for screen in SCREENS
            }
            for resolution in ("2013", "2016", "R3")
        }
        for effect in PRODUCTS
    }
    return {
        "schema_version": 1,
        "status": "validated" if thesis_eligible else "provisional",
        "upstream_validation": {
            "status": inventory.get("status", "missing"),
            "thesis_eligible": thesis_eligible,
            "counts": inventory.get("counts", {}),
        },
        "configuration": {
            "destination_group": destination_group,
            "materiality_share": materiality_share,
            "balance_min": balance_min,
            "conversion": "kg / 1e9 = Mt",
            "D_C_key": list(PRODUCTS["D_C"].pair_key),
            "D_YF_key": list(PRODUCTS["D_YF"].pair_key),
        },
        "completeness": completeness,
        "aggregate": aggregate,
        "persistence": persistence_output,
        "runs": runs,
        "interpretation": "Accounting co-movement consistent with substitution; not causal.",
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offshoring-root", required=True, type=Path)
    parser.add_argument("--validation-inventory", required=True, type=Path)
    parser.add_argument("--destination-group", choices=GROUPS, default="Red")
    parser.add_argument("--materiality-share", type=float, default=0.01)
    parser.add_argument("--balance-min", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    result = audit(
        args.offshoring_root, args.validation_inventory,
        destination_group=args.destination_group,
        materiality_share=args.materiality_share,
        balance_min=args.balance_min,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
