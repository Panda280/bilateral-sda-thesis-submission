#!/usr/bin/env python
"""Summarize persisted offshoring-substitution cell-year outputs.

This is a read-only companion to ``analyze_offshoring_substitution.py``.  It
does not reopen or reinterpret the source interaction cubes.  All calculations
start from ``cell_year_signatures.parquet`` and preserve resolution and effect
as grouping keys.  Values remain kg CO2 until display conversion by 1e9 to
Mt CO2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KG_PER_MT = 1_000_000_000.0
GROUP_KEYS = ["resolution", "effect"]
CELL_KEYS = ["destination_group", "sector_code"]


def _safe_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _period_metrics(frame: pd.DataFrame, name: str) -> list[dict]:
    rows = []
    for (resolution, effect), group in frame.groupby(GROUP_KEYS, sort=False):
        strict = group.loc[group["strict_signature"]]
        positive_np = group["np_cross_border_kg_co2"].clip(lower=0).sum()
        transitions = group["start_year"].nunique()
        rows.append({
            "period": name,
            "resolution": resolution,
            "effect": effect,
            "transitions": transitions,
            "strict_cell_years": len(strict),
            "eligible_cell_years": len(group),
            "prevalence_share": len(strict) / len(group),
            "np_capture_share": (
                strict["np_cross_border_kg_co2"].sum() / positive_np
                if positive_np else None
            ),
            "strict_np_cross_border_mt_co2": (
                strict["np_cross_border_kg_co2"].sum() / KG_PER_MT
            ),
            "strict_domestic_mt_co2": strict["domestic_kg_co2"].sum() / KG_PER_MT,
            "strict_same_group_mt_co2": strict["same_group_kg_co2"].sum() / KG_PER_MT,
            "matched_overlap_mt_co2": strict["matched_overlap_kg_co2"].sum() / KG_PER_MT,
            "annual_mean_overlap_mt_co2": (
                strict["matched_overlap_kg_co2"].sum() / KG_PER_MT / transitions
            ),
        })
    return rows


def _annual_metrics(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for key, group in frame.groupby(GROUP_KEYS + ["start_year", "end_year"], sort=False):
        strict = group.loc[group["strict_signature"]]
        positive_np = group["np_cross_border_kg_co2"].clip(lower=0).sum()
        rows.append({
            **dict(zip(GROUP_KEYS + ["start_year", "end_year"], key, strict=True)),
            "strict_cell_years": len(strict),
            "eligible_cells": len(group),
            "prevalence_share": len(strict) / len(group),
            "np_capture_share": (
                strict["np_cross_border_kg_co2"].sum() / positive_np
                if positive_np else None
            ),
            "matched_overlap_mt_co2": strict["matched_overlap_kg_co2"].sum() / KG_PER_MT,
        })
    return rows


def _threshold_metrics(frame: pd.DataFrame, thresholds_mt: list[float]) -> list[dict]:
    rows = []
    for (resolution, effect), group in frame.groupby(GROUP_KEYS, sort=False):
        positive_np = group["np_cross_border_kg_co2"].clip(lower=0).sum()
        for threshold_mt in thresholds_mt:
            threshold_kg = threshold_mt * KG_PER_MT
            hit = (
                (group["np_cross_border_kg_co2"] > threshold_kg)
                & (group["domestic_kg_co2"] < -threshold_kg)
                & (group["same_group_kg_co2"] < -threshold_kg)
            )
            selected = group.loc[hit]
            overlap = np.minimum(
                selected["np_cross_border_kg_co2"],
                -selected["same_group_kg_co2"],
            ).sum()
            rows.append({
                "resolution": resolution,
                "effect": effect,
                "threshold_mt_co2": threshold_mt,
                "strict_cell_years": len(selected),
                "eligible_cell_years": len(group),
                "prevalence_share": len(selected) / len(group),
                "np_capture_share": (
                    selected["np_cross_border_kg_co2"].sum() / positive_np
                    if positive_np else None
                ),
                "matched_overlap_mt_co2": overlap / KG_PER_MT,
            })
    return rows


def _persistence_and_leaders(frame: pd.DataFrame, leader_count: int) -> tuple[list[dict], list[dict]]:
    persistence = []
    leaders = []
    for (resolution, effect), group in frame.groupby(GROUP_KEYS, sort=False):
        by_cell = (
            group.groupby(CELL_KEYS, as_index=False)
            .agg(strict_years=("strict_signature", "sum"))
        )
        persistence.append({
            "resolution": resolution,
            "effect": effect,
            "cells": len(by_cell),
            "maximum_strict_years": int(by_cell["strict_years"].max()),
            "cells_ever_strict": int((by_cell["strict_years"] >= 1).sum()),
            "cells_at_least_5_years": int((by_cell["strict_years"] >= 5).sum()),
            "cells_at_least_7_years": int((by_cell["strict_years"] >= 7).sum()),
            "cells_at_least_10_years": int((by_cell["strict_years"] >= 10).sum()),
            "cells_all_years": int((by_cell["strict_years"] == group["start_year"].nunique()).sum()),
        })
        strict = group.loc[group["strict_signature"]]
        ranked = (
            strict.groupby(CELL_KEYS, as_index=False)
            .agg(
                strict_years=("strict_signature", "size"),
                np_cross_border_kg_co2=("np_cross_border_kg_co2", "sum"),
                domestic_kg_co2=("domestic_kg_co2", "sum"),
                same_group_kg_co2=("same_group_kg_co2", "sum"),
                matched_overlap_kg_co2=("matched_overlap_kg_co2", "sum"),
            )
            .sort_values(["matched_overlap_kg_co2", "strict_years"], ascending=False)
            .head(leader_count)
        )
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            leaders.append({
                "resolution": resolution,
                "effect": effect,
                "rank": rank,
                "destination_group": row.destination_group,
                "sector_code": str(row.sector_code),
                "strict_years": int(row.strict_years),
                "np_cross_border_mt_co2": row.np_cross_border_kg_co2 / KG_PER_MT,
                "domestic_mt_co2": row.domestic_kg_co2 / KG_PER_MT,
                "same_group_mt_co2": row.same_group_kg_co2 / KG_PER_MT,
                "matched_overlap_mt_co2": row.matched_overlap_kg_co2 / KG_PER_MT,
            })
    return persistence, leaders


def summarize(input_dir: Path, early_end_year: int, thresholds_mt: list[float], leader_count: int) -> dict:
    coverage = json.loads((input_dir / "coverage.json").read_text(encoding="utf-8"))
    source_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(input_dir / "cell_year_signatures.parquet")
    if coverage["missing_selected_products"]:
        raise ValueError("coverage contains missing selected products")
    if source_summary["units"]["kg_per_mt"] != KG_PER_MT:
        raise ValueError("source summary does not declare 1e9 kg per Mt")
    early = frame.loc[frame["end_year"] <= early_end_year]
    persistence, leaders = _persistence_and_leaders(frame, leader_count)
    return {
        "schema_version": 1,
        "source_directory": str(input_dir.resolve()),
        "calculation_units": "kg CO2",
        "display_conversion": "kg / 1e9 = Mt CO2",
        "coverage": {
            "mode": coverage["mode"],
            "transitions": coverage["selected_transitions"],
            "products": len(coverage["inventory"]),
            "missing_products": len(coverage["missing_selected_products"]),
        },
        "period_metrics": _period_metrics(early, f"through_{early_end_year}")
        + _period_metrics(frame, "full"),
        "annual_metrics": _annual_metrics(frame),
        "threshold_robustness": _threshold_metrics(frame, thresholds_mt),
        "persistence": persistence,
        "leaders_by_cumulative_annual_overlap": leaders,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--early-end-year", type=int, default=2005)
    parser.add_argument("--thresholds-mt", nargs="+", type=float, default=(0.0, 0.01, 0.1, 1.0))
    parser.add_argument("--leaders", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional JSON path; otherwise print to stdout")
    args = parser.parse_args(argv)
    payload = summarize(
        args.input_dir.resolve(), args.early_end_year, list(args.thresholds_mt), args.leaders
    )
    rendered = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
