#!/usr/bin/env python3
"""Build the non-legal Kyoto-period fossil-CO2 footprint comparison.

This script deliberately does not infer Kyoto compliance. Official outcomes and
effective party targets must be supplied through a separately reviewed UNFCCC
crosswalk before a target analogue can be calculated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sheet(path: Path, sheet: str, header_row: int, data_row: int) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    countries = raw.iloc[header_row, 1:].astype("string")
    values = raw.iloc[data_row:, :].copy()
    values = values.rename(columns={0: "year"})
    values.columns = ["year", *countries.tolist()]
    values["year"] = pd.to_numeric(values["year"], errors="coerce")
    values = values.dropna(subset=["year"])
    values["year"] = values["year"].astype(int)
    return values.melt("year", var_name="country_name", value_name="mtc")


def annex_b_names(path: Path) -> set[str]:
    regions = pd.read_excel(path, sheet_name="Regions", header=None)
    row = regions.loc[regions.iloc[:, 0].eq("KP Annex B")]
    if len(row) != 1:
        raise ValueError("Workbook must contain exactly one KP Annex B region row")
    return {x.strip() for x in str(row.iloc[0, 1]).split(",")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    territorial = read_sheet(args.workbook, "Territorial Emissions", 11, 12)
    consumption = read_sheet(args.workbook, "Consumption Emissions", 8, 9)
    members = annex_b_names(args.workbook)
    joined = territorial.merge(consumption, on=["year", "country_name"], suffixes=("_territorial", "_consumption"))
    joined = joined[joined.country_name.isin(members) & joined.year.between(1990, 2012)]

    rows = []
    for country, part in joined.groupby("country_name"):
        base = part.loc[part.year.eq(1990)]
        period = part.loc[part.year.between(2008, 2012)]
        complete = len(base) == 1 and len(period) == 5 and base[["mtc_territorial", "mtc_consumption"]].notna().all(axis=None) and period[["mtc_territorial", "mtc_consumption"]].notna().all(axis=None)
        record = {
            "country_name": country,
            "workbook_region_label": "KP Annex B",
            "legal_membership_review_status": "unreviewed_workbook_label",
            "base_year": 1990,
            "evaluation_period": "2008-2012 mean",
            "scope": "fossil fuel combustion and oxidation plus cement CO2; excludes national bunker fuels",
            "unit": "Mt CO2/yr",
            "coverage_complete": bool(complete),
            "official_kyoto_outcome": "unresolved_requires_unfccc_crosswalk",
            "effective_cp1_target_percent": pd.NA,
            "effective_target_basis": "pending reviewed party target; EU parties require individual burden-sharing allocation",
            "consumption_target_analogue": "not_calculated_without_reviewed_effective_party_target",
        }
        if complete:
            t0 = float(base.mtc_territorial.iloc[0]) * 3.664
            c0 = float(base.mtc_consumption.iloc[0]) * 3.664
            te = float(period.mtc_territorial.mean()) * 3.664
            ce = float(period.mtc_consumption.mean()) * 3.664
            gt = (te / t0 - 1) * 100 if t0 else pd.NA
            gc = (ce / c0 - 1) * 100 if c0 else pd.NA
            record.update({
                "territorial_1990_mtco2": t0,
                "territorial_2008_2012_mean_mtco2": te,
                "territorial_change_percent": gt,
                "consumption_1990_mtco2": c0,
                "consumption_2008_2012_mean_mtco2": ce,
                "consumption_change_percent": gc,
                "consumption_change_direction": "fell" if gc < 0 else ("rose" if gc > 0 else "unchanged"),
                "consumption_minus_territorial_change_pp": gc - gt,
            })
        rows.append(record)
    result = pd.DataFrame(rows).sort_values("country_name")
    output = args.output_dir / "party_footprint_comparison.csv"
    result.to_csv(output, index=False)

    summary = pd.DataFrame([
        {"metric": "workbook_KP_Annex_B_labels", "value": len(members)},
        {"metric": "matched_countries", "value": len(result)},
        {"metric": "complete_1990_and_2008_2012", "value": int(result.coverage_complete.sum())},
        {"metric": "consumption_footprint_fell", "value": int(result.consumption_change_direction.eq("fell").sum())},
        {"metric": "consumption_footprint_rose", "value": int(result.consumption_change_direction.eq("rose").sum())},
        {"metric": "official_outcomes_populated", "value": 0},
    ])
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    report = args.output_dir / "report.md"
    report.write_text(
        "# Kyoto-period consumption-footprint comparison\n\n"
        "This artifact compares 1990 with the mean of 2008--2012 for matched "
        "territorial and consumption-based fossil-and-cement CO2. It is not Kyoto "
        "compliance: legal membership, effective targets and official outcomes remain "
        "unresolved pending a reviewed UNFCCC crosswalk. The workbook's `KP Annex B` "
        "region is retained only as an unreviewed source label.\n",
        encoding="utf-8",
    )
    outputs = [output, summary_path, report]
    manifest = {
        "schema_version": 1,
        "analysis_id": "kyoto-consumption-footprint-comparison-v1",
        "status": "complete_descriptive_footprint_pending_legal_crosswalk",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"script": "scripts/build_kyoto_consumption_footprint_comparison.py", "command": sys.argv},
        "input": {"path": str(args.workbook), "sha256": sha256(args.workbook)},
        "outputs": [{"path": p.name, "sha256": sha256(p)} for p in outputs],
        "validation": {
            "conversion_mtC_to_mtCO2": 3.664,
            "official_legal_outcomes_inferred": False,
            "effective_targets_inferred": False,
            "effective_target_rule": "party-specific CP1 target; individual EU burden-sharing allocation for joint-fulfilment parties",
            "canada_analytical_group_rule": "retain Reduction; do not overwrite with dated legal status",
            "required_years": [1990, 2008, 2009, 2010, 2011, 2012],
        },
        "limitations": [
            "Fossil-and-cement CO2 does not match the Kyoto covered-gas and accounting scope.",
            "The workbook region label is not an authoritative dated legal membership registry.",
            "The consumption target analogue cannot be evaluated without reviewed effective party targets.",
            "The current footprint indicator is fossil-and-cement CO2 and remains a provisional, non-legal analogue.",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
