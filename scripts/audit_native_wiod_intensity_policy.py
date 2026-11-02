#!/usr/bin/env python3
"""Quantify changes from the SDA direct-intensity sanitization policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", required=True, help="YEAR:PRICE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-relative-change", type=float, default=1e-6)
    args = parser.parse_args()

    # Environment variables must be set before these project modules import config.
    import config
    import data_loader

    rows = []
    for state in args.state:
        year_text, price = state.split(":", 1)
        year = int(year_text)
        z, y = data_loader.read_year(
            year, price, config.WIOT_VERSION,
            source_wiot_version=config.FLOW_SOURCE_WIOT_VERSION,
        )
        x = z.sum(axis=1).add(y.sum(axis=1), fill_value=0.0).astype(float)
        if config.IMPACT_DATABASE == "EXIOBASE":
            raw = data_loader.build_emission_factors(year, config.IMPACT_CATEGORIES, z.index)
        else:
            raw = data_loader._native_impact_factors(year, config.IMPACT_CATEGORIES, z.index)
        emissions = raw.iloc[:, 0].to_numpy(dtype=float)
        output = x.reindex(z.index).to_numpy(dtype=float)
        zero = output == 0
        intensity = np.divide(emissions, output, out=np.zeros_like(emissions), where=~zero)
        reshaped = intensity.reshape(config.M, config.N)
        median = np.median(reshaped, axis=0, keepdims=True)
        outlier = np.abs(reshaped) > config.V_SECTOR_OUTLIER_MULTIPLE * np.abs(median)
        replaced = np.where(outlier, np.broadcast_to(median, reshaped.shape), reshaped)
        clamped = np.clip(replaced, -config.V_INTENSITY_MAX, config.V_INTENSITY_MAX)
        effective = clamped.reshape(-1) * output
        source_total = float(emissions.sum())
        effective_total = float(effective.sum())
        change = effective_total - source_total
        rows.append({
            "year": year,
            "price_basis": price,
            "source_total_kg_co2": source_total,
            "effective_total_kg_co2": effective_total,
            "change_kg_co2": change,
            "relative_change": abs(change) / max(abs(source_total), 1.0),
            "zero_output_cells": int(zero.sum()),
            "emissions_in_zero_output_cells_kg_co2": float(emissions[zero].sum()),
            "sector_median_replacements": int(outlier.sum()),
            "emissions_in_replaced_cells_kg_co2": float(emissions[outlier.reshape(-1)].sum()),
            "absolute_clamps_after_replacement": int((np.abs(replaced) > config.V_INTENSITY_MAX).sum()),
        })

    passed = all(row["relative_change"] <= args.maximum_relative_change for row in rows)
    manifest = {
        "schema_version": 1,
        "audit_id": "native-wiod-effective-impact-conservation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "impact_database": config.IMPACT_DATABASE,
            "target_wiot_version": config.WIOT_VERSION,
            "flow_source_wiot_version": config.FLOW_SOURCE_WIOT_VERSION,
            "units": "kg CO2",
            "household_direct_emissions_included": False,
            "sector_outlier_multiple": config.V_SECTOR_OUTLIER_MULTIPLE,
            "absolute_intensity_limit": config.V_INTENSITY_MAX,
        },
        "states": rows,
        "gate": {
            "maximum_relative_change": args.maximum_relative_change,
            "passed": passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed, "states": rows}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
