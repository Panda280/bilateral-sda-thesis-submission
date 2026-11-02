#!/usr/bin/env python3
"""Aggregate matched D_C/D_YF sourcing interactions over time.

Values in campaign products are kg CO2. All exported value columns are Mt CO2
using the exact conversion 1 Mt = 1e9 kg. Sector codes are aggregated without
assigning names.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RESOLUTIONS = ("2013", "2016", "R3")
EFFECT_PATHS = {
    "D_C": ("source_group_sector_interaction", "D_C.source_destination_group_sector_interaction.parquet"),
    "D_YF": ("yf_group_product_interaction", "D_YF.source_destination_group_product_interaction.parquet"),
}
KG_PER_MT = 1_000_000_000.0


def inventory(root: Path) -> pd.DataFrame:
    records = []
    for effect, (folder, filename) in EFFECT_PATHS.items():
        for path in sorted((root / "offshoring").glob(f"*/{folder}/{filename}")):
            run_id = path.parents[1].name
            previous, current, resolution = run_id.rsplit("_", 2)
            report = (path.parent / "source_group_sector_interaction_report.json") if effect == "D_C" else (path.parents[1] / "yf_source_group_sector" / "yf_source_group_sector_cube_report.json")
            records.append({"effect": effect, "year_previous": int(previous), "year_current": int(current), "resolution": resolution, "path": str(path), "report_path": str(report)})
    found = pd.DataFrame(records)
    if found.empty:
        raise RuntimeError("no sourcing interaction products found")
    keys = ["year_previous", "year_current", "resolution"]
    counts = found.groupby(keys)["effect"].nunique()
    complete = counts[counts == len(EFFECT_PATHS)].index
    matched = found.set_index(keys).loc[complete].reset_index()
    expected = {(year, year + 1, resolution) for year in range(2000, 2014) for resolution in RESOLUTIONS}
    actual = set(map(tuple, matched[keys].drop_duplicates().itertuples(index=False, name=None)))
    if actual != expected:
        raise RuntimeError(f"complete matched transition inventory differs from expected: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return matched.sort_values(keys + ["effect"])


def signed_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    values = frame.assign(
        positive_kg=frame["Value"].clip(lower=0),
        negative_kg=frame["Value"].clip(upper=0),
        gross_kg=frame["Value"].abs(),
    )
    result = values.groupby(group_columns, dropna=False, as_index=False).agg(
        net_kg=("Value", "sum"), positive_kg=("positive_kg", "sum"),
        negative_kg=("negative_kg", "sum"), gross_kg=("gross_kg", "sum"),
        cell_count=("Value", "size"),
    )
    for name in ("net", "positive", "negative", "gross"):
        result[f"{name}_mt"] = result.pop(f"{name}_kg") / KG_PER_MT
    result["cancellation_ratio"] = np.where(result.gross_mt > 0, 1 - result.net_mt.abs() / result.gross_mt, 0.0)
    return result


def load_annual(rows: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for record in rows.itertuples(index=False):
        frame = pd.read_parquet(record.path)
        report = json.loads(Path(record.report_path).read_text(encoding="utf-8"))
        required = {"Relationship", "Source_Group", "Destination_Group", "Value"}
        if not required.issubset(frame.columns) or not np.isfinite(frame.Value).all():
            raise RuntimeError(f"invalid schema or non-finite values: {record.path}")
        if report.get("effect") != record.effect or report.get("units") != "kg CO2":
            raise RuntimeError(f"effect or unit mismatch in {record.report_path}")
        reconciliation = report["reconciliation"] if record.effect == "D_C" else report["reconciliation"]["parent"]
        if reconciliation.get("failing_index") is not None:
            raise RuntimeError(f"published reconciliation failed: {record.report_path}")
        if not np.isclose(frame.Value.sum(), report["total"], rtol=1e-5, atol=float(reconciliation["atol"])):
            raise RuntimeError(f"interaction total does not reconcile to source report: {record.path}")
        base = {"effect": record.effect, "resolution": record.resolution, "year_previous": record.year_previous, "year_current": record.year_current}
        levels = [
            ("overall", [], {}),
            ("relationship", ["Relationship"], {"Relationship": "relationship"}),
            ("source_group", ["Source_Group"], {"Source_Group": "source_group"}),
            ("destination_group", ["Destination_Group"], {"Destination_Group": "destination_group"}),
            ("source_destination", ["Source_Group", "Destination_Group"], {"Source_Group": "source_group", "Destination_Group": "destination_group"}),
        ]
        for level, dimensions, rename in levels:
            working = frame.copy() if dimensions else frame.assign(_all="ALL")
            dimensions = dimensions or ["_all"]
            metrics = signed_metrics(working, dimensions).rename(columns=rename)
            metrics["dimension_level"] = level
            for column in ("relationship", "source_group", "destination_group"):
                if column not in metrics:
                    metrics[column] = "ALL"
            for key, value in base.items():
                metrics[key] = value
            outputs.append(metrics.drop(columns=["_all"], errors="ignore"))
    columns = ["effect", "resolution", "year_previous", "year_current", "dimension_level", "relationship", "source_group", "destination_group", "net_mt", "positive_mt", "negative_mt", "gross_mt", "cancellation_ratio", "cell_count"]
    return pd.concat(outputs, ignore_index=True)[columns].sort_values(columns[:8])


def trends(annual: pd.DataFrame) -> pd.DataFrame:
    keys = ["effect", "resolution", "dimension_level", "relationship", "source_group", "destination_group"]
    records = []
    for key, group in annual.groupby(keys, dropna=False):
        group = group.sort_values("year_current")
        x = group.year_current.to_numpy(float)
        y = group.net_mt.to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(group) > 1 else np.nan
        signs = np.sign(y)
        nonzero = signs[signs != 0]
        records.append(dict(zip(keys, key)) | {
            "n_transitions": len(group), "cumulative_net_mt": y.sum(), "cumulative_gross_mt": group.gross_mt.sum(),
            "mean_annual_net_mt": y.mean(), "linear_slope_net_mt_per_year": slope,
            "sign_persistence": (max((nonzero > 0).sum(), (nonzero < 0).sum()) / len(nonzero) if len(nonzero) else 0.0),
            "adjacent_sign_reversals": int(np.sum(signs[1:] * signs[:-1] < 0)),
        })
    return pd.DataFrame(records)


def break_candidates(annual: pd.DataFrame) -> pd.DataFrame:
    keys = ["effect", "resolution", "dimension_level", "relationship", "source_group", "destination_group"]
    records = []
    for key, group in annual.groupby(keys, dropna=False):
        group = group.sort_values("year_current")
        changes = group.net_mt.diff()
        valid = changes.dropna()
        if len(valid) < 3:
            continue
        centre = valid.median()
        mad = (valid - centre).abs().median()
        zero_mad = bool(not mad)
        scores = (0.67448975 * (changes - centre) / mad) if mad else pd.Series(0.0, index=changes.index)
        for idx in scores.index[scores.abs() >= 3.5]:
            records.append(dict(zip(keys, key)) | {"year_current": int(group.loc[idx, "year_current"]), "change_in_net_mt": changes.loc[idx], "robust_mad_score": scores.loc[idx], "zero_mad": zero_mad, "threshold": 3.5})
    return pd.DataFrame(records)


def robustness(annual: pd.DataFrame) -> pd.DataFrame:
    keys = ["effect", "dimension_level", "relationship", "source_group", "destination_group"]
    records = []
    for key, group in annual.groupby(keys, dropna=False):
        for left, right in combinations(RESOLUTIONS, 2):
            a = group[group.resolution == left][["year_previous", "year_current", "net_mt", "gross_mt"]]
            b = group[group.resolution == right][["year_previous", "year_current", "net_mt", "gross_mt"]]
            pair = a.merge(b, on=["year_previous", "year_current"], suffixes=("_left", "_right"))
            if pair.empty:
                continue
            denominator = (pair.gross_mt_left.abs().mean() + pair.gross_mt_right.abs().mean()) / 2
            records.append(dict(zip(keys, key)) | {"resolution_left": left, "resolution_right": right, "n_matched": len(pair),
                "pearson_net": pair.net_mt_left.corr(pair.net_mt_right, method="pearson"),
                "spearman_net": pair.net_mt_left.corr(pair.net_mt_right, method="spearman"),
                "sign_agreement": (np.sign(pair.net_mt_left) == np.sign(pair.net_mt_right)).mean(),
                "normalized_mae_by_mean_gross": ((pair.net_mt_left-pair.net_mt_right).abs().mean()/denominator if denominator else 0.0)})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched = inventory(args.campaign_root)
    annual = load_annual(matched)
    trend = trends(annual)
    breaks = break_candidates(annual)
    robust = robustness(annual)
    matched.drop(columns=["path", "report_path"]).to_csv(args.output_dir / "matched_inventory.csv", index=False)
    annual.to_csv(args.output_dir / "annual_group_metrics.csv", index=False)
    trend.to_csv(args.output_dir / "period_trends.csv", index=False)
    breaks.to_csv(args.output_dir / "structural_break_candidates.csv", index=False)
    robust.to_csv(args.output_dir / "cross_resolution_robustness.csv", index=False)
    manifest = {"campaign_root": str(args.campaign_root), "units_input": "kg CO2", "units_output": "Mt CO2", "kg_per_mt": KG_PER_MT,
        "matched_transitions_per_effect_resolution": 14, "input_products": len(matched), "annual_rows": len(annual), "trend_rows": len(trend),
        "break_candidate_rows": len(breaks), "robustness_rows": len(robust), "effects_separate": ["D_C", "D_YF"],
        "limitations": ["descriptive, not causal", "robust MAD flags are candidates, not formal break tests", "sector codes aggregated without names"]}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
