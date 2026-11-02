#!/usr/bin/env python3
"""Build role-safe analytical-group products for the three Kyoto periods.

The group-effect table is built from dashboard-contract-v2 country panels.  The
bilateral interaction table is built from the underlying D_C and D_YF sourcing
interaction parquet files, because the older ``annual_group_metrics.csv`` does
not retain relationship jointly with source and destination group.

Only P1, P2 and P3 are output.  Transition year ``t`` means ``t-1 -> t``.
Input and output values are kg CO2; no numerical unit conversion is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = {"P1": tuple(range(2001, 2005)), "P2": tuple(range(2005, 2008)), "P3": tuple(range(2008, 2013))}
PERIOD_LABELS = {"P1": "pre-entry into force", "P2": "in force, pre-commitment", "P3": "first commitment period"}
GROUPS = ("Red", "Non-Red", "Non-Participant")
RESOLUTIONS = ("2013", "2016", "R3")
CONFIGURATIONS = {resolution: f"exio_{resolution}" for resolution in RESOLUTIONS}
IMPACT_SOURCE = "EXIOBASE fossil-origin CO2"
FLOW_SOURCE = "EXIOBASE-derived CYP/PYP campaign flows"
PRICE_BASIS = "CYP_(t-1) -> PYP_t"

# Every available country attribution is retained.  Effects and roles are never
# pooled because they answer different accounting questions.
ROLE_PANELS = {
    "emitter": ("emitter_country.parquet", ("D_v", "D_C", "D_A_x", "D_T_coef", "D_w_coef", "D_YF", "D_YB", "D_y")),
    "structural_driver": ("driver_country.parquet", ("D_C", "D_A_x", "D_T_coef", "D_w_coef")),
    "consumer": ("consumer_country.parquet", ("D_YF", "D_YB", "D_y")),
}
INTERACTION_SPECS = {
    "D_C": ("structural_driver", "source_group_sector_interaction", "D_C.source_destination_group_sector_interaction.parquet"),
    "D_YF": ("consumer", "yf_group_product_interaction", "D_YF.source_destination_group_product_interaction.parquet"),
}
METRIC_INPUTS = {
    "net": "contribution", "positive": "positive_contribution",
    "negative": "negative_contribution", "gross": "gross_contribution",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def period_for_year(year: int) -> str | None:
    return next((period for period, years in PERIODS.items() if year in years), None)


def cancellation(net: pd.Series, gross: pd.Series) -> np.ndarray:
    return np.where(gross > 0, 1.0 - net.abs() / gross, 0.0)


def _assert_metric_identities(frame: pd.DataFrame, label: str) -> None:
    if not np.allclose(frame["net"], frame["positive"] + frame["negative"], rtol=1e-10, atol=1e-3):
        raise ValueError(f"{label}: net != positive + negative")
    if not np.allclose(frame["gross"], frame["positive"] - frame["negative"], rtol=1e-10, atol=1e-3):
        raise ValueError(f"{label}: gross != positive - negative")
    if (frame["positive"] < 0).any() or (frame["negative"] > 0).any() or (frame["gross"] < 0).any():
        raise ValueError(f"{label}: invalid metric signs")


def load_country_group_map(country_concordance: Path, group_registry: Path) -> dict[tuple[str, str], str]:
    countries = pd.read_csv(country_concordance, dtype=str)
    groups = pd.read_csv(group_registry, dtype=str)
    if set(groups["analysis_group"]) != set(GROUPS):
        raise ValueError("analytical group registry differs from the approved three-group classification")
    merged = countries.merge(groups[["canonical_country", "analysis_group"]], on="canonical_country", how="left", validate="many_to_one")
    if merged["analysis_group"].isna().any() or (merged["review_status"] != "reviewed").any():
        raise ValueError("country concordance contains unmapped or unreviewed rows")
    if merged.duplicated(["resolution", "observed_country"]).any():
        raise ValueError("country concordance key is not unique")
    return {(row.resolution, row.observed_country): row.analysis_group for row in merged.itertuples(index=False)}


def build_group_effects(panel_root: Path, country_map: dict[tuple[str, str], str]) -> tuple[pd.DataFrame, dict]:
    rows = []
    annual_for_gate = []
    for role, (filename, effects) in ROLE_PANELS.items():
        frame = pd.read_parquet(panel_root / filename)
        frame = frame[frame["effect_id"].isin(effects)].copy()
        if not (frame["is_publishable"].astype(bool) & frame["validation_status"].eq("passed")).all():
            raise ValueError(f"{filename}: non-publishable or failed source rows")
        if set(frame["units"]) != {"kg CO2"} or not np.isfinite(frame[list(METRIC_INPUTS.values())]).all().all():
            raise ValueError(f"{filename}: invalid units or non-finite values")
        frame["period"] = frame["year_current"].map(period_for_year)
        frame = frame[frame["period"].notna()].copy()
        frame["group"] = [country_map.get((str(res), str(country))) for res, country in zip(frame["resolution"], frame["label"])]
        if frame["group"].isna().any():
            missing = sorted(set(zip(frame.loc[frame.group.isna(), "resolution"], frame.loc[frame.group.isna(), "label"])))
            raise ValueError(f"{filename}: countries absent from reviewed concordance: {missing}")
        frame["role"] = role
        annual_for_gate.append(frame)
        keys = ["period", "resolution", "role", "group", "effect_id"]
        agg = frame.groupby(keys, as_index=False).agg(**{name: (column, "sum") for name, column in METRIC_INPUTS.items()}, transition_count=("year_current", "nunique"))
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True).rename(columns={"effect_id": "effect"})
    out["configuration"] = out["resolution"].map(CONFIGURATIONS)
    out["impact_source"] = IMPACT_SOURCE
    out["flow_source"] = FLOW_SOURCE
    out["units"] = "kg CO2"
    out["price_basis"] = PRICE_BASIS
    out["coverage_status"] = "complete"
    out["cancellation"] = cancellation(out["net"], out["gross"])
    expected = out["period"].map({key: len(value) for key, value in PERIODS.items()})
    if not out["transition_count"].eq(expected).all():
        raise ValueError("group effects fail the exact P1/P2/P3 transition-count gate")
    _assert_metric_identities(out, "group effects")

    # Each role is a full partition of its effect. Check all four signed pools
    # against the corresponding ungrouped role panel. The hierarchy table has a
    # coarser pre-metric grain, so only its signed net is comparable; its gross
    # must not be substituted for a role-panel gross.
    hierarchy = pd.read_parquet(panel_root / "global_hierarchy.parquet")
    hierarchy["period"] = hierarchy["year_current"].map(period_for_year)
    hierarchy = hierarchy[hierarchy["period"].notna()]
    global_period = hierarchy.groupby(["period", "resolution", "effect_id"], as_index=False).agg(net=("contribution", "sum"))
    source_role_period = pd.concat(annual_for_gate).groupby(["period", "resolution", "role", "effect_id"], as_index=False).agg(
        **{name: (column, "sum") for name, column in METRIC_INPUTS.items()})
    grouped_role_period = out.groupby(["period", "resolution", "role", "effect"], as_index=False)[list(METRIC_INPUTS)].sum().rename(columns={"effect": "effect_id"})
    checked = source_role_period.merge(grouped_role_period, on=["period", "resolution", "role", "effect_id"], suffixes=("_source", "_groups"), validate="one_to_one")
    gate_errors = {}
    for metric in METRIC_INPUTS:
        error = (checked[f"{metric}_groups"] - checked[f"{metric}_source"]).abs()
        gate_errors[metric] = float(error.max())
        if not np.allclose(checked[f"{metric}_groups"], checked[f"{metric}_source"], rtol=1e-10, atol=1e-3):
            raise ValueError(f"group sums do not reconstruct role-panel {metric}")
    net_check = source_role_period.merge(global_period, on=["period", "resolution", "effect_id"], suffixes=("_role", "_global"), validate="many_to_one")
    global_net_error = (net_check["net_role"] - net_check["net_global"]).abs()
    if not np.allclose(net_check["net_role"], net_check["net_global"], rtol=1e-10, atol=1e-2):
        raise ValueError("role-panel net does not reconstruct registered global effect")
    columns = ["period", "configuration", "impact_source", "flow_source", "resolution", "role", "group", "effect",
               "net", "positive", "negative", "gross", "cancellation", "transition_count", "coverage_status", "units", "price_basis"]
    return out[columns].sort_values(columns[:8]).reset_index(drop=True), {
        "max_abs_group_sum_error_kg": gate_errors,
        "max_abs_role_net_to_global_error_kg": float(global_net_error.max()),
        "gross_comparison_note": "signed pools reconstruct the same-grain role panel; only net is compared with coarser global hierarchy",
    }


def interaction_inventory(campaign_root: Path) -> pd.DataFrame:
    records = []
    for effect, (role, folder, filename) in INTERACTION_SPECS.items():
        for path in sorted((campaign_root / "offshoring").glob(f"*/{folder}/{filename}")):
            run_id = path.parents[1].name
            try:
                previous, current, resolution = run_id.rsplit("_", 2)
                previous, current = int(previous), int(current)
            except ValueError:
                continue
            period = period_for_year(current)
            if period is not None and resolution in RESOLUTIONS:
                records.append({"effect": effect, "role": role, "period": period, "year_previous": previous,
                                "year_current": current, "resolution": resolution, "path": str(path)})
    found = pd.DataFrame(records)
    if found.empty:
        raise FileNotFoundError(f"no P1/P2/P3 interaction parquet files under {campaign_root}")
    expected = {(effect, period, year - 1, year, resolution) for effect in INTERACTION_SPECS for period, years in PERIODS.items()
                for year in years for resolution in RESOLUTIONS}
    actual = set(found[["effect", "period", "year_previous", "year_current", "resolution"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"interaction coverage differs from exact P1/P2/P3 inventory: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return found


def build_interactions(campaign_root: Path, registered_metrics: Path | None = None) -> tuple[pd.DataFrame, dict]:
    inventory = interaction_inventory(campaign_root)
    annual = []
    cross_group_domestic_placeholders = 0
    for record in inventory.itertuples(index=False):
        frame = pd.read_parquet(record.path, columns=["Relationship", "Source_Group", "Destination_Group", "Value"])
        if not np.isfinite(frame["Value"]).all() or set(frame["Relationship"]) != {"Domestic", "Cross-border"}:
            raise ValueError(f"invalid interaction values or relationships: {record.path}")
        if set(frame["Source_Group"]) != set(GROUPS) or set(frame["Destination_Group"]) != set(GROUPS):
            raise ValueError(f"interaction groups differ from approved registry: {record.path}")
        impossible = frame["Relationship"].eq("Domestic") & frame["Source_Group"].ne(frame["Destination_Group"])
        if (frame.loc[impossible, "Value"] != 0).any():
            raise ValueError(f"non-zero domestic cross-group contribution: {record.path}")
        cross_group_domestic_placeholders += int(impossible.sum())
        frame = frame.loc[~impossible].copy()  # omit structural zero placeholders from the published cube
        frame["positive"] = frame["Value"].clip(lower=0)
        frame["negative"] = frame["Value"].clip(upper=0)
        frame["gross"] = frame["Value"].abs()
        keys = ["Relationship", "Source_Group", "Destination_Group"]
        agg = frame.groupby(keys, as_index=False).agg(net=("Value", "sum"), positive=("positive", "sum"),
                                                      negative=("negative", "sum"), gross=("gross", "sum"))
        for key, value in {"effect": record.effect, "role": record.role, "period": record.period,
                           "year_previous": record.year_previous, "year_current": record.year_current,
                           "resolution": record.resolution}.items():
            agg[key] = value
        annual.append(agg)
    annual = pd.concat(annual, ignore_index=True)
    annual = annual.rename(columns={"Relationship": "relationship", "Source_Group": "source_group", "Destination_Group": "destination_group"})
    keys = ["period", "resolution", "effect", "role", "source_group", "destination_group", "relationship"]
    out = annual.groupby(keys, as_index=False).agg(net=("net", "sum"), positive=("positive", "sum"),
                                                   negative=("negative", "sum"), gross=("gross", "sum"),
                                                   transition_count=("year_current", "nunique"))
    _assert_metric_identities(out, "group interactions")
    expected_counts = out["period"].map({key: len(value) for key, value in PERIODS.items()})
    if not out["transition_count"].eq(expected_counts).all():
        raise ValueError("interactions fail exact P1/P2/P3 transition-count gate")
    if ((out.relationship == "Domestic") & (out.source_group != out.destination_group)).any():
        raise ValueError("published cube contains domestic cross-group cells")

    pair = out.groupby(["period", "resolution", "effect", "role", "source_group", "destination_group"], as_index=False).agg(pair_gross=("gross", "sum"))
    domestic = out[out.relationship == "Domestic"].groupby(["period", "resolution", "effect", "role", "source_group", "destination_group"], as_index=False).agg(domestic_gross=("gross", "sum"))
    out = out.merge(pair, validate="many_to_one").merge(domestic, how="left", validate="many_to_one")
    out["domestic_gross"] = out["domestic_gross"].fillna(0.0)
    out["domestic_gross_share"] = np.where(out.pair_gross > 0, out.domestic_gross / out.pair_gross, 0.0)
    out["foreign_gross_share"] = np.where(out.pair_gross > 0, 1.0 - out.domestic_gross_share, 0.0)
    out["cancellation"] = cancellation(out["net"], out["gross"])
    out["configuration"] = out["resolution"].map(CONFIGURATIONS)
    out["impact_source"] = IMPACT_SOURCE
    out["flow_source"] = FLOW_SOURCE
    out["units"] = "kg CO2"
    out["price_basis"] = PRICE_BASIS
    out["coverage_status"] = "complete"

    # Validate relationship sums and the complete 3x3 pair inventory against
    # independently pooled annual totals.
    pair_counts = out.groupby(["period", "resolution", "effect"]).size()
    if not pair_counts.eq(12).all():  # 3 diagonal pairs x 2 relationships + 6 cross-group foreign pairs
        raise ValueError(f"interaction relationship inventory is incomplete: {pair_counts.to_dict()}")
    pooled = annual.groupby(["period", "resolution", "effect"], as_index=False)[["net", "positive", "negative", "gross"]].sum()
    rebuilt = out.groupby(["period", "resolution", "effect"], as_index=False)[["net", "positive", "negative", "gross"]].sum()
    check = pooled.merge(rebuilt, on=["period", "resolution", "effect"], suffixes=("_raw", "_cube"), validate="one_to_one")
    errors = {}
    for metric in ("net", "positive", "negative", "gross"):
        error = (check[f"{metric}_raw"] - check[f"{metric}_cube"]).abs()
        errors[metric] = float(error.max())
        if not np.allclose(check[f"{metric}_raw"], check[f"{metric}_cube"], rtol=1e-10, atol=1e-3):
            raise ValueError(f"relationship/group-pair sums fail for {metric}")
    registered_errors = {}
    if registered_metrics is not None:
        registered = pd.read_csv(registered_metrics)
        registered = registered[(registered["dimension_level"] == "overall") &
                                registered["year_current"].map(period_for_year).notna()].copy()
        registered = registered[["effect", "resolution", "year_previous", "year_current",
                                 "net_mt", "positive_mt", "negative_mt", "gross_mt"]]
        raw_annual = annual.groupby(["effect", "resolution", "year_previous", "year_current"], as_index=False)[
            ["net", "positive", "negative", "gross"]].sum()
        registered = registered.rename(columns={f"{metric}_mt": f"{metric}_registered" for metric in ("net", "positive", "negative", "gross")})
        joined = raw_annual.merge(registered, on=["effect", "resolution", "year_previous", "year_current"], validate="one_to_one")
        if len(joined) != len(raw_annual):
            raise ValueError("registered overall sourcing metrics do not cover every raw annual interaction")
        for metric in ("net", "positive", "negative", "gross"):
            expected_kg = joined[f"{metric}_registered"] * 1_000_000_000.0
            error = (joined[metric] - expected_kg).abs()
            registered_errors[metric] = float(error.max())
            if not np.allclose(joined[metric], expected_kg, rtol=1e-10, atol=1e-2):
                raise ValueError(f"raw interaction {metric} does not reconstruct registered annual sourcing metric")
    columns = ["period", "configuration", "impact_source", "flow_source", "resolution", "effect", "role", "source_group",
               "destination_group", "relationship", "net", "positive", "negative", "gross", "cancellation",
               "domestic_gross_share", "foreign_gross_share", "transition_count", "coverage_status", "units", "price_basis"]
    return out[columns].sort_values(columns[:10]).reset_index(drop=True), {
        "max_abs_relationship_sum_error_kg": errors,
        "max_abs_registered_annual_error_kg": registered_errors,
        "omitted_zero_domestic_cross_group_source_rows": cross_group_domestic_placeholders,
    }


def build_coverage() -> pd.DataFrame:
    rows = []
    for resolution, period in product(RESOLUTIONS, PERIODS):
        years = PERIODS[period]
        rows.append({"configuration": CONFIGURATIONS[resolution], "period": period, "period_label": PERIOD_LABELS[period],
                     "impact_source": IMPACT_SOURCE, "impact_verified_years": "2000-2014",
                     "flow_source": FLOW_SOURCE, "flow_verified_years": "2000-2014", "resolution": resolution,
                     "expected_transitions": len(years), "observed_transitions": len(years), "observed_years": ",".join(map(str, years)),
                     "coverage_status": "complete", "exclusion_reason": ""})
    return pd.DataFrame(rows)


def build_support_inventory() -> pd.DataFrame:
    supported = {("D_C", "structural_driver"), ("D_YF", "consumer")}
    rows = []
    for role, (_, effects) in ROLE_PANELS.items():
        for effect in effects:
            available = (effect, role) in supported
            rows.append({"effect": effect, "role": role, "bilateral_status": "available" if available else "unavailable_not_inferred",
                         "reason": "registered raw bilateral sourcing product" if available else "no registered bilateral product for this effect-role; no allocation inferred"})
    return pd.DataFrame(rows).sort_values(["effect", "role"]).reset_index(drop=True)


def write_report(path: Path, effects: pd.DataFrame, interactions: pd.DataFrame, coverage: pd.DataFrame, gates: dict) -> None:
    lines = ["# Kyoto-period analytical-group products", "", "Status: generated and validation gates passed.", "",
             "This product contains only P1 (2001--2004), P2 (2005--2007), and P3 (2008--2012). Result year `t` is the transition `t-1 -> t`. Values are kg CO2 on the `CYP_(t-1) -> PYP_t` basis.", "",
             "Group effects retain emitter, structural-driver, and consumer attribution roles separately. Bilateral interactions are available only for `D_C` in the structural-driver role and `D_YF` in the consumer role. Missing effect-role interactions are marked `unavailable_not_inferred` in the support inventory.", "",
             "Domestic means source country equals destination country. Same-group foreign values remain `Cross-border`; within-group is therefore not equated with domestic. Zero-valued domestic cross-group placeholders present in the raw exports were verified as zero and omitted.", "",
             "## Coverage", "", "| configuration | P1 | P2 | P3 |", "|---|---:|---:|---:|"]
    for configuration in sorted(coverage.configuration.unique()):
        values = coverage[coverage.configuration == configuration].set_index("period")["observed_transitions"]
        lines.append(f"| {configuration} | {values['P1']}/4 | {values['P2']}/3 | {values['P3']}/5 |")
    lines += ["", "## Validation", "", f"- Group-effect rows: {len(effects)}.", f"- Interaction rows: {len(interactions)} (12 per period/effect/configuration).",
              "- Net = positive + negative and gross = positive - negative passed for every output row.",
              "- Country-group sums reconstruct the same-grain role panel in net, positive, negative, and gross pools; role-panel nets also reconstruct the registered global effect.",
              "- Relationship and group-pair sums reconstruct the raw interaction pools.",
              "- Every P1/P2/P3 series contains exactly 4/3/5 transitions.",
              f"- Omitted zero domestic cross-group source rows: {gates['interactions']['omitted_zero_domestic_cross_group_source_rows']}.", "",
              "These are accounting attributions, not causal policy estimates. Analytical groups are static classifications, not legal compliance categories.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True, help="Directory containing cross_year_panel parquet files")
    parser.add_argument("--campaign-root", type=Path, required=True, help="Campaign root containing offshoring/<run_id> products")
    parser.add_argument("--country-concordance", type=Path, required=True)
    parser.add_argument("--group-registry", type=Path, required=True)
    parser.add_argument("--registered-sourcing-metrics", type=Path, required=True,
                        help="Accepted annual_group_metrics.csv used for overall reconstruction gates")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    country_map = load_country_group_map(args.country_concordance, args.group_registry)
    effects, effects_gates = build_group_effects(args.panel_root, country_map)
    interactions, interaction_gates = build_interactions(args.campaign_root, args.registered_sourcing_metrics)
    coverage = build_coverage()
    support = build_support_inventory()
    outputs = {
        "kyoto_period_group_effects.csv": effects,
        "kyoto_period_group_interactions.csv": interactions,
        "coverage_register.csv": coverage,
        "bilateral_support_inventory.csv": support,
    }
    artifacts = []
    for filename, frame in outputs.items():
        target = args.output_dir / filename
        frame.to_csv(target, index=False)
        artifacts.append({"path": filename, "rows": len(frame), "sha256": sha256(target)})
    gates = {"group_effects": effects_gates, "interactions": interaction_gates,
             "coverage_counts": {period: len(years) for period, years in PERIODS.items()},
             "metric_identities": "passed", "group_sums": "passed", "relationship_sums": "passed",
             "no_nonzero_domestic_cross_group_cells": "passed", "unsupported_cells": "unavailable_not_inferred"}
    report = args.output_dir / "VALIDATION_REPORT.md"
    write_report(report, effects, interactions, coverage, gates)
    artifacts.append({"path": report.name, "sha256": sha256(report)})
    manifest = {"schema_version": 1, "analysis_id": "kyoto-period-group-products-v1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "generator": {"script": "implementation/bilateral_sda_attribution/scripts/build_kyoto_period_group_products.py", "git_commit": args.git_commit},
                "inputs": {"panel_root": str(args.panel_root), "campaign_root": str(args.campaign_root),
                           "country_concordance": str(args.country_concordance), "group_registry": str(args.group_registry),
                           "registered_sourcing_metrics": str(args.registered_sourcing_metrics)},
                "scope": {"periods": {period: list(years) for period, years in PERIODS.items()}, "groups": list(GROUPS),
                          "resolutions": list(RESOLUTIONS), "units": "kg CO2", "price_basis": PRICE_BASIS,
                          "bilateral_effect_roles": {"D_C": "structural_driver", "D_YF": "consumer"}},
                "validation": {"passed": True, "gates": gates}, "outputs": artifacts,
                "limitations": ["accounting attribution, not causal", "static analytical groups, not legal Kyoto categories",
                                "bilateral cells unavailable and not inferred outside D_C/structural_driver and D_YF/consumer"]}
    manifest_path = args.output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "group_effect_rows": len(effects),
                      "interaction_rows": len(interactions), "validation": "passed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
