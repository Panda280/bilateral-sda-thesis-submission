"""Build a provisional group/main-effect report from completed annual reports.

The report is deliberately descriptive.  It uses emitter-country attribution,
the versioned static Kyoto analytical groups, and only years available for all
requested resolutions in its main comparison window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PRIMARY_COMPONENTS = {
    "emission_intensity": ("D_v",),
    "production_structure": ("D_A_x", "D_C"),
    "final_demand": ("D_YB", "D_YF", "D_y"),
}


def _fmt_gt(value: float) -> str:
    return f"{value / 1e12:+.3f}"


def build_report(annual_root: Path, groups_csv: Path, countries_csv: Path, output_dir: Path) -> None:
    paths = sorted(annual_root.glob("*/annual_drilldown.parquet"))
    if not paths:
        raise FileNotFoundError(f"no annual drilldowns under {annual_root}")

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    drilldown = pd.concat(frames, ignore_index=True)
    mapping = pd.read_csv(groups_csv).drop_duplicates("country_code")
    group_map = mapping.set_index("country_code")["analysis_group"]
    countries = pd.read_csv(countries_csv)
    canonical_code = countries.drop_duplicates("EXIO").set_index("EXIO")["WIOT2016"]
    valid = drilldown["is_publishable"].astype(bool) & (drilldown["validation_status"] == "passed")
    keys = ["resolution", "year_previous", "year_current", "analysis_group", "attribution_role"]
    effect_rows = []
    for label, components in PRIMARY_COMPONENTS.items():
        country_column = "consumer_country" if label == "final_demand" else "emitter_country"
        grain = country_column
        subset = drilldown.loc[valid & (drilldown["grain_id"] == grain) & drilldown["effect_id"].isin(components)].copy()
        subset["canonical_country"] = subset[country_column].map(canonical_code).fillna(subset[country_column])
        subset["analysis_group"] = subset["canonical_country"].map(group_map)
        missing = sorted(subset.loc[subset["analysis_group"].isna(), country_column].unique())
        if missing:
            raise ValueError(f"{country_column} codes missing from group registry: {missing}")
        subset["attribution_role"] = "consumer" if label == "final_demand" else "emitter"
        values = subset.groupby(keys, as_index=False)["contribution"].sum()
        values["main_effect"] = label
        effect_rows.append(values)
    primary = pd.concat(effect_rows, ignore_index=True)

    coverage = primary.groupby("resolution")["year_current"].agg(["min", "max", "nunique"])
    common_years = sorted(set.intersection(*(
        set(group["year_current"].unique()) for _, group in primary.groupby("resolution")
    )))
    if not common_years:
        raise ValueError("no common years across resolutions")
    common = primary.loc[primary["year_current"].isin(common_years)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(output_dir / "provisional_group_main_effects_annual.csv", index=False)
    cumulative = common.groupby(["resolution", "analysis_group", "attribution_role", "main_effect"], as_index=False).agg(
        cumulative_contribution=("contribution", "sum"),
        positive_years=("contribution", lambda x: int((x > 0).sum())),
        negative_years=("contribution", lambda x: int((x < 0).sum())),
    )
    cumulative.to_csv(output_dir / "provisional_group_main_effects_cumulative.csv", index=False)

    world_effects = common.groupby(["resolution", "year_current", "main_effect"], as_index=False)["contribution"].sum()
    world_totals = world_effects.groupby(["resolution", "year_current"], as_index=False)["contribution"].sum()
    world_totals["main_effect"] = "total_change"
    world = pd.concat([world_totals, world_effects], ignore_index=True)
    world.to_csv(output_dir / "provisional_world_main_effects_annual.csv", index=False)

    lines = [
        "# Provisional report: analytical groups and main SDA effects",
        "",
        "## Status and scope",
        "",
        "This is a provisional descriptive report based on the annual outputs available while the full campaign is still running. "
        "It is not thesis-ready evidence. Upstream balance failures were provisionally accepted for computation, remain recorded, "
        "and mean these outputs are not currently thesis-eligible.",
        "",
        f"The common comparison window is {min(common_years)}--{max(common_years)} ({len(common_years)} transitions) across "
        f"{', '.join(sorted(common['resolution'].unique()))}. Result year t denotes transition t-1 to t. Values below are "
        "gigatonnes of CO2 (Gt CO2), derived from stored kg CO2 outputs. Groups are static analytical classifications and do not "
        "measure legal Kyoto compliance.",
        "",
        "Primary effects are reconstructed without mixing hierarchy levels: emission intensity = D_v; production structure = "
        "D_A_x + D_C; final demand = D_YB + D_YF + D_y. Intensity and production effects are grouped by emitter country; "
        "final-demand effects are grouped by consumer country. A group-level total is intentionally not calculated because that "
        "would add unlike attribution roles.",
        "",
        "## Available coverage",
        "",
        "| Resolution | First result year | Last result year | Years available |",
        "|---|---:|---:|---:|",
    ]
    for resolution, row in coverage.sort_index().iterrows():
        lines.append(f"| {resolution} | {int(row['min'])} | {int(row['max'])} | {int(row['nunique'])} |")

    for resolution in sorted(common["resolution"].unique()):
        lines.extend(["", f"## {resolution}: cumulative effects, {min(common_years)}--{max(common_years)}", ""])
        table = cumulative.loc[cumulative["resolution"] == resolution].pivot(
            index="analysis_group", columns="main_effect", values="cumulative_contribution"
        )
        order = ["Red", "Non-Red", "Non-Participant"]
        cols = ["emission_intensity", "production_structure", "final_demand"]
        lines.extend([
            "| Group | Emission intensity (emitter) | Production structure (emitter) | Final demand (consumer) |",
            "|---|---:|---:|---:|",
        ])
        for group in order:
            if group in table.index:
                lines.append("| " + group + " | " + " | ".join(_fmt_gt(table.loc[group, col]) for col in cols) + " |")

        annual_world = world.loc[world["resolution"] == resolution].pivot(
            index="year_current", columns="main_effect", values="contribution"
        )
        world_cols = ["total_change", "emission_intensity", "production_structure", "final_demand"]
        lines.extend([
            "",
            "### World annual main effects",
            "",
            "| Result year | Total change | Emission intensity | Production structure | Final demand |",
            "|---:|---:|---:|---:|---:|",
        ])
        for year, row in annual_world.sort_index().iterrows():
            lines.append(f"| {int(year)} | " + " | ".join(_fmt_gt(row[col]) for col in world_cols) + " |")

    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "Signs describe contributions to annual attributed emission change: positive values increase emissions and negative values "
        "decrease them within this decomposition. Net values can conceal large offsetting positive and negative cells. Differences "
        "between resolutions may reflect aggregation and classification as well as substantive patterns. No causal, policy-effect, "
        "or legal-compliance conclusion is supported by this provisional summary.",
        "",
    ])
    (output_dir / "provisional_group_main_effect_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path, required=True)
    parser.add_argument("--countries-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.annual_root, args.groups_csv, args.countries_csv, args.output_dir)


if __name__ == "__main__":
    main()
