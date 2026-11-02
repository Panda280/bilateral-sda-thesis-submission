#!/usr/bin/env python3
"""Render static, signed temporal and analytical-group result figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, output: Path) -> list[dict]:
    records = []
    for extension in ("pdf", "png"):
        path = output.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        records.append({"path": path.name, "sha256": sha256(path)})
    plt.close(fig)
    return records


def effect_stacks(hierarchy: pd.DataFrame, output_dir: Path) -> list[dict]:
    effects = [effect for effect in ("emission_intensity", "leontief_production_structure", "final_demand") if effect in set(hierarchy.effect)]
    resolutions = [r for r in ("2013", "2016", "R3") if r in set(hierarchy.resolution.astype(str))]
    fig, axes = plt.subplots(len(resolutions), 1, figsize=(9, 7.5), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {"emission_intensity": "#4477AA", "leontief_production_structure": "#EE6677", "final_demand": "#228833"}
    for ax, resolution in zip(axes, resolutions):
        subset = hierarchy[hierarchy.resolution.astype(str).eq(resolution)]
        pivot = subset[subset.effect.isin(effects)].pivot(index="year_current", columns="effect", values="value").sort_index()
        positive = np.zeros(len(pivot)); negative = np.zeros(len(pivot))
        for effect in effects:
            values = pivot[effect].to_numpy(float) / 1e9
            bottoms = np.where(values >= 0, positive, negative)
            ax.bar(pivot.index, values, bottom=bottoms, label=effect, color=colors[effect], width=.75)
            positive += np.maximum(values, 0); negative += np.minimum(values, 0)
        ax.axhline(0, color="black", linewidth=.7); ax.set_ylabel(f"{resolution}\nMt CO$_2$")
    axes[0].legend(ncol=len(effects), frameon=False, loc="upper left")
    axes[-1].set_xlabel("Transition end year")
    fig.suptitle("Annual primary-effect contributions by resolution")
    return save(fig, output_dir / "temporal_primary_effect_stacks")


def anomaly_heatmap(candidates: pd.DataFrame, output_dir: Path) -> list[dict]:
    candidates = candidates.copy()
    candidates["series"] = candidates.effect.astype(str) + " · " + candidates.resolution.astype(str) + " · " + candidates.metric.astype(str)
    matrix = candidates.pivot_table(index="series", columns="year_current", values="robust_mad_score", aggfunc="max", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, max(4, .24 * len(matrix))))
    maximum = max(3.5, float(np.abs(matrix.to_numpy()).max()))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=6)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45)
    ax.set_xlabel("Transition end year"); ax.set_title("Robust temporal anomaly candidates (median/MAD score)")
    fig.colorbar(image, ax=ax, label="Robust MAD score")
    return save(fig, output_dir / "temporal_anomaly_heatmap")


def group_scorecards(groups: pd.DataFrame, output_dir: Path) -> list[dict]:
    cumulative = groups.groupby(["effect", "resolution", "group"], as_index=False).agg(net=("value", "sum"), gross=("gross_contribution", "sum"))
    effects = sorted(cumulative.effect.unique()); resolutions = [r for r in ("2013", "2016", "R3") if r in set(cumulative.resolution.astype(str))]
    fig, axes = plt.subplots(len(effects), len(resolutions), figsize=(12, 4 * len(effects)), squeeze=False,
                             constrained_layout=True)
    palette = {"Red": "#CC6677", "Non-Red": "#4477AA", "Non-Participant": "#228833"}
    for i, effect in enumerate(effects):
        for j, resolution in enumerate(resolutions):
            ax = axes[i, j]; subset = cumulative[(cumulative.effect == effect) & cumulative.resolution.astype(str).eq(resolution)].sort_values("net")
            ax.barh(subset.group, subset.net, color=[palette.get(x, "#999999") for x in subset.group])
            ax.axvline(0, color="black", linewidth=.7); ax.set_title(f"{effect} · {resolution}", pad=8)
            if i == len(effects) - 1:
                ax.set_xlabel("Cumulative net (Mt CO$_2$)")
    fig.suptitle("Descriptive analytical-group cumulative scorecards (not legal Kyoto compliance)")
    return save(fig, output_dir / "analytical_group_cumulative_scorecards")


def country_scorecards(country_path: Path, output_dir: Path) -> list[dict]:
    countries = pd.read_csv(country_path, dtype={"resolution": str})
    countries = countries[countries.role.eq("source")]
    effects = sorted(countries.effect.unique())
    resolutions = [r for r in ("2013", "2016", "R3") if r in set(countries.resolution)]
    fig, axes = plt.subplots(len(effects), len(resolutions), figsize=(13, 8), squeeze=False,
                             constrained_layout=True)
    for i, effect in enumerate(effects):
        for j, resolution in enumerate(resolutions):
            ax = axes[i, j]
            subset = countries[(countries.effect == effect) & (countries.resolution == resolution)].nsmallest(12, "cumulative_net").sort_values("cumulative_net")
            colors = np.where(subset.cumulative_net < 0, "#4477AA", "#CC6677")
            ax.barh(subset.canonical_country, subset.cumulative_net / 1e9, color=colors)
            ax.axvline(0, color="black", linewidth=.7)
            ax.set_title(f"{effect} source · {resolution}")
            if i == len(effects) - 1:
                ax.set_xlabel("Cumulative net (Mt CO$_2$)")
    fig.suptitle("Countries with the lowest cumulative source-subeffect contributions")
    return save(fig, output_dir / "country_cumulative_source_scorecards")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-hierarchy", type=Path, required=True)
    parser.add_argument("--anomaly-candidates", type=Path, required=True)
    parser.add_argument("--group-trends", type=Path, required=True)
    parser.add_argument("--kyoto-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy = pd.read_csv(args.annual_hierarchy, dtype={"resolution": str})
    anomalies = pd.read_csv(args.anomaly_candidates, dtype={"resolution": str})
    groups = pd.read_csv(args.group_trends, dtype={"resolution": str})
    outputs = effect_stacks(hierarchy, args.output_dir)
    outputs += anomaly_heatmap(anomalies, args.output_dir)
    outputs += group_scorecards(groups, args.output_dir)
    input_paths = [args.annual_hierarchy, args.anomaly_candidates, args.group_trends]
    if args.kyoto_dir:
        country_path = args.kyoto_dir / "cumulative_country_scorecards.csv"
        outputs += country_scorecards(country_path, args.output_dir)
        input_paths.extend([country_path, args.kyoto_dir / "analysis_manifest.json"])
    manifest = {"schema_version": 1, "analysis_id": "temporal-kyoto-static-figures-v1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "accepted_provisional",
                "scope": {"units": "Mt CO2 (exact conversion from kg CO2 by 1e9 where required)",
                          "price_basis": "CYP_(t-1) -> PYP_t", "indicator": "fossil-origin CO2"},
                "inputs": [{"path": str(p), "sha256": sha256(p)} for p in input_paths],
                "outputs": outputs,
                "limitations": ["Accepted provisional because upstream validation remains unresolved.",
                                "Analytical groups are descriptive and do not measure legal Kyoto compliance."]}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
