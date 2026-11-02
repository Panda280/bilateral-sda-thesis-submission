#!/usr/bin/env python3
"""Render the registered annual sourcing-result paths used in Chapter 4.

The script is deliberately limited to descriptive accounting quantities.  It
does not run, or imply the results of, an association or displacement model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EFFECT_LABEL = {"D_C": r"$\mathbf{D}_{C}$", "D_YF": r"$\mathbf{D}_{Y_F}$"}
RESOLUTION_STYLE = {"2013": "-", "2016": "--", "R3": ":"}
GROUP_COLOUR = {"Red": "#2166ac", "Non-Red": "#b2182b", "Non-Participant": "#4d4d4d"}
REL_COLOUR = {"Cross-border": "#b2182b", "Domestic": "#2166ac"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configure() -> None:
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_relationships(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), sharex=True)
    for row, effect in enumerate(("D_C", "D_YF")):
        for col, resolution in enumerate(("2013", "2016", "R3")):
            ax = axes[row, col]
            selected = frame[(frame.effect == effect) & (frame.resolution.astype(str) == resolution)]
            for relationship in ("Cross-border", "Domestic"):
                series = selected[selected.relationship == relationship].sort_values("year_current")
                ax.plot(series.year_current, series.value, color=REL_COLOUR[relationship], lw=1.5,
                        marker="o", ms=2.4, label=relationship)
            ax.axhline(0, color="#777777", lw=.7)
            ax.set_title(f"{EFFECT_LABEL[effect]}, {resolution}")
            ax.grid(axis="y", color="#dddddd", lw=.5)
            if col == 0:
                ax.set_ylabel("Net contribution (Mt CO$_2$)")
            if row == 1:
                ax.set_xlabel("Current year")
            ax.set_xticks(range(2001, 2015, 3))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Annual domestic and cross-border sourcing contributions", y=.995, fontsize=11)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .958),
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .895))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_groups(frame: pd.DataFrame, field: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0), sharex=True)
    for row, effect in enumerate(("D_C", "D_YF")):
        for col, role in enumerate(("source", "destination")):
            ax = axes[row, col]
            selected = frame[(frame.effect == effect) & (frame.role == role)]
            for group in ("Red", "Non-Red", "Non-Participant"):
                for resolution in ("2013", "2016", "R3"):
                    series = selected[(selected.group == group) &
                                      (selected.resolution.astype(str) == resolution)].sort_values("year_current")
                    ax.plot(series.year_current, series[field], color=GROUP_COLOUR[group],
                            ls=RESOLUTION_STYLE[resolution], lw=1.25)
            if field == "value":
                ax.axhline(0, color="#777777", lw=.7)
            ax.set_title(f"{EFFECT_LABEL[effect]}, {role} attribution")
            ax.grid(axis="y", color="#dddddd", lw=.5)
            if col == 0:
                label = "Net" if field == "value" else "Gross"
                ax.set_ylabel(f"{label} contribution (Mt CO$_2$)")
            if row == 1:
                ax.set_xlabel("Current year")
            ax.set_xticks(range(2001, 2015, 3))
    group_handles = [plt.Line2D([0], [0], color=GROUP_COLOUR[g], lw=2, label=g)
                     for g in ("Red", "Non-Red", "Non-Participant")]
    res_handles = [plt.Line2D([0], [0], color="black", ls=RESOLUTION_STYLE[r], label=r)
                   for r in ("2013", "2016", "R3")]
    fig.legend(group_handles + res_handles,
               [h.get_label() for h in group_handles + res_handles],
               loc="upper center", bbox_to_anchor=(.5, .955), ncol=6, frameon=False)
    descriptor = "net" if field == "value" else "gross"
    fig.suptitle(f"Annual analytical-group {descriptor} sourcing contributions", y=.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relationship_input = args.input_dir / "international_domestic.csv"
    group_input = args.input_dir / "group_trends.csv"
    relationships = pd.read_csv(relationship_input, dtype={"resolution": str})
    groups = pd.read_csv(group_input, dtype={"resolution": str})
    required_rel = {"D_C", "D_YF"}, {"Domestic", "Cross-border"}, {"2013", "2016", "R3"}
    observed_rel = set(relationships.effect), set(relationships.relationship), set(relationships.resolution)
    if observed_rel != required_rel or set(relationships.units) != {"Mt CO2"}:
        raise ValueError("unexpected relationship-panel scope")
    if (set(groups.effect) != {"D_C", "D_YF"} or set(groups.role) != {"source", "destination"}
            or set(groups.group) != {"Red", "Non-Red", "Non-Participant"}
            or set(groups.resolution) != {"2013", "2016", "R3"} or set(groups.units) != {"Mt CO2"}):
        raise ValueError("unexpected analytical-group panel scope")
    outputs = {
        "international_domestic": args.output_dir / "international_domestic_annual_paths.pdf",
        "group_net": args.output_dir / "analytical_group_annual_net.pdf",
        "group_gross": args.output_dir / "analytical_group_annual_gross.pdf",
    }
    configure()
    plot_relationships(relationships, outputs["international_domestic"])
    plot_groups(groups, "value", outputs["group_net"])
    plot_groups(groups, "gross_contribution", outputs["group_gross"])
    manifest = {
        "schema_version": 1,
        "analysis_id": "chapter-4-sourcing-path-figures-v1",
        "status": "accepted_provisional",
        "indicator": "fossil-origin CO2",
        "price_basis": "CYP_(t-1) -> PYP_t",
        "units": "Mt CO2",
        "generator": str(Path(__file__).as_posix()),
        "inputs": [{"path": str(p), "sha256": sha256(p)} for p in (relationship_input, group_input)],
        "outputs": [{"path": str(p), "sha256": sha256(p)} for p in outputs.values()],
        "interpretation_limit": "Descriptive signed accounting contributions; not association or causal displacement evidence.",
    }
    (args.output_dir / "sourcing_path_figures.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
