"""Regenerate annual_hierarchy_net_gross.pdf restricted to 2001-2012 (end of
Kyoto CP1). Standalone copy of the hierarchy_figure() logic from
build_hierarchy_resolution_figures.py, scoped to avoid touching that shared
script (which also emits matched_cross_resolution_annual.pdf, owned by a
different results section) or its 14-transition validation gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESOLUTIONS = ("2013", "2016", "R3")
EFFECTS = ("delta_g", "emission_intensity", "leontief_production_structure", "final_demand")
LABELS = {
    "delta_g": "Total change",
    "emission_intensity": "Emission intensity",
    "leontief_production_structure": "Production structure",
    "final_demand": "Final demand",
}
COLORS = {"2013": "#4477AA", "2016": "#EE6677", "R3": "#228833"}
YEAR_LAST = 2012


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"resolution": str})
    subset = frame[frame.effect.isin(EFFECTS) & (frame.year_current <= YEAR_LAST)].copy()
    n_years = subset.year_current.nunique()
    if set(subset.effect) != set(EFFECTS) or set(subset.resolution) != set(RESOLUTIONS):
        raise ValueError("incomplete effect or resolution coverage")
    if len(subset) != n_years * len(EFFECTS) * len(RESOLUTIONS):
        raise ValueError("annual semantic key is not complete and unique")
    numeric = ["value", "positive_contribution", "negative_contribution", "gross_contribution"]
    if not np.isfinite(subset[numeric].to_numpy()).all():
        raise ValueError("non-finite contribution")
    if not np.allclose(subset.positive_contribution + subset.negative_contribution, subset.value,
                        rtol=1e-8, atol=1.0):
        raise ValueError("positive + negative does not reproduce net")
    if not np.allclose(subset.positive_contribution - subset.negative_contribution,
                        subset.gross_contribution, rtol=1e-8, atol=1.0):
        raise ValueError("positive - negative does not reproduce gross")
    return subset


def hierarchy_figure(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    primary = frame[frame.effect != "delta_g"]
    summary = primary.groupby(["effect", "resolution"], as_index=False)[
        ["value", "positive_contribution", "negative_contribution", "gross_contribution"]
    ].sum()
    for column in ["value", "positive_contribution", "negative_contribution", "gross_contribution"]:
        summary[column] /= 1e12
    summary["units"] = "Gt CO2"
    measures = [("value", "Net"), ("positive_contribution", "Positive"),
                ("negative_contribution", "Negative"), ("gross_contribution", "Gross")]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharey=True)
    x = np.arange(4)
    width = .23
    for ax, effect in zip(axes, EFFECTS[1:]):
        part = summary[summary.effect == effect].set_index("resolution")
        for i, resolution in enumerate(RESOLUTIONS):
            ax.bar(x + (i - 1) * width, [part.loc[resolution, col] for col, _ in measures],
                   width, label=resolution, color=COLORS[resolution])
        ax.axhline(0, color="#333333", linewidth=.7)
        ax.set_title(LABELS[effect])
        ax.set_xticks(x, [label for _, label in measures], rotation=28, ha="right")
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel(f"Cumulative contribution, 2001–{YEAR_LAST} (Gt CO$_2$)")
    axes[-1].legend(title="Target grid", frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annual = load(args.annual_input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = hierarchy_figure(annual, args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
