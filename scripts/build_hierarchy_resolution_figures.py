"""Build manifest-backed Results figures for annual hierarchy and resolution agreement."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"resolution": str})
    required = {"year_previous", "year_current", "effect", "resolution", "units", "value",
                "positive_contribution", "negative_contribution", "gross_contribution"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    subset = frame[frame.effect.isin(EFFECTS)].copy()
    if set(subset.effect) != set(EFFECTS) or set(subset.resolution) != set(RESOLUTIONS):
        raise ValueError("incomplete effect or resolution coverage")
    if set(subset.units) != {"kg CO2"}:
        raise ValueError(f"unexpected units: {sorted(subset.units.unique())}")
    key = ["year_previous", "year_current", "effect", "resolution"]
    if subset.duplicated(key).any() or len(subset) != 14 * 4 * 3:
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
    return subset.sort_values(key)


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
    x = np.arange(4); width = .23
    for ax, effect in zip(axes, EFFECTS[1:]):
        part = summary[summary.effect == effect].set_index("resolution")
        for i, resolution in enumerate(RESOLUTIONS):
            ax.bar(x + (i - 1) * width, [part.loc[resolution, col] for col, _ in measures],
                   width, label=resolution, color=COLORS[resolution])
        ax.axhline(0, color="#333333", linewidth=.7)
        ax.set_title(LABELS[effect])
        ax.set_xticks(x, [label for _, label in measures], rotation=28, ha="right")
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Cumulative contribution, 2001–2014 (Gt CO$_2$)")
    axes[-1].legend(title="Target grid", frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return summary


def resolution_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = (("2013", "2016"), ("2013", "R3"), ("2016", "R3"))
    for effect in EFFECTS:
        wide = frame[frame.effect == effect].pivot(index="year_current", columns="resolution", values="value")
        gross = frame[frame.effect == effect].pivot(index="year_current", columns="resolution", values="gross_contribution")
        for left, right in pairs:
            difference = wide[left] - wide[right]
            denominator = (gross[left].abs() + gross[right].abs()) / 2
            rows.append({
                "effect": effect, "resolution_left": left, "resolution_right": right,
                "n_matched": len(wide),
                "sign_agreement": float((np.sign(wide[left]) == np.sign(wide[right])).mean()),
                "pearson_r": float(wide[left].corr(wide[right], method="pearson")),
                "spearman_rho": float(wide[left].corr(wide[right], method="spearman")),
                "mean_absolute_difference_kg_co2": float(difference.abs().mean()),
                "normalized_mean_absolute_difference": float(difference.abs().sum() / denominator.sum()),
                "normalization_denominator": "sum of pairwise mean annual gross contributions",
            })
    return pd.DataFrame(rows)


def resolution_figure(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), sharex=True)
    for ax, effect in zip(axes.flat, EFFECTS):
        part = frame[frame.effect == effect]
        for resolution in RESOLUTIONS:
            series = part[part.resolution == resolution]
            ax.plot(series.year_current, series.value / 1e12, marker="o", markersize=2.8,
                    linewidth=1.25, color=COLORS[resolution], label=resolution)
        ax.axhline(0, color="#333333", linewidth=.7)
        ax.set_title(LABELS[effect])
        ax.grid(alpha=.22)
    axes[0, 0].set_ylabel("Annual contribution (Gt CO$_2$)")
    axes[1, 0].set_ylabel("Annual contribution (Gt CO$_2$)")
    axes[1, 0].set_xlabel("Transition ending in year")
    axes[1, 1].set_xlabel("Transition ending in year")
    axes[0, 1].legend(title="Target grid", frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-input", type=Path, required=True)
    parser.add_argument("--cross-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annual = load(args.annual_input)
    cross = load(args.cross_input)
    hierarchy_pdf = args.output_dir / "annual_hierarchy_net_gross.pdf"
    resolution_pdf = args.output_dir / "matched_cross_resolution_annual.pdf"
    summary = hierarchy_figure(annual, hierarchy_pdf)
    resolution_figure(cross, resolution_pdf)
    metrics = resolution_metrics(cross)
    summary_path = args.output_dir / "annual_hierarchy_cumulative.csv"
    metrics_path = args.output_dir / "cross_resolution_metrics.csv"
    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    outputs = [hierarchy_pdf, resolution_pdf, summary_path, metrics_path]
    manifest = {
        "schema_version": 1, "analysis_id": "results-hierarchy-resolution-figures-v1",
        "status": "accepted_provisional", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"script": str(Path(__file__)), "command": " ".join(__import__("sys").argv)},
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in (args.annual_input, args.cross_input)],
        "outputs": [{"path": p.name, "sha256": sha(p), **({"rows": len(summary) if p == summary_path else len(metrics)} if p.suffix == ".csv" else {})} for p in outputs],
        "coverage": {"transitions": 14, "year_current": [2001, 2014], "effects": list(EFFECTS), "resolutions": list(RESOLUTIONS)},
        "scope": {"indicator": "fossil-origin CO2", "price_basis": "CYP_(t-1) -> PYP_t", "input_units": "kg CO2", "display_units": "Gt CO2"},
        "validation": {"passed": True, "gates": {"complete_unique_panel": True, "finite": True, "signed_identity": True, "gross_identity": True}},
        "limitations": ["Accepted provisional; downstream reconciliation does not resolve upstream CYP/PYP validation failures.", "Target-grid agreement does not imply database interchangeability."],
    }
    manifest_path = args.output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
