"""Plot the complete registered SDA leaf hierarchy as signed annual stacks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEAVES = ["D_v", "D_C", "D_T_coef", "D_w_coef", "D_YF", "D_YB", "D_y"]
LABELS = {
    "D_v": r"$D_v$ intensity",
    "D_C": r"$D_C$ intermediate sourcing",
    "D_T_coef": r"$D_T$ recipe composition",
    "D_w_coef": r"$D_w$ input weight",
    "D_YF": r"$D_{Y_F}$ final-demand sourcing",
    "D_YB": r"$D_{Y_B}$ final-demand composition",
    "D_y": r"$D_y$ final-demand volume",
}
COLORS = {
    "D_v": "#6a3d9a",
    "D_C": "#1f78b4",
    "D_T_coef": "#33a02c",
    "D_w_coef": "#b2df8a",
    "D_YF": "#e31a1c",
    "D_YB": "#fb9a99",
    "D_y": "#ff7f00",
}
KG_PER_GT = 1e12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(input_csv: Path, output_dir: Path, year_last: int = 2014) -> dict:
    data = pd.read_csv(input_csv)
    data = data[data.year_current <= year_last]
    n_years = year_last - 2000
    required = {"year_current", "effect", "resolution", "units", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if set(data.units) != {"kg CO2"}:
        raise ValueError(f"unexpected units: {sorted(data.units.unique())}")
    available = set(data.effect)
    if set(LEAVES) - available:
        raise ValueError(f"missing hierarchy leaves: {sorted(set(LEAVES) - available)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 10.2), sharex=True, sharey=True)
    for ax, resolution in zip(axes, ["2013", "2016", "R3"]):
        subset = data[(data.resolution.astype(str) == resolution) & data.effect.isin(LEAVES)]
        pivot = subset.pivot(index="year_current", columns="effect", values="value").reindex(columns=LEAVES)
        if pivot.isna().any().any() or len(pivot) != n_years:
            raise ValueError(f"incomplete hierarchy for {resolution}")
        values = pivot.to_numpy(float) / KG_PER_GT
        years = pivot.index.to_numpy()
        positive_bottom = np.zeros(len(years))
        negative_bottom = np.zeros(len(years))
        for index, effect in enumerate(LEAVES):
            series = values[:, index]
            positive = np.where(series > 0, series, 0)
            negative = np.where(series < 0, series, 0)
            ax.bar(years, positive, bottom=positive_bottom, width=.72,
                   color=COLORS[effect], label=LABELS[effect])
            ax.bar(years, negative, bottom=negative_bottom, width=.72,
                   color=COLORS[effect])
            positive_bottom += positive
            negative_bottom += negative
        total = data[(data.resolution.astype(str) == resolution) & (data.effect == "delta_g")]
        total = total.set_index("year_current").reindex(years).value.to_numpy(float) / KG_PER_GT
        leaf_sum = values.sum(axis=1)
        if not np.allclose(leaf_sum, total, rtol=2e-5, atol=2e-5):
            raise ValueError(f"leaf reconstruction failed for {resolution}")
        ax.plot(years, total, color="black", marker="o", linewidth=1.3,
                markersize=3.5, label=r"$D_{total}$")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_title(f"Target grid {resolution}", loc="left", fontweight="bold")
        ax.set_ylabel("Annual contribution (Gt CO$_2$)")
        ax.grid(axis="y", alpha=.25)
    axes[-1].set_xlabel("Transition ending in year")
    axes[-1].set_xticks(range(2001, year_last + 1))
    axes[-1].tick_params(axis="x", rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .995))
    fig.suptitle("Complete signed SDA hierarchy: all registered leaf subeffects",
                 y=1.035, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))

    pdf = output_dir / "detailed_hierarchical_signed_stack.pdf"
    png = output_dir / "detailed_hierarchical_signed_stack.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "analysis_id": "detailed-hierarchical-signed-stack-v1",
        "status": "accepted_provisional",
        "input": str(input_csv),
        "input_sha256": sha256(input_csv),
        "scope": {
            "years": f"2001-{year_last} adjacent transitions",
            "resolutions": ["2013", "2016", "R3"],
            "leaves": LEAVES,
            "units_input": "kg CO2",
            "display_conversion": "kg / 1e12 = Gt CO2",
            "price_basis": "CYP_(t-1) -> PYP_t",
        },
        "validation": {
            "leaf_sum_reconstructs_total": True,
            "signed_values_preserved": True,
            "positive_and_negative_stacked_separately": True,
        },
        "outputs": [
            {"path": pdf.name, "sha256": sha256(pdf)},
            {"path": png.name, "sha256": sha256(png)},
        ],
        "limitation": "Accepted-provisional fossil-origin CO2; upstream audit remains unresolved.",
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-last", type=int, default=2014)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output_dir, args.year_last), indent=2))


if __name__ == "__main__":
    main()
