"""Decompose the registered 2008--2009 cross-resolution divergence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EFFECTS = ["delta_g", "D_v", "D_C", "D_A_x", "D_T_coef", "D_w_coef", "D_YF", "D_YB", "D_y"]
PAIRS = [("R3", "2013"), ("R3", "2016"), ("2016", "2013")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, output_dir: Path) -> dict:
    data = pd.read_csv(source)
    required = {"year_current", "effect", "resolution", "units", "value"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    data = data[data["year_current"].isin([2008, 2009]) & data["effect"].isin(EFFECTS)].copy()
    if set(data["resolution"]) != {"2013", "2016", "R3"}:
        raise ValueError("Expected 2013, 2016 and R3 resolutions")
    if set(data["units"]) != {"kg CO2"}:
        raise ValueError("Expected kg CO2")

    wide = data.pivot(index=["year_current", "resolution"], columns="effect", values="value").reset_index()
    if wide[EFFECTS].isna().any().any():
        raise ValueError("Incomplete hierarchy for 2008--2009")
    wide["leontief_with_sourcing"] = wide["D_C"] + wide["D_A_x"]
    wide["final_demand_parent"] = wide["D_YF"] + wide["D_YB"] + wide["D_y"]
    wide["reconstructed_total"] = wide["D_v"] + wide["leontief_with_sourcing"] + wide["final_demand_parent"]
    wide["reconstruction_residual"] = wide["delta_g"] - wide["reconstructed_total"]
    wide["component_absolute_total"] = wide[["D_v", "D_C", "D_A_x", "D_YF", "D_YB", "D_y"]].abs().sum(axis=1)
    wide["residual_relative_to_component_absolute_total"] = (
        wide["reconstruction_residual"].abs() / wide["component_absolute_total"]
    )

    gaps = []
    compare = EFFECTS + ["leontief_with_sourcing", "final_demand_parent"]
    for year in [2008, 2009]:
        block = wide[wide["year_current"] == year].set_index("resolution")
        for left, right in PAIRS:
            total_gap = float(block.loc[left, "delta_g"] - block.loc[right, "delta_g"])
            for effect in compare:
                gap = float(block.loc[left, effect] - block.loc[right, effect])
                gaps.append({
                    "year_current": year,
                    "left_resolution": left,
                    "right_resolution": right,
                    "effect": effect,
                    "gap_kg_co2": gap,
                    "absolute_gap_kg_co2": abs(gap),
                    "total_gap_kg_co2": total_gap,
                })
    gaps_df = pd.DataFrame(gaps)
    gaps_df["absolute_gap_rank"] = gaps_df.groupby(
        ["year_current", "left_resolution", "right_resolution"]
    )["absolute_gap_kg_co2"].rank(method="first", ascending=False).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    wide_path = output_dir / "episode_effects.csv"
    gaps_path = output_dir / "resolution_effect_gaps.csv"
    wide.to_csv(wide_path, index=False)
    gaps_df.to_csv(gaps_path, index=False)
    manifest = {
        "analysis_id": "divergence-2008-2009-v1",
        "status": "accepted_provisional",
        "input": str(source),
        "input_sha256": sha256(source),
        "scope": {"years": [2008, 2009], "resolutions": ["2013", "2016", "R3"], "units": "kg CO2"},
        "validation": {
            "hierarchy_complete": True,
            "maximum_absolute_reconstruction_residual_kg": float(wide["reconstruction_residual"].abs().max()),
            "maximum_relative_reconstruction_residual": float(wide["residual_relative_to_component_absolute_total"].max()),
            "relative_reconstruction_tolerance": 2e-6,
            "reconstruction_passed": bool(wide["residual_relative_to_component_absolute_total"].max() <= 2e-6),
            "causal_interpretation": False,
        },
        "outputs": [
            {"path": wide_path.name, "rows": len(wide), "sha256": sha256(wide_path)},
            {"path": gaps_path.name, "rows": len(gaps_df), "sha256": sha256(gaps_path)},
        ],
        "limitation": "Resolution-labelled differences are accounting decompositions, not identified database causes.",
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
