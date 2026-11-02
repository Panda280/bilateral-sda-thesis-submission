"""Build manifest-backed Chapter 4 case and Leontief-sub-effect artifacts.

The script consumes only the accepted static annual hierarchy and the v2
sign-stratified case-selection register.  It does not rerank candidates.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "implementation/bilateral_sda_attribution/docs/analysis/static_inputs"
CASES = ROOT / "implementation/bilateral_sda_attribution/docs/analysis/case_selection_v2_20260831b"
OUT = ROOT / "implementation/bilateral_sda_attribution/docs/analysis/chapter4_cases_leontief_20260831"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> dict:
    path = OUT / name
    frame.to_csv(path, index=False)
    return {"path": name, "rows": len(frame), "sha256": sha256(path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected_path = CASES / "selected_cases.csv"
    hierarchy_path = STATIC / "annual_hierarchy.csv"
    selected = pd.read_csv(selected_path)
    hierarchy = pd.read_csv(hierarchy_path)

    # Coverage summary preserves every registered selection stratum.
    case_summary = (
        selected.groupby(["entity_type", "selection_sign", "selection_scope"], dropna=False)
        .size().rename("selected_count").reset_index()
        .sort_values(["entity_type", "selection_sign", "selection_scope"])
    )

    # A compact audit table, not a new selection: the largest already-selected
    # cumulative absolute contribution in each entity-type/sign stratum.
    lead = (
        selected.sort_values(
            ["entity_type", "selection_sign", "absolute_total", "result_id"],
            ascending=[True, True, False, True],
        )
        .groupby(["entity_type", "selection_sign"], as_index=False, sort=True)
        .head(1)
        [["result_id", "entity_type", "canonical_entity", "selection_sign", "effect",
          "attribution_role", "resolution", "selection_scope", "signed_total",
          "absolute_total", "materiality", "sign_persistence", "units"]]
        .sort_values(["entity_type", "selection_sign"])
    )

    effects = ["leontief_production_structure", "D_C", "D_T_coef", "D_w_coef"]
    sub = hierarchy[hierarchy["effect"].isin(effects)].copy()
    keys = ["year_previous", "year_current", "resolution", "units"]
    value_wide = sub.pivot(index=keys, columns="effect", values="value").reset_index()
    value_wide["reconstructed_leontief"] = (
        value_wide["D_C"] + value_wide["D_T_coef"] + value_wide["D_w_coef"]
    )
    value_wide["reconstruction_residual"] = (
        value_wide["leontief_production_structure"] - value_wide["reconstructed_leontief"]
    )
    denom = value_wide[["D_C", "D_T_coef", "D_w_coef"]].abs().sum(axis=1)
    value_wide["residual_relative_to_component_gross"] = (
        value_wide["reconstruction_residual"].abs() / denom.where(denom != 0)
    )

    composition = sub[keys + ["effect", "value", "positive_contribution",
                               "negative_contribution", "gross_contribution"]].sort_values(keys + ["effect"])
    cumulative = (
        sub.groupby(["resolution", "units", "effect"], as_index=False)
        [["value", "positive_contribution", "negative_contribution", "gross_contribution"]].sum()
        .sort_values(["resolution", "effect"])
    )
    residual_summary = value_wide.groupby("resolution", as_index=False).agg(
        cumulative_leontief=("leontief_production_structure", "sum"),
        cumulative_D_C=("D_C", "sum"),
        cumulative_D_T_coef=("D_T_coef", "sum"),
        cumulative_D_w_coef=("D_w_coef", "sum"),
        cumulative_residual=("reconstruction_residual", "sum"),
        maximum_absolute_annual_residual=("reconstruction_residual", lambda x: x.abs().max()),
        maximum_relative_annual_residual=("residual_relative_to_component_gross", "max"),
    )
    residual_summary["units"] = "kg CO2"

    outputs = [
        write_csv(case_summary, "selected_case_summary.csv"),
        write_csv(lead, "lead_selected_cases.csv"),
        write_csv(composition, "leontief_subeffect_composition.csv"),
        write_csv(value_wide, "leontief_reconstruction_annual.csv"),
        write_csv(cumulative, "leontief_subeffect_cumulative.csv"),
        write_csv(residual_summary, "leontief_reconstruction_summary.csv"),
    ]
    max_rel = float(value_wide["residual_relative_to_component_gross"].max())
    manifest = {
        "schema_version": 1,
        "analysis_id": "chapter4-cases-leontief-20260831",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "inputs": [
            {"path": str(selected_path.relative_to(ROOT)), "sha256": sha256(selected_path)},
            {"path": str(hierarchy_path.relative_to(ROOT)), "sha256": sha256(hierarchy_path)},
        ],
        "outputs": outputs,
        "validation": {
            "passed": bool(
                len(selected) == 795
                and len(value_wide) == 42
                and selected["result_id"].is_unique
                and max_rel < 1e-5
                and set(sub["units"]) == {"kg CO2"}
            ),
            "selected_cases": len(selected),
            "annual_reconstructions": len(value_wide),
            "stable_case_ids_unique": bool(selected["result_id"].is_unique),
            "maximum_relative_annual_residual": max_rel,
            "units": sorted(sub["units"].unique().tolist()),
            "thesis_status": "accepted_provisional",
        },
        "notes": [
            "Lead cases are descriptive views of already-selected cases, not a second ranking exercise.",
            "Positive, negative and gross columns are retained per independently aggregated effect.",
            "The reconstruction residual is defined for net values at the world-resolution-year level.",
        ],
    }
    manifest_path = OUT / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not manifest["validation"]["passed"]:
        raise SystemExit("Validation failed; see analysis_manifest.json")


if __name__ == "__main__":
    main()
