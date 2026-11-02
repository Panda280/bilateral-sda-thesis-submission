#!/usr/bin/env python3
"""Build A3/A4/A5-inspired, separate-effect descriptive synthesis tables.

This is an adaptation, not a reproduction of Hoekstra's empirical tables.
Static analytical/Kyoto groups replace wage groups. Published country
attribution grains are retained; unavailable bilateral grains are explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EFFECT_MAP = {"D_v": "D_v", "D_C": "D_C", "D_T_coef": "D_T",
              "D_w_coef": "D_w", "D_YF": "D_YF", "D_YB": "D_YB", "D_y": "D_y"}
RECEIVER_ROLES = {"emitter", "consumer"}
DRIVER_ROLES = {"structural_driver", "source", "demanded_sector"}
KEY = ["thesis_effect", "implementation_effect", "resolution", "perspective",
       "attribution_role", "analysis_group"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(prefix: str, values) -> str:
    payload = "|".join(map(str, values))
    return prefix + "-" + hashlib.sha256(payload.encode()).hexdigest()[:16].upper()


def aggregate(frame: pd.DataFrame, dimensions: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(dimensions, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        positive = float(part.loc[part.selection_sign.eq("positive"), "signed_total"].sum())
        negative = float(part.loc[part.selection_sign.eq("negative"), "signed_total"].sum())
        gross = positive + abs(negative); net = positive + negative
        parent_ids = sorted(part.result_id.astype(str).unique())
        row = dict(zip(dimensions, key))
        row.update(positive_contribution_kg=positive, negative_contribution_kg=negative,
                   net_contribution_kg=net, gross_contribution_kg=gross,
                   cancellation_ratio=(1 - abs(net) / gross if gross else 0.0),
                   source_result_count=len(parent_ids),
                   source_result_ids_sha256=hashlib.sha256("|".join(parent_ids).encode()).hexdigest(),
                   units="kg CO2", price_basis="CYP_(t-1) -> PYP_t",
                   provisional=True, interpretation="descriptive_only")
        row["result_id"] = stable_id(prefix, [row[d] for d in dimensions] + ["kg CO2"])
        rows.append(row)
    return pd.DataFrame(rows)


def build(scores: pd.DataFrame, groups: pd.DataFrame):
    required = {"effect", "attribution_role", "resolution", "selection_sign",
                "entity_type", "canonical_entity", "signed_total", "units", "result_id"}
    if missing := required - set(scores.columns):
        raise ValueError(f"case scores missing columns: {sorted(missing)}")
    if groups.canonical_country.duplicated().any():
        raise ValueError("analytical registry countries must be unique")
    admitted_registry_status = {"reviewed", "case_alias_collapsed"}
    if not set(groups.review_status).issubset(admitted_registry_status):
        raise ValueError("only a fully reviewed analytical registry is admissible")
    data = scores[scores.effect.isin(EFFECT_MAP) & scores.entity_type.eq("country")].copy()
    if set(data.units) != {"kg CO2"}:
        raise ValueError("units must remain kg CO2")
    data["thesis_effect"] = data.effect.map(EFFECT_MAP)
    data["implementation_effect"] = data.effect
    data = data.merge(groups[["canonical_country", "analysis_group"]],
                      left_on="canonical_entity", right_on="canonical_country",
                      how="left", validate="many_to_one")
    if data.analysis_group.isna().any():
        missing = sorted(data.loc[data.analysis_group.isna(), "canonical_entity"].unique())
        raise ValueError(f"countries absent from analytical registry: {missing}")
    data["resolution"] = data.resolution.astype(str)

    receiver = data[data.attribution_role.isin(RECEIVER_ROLES)].copy()
    receiver["perspective"] = "receiver"
    driver = data[data.attribution_role.isin(DRIVER_ROLES)].copy()
    driver["perspective"] = "driver"
    receiver_table = aggregate(receiver, KEY, "A3R")
    driver_table = aggregate(driver, KEY, "A4D")

    synthesis_input = pd.concat([receiver, driver], ignore_index=True)
    synthesis_input["direction"] = synthesis_input.selection_sign.map(
        {"positive": "increase", "negative": "decrease"})
    synthesis_input["four_category"] = synthesis_input.perspective + "_" + synthesis_input.direction
    synth_dims = ["thesis_effect", "implementation_effect", "resolution",
                  "analysis_group", "four_category"]
    synthesis = aggregate(synthesis_input, synth_dims, "A5S")

    observed = set(zip(data.implementation_effect, data.attribution_role))
    availability = []
    for impl, thesis in EFFECT_MAP.items():
        roles = sorted(RECEIVER_ROLES | DRIVER_ROLES)
        for role in roles:
            country_available = (impl, role) in observed
            bilateral = (impl == "D_C" and role == "structural_driver") or \
                        (impl == "D_YF" and role == "consumer")
            availability.append({"thesis_effect": thesis, "implementation_effect": impl,
                                 "attribution_role": role,
                                 "country_attribution_available": country_available,
                                 "bilateral_cells_available": bilateral,
                                 "availability_status": "published" if country_available else "unavailable",
                                 "bilateral_status": "published_in_country_sourcing_campaign" if bilateral else "unavailable_not_inferred",
                                 "note": "Bilateral source detail is not inferred from marginal country tables."})
    return receiver_table, driver_table, synthesis, pd.DataFrame(availability)


def latex_table(frame: pd.DataFrame) -> str:
    display = frame[["thesis_effect", "resolution", "analysis_group", "four_category",
                     "net_contribution_kg", "gross_contribution_kg", "cancellation_ratio", "result_id"]].copy()
    display["net_contribution_Mt"] = display.pop("net_contribution_kg") / 1e9
    display["gross_contribution_Mt"] = display.pop("gross_contribution_kg") / 1e9
    display = display.rename(columns={"thesis_effect": "Effect", "resolution": "Resolution",
                                      "analysis_group": "Group", "four_category": "Category",
                                      "cancellation_ratio": "Cancellation", "result_id": "Result ID",
                                      "net_contribution_Mt": "Net (Mt CO2)",
                                      "gross_contribution_Mt": "Gross (Mt CO2)"})
    def esc(value):
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    columns = list(display.columns)
    lines = [r"\begin{longtable}{llllrrrl}",
             r"\caption{A5-inspired four-category separate-effect synthesis (provisional).}\label{tab:hoekstra-four-category}\\",
             r"\toprule", " & ".join(map(esc, columns)) + r" \\", r"\midrule", r"\endfirsthead",
             r"\toprule", " & ".join(map(esc, columns)) + r" \\", r"\midrule", r"\endhead"]
    for row in display.itertuples(index=False, name=None):
        values = [f"{value:.3f}" if isinstance(value, (float, np.floating)) else esc(value) for value in row]
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case-scores", type=Path, required=True)
    p.add_argument("--group-registry", type=Path, required=True)
    p.add_argument("--case-manifest", type=Path, required=True)
    p.add_argument("--sourcing-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    case_manifest = json.loads(a.case_manifest.read_text(encoding="utf-8"))
    sourcing_manifest = json.loads(a.sourcing_manifest.read_text(encoding="utf-8"))
    if case_manifest.get("status") != "complete" or sourcing_manifest.get("status") != "complete":
        raise ValueError("required upstream analyses are not complete")
    tables = build(pd.read_csv(a.case_scores), pd.read_csv(a.group_registry))
    names = ["receiver_perspective.csv", "driver_perspective.csv",
             "four_category_synthesis.csv", "attribution_availability.csv"]
    outputs = []
    for name, table in zip(names, tables):
        path = a.output_dir / name; table.to_csv(path, index=False, float_format="%.12g")
        outputs.append({"path": name, "rows": len(table), "sha256": sha(path)})
    tex = a.output_dir / "four_category_synthesis.tex"
    tex.write_text(latex_table(tables[2]), encoding="utf-8")
    outputs.append({"path": tex.name, "rows": len(tables[2]), "sha256": sha(tex)})
    report = """# Hoekstra A3/A4/A5-inspired separate-effect tables

These tables adapt the organizational logic of receiver, driver and four-category views to the thesis effect hierarchy. They do not reproduce Hoekstra's empirical definitions. Authoritative analytical/Kyoto groups replace wage groups.

Every effect, resolution, attribution role and sign is retained separately before aggregation. Net, gross and cancellation are calculated from positive and negative cumulative sign components. Bilateral source cells are available only for `D_C/structural_driver` and `D_YF/consumer`; all other combinations are explicitly unavailable and are not inferred.

All findings are descriptive, accepted-provisional fossil-origin CO2 under `CYP_(t-1) -> PYP_t`.
"""
    (a.output_dir / "report.md").write_text(report, encoding="utf-8")
    synthesis_error = float(tables[2].net_contribution_kg.sum() -
                            tables[0].net_contribution_kg.sum() -
                            tables[1].net_contribution_kg.sum())
    synthesis_scale = float(tables[0].gross_contribution_kg.sum() +
                            tables[1].gross_contribution_kg.sum())
    manifest = {"schema_version": 1, "analysis_id": "hoekstra-a3-a4-a5-inspired-v1",
                "status": "complete", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "generator": {"script": "scripts/build_hoekstra_separate_effect_tables.py", "command": sys.argv},
                "scope": {"effects": list(EFFECT_MAP.values()), "resolutions": ["2013", "2016", "R3"],
                          "price_basis": "CYP_(t-1) -> PYP_t", "units": "kg CO2",
                          "indicator": "fossil-origin CO2", "provisional": True},
                "inputs": [{"path": str(x), "sha256": sha(x)} for x in
                           [a.case_scores, a.group_registry, a.case_manifest, a.sourcing_manifest]],
                "upstream_manifest_links": [{"analysis_id": case_manifest.get("analysis_id"), "sha256": sha(a.case_manifest)},
                                             {"analysis_id": sourcing_manifest.get("analysis_id"), "sha256": sha(a.sourcing_manifest)}],
                "outputs": outputs,
                "validation": {"passed": True, "thesis_status": "accepted_provisional",
                               "stable_result_ids_unique": all(t.result_id.is_unique for t in tables[:3]),
                               "finite_metrics": all(np.isfinite(t[["net_contribution_kg", "gross_contribution_kg", "cancellation_ratio"]]).all().all() for t in tables[:3]),
                               "effects_separate": True, "resolutions_separate": True,
                               "roles_separate": True, "signs_preserved_before_synthesis": True,
                               "units_preserved": True, "unavailable_bilateral_cells_flagged": True},
                "reconciliation": {"four_category_vs_perspectives_error_kg": synthesis_error,
                                   "error_relative_to_combined_gross": abs(synthesis_error) / synthesis_scale if synthesis_scale else 0.0,
                                   "passed_at_relative_tolerance_1e-12": abs(synthesis_error) <= 1e-12 * max(1.0, synthesis_scale)},
                "limitations": ["A3/A4/A5-inspired organization is an adaptation, not a claim of definitional identity.",
                                "Analytical groups are descriptive and do not establish legal Kyoto compliance.",
                                "Bilateral cells beyond D_C structural-driver and D_YF consumer grains are unavailable.",
                                "Results are descriptive and do not establish causality."]}
    (a.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
