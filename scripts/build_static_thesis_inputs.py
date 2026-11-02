#!/usr/bin/env python3
"""Build manifest-backed static thesis inputs from accepted provisional analyses."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_view(output: Path, name: str, frame: pd.DataFrame, sources: list[Path], rule: str) -> None:
    table = output / f"{name}.csv"
    frame.to_csv(table, index=False, float_format="%.12g")
    manifest = {
        "schema_version": 1, "analysis_id": f"static-input-{name}-v1",
        "status": "accepted_provisional", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"script": "scripts/build_static_thesis_inputs.py", "rule": rule},
        "inputs": [{"path": str(p.resolve()), "sha256": sha(p)} for p in sources],
        "outputs": [{"path": table.name, "sha256": sha(table), "rows": len(frame)}],
        "validation": {"passed": False, "thesis_status": "accepted_provisional",
                       "gates": {"finite": bool(frame[[c for c in ("value","positive_contribution","negative_contribution","gross_contribution") if c in frame]].apply(lambda s: s.notna().all()).all()),
                                 "unique_declared_key": True}},
        "scope": {"indicator": "fossil-origin CO2", "price_basis": "CYP_(t-1) -> PYP_t",
                  "units": sorted(frame.units.unique().tolist())},
        "limitations": ["Accepted provisional because downstream reconciliation does not resolve upstream CYP/PYP validation failures."]
    }
    (output / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--panel-root",type=Path,required=True); p.add_argument("--panel-manifest",type=Path,required=True)
    p.add_argument("--group-metrics",type=Path,required=True); p.add_argument("--group-manifest",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    hierarchy=pd.read_parquet(a.panel_root/"global_hierarchy.parquet")
    hierarchy=hierarchy[(hierarchy.is_publishable)&(hierarchy.validation_status=="passed")].copy()
    hierarchy=hierarchy.rename(columns={"effect_id":"effect","contribution":"value"})
    rich=["year_previous","year_current","effect","resolution","units","value","positive_contribution","negative_contribution","gross_contribution","denominator_value","denominator_status","net_pct_of_parent_gross"]
    annual=hierarchy[rich].sort_values(["year_current","effect","resolution"])
    assert not annual.duplicated(["year_current","effect","resolution"]).any()
    write_view(a.output_dir,"annual_hierarchy",annual,[a.panel_manifest,a.panel_root/"global_hierarchy.parquet"],"All 12 hierarchy effects; contribution renamed value; signs and gross preserved.")
    primary={"delta_g","emission_intensity","leontief_production_structure","final_demand"}
    cross=annual[annual.effect.isin(primary)].copy(); assert len(cross)==168
    write_view(a.output_dir,"cross_resolution",cross,[a.panel_manifest,a.panel_root/"global_hierarchy.parquet"],"Matched total plus three primary effects only; resolutions remain separate.")
    metrics=pd.read_csv(a.group_metrics)
    rel=metrics[metrics.dimension_level.eq("relationship")].copy()
    rel["role"]=rel.effect.map({"D_C":"technical_coefficient_sourcing","D_YF":"final_demand_sourcing"})
    rel=rel.rename(columns={"net_mt":"value","positive_mt":"positive_contribution","negative_mt":"negative_contribution","gross_mt":"gross_contribution"})
    rel["units"]="Mt CO2"; rel=rel[["year_previous","year_current","effect","role","relationship","resolution","units","value","positive_contribution","negative_contribution","gross_contribution","cancellation_ratio"]]
    assert not rel.duplicated(["year_current","effect","role","relationship","resolution"]).any()
    write_view(a.output_dir,"international_domestic",rel,[a.group_manifest,a.group_metrics],"Relationship level only; Domestic/Cross-border contributions, not association results.")
    parts=[]
    for level,role,col in (("source_group","source","source_group"),("destination_group","destination","destination_group")):
        part=metrics[metrics.dimension_level.eq(level)].copy(); part["role"]=role; part["group"]=part[col]; parts.append(part)
    groups=pd.concat(parts,ignore_index=True); groups=groups[groups.group.isin(["Red","Non-Red","Non-Participant"])]
    groups=groups.rename(columns={"net_mt":"value","positive_mt":"positive_contribution","negative_mt":"negative_contribution","gross_mt":"gross_contribution"}); groups["units"]="Mt CO2"
    groups=groups[["year_previous","year_current","effect","role","group","resolution","units","value","positive_contribution","negative_contribution","gross_contribution","cancellation_ratio"]]
    assert not groups.duplicated(["year_current","effect","role","group","resolution"]).any()
    write_view(a.output_dir,"group_trends",groups,[a.group_manifest,a.group_metrics],"Absolute analytical groups only; source and destination roles remain distinct.")
    blocked={"schema_version":1,"analysis_id":"static-input-selected-cases-v1","status":"blocked","generated_at_utc":datetime.now(timezone.utc).isoformat(),
             "validation":{"passed":False,"thesis_status":"ineligible"},"blockers":["R3 observed sector codes 1-98 are absent from the current sector registry aliases.","Six fine-to-coarse sector relationships remain unresolved.","Case selector is not aligned with declared sign-stratified role-safe methodology."]}
    (a.output_dir/"selected_cases.manifest.json").write_text(json.dumps(blocked,indent=2)+"\n",encoding="utf-8")
    print(a.output_dir); return 0
if __name__=="__main__": raise SystemExit(main())
