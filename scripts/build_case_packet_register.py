"""Build evidence-bounded narrative case packets from the integrated register."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.selected.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"integrated_result_id", "parent_result_id", "effect", "attribution_role",
                "resolution", "selection_sign", "entity_type", "selection_scope",
                "canonical_entity", "units", "signed_total", "integrated_score"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    output = []
    for row in rows:
        claim_class = "descriptive attribution candidate"
        limitation = (
            "Provisional fossil-origin CO2; screening result, not causal evidence. "
            "Preserve effect, attribution role, sign, resolution and selection scope."
        )
        if row["effect"] in {"D_T_coef", "D_w_coef", "D_A_x"}:
            limitation += " T/w labels have mathematical, not validated wage/value-added or physical-input, meaning."
        output.append({
            "packet_id": "PACKET-" + hashlib.sha256(row["integrated_result_id"].encode()).hexdigest()[:16].upper(),
            "integrated_result_id": row["integrated_result_id"],
            "parent_result_id": row["parent_result_id"],
            "registered_results": "R-2026-009;R-2026-015",
            "effect": row["effect"],
            "attribution_role": row["attribution_role"],
            "resolution": row["resolution"],
            "selection_sign": row["selection_sign"],
            "entity_type": row["entity_type"],
            "selection_scope": row["selection_scope"],
            "canonical_entity": row["canonical_entity"],
            "units": row["units"],
            "signed_total": row["signed_total"],
            "integrated_score": row["integrated_score"],
            "selection_rationale": "selected within exact semantic stratum by the registered integrated screening score",
            "claim_class": claim_class,
            "interpretation_limit": limitation,
            "numerical_evidence": str(args.selected).replace("\\", "/"),
            "pipeline": "implementation/bilateral_sda_attribution/scripts/build_integrated_drilldown_cases.py",
            "upstream_manifest": "implementation/bilateral_sda_attribution/docs/analysis/integrated_case_selection_20260831/analysis_manifest.json",
            "narrative_status": "candidate_not_interpreted",
            "external_evidence_status": "not_assessed",
        })

    register = args.output_dir / "case_packet_register.csv"
    with register.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    manifest = {
        "schema_version": 1,
        "analysis_id": "narrative-case-packet-register-v1",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"indicator": "fossil-origin CO2", "price_basis": "CYP_(t-1) -> PYP_t", "provisional": True},
        "input": {"path": str(args.selected).replace("\\", "/"), "sha256": digest(args.selected)},
        "output": {"path": str(register).replace("\\", "/"), "sha256": digest(register), "rows": len(output)},
        "gates": {"packet_ids_unique": len({x["packet_id"] for x in output}) == len(output),
                  "result_ids_unique": len({x["integrated_result_id"] for x in output}) == len(output),
                  "units_preserved": all(x["units"] == "kg CO2" for x in output)},
        "interpretation_policy": "descriptive candidates only; external mechanisms require separately registered evidence",
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not all(manifest["gates"].values()):
        raise RuntimeError(f"validation failure: {manifest['gates']}")


if __name__ == "__main__":
    main()
