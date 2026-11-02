#!/usr/bin/env python3
"""Build annual global-change reports for the native-WIOD CO2 campaign.

Task B of `docs/HANDOVER_CURRENT_STATE_20260902.md` §6. The stock
`build_annual_reports.py` consumes a canonical `sda.output` publication
(`manifest.json` + role-typed `.npy` arrays); the native campaign was executed
with the Vibranium flat `SDA/main.py` module and produced
`CO2_7/{year}_{effect}_{variant}.parquet` instead, so it cannot be fed to that
pipeline without an adapter (see
`docs/analysis/native_wiod_campaign_validation_20260903/TASK_B_BLOCKER.md`).

This script produces the same dashboard artefacts
(`dashboard_reporting_contract_v2.json`) directly from the native parquet
outputs: the 12-node `annual_hierarchy.parquet`, an emitter-grain
`annual_drilldown.parquet`, `annual_overview.md`, `annual_summary.v2.json` and
`manifest.v2.json`, one directory per transition, published atomically.

Publishability is gated on the post-run (task A) standard: aggregate hierarchy
reconstruction at relative 2e-5 / absolute 1 kg CO2 on the emitter totals. Where
a run fails an identity, the affected nodes are, by default, published as
`validation_status="accepted_provisional"` with the reconstruction residual
attached (2026-09-03 decision: the 2e-5 gate is tight and the residuals
physically reasonable); `--strict-withhold` instead publishes them with
`is_publishable=false`. Either way the run is not dropped. The canonical
elementwise `check_scaled_additivity` is recorded per run as a diagnostic.

Native effect -> canonical node:

    Total -> delta_g                 Dint -> emission_intensity / D_v
    Leontief_total -> leontief_production_structure
    D_C_coef -> D_C
    D_A_x := D_T_coef + D_w_coef      D_T_coef, D_w_coef -> same
    FD_total -> final_demand          YF -> D_YF   YB -> D_YB   DYy -> D_y

The coefficient-form Leontief split (`D_C_coef`, `D_T_coef`, `D_w_coef`) is used,
not the raw split (`D_C2`, `D_Ax2`): the two agree at `Leontief_total` but not
component-wise, and mixing raw `D_A_x` with its `D_T_coef`/`D_w_coef` split is
forbidden (handover section 9). `D_A_x` is therefore the sum of its children by
construction, so `leontief_children_sum_to_parent`
(`D_C + D_A_x == Leontief_total`) is exactly the post-run `leontief_split`
identity.

Drilldown coverage is emitter grain only (`emitter_country`, `emitter_sector`,
`emitter_country_group`). Driver / consumer / demanded-sector / source grains
need the native `Supplier`/`Destination` matrix axes to be mapped to canonical
roles first; that provenance is not in the campaign outputs and those grains are
deferred.

Usage::

    .venv/bin/python scripts/build_native_wiod_annual_reports.py \
      --campaign-root  /local/.../native-wiod-co2-2001-2009-20260902 \
      --validation-manifest /local/.../validation_20260903/validation_manifest.json \
      --output-root    /local/.../native-wiod-co2-2001-2009-annual-20260903 \
      --repo-src       /local/s2053268/ie-thesis/IE_Thesis_EXIO_WIOT   # optional, for sda import
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- canonical contract imports -------------------------------------------------
# The sda package lives under this repo's src/; allow an override for Vibranium.
_DEFAULT_SRC = Path(__file__).resolve().parents[1] / "src"
for _candidate in (_DEFAULT_SRC,):
    if (_candidate / "sda").is_dir():
        sys.path.insert(0, str(_candidate))
        break

try:
    from sda.output.annual import load_dashboard_registry
    from sda.validation.attribution import check_scaled_additivity
except ImportError as exc:  # pragma: no cover - surfaced to the operator
    raise SystemExit(
        "cannot import the sda package; pass --repo-src pointing at a checkout "
        f"that contains src/sda/ ({exc})"
    )

UNITS = "kg CO2"
INDICATOR = "CO2 only, kg CO2"
INDICATOR_BOUNDARY = "industry CO2 only; household direct emissions excluded"

# node -> native effect whose `_tot_baseline` emitter vector is read directly.
# "D_A_x" is derived (D_T_coef + D_w_coef) and is not in this map.
NODE_TO_NATIVE = {
    "delta_g": "Total",
    "emission_intensity": "Dint",
    "D_v": "Dint",
    "leontief_production_structure": "Leontief_total",
    "D_C": "D_C_coef",
    "D_T_coef": "D_T_coef",
    "D_w_coef": "D_w_coef",
    "final_demand": "FD_total",
    "D_YF": "YF",
    "D_YB": "YB",
    "D_y": "DYy",
}
# native effect used for the emitter-grain drilldown of each canonical node.
NODE_DRILLDOWN_NATIVE = {**NODE_TO_NATIVE, "D_A_x": "D_Ax2"}
DEPTH = {
    "delta_g": 0,
    "emission_intensity": 1, "leontief_production_structure": 1, "final_demand": 1,
    "D_v": 2, "D_C": 2, "D_A_x": 2, "D_YF": 2, "D_YB": 2, "D_y": 2,
    "D_T_coef": 3, "D_w_coef": 3,
}

CONFIG_META = {
    "jrc2019_to_2016": {
        "impact_database": "JRC2019", "flow_source": "WIOD2016", "target_grid": "WIOD2016 (44x50)",
        "harmonization_treatment": "none (native 2016 target grid)",
    },
    "jrc2019_to_2013": {
        "impact_database": "JRC2019", "flow_source": "WIOD2016", "target_grid": "WIOD2013 (35 sectors)",
        "harmonization_treatment": (
            "WIOD2016 56->50->35; non-nested source C13-C15 split 50/50 across "
            "target 17t18 and 19 by the reviewed crosswalk (harmonization assumption, "
            "not pure aggregation)"
        ),
    },
    "wiod2013_native": {
        "impact_database": "WIOD2013", "flow_source": "WIOD2013", "target_grid": "WIOD2013 (41x35)",
        "harmonization_treatment": "none (native 2013 grid)",
    },
    # mixed emission-account x flow-table cells (v2), 41x35 grid
    "jrc2019_flows2013": {
        "impact_database": "JRC2019", "flow_source": "WIOD2013", "target_grid": "WIOD2013 (41x35)",
        "harmonization_treatment": (
            "JRC emissions WIOD2016 56->50->35 (non-nested C13-C15 split 50/50); "
            "WIOD2013 native flows"
        ),
    },
    "wiod2013_flows2016": {
        "impact_database": "WIOD2013", "flow_source": "WIOD2016", "target_grid": "WIOD2013 (41x35)",
        "harmonization_treatment": (
            "WIOD2013 native emissions; WIOD2016 flows harmonised WIOD2016->2013 "
            "(non-nested C13-C15 split 50/50)"
        ),
    },
    # Phase 1 of the flow x emission matrix: genuine WIOD2013-PUBLISHED flow
    # tables (SDA_FLOW_TABLE_SOURCE=WIOD), 41x35 grid. The pre-existing cells
    # above all use EXIOBASE-DERIVED flows despite their labels -- corrected in
    # the Chapter 4 / register reconciliation pass.
    "wiod2013_true_native": {
        "impact_database": "WIOD2013", "flow_source": "WIOD2013 published", "target_grid": "WIOD2013 (41x35)",
        "harmonization_treatment": "none -- WIOD2013 native emissions and WIOD2013 published flows, fully self-consistent",
    },
    "jrc2019_wiodflows2013": {
        "impact_database": "JRC2019", "flow_source": "WIOD2013 published", "target_grid": "WIOD2013 (41x35)",
        "harmonization_treatment": "JRC emissions WIOD2016 56->50->35 (non-nested C13-C15 split 50/50); WIOD2013 published flows",
    },
    "exio_wiodflows2013": {
        "impact_database": "EXIOBASE", "flow_source": "WIOD2013 published", "target_grid": "WIOD2013 (41x35)",
        "harmonization_treatment": "EXIOBASE3 PXP CO2 aggregated to 41x35 via the EXIOBASE->WIOD concordance; WIOD2013 published flows",
    },
    # Phase 2: genuine WIOD2016-PUBLISHED flow tables (SDA_FLOW_TABLE_SOURCE=WIOD)
    # on the 44x50 grid, JRC2019 (WIOD2016 air) emissions. WIOD2016 native.
    "jrc2019_wiod2016flows_2016": {
        "impact_database": "JRC2019", "flow_source": "WIOD2016 published", "target_grid": "WIOD2016 (44x50)",
        "harmonization_treatment": "WIOD2016 published 56->50 by the reviewed weight-1 map; JRC2019 (WIOD2016 air) emissions",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def emitter_vector(effect_dir: Path, year: int, native_effect: str) -> np.ndarray:
    path = effect_dir / f"{year}_{native_effect}_tot_baseline.parquet"
    frame = pd.read_parquet(path)
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError(f"no numeric column in {path}")
    values = numeric.to_numpy(dtype=np.float64).ravel()
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values in {path}")
    return values


def matrix_marginal(effect_dir: Path, year: int, native_effect: str, variant: str) -> pd.Series:
    """Row (Supplier) marginal of a native bilateral matrix, indexed by label."""
    path = effect_dir / f"{year}_{native_effect}_{variant}_baseline.parquet"
    frame = pd.read_parquet(path)
    numeric = frame.select_dtypes(include="number")
    return numeric.sum(axis=1).astype(float)


def metric(values: np.ndarray, denominator: float | None) -> dict:
    array = np.asarray(values, dtype=np.float64).ravel()
    positive = float(array[array > 0].sum())
    negative = float(array[array < 0].sum())
    contribution = float(array.sum())
    gross = positive - negative
    if denominator is None:
        status, percentage = "not_applicable", None
    elif denominator == 0:
        status, percentage = "zero", None
    elif abs(denominator) <= np.finfo(np.float64).eps * 256:
        status, percentage = "near_zero", None
    else:
        status, percentage = "ok", 100.0 * contribution / denominator
    return {
        "contribution": contribution,
        "positive_contribution": positive,
        "negative_contribution": negative,
        "gross_contribution": gross,
        "denominator_value": denominator,
        "denominator_status": status,
        "net_pct_of_parent_gross": percentage,
    }


def gate_dict(report) -> dict:
    return {
        "passed": bool(report.passed),
        "max_abs_error": float(report.max_abs_error),
        "rtol": float(report.rtol),
        "atol": float(report.atol),
        "failing_index": list(report.failing_index) if report.failing_index else None,
    }


def build_run(run_record, campaign, validation_run, registry, output_root, source_files,
              *, accept_provisional=True):
    run_id = run_record["run_id"]
    parts = run_id.split("_")
    year_prev, year = int(parts[0]), int(parts[1])
    config_id = "_".join(parts[2:])
    effect_dir = Path(run_record["output"]) / "CO2_7"

    v = {node: emitter_vector(effect_dir, year, native) for node, native in NODE_TO_NATIVE.items()}
    v["D_A_x"] = v["D_T_coef"] + v["D_w_coef"]  # coefficient-consistent parent

    # Publishability is gated on the same standard as the post-run validation
    # (task A): aggregate hierarchy reconstruction at relative 2e-5 / absolute
    # 1 kg on the emitter totals, matching validate_native_wiod_smoke.py. The
    # canonical elementwise check_scaled_additivity is recorded alongside as a
    # diagnostic but is not the gate -- on float32 Vibranium arithmetic it is
    # materially stricter and would contradict the accepted task-A verdicts.
    HIER_REL, HIER_ABS = 2e-5, 1.0

    def agg_identity(children, parent):
        child_sum = float(sum(v[c].sum() for c in children))
        parent_sum = float(v[parent].sum())
        residual = child_sum - parent_sum
        scale = max(abs(child_sum), abs(parent_sum), 1.0)
        return {
            "children": list(children), "parent": parent,
            "child_sum_kg_co2": child_sum, "parent_kg_co2": parent_sum,
            "residual_kg_co2": residual, "relative_residual": abs(residual) / scale,
            "passed": abs(residual) <= HIER_ABS + HIER_REL * scale,
        }

    identities = {
        "primary_effects_sum_to_delta_g": agg_identity(
            ["D_v", "leontief_production_structure", "final_demand"], "delta_g"),
        "leontief_children_sum_to_parent": agg_identity(
            ["D_C", "D_A_x"], "leontief_production_structure"),
        "production_coefficient_children_sum_to_d_a_x": agg_identity(
            ["D_T_coef", "D_w_coef"], "D_A_x"),
        "final_demand_children_sum_to_parent": agg_identity(
            ["D_YF", "D_YB", "D_y"], "final_demand"),
    }
    elementwise = {
        "primary_effects_sum_to_delta_g": check_scaled_additivity(
            (v["D_v"], v["leontief_production_structure"], v["final_demand"]), v["delta_g"]),
        "leontief_children_sum_to_parent": check_scaled_additivity(
            (v["D_C"], v["D_A_x"]), v["leontief_production_structure"]),
        "production_coefficient_children_sum_to_d_a_x": check_scaled_additivity(
            (v["D_T_coef"], v["D_w_coef"]), v["D_A_x"]),
        "final_demand_children_sum_to_parent": check_scaled_additivity(
            (v["D_YF"], v["D_YB"], v["D_y"]), v["final_demand"]),
    }
    # node -> the failed identity that governs it (for provisional annotation)
    failing_identity_for: dict[str, str] = {}
    affected: set[str] = set()
    if not identities["primary_effects_sum_to_delta_g"]["passed"]:
        for n in set(DEPTH) - {"delta_g"}:
            affected.add(n); failing_identity_for[n] = "primary_effects_sum_to_delta_g"
    if not identities["leontief_children_sum_to_parent"]["passed"]:
        for n in ("D_C", "D_A_x", "D_T_coef", "D_w_coef"):
            affected.add(n); failing_identity_for[n] = "leontief_children_sum_to_parent"
    if not identities["production_coefficient_children_sum_to_d_a_x"]["passed"]:
        for n in ("D_T_coef", "D_w_coef"):
            affected.add(n); failing_identity_for[n] = "production_coefficient_children_sum_to_d_a_x"
    if not identities["final_demand_children_sum_to_parent"]["passed"]:
        for n in ("D_YF", "D_YB", "D_y"):
            affected.add(n); failing_identity_for[n] = "final_demand_children_sum_to_parent"

    # `accept_provisional` (default): affected nodes are published as
    # `accepted_provisional` with the reconstruction residual attached.
    # Otherwise they are withheld (`is_publishable=false`).
    withheld = set() if accept_provisional else set(affected)
    provisional = set(affected) if accept_provisional else set()

    parent_of = {spec["node_id"]: spec["parent_node_id"] for spec in registry["nodes"]}
    context = {
        "run_id": run_id, "year_previous": year_prev, "year_current": year,
        "resolution": config_id, "units": UNITS,
    }

    gross = {node: metric(vec, None)["gross_contribution"] for node, vec in v.items()}
    hierarchy_rows = []
    for spec in registry["nodes"]:
        node = spec["node_id"]
        parent = parent_of[node]
        if node in withheld:
            status = "failed"
        elif node in provisional:
            status = "accepted_provisional"
        else:
            status = "passed"
        row = {
            **context, "node_id": node, "parent_node_id": parent, "depth": DEPTH[node],
            **metric(v[node], gross[parent] if parent else None),
            "is_publishable": node not in withheld,
            "validation_status": status,
        }
        if node in failing_identity_for:
            ident = identities[failing_identity_for[node]]
            row["reconstruction_identity"] = failing_identity_for[node]
            row["reconstruction_residual_kg_co2"] = ident["residual_kg_co2"]
            row["reconstruction_relative_residual"] = ident["relative_residual"]
        hierarchy_rows.append(row)
    hierarchy_rows.sort(key=lambda r: (r["depth"], r["node_id"]))

    # --- emitter-grain drilldown -------------------------------------------------
    role_columns = registry["products"]["annual_drilldown"]["columns"]
    grain_sources = {
        "emitter_country": "country_matrix",
        "emitter_sector": "sector_matrix",
        "emitter_country_group": "group_matrix",
    }
    drilldown_rows = []
    for spec in registry["nodes"]:
        node = spec["node_id"]
        if node not in registry["effects"] or node in withheld:
            continue  # only the 8 leaf/effect nodes carry drilldowns
        # D_A_x is D_T_coef + D_w_coef; its marginal is the sum of theirs.
        natives = ["D_T_coef", "D_w_coef"] if node == "D_A_x" else [NODE_DRILLDOWN_NATIVE[node]]
        parent_gross = metric(v[node], None)["gross_contribution"]
        allowed = set(registry["effects"][node]["allowed_grains"])
        for grain, variant in grain_sources.items():
            if grain not in allowed:
                continue
            marginal = matrix_marginal(effect_dir, year, natives[0], variant)
            for nat in natives[1:]:
                marginal = marginal.add(matrix_marginal(effect_dir, year, nat, variant), fill_value=0.0)
            if not np.isclose(float(marginal.sum()), float(v[node].sum()), rtol=1e-4,
                              atol=256 * 1.1920929e-7 * max(1.0, float(np.abs(v[node]).max()))):
                raise ValueError(f"{run_id} {node}/{grain} marginal does not reconcile to the emitter total")
            for label, value in marginal.items():
                roles = {col: None for col in role_columns if col not in (
                    "run_id", "year_previous", "year_current", "resolution", "units",
                    "effect_id", "grain_id", "contribution", "positive_contribution",
                    "negative_contribution", "gross_contribution", "denominator_value",
                    "denominator_status", "net_pct_of_parent_gross", "is_publishable",
                    "validation_status")}
                roles[grain] = str(label)
                drilldown_rows.append({
                    **context, "effect_id": node, "grain_id": grain, **roles,
                    **metric(np.array([value]), parent_gross),
                    "is_publishable": True,
                    "validation_status": "accepted_provisional" if node in provisional else "passed",
                })
    drilldown_rows.sort(key=lambda r: (r["effect_id"], r["grain_id"], r["contribution"]))

    if all(i["passed"] for i in identities.values()):
        run_status = "passed"
    elif accept_provisional:
        run_status = "accepted_provisional"
    else:
        run_status = "partial"

    # --- publish atomically ----------------------------------------------------
    destination = output_root / f"{run_id}"
    if destination.exists():
        raise FileExistsError(destination)
    staging = output_root / f".staging-{run_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        _write_parquet(hierarchy_rows, staging / "annual_hierarchy.parquet",
                       registry["products"]["annual_hierarchy"]["columns"])
        _write_parquet(drilldown_rows, staging / "annual_drilldown.parquet", role_columns)

        lines = ["# Annual SDA accounting attribution (native WIOD CO2)", "",
                 f"Transition {year_prev}->{year}, configuration `{config_id}`, units {UNITS}.", "",
                 f"Run validation status: **{run_status}**.", "",
                 "| Node | Contribution | Positive | Negative | Gross | % parent gross | Denom | Node status |",
                 "|---|---:|---:|---:|---:|---:|---|---|"]
        for row in hierarchy_rows:
            pct = "n/a" if row["net_pct_of_parent_gross"] is None else f'{row["net_pct_of_parent_gross"]:.6g}'
            lines.append(
                f'| {row["node_id"]} | {row["contribution"]:.8g} | {row["positive_contribution"]:.8g} | '
                f'{row["negative_contribution"]:.8g} | {row["gross_contribution"]:.8g} | {pct} | '
                f'{row["denominator_status"]} | {row["validation_status"]} |')
        if withheld:
            lines += ["", f"Withheld nodes (failed identity gate, not published): "
                      f"{', '.join(sorted(withheld))}."]
        if provisional:
            lines += ["", "Accepted-provisional nodes (published; reconstruction residual "
                      "exceeds the 2e-5 gate but accepted per the 2026-09-03 decision):"]
            for n in sorted(provisional):
                ident = identities[failing_identity_for[n]]
                lines.append(f"- `{n}`: identity `{failing_identity_for[n]}` residual "
                             f"{ident['residual_kg_co2']:+.4g} kg CO2 "
                             f"({ident['relative_residual']:.3e} relative)")
        (staging / "annual_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        summary = {
            "report_schema_version": 2, **context,
            "validation_status": run_status,
            "hierarchy": hierarchy_rows,
            "row_counts": {"annual_hierarchy": len(hierarchy_rows), "annual_drilldown": len(drilldown_rows)},
        }
        (staging / "annual_summary.v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        artefacts = ["annual_hierarchy.parquet", "annual_drilldown.parquet",
                     "annual_overview.md", "annual_summary.v2.json"]
        manifest = {
            "report_schema_version": 2, **context,
            "year_pair": [year_prev, year],
            "registry": "dashboard_reporting_contract_v2.json",
            "roles": ["emitter"],
            "drilldown_grain_coverage": sorted(grain_sources),
            "deferred_drilldown_grains": [
                "driver_*", "consumer_*", "demanded_sector_*", "source_*",
                "emitter_sector_group",
            ],
            "native_campaign": {
                "campaign_id": campaign.get("campaign_id"),
                "configuration_id": config_id,
                **CONFIG_META[config_id],
                "indicator": INDICATOR,
                "indicator_boundary": INDICATOR_BOUNDARY,
                "like_for_like_with_main_gwp100_campaign": False,
                "comparability_note": campaign.get("comparability", {}).get(
                    "wiod2016_to_2013_non_nested_assumption"),
            },
            "source_files": source_files,
            "validation": {
                "gate": "aggregate hierarchy reconstruction, relative 2e-5 / absolute 1 kg CO2",
                "identities": identities,
                "elementwise_reconciliation_diagnostic": {
                    name: gate_dict(rep) for name, rep in elementwise.items()
                },
            },
            "validation_status": run_status,
            "acceptance_decision": (
                "2026-09-03: the 2e-5 aggregate hierarchy tolerance is regarded as "
                "extremely tight and the residuals as physically reasonable; "
                "identity-failing nodes are published as accepted_provisional "
                "rather than withheld."
            ) if provisional else None,
            "withheld_nodes": sorted(withheld),
            "provisional_nodes": sorted(provisional),
            "provisional_node_residuals": {
                n: {"identity": failing_identity_for[n],
                    "residual_kg_co2": identities[failing_identity_for[n]]["residual_kg_co2"],
                    "relative_residual": identities[failing_identity_for[n]]["relative_residual"]}
                for n in sorted(provisional)
            },
            "coverage": {"hierarchy_nodes_total": len(registry["nodes"]),
                         "hierarchy_nodes_published": len(registry["nodes"]) - len(withheld),
                         "hierarchy_nodes_accepted_provisional": len(provisional)},
            "upstream": {
                "postrun_validation_run_passed": bool(validation_run and validation_run.get("passed")),
                "postrun_validation_advisory_passed": bool(validation_run and validation_run.get("advisory_passed")),
            },
            "row_counts": summary["row_counts"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {name: {"bytes": (staging / name).stat().st_size,
                                 "sha256": sha256(staging / name)} for name in artefacts},
        }
        (staging / "manifest.v2.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"run_id": run_id, "validation_status": run_status,
            "withheld_nodes": sorted(withheld), "provisional_nodes": sorted(provisional),
            "identity_gates": {name: i["passed"] for name, i in identities.items()},
            "worst_identity_relative_residual": max(i["relative_residual"] for i in identities.values()),
            "directory": str(destination)}


def _write_parquet(rows, path: Path, columns) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist([{col: row.get(col) for col in columns} for row in rows])
    pq.write_table(table, path, row_group_size=50_000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path,
                        help="checkout containing src/sda/ (default: this repo)")
    parser.add_argument("--strict-withhold", action="store_true",
                        help="withhold identity-failing nodes instead of publishing "
                             "them as accepted_provisional (default: accept provisional)")
    args = parser.parse_args()
    accept_provisional = not args.strict_withhold

    campaign_manifest_path = args.campaign_root / "campaign_manifest.json"
    campaign = json.loads(campaign_manifest_path.read_text())
    validation = json.loads(args.validation_manifest.read_text())
    validation_by_run = {r["run_id"]: r for r in validation.get("runs", [])}
    registry = load_dashboard_registry()

    source_files = [source_record(campaign_manifest_path), source_record(args.validation_manifest)]

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for run_record in campaign["runs"]:
        print(f"[annual] {run_record['run_id']}", flush=True)
        results.append(build_run(run_record, campaign, validation_by_run.get(run_record["run_id"]),
                                 registry, args.output_root, source_files,
                                 accept_provisional=accept_provisional))

    index = {
        "schema_version": 1,
        "index_id": "native-wiod-co2-2001-2009-annual-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign.get("campaign_id"),
        "campaign_manifest_sha256": sha256(campaign_manifest_path),
        "postrun_validation_manifest_sha256": sha256(args.validation_manifest),
        "units": UNITS,
        "indicator": INDICATOR,
        "acceptance_mode": "accept_provisional" if accept_provisional else "strict_withhold",
        "reports": results,
        "summary": {
            "reports_total": len(results),
            "fully_passed": sum(r["validation_status"] == "passed" for r in results),
            "accepted_provisional": sum(r["validation_status"] == "accepted_provisional" for r in results),
            "partial": sum(r["validation_status"] == "partial" for r in results),
            "non_passed_run_ids": [r["run_id"] for r in results if r["validation_status"] != "passed"],
        },
    }
    (args.output_root / "campaign_annual_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    s = index["summary"]
    cov = ["# Native-WIOD annual reports — coverage", "",
           f"Generated {index['generated_at_utc']}. Units {UNITS}. Indicator: {INDICATOR}.",
           f"Acceptance mode: {index['acceptance_mode']}.", "",
           f"{s['fully_passed']}/{len(results)} fully passed; "
           f"{s['accepted_provisional']} accepted-provisional; {s['partial']} partial.", "",
           "| run | status | provisional / withheld nodes |", "|---|---|---|"]
    for r in results:
        nodes = r["provisional_nodes"] or r["withheld_nodes"]
        cov.append(f"| `{r['run_id']}` | {r['validation_status']} | "
                   f"{', '.join(nodes) or '—'} |")
    cov += ["",
            "Accepted-provisional runs publish all 12 nodes; the Leontief-split "
            "nodes (`D_C`, `D_A_x`, `D_T_coef`, `D_w_coef`) carry "
            "`validation_status=accepted_provisional` and their reconstruction "
            "residual in `manifest.v2.json` / `annual_summary.v2.json` "
            "(2026-09-03 decision: the 2e-5 gate is tight, the residuals "
            "physically reasonable).", "",
            "Drilldown grains published: emitter_country, emitter_sector, "
            "emitter_country_group. Driver / consumer / demanded-sector / source "
            "grains are deferred pending a Supplier/Destination -> canonical-role "
            "mapping for the native matrices."]
    (args.output_root / "COVERAGE.md").write_text("\n".join(cov) + "\n", encoding="utf-8")

    print(json.dumps(index["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
