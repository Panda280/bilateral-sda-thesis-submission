"""Registry-driven, atomic annual and cross-year dashboard reporting."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..hierarchy import TOP_LEVEL_EFFECTS
from ..model import EffectResult, SDAResult
from ..validation.attribution import check_scaled_additivity

ANNUAL_REPORT_SCHEMA_VERSION = 2
REGISTRY_PATH = Path(__file__).with_name("dashboard_reporting_contract_v2.json")
ROLE_COLUMNS = (
    "emitter_country", "emitter_sector", "emitter_country_group", "emitter_sector_group",
    "driver_country", "driver_sector", "driver_country_group", "driver_sector_group",
    "consumer_country", "consumer_country_group", "demanded_sector", "demanded_sector_group",
    "source_country", "source_country_group", "source_group", "source_sector", "source_sector_group",
)


def load_dashboard_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("contract_version") != ANNUAL_REPORT_SCHEMA_VERSION:
        raise ValueError("dashboard registry contract_version mismatch")
    if set(registry.get("products", {})) != {"annual_hierarchy", "annual_drilldown", "cross_year_panel"}:
        raise ValueError("dashboard registry product inventory mismatch")
    if len(registry.get("panels", [])) != 11:
        raise ValueError("dashboard registry must define exactly 11 panels")
    return registry


def _pair(label: Any, role: str) -> tuple[str, str]:
    if isinstance(label, tuple) and len(label) == 2:
        return str(label[0]), str(label[1])
    if isinstance(label, str) and label.count(":") == 1:
        left, right = label.split(":")
        if left and right:
            return left, right
    raise ValueError(f"{role} labels must be explicit country-sector pairs")


def _metric(values: Any, denominator: float | None) -> dict[str, Any]:
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
    return {"contribution": contribution, "positive_contribution": positive,
            "negative_contribution": negative, "gross_contribution": gross,
            "denominator_value": denominator, "denominator_status": status,
            "net_pct_of_parent_gross": percentage}


def _emitter_vector(effect: EffectResult) -> np.ndarray:
    values = np.asarray(effect.baseline, dtype=np.float64)
    return values if effect.dimensions.column is None else values.sum(axis=1)


def _hierarchy(result: SDAResult, context: Mapping[str, Any], registry: Mapping[str, Any]):
    vectors = {name: _emitter_vector(effect) for name, effect in result.effects.items()}
    delta = np.asarray(result.total_change, dtype=np.float64)
    gates = {
        "top_level_effects_sum_to_delta_g": check_scaled_additivity((vectors[n] for n in TOP_LEVEL_EFFECTS), delta),
        "leontief_children_sum_to_parent": check_scaled_additivity((vectors["D_C"], vectors["D_A_x"]), vectors["D_C"] + vectors["D_A_x"]),
        "production_coefficient_children_sum_to_d_a_x": check_scaled_additivity((vectors["D_T_coef"], vectors["D_w_coef"]), vectors["D_A_x"]),
        "final_demand_children_sum_to_parent": check_scaled_additivity((vectors["D_YF"], vectors["D_YB"], vectors["D_y"]), vectors["D_YF"] + vectors["D_YB"] + vectors["D_y"]),
    }
    failed = [name for name, report in gates.items() if not report.passed]
    if failed:
        raise ValueError(f"annual hierarchy validation failed: {failed}")
    values = {"delta_g": delta, "emission_intensity": vectors["D_v"], "D_v": vectors["D_v"],
              "leontief_production_structure": vectors["D_C"] + vectors["D_A_x"],
              "D_C": vectors["D_C"], "D_A_x": vectors["D_A_x"], "D_T_coef": vectors["D_T_coef"],
              "D_w_coef": vectors["D_w_coef"], "final_demand": vectors["D_YF"] + vectors["D_YB"] + vectors["D_y"],
              "D_YF": vectors["D_YF"], "D_YB": vectors["D_YB"], "D_y": vectors["D_y"]}
    depth = {"delta_g": 0, "emission_intensity": 1, "leontief_production_structure": 1, "final_demand": 1,
             "D_v": 2, "D_C": 2, "D_A_x": 2, "D_YF": 2, "D_YB": 2, "D_y": 2,
             "D_T_coef": 3, "D_w_coef": 3}
    gross = {key: _metric(value, None)["gross_contribution"] for key, value in values.items()}
    rows = []
    for spec in registry["nodes"]:
        if not spec["is_publishable"]:
            continue
        node, parent = spec["node_id"], spec["parent_node_id"]
        rows.append({**context, "node_id": node, "parent_node_id": parent, "depth": depth[node],
                     **_metric(values[node], gross[parent] if parent else None),
                     "is_publishable": True, "validation_status": "passed"})
    rows.sort(key=lambda row: (row["depth"], row["node_id"]))
    return rows, {name: {"passed": report.passed, "max_abs_error": report.max_abs_error,
                         "rtol": report.rtol, "atol": report.atol} for name, report in gates.items()}


def _axis_specs(effect: EffectResult, country_groups: Mapping[str, str], sector_groups: Mapping[str, str]):
    emitters = [_pair(label, "emitter") for label in effect.dimensions.row.labels]
    specs = [("emitter_country", [c for c, _ in emitters], 0), ("emitter_sector", [s for _, s in emitters], 0),
             ("emitter_country_group", [country_groups.get(c, c) for c, _ in emitters], 0),
             ("emitter_sector_group", [sector_groups.get(s, s) for _, s in emitters], 0)]
    if effect.dimensions.column is None:
        return specs
    role, labels = effect.dimensions.column.role.lower().replace(" ", "_").replace("-", "_"), list(effect.dimensions.column.labels)
    if "structural_driver" in role:
        pairs = [_pair(label, "structural driver") for label in labels]
        specs += [("driver_country", [c for c, _ in pairs], 1), ("driver_sector", [s for _, s in pairs], 1),
                  ("driver_country_group", [country_groups.get(c, c) for c, _ in pairs], 1),
                  ("driver_sector_group", [sector_groups.get(s, s) for _, s in pairs], 1)]
    elif "consumer" in role and "demanded" in role:
        pairs = [_pair(label, "consumer/demanded sector") for label in labels]
        specs += [("consumer_country", [c for c, _ in pairs], 1),
                  ("consumer_country_group", [country_groups.get(c, c) for c, _ in pairs], 1),
                  ("demanded_sector", [s for _, s in pairs], 1),
                  ("demanded_sector_group", [sector_groups.get(s, s) for _, s in pairs], 1)]
    elif "consumer" in role:
        specs += [("consumer_country", [str(x) for x in labels], 1),
                  ("consumer_country_group", [country_groups.get(str(x), str(x)) for x in labels], 1)]
    return specs


def _drilldowns(effect: EffectResult, context: Mapping[str, Any], registry: Mapping[str, Any],
                country_groups: Mapping[str, str], sector_groups: Mapping[str, str]):
    values = np.asarray(effect.baseline, dtype=np.float64); parent_gross = _metric(values, None)["gross_contribution"]
    allowed = set(registry["effects"][effect.name]["allowed_grains"])
    for grain, labels, dimension in _axis_specs(effect, country_groups, sector_groups):
        if grain not in allowed:
            raise ValueError(f"registry disallows generated grain {effect.name}/{grain}")
        reconstructed = 0.0
        for label in dict.fromkeys(labels):
            indices = [i for i, item in enumerate(labels) if item == label]
            selected = values[indices] if values.ndim == 1 or dimension == 0 else values[:, indices]
            reconstructed += float(selected.sum()); roles = {name: None for name in ROLE_COLUMNS}; roles[grain] = label
            yield {**context, "effect_id": effect.name, "grain_id": grain, **roles,
                   **_metric(selected, parent_gross), "is_publishable": True, "validation_status": "passed"}
        if not np.isclose(reconstructed, values.sum()):
            raise ValueError(f"{effect.name}/{grain} does not reconcile")
    if effect.source_categories:
        if "source_group" not in allowed:
            raise ValueError(f"registry disallows source groups for {effect.name}")
        reconstructed = 0.0
        for label, child in effect.source_categories.items():
            selected = np.asarray(child.baseline); reconstructed += float(selected.sum())
            roles = {name: None for name in ROLE_COLUMNS}; roles["source_group"] = label
            yield {**context, "effect_id": effect.name, "grain_id": "source_group", **roles,
                   **_metric(selected, parent_gross), "is_publishable": True, "validation_status": "passed"}
        if not np.isclose(reconstructed, values.sum()):
            raise ValueError(f"{effect.name}/source_group does not reconcile")


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("annual reports require the parquet optional extra") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([{column: row.get(column) for column in columns} for row in rows])
    pq.write_table(table, path, row_group_size=50_000)


def _checksum(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def publish_annual_report(result: SDAResult, output_root: str | Path, *, run_id: str, year_prev: int,
                          year: int, resolution: str, units: str, source_files: Sequence[Mapping[str, Any]],
                          country_groups: Mapping[str, str] | None = None,
                          sector_groups: Mapping[str, str] | None = None,
                          supplementary_drilldown: Sequence[Mapping[str, Any]] = ()) -> Path:
    result.validation.raise_if_invalid(); registry = load_dashboard_registry()
    if year != year_prev + 1 or not units or not resolution or not source_files:
        raise ValueError("annual metadata requires adjacent years, units, resolution, and source files")
    context = {"run_id": run_id, "year_previous": year_prev, "year_current": year, "resolution": resolution, "units": units}
    hierarchy, gates = _hierarchy(result, context, registry)
    drilldown = [row for effect in result.effects.values() for row in _drilldowns(effect, context, registry, country_groups or {}, sector_groups or {})]
    drilldown.extend({**context, **row} for row in supplementary_drilldown if row.get("is_publishable") is True)
    if any(not row["is_publishable"] for row in hierarchy + drilldown):
        raise ValueError("non-publishable row reached publication")
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True); destination = root / run_id
    if destination.exists(): raise FileExistsError(destination)
    staging = root / f".staging-{run_id}-{uuid.uuid4().hex}"; staging.mkdir()
    try:
        products = registry["products"]
        _write_parquet(hierarchy, staging / "annual_hierarchy.parquet", products["annual_hierarchy"]["columns"])
        _write_parquet(drilldown, staging / "annual_drilldown.parquet", products["annual_drilldown"]["columns"])
        lines = ["# Annual SDA accounting attribution", "", "| Node | Contribution | Positive | Negative | Gross | % parent gross | Status |", "|---|---:|---:|---:|---:|---:|---|"]
        for row in hierarchy:
            pct = "n/a" if row["net_pct_of_parent_gross"] is None else f'{row["net_pct_of_parent_gross"]:.6g}'
            lines.append(f'| {row["node_id"]} | {row["contribution"]:.8g} | {row["positive_contribution"]:.8g} | {row["negative_contribution"]:.8g} | {row["gross_contribution"]:.8g} | {pct} | {row["denominator_status"]} |')
        (staging / "annual_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary = {"report_schema_version": 2, **context, "validation_status": "passed", "hierarchy": hierarchy,
                   "row_counts": {"annual_hierarchy": len(hierarchy), "annual_drilldown": len(drilldown)}}
        (staging / "annual_summary.v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        names = ["annual_hierarchy.parquet", "annual_drilldown.parquet", "annual_overview.md", "annual_summary.v2.json"]
        manifest = {"report_schema_version": 2, **context, "year_pair": [year_prev, year], "registry": REGISTRY_PATH.name,
                    "roles": ["emitter", "structural_driver", "consumer", "demanded_sector", "source"],
                    "source_files": list(source_files), "validation_status": "passed", "validation": gates,
                    "row_counts": summary["row_counts"], "generated_at": datetime.now(timezone.utc).isoformat(),
                    "artifacts": {name: _checksum(staging / name) for name in names}}
        (staging / "manifest.v2.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    return destination


def publish_panel(annual_directories: Sequence[str | Path], output_root: str | Path, *, run_id: str) -> Path:
    import pyarrow.parquet as pq
    registry = load_dashboard_registry(); hierarchy, drilldown, inventory = [], [], set()
    for directory in annual_directories:
        root = Path(directory); manifest = json.loads((root / "manifest.v2.json").read_text(encoding="utf-8"))
        if manifest.get("validation_status") != "passed": raise ValueError(f"invalid annual run: {root}")
        key = (*manifest["year_pair"], manifest["resolution"])
        if key in inventory: raise ValueError(f"duplicate annual transition/resolution: {key}")
        inventory.add(key); hierarchy.extend(pq.read_table(root / "annual_hierarchy.parquet").to_pylist())
        drilldown.extend(pq.read_table(root / "annual_drilldown.parquet").to_pylist())
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True); destination = root / run_id
    if destination.exists(): raise FileExistsError(destination)
    staging = root / f".staging-{run_id}-{uuid.uuid4().hex}"; staging.mkdir()
    try:
        product, artifacts, total_rows = registry["products"]["cross_year_panel"], {}, 0
        for spec in registry["panels"]:
            if spec["source"] == "annual_hierarchy":
                selected = [{**row, "panel_id": spec["panel_id"], "effect_id": row["node_id"], "grain_id": "node_id", "label": row["node_id"]} for row in hierarchy]
            else:
                selected = [{**row, "panel_id": spec["panel_id"], "label": row.get(spec["grain_id"])} for row in drilldown if row["grain_id"] == spec["grain_id"]]
            selected = [row for row in selected if row["is_publishable"] is True]
            selected.sort(key=lambda row: (row["resolution"], row["year_current"], row["effect_id"], str(row["label"])))
            path = staging / "cross_year_panel" / f'{spec["panel_id"]}.parquet'; _write_parquet(selected, path, product["columns"])
            artifacts[str(path.relative_to(staging))] = _checksum(path); total_rows += len(selected)
        manifest = {"report_schema_version": 2, "run_id": run_id, "validation_status": "passed",
                    "annual_runs": len(annual_directories), "panel_count": 11, "rows": total_rows,
                    "inventory": [list(item) for item in sorted(inventory)], "partition_keys": product["partition_keys"],
                    "artifacts": artifacts}
        (staging / "panel_summary.v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    return destination
