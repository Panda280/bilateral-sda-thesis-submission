#!/usr/bin/env python
"""Build one validated annual report or a cross-year panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sda.output import load_result, publish_annual_report, publish_panel  # noqa: E402
from sda.attribution.structural import attribute_structural_driver  # noqa: E402
from sda.effects.structural import nested_technical_components  # noqa: E402
from sda.validation.attribution import check_scaled_additivity  # noqa: E402


def _metrics(values, denominator):
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    positive = float(array[array > 0].sum()); negative = float(array[array < 0].sum())
    net = float(array.sum()); gross = positive - negative
    status = "zero" if denominator == 0 else ("near_zero" if abs(denominator) <= np.finfo(np.float64).eps * 256 else "ok")
    return {"contribution": net, "positive_contribution": positive, "negative_contribution": negative,
            "gross_contribution": gross, "denominator_value": denominator, "denominator_status": status,
            "net_pct_of_parent_gross": None if status != "ok" else 100.0 * net / denominator}


def _supplemental_row(effect, grain, label, values, denominator, **roles):
    row = {"effect_id": effect, "grain_id": grain, "is_publishable": True, "validation_status": "passed", **_metrics(values, denominator)}
    row.update(roles); row[grain] = label
    return row


def _validate_sourcing_reconciliation(emitter_totals, parent, *, effect, grain):
    """Reconcile a long sourcing product at the canonical emitter grain.

    Comparing only the global net totals is numerically unstable when large
    positive and negative emitter contributions cancel.  The sourcing cubes
    retain the emitter roles, so validate their emitter vector against the
    corresponding canonical emitter vector using the same scale-aware policy
    as the canonical attribution gates.
    """
    import numpy as np

    expected = np.asarray(parent, dtype=np.float64)
    expected = expected if expected.ndim == 1 else expected.sum(axis=1)
    reconstructed = np.asarray(emitter_totals, dtype=np.float64)
    report = check_scaled_additivity((reconstructed,), expected)
    if not report.passed:
        raise ValueError(
            f"{effect}/{grain} sourcing product does not reconcile at emitter grain "
            f"(max_abs_error={report.max_abs_error:.12g}, "
            f"failing_index={report.failing_index}, rtol={report.rtol:.12g}, "
            f"atol={report.atol:.12g})"
        )


def _sourcing_rows(directory: Path | None, result, country_groups):
    if directory is None: return []
    import numpy as np
    import pyarrow.parquet as pq
    products = [
        ("D_C", directory / "source_country" / "D_C.source_country_cube.parquet", ("source_country",)),
        ("D_C", directory / "source_group_sector" / "D_C.source_group_sector_cube.parquet", ("source_group", "source_sector")),
        ("D_YF", directory / "yf_source_country" / "D_YF.source_country_cube.parquet", ("source_country",)),
        ("D_YF", directory / "yf_source_group_sector" / "D_YF.source_group_sector_cube.parquet", ("source_group", "source_sector")),
    ]
    rows = []
    for effect, path, grains in products:
        if not path.is_file(): continue
        parent = result.effects[effect].baseline
        denominator = _metrics(parent, 1)["gross_contribution"]
        accumulators = {grain: defaultdict(list) for grain in grains}
        emitter_labels = [tuple(map(str, label)) if isinstance(label, tuple) else tuple(str(label).split(":", 1))
                          for label in result.effects[effect].dimensions.row.labels]
        emitter_index = {label: index for index, label in enumerate(emitter_labels)}
        emitter_totals = np.zeros(len(emitter_labels), dtype=np.float64)
        grain_columns = {"source_country": "Source_Country", "source_group": "Source_Group", "source_sector": "Source_Sector"}
        columns = ["Emitter_Country", "Emitter_Sector", *(grain_columns[g] for g in grains), "Value"]
        for batch in pq.ParquetFile(path).iter_batches(columns=columns):
            data = batch.to_pydict(); values = data["Value"]
            for country, sector, value in zip(data["Emitter_Country"], data["Emitter_Sector"], values, strict=True):
                label = (str(country), str(sector))
                if label not in emitter_index:
                    raise ValueError(f"{effect} sourcing product contains unknown emitter {label!r}")
                emitter_totals[emitter_index[label]] += float(value)
            for grain in grains:
                column = grain_columns[grain]
                for label, value in zip(data[column], values, strict=True): accumulators[grain][str(label)].append(float(value))
        for grain, grouped in accumulators.items():
            _validate_sourcing_reconciliation(emitter_totals, parent, effect=effect, grain=grain)
            for label, values in grouped.items():
                rows.append(_supplemental_row(effect, grain, label, values, denominator))
                if grain == "source_country":
                    group = country_groups.get(label, label)
                    # country-group rows are rebuilt below to avoid duplicated totals.
        if "source_country" in grains:
            grouped = defaultdict(list)
            for label, values in accumulators["source_country"].items(): grouped[country_groups.get(label, label)].extend(values)
            for label, values in grouped.items(): rows.append(_supplemental_row(effect, "source_country_group", label, values, denominator))
    return rows


def _technical_source_sector_rows(bundle_dir: Path | None, result, sector_groups):
    if bundle_dir is None: return []
    import numpy as np
    config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    with np.load(bundle_dir / config["input_npz"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[member], dtype=np.float64) for name, member in config["arrays"].items()}
    labels = config["semantics"]["emitter_labels"]
    sectors = [str(label).split(":", 1)[1] for label in labels]
    nested = nested_technical_components(arrays["C0"], arrays["C1"], arrays["T_coef0"], arrays["T_coef1"], arrays["w_coef0"], arrays["w_coef1"])
    supplier_sector = arrays["supplier_sector"].astype(int)
    y0 = (arrays["F0"] * arrays["B0"][supplier_sector, :] * arrays["levels0"][None, :]).sum(axis=1)
    y1 = (arrays["F1"] * arrays["B1"][supplier_sector, :] * arrays["levels1"][None, :]).sum(axis=1)
    propagation = {"v0": arrays["v0"], "delta_v": arrays["v1"] - arrays["v0"], "L0": arrays["L0"],
                   "L1": arrays["L1"], "y0": y0, "delta_y": y1 - y0}
    rows = []
    for effect, component in (("D_T_coef", nested.T_coef), ("D_w_coef", nested.w_coef)):
        splits = {}
        for sector in dict.fromkeys(sectors):
            mask = np.asarray([item == sector for item in sectors])[:, None]
            splits[sector] = attribute_structural_driver(component * mask, **propagation)
        reconstructed = np.sum(np.stack(list(splits.values())), axis=0)
        parent = np.asarray(result.effects[effect].baseline)
        if not np.allclose(reconstructed, parent, rtol=1e-5, atol=np.finfo(np.float32).eps * 256 * max(1, abs(parent).max())):
            raise ValueError(f"{effect}/source_sector split does not reconcile")
        denominator = _metrics(parent, 1)["gross_contribution"]
        for sector, values in splits.items(): rows.append(_supplemental_row(effect, "source_sector", sector, values, denominator))
        grouped = defaultdict(list)
        for sector, values in splits.items(): grouped[sector_groups.get(sector, sector)].append(values)
        for group, values in grouped.items(): rows.append(_supplemental_row(effect, "source_sector_group", group, np.stack(values), denominator))
    return rows


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _mapping(path: Path | None) -> dict[str, str]:
    if path is None: return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"mapping must be a string-to-string JSON object: {path}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    annual = sub.add_parser("annual")
    annual.add_argument("--canonical-run", required=True, type=Path)
    annual.add_argument("--sourcing-dir", type=Path)
    annual.add_argument("--bundle-dir", type=Path)
    annual.add_argument("--output-root", required=True, type=Path)
    annual.add_argument("--run-id", required=True)
    annual.add_argument("--year-prev", required=True, type=int)
    annual.add_argument("--year", required=True, type=int)
    annual.add_argument("--resolution", required=True)
    annual.add_argument("--units", required=True)
    annual.add_argument("--country-groups", type=Path)
    annual.add_argument("--sector-groups", type=Path)
    panel = sub.add_parser("panel")
    panel.add_argument("--annual-root", required=True, type=Path)
    panel.add_argument("--output-root", required=True, type=Path)
    panel.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "panel":
            annual_dirs = sorted(p.parent for p in args.annual_root.glob("*/manifest.v2.json"))
            if not annual_dirs: raise ValueError("no validated annual reports found")
            print(publish_panel(annual_dirs, args.output_root, run_id=args.run_id)); return 0
        sources = [_record(args.canonical_run / "manifest.json")]
        for candidate in (args.canonical_run / "source_provenance.json",):
            if candidate.is_file(): sources.append(_record(candidate))
        if args.sourcing_dir:
            sources.extend(_record(path) for path in sorted(args.sourcing_dir.rglob("*report.json")))
        if args.bundle_dir:
            sources.extend(_record(path) for path in (args.bundle_dir / "config.json", args.bundle_dir / "bundle_manifest.json") if path.is_file())
        result = load_result(args.canonical_run)
        country_groups, sector_groups = _mapping(args.country_groups), _mapping(args.sector_groups)
        supplemental = _sourcing_rows(args.sourcing_dir, result, country_groups)
        supplemental += _technical_source_sector_rows(args.bundle_dir, result, sector_groups)
        print(publish_annual_report(result, args.output_root, run_id=args.run_id,
            year_prev=args.year_prev, year=args.year, resolution=args.resolution, units=args.units,
            source_files=sources, country_groups=country_groups, sector_groups=sector_groups,
            supplementary_drilldown=supplemental))
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
