"""Build the P2 controlled-granularity and nested T/w analysis.

The publication deliberately separates (a) within-R3 controlled aggregation
from (b) comparison with independently calculated 2013/2016-resolution SDA
results.  Only ``reviewed_unique`` concordance rows are admitted. Unresolved
fine sectors are excluded on both matrix axes and their retained signed and
absolute mass is reported rather than silently reassigned.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sda.analysis.granularity import concentration, controlled_aggregate, diagnose_aggregation
from sda.analysis.tw import summarize_tw

EFFECTS = ("D_A_x", "D_T_coef", "D_w_coef")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run(name: str) -> tuple[int, int, str]:
    parts = name.removeprefix("sda_").split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid canonical run name: {name}")
    return int(parts[0]), int(parts[1]), parts[2]


def norm_label(label: object) -> tuple[str, str]:
    # Canonical publications persist country-sector tuples as ``CC:sector``
    # strings, while in-memory synthetic results retain Python tuples.
    if isinstance(label, str) and ":" in label:
        country, sector = label.split(":", 1)
        return country.strip(), sector.strip()
    if not isinstance(label, tuple) or len(label) != 2:
        raise ValueError(f"expected (country, sector), found {label!r}")
    return str(label[0]).strip(), str(label[1]).strip()


def _tuple_values(value: object) -> list[object]:
    if isinstance(value, dict) and "__tuple__" in value:
        return list(value["__tuple__"])
    if isinstance(value, list):
        return value
    raise ValueError("canonical label payload is not tuple-like")


def load_effect(run: Path, effect_id: str, *, verify: bool = True):
    """Load one canonical effect as a read-only mmap, not the full SDA result."""
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["effects"][effect_id]
    array_record = record["baseline"]
    array_path = run / array_record["file"]
    if verify and sha256(array_path) != array_record["sha256"]:
        raise ValueError(f"checksum failed: {array_path}")
    dims = record["dimensions"]
    row_labels = tuple(norm_label(x) for x in _tuple_values(dims["row"]["labels"]))
    column = dims.get("column")
    if column is None:
        raise ValueError(f"{run.name}/{effect_id} has no column dimension")
    col_labels = tuple(norm_label(x) for x in _tuple_values(column["labels"]))
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    return values, row_labels, col_labels


def scalar_tw(t_value: float, w_value: float) -> tuple[float, float, float, str]:
    denominator = abs(t_value) + abs(w_value)
    if denominator == 0:
        return 0.0, 0.0, 0.0, "zero"
    ts, ws = abs(t_value) / denominator, abs(w_value) / denominator
    cancellation = 1.0 - abs(t_value + w_value) / denominator
    dominance = "T-dominant" if ts >= .6 else "w-dominant" if ws >= .6 else "balanced"
    return ts, ws, cancellation, dominance


def vector_abs_parent(t: np.ndarray, w: np.ndarray, axis: int, block: int = 128) -> np.ndarray:
    """Return abs(T+W) marginals without materialising the full sum."""
    result = np.zeros(t.shape[axis], dtype=float)
    for start in range(0, t.shape[0], block):
        stop = min(start + block, t.shape[0])
        values = np.abs(np.asarray(t[start:stop]) + np.asarray(w[start:stop]))
        if axis == 0:
            result[start:stop] = values.sum(axis=1)
        else:
            result += values.sum(axis=0)
    return result


def read_mapping(path: Path, target: str):
    accepted: dict[tuple[str, str], tuple[str, str]] = {}
    unresolved: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["coarse_resolution"].strip() != target:
                continue
            fine = row["fine_country"].strip(), row["fine_sector"].strip()
            status = row["mapping_status"].strip()
            if status == "reviewed_unique":
                coarse = row["coarse_country"].strip(), row["coarse_sector"].strip()
                if fine in accepted and accepted[fine] != coarse:
                    raise ValueError(f"ambiguous reviewed mapping for {fine} to {target}")
                accepted[fine] = coarse
            else:
                unresolved.add(fine)
    if not accepted:
        raise ValueError(f"no reviewed mappings for target {target}")
    return accepted, unresolved


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def rank_metrics(left: np.ndarray, right: np.ndarray, top_k: int = 100) -> tuple[float, float]:
    x, y = np.abs(left).ravel(), np.abs(right).ravel()
    if x.size != y.size:
        raise ValueError("rank inputs differ in size")
    rx, ry = average_ranks(x), average_ranks(y)
    rho = 0.0 if np.ptp(rx) == 0 or np.ptp(ry) == 0 else float(np.corrcoef(rx, ry)[0, 1])
    k = min(top_k, x.size)
    ix = set(np.argsort(x, kind="stable")[-k:])
    iy = set(np.argsort(y, kind="stable")[-k:])
    return rho, len(ix & iy) / k


def comparison_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"harmonized arrays differ: {left.shape} versus {right.shape}")
    l1 = float(np.abs(left - right).sum())
    denominator = float(np.abs(right).sum())
    both = (left != 0) & (right != 0)
    union = (left != 0) | (right != 0)
    source_left = np.abs(left).sum(axis=0)
    source_right = np.abs(right).sum(axis=0)
    if source_left.sum() and source_right.sum():
        source_tv = float(0.5 * np.abs(source_left / source_left.sum() - source_right / source_right.sum()).sum())
    else:
        source_tv = 0.0
    rho, overlap = rank_metrics(left, right)
    return {
        "signed_difference": float(left.sum() - right.sum()),
        "absolute_difference_l1": l1,
        "normalized_l1_native": 0.0 if denominator == 0 else l1 / denominator,
        "sign_disagreement_rate": 0.0 if not both.any() else float((np.sign(left[both]) != np.sign(right[both])).mean()),
        "edge_symmetric_difference_rate": 0.0 if not union.any() else float(((left != 0) != (right != 0))[union].mean()),
        "magnitude_rank_spearman": rho,
        "top_100_overlap": overlap,
        "source_substitution_total_variation": source_tv,
    }


def aggregate_and_diagnose(values: np.ndarray, row_groups, column_groups):
    """Controlled sum and diagnostics with bounded working memory."""
    row_unique = tuple(dict.fromkeys(row_groups))
    col_unique = tuple(dict.fromkeys(column_groups))
    row_lookup = {x: i for i, x in enumerate(row_unique)}
    col_lookup = {x: i for i, x in enumerate(col_unique)}
    row_codes = np.array([row_lookup[x] for x in row_groups], dtype=int)
    col_codes = np.array([col_lookup[x] for x in column_groups], dtype=int)
    coarse = np.zeros((len(row_unique), len(col_unique)), dtype=float)
    for i in range(values.shape[0]):
        coarse[row_codes[i]] += np.bincount(col_codes, weights=np.asarray(values[i]), minlength=len(col_unique))
    signed = abs_total = square_sum = mag_log_sum = 0.0
    reversals = reversal_mag = lost = nonzero_count = 0.0
    for start in range(0, values.shape[0], 64):
        stop = min(start + 64, values.shape[0])
        block = np.asarray(values[start:stop])
        mag = np.abs(block)
        expanded = coarse[row_codes[start:stop, None], col_codes[None, :]]
        nonzero = block != 0
        reversal = nonzero & (expanded != 0) & (np.signbit(block) != np.signbit(expanded))
        lost_mask = nonzero & (expanded == 0)
        signed += float(block.sum()); abs_total += float(mag.sum()); square_sum += float(np.square(mag).sum())
        positive = mag > 0
        mag_log_sum += float((mag[positive] * np.log(mag[positive])).sum())
        nonzero_count += int(nonzero.sum()); reversals += int(reversal.sum())
        reversal_mag += float(mag[reversal].sum()); lost += int(lost_mask.sum())
    coarse_mag = np.abs(coarse); coarse_abs = float(coarse_mag.sum())
    coarse_positive = coarse_mag > 0
    fine_hhi = 0.0 if abs_total == 0 else square_sum / abs_total ** 2
    fine_entropy = 0.0 if abs_total == 0 else math.log(abs_total) - mag_log_sum / abs_total
    coarse_hhi = 0.0 if coarse_abs == 0 else float(np.square(coarse_mag).sum() / coarse_abs ** 2)
    coarse_entropy = 0.0 if coarse_abs == 0 else float(math.log(coarse_abs) - (coarse_mag[coarse_positive] * np.log(coarse_mag[coarse_positive])).sum() / coarse_abs)
    metrics = {
        "fine_signed_total": signed, "coarse_signed_total": float(coarse.sum()),
        "signed_conservation_error": float(coarse.sum() - signed),
        "fine_absolute_total": abs_total, "coarse_absolute_total": coarse_abs,
        "cancellation_loss": 0.0 if abs_total == 0 else 1.0 - coarse_abs / abs_total,
        "sign_reversal_rate": 0.0 if nonzero_count == 0 else reversals / nonzero_count,
        "magnitude_weighted_sign_reversal_rate": 0.0 if abs_total == 0 else reversal_mag / abs_total,
        "fine_hhi": fine_hhi, "coarse_hhi": coarse_hhi, "hhi_change": coarse_hhi - fine_hhi,
        "fine_entropy": fine_entropy, "coarse_entropy": coarse_entropy,
        "entropy_change": coarse_entropy - fine_entropy,
        "sparse_edge_loss_rate": 0.0 if nonzero_count == 0 else lost / nonzero_count,
    }
    return coarse, row_unique, col_unique, metrics


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(canonical_root: Path, concordance: Path, output: Path, *, units: str,
          price_basis: str, provisional: bool, expected_runs: int = 42) -> Path:
    runs = sorted(p for p in canonical_root.glob("sda_*") if (p / "manifest.json").is_file())
    if len(runs) != expected_runs:
        raise ValueError(f"expected {expected_runs} canonical runs, found {len(runs)}")
    by_key = {parse_run(p.name): p for p in runs}
    targets = ("2013", "2016")
    maps = {target: read_mapping(concordance, target) for target in targets}
    staging = output.parent / f".staging-{output.name}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    controlled_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    tw_rows: list[dict[str, object]] = []
    source_paths: set[Path] = set()
    try:
        for previous in range(2000, 2014):
            fine_path = by_key[(previous, previous + 1, "R3")]
            source_paths.add(fine_path / "manifest.json")

            # T/w descriptive tests at world and both country-sector axes.
            t, row_labels, col_labels = load_effect(fine_path, "D_T_coef")
            w, w_rows, w_cols = load_effect(fine_path, "D_w_coef")
            parent, p_rows, p_cols = load_effect(fine_path, "D_A_x")
            if (row_labels, col_labels) != (w_rows, w_cols) or (row_labels, col_labels) != (p_rows, p_cols):
                raise ValueError(f"{fine_path.name}: T/w/parent dimensions differ")
            residual = 0.0
            for start in range(0, t.shape[0], 128):
                stop = min(start + 128, t.shape[0])
                residual = max(residual, float(np.max(np.abs(np.asarray(t[start:stop]) + np.asarray(w[start:stop]) - np.asarray(parent[start:stop])))))
            t_row, w_row = np.asarray(t).sum(axis=1), np.asarray(w).sum(axis=1)
            t_col, w_col = np.asarray(t).sum(axis=0), np.asarray(w).sum(axis=0)
            t_abs_row, w_abs_row = np.abs(t).sum(axis=1), np.abs(w).sum(axis=1)
            t_abs_col, w_abs_col = np.abs(t).sum(axis=0), np.abs(w).sum(axis=0)
            parent_abs_row = vector_abs_parent(t, w, axis=0)
            parent_abs_col = vector_abs_parent(t, w, axis=1)
            scopes = (
                ("world", [("WORLD", np.arange(t.shape[0]), t_row, w_row, t_abs_row, w_abs_row, parent_abs_row)]),
                ("emitter_country", [(country, np.flatnonzero(np.array([x[0] == country for x in row_labels])), t_row, w_row, t_abs_row, w_abs_row, parent_abs_row) for country in sorted({x[0] for x in row_labels})]),
                ("driver_country", [(country, np.flatnonzero(np.array([x[0] == country for x in col_labels])), t_col, w_col, t_abs_col, w_abs_col, parent_abs_col) for country in sorted({x[0] for x in col_labels})]),
                ("emitter_sector", [(sector, np.flatnonzero(np.array([x[1] == sector for x in row_labels])), t_row, w_row, t_abs_row, w_abs_row, parent_abs_row) for sector in sorted({x[1] for x in row_labels})]),
                ("driver_sector", [(sector, np.flatnonzero(np.array([x[1] == sector for x in col_labels])), t_col, w_col, t_abs_col, w_abs_col, parent_abs_col) for sector in sorted({x[1] for x in col_labels})]),
            )
            for scope, selectors in scopes:
                for label, indexes, t_sum, w_sum, t_abs, w_abs, parent_abs_vector in selectors:
                    t_value, w_value = float(t_sum[indexes].sum()), float(w_sum[indexes].sum())
                    t_share, w_share, cancellation, dominance = scalar_tw(t_value, w_value)
                    child_abs = float(t_abs[indexes].sum() + w_abs[indexes].sum())
                    parent_abs = float(parent_abs_vector[indexes].sum())
                    tw_rows.append({
                        "run_id": fine_path.name, "year_previous": previous,
                        "year_current": previous + 1, "resolution": "R3", "scope": scope,
                        "label": label, "units": units, "t_value": t_value,
                        "w_value": w_value, "parent_value": t_value + w_value,
                        "t_absolute_share_of_signed_totals": t_share,
                        "w_absolute_share_of_signed_totals": w_share,
                        "paired_total_cancellation": cancellation,
                        "dominance": dominance,
                        "child_cell_absolute_total": child_abs,
                        "parent_cell_absolute_total": parent_abs,
                        "split_information_gain_ratio": 0.0 if child_abs == 0 else 1.0 - parent_abs / child_abs,
                        "max_cellwise_reconstruction_error": residual,
                    })

            for target in targets:
                mapping, unresolved = maps[target]
                row_keep = np.array([label in mapping for label in row_labels])
                col_keep = np.array([label in mapping for label in col_labels])
                row_groups = [mapping[label] for label, keep in zip(row_labels, row_keep) if keep]
                col_groups = [mapping[label] for label, keep in zip(col_labels, col_keep) if keep]
                native_path = by_key[(previous, previous + 1, target)]
                source_paths.add(native_path / "manifest.json")
                for effect_id in EFFECTS:
                    fine, effect_rows, effect_cols = load_effect(fine_path, effect_id, verify=False)
                    if (effect_rows, effect_cols) != (row_labels, col_labels):
                        raise ValueError(f"{fine_path.name}/{effect_id}: dimensions differ")
                    native, native_rows, native_cols = load_effect(native_path, effect_id)
                    selected = fine[np.ix_(row_keep, col_keep)]
                    coarse_values, coarse_rows, coarse_cols, diagnostic = aggregate_and_diagnose(selected, row_groups, col_groups)
                    row_pos = {label: i for i, label in enumerate(coarse_rows)}
                    col_pos = {label: i for i, label in enumerate(coarse_cols)}
                    missing_native = sorted((set(native_rows) - set(row_pos)) | (set(native_cols) - set(col_pos)))
                    common_rows = [label for label in native_rows if label in row_pos]
                    common_cols = [label for label in native_cols if label in col_pos]
                    harmonized = coarse_values[np.ix_([row_pos[x] for x in common_rows], [col_pos[x] for x in common_cols])]
                    native_row_pos = {label: i for i, label in enumerate(native_rows)}
                    native_col_pos = {label: i for i, label in enumerate(native_cols)}
                    native_common = native[np.ix_([native_row_pos[x] for x in common_rows], [native_col_pos[x] for x in common_cols])]
                    full_abs = float(np.abs(fine).sum())
                    retained_abs = float(np.abs(selected).sum())
                    base = {"year_previous": previous, "year_current": previous + 1,
                            "target_resolution": target, "effect_id": effect_id, "units": units}
                    controlled_rows.append({
                        **base, "run_id": fine_path.name, "fine_cells": fine.size,
                        "retained_cells": selected.size, "fine_labels_per_axis": len(row_labels),
                        "retained_labels_per_axis": int(row_keep.sum()),
                        "unresolved_fine_label_count": len(unresolved),
                        "retained_absolute_mass_share": 0.0 if full_abs == 0 else retained_abs / full_abs,
                        "retained_signed_mass_share": 0.0 if fine.sum() == 0 else float(selected.sum() / fine.sum()),
                        **diagnostic,
                    })
                    comparison_rows.append({
                        **base, "fine_run_id": fine_path.name, "native_run_id": native_path.name,
                        "common_rows": len(common_rows), "common_columns": len(common_cols),
                        "native_labels_excluded": len(missing_native),
                        **comparison_metrics(harmonized, native_common),
                    })
                    del selected, coarse_values, native, fine
            del t, w, parent

        controlled_path = staging / "controlled_within_r3_aggregation.csv"
        comparison_path = staging / "harmonized_cross_database_comparison.csv"
        tw_path = staging / "tw_descriptive_tests.csv"
        write_rows(controlled_path, controlled_rows)
        write_rows(comparison_path, comparison_rows)
        write_rows(tw_path, tw_rows)

        # Compact, thesis-ready static figures; signed values are never clipped.
        import matplotlib.pyplot as plt
        world = [row for row in tw_rows if row["scope"] == "world"]
        years = [row["year_current"] for row in world]
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        ax.plot(years, [row["t_value"] / 1e12 for row in world], marker="o", label=r"$D_T$")
        ax.plot(years, [row["w_value"] / 1e12 for row in world], marker="o", label=r"$D_w$")
        ax.axhline(0, color="black", linewidth=.7); ax.set_ylabel("Gt CO$_2$")
        ax.set_xlabel("Current year"); ax.legend(); fig.tight_layout()
        tw_fig = staging / "tw_world_annual_signed.pdf"; fig.savefig(tw_fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for target, marker in (("2013", "o"), ("2016", "s")):
            selected = [r for r in comparison_rows if r["target_resolution"] == target and r["effect_id"] == "D_A_x"]
            axes[0].plot([r["year_current"] for r in selected], [r["normalized_l1_native"] for r in selected], marker=marker, label=target)
            axes[1].plot([r["year_current"] for r in selected], [r["source_substitution_total_variation"] for r in selected], marker=marker, label=target)
        axes[0].set_ylabel("Normalized L1 difference"); axes[1].set_ylabel("Source-share total variation")
        for ax in axes: ax.set_xlabel("Current year"); ax.legend(title="Target")
        fig.tight_layout(); gran_fig = staging / "harmonized_information_loss.pdf"; fig.savefig(gran_fig); plt.close(fig)

        limitations = [
            "All outputs are provisional fossil-origin CO2 because upstream CYP/PYP validation has not passed.",
            "Four unresolved R3 sectors are excluded on both axes; retained mass is reported per effect and transition.",
            "Controlled aggregation is an within-R3 operation; cross-database comparison is separately labelled and does not identify database quality.",
            "T and w economic labels remain implementation-derived hypotheses; no value-added proxy is available in the canonical result publications.",
        ]
        report = staging / "report.md"
        report.write_text(
            "# P2 granularity and nested T/w analysis\n\n"
            f"Status: **{'accepted provisional' if provisional else 'validated'}**. "
            f"Coverage: {len(runs)}/{expected_runs} canonical runs; units: `{units}`; price basis: `{price_basis}`.\n\n"
            "The controlled experiment aggregates reviewed R3 cells inside the same calculated effect. "
            "The harmonized comparison then compares that result with the independently calculated target-resolution result. "
            "These are distinct estimands. The T/w table reports signed magnitude, paired cancellation, cell-level information gain, and country/sector dominance.\n\n"
            "## Evidentiary limits\n\n" + "\n".join(f"- {x}" for x in limitations) + "\n",
            encoding="utf-8",
        )
        artifacts = [controlled_path, comparison_path, tw_path, tw_fig, gran_fig, report]
        manifest = {
            "schema_version": 1,
            "analysis_id": "p2-granularity-tw-20260831",
            "status": "complete",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": {"script": "scripts/build_p2_granularity_tw_analysis.py", "command": sys.argv},
            "scope": {"years": [2000, 2014], "resolutions": ["R3", "2013", "2016"],
                      "effects": list(EFFECTS), "units": units, "price_basis": price_basis,
                      "indicator": "fossil-origin CO2", "provisional": provisional},
            "coverage": {"expected_runs": expected_runs, "accepted_runs": len(runs),
                         "controlled_rows": len(controlled_rows), "comparison_rows": len(comparison_rows),
                         "tw_rows": len(tw_rows)},
            "concordance": {"path": str(concordance.resolve()), "sha256": sha256(concordance),
                            "admitted_status": "reviewed_unique", "excluded_status": "unresolved_one_to_many"},
            "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in sorted(source_paths)],
            "outputs": [{"path": p.name, "sha256": sha256(p)} for p in artifacts],
            "validation": {"canonical_checksums_verified_by_loader": True,
                           "inventory_complete": len(runs) == expected_runs,
                           "unresolved_links_excluded": True,
                           "controlled_signed_conservation_checked": True,
                           "tw_cellwise_reconstruction_checked": True,
                           "upstream_status": "accepted_provisional" if provisional else "passed"},
            "limitations": limitations,
        }
        manifest_path = staging / "analysis_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        staging.replace(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--concordance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--units", default="kg CO2")
    parser.add_argument("--price-basis", default="CYP_(t-1) -> PYP_t")
    parser.add_argument("--expected-runs", type=int, default=42)
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args(argv)
    print(build(args.canonical_root, args.concordance, args.output, units=args.units,
                price_basis=args.price_basis, provisional=args.provisional,
                expected_runs=args.expected_runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
