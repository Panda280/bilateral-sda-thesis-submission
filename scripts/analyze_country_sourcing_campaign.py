#!/usr/bin/env python3
"""Role-safe longitudinal sourcing indicators for a completed SDA campaign.

This is descriptive attribution analysis.  The offshoring-consistent ranking
does not identify offshoring causally.  D_C structural-driver countries and
D_YF consumer countries are deliberately kept as different roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KG_PER_MT = 1_000_000_000.0
SPECS = {
    "D_C": ("source_country", "D_C.source_country_cube.parquet",
            "source_country_cube_report.json", "Driver_Country",
            "structural_driver_country"),
    "D_YF": ("yf_source_country", "D_YF.source_country_cube.parquet",
             "yf_source_country_cube_report.json", "Consumer_Country",
             "consumer_country"),
}
SECTOR_SPECS = {
    "D_C": ("source_group_sector", "D_C.source_group_sector_cube.parquet",
            "source_group_sector_cube_report.json"),
    "D_YF": ("yf_source_group_sector", "D_YF.source_group_sector_cube.parquet",
             "yf_source_group_sector_cube_report.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_report(path: Path, effect: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("effect") != effect or report.get("units") != "kg CO2":
        raise RuntimeError(f"effect/unit mismatch: {path}")
    reconciliation = report.get("reconciliation", {})
    parent = reconciliation.get("parent", reconciliation)
    if parent.get("failing_index") is not None or parent.get("passed") is False:
        raise RuntimeError(f"failed reconciliation: {path}")
    return report


def inventory(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for run in sorted((campaign_root / "offshoring").glob("*_*_*")):
        if not run.is_dir():
            continue
        try:
            previous, current, resolution = run.name.rsplit("_", 2)
            previous, current = int(previous), int(current)
        except ValueError:
            continue
        for effect, (folder, filename, report_name, focal_col, focal_role) in SPECS.items():
            product, report_path = run / folder / filename, run / folder / report_name
            if not product.is_file() or not report_path.is_file():
                continue
            report = validate_report(report_path, effect)
            stat = product.stat()
            rows.append({
                "run_id": run.name, "year_previous": previous,
                "year_current": current, "resolution": resolution,
                "effect_id": effect, "focal_column": focal_col,
                "focal_role": focal_role, "product_path": str(product),
                "report_path": str(report_path), "report_sha256": sha256(report_path),
                "product_bytes": stat.st_size,
                "product_mtime_ns": stat.st_mtime_ns,
                "reported_total_kg": report["total"],
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no complete source-country products found")
    keys = ["year_previous", "year_current", "resolution"]
    counts = result.groupby(keys).effect_id.nunique()
    if (counts != 2).any():
        bad = counts[counts != 2].to_dict()
        raise RuntimeError(f"unmatched D_C/D_YF inventory: {bad}")
    return result.sort_values(keys + ["effect_id"])


def analyse_product(row) -> tuple[list[dict], list[dict]]:
    frame = pd.read_parquet(row.product_path)
    required = {row.focal_column, "Source_Country", "Value"}
    if not required.issubset(frame.columns) or not np.isfinite(frame.Value).all():
        raise RuntimeError(f"invalid product schema/values: {row.product_path}")
    if not np.isclose(frame.Value.sum(), row.reported_total_kg, rtol=1e-8,
                      atol=max(1e-6, abs(row.reported_total_kg) * 1e-10)):
        raise RuntimeError(f"product total mismatch: {row.product_path}")
    collapsed = frame.groupby([row.focal_column, "Source_Country"], as_index=False).Value.sum()
    indicators, shares = [], []
    for focal, group in collapsed.groupby(row.focal_column, sort=True):
        values = group.Value.to_numpy(float)
        sources = group.Source_Country.astype(str).to_numpy()
        total, gross = float(values.sum()), float(np.abs(values).sum())
        domestic = float(values[sources == str(focal)].sum())
        foreign = total - domestic
        foreign_gross = float(np.abs(values[sources != str(focal)]).sum())
        intensity = foreign_gross / gross if gross else np.nan
        absolute_shares = np.abs(values) / gross if gross else np.zeros(len(values))
        hhi = float(np.square(absolute_shares).sum()) if gross else np.nan
        entropy = float(-sum(p * math.log(p) for p in absolute_shares if p > 0)) if gross else np.nan
        evidence = "NO"
        if domestic < 0 and foreign > 0 and np.isfinite(intensity):
            evidence = "STRONG" if intensity >= .75 else "MODERATE" if intensity >= .5 else "WEAK"
        base = {"run_id": row.run_id, "year_previous": row.year_previous,
                "year_current": row.year_current, "resolution": row.resolution,
                "effect_id": row.effect_id, "focal_role": row.focal_role,
                "focal_country": str(focal)}
        indicators.append(base | {
            "net_kg": total, "domestic_kg": domestic, "foreign_kg": foreign,
            "gross_kg": gross, "absolute_foreign_intensity": intensity,
            "source_hhi": hhi, "source_entropy": entropy,
            "effective_source_count": math.exp(entropy) if np.isfinite(entropy) else np.nan,
            "cancellation_ratio": (gross - abs(total)) / gross if gross else np.nan,
            "offshoring_consistent_evidence": evidence,
        })
        for source, value, share in zip(sources, values, absolute_shares):
            shares.append(base | {"source_country": source,
                                  "signed_contribution_kg": float(value),
                                  "absolute_share": float(share),
                                  "signed_share_of_net": float(value / total) if total else np.nan})
    return indicators, shares


def longitudinal(indicators: pd.DataFrame, shares: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["resolution", "effect_id", "focal_role", "focal_country"]
    periods, cases = [], []
    for key, group in indicators.groupby(keys, dropna=False):
        group = group.sort_values("year_current")
        evidence_years = int((group.offshoring_consistent_evidence != "NO").sum())
        strong_years = int((group.offshoring_consistent_evidence == "STRONG").sum())
        periods.append(dict(zip(keys, key)) | {
            "n_transitions": len(group), "evidence_years": evidence_years,
            "strong_years": strong_years,
            "evidence_persistence": evidence_years / len(group),
            "cumulative_net_kg": group.net_kg.sum(),
            "cumulative_domestic_kg": group.domestic_kg.sum(),
            "cumulative_foreign_kg": group.foreign_kg.sum(),
            "cumulative_gross_kg": group.gross_kg.sum(),
            "mean_absolute_foreign_intensity": group.absolute_foreign_intensity.mean(),
            "mean_source_hhi": group.source_hhi.mean(),
            "mean_source_entropy": group.source_entropy.mean(),
            "mean_effective_source_count": group.effective_source_count.mean(),
            "mean_cancellation_ratio": group.cancellation_ratio.mean(),
        })
    period = pd.DataFrame(periods)
    for key, group in shares.groupby(keys, dropna=False):
        pivot = group.pivot_table(index="year_current", columns="source_country",
                                  values="absolute_share", aggfunc="sum", fill_value=0)
        dominant = pivot.idxmax(axis=1) if len(pivot.columns) else pd.Series(dtype=str)
        adjacent_same = float((dominant.iloc[1:].to_numpy() == dominant.iloc[:-1].to_numpy()).mean()) if len(dominant) > 1 else np.nan
        cumulative = group.groupby("source_country").signed_contribution_kg.sum()
        top_source = cumulative.abs().idxmax() if len(cumulative) else ""
        cases.append(dict(zip(keys, key)) | {
            "dominant_source_adjacent_persistence": adjacent_same,
            "period_dominant_source_country": top_source,
            "period_dominant_source_signed_kg": cumulative.get(top_source, np.nan),
        })
    persistence = pd.DataFrame(cases)
    period = period.merge(persistence, on=keys, how="left")
    eligible = period[(period.cumulative_domestic_kg < 0) & (period.cumulative_foreign_kg > 0)].copy()
    eligible["materiality_share"] = eligible.cumulative_gross_kg / eligible.groupby(["resolution", "effect_id"]).cumulative_gross_kg.transform("sum")
    eligible["dominant_source_persistence_for_score"] = eligible.dominant_source_adjacent_persistence.fillna(0)
    scoring_group = eligible.groupby(["resolution", "effect_id"])
    eligible["case_score"] = (
        scoring_group.evidence_persistence.rank(pct=True) +
        scoring_group.materiality_share.rank(pct=True) +
        scoring_group.mean_absolute_foreign_intensity.rank(pct=True) +
        scoring_group.dominant_source_persistence_for_score.rank(pct=True)
    ) / 4
    eligible = eligible.sort_values(["resolution", "effect_id", "case_score"], ascending=[True, True, False])
    eligible["rank_within_resolution_effect"] = eligible.groupby(["resolution", "effect_id"]).cumcount() + 1
    eligible["result_id"] = eligible.apply(lambda r: "SRC-" + hashlib.sha256(
        f"{r.resolution}|{r.effect_id}|{r.focal_role}|{r.focal_country}".encode()).hexdigest()[:12], axis=1)
    eligible["interpretation_limit"] = "descriptive offshoring-consistent pattern; not causal proof"
    return period, eligible


def source_sectors(campaign_root: Path, inv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in inv.itertuples(index=False):
        folder, filename, report_name = SECTOR_SPECS[record.effect_id]
        run_root = campaign_root / "offshoring" / record.run_id
        product, report_path = run_root / folder / filename, run_root / folder / report_name
        report = validate_report(report_path, record.effect_id)
        frame = pd.read_parquet(product, columns=["Source_Group", "Source_Sector", "Value"])
        if not np.isfinite(frame.Value).all() or not np.isclose(frame.Value.sum(), report["total"], rtol=1e-8,
                atol=max(1e-6, abs(report["total"]) * 1e-10)):
            raise RuntimeError(f"source-sector product invalid: {product}")
        grouped = frame.groupby(["Source_Group", "Source_Sector"], as_index=False).Value.sum()
        grouped.insert(0, "effect_id", record.effect_id); grouped.insert(0, "resolution", record.resolution)
        grouped.insert(0, "year_current", record.year_current); grouped.insert(0, "year_previous", record.year_previous)
        rows.append(grouped.rename(columns={"Source_Group": "source_group", "Source_Sector": "source_sector",
                                            "Value": "signed_contribution_kg"}))
    annual = pd.concat(rows, ignore_index=True)
    keys = ["resolution", "effect_id", "source_group", "source_sector"]
    cumulative = annual.groupby(keys, as_index=False).signed_contribution_kg.sum()
    gross = annual.assign(gross_period_kg=lambda x: x.signed_contribution_kg.abs()).groupby(
        keys, as_index=False).gross_period_kg.sum()
    return cumulative.merge(gross, on=keys, validate="one_to_one")


def figures(indicators: pd.DataFrame, shares: pd.DataFrame, sectors: pd.DataFrame,
            cases: pd.DataFrame, output: Path,
            edge_share_threshold: float, max_edges: int) -> list[dict]:
    plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9})
    records = []
    annual = indicators.groupby(["resolution", "effect_id", "year_current"])[["domestic_kg", "foreign_kg"]].sum() / KG_PER_MT
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, ((resolution, effect), data) in zip(axes.flat, annual.groupby(level=[0, 1])):
        d = data.droplevel([0, 1]); ax.plot(d.index, d.domestic_kg, marker="o", label="Domestic")
        ax.plot(d.index, d.foreign_kg, marker="o", label="Foreign"); ax.axhline(0, color="black", lw=.7)
        ax.set_title(f"{effect}, {resolution}"); ax.set_xlabel("Current year"); ax.set_ylabel("Signed contribution (Mt CO$_2$)"); ax.legend()
    path = output / "domestic_foreign_annual.pdf"; fig.savefig(path); plt.close(fig)
    records.append({"figure_id": "SRC-FIG-01", "file": path.name,
                    "description": "Signed domestic and foreign annual contributions, with effects and resolutions separate."})

    top = cases[cases.rank_within_resolution_effect <= 10].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for ax, ((resolution, effect), data) in zip(axes.flat, top.groupby(["resolution", "effect_id"], sort=True)):
        # The panel title already identifies effect and resolution, whose
        # contract fixes the focal role. Repeating long role names on every
        # tick made the thesis export unreadable without adding information.
        data = data.sort_values("case_score"); labels = data.focal_country
        ax.barh(labels, data.case_score); ax.set_title(f"{effect}, {resolution}"); ax.set_xlim(0, 1)
    path = output / "offshoring_consistent_case_scores.pdf"; fig.savefig(path); plt.close(fig)
    records.append({"figure_id": "SRC-FIG-02", "file": path.name,
                    "description": "Descriptive non-causal offshoring-consistent ranking."})

    cumulative = shares.groupby(["resolution", "effect_id", "focal_role", "focal_country", "source_country"], as_index=False).signed_contribution_kg.sum()
    cumulative["abs_kg"] = cumulative.signed_contribution_kg.abs()
    cumulative["share_of_effect_resolution_gross"] = cumulative.abs_kg / cumulative.groupby(["resolution", "effect_id"]).abs_kg.transform("sum")
    edges = cumulative[cumulative.share_of_effect_resolution_gross >= edge_share_threshold].copy()
    edges = edges.sort_values("abs_kg", ascending=False).head(max_edges)
    edges.to_csv(output / "network_edges_thresholded.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, max(4, len(edges) * .18)))
    labels = edges.focal_country + " <- " + edges.source_country + " [" + edges.effect_id + "/" + edges.resolution + "]"
    colors = np.where(edges.signed_contribution_kg >= 0, "#b2182b", "#2166ac")
    order = np.arange(len(edges)); ax.barh(order, edges.signed_contribution_kg / KG_PER_MT, color=colors)
    ax.set_yticks(order, labels); ax.axvline(0, color="black", lw=.7); ax.set_xlabel("Signed cumulative contribution (Mt CO$_2$)")
    ax.set_title(f"Bilateral source edges: abs share >= {edge_share_threshold:.1%}; top {max_edges}")
    path = output / "bilateral_source_network_edges.pdf"; fig.savefig(path); plt.close(fig)
    records.append({"figure_id": "SRC-FIG-03", "file": path.name,
                    "description": "Signed bilateral edges under an explicit absolute-share threshold."})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
    for ax, ((resolution, effect), data) in zip(axes.flat, sectors.groupby(["resolution", "effect_id"], sort=True)):
        data = data.nlargest(12, "gross_period_kg").sort_values("signed_contribution_kg")
        labels = data.source_group + ":" + data.source_sector.astype(str)
        colors = np.where(data.signed_contribution_kg >= 0, "#b2182b", "#2166ac")
        ax.barh(labels, data.signed_contribution_kg / KG_PER_MT, color=colors)
        ax.axvline(0, color="black", lw=.7); ax.set_title(f"{effect}, {resolution}")
        ax.set_xlabel("Signed cumulative contribution (Mt CO$_2$)")
    path = output / "source_sector_cumulative_top_gross.pdf"; fig.savefig(path); plt.close(fig)
    records.append({"figure_id": "SRC-FIG-04", "file": path.name,
                    "description": "Source-group/sector cells selected by period gross magnitude and plotted signed."})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-share-threshold", type=float, default=.005)
    parser.add_argument("--max-network-edges", type=int, default=50)
    args = parser.parse_args()
    if not 0 <= args.edge_share_threshold <= 1 or args.max_network_edges < 1:
        raise ValueError("invalid network threshold or edge count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inv = inventory(args.campaign_root)
    indicator_rows, share_rows = [], []
    for row in inv.itertuples(index=False):
        a, b = analyse_product(row); indicator_rows.extend(a); share_rows.extend(b)
    indicators, shares = pd.DataFrame(indicator_rows), pd.DataFrame(share_rows)
    period, cases = longitudinal(indicators, shares)
    sectors = source_sectors(args.campaign_root, inv)
    inv.drop(columns=["product_path", "report_path"]).to_csv(args.output_dir / "input_inventory.csv", index=False)
    indicators.to_parquet(args.output_dir / "annual_country_sourcing.parquet", index=False)
    shares.to_parquet(args.output_dir / "annual_bilateral_source_shares.parquet", index=False)
    period.to_csv(args.output_dir / "country_period_scorecards.csv", index=False)
    cases.to_csv(args.output_dir / "offshoring_consistent_cases.csv", index=False)
    sectors.to_csv(args.output_dir / "cumulative_source_sector.csv", index=False)
    figure_records = figures(indicators, shares, sectors, cases, args.output_dir,
                             args.edge_share_threshold, args.max_network_edges)
    manifest = {
        "analysis_id": "country-sourcing-campaign-v1", "status": "complete",
        "campaign_root": str(args.campaign_root), "units_input": "kg CO2",
        "display_conversion": "1 Mt CO2 = 1e9 kg CO2", "effects_kept_separate": ["D_C", "D_YF"],
        "role_semantics": {"D_C": "structural_driver_country", "D_YF": "consumer_country"},
        "input_products": len(inv), "annual_indicator_rows": len(indicators),
        "bilateral_rows": len(shares), "period_rows": len(period), "ranked_case_rows": len(cases),
        "source_sector_rows": len(sectors),
        "network_threshold": {"absolute_share_of_effect_resolution_gross": args.edge_share_threshold,
                              "maximum_edges": args.max_network_edges, "signed_values_preserved": True},
        "figures": figure_records,
        "limitations": ["descriptive attribution, not causal proof of offshoring",
                        "current campaign remains provisional pending upstream validation",
                        "cross-resolution rows are not pooled in scorecards or rankings"],
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
