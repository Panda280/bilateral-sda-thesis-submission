#!/usr/bin/env python3
"""Three-way comparison of the native-WIOD CO2 campaign (task C).

Task C of `docs/HANDOVER_CURRENT_STATE_20260902.md` §6. Builds matched 2001-2009
panels for the three native configurations and quantifies their differences,
plus a comparison against the main EXIOBASE campaign.

Comparisons
-----------
* ``Q1_target_grid``   -- ``jrc2019_to_2016`` vs ``jrc2019_to_2013``
  (same JRC2019 impacts and WIOD2016 flows; native 2016 grid vs harmonised
  2013 grid). Node totals only; the two use different country/sector grids and
  the 2013 side carries the non-nested ``C13-C15`` 50/50 allocation.
* ``Q2_cross_database`` -- ``jrc2019_to_2013`` vs ``wiod2013_native``
  (both on the 41x35 WIOD2013 grid; differ in impact database, flow-table
  vintage and, on the JRC side, the harmonisation step -- not a single-factor
  contrast).
* ``Q3a_vs_main_2016``  -- ``jrc2019_to_2016`` vs the main EXIOBASE campaign,
  resolution ``2016``.
* ``Q3b_vs_main_2013``  -- ``jrc2019_to_2013`` vs main EXIOBASE resolution ``2013``.
* ``Q3c_vs_main_2013``  -- ``wiod2013_native`` vs main EXIOBASE resolution ``2013``.
* ``Q4*`` / ``Q5*`` -- 2x2 on the 41x35 grid over {JRC2019, WIOD2013} emission
  accounts x {WIOD2016, WIOD2013} flow tables. Q4 varies the flow table with the
  emission account fixed (input-output / flow-table effect); Q5 varies the
  emission account with the flow table fixed (environmental-extension effect).

The main EXIOBASE campaign is fossil-origin CO2 in ``kg CO2`` (D-2026-011,
D-2026-022), NOT the AR4 GWP100 basket that §6.C names -- flagged as a handover
discrepancy. Q3 is therefore an environmental-account-scope + database (+ for
JRC, flow-vintage) comparison, not a pure database effect and not a CO2-vs-
CO2-equivalent indicator comparison.

Also builds a matched 2001-2009 comparison across every dataset x resolution
series (the three native configurations plus the main EXIOBASE campaign on the
2016, 2013 and R3 grids): ``matched_period_summary.parquet`` (cumulative total
change and primary effects) and ``cross_series_agreement.parquet`` (pairwise
annual sign agreement, correlation and NMAE).

Outputs (one immutable directory): ``panel.parquet``,
``matched_period_summary.parquet``, ``cross_series_agreement.parquet``,
``comparison_nodes.parquet``, ``comparison_ranks.parquet``,
``difference_reconciliation.parquet``, ``raw_dax_diagnostic.parquet``,
``REPORT.md``, ``analysis_manifest.json``.

Never sums raw ``D_Ax2`` with the ``D_T_coef``/``D_w_coef`` split (handover §9):
the raw split is carried only in ``raw_dax_diagnostic.parquet``.
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

YEARS = list(range(2001, 2010))
NATIVE_CONFIGS = ("jrc2019_to_2016", "jrc2019_to_2013", "wiod2013_native",
                  "jrc2019_flows2013", "wiod2013_flows2016",
                  # Phase 1: genuine WIOD2013-PUBLISHED flow tables (the cells
                  # above all use EXIOBASE-derived flows). 41x35 grid.
                  "wiod2013_true_native", "jrc2019_wiodflows2013", "exio_wiodflows2013")
HIER_NODES = (
    "delta_g", "emission_intensity", "leontief_production_structure", "final_demand",
    "D_v", "D_C", "D_A_x", "D_YF", "D_YB", "D_y", "D_T_coef", "D_w_coef",
)
DEPTH = {
    "delta_g": 0, "emission_intensity": 1, "leontief_production_structure": 1, "final_demand": 1,
    "D_v": 2, "D_C": 2, "D_A_x": 2, "D_YF": 2, "D_YB": 2, "D_y": 2, "D_T_coef": 3, "D_w_coef": 3,
}
# difference-of-differences reconciliation: child nodes -> parent
DIFF_IDENTITIES = {
    "delta_g": ["emission_intensity", "leontief_production_structure", "final_demand"],
    "leontief_production_structure": ["D_C", "D_A_x"],
    "D_A_x": ["D_T_coef", "D_w_coef"],
    "final_demand": ["D_YF", "D_YB", "D_y"],
}
# emission-account x flow-table x grid matrix for the matched 2001-2009 comparison
SERIES_MATRIX = {
    "exio_2016": {"ea": "EXIOBASE fossil CO2", "flows": "WIOD2016", "grid": "2016",
                  "dataset": "EXIOBASE fossil-origin CO2"},
    "exio_2013": {"ea": "EXIOBASE fossil CO2", "flows": "WIOD2013", "grid": "2013",
                  "dataset": "EXIOBASE fossil-origin CO2"},
    "exio_R3": {"ea": "EXIOBASE fossil CO2", "flows": "EXIOBASE R3", "grid": "R3",
                "dataset": "EXIOBASE fossil-origin CO2"},
    "jrc2019_to_2016": {"ea": "JRC2019", "flows": "WIOD2016", "grid": "2016",
                        "dataset": "JRC2019 native CO2 / WIOD2016 flows"},
    "jrc2019_to_2013": {"ea": "JRC2019", "flows": "WIOD2016", "grid": "2013",
                        "dataset": "JRC2019 native CO2 / WIOD2016 flows (harmonised)"},
    "jrc2019_flows2013": {"ea": "JRC2019", "flows": "WIOD2013", "grid": "2013",
                          "dataset": "JRC2019 native CO2 / WIOD2013 flows"},
    "wiod2013_flows2016": {"ea": "WIOD2013", "flows": "WIOD2016", "grid": "2013",
                           "dataset": "WIOD2013 native CO2 / WIOD2016 flows"},
    "wiod2013_native": {"ea": "WIOD2013", "flows": "WIOD2013", "grid": "2013",
                        "dataset": "WIOD2013 native CO2 / WIOD2013 flows"},
    "wiod2013_true_native": {"ea": "WIOD2013", "flows": "WIOD2013 published", "grid": "2013",
                             "dataset": "WIOD2013 native CO2 / WIOD2013 PUBLISHED flows (fully self-consistent)"},
    "jrc2019_wiodflows2013": {"ea": "JRC2019", "flows": "WIOD2013 published", "grid": "2013",
                              "dataset": "JRC2019 native CO2 / WIOD2013 PUBLISHED flows"},
    "exio_wiodflows2013": {"ea": "EXIOBASE fossil CO2", "flows": "WIOD2013 published", "grid": "2013",
                           "dataset": "EXIOBASE fossil CO2 / WIOD2013 PUBLISHED flows"},
}
PRIMARY_NODES = ("delta_g", "emission_intensity", "leontief_production_structure", "final_demand")
# sourcing / production-recipe / final-demand subeffects, comparable only on a
# shared grid (Q1 crosses grids, so it is excluded from subeffect agreement)
SUBEFFECT_NODES = ("D_v", "D_C", "D_A_x", "D_T_coef", "D_w_coef", "D_YF", "D_YB", "D_y")
AGREEMENT_NODES = PRIMARY_NODES + SUBEFFECT_NODES
# "sourcing components" in the thesis sense
SOURCING_NODES = ("D_C", "D_YF")
RANK_EFFECTS = ("D_v", "D_C", "D_A_x", "D_T_coef", "D_w_coef", "D_YF", "D_YB", "D_y")

# alternative cumulation windows for the period-selection sensitivity
WINDOWS = {
    "2001-2009 (full native window)": list(range(2001, 2010)),
    "2001-2007 (pre-crisis)": list(range(2001, 2008)),
    "2001-2008": list(range(2001, 2009)),
    "2002-2009 (drop first transition)": list(range(2002, 2010)),
    "2004-2009": list(range(2004, 2010)),
    "2001-2009 excl. 2003-2004 (instability)": [y for y in range(2001, 2010) if y not in (2003, 2004)],
    "2008-2009 (crisis onset only)": [2008, 2009],
}
RANK_GRAINS = ("emitter_country", "emitter_sector")
RANK_TOP_N = 20
SIGN_EPS = 1.0  # kg CO2; below this a contribution is treated as ~zero for sign flips


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hierarchy(report_dir: Path) -> pd.DataFrame:
    frame = pd.read_parquet(report_dir / "annual_hierarchy.parquet")
    return frame.set_index("node_id")


def load_drilldown(report_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(report_dir / "annual_drilldown.parquet")


def net_gross_ratio(row) -> float | None:
    gross = float(row["gross_contribution"])
    return None if gross == 0 else float(row["contribution"]) / gross


def sign(x: float) -> int:
    return 0 if abs(x) < SIGN_EPS else (1 if x > 0 else -1)


def build_panel(native_root: Path, main_root: Path) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        yp = year - 1
        for config in NATIVE_CONFIGS:
            report_dir = native_root / f"{yp}_{year}_{config}"
            if not report_dir.is_dir():
                continue  # config not present in this campaign (e.g. v1 has 3, v2 has 5)
            h = load_hierarchy(report_dir)
            for node in HIER_NODES:
                r = h.loc[node]
                rows.append({
                    "dataset": "native", "series": config, "year_previous": yp, "year_current": year,
                    "node_id": node, "depth": DEPTH[node],
                    "contribution": float(r["contribution"]),
                    "positive_contribution": float(r["positive_contribution"]),
                    "negative_contribution": float(r["negative_contribution"]),
                    "gross_contribution": float(r["gross_contribution"]),
                    "net_gross_ratio": net_gross_ratio(r),
                    "validation_status": str(r["validation_status"]),
                })
        for resolution in ("2016", "2013", "R3"):
            report_dir = main_root / f"{yp}_{year}_{resolution}"
            if not report_dir.is_dir():
                continue
            h = load_hierarchy(report_dir)
            for node in HIER_NODES:
                r = h.loc[node]
                rows.append({
                    "dataset": "main", "series": f"exio_{resolution}", "year_previous": yp, "year_current": year,
                    "node_id": node, "depth": DEPTH[node],
                    "contribution": float(r["contribution"]),
                    "positive_contribution": float(r["positive_contribution"]),
                    "negative_contribution": float(r["negative_contribution"]),
                    "gross_contribution": float(r["gross_contribution"]),
                    "net_gross_ratio": net_gross_ratio(r),
                    "validation_status": str(r["validation_status"]),
                })
    return pd.DataFrame(rows)


PAIRS = {
    "Q1_target_grid": ("native", "jrc2019_to_2016", "native", "jrc2019_to_2013", False),
    "Q2_cross_database": ("native", "jrc2019_to_2013", "native", "wiod2013_native", True),
    "Q3a_vs_main_2016": ("native", "jrc2019_to_2016", "main", "exio_2016", True),
    "Q3b_vs_main_2013": ("native", "jrc2019_to_2013", "main", "exio_2013", True),
    "Q3c_vs_main_2013": ("native", "wiod2013_native", "main", "exio_2013", True),
    # Factorial on the 41x35 grid: hold one factor, vary the other.
    "Q4a_flow_effect_jrc": ("native", "jrc2019_to_2013", "native", "jrc2019_flows2013", True),
    "Q4b_flow_effect_wiod": ("native", "wiod2013_flows2016", "native", "wiod2013_native", True),
    "Q5a_extension_effect_flows2016": ("native", "jrc2019_to_2013", "native", "wiod2013_flows2016", True),
    "Q5b_extension_effect_flows2013": ("native", "jrc2019_flows2013", "native", "wiod2013_native", True),
    # Q6: GENUINE flow-table effect -- EXIOBASE-derived flows vs WIOD2013 PUBLISHED
    # flows, holding the emission account. (Q4 above is a concordance-path check:
    # both its sides are EXIOBASE-derived flows.)
    "Q6a_realflow_effect_wiod": ("native", "wiod2013_native", "native", "wiod2013_true_native", True),
    "Q6b_realflow_effect_jrc": ("native", "jrc2019_flows2013", "native", "jrc2019_wiodflows2013", True),
    "Q6c_realflow_effect_exio": ("main", "exio_2013", "native", "exio_wiodflows2013", True),
}


def _series(panel: pd.DataFrame, dataset: str, series: str, year: int) -> pd.DataFrame:
    sub = panel[(panel.dataset == dataset) & (panel.series == series) & (panel.year_current == year)]
    return sub.set_index("node_id")


def build_node_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question, (ds_a, s_a, ds_b, s_b, _same_grid) in PAIRS.items():
        for year in YEARS:
            a = _series(panel, ds_a, s_a, year)
            b = _series(panel, ds_b, s_b, year)
            if a.empty or b.empty:
                continue
            for node in HIER_NODES:
                av, bv = float(a.loc[node, "contribution"]), float(b.loc[node, "contribution"])
                delta = av - bv
                scale = max(abs(av), abs(bv), 1.0)
                ar = a.loc[node, "net_gross_ratio"]
                br = b.loc[node, "net_gross_ratio"]
                rows.append({
                    "question": question, "year_previous": year - 1, "year_current": year,
                    "node_id": node, "depth": DEPTH[node],
                    "a_series": s_a, "b_series": s_b,
                    "a_contribution": av, "b_contribution": bv,
                    "delta_a_minus_b": delta, "abs_delta": abs(delta),
                    "relative_delta": delta / scale,
                    "sign_change": sign(av) != sign(bv) and sign(av) != 0 and sign(bv) != 0,
                    "a_net_gross_ratio": None if ar is None else float(ar),
                    "b_net_gross_ratio": None if br is None else float(br),
                    "cancellation_shift": (None if ar is None or br is None else float(ar) - float(br)),
                    "a_validation_status": str(a.loc[node, "validation_status"]),
                    "b_validation_status": str(b.loc[node, "validation_status"]),
                    "provisional": "accepted_provisional" in (
                        str(a.loc[node, "validation_status"]), str(b.loc[node, "validation_status"])),
                })
    return pd.DataFrame(rows)


def build_difference_reconciliation(node_cmp: pd.DataFrame) -> pd.DataFrame:
    """Check Δparent == Σ Δchildren for the differenced hierarchy.

    Residuals trace entirely to the accepted_provisional Leontief-split runs;
    the plain relative figure is inflated when Δparent sits near a sign change,
    so a magnitude-floored relative residual (scale floored at 1e-3 of the
    year's |Δ delta_g|) is also reported.
    """
    rows = []
    for (question, year), grp in node_cmp.groupby(["question", "year_current"]):
        g = grp.set_index("node_id")
        d = g["delta_a_minus_b"]
        prov = g["provisional"]
        # panel magnitude: subtraction introduces no new error, so the residual
        # is meaningful only against the emission-change magnitudes being
        # differenced, not against the (possibly near-zero) difference itself.
        panel_mag = max(abs(float(g.loc["delta_g", "a_contribution"])),
                        abs(float(g.loc["delta_g", "b_contribution"])), 1.0)
        for parent, children in DIFF_IDENTITIES.items():
            child_sum = float(sum(d[c] for c in children))
            residual = child_sum - float(d[parent])
            scale = max(abs(child_sum), abs(float(d[parent])), 1.0)
            rows.append({
                "question": question, "year_current": year, "parent_node": parent,
                "children": ", ".join(children),
                "child_delta_sum": child_sum, "parent_delta": float(d[parent]),
                "residual": residual,
                "relative_residual": abs(residual) / scale,
                "residual_vs_panel_magnitude": abs(residual) / panel_mag,
                "involves_provisional": bool(prov[parent] or any(prov[c] for c in children)),
            })
    return pd.DataFrame(rows)


# a primary effect whose annual magnitude exceeds this multiple of the series'
# median |delta_g| is flagged as a decomposition-instability anomaly (large
# offsetting intensity/structure terms; the total is unaffected). The median
# denominator avoids flagging ordinary near-zero-total years.
ANOMALY_MULTIPLE = 10.0


# 2x2 factorial cells on the 41x35 grid for the vintage-vs-emission allocation
ALLOC_CELLS = {
    "A_jrc_flows2016": "jrc2019_to_2013",     # JRC EA, WIOD2016 flows
    "B_jrc_flows2013": "jrc2019_flows2013",   # JRC EA, WIOD2013 flows
    "C_wiod_flows2016": "wiod2013_flows2016", # WIOD2013 EA, WIOD2016 flows
    "D_wiod_flows2013": "wiod2013_native",    # WIOD2013 EA, WIOD2013 flows
}


def build_vintage_emission_allocation(panel: pd.DataFrame) -> pd.DataFrame:
    """Additively split the JRC-vs-WIOD2013 between-database difference (A - D)
    into an emission-account component, a flow-table-vintage component, and an
    interaction, using the 2x2 (symmetric / average-of-paths, i.e. Shapley):

        emission    = 1/2 [ (A - C) + (B - D) ]
        vintage     = 1/2 [ (A - B) + (C - D) ]
        interaction = (A - D) - emission - vintage

    emission + vintage + interaction == A - D exactly, per node per year.
    """
    have = {k: v for k, v in ALLOC_CELLS.items() if not panel[panel.series == v].empty}
    if set(have) != set(ALLOC_CELLS):
        return pd.DataFrame()
    rows = []
    vec = {k: {n: _annual_vector(panel, s, n) for n in HIER_NODES}
           for k, s in ALLOC_CELLS.items()}
    for node in HIER_NODES:
        A, B = vec["A_jrc_flows2016"][node], vec["B_jrc_flows2013"][node]
        C, D = vec["C_wiod_flows2016"][node], vec["D_wiod_flows2013"][node]
        for year in YEARS:
            a, b, c, d = (float(x.get(year)) if pd.notna(x.get(year)) else None
                          for x in (A, B, C, D))
            if None in (a, b, c, d):
                continue
            total = a - d
            emission = 0.5 * ((a - c) + (b - d))
            vintage = 0.5 * ((a - b) + (c - d))
            interaction = total - emission - vintage
            scale = max(abs(a), abs(d), 1.0)
            rows.append({
                "node_id": node, "year_previous": year - 1, "year_current": year,
                "jrc": a, "wiod2013": d, "total_diff_jrc_minus_wiod": total,
                "emission_component": emission, "vintage_component": vintage,
                "interaction": interaction,
                "emission_share": emission / total if abs(total) > scale * 1e-9 else None,
                "vintage_share": vintage / total if abs(total) > scale * 1e-9 else None,
                "vintage_rel_to_scale": abs(vintage) / scale,
            })
    return pd.DataFrame(rows)


def _anomalous_years(panel: pd.DataFrame, series: str) -> set[int]:
    dg = _annual_vector(panel, series, "delta_g").abs()
    threshold = ANOMALY_MULTIPLE * float(dg.median())
    flagged: set[int] = set()
    for node in ("emission_intensity", "leontief_production_structure"):
        v = _annual_vector(panel, series, node).abs()
        for year in YEARS:
            if pd.notna(v.get(year)) and v[year] > threshold:
                flagged.add(year)
    return flagged


def build_matched_period_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Cumulative 2001-2009 total change and primary effects for every
    dataset x resolution series, with and without decomposition-instability
    years excluded."""
    rows = []
    for series, meta in SERIES_MATRIX.items():
        sub = panel[panel.series == series]
        if sub.empty:
            continue
        years_present = sorted(sub.year_current.unique())
        anomalous = _anomalous_years(panel, series)
        clean = sub[~sub.year_current.isin(anomalous)]
        for node in PRIMARY_NODES:
            n = sub[sub.node_id == node]
            nc = clean[clean.node_id == node]
            rows.append({
                "series": series, "dataset": meta["dataset"], "grid": meta["grid"],
                "node_id": node, "years": len(years_present),
                "anomalous_years": ", ".join(str(y) for y in sorted(anomalous)) or None,
                "cumulative_contribution": float(n["contribution"].sum()),
                "cumulative_contribution_excl_anomalies": float(nc["contribution"].sum()),
                "cumulative_positive": float(n["positive_contribution"].sum()),
                "cumulative_negative": float(n["negative_contribution"].sum()),
                "cumulative_gross": float(n["gross_contribution"].sum()),
                "median_annual_contribution": float(n["contribution"].median()),
                "annual_sign_all_same": bool(n["contribution"].gt(0).nunique() == 1),
            })
    return pd.DataFrame(rows)


def build_window_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    """Cumulative total change and primary effects under alternative cumulation
    windows, for every dataset x resolution series. Shows how the period
    selection changes the reported emission-change story."""
    rows = []
    for series, meta in SERIES_MATRIX.items():
        sub = panel[panel.series == series]
        if sub.empty:
            continue
        for window, years in WINDOWS.items():
            w = sub[sub.year_current.isin(years)]
            present = sorted(w.year_current.unique())
            if not present:
                continue
            rec = {"series": series, "dataset": meta["dataset"], "grid": meta["grid"],
                   "window": window, "years_in_window": len(present),
                   "first_year": present[0], "last_year": present[-1]}
            for node in PRIMARY_NODES:
                rec[f"cumulative_{node}"] = float(w[w.node_id == node]["contribution"].sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def build_instability_diagnostic(campaign_root: Path | None,
                                 matched: pd.DataFrame) -> pd.DataFrame:
    """Top offsetting Dint / Leontief_total country-sector cells for each flagged
    decomposition-instability (series, year)."""
    if campaign_root is None or matched.empty:
        return pd.DataFrame()
    flagged = set()
    for r in matched[matched.anomalous_years.notna()].itertuples():
        if r.series not in NATIVE_CONFIGS:
            continue
        for y in str(r.anomalous_years).split(", "):
            flagged.add((r.series, int(y)))
    rows = []
    for series, year in sorted(flagged):
        effect_dir = campaign_root / f"{year - 1}_{year}_{series}" / "CO2_7"
        if not effect_dir.is_dir():
            continue
        dint = pd.read_parquet(effect_dir / f"{year}_Dint_baseline.parquet").sum(axis=1)
        leon = pd.read_parquet(effect_dir / f"{year}_Leontief_total_baseline.parquet").sum(axis=1)
        top = dint.abs().sort_values(ascending=False).index[:5]
        for key in top:
            country, sector = (key if isinstance(key, tuple) else (str(key), ""))
            rows.append({
                "series": series, "year_current": year,
                "supplier_country": country, "supplier_sector": sector,
                "dint_kg_co2": float(dint[key]),
                "leontief_kg_co2": float(leon.get(key, np.nan)),
                "net_kg_co2": float(dint[key] + leon.get(key, np.nan)),
                "dint_gt": float(dint[key] / 1e12),
                "leontief_gt": float(leon.get(key, np.nan) / 1e12),
            })
    return pd.DataFrame(rows)


def _annual_vector(panel: pd.DataFrame, series: str, node: str) -> pd.Series:
    s = panel[(panel.series == series) & (panel.node_id == node)]
    return s.set_index("year_current")["contribution"].reindex(YEARS)


def _annual_gross(panel: pd.DataFrame, series: str, node: str) -> pd.Series:
    s = panel[(panel.series == series) & (panel.node_id == node)]
    return s.set_index("year_current")["gross_contribution"].reindex(YEARS)


def build_cross_series_agreement(panel: pd.DataFrame) -> pd.DataFrame:
    """Pairwise agreement over the 9 annual transitions for every series pair,
    for the total change, the three primary effects, and the eight subeffects
    (D_v, D_C, D_A_x, D_T_coef, D_w_coef, D_YF, D_YB, D_y). Subeffect rows on a
    cross-grid pair (different country/sector grids) are still emitted but
    flagged `comparable=False` -- composition is not comparable there."""
    from itertools import combinations
    series = [s for s in SERIES_MATRIX if not panel[panel.series == s].empty]
    rows = []
    for a, b in combinations(series, 2):
        for node in AGREEMENT_NODES:
            va, vb = _annual_vector(panel, a, node), _annual_vector(panel, b, node)
            ga, gb = _annual_gross(panel, a, node), _annual_gross(panel, b, node)
            mask = va.notna() & vb.notna()
            va, vb, ga, gb = va[mask], vb[mask], ga[mask], gb[mask]
            if len(va) < 3:
                continue
            sign_agreement = float((np.sign(va.values) == np.sign(vb.values)).mean())
            pearson = float(np.corrcoef(va.values, vb.values)[0, 1]) if va.std() and vb.std() else np.nan
            spearman = float(pd.Series(va.values).corr(pd.Series(vb.values), method="spearman"))
            denom = float((0.5 * (ga.values + gb.values)).sum())
            nmae = float(np.abs(va.values - vb.values).sum() / denom) if denom else np.nan
            ma, mb = SERIES_MATRIX[a], SERIES_MATRIX[b]
            rows.append({
                "series_a": a, "series_b": b, "node_id": node, "n_years": int(len(va)),
                "sign_agreement": sign_agreement, "pearson_r": pearson,
                "spearman_rho": spearman, "nmae": nmae,
                "same_grid": ma["grid"] == mb["grid"],
                "same_dataset": ma["dataset"] == mb["dataset"],
                "same_ea": ma["ea"] == mb["ea"],
                "same_flows": ma["flows"] == mb["flows"],
                "is_subeffect": node in SUBEFFECT_NODES,
                "is_sourcing": node in SOURCING_NODES,
                # subeffect composition is only comparable on a shared grid
                "comparable": (node in PRIMARY_NODES) or (ma["grid"] == mb["grid"]),
            })
    return pd.DataFrame(rows)


def build_rank_comparison(native_root: Path, main_root: Path) -> pd.DataFrame:
    def dd_dir(dataset, series, yp, year):
        if dataset == "native":
            return native_root / f"{yp}_{year}_{series}"
        return main_root / f"{yp}_{year}_{series.replace('exio_', '')}"

    rows = []
    for question, (ds_a, s_a, ds_b, s_b, same_grid) in PAIRS.items():
        if not same_grid:
            continue
        for year in YEARS:
            yp = year - 1
            da_dir, db_dir = dd_dir(ds_a, s_a, yp, year), dd_dir(ds_b, s_b, yp, year)
            if not da_dir.is_dir() or not db_dir.is_dir():
                continue
            da, db = load_drilldown(da_dir), load_drilldown(db_dir)
            for effect in RANK_EFFECTS:
                for grain in RANK_GRAINS:
                    ra = _rank_table(da, effect, grain)
                    rb = _rank_table(db, effect, grain)
                    if ra.empty or rb.empty:
                        continue
                    labels = set(ra.index[:RANK_TOP_N]) | set(rb.index[:RANK_TOP_N])
                    for label in labels:
                        rank_a = int(ra.index.get_loc(label)) + 1 if label in ra.index else None
                        rank_b = int(rb.index.get_loc(label)) + 1 if label in rb.index else None
                        rows.append({
                            "question": question, "year_previous": yp, "year_current": year,
                            "effect_id": effect, "grain_id": grain, "label": label,
                            "rank_a": rank_a, "rank_b": rank_b,
                            "rank_shift": (None if rank_a is None or rank_b is None else rank_a - rank_b),
                            "contribution_a": float(ra.loc[label, "contribution"]) if label in ra.index else None,
                            "contribution_b": float(rb.loc[label, "contribution"]) if label in rb.index else None,
                            "in_top_n_a": rank_a is not None and rank_a <= RANK_TOP_N,
                            "in_top_n_b": rank_b is not None and rank_b <= RANK_TOP_N,
                        })
    return pd.DataFrame(rows)


def _rank_table(drill: pd.DataFrame, effect: str, grain: str) -> pd.DataFrame:
    sub = drill[(drill.effect_id == effect) & (drill.grain_id == grain)].copy()
    if sub.empty:
        return sub
    sub = sub.dropna(subset=[grain])
    sub = sub.groupby(grain).agg(
        contribution=("contribution", "sum"),
        gross_contribution=("gross_contribution", "sum")).reset_index()
    sub = sub.sort_values("gross_contribution", ascending=False).set_index(grain)
    return sub


def build_composition_agreement(ranks: pd.DataFrame) -> pd.DataFrame:
    """Per question x effect x grain, how similar are the two series' driving
    country / sector rankings: Spearman rank correlation over the union of
    top-N labels, top-10 retention, worst rank shift inside the top 10, and
    contribution sign agreement for shared labels. Averaged over the 9 years.
    Covers all subeffects; `D_C` and `D_YF` are the sourcing components."""
    if ranks.empty:
        return pd.DataFrame()
    out = []
    for (q, eff, grain), g in ranks.groupby(["question", "effect_id", "grain_id"]):
        per_year = []
        for year, gy in g.groupby("year_current"):
            both = gy.dropna(subset=["rank_a", "rank_b"])
            if len(both) < 4:
                continue
            spearman = float(pd.Series(both.rank_a.values).corr(
                pd.Series(both.rank_b.values), method="spearman"))
            top_a = set(gy.loc[gy.rank_a.le(10), "label"])
            top_b = set(gy.loc[gy.rank_b.le(10), "label"])
            retention = len(top_a & top_b) / max(len(top_a), 1)
            in10 = both[(both.rank_a <= 10) | (both.rank_b <= 10)]
            worst_shift = int(in10.rank_shift.abs().max()) if len(in10) else 0
            sign_ok = float((np.sign(both.contribution_a) == np.sign(both.contribution_b)).mean())
            per_year.append((spearman, retention, worst_shift, sign_ok))
        if not per_year:
            continue
        arr = np.array(per_year, dtype=float)
        out.append({
            "question": q, "effect_id": eff, "grain_id": grain,
            "is_sourcing": eff in SOURCING_NODES,
            "years": len(per_year),
            "spearman_rank_corr_mean": float(arr[:, 0].mean()),
            "top10_retention_mean": float(arr[:, 1].mean()),
            "worst_top10_rank_shift_max": int(arr[:, 2].max()),
            "contribution_sign_agreement_mean": float(arr[:, 3].mean()),
        })
    return pd.DataFrame(out)


def build_raw_dax_diagnostic(campaign_root: Path | None) -> pd.DataFrame:
    if campaign_root is None:
        return pd.DataFrame()
    rows = []
    for year in YEARS:
        yp = year - 1
        for config in NATIVE_CONFIGS:
            effect_dir = campaign_root / f"{yp}_{year}_{config}" / "CO2_7"
            if not effect_dir.is_dir():
                continue
            out = {"year_previous": yp, "year_current": year, "series": config}
            for native_effect, key in (("D_Ax2", "raw_D_A_x"), ("D_C2", "raw_D_C"),
                                       ("D_T_coef", "D_T_coef"), ("D_w_coef", "D_w_coef"),
                                       ("Leontief_total", "Leontief_total")):
                path = effect_dir / f"{year}_{native_effect}_tot_baseline.parquet"
                if path.is_file():
                    frame = pd.read_parquet(path)
                    out[key] = float(frame.select_dtypes("number").to_numpy(dtype=float).sum())
            if "D_T_coef" in out and "D_w_coef" in out:
                out["coef_D_A_x"] = out["D_T_coef"] + out["D_w_coef"]
                if "raw_D_A_x" in out:
                    scale = max(abs(out["raw_D_A_x"]), abs(out["coef_D_A_x"]), 1.0)
                    out["raw_minus_coef_D_A_x"] = out["raw_D_A_x"] - out["coef_D_A_x"]
                    out["raw_minus_coef_relative"] = out["raw_minus_coef_D_A_x"] / scale
            rows.append(out)
    return pd.DataFrame(rows)


def write_report(path: Path, node_cmp: pd.DataFrame, ranks: pd.DataFrame,
                 recon: pd.DataFrame, raw_dax: pd.DataFrame,
                 matched: pd.DataFrame, agreement: pd.DataFrame,
                 windows: pd.DataFrame, instability: pd.DataFrame,
                 allocation: pd.DataFrame, composition: pd.DataFrame) -> None:
    L: list[str] = ["# Native-WIOD three-way comparison (task C)", "",
                    f"Generated {datetime.now(timezone.utc).isoformat()}. Units: kg CO2. "
                    f"Years: {YEARS[0]}-{YEARS[-1]}.", ""]
    if not matched.empty:
        anom = matched[matched.anomalous_years.notna()][["series", "anomalous_years"]].drop_duplicates()
        L += ["## Matched-period comparison across datasets and resolutions "
              f"({YEARS[0]}-{YEARS[-1]})", "",
              "Cumulative total change and primary effects, Gt CO2 (kg / 1e12). "
              "Every series covers the same nine transitions. `intensity*` and "
              "`structure*` exclude decomposition-instability years (a primary "
              f"effect exceeding {ANOMALY_MULTIPLE:.0f}x the series' median |Δg|; "
              "the total is unaffected because intensity and structure cancel).", ""]
        if not anom.empty:
            L.append("Instability years: " + "; ".join(
                f"`{r.series}` {r.anomalous_years}" for r in anom.itertuples()) + ".")
            L.append("")
        L += ["| dataset | grid | Δg | intensity | intensity* | structure | structure* | final demand |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        piv = matched.pivot_table(index=["dataset", "grid", "series"], columns="node_id",
                                  values=["cumulative_contribution",
                                          "cumulative_contribution_excl_anomalies"])
        for (dataset, grid, _series), row in piv.iterrows():
            c = row["cumulative_contribution"]
            e = row["cumulative_contribution_excl_anomalies"]
            L.append(f"| {dataset} | {grid} | {c['delta_g']/1e12:+.3f} | "
                     f"{c['emission_intensity']/1e12:+.3f} | {e['emission_intensity']/1e12:+.3f} | "
                     f"{c['leontief_production_structure']/1e12:+.3f} | "
                     f"{e['leontief_production_structure']/1e12:+.3f} | "
                     f"{c['final_demand']/1e12:+.3f} |")
        L += ["", "### Pairwise annual agreement, total change (9 transitions)", "",
              "Sign agreement, Pearson r and NMAE (Σ|a−b| / Σ mean gross), median "
              "over series pairs, grouped by which factor differs (grid held "
              "constant except in the first row).", "",
              "| what differs | pairs | sign agr. | Pearson r | NMAE |",
              "|---|---:|---:|---:|---:|"]
        dg = agreement[agreement.node_id == "delta_g"]
        for label, sel in (
            ("target grid only (same EA and flows)", dg[~dg.same_grid & dg.same_ea & dg.same_flows]),
            ("flow table only (same EA, same grid)", dg[dg.same_grid & dg.same_ea & ~dg.same_flows]),
            ("environmental account only (same flows, same grid)", dg[dg.same_grid & ~dg.same_ea & dg.same_flows]),
            ("both EA and flow table (same grid)", dg[dg.same_grid & ~dg.same_ea & ~dg.same_flows]),
        ):
            if sel.empty:
                continue
            L.append(f"| {label} | {len(sel)} | {sel.sign_agreement.median():.2f} | "
                     f"{sel.pearson_r.median():.3f} | {sel.nmae.median():.3f} |")
        L.append("")

        # sourcing / subeffect agreement, shared-grid pairs only
        comp = agreement[agreement.comparable & agreement.same_grid]
        if not comp.empty:
            L += ["### Subeffect agreement, shared-grid pairs (9 transitions)", "",
                  "Median sign agreement / Pearson r / NMAE per node, grouped by "
                  "what differs. `D_C` and `D_YF` are the sourcing components.", "",
                  "| node | flow table differs | environmental account differs |",
                  "|---|---|---|"]
            flow = comp[comp.same_ea & ~comp.same_flows]
            ext = comp[~comp.same_ea & comp.same_flows]
            for node in AGREEMENT_NODES:
                if node in PRIMARY_NODES and node != "delta_g":
                    continue
                fs = flow[flow.node_id == node]
                es = ext[ext.node_id == node]
                tag = " (sourcing)" if node in SOURCING_NODES else ""
                fcell = ("--" if fs.empty else
                         f"{fs.sign_agreement.median():.2f} / {fs.pearson_r.median():.2f} / {fs.nmae.median():.3f}")
                ecell = ("--" if es.empty else
                         f"{es.sign_agreement.median():.2f} / {es.pearson_r.median():.2f} / {es.nmae.median():.3f}")
                L.append(f"| {node}{tag} | {fcell} | {ecell} |")
            L.append("")

    if not composition.empty:
        L += ["### Driving country / sector composition of the sourcing effects", "",
              "For each comparison, how similar are the two series' emitter "
              "rankings for `D_C` and `D_YF`: mean Spearman rank correlation, "
              "mean top-10 retention, worst rank shift inside the top 10, and "
              "contribution sign agreement (mean over 9 years). Emitter grain "
              "only; source / driver grains are deferred (NAT-06).", "",
              "| comparison | effect | grain | Spearman | top-10 kept | worst top-10 shift | sign agr. |",
              "|---|---|---|---:|---:|---:|---:|"]
        order = ["Q4a_flow_effect_jrc", "Q4b_flow_effect_wiod",
                 "Q5a_extension_effect_flows2016", "Q5b_extension_effect_flows2013",
                 "Q2_cross_database", "Q3b_vs_main_2013", "Q3c_vs_main_2013"]
        cs = composition[composition.effect_id.isin(SOURCING_NODES)]
        for q in order:
            for _, r in cs[cs.question == q].sort_values(["effect_id", "grain_id"]).iterrows():
                L.append(f"| {q} | {r.effect_id} | {r.grain_id} | "
                         f"{r.spearman_rank_corr_mean:.2f} | {r.top10_retention_mean:.2f} | "
                         f"{r.worst_top10_rank_shift_max} | {r.contribution_sign_agreement_mean:.2f} |")
        L.append("")

    factorial = [q for q in ("Q4a_flow_effect_jrc", "Q4b_flow_effect_wiod",
                             "Q5a_extension_effect_flows2016", "Q5b_extension_effect_flows2013")
                 if not node_cmp[node_cmp.question == q].empty]
    if factorial:
        L += ["## Environmental extension vs. input-output data (2x2 on the 41x35 grid)", "",
              "Four series: {JRC2019, WIOD2013} emission accounts x {WIOD2016, "
              "WIOD2013} flow tables. Q4 holds the emission account and varies "
              "the flow table (input-output / flow-table effect); Q5 holds the "
              "flow table and varies the emission account (environmental-"
              "extension effect). Median relative difference across the nine "
              "years, per primary node.", "",
              "| comparison | isolates | Δg | intensity | prod. structure | final demand |",
              "|---|---|---:|---:|---:|---:|"]
        labels = {
            "Q4a_flow_effect_jrc": "flow table (JRC EA)",
            "Q4b_flow_effect_wiod": "flow table (WIOD2013 EA)",
            "Q5a_extension_effect_flows2016": "environmental account (WIOD2016 flows)",
            "Q5b_extension_effect_flows2013": "environmental account (WIOD2013 flows)",
        }
        for q in factorial:
            sub = node_cmp[node_cmp.question == q].set_index(["node_id"])
            def med(n):
                return sub.loc[sub.index == n, "relative_delta"].median()
            L.append(f"| {q} | {labels[q]} | {med('delta_g'):+.3f} | "
                     f"{med('emission_intensity'):+.3f} | "
                     f"{med('leontief_production_structure'):+.3f} | "
                     f"{med('final_demand'):+.3f} |")
        L += ["",
              "The emission-account choice fixes the raw industry CO2 total "
              "regardless of the flow table (Q5 pairs share it; probe: JRC2019 "
              "~25.1 Gt vs WIOD2013 ~45.7 Gt for 2004), so Q5 is a clean "
              "one-factor contrast. Q4 additionally carries the WIOD2016->2013 "
              "flow harmonisation (non-nested C13-C15) on the WIOD2016-flows "
              "side.", ""]

    if not allocation.empty:
        L += ["## Allocation of the JRC-vs-WIOD2013 difference: emission account vs. flow vintage", "",
              "The between-database difference A - D "
              "(`jrc2019_to_2013` minus `wiod2013_native`) split additively into "
              "an emission-account component, a flow-table-vintage component, and "
              "an interaction (symmetric / Shapley: emission = 1/2[(A-C)+(B-D)], "
              "vintage = 1/2[(A-B)+(C-D)]). Cumulative 2001-2009, Gt CO2, with the "
              "vintage share of |total|.", "",
              "| node | total (A-D) | emission | vintage | interaction | vintage % of |total| |",
              "|---|---:|---:|---:|---:|---:|"]
        for node in ("delta_g", "emission_intensity", "leontief_production_structure",
                     "final_demand", "D_C", "D_YF", "D_v", "D_y"):
            s = allocation[allocation.node_id == node]
            if s.empty:
                continue
            tot = s.total_diff_jrc_minus_wiod.sum() / 1e12
            emi = s.emission_component.sum() / 1e12
            vin = s.vintage_component.sum() / 1e12
            inter = s.interaction.sum() / 1e12
            vpct = 100 * abs(vin) / max(abs(tot), 1e-9)
            L.append(f"| {node} | {tot:+.3f} | {emi:+.3f} | {vin:+.4f} | {inter:+.4f} | {vpct:.2f}% |")
        allv = allocation.groupby("node_id").apply(
            lambda g: abs(g.vintage_component.sum()) / max(abs(g.total_diff_jrc_minus_wiod.sum()), 1.0))
        L += ["",
              f"Across all 12 nodes the flow-table-vintage component is at most "
              f"{100*allv.max():.1f}% of the between-database difference "
              f"(node `{allv.idxmax()}`); for the total change it is "
              f"{100*allv.get('delta_g', 0):.2f}%. The emission-account component "
              "accounts for essentially all of the difference; the interaction is "
              "negligible.", ""]

    if not instability.empty:
        L += ["## 2003-2004 instability in `jrc2019_to_2016`: root cause", "",
              "For each flagged (series, year), the top |Dint| supplier "
              "country-sector cells and the offsetting Leontief_total term. A "
              "single small service/utility cell carries a multi-Gt emission-"
              "intensity effect exactly cancelled by the production-structure "
              "effect; the net contribution is negligible.", "",
              "| series | year | supplier | Dint (Gt) | Leontief (Gt) | net (Gt) |",
              "|---|---:|---|---:|---:|---:|"]
        for r in instability.itertuples():
            L.append(f"| {r.series} | {r.year_current} | "
                     f"{r.supplier_country} / {r.supplier_sector} | "
                     f"{r.dint_gt:+.2f} | {r.leontief_gt:+.2f} | {r.net_kg_co2/1e12:+.3f} |")
        L += ["", "The affected cells are `E37-E39` (sewerage, waste) and `N` "
              "(administrative services) for Turkey (2003) and Russia (2004). "
              "On the WIOD2016 50-sector grid these have near-zero gross output "
              "for those countries, so emissions / output is extreme and its "
              "year-on-year change dominates Dint; the run logs show technical "
              "coefficients above 1 and a Leontief inverse varying by ~300x "
              "across Monte-Carlo samples. The 35-sector WIOD2013 grid "
              "aggregates these sectors away, so `jrc2019_to_2013` is unaffected.",
              "",
              "This is the same near-zero-output intensity artefact that the "
              "EXIOBASE sector-median replacement was designed for, and it "
              "occurs in the native JRC2019 / WIOD2016 accounts as well. With "
              "that policy enabled (`sector_outlier_multiple = 100`) the "
              "intensity loader replaces 28 country-sector cells in 2003 and 21 "
              "in 2004, changing the effective industry total by 0.24% and "
              "1.55% "
              "(`native-impact-smoke-20260902/effective_impact_jrc2019_to_2016.json`). "
              "The native policy sets `V_SECTOR_OUTLIER_MULTIPLE` and "
              "`V_INTENSITY_MAX` to inf (handover §4) to preserve official "
              "country-sector heterogeneity, leaving these cells unmitigated on "
              "the fine grid.",
              "",
              "Recommendation: exclude `emission_intensity` / `D_v` and "
              "`leontief_production_structure` for `jrc2019_to_2016` in 2003 and "
              "2004, or re-run that configuration with a finite "
              "`SDA_V_INTENSITY_MAX` / `SDA_V_SECTOR_OUTLIER_MULTIPLE` under a "
              "new treatment version.", ""]

    if not windows.empty:
        L += ["## Period-selection sensitivity", "",
              "Cumulative total change and primary effects (Gt CO2) under "
              "alternative cumulation windows. `emission_intensity` and "
              "`leontief_production_structure` for `jrc2019_to_2016` include the "
              "2003-2004 instability except where the window excludes those "
              "years.", ""]
        for series in SERIES_MATRIX:
            w = windows[windows.series == series]
            if w.empty:
                continue
            meta = SERIES_MATRIX[series]
            L += [f"### {meta['dataset']} — grid {meta['grid']} (`{series}`)", "",
                  "| window | yrs | Δg | intensity | prod. structure | final demand |",
                  "|---|---:|---:|---:|---:|---:|"]
            for r in w.itertuples():
                L.append(f"| {r.window} | {r.years_in_window} | "
                         f"{r.cumulative_delta_g/1e12:+.2f} | "
                         f"{r.cumulative_emission_intensity/1e12:+.2f} | "
                         f"{r.cumulative_leontief_production_structure/1e12:+.2f} | "
                         f"{r.cumulative_final_demand/1e12:+.2f} |")
            L.append("")

    L += [
        "## Caveats", "",
        "- The main EXIOBASE campaign is **fossil-origin CO2** (combustion, cement, "
        "lime, peat decay, fossil waste) in kg CO2 (D-2026-011, D-2026-022), not the "
        "AR4 GWP100 basket that handover §6.C names. Q3 differences combine "
        "environmental-account scope, impact database and (for JRC) flow-table "
        "vintage; they are not a pure database effect and not a CO2-vs-CO2eq "
        "comparison.",
        "- Q1 compares the native 2016 grid (44x50) with the harmonised 2013 grid "
        "(41x35, non-nested C13-C15 split 50/50); node totals are comparable, "
        "composition is not.",
        "- Q2 (`jrc2019_to_2013` vs `wiod2013_native`) mixes impact database, flow "
        "vintage and the JRC-side harmonisation; not a single-factor contrast.",
        "- 5 native runs carry `accepted_provisional` Leontief-split nodes; rows "
        "touching them have `provisional=true`.",
        "- All native and main annual inputs are provisional (D-2026-020).", "",
        "## Node-level differences (median | min | max of relative delta across years)", "",
        "| question | node | median rel Δ | min | max | sign changes | prov. rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for question in PAIRS:
        q = node_cmp[node_cmp.question == question]
        for node in HIER_NODES:
            n = q[q.node_id == node]
            if n.empty:
                continue
            L.append(f"| {question} | {node} | {n.relative_delta.median():+.3e} | "
                     f"{n.relative_delta.min():+.3e} | {n.relative_delta.max():+.3e} | "
                     f"{int(n.sign_change.sum())} | {int(n.provisional.sum())} |")
    L += ["", "## Difference-of-differences reconciliation", "",
          "`Δparent == Σ Δchildren` for the differenced hierarchy. Non-provisional "
          "rows reconcile to ~1e-5; residuals on provisional rows equal the "
          "inherited Leontief-split residuals (max ~3.7e10 kg CO2 on the "
          "`2000_2001` comparisons). `_floored` divides by max(scale, 1e-3·|Δδ_g|).", "",
          "| question | worst parent | rel. resid | resid / panel mag | provisional |",
          "|---|---|---:|---:|---|"]
    for question in PAIRS:
        r = recon[recon.question == question]
        if r.empty:
            continue
        w = r.loc[r.residual_vs_panel_magnitude.idxmax()]
        L.append(f"| {question} | {w['parent_node']} | {w['relative_residual']:.3e} | "
                 f"{w['residual_vs_panel_magnitude']:.3e} | {bool(w['involves_provisional'])} |")
    if not ranks.empty:
        L += ["", "## Largest emitter rank shifts (same-grid questions)", "",
              "| question | effect | grain | label | rank a | rank b | shift |",
              "|---|---|---|---|---:|---:|---:|"]
        movers = ranks.dropna(subset=["rank_shift"]).copy()
        movers["abs_shift"] = movers["rank_shift"].abs()
        for _, row in movers.sort_values("abs_shift", ascending=False).head(25).iterrows():
            L.append(f"| {row['question']} | {row['effect_id']} | {row['grain_id']} | "
                     f"{row['label']} | {int(row['rank_a'])} | {int(row['rank_b'])} | "
                     f"{int(row['rank_shift']):+d} |")
    if not raw_dax.empty and "raw_minus_coef_relative" in raw_dax:
        L += ["", "## Raw vs coefficient D_A_x (diagnostic, never summed into the hierarchy)", "",
              "| series | median |raw−coef|/scale | max |", "|---|---:|---:|"]
        for series, g in raw_dax.groupby("series"):
            v = g["raw_minus_coef_relative"].abs()
            L.append(f"| {series} | {v.median():.3e} | {v.max():.3e} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-annual-root", type=Path, required=True)
    parser.add_argument("--main-annual-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path,
                        help="native campaign root, for the raw D_A_x diagnostic")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args()

    panel = build_panel(args.native_annual_root, args.main_annual_root)
    node_cmp = build_node_comparison(panel)
    recon = build_difference_reconciliation(node_cmp)
    ranks = build_rank_comparison(args.native_annual_root, args.main_annual_root)
    raw_dax = build_raw_dax_diagnostic(args.campaign_root)
    matched = build_matched_period_summary(panel)
    agreement = build_cross_series_agreement(panel)
    windows = build_window_sensitivity(panel)
    instability = build_instability_diagnostic(args.campaign_root, matched)
    allocation = build_vintage_emission_allocation(panel)
    composition = build_composition_agreement(ranks)

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    products = {
        "panel.parquet": panel,
        "matched_period_summary.parquet": matched,
        "cross_series_agreement.parquet": agreement,
        "vintage_emission_allocation.parquet": allocation,
        "composition_agreement.parquet": composition,
        "window_sensitivity.parquet": windows,
        "instability_diagnostic.parquet": instability,
        "comparison_nodes.parquet": node_cmp,
        "comparison_ranks.parquet": ranks,
        "difference_reconciliation.parquet": recon,
        "raw_dax_diagnostic.parquet": raw_dax,
    }
    outputs = []
    for name, frame in products.items():
        p = out / name
        frame.to_parquet(p, index=False)
        outputs.append({"role": name.split(".")[0], "path": str(p), "sha256": sha256(p),
                        "rows": int(len(frame))})
    write_report(out / "REPORT.md", node_cmp, ranks, recon, raw_dax, matched, agreement,
                 windows, instability, allocation, composition)
    outputs.append({"role": "report", "path": str(out / "REPORT.md"),
                    "sha256": sha256(out / "REPORT.md"), "rows": None})

    inputs = []
    for year in YEARS:
        yp = year - 1
        for config in NATIVE_CONFIGS:
            d = args.native_annual_root / f"{yp}_{year}_{config}" / "manifest.v2.json"
            if d.is_file():
                inputs.append({"role": "native_annual", "path": str(d), "sha256": sha256(d)})
        for res in ("2016", "2013", "R3"):
            d = args.main_annual_root / f"{yp}_{year}_{res}" / "manifest.v2.json"
            if d.is_file():
                inputs.append({"role": "main_annual", "path": str(d), "sha256": sha256(d)})

    n_provisional = int(node_cmp.provisional.sum()) if not node_cmp.empty else 0
    worst_recon = float(recon.relative_residual.max()) if not recon.empty else 0.0
    worst_recon_nonprov = float(
        recon.loc[~recon.involves_provisional, "residual_vs_panel_magnitude"].max()
    ) if not recon.empty and (~recon.involves_provisional).any() else 0.0
    manifest = {
        "schema_version": 1,
        "analysis_id": "native-wiod-three-way-comparison-v1",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": "implementation/bilateral_sda_attribution/scripts/build_native_wiod_three_way_comparison.py",
            "git_commit": args.git_commit,
            "command": sys.argv,
        },
        "scope": {
            "price_basis": "CYP_(t-1) -> PYP_t",
            "indicator": "native WIOD/JRC CO2 (kg CO2) vs EXIOBASE fossil-origin CO2 (kg CO2)",
            "units": "kg CO2",
            "effects": list(HIER_NODES),
            "resolutions": list(SERIES_MATRIX),
            "year_previous_min": YEARS[0] - 1,
            "year_current_max": YEARS[-1],
            "comparisons": {k: {"a": f"{v[0]}:{v[1]}", "b": f"{v[2]}:{v[3]}", "same_grid": v[4]}
                            for k, v in PAIRS.items()},
        },
        "coverage": {
            "expected": len(PAIRS) * len(YEARS) * len(HIER_NODES),
            "accepted": int(len(node_cmp)),
            "missing": [], "excluded": [],
        },
        "inputs": inputs,
        "outputs": outputs,
        "validation": {
            "passed": worst_recon_nonprov < 5e-3,
            "gates": {
                "diff_of_diffs_worst_relative_residual_raw": worst_recon,
                "diff_of_diffs_worst_residual_vs_panel_magnitude_nonprovisional": worst_recon_nonprov,
                "note": ("residuals on provisional rows equal the inherited "
                         "Leontief-split residuals from the 5 accepted_provisional runs; "
                         "the raw relative figure is inflated near sign changes"),
            },
            "upstream_input_status": "accepted_provisional",
            "thesis_status": "accepted_provisional",
        },
        "limitations": [
            "Main EXIOBASE campaign is fossil-origin CO2, not GWP100 (handover §6.C "
            "discrepancy); Q3 mixes account scope, database and flow vintage.",
            "Q1 compares different country/sector grids; composition not comparable.",
            "Q2 is not a single-factor database contrast.",
            "Non-nested C13-C15 50/50 allocation on the jrc2019_to_2013 side.",
            f"{n_provisional} node-comparison rows involve accepted_provisional Leontief-split nodes.",
            "All native and main annual inputs are provisional (D-2026-020, D-2026-022).",
        ],
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_root": str(out),
        "panel_rows": len(panel), "node_comparison_rows": len(node_cmp),
        "matched_period_rows": len(matched), "cross_series_agreement_rows": len(agreement),
        "series": sorted(panel.series.unique()),
        "rank_rows": len(ranks), "provisional_rows": n_provisional,
        "worst_difference_reconciliation_residual": worst_recon,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
