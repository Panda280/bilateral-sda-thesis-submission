#!/usr/bin/env python3
"""Three-grid comparison tables: EXIOBASE impact AND EXIOBASE flows throughout,
varying only the target country-sector grid.

  R3    49 x 98   finest
  2016  44 x 50   WIOD2016-compatible
  2013  41 x 35   WIOD2013-compatible

Because the impact vector and the flow source are held fixed, every difference
between the three columns is a property of the target grid alone. Registered
sources, read-only:

  R-2026-035/036   by_period_hierarchy.parquet          global hierarchy
  R-2026-038       kyoto_period_group_effects.csv       group marginals
  R-2026-009/011   annual_country_scorecards.csv        country detail
  R-2026-019       reviewed_cross_resolution_candidates.csv  native vs R3-mapped
No value is hand-entered.
"""
from __future__ import annotations
import csv, json, hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "implementation/bilateral_sda_attribution/docs/analysis"
HIER = AN / "native_wiod_v2_20260903/kyoto-period-hierarchy-20260906/by_period_hierarchy.parquet"
PROD = AN / "kyoto_period_group_products_20260908/kyoto_period_group_effects.csv"
SCORE = AN / "kyoto_country_scorecards_cp1_20260907/annual_country_scorecards.csv"
XRES = AN / "kyoto_group_sector_deep_dive_cp1_20260907/reviewed_cross_resolution_candidates.csv"
OUT = ROOT / "src/chapters/results_revised/generated"
GT, MT = 1e12, 1e9
# finest first, so the reader moves from most to least resolved
GRIDS = [("exio_R3", "R3", r"R3 $49\times98$"),
         ("exio_2016", "2016", r"$44\times50$"),
         ("exio_2013", "2013", r"$41\times35$")]
PERIODS = [("P1_pre_entry_into_force", "P1", "P1, 2001--2004"),
           ("P2_ratification", "P2", "P2, 2005--2007"),
           ("P3_first_commitment_period", "P3", "P3, 2008--2012")]
GRP = [("Red", "Reduction"), ("Non-Red", "Non-Reduction"), ("Non-Participant", "Non-Participation")]
NODES = [("delta_g", r"\textbf{Total change}", True),
         ("emission_intensity", r"$\mathbf D_{\mathrm{int}}$", False),
         ("leontief_production_structure", r"$\mathbf D_L$", False),
         ("D_C", r"\quad $\mathbf D_C$", False),
         ("D_A_x", r"\quad $\mathbf D_{A_x}$", False),
         ("D_T_coef", r"\qquad $\mathbf D_T$", False),
         ("D_w_coef", r"\qquad $\mathbf D_w$", False),
         ("final_demand", r"$\mathbf D_{\mathrm{fd}}$", False),
         ("D_YF", r"\quad $\mathbf D_{Y_F}$", False),
         ("D_YB", r"\quad $\mathbf D_{Y_B}$", False),
         ("D_y", r"\quad $\mathbf D_{Y_y}$", False)]


def f(v, dp=3, unit=GT):
    return "---" if v is None else f"${v/unit:+.{dp}f}$".replace("+-", "-")


def spread(vals):
    return max(vals) - min(vals)


# ------------------------------------------------------------------ global
def global_grids():
    df = pd.read_parquet(HIER)

    def val(series, period, node):
        m = df[(df.series == series) & (df.period == period) & (df.node_id == node)]
        return None if not len(m) or pd.isna(m["cumulative_contribution"].iloc[0]) \
            else float(m["cumulative_contribution"].iloc[0])

    L = [r"\begin{table}[htbp]", r"  \centering", r"  \scriptsize",
         r"  \caption{Level 1 on the three target grids, Gt CO$_2$. The EXIOBASE"
         r" impact vector and the EXIOBASE-derived flow tables are held fixed, and"
         r" only the target country--sector grid changes, so every difference across"
         r" a row is a property of the grid alone. Indented rows are children of the"
         r" effect above them and are not added to it. \emph{Spr.} is the largest"
         r" minus the smallest of the three grids"
         r" (Results R-2026-035 and R-2026-036).}",
         r"  \label{tab:grid-global-subeffects}", r"  \setlength{\tabcolsep}{2.5pt}",
         r"  \begin{tabular}{l" + "rrrr" * 3 + "}", r"  \toprule",
         r"  & \multicolumn{4}{c}{P1, 2001--2004} & \multicolumn{4}{c}{P2, 2005--2007}"
         r" & \multicolumn{4}{c}{P3, 2008--2012} \\",
         r"  \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}",
         r"  Effect" + r" & R3 & 2016 & 2013 & Spr." * 3 + r" \\", r"  \midrule"]
    for node, sym, rule in NODES:
        cells = []
        for pkey, _, _ in PERIODS:
            vs = [val(s, pkey, node) for s, _, _ in GRIDS]
            cells += [f(v, 2) for v in vs]
            cells.append(f(spread([v for v in vs if v is not None]), 2) if all(v is not None for v in vs) else "---")
        L.append(f"  {sym} & " + " & ".join(cells) + r" \\")
        if rule:
            L.append(r"  \midrule")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- group
def group_grids():
    E = list(csv.DictReader(open(PROD)))

    def eff(period, effect, group, cfg, role="emitter"):
        for r in E:
            if (r["period"] == period and r["configuration"] == cfg and r["effect"] == effect
                    and r["role"] == role and r["group"] == group):
                return float(r["net"])
        return None

    ROWS = [("D_v", r"$\mathbf D_{\mathrm{int}}$"), ("D_A_x", r"$\mathbf D_{A_x}$"),
            ("D_C", r"$\mathbf D_C$"), ("D_YF", r"$\mathbf D_{Y_F}$"),
            ("D_YB", r"$\mathbf D_{Y_B}$"), ("D_y", r"$\mathbf D_{Y_y}$")]
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \scriptsize",
         r"  \caption{Level 2 on the three target grids, Mt CO$_2$, emitter role,"
         r" impact vector and flow source held fixed. A shaded \emph{Spr.} entry"
         r" marks a quantity on which the three grids do not agree in sign"
         r" (Result R-2026-038).}",
         r"  \label{tab:grid-group-effects}", r"  \setlength{\tabcolsep}{3pt}",
         r"  \begin{tabular}{ll" + "rrrr" * 3 + "}", r"  \toprule",
         r"  & & \multicolumn{4}{c}{P1} & \multicolumn{4}{c}{P2} & \multicolumn{4}{c}{P3} \\",
         r"  \cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(l){11-14}",
         r"  Group & Effect" + r" & R3 & 2016 & 2013 & Spr." * 3 + r" \\", r"  \midrule"]
    for gc, gl in GRP:
        for k, (code, sym) in enumerate(ROWS):
            cells = []
            for pkey, plab, _ in PERIODS:
                vs = [eff(plab, code, gc, cfg) for cfg, _, _ in GRIDS]
                cells += [f(v, 0, MT) for v in vs]
                sp = spread(vs)
                flip = len({v > 0 for v in vs}) > 1
                cells.append((r"\textbf{" + f"{sp/MT:,.0f}" + r"}$^{\dagger}$") if flip else f"{sp/MT:,.0f}")
            L.append(f"  {gl if k == 0 else ''} & {sym} & " + " & ".join(cells) + r" \\")
        if gc != GRP[-1][0]:
            L.append(r"  \addlinespace")
    L += [r"  \bottomrule", r"  \end{tabular}",
          r"  \\[2pt] {\scriptsize $\dagger$ the three grids disagree on the sign of this quantity.}",
          r"\end{table}"]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------- country
def country_grids(topn=4):
    rows = list(csv.DictReader(open(SCORE)))
    agg = defaultdict(float)
    grp = {}
    for r in rows:
        y = int(r["year_current"])
        if 2001 <= y <= 2012 and r["role"] == "emitter":
            agg[(r["resolution"], r["effect"], r["canonical_country"])] += float(r["net"])
            grp[r["canonical_country"]] = r["analysis_group"]
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \scriptsize",
         r"  \caption{Level 3, the leading countries on the three target grids,"
         r" cumulated over $2001$--$2012$, Mt CO$_2$, emitter role, impact vector and"
         r" flow source held fixed. Countries are the four largest by absolute"
         r" $\mathbf D_C$ within their group on any grid. A dagger marks a country"
         r" whose $\mathbf D_C$ or $\mathbf D_{Y_F}$ does not keep its sign across the"
         r" three grids"
         r" (Results R-2026-009 and R-2026-011).}",
         r"  \label{tab:grid-country}", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{ll" + "rrr" * 2 + "}", r"  \toprule",
         r"  & & \multicolumn{3}{c}{$\mathbf D_C$} & \multicolumn{3}{c}{$\mathbf D_{Y_F}$} \\",
         r"  \cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"  Group & Country & R3 & 2016 & 2013 & R3 & 2016 & 2013 \\", r"  \midrule"]
    for gc, gl in GRP:
        members = [c for c, g in grp.items() if g == gc]
        ranked = sorted(members, key=lambda c: -max(abs(agg.get((res, "D_C", c), 0.0))
                                                    for _, res, _ in GRIDS))[:topn]
        for k, c in enumerate(ranked):
            cells, flip = [], False
            for eff in ("D_C", "D_YF"):
                vs = [agg.get((res, eff, c), 0.0) for _, res, _ in GRIDS]
                if len({v > 0 for v in vs}) > 1:
                    flip = True
                cells += [f(v, 0, MT) for v in vs]
            name = f"\\texttt{{{c}}}" + (r"$^{\dagger}$" if flip else "")
            L.append(f"  {gl if k == 0 else ''} & {name} & " + " & ".join(cells) + r" \\")
        if gc != GRP[-1][0]:
            L.append(r"  \addlinespace")
    L += [r"  \bottomrule", r"  \end{tabular}",
          r"  \\[2pt] {\scriptsize $\dagger$ this country's $\mathbf D_C$ or $\mathbf D_{Y_F}$ does not keep its sign across the three grids.}",
          r"\end{table}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ sector
def sector_hidden(topn=6):
    """Two panels: coarse sectors whose sign the fine grid contradicts, and
    coarse sectors that report far less activity than their own fine cells."""
    R = [r for r in csv.DictReader(open(XRES))]
    for r in R:
        r["_nat"] = float(r["native_signed_kg"] or 0.0)
        r["_map"] = float(r["mapped_r3_signed_kg"] or 0.0)
        r["_gap"] = abs(r["_map"]) - abs(r["_nat"])

    flips = [r for r in R if r["sign_agreement"] == "False"]
    flips.sort(key=lambda r: -abs(r["_map"]))
    hidden = [r for r in R if r["sign_agreement"] == "True" and r["_gap"] > 0]
    hidden.sort(key=lambda r: -r["_gap"])

    def block(rows, title, note):
        out = [r"  \multicolumn{7}{l}{\emph{" + title + r"}} \\"]
        for r in rows[:topn]:
            out.append("  \\quad \\texttt{{{}}} & {} & {} & {} & {} & {} & {} \\\\".format(
                r["sector_code"].replace("_", r"\_"), r["resolution"],
                {"Red": "Red", "Non-Red": "N-Red", "Non-Participant": "N-Part"}[r["destination_group"]],
                r["effect"].replace("_", r"\_").replace(r"D\_C", r"$\mathbf D_C$").replace(r"D\_YF", r"$\mathbf D_{Y_F}$"),
                f(r["_nat"], 0, MT), f(r["_map"], 0, MT), r["absolute_rank_change"]))
        return out

    L = [r"\begin{table}[htbp]", r"  \centering", r"  \scriptsize",
         r"  \caption{Level 3, sectors that a coarser grid does not resolve,"
         r" $2001$--$2012$, Mt CO$_2$. For each coarse sector, \emph{native} is what"
         r" that grid computes for it and \emph{R3 cells} is the sum of the R3 cells"
         r" that map into it under the reviewed concordance; \emph{rank} is how far"
         r" the sector moves in its group's absolute ranking between the two. The"
         r" upper panel lists the largest cases where the two disagree in sign, the"
         r" lower panel the largest cases where they agree in sign but the coarse"
         r" grid reports far less activity. Of $462$ reviewed coarse sectors, $85$"
         r" disagree in sign with their own fine cells (Result R-2026-019).}",
         r"  \label{tab:grid-sector-hidden}", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{llllrrr}", r"  \toprule",
         r"  Sector & Grid & Group & Effect & Native & R3 cells & Rank \\", r"  \midrule"]
    L += block(flips, r"Panel A: the fine cells contradict the coarse sector's sign",
               "")
    L.append(r"  \addlinespace")
    L += block(hidden, r"Panel B: the coarse sector reports far less than its own fine cells", "")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n", len(flips), len(R)



# ------------------------------------------------ flow-table vintage contrast
def flow_vintage():
    """EXIOBASE-derived against WIOD2013-published flow tables, 41x35 grid.

    The impact vector (EXIOBASE), the target grid and the window are identical
    on both sides, so the only thing that differs is the flow table itself.
    P3 is omitted: the published-flow campaign holds 2 of its 5 transitions, so
    a P3 column would report missing years, not a vintage effect.
    """
    df = pd.read_parquet(HIER)

    def val(series, period, node):
        m = df[(df.series == series) & (df.period == period) & (df.node_id == node)]
        return None if not len(m) or pd.isna(m["cumulative_contribution"].iloc[0]) \
            else float(m["cumulative_contribution"].iloc[0])

    L = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
         r"  \caption{Dataset vintage at a fixed grid: the EXIOBASE-derived flow"
         r" tables against the official WIOD2013-published tables, both at"
         r" $41\times35$, both with the EXIOBASE fossil-origin CO$_2$ impact vector,"
         r" Gt CO$_2$. The impact account, the target grid and the window are"
         r" identical on the two sides, so every difference is the flow table"
         r" alone. Indented rows are children of the effect above and are not added"
         r" to it. P3 is omitted because the published-flow campaign holds two of"
         r" its five transitions; a P3 column would report missing years rather"
         r" than a vintage effect (Decision D-2026-044; Results R-2026-035 and"
         r" R-2026-036).}",
         r"  \label{tab:flow-vintage}", r"  \setlength{\tabcolsep}{5pt}",
         r"  \begin{tabular}{lrrrrrr}", r"  \toprule",
         r"  & \multicolumn{3}{c}{P1, 2001--2004} & \multicolumn{3}{c}{P2, 2005--2007} \\",
         r"  \cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"  Effect & Derived & Published & Diff. & Derived & Published & Diff. \\",
         r"  \midrule"]
    for node, sym, rule in NODES:
        cells = []
        for pkey in ("P1_pre_entry_into_force", "P2_ratification"):
            a = val("exio_2013", pkey, node)
            b = val("exio_wiodflows2013", pkey, node)
            cells += [f(a), f(b), f(b - a) if a is not None and b is not None else "---"]
        L.append(f"  {sym} & " + " & ".join(cells) + r" \\")
        if rule:
            L.append(r"  \midrule")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sec, nflip, ntot = sector_hidden()
    files = {"tab_grid_global_subeffects.tex": global_grids(),
             "tab_grid_group_effects.tex": group_grids(),
             "tab_grid_country.tex": country_grids(),
             "tab_grid_sector_hidden.tex": sec,
             "tab_flow_vintage.tex": flow_vintage()}
    man = {}
    for n, b in files.items():
        (OUT / n).write_text(b, encoding="utf-8", newline="\n")
        man[n] = hashlib.sha256(b.encode()).hexdigest()[:16]
        print(f"wrote {n:36s} {len(b.splitlines()):3d} lines")
    print(f"\ncross-resolution sign disagreements: {nflip} of {ntot} reviewed coarse sectors")
    (OUT / "GENERATED_MANIFEST_GRIDS.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_chapter4_grid_vintage_tables.py",
        "design": "EXIOBASE impact and EXIOBASE-derived flows held fixed; only the target grid varies",
        "grids": [g[2] for g in GRIDS],
        "results": ["R-2026-009", "R-2026-011", "R-2026-019", "R-2026-035", "R-2026-036", "R-2026-038"],
        "sign_disagreements": {"count": nflip, "of": ntot}, "outputs": man,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
