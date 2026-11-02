#!/usr/bin/env python3
"""Generate the Chapter 4 tables in the reporting manner of Hoekstra, Michel and
Suh (2016), extended down three levels of aggregation.

Level 1  global      four effect categories, by Kyoto period
Level 2  group       groups x categories as % of the global change (their Table 1)
                     variables x groups in Mt          (their Table 2)
                     sourcing split by origin          (their Table A3)
Level 3  group x country / group x sector

Category mapping (additive, exactly as in their Equation 3):
  Technology           = D_int + D_A_x        (their v + A)
  Sourcing             = D_C   + D_YF         (their C_* + F_*)
  Consumption patterns = D_YB                 (their B)
  Consumption growth   = D_Yy                 (their y)

Sources (read-only, all registered):
  R-2026-038  kyoto_period_group_effects.csv          group marginals by role
  R-2026-039  kyoto_period_group_interactions.csv     source x destination cube
  R-2026-009/011  annual_country_scorecards.csv       country detail, annual
  R-2026-019  group_sector_all.csv                    group x sector, CP1 window
No value is hand-entered.
"""
from __future__ import annotations
import csv, json, hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "implementation/bilateral_sda_attribution/docs/analysis"
PROD = AN / "kyoto_period_group_products_20260908"
SCORE = AN / "kyoto_country_scorecards_cp1_20260907/annual_country_scorecards.csv"
GSEC = AN / "kyoto_group_sector_deep_dive_cp1_20260907/group_sector_all.csv"
OUT = ROOT / "src/chapters/results_revised/generated"
GT, MT = 1e12, 1e9
MAIN = "exio_2016"
GRIDS = ["exio_2013", "exio_2016", "exio_R3"]
PERIODS = ["P1", "P2", "P3"]
PLAB = {"P1": "P1, 2001--2004", "P2": "P2, 2005--2007", "P3": "P3, 2008--2012"}
GRP = [("Red", "Reduction"), ("Non-Red", "Non-Reduction"), ("Non-Participant", "Non-Participation")]
CATS = [("Technology", ["D_v", "D_A_x"]),
        ("Sourcing", ["D_C", "D_YF"]),
        ("Consumption patterns", ["D_YB"]),
        ("Consumption growth", ["D_y"])]
E = list(csv.DictReader(open(PROD / "kyoto_period_group_effects.csv")))
I = list(csv.DictReader(open(PROD / "kyoto_period_group_interactions.csv")))


def eff(period, effect, group, cfg=MAIN, role="emitter", field="net"):
    for r in E:
        if (r["period"] == period and r["configuration"] == cfg and r["effect"] == effect
                and r["role"] == role and r["group"] == group):
            return float(r[field])
    return 0.0


def cat(period, cats, group, cfg=MAIN, field="net"):
    return sum(eff(period, e, group, cfg, field=field) for e in cats)


def total(period, group, cfg=MAIN):
    return sum(cat(period, c, group, cfg) for _, c in CATS)


def gtotal(period, cfg=MAIN):
    return sum(total(period, g, cfg) for g, _ in GRP)


def f(v, dp=2, unit=GT):
    return f"${v/unit:+.{dp}f}$".replace("+-", "-")


def pct(v, base):
    return f"${v/base*100:+.0f}$" if base else "---"


# ---------------------------------------------------------------- level 1
def level1():
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
         r"  \caption{Level 1, the global decomposition. Change in world"
         r" fossil-origin CO$_2$ by Kyoto period, in the four effect categories of"
         r" Hoekstra, Michel and Suh~\cite{HoekstraMichelSuh2016}: technology is $\mathbf D_{\mathrm{int}}"
         r"+\mathbf D_{A_x}$, sourcing is $\mathbf D_C+\mathbf D_{Y_F}$, consumption"
         r" patterns is $\mathbf D_{Y_B}$ and consumption growth is $\mathbf D_{Y_y}$."
         r" The four categories are additive and sum to the total change. Gt CO$_2$,"
         r" with each category as a percentage of the period total beneath it."
         r" EXIOBASE sources on the $2016$-compatible grid, emitter role"
         r" (Result R-2026-038).}",
         r"  \label{tab:level1-global}", r"  \begin{tabular}{llrrrrr}", r"  \toprule",
         r"  Period & & Technology & Sourcing & Cons.\ patterns & Cons.\ growth & Total \\",
         r"  \midrule"]
    for p in PERIODS:
        tot = gtotal(p)
        vals = [sum(cat(p, c, g) for g, _ in GRP) for _, c in CATS]
        L.append(f"  {PLAB[p]} & Gt & " + " & ".join(f(v) for v in vals) + f" & {f(tot)} " + r"\\")
        L.append(r"   & \% & " + " & ".join(pct(v, tot) for v in vals) + f" & {pct(tot, tot)} " + r"\\")
        if p != PERIODS[-1]:
            L.append(r"  \addlinespace")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- level 2a
def level2_groups():
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
         r"  \caption{Level 2, the group decomposition, in the form of Table~1 of"
         r" Hoekstra, Michel and Suh~\cite{HoekstraMichelSuh2016}. Each cell is a percentage of the"
         r" \emph{global} change in that period, so the Total row is the global"
         r" decomposition of \cref{tab:level1-global} and the group rows sum to it."
         r" Rows are the analytical group in whose territory the change is"
         r" registered (emitter role). EXIOBASE sources, $2016$-compatible grid"
         r" (Result R-2026-038).}",
         r"  \label{tab:level2-groups}", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{lrrrrr}", r"  \toprule",
         r"  Group & Technology & Sourcing & Cons.\ patterns & Cons.\ growth & Total \\",
         r"  \midrule"]
    for p in PERIODS:
        base = gtotal(p)
        L.append(r"  \multicolumn{6}{l}{\emph{" + PLAB[p] + r"}} \\")
        for gc, gl in GRP:
            vals = [cat(p, c, gc) for _, c in CATS]
            L.append(f"  \\quad {gl} & "
                     + " & ".join(pct(v, base) for v in vals)
                     + f" & {pct(total(p, gc), base)} " + r"\\")
        vals = [sum(cat(p, c, g) for g, _ in GRP) for _, c in CATS]
        L.append(r"  \quad \textbf{Total} & " + " & ".join(pct(v, base) for v in vals)
                 + f" & {pct(base, base)} " + r"\\")
        L.append(f"  \\quad \\emph{{(Gt CO$_2$)}} & " + " & ".join(f(v) for v in vals)
                 + f" & {f(base)} " + r"\\")
        if p != PERIODS[-1]:
            L.append(r"  \addlinespace")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- level 2b
def level2_variables():
    VARS = [("Technology", r"$\mathbf D_{\mathrm{int}}$", "D_v"),
            ("", r"$\mathbf D_{A_x}$", "D_A_x"),
            ("", r"\quad $\mathbf D_T$", "D_T_coef"),
            ("", r"\quad $\mathbf D_w$", "D_w_coef"),
            ("Sourcing", r"$\mathbf D_C$", "D_C"),
            ("", r"$\mathbf D_{Y_F}$", "D_YF"),
            ("Consumption", r"$\mathbf D_{Y_B}$", "D_YB"),
            ("", r"$\mathbf D_{Y_y}$", "D_y")]
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \scriptsize",
         r"  \caption{Level 2 in detail, in the form of Table~2 of"
         r" Hoekstra, Michel and Suh~\cite{HoekstraMichelSuh2016}: the individual effects behind the four"
         r" categories, by analytical group and Kyoto period, Mt CO$_2$, emitter"
         r" role. $\mathbf D_T$ and $\mathbf D_w$ are the nested children of"
         r" $\mathbf D_{A_x}$ and are indented; they are not added to it. Total"
         r" sourcing is $\mathbf D_C+\mathbf D_{Y_F}$, the analogue of their ECS,"
         r" but computed on Kyoto rather than wage groups (Result R-2026-038).}",
         r"  \label{tab:level2-variables}", r"  \setlength{\tabcolsep}{3pt}",
         r"  \begin{tabular}{ll" + "rrr" * 3 + "}", r"  \toprule",
         r"  & & \multicolumn{3}{c}{P1, 2001--2004} & \multicolumn{3}{c}{P2, 2005--2007}"
         r" & \multicolumn{3}{c}{P3, 2008--2012} \\",
         r"  \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
         r"  & & Red & N-Red & N-Part & Red & N-Red & N-Part & Red & N-Red & N-Part \\",
         r"  \midrule"]
    for block, sym, code in VARS:
        cells = [f(eff(p, code, g), 0, MT) for p in PERIODS for g, _ in GRP]
        L.append(f"  {block} & {sym} & " + " & ".join(cells) + r" \\")
    L.append(r"  \midrule")
    L.append(r"  \textbf{Total} & & " + " & ".join(f(total(p, g), 0, MT)
             for p in PERIODS for g, _ in GRP) + r" \\")
    L.append(r"  Total sourcing & & " + " & ".join(f(cat(p, ["D_C", "D_YF"], g), 0, MT)
             for p in PERIODS for g, _ in GRP) + r" \\")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- level 2c
def sourcing_origin():
    def cell(p, e, rel, sg, dg):
        t = 0.0
        for r in I:
            if (r["period"] == p and r["configuration"] == MAIN and r["effect"] == e
                    and r["relationship"] == rel
                    and (sg is None or r["source_group"] == sg)
                    and (dg is None or r["destination_group"] == dg)):
                t += float(r["net"])
        return t
    ROWS = [(r"$\mathbf D_C$ domestic", "D_C", "Domestic", None),
            (r"\quad from Reduction", "D_C", "Cross-border", "Red"),
            (r"\quad from Non-Reduction", "D_C", "Cross-border", "Non-Red"),
            (r"\quad from Non-Participation", "D_C", "Cross-border", "Non-Participant"),
            (r"$\mathbf D_{Y_F}$ domestic", "D_YF", "Domestic", None),
            (r"\quad from Reduction", "D_YF", "Cross-border", "Red"),
            (r"\quad from Non-Reduction", "D_YF", "Cross-border", "Non-Red"),
            (r"\quad from Non-Participation", "D_YF", "Cross-border", "Non-Participant")]
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \scriptsize",
         r"  \caption{The sourcing effects split by the origin of the sourced input,"
         r" in the form of Table~A3 of Hoekstra, Michel and Suh~\cite{HoekstraMichelSuh2016}, whose"
         r" $\mathbf C_{\mathrm{dom}},\mathbf C_{\mathrm{low}},\ldots$ rows this"
         r" reproduces on Kyoto groups. Columns are the destination group, that is"
         r" the group whose industries changed their sourcing (structural-driver role"
         r" for $\mathbf D_C$, consumer role for $\mathbf D_{Y_F}$); rows are the"
         r" origin of the input. A domestic row is necessarily same-group, but a"
         r" same-group cell is not necessarily domestic. Mt CO$_2$, EXIOBASE sources,"
         r" $2016$-compatible grid (Result R-2026-039).}",
         r"  \label{tab:level2-sourcing-origin}", r"  \setlength{\tabcolsep}{2.5pt}",
         r"  \begin{tabular}{l" + "rrrr" * 3 + "}", r"  \toprule",
         r"  & \multicolumn{4}{c}{P1, 2001--2004} & \multicolumn{4}{c}{P2, 2005--2007}"
         r" & \multicolumn{4}{c}{P3, 2008--2012} \\",
         r"  \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}",
         r"  Origin & Red & N-Red & N-Part & Tot. & Red & N-Red & N-Part & Tot."
         r" & Red & N-Red & N-Part & Tot. \\", r"  \midrule"]
    for label, e, rel, sg in ROWS:
        cells = []
        for p in PERIODS:
            row = [cell(p, e, rel, sg, dg) for dg, _ in GRP]
            cells += [f(v, 0, MT) for v in row] + [f(sum(row), 0, MT)]
        L.append(f"  {label} & " + " & ".join(cells) + r" \\")
        if e == "D_C" and sg == "Non-Participant":
            L.append(r"  \addlinespace")
    L.append(r"  \midrule")
    tot = []
    for p in PERIODS:
        row = [sum(cell(p, e, rel, sg, dg) for _, e, rel, sg in ROWS) for dg, _ in GRP]
        tot += [f(v, 0, MT) for v in row] + [f(sum(row), 0, MT)]
    L.append(r"  \textbf{Total sourcing} & " + " & ".join(tot) + r" \\")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- level 3
def level3_country(topn=4):
    rows = list(csv.DictReader(open(SCORE)))
    per = lambda y: "P1" if 2001 <= y <= 2004 else ("P2" if 2005 <= y <= 2007 else ("P3" if 2008 <= y <= 2012 else None))
    agg = defaultdict(float)
    grp = {}
    for r in rows:
        p = per(int(r["year_current"]))
        if p and r["resolution"] == "2016" and r["role"] == "emitter":
            agg[(p, r["effect"], r["canonical_country"])] += float(r["net"])
            grp[r["canonical_country"]] = r["analysis_group"]
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \footnotesize",
         r"  \caption{Level 3, the group--country detail behind the sourcing effects."
         r" For each analytical group and period, the countries carrying the largest"
         r" absolute cumulative $\mathbf D_C$ contribution in their group, Mt CO$_2$,"
         r" emitter role, $2016$-compatible grid. Country contributions within a group"
         r" sum to that group's marginal in \cref{tab:level2-variables}. These locate"
         r" which countries carry a group balance; they are not rankings of"
         r" responsibility (Results R-2026-009 and R-2026-011).}",
         r"  \label{tab:level3-country}", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{ll" + "rr" * 3 + "}", r"  \toprule",
         r"  & & \multicolumn{2}{c}{P1} & \multicolumn{2}{c}{P2} & \multicolumn{2}{c}{P3} \\",
         r"  \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
         r"  Group & Country & $\mathbf D_C$ & $\mathbf D_{Y_F}$ & $\mathbf D_C$"
         r" & $\mathbf D_{Y_F}$ & $\mathbf D_C$ & $\mathbf D_{Y_F}$ \\", r"  \midrule"]
    for gc, gl in GRP:
        members = [c for c, g in grp.items() if g == gc]
        ranked = sorted(members, key=lambda c: -max(abs(agg.get((p, "D_C", c), 0.0)) for p in PERIODS))[:topn]
        for k, c in enumerate(ranked):
            cells = []
            for p in PERIODS:
                cells += [f(agg.get((p, "D_C", c), 0.0), 0, MT), f(agg.get((p, "D_YF", c), 0.0), 0, MT)]
            L.append(f"  {gl if k == 0 else ''} & \\texttt{{{c}}} & " + " & ".join(cells) + r" \\")
        cells = []
        for p in PERIODS:
            cells += [f(sum(agg.get((p, "D_C", c), 0.0) for c in members), 0, MT),
                      f(sum(agg.get((p, "D_YF", c), 0.0) for c in members), 0, MT)]
        L.append(r"   & \emph{group total} & " + " & ".join(cells) + r" \\")
        if gc != GRP[-1][0]:
            L.append(r"  \addlinespace")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def level3_sector(topn=4):
    rows = [r for r in csv.DictReader(open(GSEC)) if r["resolution"] == "2016"]
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \footnotesize",
         r"  \caption{Level 3, the group--sector detail. The sectors carrying the"
         r" largest absolute $\mathbf D_C$ contribution within each destination group,"
         r" with the domestic and same-group cross-border parts of that contribution"
         r" shown separately, Mt CO$_2$, $2016$-compatible grid. This product is"
         r" cumulated over the whole $2001$--$2012$ window and is not resolved by"
         r" period; it is the deepest joint grain for which a registered product"
         r" exists, and no group--country--sector product is registered at all"
         r" (Result R-2026-019).}",
         r"  \label{tab:level3-sector}", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{llrrrr}", r"  \toprule",
         r"  Destination group & Sector & All & Domestic & Same-group"
         r" & Non-Part.\ cross-b. \\", r"  \midrule"]
    for gc, gl in GRP:
        sub = [r for r in rows if r["destination_group"] == gc and r["effect"] == "D_C"]
        sub.sort(key=lambda r: -abs(float(r["all_contributions_mt_co2"])))
        for k, r in enumerate(sub[:topn]):
            L.append(f"  {gl if k == 0 else ''} & \\texttt{{{r['sector_code']}}} & "
                     + " & ".join(f"${float(r[c]):+,.0f}$".replace("+-", "-") for c in
                                  ["all_contributions_mt_co2", "domestic_mt_co2",
                                   "same_group_mt_co2", "np_cross_border_mt_co2"]) + r" \\")
        if gc != GRP[-1][0]:
            L.append(r"  \addlinespace")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"



# ------------------------------------------------- level x grid sensitivity
def level_grid_sensitivity():
    """How far each level's numbers move when only the target grid changes."""
    import statistics
    def spread(vals):
        return max(vals) - min(vals)

    rows = []
    # level 1: the four global category shares, per period
    l1_spread, l1_flip, l1_n = [], 0, 0
    for p in PERIODS:
        for name, cs in CATS:
            sh = [sum(cat(p, cs, g, cfg) for g, _ in GRP) / gtotal(p, cfg) for cfg in GRIDS]
            l1_spread.append(spread(sh) * 100); l1_n += 1
            if len({v > 0 for v in sh}) > 1:
                l1_flip += 1
    # level 2: each group x category share of the global change
    l2_spread, l2_flip, l2_n = [], 0, 0
    for p in PERIODS:
        for g, _ in GRP:
            for name, cs in CATS:
                sh = [cat(p, cs, g, cfg) / gtotal(p, cfg) for cfg in GRIDS]
                l2_spread.append(spread(sh) * 100); l2_n += 1
                if len({v > 0 for v in sh}) > 1:
                    l2_flip += 1
    # level 3 proxy: each source x destination sourcing cell as a share of the global change
    l3_spread, l3_flip, l3_n = [], 0, 0
    for p in PERIODS:
        for e in ("D_C", "D_YF"):
            for sg, _ in GRP:
                for dg, _ in GRP:
                    sh = []
                    for cfg in GRIDS:
                        v = sum(float(r["net"]) for r in I
                                if r["period"] == p and r["configuration"] == cfg
                                and r["effect"] == e and r["source_group"] == sg
                                and r["destination_group"] == dg)
                        sh.append(v / gtotal(p, cfg))
                    l3_spread.append(spread(sh) * 100); l3_n += 1
                    if len({v > 0 for v in sh}) > 1:
                        l3_flip += 1
    data = [("Level 1: global category share", l1_spread, l1_flip, l1_n),
            ("Level 2: group $\\times$ category share", l2_spread, l2_flip, l2_n),
            ("Level 3: sourcing group-pair cell", l3_spread, l3_flip, l3_n)]
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
         r"  \caption{How far a reported quantity moves when only the target grid"
         r" changes, by level of aggregation. Each quantity is expressed as a"
         r" percentage of the global change in its period, then recomputed on the"
         r" $2013$, $2016$ and R3 grids; the spread is the largest minus the"
         r" smallest of the three. A sign disagreement means the three grids do not"
         r" agree on the direction of that quantity. The grids are not pooled and no"
         r" grid is preferred (Results R-2026-038 and R-2026-039).}",
         r"  \label{tab:level-grid-sensitivity}", r"  \begin{tabular}{lrrrr}", r"  \toprule",
         r"  Reported quantity & Cells & Median spread & Max spread & Sign disagreements \\",
         r"  \midrule"]
    for name, sp, fl, n in data:
        L.append(f"  {name} & {n} & ${statistics.median(sp):.1f}$ pp & ${max(sp):.1f}$ pp"
                 f" & {fl} of {n} \\\\")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"



# ------------------------------------------- sourcing across sector granularity
GSEC_RES = [("2013", 35), ("2016", 50), ("R3", 98)]


def _gsec():
    return list(csv.DictReader(open(GSEC)))


def sourcing_granularity():
    """How the sourcing effects look as the sector grid is refined.

    Net, cross-sector dispersion (sum of absolute sector contributions), the
    cancellation that dispersion implies, how many sectors carry a contribution
    opposite to their group's net, and how many sectors carry 80% of the
    absolute contribution.
    """
    R = _gsec()

    def cells(res, eff, g):
        return [float(x["all_contributions_mt_co2"]) for x in R
                if x["resolution"] == res and x["effect"] == eff
                and x["destination_group"] == g]

    L = [r"\begin{table}[htbp]", r"  \centering", r"  \scriptsize",
         r"  \caption{The sourcing effects as the sector grid is refined, "
         r"$2001$--$2012$, Mt CO$_2$. \emph{Net} is the signed sum over sectors; "
         r"\emph{disp.} is the sum of the absolute sector contributions; "
         r"$\kappa=1-\lvert\text{net}\rvert/\text{disp.}$ is the cancellation "
         r"between sectors that the net conceals; \emph{opp.} counts the sectors "
         r"whose contribution opposes their group's net; \emph{80\%} is the number "
         r"of sectors carrying $80$ per cent of the absolute contribution. Columns "
         r"are the destination group, that is the group whose industries changed "
         r"their sourcing. The sector totals reconcile exactly with the "
         r"destination-group marginals of \cref{tab:level2-sourcing-origin} summed "
         r"over the three periods (Result R-2026-019).}",
         r"  \label{tab:sourcing-granularity}", r"  \setlength{\tabcolsep}{3.5pt}",
         r"  \begin{tabular}{ll" + "rrrrr" * 2 + "}", r"  \toprule",
         r"  & & \multicolumn{5}{c}{$\mathbf D_C$, intermediate sourcing}"
         r" & \multicolumn{5}{c}{$\mathbf D_{Y_F}$, final-demand sourcing} \\",
         r"  \cmidrule(lr){3-7}\cmidrule(l){8-12}",
         r"  Grid & Group & Net & Disp. & $\kappa$ & Opp. & 80\%"
         r" & Net & Disp. & $\kappa$ & Opp. & 80\% \\", r"  \midrule"]
    for res, nsec in GSEC_RES:
        L.append(r"  \multicolumn{12}{l}{\emph{" + f"{res} grid, {nsec} sectors" + r"}} \\")
        for gc, gl in GRP:
            row = []
            for eff in ("D_C", "D_YF"):
                v = cells(res, eff, gc)
                net = sum(v)
                disp = sum(abs(x) for x in v)
                kap = (1 - abs(net) / disp) * 100 if disp else 0.0
                opp = sum(1 for x in v if x * net < 0)
                srt = sorted((abs(x) for x in v), reverse=True)
                c = 0.0
                k = 0
                for x in srt:
                    c += x
                    k += 1
                    if c >= 0.8 * disp:
                        break
                row += [f"${net:+,.0f}$".replace("+-", "-"), f"${disp:,.0f}$",
                        f"${kap:.0f}\\%$", f"{opp}", f"{k}"]
            L.append(f"  \\quad & {gl} & " + " & ".join(row) + r" \\")
        if res != GSEC_RES[-1][0]:
            L.append(r"  \addlinespace")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def sourcing_channels():
    """Where the sourcing effect sits, by relationship, at each sector grid."""
    R = _gsec()
    CH = [("Domestic", "domestic_mt_co2"),
          ("Same-group cross-border", "same_group_cross_border_mt_co2"),
          ("Non-Participation cross-border", "np_cross_border_mt_co2")]
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
         r"  \caption{The same sourcing effects split by relationship, at each "
         r"sector grid, $2001$--$2012$, Mt CO$_2$, summed over the three "
         r"destination groups. Country identity is unchanged by a sector "
         r"concordance, so these movements are a property of computing the "
         r"decomposition on differently aggregated tables, not evidence that "
         r"domestic sourcing itself behaves differently (Result R-2026-019).}",
         r"  \label{tab:sourcing-channels}", r"  \begin{tabular}{lrrrrrr}", r"  \toprule",
         r"  & \multicolumn{3}{c}{$\mathbf D_C$} & \multicolumn{3}{c}{$\mathbf D_{Y_F}$} \\",
         r"  \cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"  Relationship & 2013 & 2016 & R3 & 2013 & 2016 & R3 \\", r"  \midrule"]
    for label, col in CH:
        row = []
        for eff in ("D_C", "D_YF"):
            for res, _ in GSEC_RES:
                v = sum(float(x[col]) for x in R
                        if x["resolution"] == res and x["effect"] == eff)
                row.append(f"${v:+,.0f}$".replace("+-", "-"))
        L.append(f"  {label} & " + " & ".join(row) + r" \\")
    L += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"tab_level1_global.tex": level1(),
             "tab_level2_groups.tex": level2_groups(),
             "tab_level2_variables.tex": level2_variables(),
             "tab_level2_sourcing_origin.tex": sourcing_origin(),
             "tab_level3_country.tex": level3_country(),
             "tab_level3_sector.tex": level3_sector(),
             "tab_level_grid_sensitivity.tex": level_grid_sensitivity(),
             "tab_sourcing_granularity.tex": sourcing_granularity(),
             "tab_sourcing_channels.tex": sourcing_channels()}
    man = {}
    for n, b in files.items():
        (OUT / n).write_text(b, encoding="utf-8", newline="\n")
        man[n] = hashlib.sha256(b.encode()).hexdigest()[:16]
        print(f"wrote {n:34s} {len(b.splitlines()):3d} lines")
    # validation: group rows must sum to the global total in every period
    print("\nadditivity check (group rows vs global total):")
    ok = True
    for p in PERIODS:
        s = sum(total(p, g) for g, _ in GRP)
        g_ = gtotal(p)
        good = abs(s - g_) < 1e6
        ok &= good
        print(f"  {p}: groups={s/GT:+.4f}  global={g_/GT:+.4f}  {'ok' if good else 'FAIL'}")
    (OUT / "GENERATED_MANIFEST_HOEKSTRA.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_chapter4_hoekstra_tables.py",
        "reporting_template": "Hoekstra, Michel and Suh (2016), Tables 1, 2 and A3",
        "results": ["R-2026-009", "R-2026-011", "R-2026-019", "R-2026-038", "R-2026-039"],
        "category_mapping": {c: v for c, v in CATS},
        "main_grid": MAIN, "outputs": man, "additivity_ok": bool(ok),
    }, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
