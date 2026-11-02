#!/usr/bin/env python3
"""Generate the Chapter 4 period/group LaTeX tables from the registered products.

Sources (read-only):
  R-2026-038  kyoto_period_group_effects.csv        group marginals
  R-2026-039  kyoto_period_group_interactions.csv   group-pair + relationship cube
  R-2026-039  coverage_register.csv                 EXIOBASE coverage
  R-2026-037  tmp/by_period_hierarchy.parquet       native WIOD2016 hierarchy
  R-2026-035/036  native_wiod_v2 kyoto-period-hierarchy-20260906  WIOD2013 series

No value is typed by hand: every number in the emitted tables is read from a
registered product. Tables are written to src/chapters/results_revised/generated/.
"""
from __future__ import annotations
import csv, json, hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROD = ROOT / "implementation/bilateral_sda_attribution/docs/analysis/kyoto_period_group_products_20260908"
NAT16 = ROOT / "tmp/by_period_hierarchy.parquet"
NAT13 = ROOT / "implementation/bilateral_sda_attribution/docs/analysis/native_wiod_v2_20260903/kyoto-period-hierarchy-20260906/by_period_hierarchy.parquet"
OUT = ROOT / "src/chapters/results_revised/generated"
GT = 1e12
MAIN = "exio_2016"
GRIDS = ["exio_2013", "exio_2016", "exio_R3"]
GROUPS = [("Red", "Reduction"), ("Non-Red", "Non-Reduction"), ("Non-Participant", "Non-Participation")]
LEAVES = [("D_v", "emitter", r"$\mathbf D_{\mathrm{int}}$", "emitter"),
          ("D_C", "structural_driver", r"$\mathbf D_C$", "str.\\ driver"),
          ("D_T_coef", "structural_driver", r"$\mathbf D_T$", "str.\\ driver"),
          ("D_w_coef", "structural_driver", r"$\mathbf D_w$", "str.\\ driver"),
          ("D_YF", "consumer", r"$\mathbf D_{Y_F}$", "consumer"),
          ("D_YB", "consumer", r"$\mathbf D_{Y_B}$", "consumer"),
          ("D_y", "consumer", r"$\mathbf D_{Y_y}$", "consumer")]
PERIOD_LABEL = {"P1": "P1, 2001--2004", "P2": "P2, 2005--2007", "P3": "P3, 2008--2012"}


def load():
    E = list(csv.DictReader(open(PROD / "kyoto_period_group_effects.csv")))
    I = list(csv.DictReader(open(PROD / "kyoto_period_group_interactions.csv")))
    C = list(csv.DictReader(open(PROD / "coverage_register.csv")))
    return E, I, C


def eff(E, period, effect, role, group, cfg=MAIN, field="net"):
    for r in E:
        if (r["period"] == period and r["configuration"] == cfg and r["effect"] == effect
                and r["role"] == role and r["group"] == group):
            return float(r[field])
    return None


def inter(I, period, effect, cfg=MAIN, rel=None, sg=None, dg=None, field="net"):
    t = None
    for r in I:
        if (r["period"] == period and r["configuration"] == cfg and r["effect"] == effect
                and (rel is None or r["relationship"] == rel)
                and (sg is None or r["source_group"] == sg)
                and (dg is None or r["destination_group"] == dg)):
            t = (t or 0.0) + float(r[field])
    return t


def fmt(v, dp=3):
    if v is None:
        return "---"
    return f"${v:+.{dp}f}$".replace("+-", "-")


def period_table(E, period):
    lines = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
             r"  \caption{Group-attributed effects in " + PERIOD_LABEL[period] +
             r". Net contributions in Gt CO$_2$ with the cancellation percentage of each"
             r" group's own gross activity. EXIOBASE sources on the $2016$-compatible target"
             r" grid; attribution roles are kept separate and never summed"
             r" (Result R-2026-038).}",
             r"  \label{tab:period-group-effects-" + period.lower() + "}",
             r"  \setlength{\tabcolsep}{4pt}",
             r"  \begin{tabular}{llrrrrrr}", r"    \toprule",
             r"    & & \multicolumn{2}{c}{Reduction} & \multicolumn{2}{c}{Non-Reduction}"
             r" & \multicolumn{2}{c}{Non-Participation} \\",
             r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
             r"    Effect & Role & Net & Canc. & Net & Canc. & Net & Canc. \\", r"    \midrule"]
    for code, role, sym, rlabel in LEAVES:
        cells = []
        for gcode, _ in GROUPS:
            net = eff(E, period, code, role, gcode)
            canc = eff(E, period, code, role, gcode, field="cancellation")
            cells.append(fmt(None if net is None else net / GT))
            cells.append("---" if canc is None else f"${canc * 100:.0f}\\%$")
        lines.append(f"    {sym} & {rlabel} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def sourcing_table(I):
    lines = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
             r"  \caption{The two registered bilateral sourcing effects by period and"
             r" relationship, EXIOBASE sources on the $2016$-compatible grid. Net"
             r" contributions in Gt CO$_2$; the gross shares are each relationship's share"
             r" of the effect's period gross activity. Domestic means the source and"
             r" destination country coincide before group aggregation, so a same-group cell"
             r" contains both domestic and cross-border contributions"
             r" (Result R-2026-039).}",
             r"  \label{tab:period-sourcing-relationship}",
             r"  \setlength{\tabcolsep}{5pt}",
             r"  \begin{tabular}{llrrrrr}", r"    \toprule",
             r"    & & \multicolumn{3}{c}{Net (Gt CO$_2$)} & \multicolumn{2}{c}{Gross share} \\",
             r"    \cmidrule(lr){3-5}\cmidrule(l){6-7}",
             r"    Effect & Period & Total & Domestic & Cross-border & Dom. & Cross-b. \\",
             r"    \midrule"]
    for code, sym in (("D_C", r"$\mathbf D_C$"), ("D_YF", r"$\mathbf D_{Y_F}$")):
        for i, p in enumerate(["P1", "P2", "P3"]):
            tot = inter(I, p, code) / GT
            dom = inter(I, p, code, rel="Domestic") / GT
            xb = inter(I, p, code, rel="Cross-border") / GT
            dg = inter(I, p, code, rel="Domestic", field="gross")
            xg = inter(I, p, code, rel="Cross-border", field="gross")
            lines.append(f"    {sym if i == 0 else ''} & {PERIOD_LABEL[p]} & {fmt(tot)} & {fmt(dom)} "
                         f"& {fmt(xb)} & ${dg / (dg + xg) * 100:.0f}\\%$ & ${xg / (dg + xg) * 100:.0f}\\%$ \\\\")
        if code == "D_C":
            lines.append(r"    \addlinespace")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def robustness_table(E, I):
    """Each retained group-period claim, evaluated on all three target grids."""
    rows = []
    for p in ["P1", "P2", "P3"]:
        rows.append((f"{p}: $\\mathbf D_C$ net, all groups",
                     [inter(I, p, "D_C", cfg=g) / GT for g in GRIDS], "net"))
        rows.append((f"{p}: $\\mathbf D_C$ domestic net",
                     [inter(I, p, "D_C", cfg=g, rel="Domestic") / GT for g in GRIDS], "net"))
        rows.append((f"{p}: $\\mathbf D_{{Y_F}}$ cross-border net",
                     [inter(I, p, "D_YF", cfg=g, rel="Cross-border") / GT for g in GRIDS], "net"))
        rows.append((f"{p}: $\\mathbf D_{{\\mathrm{{int}}}}$ Non-Participation net",
                     [eff(E, p, "D_v", "emitter", "Non-Participant", cfg=g) / GT for g in GRIDS], "net"))
    lines = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
             r"  \caption{Target-grid robustness of the retained group-period statements."
             r" Each row is the same quantity recomputed on the three target grids"
             r" (Gt CO$_2$). \emph{Sign} records whether all three grids agree in direction;"
             r" the grids are not pooled and no preferred grid is selected"
             r" (Results R-2026-038 and R-2026-039).}",
             r"  \label{tab:period-group-grid-robustness}",
             r"  \setlength{\tabcolsep}{5pt}",
             r"  \begin{tabular}{lrrrl}", r"    \toprule",
             r"    Statement & $2013$ & $2016$ & R3 & Sign \\", r"    \midrule"]
    prev = None
    for label, vals, _ in rows:
        if prev and label[:2] != prev:
            lines.append(r"    \addlinespace")
        prev = label[:2]
        agree = "same" if len({v > 0 for v in vals}) == 1 else r"\textbf{differs}"
        lines.append(f"    {label} & " + " & ".join(fmt(v) for v in vals) + f" & {agree} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def coverage_table(C):
    n16 = pd.read_parquet(NAT16)
    n13 = pd.read_parquet(NAT13)

    def counts(df, series):
        out = {}
        for _, r in df[(df.series == series) & (df.node_id == "delta_g")].iterrows():
            p = str(r["period"])[:2]
            out[p] = (int(r["n_transitions"]), str(r["status"]))
        return out

    series_rows = [
        ("EXIOBASE fossil-origin CO$_2$", "EXIOBASE-derived", "$2013$, $2016$, R3",
         {p: (int(next(x for x in C if x["configuration"] == "exio_2016" and x["period"] == p)["observed_transitions"]), "complete") for p in ["P1", "P2", "P3"]},
         "Group marginals for the registered roles; bilateral interactions for $\\mathbf D_C$ and $\\mathbf D_{Y_F}$ only"),
        ("JRC2019 (WIOD-compatible)", "WIOD2016-published", "$44\\times50$",
         counts(n16, "jrc2019_wiod2016flows_2016"),
         "Hierarchy only: total, primary effects and registered subeffects. No group or bilateral product"),
        ("WIOD2013 accounts", "WIOD2013-published", "$41\\times35$",
         counts(n13, "wiod2013_true_native"),
         "Hierarchy only, P1 and P2. Excluded from P3: the account ends in 2009"),
        ("WIOD2013 accounts", "EXIOBASE-derived", "$41\\times35$",
         counts(n13, "wiod2013_native"),
         "Hierarchy only, P1 and P2. A mixed-source series (D-2026-033), not native WIOD2013"),
    ]
    lines = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
             r"  \caption{Coverage and admissible use of the active empirical series."
             r" Counts are adjacent-year transitions; P1, P2 and P3 require exactly four,"
             r" three and five. A dash is an exclusion, not a zero. Every series is named by"
             r" its emission account, its flow source and its target grid, because a change"
             r" of emission account alone does not make a series native to a database"
             r" (Decisions D-2026-033 and D-2026-042; Results R-2026-037, R-2026-038 and"
             r" R-2026-039).}",
             r"  \label{tab:results-coverage}", r"  \setlength{\tabcolsep}{4pt}",
             r"  \begin{tabular}{>{\raggedright\arraybackslash}p{0.17\textwidth}"
             r">{\raggedright\arraybackslash}p{0.13\textwidth}"
             r">{\raggedright\arraybackslash}p{0.09\textwidth}ccc"
             r">{\raggedright\arraybackslash}p{0.29\textwidth}}", r"    \toprule",
             r"    Emission account & Flow source & Grid & P1 & P2 & P3 & Admissible use \\",
             r"    \midrule"]
    for acct, flow, grid, cnt, use in series_rows:
        cells = []
        for p in ["P1", "P2", "P3"]:
            n, status = cnt.get(p, (0, "missing"))
            cells.append(str(n) if status == "complete" and n else "---")
        lines.append(f"    {acct} & {flow} & {grid} & " + " & ".join(cells) + f" & {use} \\\\")
        lines.append(r"    \addlinespace[2pt]")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    E, I, C = load()
    OUT.mkdir(parents=True, exist_ok=True)
    written = {}
    files = {"tab_coverage.tex": coverage_table(C),
             "tab_sourcing_cross_period.tex": sourcing_table(I),
             "tab_grid_robustness.tex": robustness_table(E, I)}
    for p in ["P1", "P2", "P3"]:
        files[f"tab_group_effects_{p}.tex"] = period_table(E, p)
    for name, body in files.items():
        (OUT / name).write_text(body, encoding="utf-8", newline="\n")
        written[name] = hashlib.sha256(body.encode()).hexdigest()[:16]
        print(f"wrote {name:34s} {len(body.splitlines()):3d} lines  sha={written[name]}")
    (OUT / "GENERATED_MANIFEST.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "implementation/bilateral_sda_attribution/scripts/build_chapter4_period_group_tables.py",
        "decision": "D-2026-042",
        "results": ["R-2026-035", "R-2026-036", "R-2026-037", "R-2026-038", "R-2026-039"],
        "main_grid": MAIN, "outputs": written,
        "note": "Every value is read from a registered product; none is hand-entered.",
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
