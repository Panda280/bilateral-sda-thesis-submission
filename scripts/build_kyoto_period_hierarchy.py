#!/usr/bin/env python3
"""Period-cumulative SDA hierarchy along the three Kyoto-timeline periods.

Decision D-2026-031:
  P1  pre-entry-into-force        transitions 2001--2004
  P2  in force, pre-commitment    transitions 2005--2007  ("ratification")
  P3  first commitment period     transitions 2008--2012  (CP1)

Consumes annual-report directories (dashboard contract v2). The main EXIOBASE
campaign covers all three periods on the 2013 / 2016 / R3 grids. The native
`wiod2013_native` configuration uses the WIOD2013 environmental accounts, which
end in 2009, so it is **excluded from P3**; the WIOD2013 grid continues into P3
through `wiod2013_flows2016` (WIOD2016 impacts) and `exio_2013` (EXIOBASE
aggregated impacts). Native rows are flagged `partial` when the campaign does
not yet cover the whole period.

Output: `by_period_hierarchy.parquet` and `KYOTO_PERIODS_REPORT.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PERIODS = {
    "P1_pre_entry_into_force": list(range(2001, 2005)),
    "P2_ratification": list(range(2005, 2008)),
    "P3_first_commitment_period": list(range(2008, 2013)),
}
NODES = ("delta_g", "emission_intensity", "leontief_production_structure", "final_demand",
         "D_v", "D_C", "D_A_x", "D_T_coef", "D_w_coef", "D_YF", "D_YB", "D_y")
SOURCING = ("D_C", "D_YF")

# series id -> (root key, report-dir suffix builder, ea-ends-year or None)
MAIN_GRIDS = {"exio_2016": "2016", "exio_2013": "2013", "exio_R3": "R3"}
NATIVE_CONFIGS = (
    "jrc2019_to_2016", "jrc2019_to_2013", "jrc2019_flows2013",
    "wiod2013_flows2016", "wiod2013_native",
    # Phase 1: genuine WIOD2013-published flow tables (41x35). The five configs
    # above all use EXIOBASE-derived flows.
    "wiod2013_true_native", "jrc2019_wiodflows2013", "exio_wiodflows2013",
    # Phase 2: genuinely native WIOD2016 series (JRC2019/WIOD2016 air
    # account with official WIOD2016-published flows, aggregated 56->50).
    "jrc2019_wiod2016flows_2016",
)
# Environmental-account coverage ceiling: any configuration built on the
# WIOD2013 environmental accounts (impact database WIOD2013) is capped at 2009
# and therefore EXCLUDED from P3 (2008--2012), per Decision D-2026-031 and the
# 2026-09-06 instruction to keep native WIOD2013 impacts out of P3. The
# JRC2019- and EXIOBASE-impact configs on WIOD2013 published flows are NOT
# capped here (their emission accounts run well past 2012); the WIOD2013
# published flow tables themselves end in 2011, but that only makes an
# extended P3 partial, not excluded, and the Phase 1 campaign in any case
# only ran through 2009 so those rows surface as ``partial_campaign``.
NATIVE_EA_LAST_YEAR = {
    "wiod2013_native": 2009,        # WIOD2013 EA + EXIOBASE-derived flows
    "wiod2013_flows2016": 2009,     # WIOD2013 EA + WIOD2016 flows
    "wiod2013_true_native": 2009,   # WIOD2013 EA + WIOD2013 published flows
}


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def load_node_values(report_dir: Path) -> dict[str, dict[str, float]]:
    h = pd.read_parquet(report_dir / "annual_hierarchy.parquet").set_index("node_id")
    out = {}
    for node in NODES:
        if node not in h.index:
            continue
        r = h.loc[node]
        out[node] = {
            "contribution": float(r["contribution"]),
            "positive": float(r["positive_contribution"]),
            "negative": float(r["negative_contribution"]),
            "gross": float(r["gross_contribution"]),
        }
    return out


def collect(series: str, year: int, main_root: Path, native_root: Path) -> dict | None:
    yp = year - 1
    if series in MAIN_GRIDS:
        d = main_root / f"{yp}_{year}_{MAIN_GRIDS[series]}"
    else:
        d = native_root / f"{yp}_{year}_{series}"
    if not (d / "annual_hierarchy.parquet").is_file():
        return None
    return load_node_values(d)


def build(main_root: Path, native_root: Path) -> pd.DataFrame:
    rows = []
    for series in list(MAIN_GRIDS) + list(NATIVE_CONFIGS):
        ea_last = NATIVE_EA_LAST_YEAR.get(series)
        for period, years in PERIODS.items():
            # D-2026-031: a native config whose environmental accounts end before
            # the period is over is not carried as a series for that period, even
            # if it covers the opening years. The grid continues via other series.
            excluded = ea_last is not None and years[-1] > ea_last
            if excluded:
                for node in NODES:
                    rows.append({"series": series, "period": period, "node_id": node,
                                 "status": "excluded",
                                 "reason": f"environmental accounts end {ea_last}; "
                                           f"period runs to {years[-1]} (grid continues via other series)",
                                 "n_transitions": 0, "years": "",
                                 "cumulative_contribution": None, "cumulative_positive": None,
                                 "cumulative_negative": None, "cumulative_gross": None})
                continue
            eligible = [y for y in years if ea_last is None or y <= ea_last]
            present, vals = [], {n: {"contribution": 0.0, "positive": 0.0,
                                     "negative": 0.0, "gross": 0.0} for n in NODES}
            for y in eligible:
                v = collect(series, y, main_root, native_root)
                if v is None:
                    continue
                present.append(y)
                for node, nv in v.items():
                    vals[node]["contribution"] += nv["contribution"]
                    vals[node]["positive"] += nv["positive"]
                    vals[node]["negative"] += nv["negative"]
                    vals[node]["gross"] += nv["gross"]
            if not present:
                status = "missing"
            elif set(present) == set(years):
                status = "complete"
            elif set(present) == set(eligible) and eligible != years:
                status = "partial_ea_coverage"      # EA ends 2009, all available years used
            else:
                status = "partial_campaign"         # campaign not yet run for all years
            for node in NODES:
                rows.append({
                    "series": series, "period": period, "node_id": node, "status": status,
                    "reason": "" if status == "complete" else
                              ("environmental accounts end 2009" if status == "partial_ea_coverage"
                               else "campaign not yet run for the full period" if status == "partial_campaign"
                               else "no reports found"),
                    "n_transitions": len(present),
                    "years": ",".join(map(str, present)),
                    "cumulative_contribution": vals[node]["contribution"] if present else None,
                    "cumulative_positive": vals[node]["positive"] if present else None,
                    "cumulative_negative": vals[node]["negative"] if present else None,
                    "cumulative_gross": vals[node]["gross"] if present else None,
                })
    return pd.DataFrame(rows)


def write_report(path: Path, df: pd.DataFrame) -> None:
    L = ["# SDA hierarchy by Kyoto-timeline period (Decision D-2026-031)", "",
         f"Generated {datetime.now(timezone.utc).isoformat()}. Units: Gt CO2 "
         "(kg / 1e12). Transition year `t` = change `t-1 -> t`.", "",
         "| period | transitions |", "|---|---|",
         "| P1 pre-entry-into-force | 2001--2004 |",
         "| P2 ratification (in force, pre-commitment) | 2005--2007 |",
         "| P3 first commitment period (CP1) | 2008--2012 |", ""]
    prim = ("delta_g", "emission_intensity", "leontief_production_structure", "final_demand")
    for period in PERIODS:
        L += [f"## {period}", "",
              "| series | status | yrs | Δg | intensity | prod. structure | final demand | D_C | D_YF |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        p = df[df.period == period]
        for series in list(MAIN_GRIDS) + list(NATIVE_CONFIGS):
            s = p[p.series == series].set_index("node_id")
            if s.empty:
                continue
            st = s["status"].iloc[0]
            n = int(s["n_transitions"].iloc[0])
            if st in ("excluded", "missing"):
                L.append(f"| {series} | {st} | 0 | -- | -- | -- | -- | -- | -- |")
                continue

            def g(node):
                v = s.loc[node, "cumulative_contribution"]
                return f"{v/1e12:+.2f}" if pd.notna(v) else "--"

            L.append(f"| {series} | {st} | {n} | {g('delta_g')} | {g('emission_intensity')} | "
                     f"{g('leontief_production_structure')} | {g('final_demand')} | "
                     f"{g('D_C')} | {g('D_YF')} |")
        L.append("")
    L += ["## Notes", "",
          "- `wiod2013_native` is **excluded from P3**: the WIOD2013 environmental "
          "accounts end in 2009. The WIOD2013 41x35 grid still appears in P3 via "
          "`wiod2013_flows2016` (WIOD2016 impacts) and `exio_2013` (EXIOBASE "
          "aggregated impacts).",
          "- `partial_ea_coverage`: all available years used, but the EA ends "
          "before the period (native WIOD2013, if ever run past P2).",
          "- `partial_campaign`: the native campaign has not yet been run for every "
          "year of the period (extend with `--year-last 2012` for the D-2026-029/030 "
          "configs; `wiod2013_native` stays 2001--2009).",
          "- 2013--2014 transitions are outside the three periods."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-annual-root", type=Path, required=True)
    ap.add_argument("--native-annual-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--git-commit", default="unknown")
    args = ap.parse_args()

    df = build(args.main_annual_root, args.native_annual_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pq = args.output_root / "by_period_hierarchy.parquet"
    df.to_parquet(pq, index=False)
    write_report(args.output_root / "KYOTO_PERIODS_REPORT.md", df)

    manifest = {
        "schema_version": 1,
        "analysis_id": "kyoto-period-hierarchy-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "D-2026-031",
        "generator": {"script": "implementation/bilateral_sda_attribution/scripts/build_kyoto_period_hierarchy.py",
                      "git_commit": args.git_commit},
        "periods": {k: {"first": v[0], "last": v[-1]} for k, v in PERIODS.items()},
        "outputs": [{"role": "by_period_hierarchy", "path": str(pq), "sha256": sha256(pq),
                     "rows": int(len(df))}],
        "excluded": {"wiod2013_native": "P3 (WIOD2013 environmental accounts end 2009)"},
    }
    (args.output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(df),
                      "status_counts": df.drop_duplicates(["series", "period"])
                      ["status"].value_counts().to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
