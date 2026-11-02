#!/usr/bin/env python3
"""Level 3 (country / sector) for the six flow-table x grid configs, EXIOBASE
impact fixed. Emitter-role cumulative net per Kyoto period + a concentration
measure (how many sectors carry 80% of the absolute contribution).

  A exio_R3  B exio_2016  C exio_2013   (drilldown, EXIOBASE-derived flows)
  D exio_wiod2016flows_2016  E exio_wiod2016flows_2013  F exio_wiodflows2013
F ends 2009 -> P3 not reported.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

MT = 1e9
CAMP = Path("/local/s2053268/sda-campaign-2000-2014-v2")
DRILL = CAMP / "annual-resume-20260830"
PERIODS = {"P1": range(2001, 2005), "P2": range(2005, 2008), "P3": range(2008, 2013)}
NO_P3 = {"F"}
CO2_7 = {"D": (CAMP / "exio-wiod2016flows-2001-2012-20260908", "exio_wiod2016flows_2016"),
         "E": (CAMP / "exio-wiod2016flows-2013grid-2001-2012-20260908", "exio_wiod2016flows_2013"),
         "F": (CAMP / "native-wiodflows-2001-2009-20260904", "exio_wiodflows2013")}
DERIVED = {"A": "R3", "B": "2016", "C": "2013"}
B2E = {"Dint": "D_int", "D_Ax2": "D_A_x", "D_C_coef": "D_C", "D_T_coef": "D_T",
       "D_w_coef": "D_w", "YF": "D_YF"}
D2E = {"D_v": "D_int", "D_A_x": "D_A_x", "D_C": "D_C", "D_T_coef": "D_T",
       "D_w_coef": "D_w", "D_YF": "D_YF"}
EFFS = list(dict.fromkeys(B2E.values()))
ISO2TO3 = {"AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU",
           "DK": "DNK", "EE": "EST", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR",
           "GR": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LT": "LTU",
           "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT",
           "RO": "ROU", "SE": "SWE", "SI": "SVN", "SK": "SVK", "AU": "AUS", "BR": "BRA",
           "CA": "CAN", "CH": "CHE", "CN": "CHN", "ID": "IDN", "IN": "IND", "JP": "JPN",
           "KR": "KOR", "MX": "MEX", "NO": "NOR", "RU": "RUS", "TR": "TUR", "TW": "TWN",
           "US": "USA"}


def drill_period(res, grain, per):
    acc = {}
    for y in PERIODS[per]:
        f = DRILL / f"{y-1}_{y}_{res}" / "annual_drilldown.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["grain_id", "effect_id", grain, "contribution"])
        d = d[d.grain_id == grain]
        for eid, eff in D2E.items():
            s = d[d.effect_id == eid].groupby(grain)["contribution"].sum()
            for k, val in s.items():
                acc.setdefault(eff, {}).setdefault(str(k), 0.0)
                acc[eff][str(k)] += float(val)
    return acc


def co2_period(root, cfg, axis, per):
    acc = {}
    for y in PERIODS[per]:
        cd = root / f"{y-1}_{y}_{cfg}" / "CO2_7"
        if not cd.exists():
            continue
        for base, eff in B2E.items():
            f = cd / f"{y}_{base}_tot_baseline.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f).reset_index()
            col = df.columns[-1]
            s = df.groupby(axis)[col].sum()
            for k, val in s.items():
                acc.setdefault(eff, {}).setdefault(str(k), 0.0)
                acc[eff][str(k)] += float(val)
    return acc


COUNTRY = {c: {} for c in "ABCDEF"}
SECTOR = {c: {} for c in "ABCDEF"}
for per in PERIODS:
    for k, res in DERIVED.items():
        COUNTRY[k][per] = drill_period(res, "emitter_country", per)
        SECTOR[k][per] = drill_period(res, "emitter_sector", per)
    for k, (root, cfg) in CO2_7.items():
        if per == "P3" and k in NO_P3:
            COUNTRY[k][per] = {}
            SECTOR[k][per] = {}
            continue
        COUNTRY[k][per] = co2_period(root, cfg, "Country", per)
        SECTOR[k][per] = co2_period(root, cfg, "Sector", per)


def norm_c(c):
    return ISO2TO3.get(c, c)


def concentration(dct):
    vs = sorted((abs(x) for x in dct.values()), reverse=True)
    tot = sum(vs)
    if tot == 0:
        return 0, len(vs)
    run = 0.0
    for i, x in enumerate(vs, 1):
        run += x
        if run >= 0.8 * tot:
            return i, len(vs)
    return len(vs), len(vs)


def top_by_abs(dct, n):
    return sorted(dct.items(), key=lambda kv: -abs(kv[1]))[:n]


def ser(dct):
    return {e: {k: round(vv / MT, 1) for k, vv in d.items()} for e, d in dct.items()}


out = {
    "country_Mt": {c: {p: ser(COUNTRY[c][p]) for p in PERIODS} for c in "ABCDEF"},
    "sector_Mt": {c: {p: ser(SECTOR[c][p]) for p in PERIODS} for c in "ABCDEF"},
    "concentration": {c: {p: {e: concentration(SECTOR[c][p].get(e, {})) for e in EFFS}
                          for p in PERIODS} for c in "ABCDEF"},
}
Path("/tmp/level3_sixconfig.json").write_text(json.dumps(out, indent=1))
print("wrote /tmp/level3_sixconfig.json")

for per in PERIODS:
    print(f"\n=== {per}  emitter country D_int (Mt) ===")
    tops = set()
    for k in "ABCDEF":
        for c, _ in top_by_abs(COUNTRY[k][per].get("D_int", {}), 6):
            tops.add(norm_c(c))
    for c3 in sorted(tops):
        row = []
        for k in "ABCDEF":
            dd = COUNTRY[k][per].get("D_int", {})
            val = dd.get(c3) or dd.get({v: kk for kk, v in ISO2TO3.items()}.get(c3, ""))
            if c3 == "ROW":
                val = sum(x for kk, x in dd.items() if kk in ("WF", "WM", "WA", "WE", "WL", "WT"))
            row.append(f"{val/MT:+.0f}" if val else "-")
        print(f"  {c3:<5} " + "  ".join(f"{k}{x}" for k, x in zip("ABCDEF", row)))
