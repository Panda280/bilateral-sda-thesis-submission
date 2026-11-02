#!/usr/bin/env python3
"""Level-2 (group) detail for the six flow-table x grid configurations,
EXIOBASE fossil-CO2 impact fixed.

  A exio_R3   B exio_2016   C exio_2013         (EXIOBASE-derived flows)
  D exio_wiod2016flows_2016  (WIOD2016 pub, 50)
  E exio_wiod2016flows_2013  (WIOD2016 pub, 35)
  F exio_wiodflows2013       (WIOD2013 pub, 35, 2001-2009)

Two products per config:
  * emitter-role group marginals for every effect  (thesis Table 4.4)
  * D_C / D_YF split by origin: domestic + cross-border by origin group,
    per destination group                          (thesis Table 4.5)

A/B/C read the registered R-2026-038/039 CSVs. D/E/F are built from the
per-transition CO2_7 group_matrix / country_matrix parquet files with the
analytical-group registry.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

G = 1e12
MT = 1e9   # 1 Mt CO2 = 1e6 t = 1e9 kg
CAMP = Path("/local/s2053268/sda-campaign-2000-2014-v2")
GP = CAMP / "analysis/kyoto_period_group_products_20260908_v2/products"
REG = pd.read_csv(CAMP / "analysis/kyoto_period_group_products_20260908_v2/_inputs/analytical_group_registry.csv")
GROUP = dict(zip(REG.canonical_country, REG.analysis_group))          # ISO3 -> Red/Non-Red/Non-Participant
GROUP_ALT = {}
for c, g in list(GROUP.items()):
    GROUP_ALT[c] = g
# some matrices use 2-letter or lower; also RoW spellings
for extra, g in [("ROW", "Non-Participant"), ("RoW", "Non-Participant"), ("ROU", "Red")]:
    GROUP_ALT[extra] = g

PERIODS = {"P1": range(2001, 2005), "P2": range(2005, 2008), "P3": range(2008, 2013)}
GROUPS = ["Red", "Non-Red", "Non-Participant"]

# CO2_7 effect basename -> (canonical effect, thesis label)
EFF_CO2_7 = {
    "Dint": "D_int", "D_Ax2": "D_A_x", "D_T_coef": "D_T", "D_w_coef": "D_w",
    "D_C_coef": "D_C", "YF": "D_YF", "YB": "D_YB", "DYy": "D_Yy",
    "Total": "total", "Leontief_total": "D_L", "FD_total": "D_fd",
}
# registered-CSV effect id -> canonical
EFF_CSV = {"D_v": "D_int", "D_A_x": "D_A_x", "D_T_coef": "D_T", "D_w_coef": "D_w",
           "D_C": "D_C", "D_YF": "D_YF", "D_YB": "D_YB", "D_y": "D_Yy"}

CO2_7_ROOT = {
    "D": (CAMP / "exio-wiod2016flows-2001-2012-20260908", "exio_wiod2016flows_2016"),
    "E": (CAMP / "exio-wiod2016flows-2013grid-2001-2012-20260908", "exio_wiod2016flows_2013"),
    "F": (CAMP / "native-wiodflows-2001-2009-20260904", "exio_wiodflows2013"),
}
DERIVED = {"A": "R3", "B": "2016", "C": "2013"}


def grp_of(country):
    c = str(country)
    if c in GROUP_ALT:
        return GROUP_ALT[c]
    return None


def canon_group(g):
    return {"Red": "Red", "Non-Red": "Non-Red", "Non-Participant": "Non-Participant",
            "Reduction": "Red", "Non-Reduction": "Non-Red"}.get(g, g)


# ---------- A/B/C from registered CSVs ----------
def abc_emitter(res):
    d = pd.read_csv(GP / "kyoto_period_group_effects.csv")
    d = d[(d.resolution == res) & (d.role == "emitter")]
    out = {}
    for per in PERIODS:
        out[per] = {}
        sub = d[d.period == per]
        for csv_e, eff in EFF_CSV.items():
            r = sub[sub.effect == csv_e]
            out[per][eff] = {canon_group(row.group): row.net for row in r.itertuples()}
    return out


def abc_origin(res):
    d = pd.read_csv(GP / "kyoto_period_group_interactions.csv")
    d = d[d.resolution == res]
    out = {}
    for per in PERIODS:
        out[per] = {}
        for eff in ["D_C", "D_YF"]:
            sub = d[(d.period == per) & (d.effect == eff)]
            # domestic[dest], cross[origin][dest]
            dom = {g: 0.0 for g in GROUPS}
            cross = {o: {g: 0.0 for g in GROUPS} for o in GROUPS}
            for row in sub.itertuples():
                dg = canon_group(row.destination_group)
                og = canon_group(row.source_group)
                if row.relationship == "Domestic":
                    dom[dg] = dom.get(dg, 0.0) + row.net
                else:
                    cross[og][dg] = cross[og].get(dg, 0.0) + row.net
            out[per][eff] = {"domestic": dom, "cross": cross}
    return out


# ---------- D/E/F from CO2_7 matrices ----------
def def_group_marginal(root, cfg, per, base):
    """emitter-role marginal (supplier row sum of the 3x3 group matrix)"""
    acc = {g: 0.0 for g in GROUPS}
    for y in PERIODS[per]:
        f = root / f"{y-1}_{y}_{cfg}" / "CO2_7" / f"{y}_{base}_group_matrix_baseline.parquet"
        if not f.exists():
            return None
        m = pd.read_parquet(f)
        m.index = [canon_group(str(i)) for i in m.index]
        m.columns = [canon_group(str(c)) for c in m.columns]
        rs = m.sum(axis=1)          # supplier marginal = emitter
        for g in GROUPS:
            acc[g] += float(rs.get(g, 0.0))
    return acc


def def_world_tot(root, cfg, per, base):
    acc = 0.0
    for y in PERIODS[per]:
        f = root / f"{y-1}_{y}_{cfg}" / "CO2_7" / f"{y}_{base}_tot_baseline.parquet"
        if not f.exists():
            return None
        acc += float(np.nansum(pd.read_parquet(f).select_dtypes("number").values))
    return acc


def def_origin_split(root, cfg, per, base):
    """domestic (diagonal) + cross-border by origin group, per destination group,
    from the NxN supplier-country x destination-country matrix"""
    dom = {g: 0.0 for g in GROUPS}
    cross = {o: {g: 0.0 for g in GROUPS} for o in GROUPS}
    for y in PERIODS[per]:
        f = root / f"{y-1}_{y}_{cfg}" / "CO2_7" / f"{y}_{base}_country_matrix_baseline.parquet"
        if not f.exists():
            return None
        m = pd.read_parquet(f)
        sup = [str(i) for i in m.index]
        des = [str(c) for c in m.columns]
        A = m.to_numpy(dtype=float)
        for i, s in enumerate(sup):
            gs = grp_of(s)
            if gs is None:
                continue
            for j, d in enumerate(des):
                gd = grp_of(d)
                if gd is None:
                    continue
                v = A[i, j]
                if s == d:
                    dom[gd] += v
                else:
                    cross[gs][gd] += v
    return {"domestic": dom, "cross": cross}


# ---------- assemble ----------
EM = {}   # config -> period -> effect -> {group: net}
OR = {}   # config -> period -> {D_C|D_YF: {domestic, cross}}
for k, res in DERIVED.items():
    EM[k] = abc_emitter(res)
    OR[k] = abc_origin(res)
for k, (root, cfg) in CO2_7_ROOT.items():
    EM[k], OR[k] = {}, {}
    for per in PERIODS:
        EM[k][per] = {}
        for base, eff in EFF_CO2_7.items():
            if eff in ("total", "D_L", "D_fd"):
                continue
            EM[k][per][eff] = def_group_marginal(root, cfg, per, base)
        OR[k][per] = {}
        for base, eff in [("D_C_coef", "D_C"), ("YF", "D_YF")]:
            OR[k][per][eff] = def_origin_split(root, cfg, per, base)

# ---------- verification: emitter marginals reconstruct the world effect ----------
print("=== VERIFY: sum of emitter group marginals vs world effect (Gt) ===")
for k in ["D", "E", "F"]:
    root, cfg = CO2_7_ROOT[k]
    for per in PERIODS:
        if k == "F" and per == "P3":
            continue
        for base, eff in [("Dint", "D_int"), ("D_C_coef", "D_C"), ("YF", "D_YF"), ("DYy", "D_Yy")]:
            gm = EM[k][per].get(eff)
            wt = def_world_tot(root, cfg, per, base)
            if gm is None or wt is None:
                continue
            s = sum(gm.values())
            flag = "" if abs(s - wt) < 5e9 else "  <-- MISMATCH"
            print(f"  {k} {per} {eff:6}: marginals {s/G:+.3f}  world {wt/G:+.3f}{flag}")

# ---------- dump for table generation ----------
def ser(d):
    if isinstance(d, dict):
        return {kk: ser(vv) for kk, vv in d.items()}
    if d is None:
        return None
    return round(float(d) / MT, 1)

Path("/tmp/level2_sixconfig.json").write_text(json.dumps(
    {"emitter_Mt": ser(EM), "origin_Mt": ser(OR)}, indent=1))
print("\nwrote /tmp/level2_sixconfig.json")

# ---------- quick trend view: emitter marginals, Mt, P1 & P2 ----------
for per in ["P1", "P2"]:
    print(f"\n=== EMITTER GROUP MARGINALS  {per}  (Mt CO2) ===")
    print(f"  {'effect/group':<22}" + "".join(f"{c:>9}" for c in "ABCDEF"))
    for eff in ["D_int", "D_A_x", "D_T", "D_w", "D_C", "D_YF", "D_YB", "D_Yy"]:
        for g in GROUPS:
            row = []
            for c in "ABCDEF":
                v = EM[c][per].get(eff, {})
                v = None if v is None else v.get(g)
                row.append(f"{v/MT:>+9.0f}" if v is not None else f"{'-':>9}")
            print(f"  {eff+' / '+g:<22}" + "".join(row))
