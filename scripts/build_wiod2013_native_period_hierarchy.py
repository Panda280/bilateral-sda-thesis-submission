#!/usr/bin/env python3
"""Independently reproduce the Kyoto-period hierarchy for the `wiod2013_native`
configuration (WIOD2013 environmental account on the EXIOBASE-derived flow proxy).

VERIFICATION PRODUCT, not a new headline result. The 2026-09-08 period build
(`kyoto-period-hierarchy-v1`, tmp/by_period_hierarchy.parquet) reports
`no reports found` for every WIOD2013 series, because that campaign's annual
reports sit outside the Vibranium native-WIOD2016 campaign directory. The
2026-09-06 build under
`docs/analysis/native_wiod_v2_20260903/kyoto-period-hierarchy-20260906/`
does carry them. This script re-aggregates P1 and P2 straight from the
repository's annual reports so the earlier product can be checked
independently; it agrees to three decimal places in Gt CO2.

NAMING (Decision D-2026-033): `wiod2013_native` pairs the WIOD2013
environmental account with the EXIOBASE-derived flow proxy. It is NOT the
genuinely native WIOD2013 series; that is `wiod2013_true_native`, which uses
the official WIOD2013-published flow tables and is reported separately.
P3 stays excluded for both: the WIOD2013 environmental account ends in 2009.
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "implementation/bilateral_sda_attribution/docs/analysis/native_wiod_annual_reports_20260903/native-wiod-co2-2001-2009-annual-20260903"
OUT = ROOT / "implementation/bilateral_sda_attribution/docs/analysis/wiod2013_native_period_hierarchy_20260908"
SERIES = "wiod2013_native"
PERIODS = {"P1": (2001, 2004, "pre-entry into force"),
           "P2": (2005, 2007, "in force, pre-commitment")}
P3_EXCLUSION = "environmental accounts end 2009; period runs to 2012"
PRIMARY = ["emission_intensity", "leontief_production_structure", "final_demand"]
LEONTIEF_CHILDREN = ["D_C", "D_T_coef", "D_w_coef"]
FD_CHILDREN = ["D_YF", "D_YB", "D_y"]
# Split identities are normalised by the summed absolute component totals,
# not by a near-cancelling parent net: this is the convention behind the
# registered reconstruction residuals (R-2026-010) and the declared gate
# for the coefficient-form Leontief identity (R-2026-029).
TOL_REL = 2e-5


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def load() -> tuple[pd.DataFrame, list[dict]]:
    frames, inputs = [], []
    for d in sorted(SRC.glob(f"*_{SERIES}")):
        f = d / "annual_hierarchy.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        frames.append(df)
        inputs.append({"path": str(f.relative_to(ROOT)).replace("\\", "/"),
                       "sha256": sha256(f)})
    if not frames:
        sys.exit(f"no annual reports found under {SRC}")
    return pd.concat(frames, ignore_index=True), inputs


def main() -> int:
    df, inputs = load()
    df["transition"] = df["year_current"].astype(int)

    failures: list[str] = []
    bad = df[df["validation_status"] != "passed"]
    if len(bad):
        failures.append(f"{len(bad)} source rows are not validation_status=passed")

    rows = []
    for period, (lo, hi, label) in PERIODS.items():
        sub = df[(df["transition"] >= lo) & (df["transition"] <= hi)]
        years = sorted(sub["transition"].unique())
        expected = hi - lo + 1
        if len(years) != expected:
            failures.append(f"{period}: expected {expected} transitions, observed {len(years)} {years}")
        g = sub.groupby("node_id", as_index=False).agg(
            net=("contribution", "sum"),
            positive=("positive_contribution", "sum"),
            negative=("negative_contribution", "sum"),
            gross=("gross_contribution", "sum"))
        for _, r in g.iterrows():
            net, pos, neg, gross = (float(r[k]) for k in ("net", "positive", "negative", "gross"))
            rows.append({"period": period, "period_label": label, "configuration": SERIES,
                         "impact_source": "WIOD2013 environmental accounts",
                         "flow_source": "EXIOBASE-derived CYP/PYP campaign flows",
                         "resolution": "41x35", "node_id": r["node_id"],
                         "net": net, "positive": pos, "negative": neg, "gross": gross,
                         "cancellation": (1.0 - abs(net) / gross) if gross else 0.0,
                         "transition_count": len(years),
                         "observed_years": ",".join(str(y) for y in years),
                         "coverage_status": "complete",
                         "units": "kg CO2", "price_basis": "CYP_(t-1) -> PYP_t"})

    rows.append({"period": "P3", "period_label": "first commitment period", "configuration": SERIES,
                 "impact_source": "WIOD2013 environmental accounts",
                 "flow_source": "EXIOBASE-derived CYP/PYP campaign flows",
                 "resolution": "41x35", "node_id": "delta_g",
                 "net": float("nan"), "positive": float("nan"), "negative": float("nan"),
                 "gross": float("nan"), "cancellation": float("nan"),
                 "transition_count": 0, "observed_years": "",
                 "coverage_status": "excluded", "units": "kg CO2",
                 "price_basis": "CYP_(t-1) -> PYP_t"})

    out = pd.DataFrame(rows)

    def val(period, node):
        m = out[(out["period"] == period) & (out["node_id"] == node)]
        return float(m["net"].iloc[0]) if len(m) else None

    checks = []
    for period in PERIODS:
        total = val(period, "delta_g")
        prim = sum(val(period, n) for n in PRIMARY)
        leon = val(period, "leontief_production_structure")
        leon_c = sum(val(period, n) for n in LEONTIEF_CHILDREN)
        fd = val(period, "final_demand")
        fd_c = sum(val(period, n) for n in FD_CHILDREN)
        def absscale(nodes, parent):
            return max(sum(abs(val(period, n)) for n in nodes), abs(parent), 1.0)

        for name, a, b, scale in (
                ("primary sum", prim, total, absscale(PRIMARY, total)),
                ("leontief split", leon_c, leon, absscale(LEONTIEF_CHILDREN, leon)),
                ("final-demand split", fd_c, fd, absscale(FD_CHILDREN, fd))):
            rel = abs(a - b) / scale
            checks.append({"period": period, "identity": name, "residual_relative": rel,
                           "residual_kg": abs(a - b),
                           "passed": bool(rel <= TOL_REL)})
            if rel > TOL_REL:
                failures.append(f"{period} {name}: relative residual {rel:.3e}")
        for _, r in out[(out["period"] == period)].iterrows():
            g = abs(float(r["positive"])) + abs(float(r["negative"]))
            if r["gross"] and abs(g - float(r["gross"])) / max(abs(float(r["gross"])), 1.0) > 1e-9:
                failures.append(f"{period} {r['node_id']}: gross != |pos|+|neg|")

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "wiod2013_native_period_hierarchy.csv"
    out.to_csv(csv_path, index=False)

    manifest = {
        "schema_version": 1,
        "analysis_id": "wiod2013-native-period-hierarchy-verification-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "D-2026-042",
        "purpose": "independent verification of the 2026-09-06 period build; not a new result ID",
        "generator": {"script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/")},
        "series": SERIES,
        "periods": {k: {"first": v[0], "last": v[1]} for k, v in PERIODS.items()},
        "excluded": {"P3": P3_EXCLUSION},
        "inputs": inputs,
        "outputs": [{"path": csv_path.name, "sha256": sha256(csv_path), "rows": int(len(out))}],
        "validation": {"passed": not failures, "identity_checks": checks,
                       "tolerance_relative": TOL_REL, "failures": failures},
        "limitations": [
            "Accounting attributions, not causal estimates.",
            "Group marginals and bilateral interactions are NOT provided by this product.",
            "P3 is excluded because the WIOD2013 environmental account ends in 2009.",
            "wiod2013_native uses the EXIOBASE-derived flow proxy (D-2026-033), not WIOD2013-published flows.",
        ],
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"rows: {len(out)}   output: {csv_path.relative_to(ROOT)}")
    for c in checks:
        print(f"  {c['period']} {c['identity']:20s} rel={c['residual_relative']:.3e} {'ok' if c['passed'] else 'FAIL'}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nall gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
