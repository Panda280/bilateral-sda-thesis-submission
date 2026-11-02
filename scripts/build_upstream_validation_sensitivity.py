"""Audit CYP/PYP inputs and paired GRAS/non-GRAS sensitivity evidence.

Inputs are an explicit CSV inventory and strict JSON evidence records. Missing
or failed checks remain visible; the script never infers an indicator from a
filename or converts downstream SDA reconciliation into upstream acceptance.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, shutil, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

CHECKS = ("gras_convergence", "reproduction", "row_balance", "column_balance", "leontief")
TREATMENTS = ("gras", "non_gras")
INVENTORY_COLUMNS = ("year", "treatment", "evidence_json", "indicator_manifest", "canonical_manifest")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def _load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    p = Path(value); return p if p.is_absolute() else (base / p).resolve()


def _block(output: Path, blockers: list[str], command: list[str], inputs: list[dict]) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    report = output / "upstream_validation_readiness.md"
    report.write_text("# Upstream validation and balancing-sensitivity readiness\n\nStatus: **blocked**.\n\n" +
                      "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
    manifest = _manifest("blocked", command, inputs, [], blockers, False)
    manifest["outputs"] = [{"role":"blocker-report", "path":report.name, "sha256":sha(report), "rows":None}]
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _manifest(status, command, inputs, outputs, limitations, passed):
    try:
        git = subprocess.run(["git","describe","--always","--dirty"], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True,text=True,check=True).stdout.strip()
    except Exception: git = "unavailable"
    return {"schema_version":1,"analysis_id":"upstream-cyp-pyp-balancing-sensitivity-v1",
            "status":status,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
            "generator":{"script":"scripts/build_upstream_validation_sensitivity.py","git_commit":git,"command":command},
            "scope":{"price_basis":"CYP_(t-1) -> PYP_t","indicator":"read from indicator manifests",
                     "units_input":"read from indicator manifests","units_output":"unchanged",
                     "effects":["total_change","D_v","D_C","D_A_x","D_T_coef","D_w_coef","D_YF","D_YB","D_y"],
                     "resolutions":[],"year_previous_min":None,"year_current_max":None},
            "coverage":{"expected":30,"accepted":0,"missing":[],"excluded":[]},
            "inputs":inputs,"outputs":outputs,
            "validation":{"passed":passed,"gates":{},"upstream_input_status":"passed" if passed else "failed",
                          "thesis_status":"eligible" if passed else "ineligible"},
            "limitations":limitations}


def build(inventory: Path, output: Path, *, command=None, expected_years=range(2000,2015)) -> Path:
    base=inventory.parent; blockers=[]; inputs=[]; records=[]; seen=set()
    if not inventory.is_file(): return _block(output,[f"missing inventory: {inventory}"],command or [],[])
    inputs.append({"role":"paired-treatment-inventory","path":str(inventory.resolve()),"sha256":sha(inventory)})
    with inventory.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or not set(INVENTORY_COLUMNS)<=set(reader.fieldnames):
            return _block(output,[f"inventory must contain {INVENTORY_COLUMNS}"],command or [],inputs)
        inventory_rows=list(reader)
    for row in inventory_rows:
        try: year=int(row["year"])
        except ValueError: blockers.append(f"invalid year: {row['year']!r}"); continue
        treatment=row["treatment"].strip()
        key=(year,treatment)
        if treatment not in TREATMENTS: blockers.append(f"{year}: invalid treatment {treatment!r}"); continue
        if key in seen: blockers.append(f"duplicate inventory row: {year}/{treatment}"); continue
        seen.add(key)
        paths={name:_resolve(base,row[name]) for name in ("evidence_json","indicator_manifest","canonical_manifest")}
        for role,p in paths.items():
            if not p.is_file(): blockers.append(f"{year}/{treatment}: missing {role}: {p}")
            else: inputs.append({"role":role,"path":str(p),"sha256":sha(p)})
        if not all(p.is_file() for p in paths.values()): continue
        try: evidence,indicator,canonical=map(_load,(paths["evidence_json"],paths["indicator_manifest"],paths["canonical_manifest"]))
        except (OSError,json.JSONDecodeError) as e: blockers.append(f"{year}/{treatment}: invalid JSON: {e}"); continue
        if evidence.get("year")!=year or evidence.get("treatment")!=treatment:
            blockers.append(f"{year}/{treatment}: evidence identity mismatch")
        dims=evidence.get("dimensions",{})
        if not all(isinstance(dims.get(k),int) and dims[k]>0 for k in ("rows","columns","final_demand_columns")):
            blockers.append(f"{year}/{treatment}: missing positive dimensions")
        checks=evidence.get("checks",{})
        for name in CHECKS:
            check=checks.get(name)
            if not isinstance(check,dict) or not isinstance(check.get("passed"),bool) or not all(k in check for k in ("max_abs_error","tolerance")):
                blockers.append(f"{year}/{treatment}: incomplete {name} check")
        consumed=evidence.get("consumed_files",[])
        if not consumed: blockers.append(f"{year}/{treatment}: no consumed_files with hashes")
        for item in consumed:
            p=Path(item.get("path",""))
            if not p.is_file(): blockers.append(f"{year}/{treatment}: consumed file missing: {p}")
            elif item.get("sha256")!=sha(p): blockers.append(f"{year}/{treatment}: consumed file hash mismatch: {p}")
        for required in ("indicator","units","characterization_method","source_files"):
            if not indicator.get(required): blockers.append(f"{year}/{treatment}: indicator manifest missing {required}")
        if indicator.get("indicator") not in ("fossil-origin CO2","GWP100"):
            blockers.append(f"{year}/{treatment}: unsupported/unresolved indicator identity")
        effects=set(canonical.get("effects",{}))
        required_effects={"D_v","D_C","D_A_x","D_T_coef","D_w_coef","D_YF","D_YB","D_y"}
        if effects!=required_effects: blockers.append(f"{year}/{treatment}: canonical effect hierarchy inventory mismatch")
        records.append({"year":year,"treatment":treatment,"indicator":indicator.get("indicator"),"units":indicator.get("units"),
                        "rows":dims.get("rows"),"columns":dims.get("columns"),"final_demand_columns":dims.get("final_demand_columns"),
                        **{f"{n}_passed":checks.get(n,{}).get("passed") for n in CHECKS},
                        **{f"{n}_max_abs_error":checks.get(n,{}).get("max_abs_error") for n in CHECKS},
                        **{f"{n}_tolerance":checks.get(n,{}).get("tolerance") for n in CHECKS}})
    expected={(y,t) for y in expected_years for t in TREATMENTS}; missing=sorted(expected-seen)
    blockers += [f"missing paired treatment: {y}/{t}" for y,t in missing]
    by_year={}
    for r in records: by_year.setdefault(r["year"],[]).append(r)
    for year,pair in by_year.items():
        if len(pair)==2:
            for key in ("indicator","units","rows","columns","final_demand_columns"):
                if pair[0][key]!=pair[1][key]: blockers.append(f"{year}: paired-treatment {key} mismatch")
    if blockers: return _block(output,blockers,command or [],inputs)
    staging=output.parent/f".staging-{output.name}-{uuid.uuid4().hex}"; staging.mkdir(parents=True)
    try:
        validation=staging/"upstream_validation.csv"; fields=list(records[0])
        with validation.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
        sensitivity=[]
        for year in expected_years:
            pair={r["treatment"]:r for r in records if r["year"]==year}
            out={"year":year,"indicator":pair["gras"]["indicator"],"units":pair["gras"]["units"]}
            for name in CHECKS: out[f"{name}_max_abs_error_difference_gras_minus_non_gras"]=pair["gras"][f"{name}_max_abs_error"]-pair["non_gras"][f"{name}_max_abs_error"]
            sensitivity.append(out)
        sens=staging/"balancing_sensitivity.csv"
        with sens.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(sensitivity[0]));w.writeheader();w.writerows(sensitivity)
        report=staging/"upstream_validation_report.md"; report.write_text("# CYP/PYP validation and paired balancing sensitivity\n\nAll required evidence records were present. Failed gates remain failed in `upstream_validation.csv`; paired differences are descriptive and do not override them. Indicator identity and units come only from hashed indicator manifests.\n",encoding="utf-8")
        outputs=[{"role":"validation-table","path":validation.name,"sha256":sha(validation),"rows":len(records)},
                 {"role":"sensitivity-table","path":sens.name,"sha256":sha(sens),"rows":len(sensitivity)},
                 {"role":"report","path":report.name,"sha256":sha(report),"rows":None}]
        all_pass=all(r[f"{n}_passed"] for r in records for n in CHECKS)
        m=_manifest("complete",command or [],inputs,outputs,["Paired error differences are descriptive; a smaller error is not automatically acceptable."],all_pass)
        m["coverage"]={"expected":len(expected),"accepted":len(records),"missing":[],"excluded":[]}
        m["scope"].update({"indicator":sorted({r['indicator'] for r in records}),"units_input":sorted({r['units'] for r in records}),
                           "year_previous_min":min(expected_years)-1,"year_current_max":max(expected_years)})
        m["validation"]["gates"]={name:all(r[f"{name}_passed"] for r in records) for name in CHECKS}
        (staging/"analysis_manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True),encoding="utf-8")
        if output.exists(): raise FileExistsError(output)
        staging.replace(output); return output
    except Exception: shutil.rmtree(staging,ignore_errors=True); raise


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);print(build(a.inventory,a.output,command=sys.argv));return 0
if __name__=="__main__": raise SystemExit(main())
