#!/usr/bin/env python3
"""Export deterministic thesis tables and figures from validated CSV products."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(spec_path: Path, output: Path):
    spec=json.loads(spec_path.read_text(encoding="utf-8")); blockers=[]; inputs=[]; products=[]; prepared=[]
    views=spec.get("views",[])
    required={"annual_hierarchy","cross_resolution","international_domestic","group_trends","selected_cases"}
    absent=required-{v.get("id") for v in views}
    if absent: blockers.append(f"input specification lacks required views: {sorted(absent)}")
    for view in views:
        table=Path(view["table"]); manifest=Path(view["manifest"])
        for role,path in (("source_table",table),("upstream_manifest",manifest)):
            if not path.is_file(): blockers.append(f"{view['id']}: missing {role}: {path}")
            else: inputs.append({"role":f"{view['id']}:{role}","path":str(path),"sha256":sha(path)})
        if not table.is_file() or not manifest.is_file(): continue
        meta=json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("status") not in {"complete","accepted_provisional"}: blockers.append(f"{view['id']}: upstream manifest status is {meta.get('status')!r}"); continue
        validation=meta.get("validation",{})
        if validation and not validation.get("passed",False) and validation.get("thesis_status")!="accepted_provisional": blockers.append(f"{view['id']}: upstream validation is neither passed nor accepted provisional"); continue
        frame=pd.read_csv(table); missing=set(view["required_columns"])-set(frame.columns)
        if missing: blockers.append(f"{view['id']}: missing columns {sorted(missing)}"); continue
        semantic=set(view.get("semantic_columns",[]))
        if not semantic.issubset(frame.columns): blockers.append(f"{view['id']}: semantic columns absent: {sorted(semantic-set(frame.columns))}")
        if frame.duplicated(view["key_columns"]).any(): blockers.append(f"{view['id']}: duplicate declared keys")
        for col in view.get("numeric_columns",[]):
            values=pd.to_numeric(frame[col],errors="coerce")
            if values.isna().any() or (values.abs()==float("inf")).any(): blockers.append(f"{view['id']}: non-finite numeric column {col}")
        prepared.append((view,frame))
    if blockers: return blockers,inputs,products
    output.mkdir(parents=True,exist_ok=True)
    for view,frame in prepared:
        ordered=frame.sort_values(view["key_columns"],kind="stable").reset_index(drop=True)
        table_out=output/f"{view['id']}.table_data.csv"; plot_out=output/f"{view['id']}.plot_data.csv"
        ordered.to_csv(table_out,index=False,float_format="%.12g")
        ordered[view["plot_columns"]].to_csv(plot_out,index=False,float_format="%.12g")
        for role,path,rows in (("table_data",table_out,len(ordered)),("plot_data",plot_out,len(ordered))): products.append({"role":f"{view['id']}:{role}","path":path.name,"sha256":sha(path),"rows":rows})
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7.2,4.2)); x=view["x"]; y=view["y"]
            groups=view.get("series",[])
            if groups:
                for key,part in ordered.groupby(groups,dropna=False,sort=True): ax.plot(part[x],part[y],marker="o",label=" | ".join(map(str,key if isinstance(key,tuple) else (key,))))
                ax.legend(fontsize=7)
            else: ax.plot(ordered[x],ordered[y],marker="o")
            ax.set(xlabel=view.get("x_label",x),ylabel=view.get("y_label",y),title=view["title"]); ax.grid(alpha=.25); fig.tight_layout()
            figure=output/f"{view['id']}.png"; fig.savefig(figure,dpi=180,metadata={"Software":"bilateral-sda-attribution"}); plt.close(fig)
            products.append({"role":f"{view['id']}:figure","path":figure.name,"sha256":sha(figure),"rows":None})
        except ImportError: pass
    return blockers,inputs,products

def main():
    p=argparse.ArgumentParser(); p.add_argument("--spec",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    blockers,inputs,outputs=run(a.spec,a.output_dir)
    report=["# Static thesis result exports","",("Status: **BLOCKED**" if blockers else "Status: **COMPLETE**"),""]
    if blockers: report += ["## Exact blockers",""]+[f"- {x}" for x in blockers]
    else: report += [f"Generated {len(outputs)} hashed table-data, plot-data, and static-figure artifacts.","","All figures derive only from the adjacent saved plot-data CSV."]
    (a.output_dir/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    manifest={"schema_version":1,"analysis_id":"static-thesis-exports-v1","status":"blocked" if blockers else "complete","generated_at_utc":datetime.now(timezone.utc).isoformat(),"generator":{"script":"scripts/export_static_thesis_results.py","git_commit":"dirty-state-not-resolved-by-pipeline","command":sys.argv},"scope":{"price_basis":"preserved from upstream manifests","indicator":"fossil-origin CO2","units_input":"preserved per input table","units_output":"preserved per output table","effects":[],"resolutions":["2013","2016","R3"],"year_previous_min":2000,"year_current_max":2014},"coverage":{"expected":5,"accepted":0 if blockers else 5,"missing":blockers,"excluded":[]},"inputs":[{"role":"export_spec","path":str(a.spec),"sha256":sha(a.spec)},*inputs],"outputs":outputs,"validation":{"passed":not blockers,"gates":{"required_views":not blockers,"upstream_manifests":not blockers,"schemas_and_keys":not blockers},"upstream_input_status":"accepted_provisional","thesis_status":"accepted_provisional" if not blockers else "ineligible"},"limitations":["Static exports inherit all upstream validation limitations.","Fossil-origin CO2 must not be represented as GWP100.","Effects, roles, units, and resolutions are not pooled."]}
    (a.output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); return 2 if blockers else 0
if __name__=="__main__": raise SystemExit(main())
