#!/usr/bin/env python3
"""Build descriptive country and analytical-group SDA scorecards.

The analytical groups are static thesis classifications.  They are not legal
Kyoto compliance categories.  D_C and D_YF remain separate effects and all
roles remain explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EFFECT_SPECS = {
    "D_C": ("source_country/D_C.source_country_cube.parquet", ("Emitter_Country", "Driver_Country", "Source_Country")),
    "D_YF": ("yf_source_country/D_YF.source_country_cube.parquet", ("Emitter_Country", "Consumer_Country", "Source_Country")),
}
ROLE_NAMES = {
    "Emitter_Country": "emitter",
    "Driver_Country": "driver",
    "Consumer_Country": "consumer",
    "Source_Country": "source",
}
GROUPS = {"Red", "Non-Red", "Non-Participant"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mappings(groups_path: Path, countries_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pd.read_csv(groups_path, dtype=str).fillna("")
    if {"country_code", "analysis_group"} <= set(groups):
        groups = groups.rename(columns={"country_code": "canonical_country"})
    required = {"canonical_country", "analysis_group"}
    if not required <= set(groups):
        raise ValueError(f"group registry lacks {sorted(required - set(groups))}")
    if groups.canonical_country.duplicated().any() or not set(groups.analysis_group) <= GROUPS:
        raise ValueError("group registry has duplicate codes or invalid groups")

    countries = pd.read_csv(countries_path, dtype=str).fillna("")
    if {"resolution", "observed_country", "canonical_country"} <= set(countries):
        countries = countries[["resolution", "observed_country", "canonical_country"]]
    elif {"WIOT2013", "WIOT2016", "EXIO"} <= set(countries):
        rows = []
        for resolution, column in (("2013", "WIOT2013"), ("2016", "WIOT2016"), ("R3", "EXIO")):
            rows.extend({"resolution": resolution, "observed_country": row[column],
                         "canonical_country": row["WIOT2016"]}
                        for _, row in countries.iterrows() if row[column])
        countries = pd.DataFrame(rows).drop_duplicates()
    else:
        raise ValueError("unsupported country concordance schema")
    target_counts = countries.groupby(["resolution", "observed_country"]).canonical_country.nunique()
    conflicting = target_counts[target_counts > 1]
    if len(conflicting):
        raise ValueError(f"ambiguous country concordance source keys: {list(conflicting.index)}")
    # The legacy table repeats WIOD ROW once for each detailed EXIO RoW region.
    # Identical source-to-target repetitions are redundant, not ambiguous.
    countries = countries.drop_duplicates(["resolution", "observed_country", "canonical_country"])
    unknown = set(countries.canonical_country) - set(groups.canonical_country)
    if unknown:
        raise ValueError(f"canonical countries absent from group registry: {sorted(unknown)}")
    return groups, countries


def summarize(frame: pd.DataFrame, entity: str) -> pd.DataFrame:
    keys = ["effect", "resolution", "role", entity, "units"]
    rows = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        ordered = group.sort_values(["year_previous", "year_current"])
        values = ordered.net.to_numpy(float)
        nonzero = values[values != 0]
        reversals = int(sum(a * b < 0 for a, b in zip(values, values[1:])))
        rows.append(dict(zip(keys, key)) | {
            "period_start": int(ordered.year_previous.min()),
            "period_end": int(ordered.year_current.max()),
            "observation_count": len(values),
            "cumulative_net": float(values.sum()),
            "cumulative_gross": float(ordered.gross.sum()),
            "mean_annual_net": float(values.mean()),
            "population_volatility": float(values.std(ddof=0)),
            "positive_count": int((values > 0).sum()),
            "negative_count": int((values < 0).sum()),
            "zero_count": int((values == 0).sum()),
            "sign_persistence": float(max((values > 0).sum(), (values < 0).sum()) / len(nonzero)) if len(nonzero) else 0.0,
            "sign_reversals": reversals,
            "cancellation_ratio": float(1 - abs(values.sum()) / ordered.gross.sum()) if ordered.gross.sum() else 0.0,
        })
    return pd.DataFrame(rows)


def mechanism_profiles(country_summary: pd.DataFrame) -> pd.DataFrame:
    source = country_summary[country_summary.role.eq("source")].copy()
    pivot = source.pivot_table(index=["resolution", "canonical_country", "units"], columns="effect",
                               values=["cumulative_net", "cumulative_gross"], aggfunc="first")
    rows = []
    for (resolution, country, units), row in pivot.iterrows():
        if not all((metric, effect) in row.index for metric in ("cumulative_net", "cumulative_gross") for effect in EFFECT_SPECS):
            continue
        dc, dyf = float(row["cumulative_net", "D_C"]), float(row["cumulative_net", "D_YF"])
        dcg, dyfg = float(row["cumulative_gross", "D_C"]), float(row["cumulative_gross", "D_YF"])
        if dc >= 0 and dyf >= 0:
            classification = "no_source_subeffect_reduction"
        elif dc < 0 and dyf >= 0:
            classification = "intermediate_sourcing_reduction_only"
        elif dyf < 0 and dc >= 0:
            classification = "final_demand_sourcing_reduction_only"
        else:
            classification = "both_source_subeffects_reducing"
        dominant = "D_C" if abs(dc) > abs(dyf) else "D_YF" if abs(dyf) > abs(dc) else "tie"
        rows.append({"resolution": resolution, "canonical_country": country, "units": units,
                     "D_C_cumulative_net": dc, "D_YF_cumulative_net": dyf,
                     "D_C_cumulative_gross": dcg, "D_YF_cumulative_gross": dyfg,
                     "descriptive_mechanism_class": classification,
                     "dominant_source_subeffect_by_absolute_net": dominant,
                     "interpretation_limit": "descriptive source-subeffect profile; not causal proof or legal compliance"})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path, required=True)
    parser.add_argument("--countries-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    groups, countries = load_mappings(args.groups_csv, args.countries_csv)
    country_map = countries.set_index(["resolution", "observed_country"]).canonical_country.to_dict()
    group_map = groups.set_index("canonical_country").analysis_group.to_dict()
    annual_rows, inventory, audit_rows = [], [], []
    for run_dir in sorted((args.campaign_root / "offshoring").glob("*_*_*")):
        parts = run_dir.name.split("_")
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        previous, current, resolution = parts
        for effect, (relative, roles) in EFFECT_SPECS.items():
            path = run_dir / relative
            if not path.is_file():
                continue
            print(f"processing {run_dir.name} {effect}: {path}", flush=True)
            columns = [*roles, "Value"]
            parquet = pq.ParquetFile(path)
            inventory.append({"run_id": run_dir.name, "effect": effect, "path": str(path), "sha256": sha256(path), "rows": parquet.metadata.num_rows})
            observed_by_role = {raw_role: set() for raw_role in roles}
            parts_by_role = {raw_role: [] for raw_role in roles}
            for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
                data = batch.to_pandas()
                values = pd.to_numeric(data.Value, errors="coerce")
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite Value in {path}")
                for raw_role in roles:
                    observed_by_role[raw_role].update(data[raw_role].astype(str).unique())
                    compact_batch = pd.DataFrame({"observed_country": data[raw_role].astype(str), "Value": values}).groupby("observed_country").Value.agg(
                        net="sum", positive=lambda x: x[x > 0].sum(), negative=lambda x: x[x < 0].sum(), gross=lambda x: x.abs().sum()).reset_index()
                    parts_by_role[raw_role].append(compact_batch)
            for raw_role in roles:
                role = ROLE_NAMES[raw_role]
                observed_codes = observed_by_role[raw_role]
                missing = sorted(code for code in observed_codes if (resolution, code) not in country_map)
                if missing:
                    raise ValueError(f"unmapped {resolution} {raw_role}: {missing}")
                for observed in sorted(observed_codes):
                    canonical = country_map[resolution, observed]
                    audit_rows.append({"resolution": resolution, "role": role, "observed_country": observed,
                                       "canonical_country": canonical, "analysis_group": group_map[canonical], "status": "mapped"})
                compact = pd.concat(parts_by_role[raw_role], ignore_index=True)
                compact["canonical_country"] = compact.observed_country.map(lambda code: country_map[resolution, code])
                compact = compact.groupby("canonical_country", as_index=False).agg(
                    net=("net", "sum"), positive=("positive", "sum"), negative=("negative", "sum"), gross=("gross", "sum"))
                compact.insert(0, "role", role); compact.insert(0, "resolution", resolution); compact.insert(0, "effect", effect)
                compact.insert(0, "year_current", int(current)); compact.insert(0, "year_previous", int(previous))
                compact["units"] = "kg CO2"
                compact["cancellation_ratio"] = np.where(compact.gross > 0, 1 - compact.net.abs() / compact.gross, 0.0)
                annual_rows.append(compact)
            print(f"completed {run_dir.name} {effect}", flush=True)
    # Expected inventory is derived from the transitions actually present under
    # `offshoring/`, not hardcoded to 14: every (resolution, effect) pair must
    # cover the same set of transitions, and that set defines the window.
    _inv = pd.DataFrame(inventory)
    _inv["_transition"] = _inv.run_id.str.rsplit("_", n=1).str[0]
    _inv["_resolution"] = _inv.run_id.str.rsplit("_", n=1).str[1]
    _per = _inv.groupby(["_resolution", "effect"])["_transition"].apply(frozenset)
    if _per.nunique() != 1:
        raise ValueError(f"source-country inventory covers inconsistent transition sets: {_per.to_dict()}")
    _transitions = sorted(next(iter(set(_per))))
    expected = len(_transitions) * 3 * 2
    if len(inventory) != expected:
        raise ValueError(
            f"incomplete source-country inventory: {len(inventory)} of {expected} "
            f"({len(_transitions)} transitions x 3 resolutions x 2 effects)")
    annual = pd.concat(annual_rows, ignore_index=True)
    annual["analysis_group"] = annual.canonical_country.map(group_map)
    group_annual = annual.groupby(["year_previous", "year_current", "effect", "resolution", "role", "analysis_group", "units"], as_index=False).agg(
        net=("net", "sum"), positive=("positive", "sum"), negative=("negative", "sum"), gross=("gross", "sum"))
    group_annual["cancellation_ratio"] = np.where(group_annual.gross > 0, 1 - group_annual.net.abs() / group_annual.gross, 0.0)
    country_summary = summarize(annual, "canonical_country")
    group_summary = summarize(group_annual.rename(columns={"analysis_group": "group"}), "group")
    profiles = mechanism_profiles(country_summary)
    outputs = {"annual_country_scorecards.csv": annual,
               "cumulative_country_scorecards.csv": country_summary,
               "annual_group_scorecards.csv": group_annual,
               "cumulative_group_scorecards.csv": group_summary,
               "country_reduction_mechanisms.csv": profiles,
               "country_mapping_audit.csv": pd.DataFrame(audit_rows).drop_duplicates(),
               "matched_inventory.csv": pd.DataFrame(inventory)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, frame in outputs.items():
        target = args.output_dir / name; frame.to_csv(target, index=False)
        records.append({"path": name, "sha256": sha256(target), "rows": len(frame)})
    manifest = {"schema_version": 1, "analysis_id": "kyoto-country-scorecards-v1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "accepted_provisional",
                "generator": {"script": "scripts/build_kyoto_country_scorecards.py", "command": sys.argv},
                "scope": {"years": f"{_transitions[0].replace('_', '->')} through {_transitions[-1].replace('_', '->')}", "resolutions": ["2013", "2016", "R3"],
                          "effects": list(EFFECT_SPECS), "units": "kg CO2", "price_basis": "CYP_(t-1) -> PYP_t"},
                "inputs": [{"path": str(args.groups_csv), "sha256": sha256(args.groups_csv)},
                           {"path": str(args.countries_csv), "sha256": sha256(args.countries_csv)}],
                "outputs": records, "validation": {"passed": True, "inventory": f"{len(inventory)}/{expected}",
                "finite": bool(np.isfinite(annual[["net", "positive", "negative", "gross", "cancellation_ratio"]]).all().all())},
                "limitations": ["Accepted provisional fossil-origin CO2 because upstream validation remains unresolved.",
                                "Static analytical groups are not legal Kyoto compliance classifications.",
                                "Mechanism labels are descriptive source-subeffect profiles and are not causal proof."]}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
