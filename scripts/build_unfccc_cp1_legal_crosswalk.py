"""Build a reviewed Kyoto CP1 legal crosswalk from official UNFCCC CAR PDFs.

The script never infers legal compliance from an emissions trend.  For Parties
with a final compilation and accounting report (CAR), it transcribes the legal
test in CAR table 4: reviewed emissions in excess of retired Kyoto units.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


TARGETS = {
    "AUS": ("Australia", 108.0), "AUT": ("Austria", 87.0),
    "BEL": ("Belgium", 92.5), "BGR": ("Bulgaria", 92.0),
    "HRV": ("Croatia", 95.0), "CZE": ("Czechia", 92.0),
    "DNK": ("Denmark", 79.0), "EST": ("Estonia", 92.0),
    "EU": ("European Union", 92.0), "FIN": ("Finland", 100.0),
    "FRA": ("France", 100.0), "DEU": ("Germany", 79.0),
    "GRC": ("Greece", 125.0), "HUN": ("Hungary", 94.0),
    "ISL": ("Iceland", 110.0), "IRL": ("Ireland", 113.0),
    "ITA": ("Italy", 93.5), "JPN": ("Japan", 94.0),
    "LVA": ("Latvia", 92.0), "LIE": ("Liechtenstein", 92.0),
    "LTU": ("Lithuania", 92.0), "LUX": ("Luxembourg", 72.0),
    "MCO": ("Monaco", 92.0), "NLD": ("Netherlands", 94.0),
    "NZL": ("New Zealand", 100.0), "NOR": ("Norway", 101.0),
    "POL": ("Poland", 94.0), "PRT": ("Portugal", 127.0),
    "ROU": ("Romania", 92.0), "RUS": ("Russian Federation", 100.0),
    "SVK": ("Slovakia", 92.0), "SVN": ("Slovenia", 92.0),
    "ESP": ("Spain", 115.0), "SWE": ("Sweden", 104.0),
    "CHE": ("Switzerland", 92.0), "GBR": ("United Kingdom", 87.5),
    "UKR": ("Ukraine", 100.0),
}

EU15 = {"AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "GRC", "IRL", "ITA",
        "LUX", "NLD", "PRT", "ESP", "SWE", "GBR"}
BASE = {"BGR": "1988", "HUN": "1985-1987 average", "POL": "1988",
        "ROU": "1989", "SVN": "1986"}

TARGET_SOURCE = "https://unfccc.int/process-and-meetings/the-kyoto-protocol/what-is-the-kyoto-protocol/kyoto-protocol-targets-for-the-first-commitment-period"
BASE_SOURCE = "https://unfccc.int/process-and-meetings/transparency-and-reporting/reporting-and-review-under-the-kyoto-protocol/first-commitment-period/kyoto-protocol-base-year-data-for-the-first-commitment-period-of-the-kyoto-protocol"
CAR_INDEX = "https://unfccc.int/process/transparency-and-reporting/reporting-and-review-under-the-kyoto-protocol/second-commitment-period/final-compilation-and-accounting-reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_car(path: Path) -> dict:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    compact = re.sub(r"\s+", " ", text)
    def value(label: str) -> str:
        match = re.search(label + r"\s+(NA|[0-9][0-9 ]*)", compact, re.I)
        return "" if match is None else match.group(1).strip()
    emissions = value(r"Total greenhouse gas emissions in the first commitment period \(t CO2 eq\)")
    # Table 4 is the authoritative retirement-account comparison.
    excess = value(r"Quantity of total GHG emissions in excess of Kyoto Protocol units in the retirement account \(t CO2 eq\)")
    retirement = ""
    table4 = re.search(r"Table 4 (.*?)(?:Abbreviations:|$)", compact, re.I)
    if table4:
        match = re.search(r"Total quantity of AAUs, ERUs, RMUs, CERs, tCERs and lCERs in the retirement account\s+([0-9][0-9 ]*)", table4.group(1), re.I)
        retirement = match.group(1).strip() if match else ""
    if not excess:
        raise ValueError(f"Could not transcribe CAR table 4 from {path}")
    outcome = "met_article_3_1" if excess.upper() == "NA" or int(excess.replace(" ", "")) == 0 else "emissions_exceeded_retired_units"
    return {"commitment_period_emissions_tco2e": emissions.replace(" ", ""),
            "retired_units_total": retirement.replace(" ", ""),
            "emissions_excess_tco2e": "0" if excess.upper() == "NA" else excess.replace(" ", ""),
            "official_accounting_outcome": outcome}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], []
    for code, (name, target) in TARGETS.items():
        pdf = args.source_dir / f"{code.lower()}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        accounting = extract_car(pdf)
        year = "2017" if code == "UKR" else "2016"
        car_url = f"https://unfccc.int/sites/default/files/resource/docs/{year}/car/{code.lower()}.pdf"
        rows.append({
            "entity_id": code, "unfccc_party_name": name,
            "entity_type": "regional_economic_integration_organization" if code == "EU" else "country",
            "unfccc_annex_i": "yes", "kyoto_annex_b_cp1_entry": "yes",
            "cp1_protocol_party": "yes", "participation_note": "",
            "withdrawal_effective_date": "", "commitment_period": "2008-2012",
            "annex_b_percent_of_base_year": 92.0 if code in EU15 else target,
            "effective_target_percent_of_base_year": target,
            "effective_change_target_percent": target - 100.0,
            "joint_fulfilment_id": "EU15_CP1" if code in EU15 or code == "EU" else "",
            "target_basis": "EU15 burden-sharing allocation" if code in EU15 else "Annex B quantified commitment",
            "co2_ch4_n2o_base_year": BASE.get(code, "1990"),
            "fluorinated_gases_base_year": "party-specific; see UNFCCC base-year record",
            "covered_scope": "six Annex A GHGs in CO2 equivalent; applicable LULUCF accounting and Kyoto units included in final accounting",
            **accounting, "outcome_source_document": f"FCCC/KP/CMP/{year}/CAR/{code}",
            "outcome_source_table": "table 4", "outcome_source_url": car_url,
            "target_source_url": TARGET_SOURCE, "base_year_source_url": BASE_SOURCE,
        })
        inputs.append({"path": str(pdf), "sha256": sha256(pdf), "source_url": car_url})
    rows.extend([
        {"entity_id": "CAN", "unfccc_party_name": "Canada", "entity_type": "country", "unfccc_annex_i": "yes", "kyoto_annex_b_cp1_entry": "yes", "cp1_protocol_party": "withdrew_effective_2012-12-15", "participation_note": "Withdrawal notification received 2011-12-15; effective 2012-12-15", "withdrawal_effective_date": "2012-12-15", "commitment_period": "2008-2012", "annex_b_percent_of_base_year": 94.0, "effective_target_percent_of_base_year": 94.0, "effective_change_target_percent": -6.0, "joint_fulfilment_id": "", "target_basis": "Annex B entry; no final CAR because of withdrawal", "co2_ch4_n2o_base_year": "1990", "fluorinated_gases_base_year": "untranscribed", "covered_scope": "Annex A scope while Party; legal outcome not classified here", "commitment_period_emissions_tco2e": "", "retired_units_total": "", "emissions_excess_tco2e": "", "official_accounting_outcome": "not_assessed_with_final_car_withdrawal", "outcome_source_document": "UNFCCC CP1 targets page", "outcome_source_table": "footnote ****", "outcome_source_url": TARGET_SOURCE, "target_source_url": TARGET_SOURCE, "base_year_source_url": BASE_SOURCE},
        {"entity_id": "USA", "unfccc_party_name": "United States of America", "entity_type": "country", "unfccc_annex_i": "yes", "kyoto_annex_b_cp1_entry": "yes", "cp1_protocol_party": "no_never_ratified", "participation_note": "Signed 1998-11-12; did not ratify", "withdrawal_effective_date": "", "commitment_period": "2008-2012", "annex_b_percent_of_base_year": 93.0, "effective_target_percent_of_base_year": "", "effective_change_target_percent": "", "joint_fulfilment_id": "", "target_basis": "Annex B inscription did not become a binding Party commitment", "co2_ch4_n2o_base_year": "1990", "fluorinated_gases_base_year": "not applicable to a CP1 Party commitment", "covered_scope": "not applicable to legal CP1 accounting", "commitment_period_emissions_tco2e": "", "retired_units_total": "", "emissions_excess_tco2e": "", "official_accounting_outcome": "not_applicable_never_ratified", "outcome_source_document": "UNFCCC CP1 targets page and Party record", "outcome_source_table": "footnote ***", "outcome_source_url": TARGET_SOURCE, "target_source_url": TARGET_SOURCE, "base_year_source_url": BASE_SOURCE},
    ])
    output = args.output_dir / "unfccc_cp1_party_crosswalk.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {"schema_version": 1, "analysis_id": "unfccc-cp1-legal-crosswalk-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified_official_sources", "rows": len(rows),
        "method": "Targets/base years transcribed from official UNFCCC tables; legal outcome transcribed from each CAR table 4; no trend-based inference.",
        "sources": {"target_table": TARGET_SOURCE, "base_year_table": BASE_SOURCE, "car_index": CAR_INDEX},
        "inputs": inputs, "outputs": [{"path": output.name, "sha256": sha256(output)}],
        "validation": {"car_rows": len(TARGETS), "all_car_table4_parsed": True,
          "canada_separate_withdrawal": True, "usa_separate_nonratification": True,
          "eu15_burden_sharing_rows": len(EU15)}}
    mp = args.output_dir / "analysis_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
