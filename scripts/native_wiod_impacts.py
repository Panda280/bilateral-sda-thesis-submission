#!/usr/bin/env python3
"""Acquire, parse, aggregate, and validate native WIOD CO2 accounts.

Raw files stay outside Git.  This module deliberately separates producing-
industry emissions from household direct emissions and converts Gg (kt) CO2
to kg CO2 only after validating the declared source unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd

GG_TO_KG = 1_000_000.0
WIOD2013_URL = "https://dataverse.nl/api/access/datafile/199110"


@dataclass(frozen=True)
class NativeImpact:
    values_kg_co2: pd.Series
    household_direct_kg_co2: pd.Series
    database: str
    release: str
    year: int
    indicator: str
    boundary: str
    native_grid: str
    raw_unit: str
    conversion_to_kg: float
    source_path: str
    source_sha256: str

    def metadata(self) -> dict:
        return {
            "database": self.database,
            "release": self.release,
            "year": self.year,
            "indicator": self.indicator,
            "boundary": self.boundary,
            "native_grid": self.native_grid,
            "raw_unit": self.raw_unit,
            "conversion_to_kg": self.conversion_to_kg,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "industry_total_kg_co2": float(self.values_kg_co2.sum()),
            "household_direct_total_kg_co2": float(self.household_direct_kg_co2.sum()),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    temporary.replace(destination)
    return sha256(destination)


def _numeric_sum(values: Iterable[object]) -> float:
    total = 0.0
    for value in values:
        if value in (None, ""):
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite emissions value: {value!r}")
        total += number
    return total


def load_wiod2013_co2(archive: Path, year: int) -> NativeImpact:
    """Load the May-2012 WIOD2013 CO2 archive (country .xls workbooks).

    Each industry row is summed over fuel columns.  ``FC_HH`` is returned on
    a separate country index and is never inserted into the industry vector.
    ``secTOT`` and ``total`` are validation rows, not observations.
    """
    if not 1995 <= year <= 2009:
        raise ValueError("WIOD2013 environmental accounts cover 1995-2009")
    try:
        import xlrd
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("xlrd>=2 is required to read official WIOD2013 .xls files") from exc

    source_hash = sha256(archive)
    industry: dict[tuple[str, str], float] = {}
    household: dict[str, float] = {}
    with zipfile.ZipFile(archive) as outer:
        member = next((n for n in outer.namelist() if n.endswith("CO2_may12.zip")), None)
        if member is None:
            raise ValueError("Official archive does not contain CO2_may12.zip")
        with zipfile.ZipFile(BytesIO(outer.read(member))) as accounts:
            names = sorted(n for n in accounts.namelist() if n.lower().endswith(".xls"))
            if len(names) != 41:
                raise ValueError(f"Expected 41 WIOD2013 country workbooks, found {len(names)}")
            for name in names:
                country = Path(name).name.split("_")[0]
                book = xlrd.open_workbook(file_contents=accounts.read(name))
                notes = book.sheet_by_name("Notes").cell_value(5, 0)
                if "CO2 emissions in Gg (kt)" not in str(notes):
                    raise ValueError(f"Unexpected unit declaration in {name}: {notes!r}")
                sheet = book.sheet_by_name(str(year))
                rows = {
                    str(sheet.cell_value(row, 1)): _numeric_sum(
                        sheet.cell_value(row, col) for col in range(2, sheet.ncols)
                    )
                    for row in range(1, sheet.nrows)
                }
                # ``secQ`` (extra-territorial organisations) is printed in
                # the EA workbooks but is not one of the 35 WIOT industries.
                # It must be zero; otherwise dropping it would change totals.
                if not math.isclose(rows.get("secQ", 0.0), 0.0, abs_tol=1e-12):
                    raise ValueError(f"{name}/{year}: non-zero secQ cannot align to the 35-sector WIOT")
                sector_rows = {
                    k: v for k, v in rows.items()
                    if k.startswith("sec") and k not in {"secTOT", "secQ"}
                }
                if len(sector_rows) != 35:
                    raise ValueError(f"{name}/{year}: expected 35 industry rows, found {len(sector_rows)}")
                if not math.isclose(sum(sector_rows.values()), rows["secTOT"], rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError(f"{name}/{year}: industry rows do not reconcile to secTOT")
                if not math.isclose(rows["secTOT"] + rows["FC_HH"], rows["total"], rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError(f"{name}/{year}: industry plus household does not reconcile to total")
                industry.update({(country, code.removeprefix("sec")): value * GG_TO_KG for code, value in sector_rows.items()})
                household[country] = rows["FC_HH"] * GG_TO_KG

    values = pd.Series(industry, name="kg CO2", dtype=float)
    values.index = pd.MultiIndex.from_tuples(values.index, names=["country", "sector"])
    household_values = pd.Series(household, name="kg CO2", dtype=float)
    household_values.index.name = "country"
    return NativeImpact(
        values.sort_index(), household_values.sort_index(), "WIOD", "2013/May-2012 EA",
        year, "CO2", "territorial producing-industry CO2; household direct separate",
        "41 countries x 35 industries", "Gg (kt) CO2", GG_TO_KG,
        str(archive.resolve()), source_hash,
    )


def load_wiod2016_jrc_canonical(path: Path, year: int) -> NativeImpact:
    """Load a lossless canonical staging CSV made from the JRC WIOD accounts.

    Required columns are ``year,country,sector,role,CO2en_Gg,CO2nen_Gg``.
    ``role`` must be ``industry`` or ``household_direct``.  Requiring both
    native CO2 components makes the boundary and summation auditable instead
    of relying on an undocumented pre-summed column.
    """
    required = {"year", "country", "sector", "role", "CO2en_Gg", "CO2nen_Gg"}
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"JRC staging file lacks required columns: {sorted(missing)}")
    frame = frame.loc[frame["year"].astype(int) == year].copy()
    if frame.empty:
        raise ValueError(f"No JRC observations for {year}")
    if set(frame["role"]) - {"industry", "household_direct"}:
        raise ValueError("role must contain only industry or household_direct")
    for column in ("CO2en_Gg", "CO2nen_Gg"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].map(math.isfinite).all() or (frame[column] < 0).any():
            raise ValueError(f"{column} contains negative or non-finite observations")
    frame["CO2_Gg"] = frame["CO2en_Gg"] + frame["CO2nen_Gg"]
    industry = frame.loc[frame.role == "industry"]
    if industry.duplicated(["country", "sector"]).any():
        raise ValueError("Duplicate JRC country-sector industry observations")
    if industry.country.nunique() != 44 or industry.sector.nunique() != 56:
        raise ValueError(
            f"Expected JRC 44x56 grid; found {industry.country.nunique()}x{industry.sector.nunique()}"
        )
    values = industry.set_index(["country", "sector"])["CO2_Gg"] * GG_TO_KG
    households = (
        frame.loc[frame.role == "household_direct"].groupby("country")["CO2_Gg"].sum()
        * GG_TO_KG
    )
    return NativeImpact(
        values.rename("kg CO2").sort_index(), households.rename("kg CO2").sort_index(),
        "JRC WIOD-compatible EA", "Corsatea et al. 2019", year, "CO2en + CO2nen",
        "territorial producing-industry CO2; household direct separate",
        "44 regions x 56 industries", "Gg CO2", GG_TO_KG,
        str(path.resolve()), sha256(path),
    )


def load_wiod2016_jrc_xlsx(path: Path, year: int) -> NativeImpact:
    """Load the official July-2019 JRC ``CO2 emissions.xlsx`` workbook.

    The workbook publishes total CO2 by country, 56 NACE Rev.2 industries,
    and year in kt, plus a separate ``FC_HH`` row. The accompanying method
    defines total CO2 as energy-related plus non-energy-related CO2, but the
    distributed CO2 workbook does not expose those components separately.
    """
    if not 2000 <= year <= 2016:
        raise ValueError("JRC WIOD-compatible CO2 accounts cover 2000-2016")
    book = pd.ExcelFile(path)
    country_sheets = [name for name in book.sheet_names if name not in {"Notes", "Sector"}]
    if len(country_sheets) != 44:
        raise ValueError(f"Expected 44 JRC country/region sheets, found {len(country_sheets)}")
    notes = pd.read_excel(path, sheet_name="Notes", header=None)
    note_text = " ".join(notes.fillna("").astype(str).to_numpy().ravel())
    if "European Commission, Joint Research Centre, July 2019 release" not in note_text:
        raise ValueError("Unexpected JRC release declaration")
    if "CO2 Emission in kt" not in note_text:
        raise ValueError("Unexpected JRC CO2 unit declaration")
    sector_sheet = pd.read_excel(path, sheet_name="Sector", header=None)
    expected_sectors = sector_sheet.iloc[2:58, 1].dropna().astype(str).tolist()
    if len(expected_sectors) != 56 or len(set(expected_sectors)) != 56:
        raise ValueError("JRC Sector sheet does not define 56 unique industries")

    industry: dict[tuple[str, str], float] = {}
    households: dict[str, float] = {}
    for sheet_name in country_sheets:
        frame = pd.read_excel(path, sheet_name=sheet_name, header=0, index_col=0)
        frame.index = frame.index.astype(str)
        year_column = next((column for column in frame.columns if str(column) == str(year)), None)
        if year_column is None:
            raise ValueError(f"{sheet_name}: missing year {year}")
        series = pd.to_numeric(frame[year_column], errors="raise").dropna()
        observed = series.reindex(expected_sectors)
        if observed.isna().any():
            raise ValueError(f"{sheet_name}/{year}: missing industry rows")
        if (observed < 0).any() or not observed.map(math.isfinite).all():
            raise ValueError(f"{sheet_name}/{year}: invalid industry CO2 value")
        country = "ROW" if sheet_name == "RoW" else sheet_name
        industry.update(
            {(country, sector): value * GG_TO_KG for sector, value in observed.items()}
        )
        if "FC_HH" not in series.index:
            raise ValueError(f"{sheet_name}/{year}: missing FC_HH row")
        households[country] = float(series.loc["FC_HH"]) * GG_TO_KG

    values = pd.Series(industry, name="kg CO2", dtype=float)
    values.index = pd.MultiIndex.from_tuples(values.index, names=["country", "sector"])
    household_values = pd.Series(households, name="kg CO2", dtype=float)
    household_values.index.name = "country"
    return NativeImpact(
        values.sort_index(), household_values.sort_index(),
        "JRC WIOD-compatible EA", "July 2019 / Corsatea et al. 2019", year,
        "published total CO2 (methodological identity CO2en + CO2nen)",
        "territorial resident producing-industry CO2; household direct separate",
        "44 regions x 56 industries", "kt CO2", GG_TO_KG,
        str(path.resolve()), sha256(path),
    )


def aggregate_2016_sectors(impact: NativeImpact, mapping_path: Path) -> NativeImpact:
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    required = {"native_sector", "target_sector", "weight", "review_status"}
    if required - set(mapping.columns):
        raise ValueError(f"Mapping lacks columns: {sorted(required - set(mapping.columns))}")
    if (mapping.review_status != "reviewed").any():
        unresolved = mapping.loc[mapping.review_status != "reviewed", "native_sector"].tolist()
        raise ValueError(f"56-to-50 mapping is not fully reviewed: {unresolved}")
    mapping["weight"] = pd.to_numeric(mapping["weight"], errors="raise")
    if not mapping.groupby("native_sector").weight.sum().map(lambda x: math.isclose(x, 1.0)).all():
        raise ValueError("Mapping weights must sum to one for every native sector")
    source = impact.values_kg_co2.rename("value").reset_index()
    missing = sorted(set(source.sector) - set(mapping.native_sector))
    if missing:
        raise ValueError(f"Mapping misses native sectors: {missing}")
    joined = source.merge(mapping[["native_sector", "target_sector", "weight"]], left_on="sector", right_on="native_sector", validate="many_to_many")
    joined["mapped"] = joined.value * joined.weight
    mapped = joined.groupby(["country", "target_sector"]).mapped.sum().rename("kg CO2")
    if not math.isclose(float(mapped.sum()), float(impact.values_kg_co2.sum()), rel_tol=1e-12):
        raise ValueError("Sector aggregation does not conserve industry CO2")
    return NativeImpact(
        mapped, impact.household_direct_kg_co2, impact.database, impact.release,
        impact.year, impact.indicator, impact.boundary,
        "44 regions x reviewed thesis WIOD2016 sectors", impact.raw_unit,
        impact.conversion_to_kg, impact.source_path, impact.source_sha256,
    )


def write_impact(impact: NativeImpact, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    impact.values_kg_co2.rename("kg CO2").to_csv(output_dir / "industry_co2_kg.csv")
    impact.household_direct_kg_co2.rename("kg CO2").to_csv(output_dir / "household_direct_co2_kg.csv")
    (output_dir / "manifest.json").write_text(json.dumps(impact.metadata(), indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("download-wiod2013")
    acquire.add_argument("destination", type=Path)
    load13 = sub.add_parser("load-wiod2013")
    load13.add_argument("archive", type=Path)
    load13.add_argument("year", type=int)
    load13.add_argument("output", type=Path)
    load16 = sub.add_parser("load-jrc2016")
    load16.add_argument("staging_csv", type=Path)
    load16.add_argument("year", type=int)
    load16.add_argument("output", type=Path)
    load16.add_argument("--sector-map", type=Path)
    args = parser.parse_args()
    if args.command == "download-wiod2013":
        print(json.dumps({"url": WIOD2013_URL, "path": str(args.destination), "sha256": download(WIOD2013_URL, args.destination)}, indent=2))
    elif args.command == "load-wiod2013":
        write_impact(load_wiod2013_co2(args.archive, args.year), args.output)
    else:
        impact = (
            load_wiod2016_jrc_xlsx(args.staging_csv, args.year)
            if args.staging_csv.suffix.lower() == ".xlsx"
            else load_wiod2016_jrc_canonical(args.staging_csv, args.year)
        )
        if args.sector_map:
            impact = aggregate_2016_sectors(impact, args.sector_map)
        write_impact(impact, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
