"""Schema-aware, dependency-light exports for canonical SDA results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..attribution.groups import SOURCE_ORIGIN_GROUPS
from ..hierarchy import CANONICAL_EFFECTS, TOP_LEVEL_EFFECTS
from ..model import EffectResult, SDAResult
from .aggregation import AxisGrouping, aggregate_effect
from .storage import SCHEMA_VERSION

REPORT_SCHEMA_VERSION = 1
SOURCE_COUNTRY_CUBE_COLUMNS = (
    "Emitter_Country",
    "Emitter_Sector",
    "Driver_Country",
    "Source_Country",
    "Value",
)
SOURCE_GROUP_SECTOR_CUBE_COLUMNS = (
    "Emitter_Country",
    "Emitter_Sector",
    "Driver_Country",
    "Source_Group",
    "Source_Sector",
    "Value",
)
YF_SOURCE_COUNTRY_CUBE_COLUMNS = (
    "Emitter_Country", "Emitter_Sector", "Consumer_Country",
    "Source_Country", "Value",
)
YF_SOURCE_GROUP_SECTOR_CUBE_COLUMNS = (
    "Emitter_Country", "Emitter_Sector", "Consumer_Country",
    "Source_Group", "Source_Sector", "Value",
)
DC_INTERACTION_COLUMNS = (
    "Relationship", "Source_Group", "Source_Sector",
    "Destination_Group", "Destination_Sector", "Value",
)
YF_INTERACTION_COLUMNS = (
    "Relationship", "Source_Group", "Destination_Group", "Product_Sector", "Value",
)


@dataclass(frozen=True)
class EffectExportSpec:
    """Public reporting contract for one canonical effect."""

    columns: tuple[str, ...]
    included_in_total: bool
    parent: str | None = None


_EMITTER = ("Emitter Country", "Emitter Sector")
_STRUCTURAL = _EMITTER + ("Driver Country", "Driver Sector")
_DEMAND = _EMITTER + ("Consumer Country", "Demanded Sector")

EFFECT_REGISTRY: Mapping[str, EffectExportSpec] = {
    "D_v": EffectExportSpec(_EMITTER + ("Value",), True),
    "D_C": EffectExportSpec(_STRUCTURAL + ("Value",), True),
    "D_A_x": EffectExportSpec(_STRUCTURAL + ("Value",), True),
    "D_T_coef": EffectExportSpec(_STRUCTURAL + ("Value",), False, "D_A_x"),
    "D_w_coef": EffectExportSpec(_STRUCTURAL + ("Value",), False, "D_A_x"),
    "D_YF": EffectExportSpec(_DEMAND + ("Value",), True),
    "D_YB": EffectExportSpec(_DEMAND + ("Value",), True),
    "D_y": EffectExportSpec(_EMITTER + ("Consumer Country", "Value"), True),
}

# Group-sector resolution: the same effects, with each country-sector axis
# re-resolved to (classification group, sector) -- see
# docs/decisions/0006-group-sector-report-resolution.md for what "group"
# means here and why it is safe to reuse EFFECT_REGISTRY's row/column shape
# conventions unchanged.
_EMITTER_GROUP = ("Emitter Group", "Emitter Sector")
_STRUCTURAL_GROUP = _EMITTER_GROUP + ("Driver Group", "Driver Sector")
_DEMAND_GROUP = _EMITTER_GROUP + ("Consumer Group", "Demanded Sector")

GROUP_EFFECT_REGISTRY: Mapping[str, EffectExportSpec] = {
    "D_v": EffectExportSpec(_EMITTER_GROUP + ("Value",), True),
    "D_C": EffectExportSpec(_STRUCTURAL_GROUP + ("Value",), True),
    "D_A_x": EffectExportSpec(_STRUCTURAL_GROUP + ("Value",), True),
    "D_T_coef": EffectExportSpec(_STRUCTURAL_GROUP + ("Value",), False, "D_A_x"),
    "D_w_coef": EffectExportSpec(_STRUCTURAL_GROUP + ("Value",), False, "D_A_x"),
    "D_YF": EffectExportSpec(_DEMAND_GROUP + ("Value",), True),
    "D_YB": EffectExportSpec(_DEMAND_GROUP + ("Value",), True),
    "D_y": EffectExportSpec(_EMITTER_GROUP + ("Consumer Group", "Value"), True),
}


def _pair(label: Any, role: str) -> tuple[Any, Any]:
    """Accept a real ``(country, sector)`` tuple (synthetic fixtures, tests),
    or the flat ``"country:sector"`` string ``sda.integration.bundle_export
    ._label`` uses for real EXIODeflator-sourced bundles -- see
    ``docs/decisions/0005-bundle-export-boundary.md``'s "Label encoding"
    note: JSON has no tuple type, so the real export path encodes
    ``emitter_labels``/``structural_driver_labels`` as flat strings rather
    than ``[country, sector]`` lists. Without this, every CSV/Parquet
    export of a real published run fails here first (row 0's emitter
    label), before any other effect-specific logic runs -- found exactly
    that way, exporting the first real 2003-2004 run's effects."""
    if isinstance(label, tuple) and len(label) == 2:
        return label
    if isinstance(label, str):
        parts = label.split(":")
        if len(parts) == 2 and all(parts):
            return parts[0], parts[1]
    raise ValueError(f"{role} labels must be a (country, sector) pair, or a 'country:sector' string")


def _rows(effect: EffectResult):
    values = np.asarray(effect.baseline)
    for row_index, emitter_label in enumerate(effect.dimensions.row.labels):
        emitter = _pair(emitter_label, "emitter")
        if effect.name == "D_v":
            yield (*emitter, float(values[row_index]))
            continue
        if effect.dimensions.column is None:
            raise ValueError(f"{effect.name} requires a column dimension")
        for column_index, column_label in enumerate(effect.dimensions.column.labels):
            if effect.name == "D_y":
                yield (*emitter, column_label, float(values[row_index, column_index]))
            else:
                column = _pair(column_label, "driver")
                yield (*emitter, *column, float(values[row_index, column_index]))


def export_effect_csv(
    effect: EffectResult,
    destination: str | Path,
    *,
    spec: EffectExportSpec | None = None,
) -> Path:
    """Write one effect with headers determined by its canonical semantics.

    ``spec`` defaults to the country-sector :data:`EFFECT_REGISTRY` entry;
    pass :data:`GROUP_EFFECT_REGISTRY`'s entry (see
    :func:`export_group_effect_csv`) to label the same row shape as a
    group-sector resolution instead.
    """
    if spec is None:
        spec = EFFECT_REGISTRY[effect.name]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(spec.columns)
        writer.writerows(_rows(effect))
    return path


def _flatten_columns(effect: EffectResult) -> tuple[np.ndarray, ...]:
    """Vectorised equivalent of :func:`_rows`, one array per output column.

    Same row order and column meaning as :func:`_rows` (and therefore the
    same schema :data:`EFFECT_REGISTRY`/:data:`GROUP_EFFECT_REGISTRY`
    describe) -- only the construction differs. ``_rows`` builds one Python
    tuple per cell via a nested loop, which is fine for CSV's row-at-a-time
    ``csv.writer`` but is the dominant cost for a full driver-resolution
    effect (n-by-n, e.g. 1435x1435 = ~2.06e6 rows): building the same rows
    via ``numpy.repeat``/``numpy.tile`` over the label arrays plus one
    ``.ravel()`` of the value matrix does the identical cross-product in
    vectorised (C-level) numpy calls instead of ~2e6 individual Python tuple
    allocations. Used by :func:`export_effect_parquet`.
    """
    values = np.asarray(effect.baseline, dtype=np.float64)
    row_labels = effect.dimensions.row.labels
    emitter_country = np.array([_pair(label, "emitter")[0] for label in row_labels], dtype=object)
    emitter_sector = np.array([_pair(label, "emitter")[1] for label in row_labels], dtype=object)

    if effect.name == "D_v":
        return emitter_country, emitter_sector, values

    if effect.dimensions.column is None:
        raise ValueError(f"{effect.name} requires a column dimension")
    column_labels = effect.dimensions.column.labels
    n, m = values.shape
    row_country = np.repeat(emitter_country, m)
    row_sector = np.repeat(emitter_sector, m)
    flat_values = values.ravel()

    if effect.name == "D_y":
        consumer = np.tile(np.array(column_labels, dtype=object), n)
        return row_country, row_sector, consumer, flat_values

    driver_country = np.array([_pair(label, "driver")[0] for label in column_labels], dtype=object)
    driver_sector = np.array([_pair(label, "driver")[1] for label in column_labels], dtype=object)
    col_country = np.tile(driver_country, n)
    col_sector = np.tile(driver_sector, n)
    return row_country, row_sector, col_country, col_sector, flat_values


def export_effect_parquet(
    effect: EffectResult,
    destination: str | Path,
    *,
    spec: EffectExportSpec | None = None,
) -> Path:
    """Write one effect as Parquet with :func:`export_effect_csv`'s schema.

    Same long-format ``(emitter[, driver/consumer][, sector], Value)``
    columns as :func:`export_effect_csv` -- only the on-disk encoding
    changes, so anything already reading the CSV schema (column names, one
    row per cell) can switch formats without adjusting how it reads the
    result. Requires the optional ``parquet`` extra (``pandas`` +
    ``pyarrow``; see ``pyproject.toml``) -- imported lazily so the base
    package stays dependency-light for callers who never use this path.
    Built from :func:`_flatten_columns`'s vectorised numpy arrays, not a
    Python-level row loop.
    """
    if spec is None:
        spec = EFFECT_REGISTRY[effect.name]
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Parquet export requires the 'parquet' optional extra (pandas + pyarrow); "
            "install with pip install 'bilateral-sda-attribution[parquet]'"
        ) from exc
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(dict(zip(spec.columns, _flatten_columns(effect))))
    frame.to_parquet(path, index=False)
    return path


def _export_effect(
    effect: EffectResult,
    destination: str | Path,
    *,
    file_format: str,
    spec: EffectExportSpec | None = None,
) -> Path:
    """Dispatch to :func:`export_effect_csv` or :func:`export_effect_parquet`."""
    if file_format == "csv":
        return export_effect_csv(effect, destination, spec=spec)
    if file_format == "parquet":
        return export_effect_parquet(effect, destination, spec=spec)
    raise ValueError(f"file_format must be 'csv' or 'parquet', got {file_format!r}")


def export_effect_report(
    effect: EffectResult,
    destination: str | Path,
    *,
    run_id: str,
    units: str,
    run_metadata: Mapping[str, Any] | None = None,
    file_format: str = "csv",
) -> Path:
    """Export one contract-valid effect with a provenance sidecar.

    ``file_format`` is ``"csv"`` (default, stdlib-only) or ``"parquet"``
    (requires the optional ``parquet`` extra) -- see
    :func:`export_effect_parquet` for why the latter is worth it for a
    full driver-resolution effect.
    """
    if not run_id or not units:
        raise ValueError("run_id and units must be non-empty")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    filename = f"{effect.name}.{file_format}"
    _export_effect(effect, root / filename, file_format=file_format)
    spec = EFFECT_REGISTRY[effect.name]
    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "storage_schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "units": units,
        "run_metadata": dict(run_metadata or {}),
        "files": {effect.name: filename},
        "effect_totals": {effect.name: float(np.asarray(effect.baseline).sum())},
        "headline_effects": [effect.name] if spec.included_in_total else [],
        "excluded_nested_effects": [effect.name] if spec.parent else [],
    }
    (root / "report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def _source_country_cube_chunks(cube: Any, chunk_rows: int):
    """Yield bounded column arrays in emitter/driver/source C-order."""

    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    values = np.asarray(cube.baseline, dtype=np.float64)
    emitters = tuple(_pair(label, "emitter") for label in cube.emitter.labels)
    drivers = np.asarray(cube.driver_country.labels, dtype=object)
    sources = np.asarray(cube.source_country.labels, dtype=object)
    driver_count, source_count = len(drivers), len(sources)
    total = values.size
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        flat = np.arange(start, stop, dtype=np.int64)
        emitter_index = flat // (driver_count * source_count)
        remainder = flat % (driver_count * source_count)
        driver_index = remainder // source_count
        source_index = remainder % source_count
        yield (
            np.asarray([emitters[index][0] for index in emitter_index], dtype=object),
            np.asarray([emitters[index][1] for index in emitter_index], dtype=object),
            drivers[driver_index],
            sources[source_index],
            values.ravel()[start:stop],
        )


def export_source_country_cube(
    cube: Any,
    destination: str | Path,
    *,
    run_id: str,
    units: str,
    run_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    file_format: str = "csv",
    chunk_rows: int = 100_000,
) -> Path:
    """Opt-in long-format export for a source-country offshoring cube.

    CSV is streamed in bounded chunks with the standard library. Parquet is
    written as successive row groups and requires the optional ``parquet``
    extra. No full long-format table or Python row list is materialized.
    """

    if not run_id or not units:
        raise ValueError("run_id and units must be non-empty")
    if file_format not in {"csv", "parquet"}:
        raise ValueError(
            f"file_format must be 'csv' or 'parquet', got {file_format!r}"
        )
    # Validate before creating report directories or files. The chunk helper
    # is a generator, so validation inside it does not run until iteration.
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    chunks = _source_country_cube_chunks(cube, chunk_rows)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    filename = f"D_C.source_country_cube.{file_format}"
    path = root / filename

    if file_format == "csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(SOURCE_COUNTRY_CUBE_COLUMNS)
            for columns in chunks:
                writer.writerows(zip(*columns))
    else:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Parquet export requires the 'parquet' optional extra (pandas + pyarrow); "
                "install with pip install 'bilateral-sda-attribution[parquet]'"
            ) from exc
        writer = None
        try:
            for columns in chunks:
                table = pa.table(dict(zip(SOURCE_COUNTRY_CUBE_COLUMNS, columns)))
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    report = cube.reconciliation
    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "product": "source_country_offshoring_cube",
        "effect": "D_C",
        "run_id": run_id,
        "units": units,
        "file_format": file_format,
        "file": filename,
        "columns": list(SOURCE_COUNTRY_CUBE_COLUMNS),
        "shape": list(np.asarray(cube.baseline).shape),
        "axes": {
            "emitter": list(cube.emitter.labels),
            "driver_country": list(cube.driver_country.labels),
            "source_country": list(cube.source_country.labels),
        },
        "total": float(np.asarray(cube.baseline).sum()),
        "cube_metadata": dict(cube.metadata),
        "run_metadata": dict(run_metadata or {}),
        "provenance": dict(provenance or {}),
        "reconciliation": {
            "passed": report.passed,
            "max_abs_error": report.max_abs_error,
            "failing_index": report.failing_index,
            "rtol": report.rtol,
            "atol": report.atol,
        },
    }
    (root / "source_country_cube_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def _source_group_sector_cube_chunks(cube: Any, chunk_rows: int):
    """Yield bounded long-format columns from the four-dimensional cube."""

    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    values = np.asarray(cube.baseline, dtype=np.float64)
    emitters = tuple(_pair(label, "emitter") for label in cube.emitter.labels)
    drivers = np.asarray(cube.driver_country.labels, dtype=object)
    groups = np.asarray(cube.source_group.labels, dtype=object)
    sectors = np.asarray(cube.source_sector.labels, dtype=object)
    driver_count, group_count, sector_count = len(drivers), len(groups), len(sectors)
    stride = driver_count * group_count * sector_count
    for start in range(0, values.size, chunk_rows):
        stop = min(start + chunk_rows, values.size)
        flat = np.arange(start, stop, dtype=np.int64)
        emitter_index = flat // stride
        remainder = flat % stride
        driver_index = remainder // (group_count * sector_count)
        remainder %= group_count * sector_count
        group_index = remainder // sector_count
        sector_index = remainder % sector_count
        yield (
            np.asarray([emitters[index][0] for index in emitter_index], dtype=object),
            np.asarray([emitters[index][1] for index in emitter_index], dtype=object),
            drivers[driver_index], groups[group_index], sectors[sector_index],
            values.ravel()[start:stop],
        )


def export_source_group_sector_cube(
    cube: Any,
    destination: str | Path,
    *,
    run_id: str,
    units: str,
    run_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    file_format: str = "parquet",
    chunk_rows: int = 100_000,
) -> Path:
    """Stream the opt-in source-group×sector offshoring product to disk."""

    if not run_id or not units:
        raise ValueError("run_id and units must be non-empty")
    if file_format not in {"csv", "parquet"}:
        raise ValueError(f"file_format must be 'csv' or 'parquet', got {file_format!r}")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    chunks = _source_group_sector_cube_chunks(cube, chunk_rows)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    filename = f"D_C.source_group_sector_cube.{file_format}"
    path = root / filename
    if file_format == "csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(SOURCE_GROUP_SECTOR_CUBE_COLUMNS)
            for columns in chunks:
                writer.writerows(zip(*columns))
    else:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Parquet export requires the 'parquet' optional extra") from exc
        writer = None
        try:
            for columns in chunks:
                table = pa.table(dict(zip(SOURCE_GROUP_SECTOR_CUBE_COLUMNS, columns)))
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "product": "source_group_sector_offshoring_cube",
        "effect": "D_C", "run_id": run_id, "units": units,
        "file_format": file_format, "file": filename,
        "columns": list(SOURCE_GROUP_SECTOR_CUBE_COLUMNS),
        "shape": list(np.asarray(cube.baseline).shape),
        "axes": {
            "emitter": list(cube.emitter.labels),
            "driver_country": list(cube.driver_country.labels),
            "source_group": list(cube.source_group.labels),
            "source_sector": list(cube.source_sector.labels),
        },
        "total": float(np.asarray(cube.baseline).sum()),
        "cube_metadata": dict(cube.metadata),
        "run_metadata": dict(run_metadata or {}),
        "provenance": dict(provenance or {}),
        "reconciliation": {
            "parent": vars(cube.reconciliation),
            "by_source_group": {
                group: vars(report) for group, report in cube.group_reconciliation.items()
            },
        },
    }
    (root / "source_group_sector_cube_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def _yf_cube_chunks(cube: Any, chunk_rows: int, *, grouped: bool):
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    values = np.asarray(cube.baseline, dtype=np.float64)
    emitters = tuple(_pair(label, "emitter") for label in cube.emitter.labels)
    emitter_countries = np.asarray([country for country, _ in emitters], dtype=object)
    emitter_sectors = np.asarray([sector for _, sector in emitters], dtype=object)
    consumers = np.asarray(cube.consumer_country.labels, dtype=object)
    tail_axes = ((np.asarray(cube.source_group.labels, dtype=object),
                  np.asarray(cube.source_sector.labels, dtype=object)) if grouped
                 else (np.asarray(cube.source_country.labels, dtype=object),))
    sizes = [len(consumers), *(len(axis) for axis in tail_axes)]
    strides = [int(np.prod(sizes[index + 1:])) for index in range(len(sizes))]
    for start in range(0, values.size, chunk_rows):
        stop = min(start + chunk_rows, values.size); flat = np.arange(start, stop, dtype=np.int64)
        emitter_index = flat // int(np.prod(sizes)); remainder = flat % int(np.prod(sizes))
        indices = []
        for size, stride in zip(sizes, strides, strict=True):
            indices.append(remainder // stride); remainder %= stride
        value_chunk = values[(emitter_index, *indices)]
        yield (emitter_countries[emitter_index], emitter_sectors[emitter_index],
               consumers[indices[0]],
               *(axis[index] for axis, index in zip(tail_axes, indices[1:], strict=True)),
               value_chunk)


def _export_yf_cube(cube: Any, destination: str | Path, *, run_id: str, units: str,
                    provenance: Mapping[str, Any] | None, file_format: str,
                    chunk_rows: int, grouped: bool) -> Path:
    if not run_id or not units:
        raise ValueError("run_id and units must be non-empty")
    if file_format not in {"csv", "parquet"}:
        raise ValueError("file_format must be csv or parquet")
    columns = YF_SOURCE_GROUP_SECTOR_CUBE_COLUMNS if grouped else YF_SOURCE_COUNTRY_CUBE_COLUMNS
    product = "final_demand_source_group_sector_cube" if grouped else "final_demand_source_country_cube"
    stem = "D_YF.source_group_sector_cube" if grouped else "D_YF.source_country_cube"
    root = Path(destination); root.mkdir(parents=True, exist_ok=True); path = root / f"{stem}.{file_format}"
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    chunks = _yf_cube_chunks(cube, chunk_rows, grouped=grouped)
    try:
        if file_format == "csv":
            with temporary_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(columns)
                for chunk in chunks: writer.writerows(zip(*chunk))
        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("Parquet export requires the parquet optional extra") from exc
            writer = None
            try:
                for chunk in chunks:
                    table = pa.table(dict(zip(columns, chunk)))
                    writer = writer or pq.ParquetWriter(temporary_path, table.schema); writer.write_table(table)
            finally:
                if writer is not None: writer.close()
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    reconciliation = {"parent": vars(cube.reconciliation)}
    if grouped:
        reconciliation["by_source_group"] = {key: vars(value) for key, value in cube.group_reconciliation.items()}
    axes = {"emitter": list(cube.emitter.labels),
            "consumer_country": list(cube.consumer_country.labels)}
    if grouped:
        axes.update(source_group=list(cube.source_group.labels), source_sector=list(cube.source_sector.labels))
    else:
        axes["source_country"] = list(cube.source_country.labels)
    manifest = {"report_schema_version": REPORT_SCHEMA_VERSION, "product": product, "effect": "D_YF",
                "attribution_scope": "final-demand sourcing",
                "sourcing_resolution": "consumer_country", "run_id": run_id, "units": units,
                "file_format": file_format, "file": path.name, "columns": list(columns),
                "shape": list(np.asarray(cube.baseline).shape), "axes": axes,
                "total": float(np.asarray(cube.baseline).sum()), "cube_metadata": dict(cube.metadata),
                "provenance": dict(provenance or {}), "reconciliation": reconciliation}
    report_name = "yf_source_group_sector_cube_report.json" if grouped else "yf_source_country_cube_report.json"
    report_path = root / report_name
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary_report.replace(report_path)
    return root


def export_yf_source_country_cube(cube: Any, destination: str | Path, *, run_id: str, units: str,
                                  provenance: Mapping[str, Any] | None = None,
                                  file_format: str = "parquet", chunk_rows: int = 100_000) -> Path:
    return _export_yf_cube(cube, destination, run_id=run_id, units=units, provenance=provenance,
                           file_format=file_format, chunk_rows=chunk_rows, grouped=False)


def export_yf_source_group_sector_cube(cube: Any, destination: str | Path, *, run_id: str, units: str,
                                       provenance: Mapping[str, Any] | None = None,
                                       file_format: str = "parquet", chunk_rows: int = 100_000) -> Path:
    return _export_yf_cube(cube, destination, run_id=run_id, units=units, provenance=provenance,
                           file_format=file_format, chunk_rows=chunk_rows, grouped=True)


def _export_interaction(interaction: Any, destination: str | Path, *, run_id: str,
                        units: str, provenance: Mapping[str, Any] | None,
                        file_format: str, yf: bool) -> Path:
    if not run_id or not units:
        raise ValueError("run_id and units must be non-empty")
    if file_format not in {"csv", "parquet"}:
        raise ValueError("file_format must be csv or parquet")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    stem = ("D_YF.source_destination_group_product_interaction" if yf
            else "D_C.source_destination_group_sector_interaction")
    columns = YF_INTERACTION_COLUMNS if yf else DC_INTERACTION_COLUMNS
    path = root / f"{stem}.{file_format}"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    values = np.asarray(interaction.baseline, dtype=np.float64)
    rows = []
    for index in np.ndindex(values.shape):
        if yf:
            relationship, source_group, destination_group, sector = index
            labels = (
                interaction.relationship.labels[relationship],
                interaction.source_group.labels[source_group],
                interaction.destination_group.labels[destination_group],
                interaction.product_sector.labels[sector],
            )
        else:
            relationship, source_group, source_sector, destination_group, destination_sector = index
            labels = (
                interaction.relationship.labels[relationship],
                interaction.source_group.labels[source_group],
                interaction.source_sector.labels[source_sector],
                interaction.destination_group.labels[destination_group],
                interaction.destination_sector.labels[destination_sector],
            )
        rows.append((*labels, float(values[index])))
    try:
        if file_format == "csv":
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(rows)
        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("Parquet export requires the parquet optional extra") from exc
            pq.write_table(pa.table(dict(zip(columns, zip(*rows, strict=True)))), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    report = interaction.reconciliation
    axes = {
        "relationship": list(interaction.relationship.labels),
        "source_group": list(interaction.source_group.labels),
        "destination_group": list(interaction.destination_group.labels),
    }
    if yf:
        axes["product_sector"] = list(interaction.product_sector.labels)
    else:
        axes["source_sector"] = list(interaction.source_sector.labels)
        axes["destination_sector"] = list(interaction.destination_sector.labels)
    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "product": "yf_group_product_interaction" if yf else "dc_group_sector_interaction",
        "effect": "D_YF" if yf else "D_C", "run_id": run_id, "units": units,
        "file_format": file_format, "file": path.name, "columns": list(columns),
        "shape": list(values.shape), "axes": axes, "total": float(values.sum()),
        "interaction_metadata": dict(interaction.metadata),
        "provenance": dict(provenance or {}), "reconciliation": vars(report),
    }
    report_path = root / ("yf_group_product_interaction_report.json" if yf
                          else "source_group_sector_interaction_report.json")
    report_temp = report_path.with_name(f".{report_path.name}.tmp")
    report_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_temp.replace(report_path)
    return root


def export_source_group_sector_interaction(interaction: Any, destination: str | Path, *,
        run_id: str, units: str, provenance: Mapping[str, Any] | None = None,
        file_format: str = "parquet") -> Path:
    return _export_interaction(interaction, destination, run_id=run_id, units=units,
                               provenance=provenance, file_format=file_format, yf=False)


def export_yf_group_product_interaction(interaction: Any, destination: str | Path, *,
        run_id: str, units: str, provenance: Mapping[str, Any] | None = None,
        file_format: str = "parquet") -> Path:
    return _export_interaction(interaction, destination, run_id=run_id, units=units,
                               provenance=provenance, file_format=file_format, yf=True)


def export_canonical_reports(
    result: SDAResult,
    destination: str | Path,
    *,
    run_id: str,
    units: str,
    run_metadata: Mapping[str, Any] | None = None,
    file_format: str = "csv",
) -> Path:
    """Export canonical reports and a JSON manifest from a validated result.

    ``file_format`` is ``"csv"`` (default, stdlib-only, one file per
    canonical effect written via :func:`export_effect_csv`) or
    ``"parquet"`` (:func:`export_effect_parquet`; requires the optional
    ``parquet`` extra). Parquet is materially smaller and faster to write
    for the full driver-resolution effects (``D_C``, ``D_A_x``,
    ``D_T_coef``, ``D_w_coef``, ``D_YF``, ``D_YB`` -- each up to
    n-by-n rows long-format) -- see the module docstring / runbook for
    measured sizes on a real published run.
    """
    result.validation.raise_if_invalid()
    if not run_id or not units:
        raise ValueError("run_id and units must be non-empty")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    effect_totals = {}
    for name in CANONICAL_EFFECTS:
        filename = f"{name}.{file_format}"
        _export_effect(result.effects[name], root / filename, file_format=file_format)
        files[name] = filename
        effect_totals[name] = float(np.asarray(result.effects[name].baseline).sum())

    # Nested children explain D_A_x and must never be added to the headline total.
    included = [name for name in TOP_LEVEL_EFFECTS if EFFECT_REGISTRY[name].included_in_total]
    canonical_total = sum(effect_totals[name] for name in included)
    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "storage_schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "units": units,
        "run_metadata": dict(run_metadata or {}),
        "files": files,
        "effect_totals": effect_totals,
        "headline_total": canonical_total,
        "headline_effects": included,
        "excluded_nested_effects": [
            name for name, spec in EFFECT_REGISTRY.items() if spec.parent is not None
        ],
        "validation": {
            "passed": result.validation.passed,
            "checks": dict(result.validation.checks),
            "max_abs_error": result.validation.max_abs_error,
            "max_rel_error": result.validation.max_rel_error,
        },
    }
    (root / "report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# Group-sector resolution
# ---------------------------------------------------------------------------


def build_group_axis(
    label_set: Any,
    country_group: Mapping[str, str],
    *,
    role: str,
) -> AxisGrouping:
    """Build the :class:`AxisGrouping` that re-resolves one axis to its group.

    ``label_set`` labels are either ``(country, sector)`` pairs -- grouped
    into ``(group, sector)`` while the sector is preserved -- or bare
    country labels (``D_y``'s consumer axis) -- grouped directly into their
    classification. Every country appearing on the axis must have an
    explicit entry in ``country_group``; there is no silent default and no
    axis is ever left at its raw country resolution by omission.
    """
    grouped: dict[Any, list[Any]] = {}
    for label in label_set.labels:
        if isinstance(label, tuple) and len(label) == 2:
            country, sector = label
            key: Any = (_group_of(country, country_group), sector)
        else:
            key = _group_of(label, country_group)
        grouped.setdefault(key, []).append(label)
    return AxisGrouping(role, {key: tuple(members) for key, members in grouped.items()})


def _group_of(country: Any, country_group: Mapping[str, str]) -> str:
    if country not in country_group:
        raise ValueError(f"missing group classification for country: {country!r}")
    return country_group[country]


def aggregate_to_group_sector(
    effect: EffectResult, country_group: Mapping[str, str]
) -> EffectResult:
    """Re-resolve one canonical effect from country-sector to group-sector.

    Row and column axes are grouped independently with the same
    classification. When ``effect`` carries source-origin categories (only
    ``D_C`` and ``D_YF`` do), :func:`~sda.output.aggregation.aggregate_effect`
    aggregates each of the four Domestic/Red/Non-Red/Non-Participant
    components to the identical group-sector resolution, so the result's
    ``source_categories`` already carry the full split.
    """
    rows = build_group_axis(
        effect.dimensions.row, country_group, role=f"{effect.dimensions.row.role} (group)"
    )
    columns = None
    if effect.dimensions.column is not None:
        columns = build_group_axis(
            effect.dimensions.column,
            country_group,
            role=f"{effect.dimensions.column.role} (group)",
        )
    return aggregate_effect(effect, rows=rows, columns=columns)


def export_group_effect_csv(effect: EffectResult, destination: str | Path) -> Path:
    """Write one group-sector-resolved effect with group-aware headers."""
    return export_effect_csv(effect, destination, spec=GROUP_EFFECT_REGISTRY[effect.name])


def export_group_effect_parquet(effect: EffectResult, destination: str | Path) -> Path:
    """Write one group-sector-resolved effect as Parquet; see :func:`export_effect_parquet`."""
    return export_effect_parquet(effect, destination, spec=GROUP_EFFECT_REGISTRY[effect.name])


def _export_group_effect(effect: EffectResult, destination: str | Path, *, file_format: str) -> Path:
    if file_format == "csv":
        return export_group_effect_csv(effect, destination)
    if file_format == "parquet":
        return export_group_effect_parquet(effect, destination)
    raise ValueError(f"file_format must be 'csv' or 'parquet', got {file_format!r}")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def build_group_sector_reports(
    result: SDAResult,
    destination: str | Path,
    *,
    run_id: str,
    units: str,
    country_group: Mapping[str, str],
    run_metadata: Mapping[str, Any] | None = None,
    file_format: str = "csv",
) -> Path:
    """Export canonical group-sector resolution tables from a validated result.

    Every canonical effect (top-level and nested) is re-resolved from
    country-sector to group-sector using ``country_group``. ``D_C`` and
    ``D_YF`` additionally carry the full Domestic/Red/Non-Red/
    Non-Participant source-origin split at this resolution: both the
    unsplit group-sector total and one file per category are written, so a
    downstream consumer never has to reconstruct Domestic as a post-hoc
    filter. Every exported subtotal reconciles to the canonical detailed
    store within the project's standard tolerance -- see the written
    manifest's ``reconciliation`` block, and
    :func:`~sda.output.aggregation.aggregate_effect`'s own conservation
    check, which this raises through if an aggregation ever failed to
    conserve its total.
    """
    result.validation.raise_if_invalid()
    if not run_id or not units:
        raise ValueError("run_id and units must be non-empty")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    source_category_files: dict[str, dict[str, str]] = {}
    effect_totals: dict[str, float] = {}
    reconciliation: dict[str, dict[str, float]] = {}

    for name in CANONICAL_EFFECTS:
        detailed = result.effects[name]
        grouped = aggregate_to_group_sector(detailed, country_group)

        filename = f"{name}.group.{file_format}"
        _export_group_effect(grouped, root / filename, file_format=file_format)
        files[name] = filename

        detailed_total = float(np.asarray(detailed.baseline).sum())
        grouped_total = float(np.asarray(grouped.baseline).sum())
        effect_totals[name] = grouped_total
        reconciliation[name] = {
            "detailed_total": detailed_total,
            "group_total": grouped_total,
            "abs_error": abs(detailed_total - grouped_total),
        }

        if grouped.source_categories:
            source_category_files[name] = {}
            for category, component in grouped.source_categories.items():
                category_filename = f"{name}.{_slug(category)}.group.{file_format}"
                _export_group_effect(component, root / category_filename, file_format=file_format)
                source_category_files[name][category] = category_filename

    included = [name for name in TOP_LEVEL_EFFECTS if EFFECT_REGISTRY[name].included_in_total]
    headline_total = sum(effect_totals[name] for name in included)

    narrative_text = render_group_narratives(result, country_group, run_id=run_id, units=units)
    (root / "narrative.txt").write_text(narrative_text, encoding="utf-8")

    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "storage_schema_version": SCHEMA_VERSION,
        "resolution": "group-sector",
        "run_id": run_id,
        "units": units,
        "run_metadata": dict(run_metadata or {}),
        "source_origin_groups": list(SOURCE_ORIGIN_GROUPS),
        "files": files,
        "source_category_files": source_category_files,
        "effect_totals": effect_totals,
        "headline_total": headline_total,
        "headline_effects": included,
        "excluded_nested_effects": [
            name for name, spec in EFFECT_REGISTRY.items() if spec.parent is not None
        ],
        "reconciliation": reconciliation,
        "narrative_file": "narrative.txt",
        "validation": {
            "passed": result.validation.passed,
            "checks": dict(result.validation.checks),
            "max_abs_error": result.validation.max_abs_error,
            "max_rel_error": result.validation.max_rel_error,
        },
    }
    (root / "group_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# Narrative templates
# ---------------------------------------------------------------------------

EFFECT_NARRATIVE_LABELS: Mapping[str, str] = {
    "D_v": "value-intensity effect",
    "D_C": "structural composition effect",
    "D_A_x": "technical coefficient effect",
    "D_T_coef": "input-structure sub-effect of the technical coefficient effect",
    "D_w_coef": "intermediate-input-share sub-effect of the technical coefficient effect",
    "D_YF": "final-demand composition effect",
    "D_YB": "final-demand sourcing effect",
    "D_y": "final-demand level effect",
}

# Canonical narrative language (handover Sec. 10.1): every generated sentence
# must describe an effect as attributed to, associated with, or traced to
# changes in a driver -- never as caused by it. This package computes
# accounting decompositions, not a causal model.
REQUIRED_NARRATIVE_PHRASES: tuple[str, ...] = (
    "attributed to",
    "associated with",
    "traced to changes in",
)

BANNED_NARRATIVE_PHRASES: tuple[str, ...] = (
    "caused by",
    "causes",
    "causation",
    "the cause of",
)


def lint_narrative_terminology(text: str) -> tuple[str, ...]:
    """Return every banned causal-attribution phrase found in ``text``.

    An empty result means the text uses only the approved
    attributed-to / associated-with / traced-to-changes-in vocabulary.
    """
    lowered = text.lower()
    return tuple(phrase for phrase in BANNED_NARRATIVE_PHRASES if phrase in lowered)


def render_effect_narrative(
    name: str, effect: EffectResult, *, run_id: str, units: str
) -> str:
    """Render one canonical-language paragraph describing a group-sector effect."""
    label = EFFECT_NARRATIVE_LABELS.get(name, name)
    total = float(np.asarray(effect.baseline).sum())
    sentence = (
        f"In run {run_id}, {total:,.6g} {units} of {name} ({label}) at group-sector "
        f"resolution is attributed to structural and final-demand changes over the "
        f"decomposition period."
    )
    if effect.source_categories:
        categories = ", ".join(effect.source_categories)
        sentence += (
            f" This total is associated with the {categories} source-origin split, "
            f"with every category traced to changes in the driver country's "
            f"classification relative to the emitter."
        )
    return sentence


def render_group_narratives(
    result: SDAResult,
    country_group: Mapping[str, str],
    *,
    run_id: str,
    units: str,
) -> str:
    """Render the full group-sector narrative text for one validated run."""
    status = "PASSED" if result.validation.passed else "FAILED"
    lines = [f"Run {run_id} -- validation status: {status}."]
    for name in CANONICAL_EFFECTS:
        grouped = aggregate_to_group_sector(result.effects[name], country_group)
        lines.append(render_effect_narrative(name, grouped, run_id=run_id, units=units))
    return "\n".join(lines)
