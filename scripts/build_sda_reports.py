"""CLI: build canonical detailed and group-sector report products from a published run.

Reads a run directory produced by ``sda.output.storage.publish_result`` (the
target tree phase-10 names ``scripts/build_sda_reports.py`` for) and writes:

* ``<output-dir>/detailed/`` -- the existing country-sector resolution CSVs
  and manifest (:func:`sda.output.export_canonical_reports`).
* ``<output-dir>/group/`` -- the new group-sector resolution tables,
  including the full Domestic/Red/Non-Red/Non-Participant source-origin
  split for ``D_C``/``D_YF`` (:func:`sda.output.build_group_sector_reports`),
  plus a narrative text file.
* ``<output-dir>/run_report.json`` -- a small top-level manifest tying both
  together with the run's validation status and ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

_SRC = Path(__file__).resolve().parent.parent / "src"
if (_SRC / "sda").is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sda.output import (  # noqa: E402
    build_group_sector_reports,
    export_canonical_reports,
    load_result,
)


class ReportBuildError(ValueError):
    """The report inputs are incomplete or unsafe to build from."""


def load_country_group_mapping(path: str | Path) -> dict[str, str]:
    """Load an external country -> classification-group mapping from JSON.

    The file must be a flat JSON object mapping each country code to its
    base classification (``Red``, ``Non-Red``, or ``Non-Participant`` --
    the same base classification :mod:`sda.attribution.groups` uses before
    the Domestic-relative override; see
    docs/decisions/0006-group-sector-report-resolution.md). This is the
    reviewed-mapping path: once a canonical run's manifest carries an
    explicit classification snapshot, prefer that instead.
    """
    mapping_path = Path(path)
    try:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportBuildError(f"cannot read country-group mapping: {mapping_path}") from exc
    if (
        not isinstance(raw, dict)
        or not raw
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in raw.items())
    ):
        raise ReportBuildError(
            "country-group mapping must be a non-empty flat JSON object of str -> str"
        )
    return dict(raw)


def build_reports(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    run_id: str | None,
    units: str,
    country_group: dict[str, str],
) -> Path:
    """Load one published canonical run and write detailed and group-sector reports."""
    run_path = Path(run_dir)
    result = load_result(run_path)
    resolved_run_id = run_id or run_path.name
    if not resolved_run_id:
        raise ReportBuildError("run_id must be provided or derivable from --run-dir")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_metadata = {"source_run_directory": str(run_path)}

    export_canonical_reports(
        result,
        output_root / "detailed",
        run_id=resolved_run_id,
        units=units,
        run_metadata=run_metadata,
    )
    build_group_sector_reports(
        result,
        output_root / "group",
        run_id=resolved_run_id,
        units=units,
        country_group=country_group,
        run_metadata=run_metadata,
    )

    top_manifest = {
        "run_id": resolved_run_id,
        "units": units,
        "source_run_directory": str(run_path),
        "validation": {
            "passed": result.validation.passed,
            "checks": dict(result.validation.checks),
            "max_abs_error": result.validation.max_abs_error,
            "max_rel_error": result.validation.max_rel_error,
        },
        "detailed_report": "detailed/report.json",
        "group_report": "group/group_report.json",
        "narrative": "group/narrative.txt",
    }
    (output_root / "run_report.json").write_text(
        json.dumps(top_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical detailed and group-sector SDA report products "
            "from a published run."
        )
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help="Published canonical run directory (sda.output.storage.publish_result output).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write report products into.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Defaults to the run directory's name.",
    )
    parser.add_argument(
        "--units", required=True,
        help="Effect units, e.g. 'kg CO2e'.",
    )
    parser.add_argument(
        "--country-group-map", required=True, type=Path,
        help="JSON file: country code -> Red/Non-Red/Non-Participant classification.",
    )
    args = parser.parse_args(argv)

    country_group = load_country_group_mapping(args.country_group_map)
    output_root = build_reports(
        args.run_dir,
        args.output_dir,
        run_id=args.run_id,
        units=args.units,
        country_group=country_group,
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
