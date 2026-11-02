#!/usr/bin/env python
"""Publish one canonical bilateral-SDA run from an exported state bundle.

Consumes the ``config.json`` / ``state.npz`` / ``bundle_manifest.json``
directory that ``EXIODeflator/scripts/export_sda_state_bundle.py`` writes
(see ``docs/decisions/0004-bridge-execution-environment.md`` and
``docs/decisions/0005-bundle-export-boundary.md``). This half is pure NumPy
and runs entirely inside this repository's own environment -- it never
imports EXIODeflator or CuPy.

Fills the ``scripts/`` directory the original target tree called for.

Usage::

    python scripts/build_production_run.py \\
        --bundle-dir /path/to/bundle_2004_2013 \\
        --output-root results/sda/bilateral_v1 \\
        --run-id prod_2004_2013

Publishes the run via ``sda.cli.execute_config`` (identical validation gates
as the CLI itself -- this script adds no new execution path) and copies the
bundle's source-file manifest into the published run directory as
``source_provenance.json``, so the acceptance gate ("a manifest that traces
back to the exact source files consumed") holds for the published run
itself, not just the intermediate bundle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sda.cli import ConfigurationError, execute_config  # noqa: E402
from sda.integration.bundle_export import CONFIG_NAME, MANIFEST_NAME  # noqa: E402

SOURCE_PROVENANCE_NAME = "source_provenance.json"


class BridgeError(ValueError):
    """The bundle directory is not a valid export_sda_state_bundle.py output."""


def build_production_run(bundle_dir: str | Path, output_root: str | Path, run_id: str) -> Path:
    """Publish one canonical run from ``bundle_dir`` and return its directory.

    ``bundle_dir`` must contain ``config.json`` (validated and consumed
    unmodified by ``sda.cli.execute_config``) and ``bundle_manifest.json``
    (this function's own provenance requirement -- a bundle without one is
    refused rather than published anonymously).
    """
    bundle_dir = Path(bundle_dir)
    config_path = bundle_dir / CONFIG_NAME
    manifest_path = bundle_dir / MANIFEST_NAME
    if not config_path.is_file():
        raise BridgeError(f"bundle directory is missing {CONFIG_NAME}: {bundle_dir}")
    if not manifest_path.is_file():
        raise BridgeError(f"bundle directory is missing {MANIFEST_NAME}: {bundle_dir}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"cannot read {manifest_path}") from exc
    consumed = manifest.get("consumed_files")
    if not isinstance(consumed, list) or not consumed:
        raise BridgeError(f"{manifest_path} must record at least one consumed source file")

    published = execute_config(config_path, output_root, run_id)
    shutil.copyfile(manifest_path, published / SOURCE_PROVENANCE_NAME)
    return published


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        published = build_production_run(args.bundle_dir, args.output_root, args.run_id)
    except (BridgeError, ConfigurationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
