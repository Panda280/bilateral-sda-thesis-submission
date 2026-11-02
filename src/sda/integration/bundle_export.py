"""Serialize a validated :class:`~sda.state.SDAState` to the flat NPZ/JSON
bundle ``sda.cli.execute_config`` already validates and consumes, unmodified.

Deliberately pure NumPy -- no EXIODeflator or CuPy import anywhere in this
module. This is the shared boundary between
``EXIODeflator/scripts/export_sda_state_bundle.py`` (which builds the
``SDAState`` from real CuPy ``TY`` objects and therefore must run inside
EXIODeflator's own CuPy environment) and this package's already-tested CLI:
the exact same serialization logic is unit-tested here against a synthetic
``SDAState`` and then reused, unmodified, by the real bridge script. See
``docs/decisions/0004-bridge-execution-environment.md`` and
``docs/decisions/0005-bundle-export-boundary.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..attribution.groups import SOURCE_ORIGIN_GROUPS
from ..cli import ARRAY_INPUTS, SCHEMA_VERSION
from ..state import SDAState

BUNDLE_NPZ_NAME = "state.npz"
CONFIG_NAME = "config.json"
MANIFEST_NAME = "bundle_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def _label(country: object, sector: object) -> str:
    """Match the flat ``"country:sector"`` label convention ``sda.cli``
    configurations already use (see ``tests/test_canonical_round_trip.py``).
    JSON has no tuple type, so a JSON list-of-pairs would deserialize to a
    list of Python lists, which ``LabelSet`` cannot hash/dedupe."""
    return f"{country}:{sector}"


def _safe_group_key(group: str) -> str:
    return group.replace("-", "_").replace(" ", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_record(path: str | Path) -> dict[str, Any]:
    """One checksummed provenance record for a consumed input file."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"cannot record provenance for missing file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def write_state_bundle(
    state: SDAState,
    bundle_dir: str | Path,
    *,
    consumed_files: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``state.npz`` + ``config.json`` + ``bundle_manifest.json`` under
    ``bundle_dir``. Returns the ``config.json`` path -- pass it straight to
    ``sda.cli.execute_config`` (or ``scripts/build_production_run.py``).

    ``config.json`` is exactly the schema ``sda.cli.execute_config`` already
    validates; no CLI change was needed for this bridge. ``bundle_manifest.json``
    records which upstream output files (paths + checksums) were consumed to
    build ``state``, per the migration handover's provenance requirement --
    every entry in ``consumed_files`` should come from
    :func:`source_file_record` (or carry the same ``path``/``sha256``/``bytes``
    shape) so a published run can always be traced back to exact source files.
    """
    if not isinstance(state, SDAState):
        raise TypeError("state must be an SDAState")
    consumed_files = list(consumed_files)
    if not consumed_files:
        raise ValueError("consumed_files must record at least one source file")

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    array_values = {name: np.asarray(getattr(state, name)) for name in ARRAY_INPUTS}

    mask_members: dict[str, str] = {}
    mask_arrays: dict[str, np.ndarray] = {}
    for family, masks in (("c", state.source_groups.C), ("yf", state.source_groups.YF)):
        for group in SOURCE_ORIGIN_GROUPS:
            member = f"{family}_mask_{_safe_group_key(group)}"
            mask_members[f"{family}:{group}"] = member
            mask_arrays[member] = np.asarray(masks[group], dtype=bool)

    npz_path = bundle_dir / BUNDLE_NPZ_NAME
    np.savez(npz_path, **array_values, **mask_arrays)

    config = {
        "schema_version": SCHEMA_VERSION,
        "input_npz": BUNDLE_NPZ_NAME,
        "arrays": {name: name for name in ARRAY_INPUTS},
        "semantics": {
            "sectors_per_country": int(state.sectors_per_country),
            "emitter_labels": [_label(c, s) for c, s in state.labels.emitters],
            "structural_driver_labels": [_label(c, s) for c, s in state.labels.structural_drivers],
            "consumer_labels": list(state.labels.consumers),
            "demanded_sector_labels": list(state.labels.demanded_sectors),
            "c_source_masks": {group: mask_members[f"c:{group}"] for group in SOURCE_ORIGIN_GROUPS},
            "yf_source_masks": {group: mask_members[f"yf:{group}"] for group in SOURCE_ORIGIN_GROUPS},
        },
    }
    config_path = bundle_dir / CONFIG_NAME
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "consumed_files": consumed_files,
        "provenance": dict(provenance or {}),
    }
    (bundle_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    return config_path
