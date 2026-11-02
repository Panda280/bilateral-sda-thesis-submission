"""Small, explicit command-line entry point for canonical SDA runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decomposition import run_canonical_sda
from .output import publish_result

SCHEMA_VERSION = 1
ARRAY_INPUTS = (
    "v0", "v1", "L0", "L1", "C0", "C1", "A_x0", "A_x1",
    "T_coef0", "T_coef1", "w_coef0", "w_coef1", "F0", "F1",
    "B0", "B1", "levels0", "levels1", "supplier_sector",
)
SEMANTIC_INPUTS = (
    "sectors_per_country", "emitter_labels", "structural_driver_labels",
    "consumer_labels", "demanded_sector_labels", "c_source_masks",
    "yf_source_masks",
)
# emitter_labels/structural_driver_labels are (country, sector) pairs per the
# canonical contract (docs/bilateral_sda_contract.md). A config may spell
# each label either as a pre-composed scalar string (e.g. "A:s0" -- used
# throughout the existing test suite; any hashable token is a valid
# LabelSet label) or as a genuine [country, sector] JSON pair. JSON has no
# tuple type, so the pair form arrives from json.loads as a 2-element list;
# LabelSet.__post_init__ needs hashable labels (a list is not), and
# sda.output.reports._pair needs an actual 2-tuple to reach CSV export --
# so pair-shaped entries are coerced to real tuples below, while scalar
# string entries pass through unchanged. consumer_labels/demanded_sector_labels
# are always flat scalar tokens and are never coerced.
PAIR_LABEL_INPUTS = frozenset({"emitter_labels", "structural_driver_labels"})


class ConfigurationError(ValueError):
    """The execution configuration is incomplete or unsafe."""


def _coerce_pair_labels(labels: Sequence[Any]) -> list[Any]:
    """Turn JSON-decoded [country, sector] lists into real tuples; leave scalar labels as-is."""

    return [tuple(item) if isinstance(item, list) else item for item in labels]


def execute_config(config_path: str | Path, output_root: str | Path, run_id: str) -> Path:
    """Load an explicit JSON/NPZ input bundle, validate it, and publish one run."""

    path = Path(config_path).resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read configuration: {path}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ConfigurationError(f"configuration schema_version must be {SCHEMA_VERSION}")
    _require_exact_mapping(config, {"schema_version", "input_npz", "arrays", "semantics"}, "configuration")

    npz_name = config["input_npz"]
    if not isinstance(npz_name, str) or not npz_name or Path(npz_name).is_absolute():
        raise ConfigurationError("input_npz must be a non-empty path relative to the configuration")
    bundle = (path.parent / npz_name).resolve()
    if path.parent not in bundle.parents:
        raise ConfigurationError("input_npz may not escape the configuration directory")

    arrays = config["arrays"]
    semantics = config["semantics"]
    _require_exact_mapping(arrays, set(ARRAY_INPUTS), "arrays")
    _require_exact_mapping(semantics, set(SEMANTIC_INPUTS), "semantics")
    if any(not isinstance(arrays[name], str) or not arrays[name] for name in ARRAY_INPUTS):
        raise ConfigurationError("every arrays value must name one NPZ member")
    _validate_semantics(semantics)

    try:
        with np.load(bundle, allow_pickle=False) as archive:
            missing = sorted(set(arrays.values()) - set(archive.files))
            if missing:
                raise ConfigurationError(f"NPZ members are missing: {missing}")
            inputs = {name: np.array(archive[arrays[name]], copy=True) for name in ARRAY_INPUTS}
            for mask_family in ("c_source_masks", "yf_source_masks"):
                inputs[mask_family] = {
                    group: np.array(archive[member], copy=True)
                    for group, member in semantics[mask_family].items()
                }
    except ConfigurationError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot safely load NPZ input: {bundle}") from exc

    inputs.update({
        name: _coerce_pair_labels(semantics[name]) if name in PAIR_LABEL_INPUTS else semantics[name]
        for name in SEMANTIC_INPUTS if not name.endswith("_masks")
    })
    result = run_canonical_sda(**inputs)
    return publish_result(result, output_root, run_id)


def _require_exact_mapping(value: Any, required: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be an object")
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - required)
    if missing or unexpected:
        raise ConfigurationError(f"{label} keys mismatch; missing={missing}, unexpected={unexpected}")


def _is_malformed_pair_entry(item: Any) -> bool:
    if isinstance(item, str):
        return not item
    if isinstance(item, list):
        return len(item) != 2 or not all(isinstance(part, str) and part for part in item)
    return True


def _validate_semantics(semantics: Mapping[str, Any]) -> None:
    if not isinstance(semantics["sectors_per_country"], int) or semantics["sectors_per_country"] <= 0:
        raise ConfigurationError("sectors_per_country must be a positive integer")
    for name in ("emitter_labels", "structural_driver_labels", "consumer_labels", "demanded_sector_labels"):
        labels = semantics[name]
        if not isinstance(labels, list) or not labels:
            raise ConfigurationError(f"{name} must be a non-empty JSON list")
        if name in PAIR_LABEL_INPUTS:
            if any(_is_malformed_pair_entry(item) for item in labels):
                raise ConfigurationError(
                    f"{name} entries must each be a non-empty string, or a "
                    "[country, sector] pair of non-empty strings"
                )
        else:
            if any(not isinstance(item, str) or not item for item in labels):
                raise ConfigurationError(f"{name} entries must be non-empty strings")
    required_groups = {"Domestic", "Red", "Non-Red", "Non-Participant"}
    for name in ("c_source_masks", "yf_source_masks"):
        masks = semantics[name]
        _require_exact_mapping(masks, required_groups, name)
        if any(not isinstance(member, str) or not member for member in masks.values()):
            raise ConfigurationError(f"{name} values must name NPZ members")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and publish a validated canonical bilateral SDA")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    published = execute_config(args.config, args.output_root, args.run_id)
    print(published)
    return 0

