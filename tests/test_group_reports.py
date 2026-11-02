import csv
import json

import numpy as np
import pytest

from sda import LabelSet
from sda.output import (
    GROUP_EFFECT_REGISTRY,
    aggregate_to_group_sector,
    build_group_axis,
    build_group_sector_reports,
)

from _group_report_fixtures import GROUPS, build_synthetic_result


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_build_group_axis_groups_country_sector_pairs_preserving_sector():
    labels = LabelSet(
        "emitter country-sector", (("A", "s0"), ("A", "s1"), ("B", "s0"), ("C", "s0"))
    )
    grouping = build_group_axis(labels, GROUPS, role="emitter (group)")

    assert grouping.role == "emitter (group)"
    assert set(grouping.groups) == {
        ("Red", "s0"), ("Red", "s1"), ("Non-Red", "s0"), ("Non-Participant", "s0"),
    }
    assert grouping.groups[("Red", "s0")] == (("A", "s0"),)
    assert grouping.groups[("Red", "s1")] == (("A", "s1"),)


def test_build_group_axis_groups_bare_country_labels():
    labels = LabelSet("consumer country", ("A", "B", "C"))
    grouping = build_group_axis(labels, GROUPS, role="consumer (group)")

    assert grouping.groups == {"Red": ("A",), "Non-Red": ("B",), "Non-Participant": ("C",)}


def test_build_group_axis_rejects_unclassified_country():
    labels = LabelSet("emitter country-sector", (("A", "s0"), ("Z", "s0")))

    with pytest.raises(ValueError, match="missing group classification"):
        build_group_axis(labels, GROUPS, role="emitter (group)")


def test_group_effect_registry_uses_group_headers_for_every_canonical_effect():
    assert GROUP_EFFECT_REGISTRY["D_v"].columns == ("Emitter Group", "Emitter Sector", "Value")
    assert GROUP_EFFECT_REGISTRY["D_C"].columns == (
        "Emitter Group", "Emitter Sector", "Driver Group", "Driver Sector", "Value",
    )
    assert GROUP_EFFECT_REGISTRY["D_A_x"].columns == GROUP_EFFECT_REGISTRY["D_C"].columns
    assert GROUP_EFFECT_REGISTRY["D_YF"].columns == (
        "Emitter Group", "Emitter Sector", "Consumer Group", "Demanded Sector", "Value",
    )
    assert GROUP_EFFECT_REGISTRY["D_y"].columns == (
        "Emitter Group", "Emitter Sector", "Consumer Group", "Value",
    )
    # Nested children stay excluded from the headline total at group resolution too.
    assert not GROUP_EFFECT_REGISTRY["D_T_coef"].included_in_total
    assert GROUP_EFFECT_REGISTRY["D_T_coef"].parent == "D_A_x"


def test_aggregate_to_group_sector_conserves_total_and_carries_source_categories():
    result = build_synthetic_result()
    grouped = aggregate_to_group_sector(result.effects["D_C"], GROUPS)

    np.testing.assert_allclose(
        grouped.baseline.sum(), result.effects["D_C"].baseline.sum(), rtol=1e-10, atol=1e-12
    )
    assert grouped.source_categories is not None
    assert set(grouped.source_categories) == {"Domestic", "Red", "Non-Red", "Non-Participant"}
    category_sum = sum(c.baseline.sum() for c in grouped.source_categories.values())
    np.testing.assert_allclose(category_sum, grouped.baseline.sum(), rtol=1e-10, atol=1e-12)


def test_aggregate_to_group_sector_leaves_effects_without_a_split_alone():
    result = build_synthetic_result()
    grouped = aggregate_to_group_sector(result.effects["D_YB"], GROUPS)
    assert grouped.source_categories is None


def test_build_group_sector_reports_writes_every_effect_and_the_full_source_split(tmp_path):
    result = build_synthetic_result()
    root = build_group_sector_reports(
        result, tmp_path / "group", run_id="grp-run", units="kg CO2e", country_group=GROUPS,
    )
    manifest = json.loads((root / "group_report.json").read_text())

    assert manifest["run_id"] == "grp-run"
    assert manifest["units"] == "kg CO2e"
    assert manifest["resolution"] == "group-sector"
    assert manifest["validation"]["passed"] is True
    assert set(manifest["files"]) == set(GROUP_EFFECT_REGISTRY)

    # D_C and D_YF alone carry the full 4-way split -- both the unsplit
    # group-sector total (in "files") and one file per source-origin
    # category, not just the total.
    assert set(manifest["source_category_files"]) == {"D_C", "D_YF"}
    for name in ("D_C", "D_YF"):
        assert set(manifest["source_category_files"][name]) == {
            "Domestic", "Red", "Non-Red", "Non-Participant",
        }
        for filename in manifest["source_category_files"][name].values():
            assert (root / filename).exists()

    for filename in manifest["files"].values():
        assert (root / filename).exists()

    assert manifest["narrative_file"] == "narrative.txt"
    assert (root / "narrative.txt").exists()

    header = _read_csv(root / manifest["files"]["D_C"])[0]
    assert header == list(GROUP_EFFECT_REGISTRY["D_C"].columns)
    header_y = _read_csv(root / manifest["files"]["D_y"])[0]
    assert header_y == list(GROUP_EFFECT_REGISTRY["D_y"].columns)


def test_build_group_sector_reports_reconciles_every_effect_total(tmp_path):
    result = build_synthetic_result()
    root = build_group_sector_reports(
        result, tmp_path / "group", run_id="grp-run-2", units="kg CO2e", country_group=GROUPS,
    )
    manifest = json.loads((root / "group_report.json").read_text())

    for name, totals in manifest["reconciliation"].items():
        assert totals["abs_error"] < 1e-9
        np.testing.assert_allclose(
            totals["detailed_total"], totals["group_total"], rtol=1e-10, atol=1e-12
        )
