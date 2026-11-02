import csv
import json

import numpy as np

from sda import CANONICAL_EFFECTS, EffectDimensions, EffectResult, LabelSet, SDAResult, ValidationReport
from sda.output import EFFECT_REGISTRY, export_canonical_reports, export_effect_csv, export_effect_report


EMITTERS = LabelSet("emitter_country_sector", (("A", "x"), ("B", "y")))
STRUCTURAL = LabelSet("structural_driver_country_sector", (("A", "m"), ("B", "n")))
DEMAND = LabelSet("consumer_demanded_sector", (("A", "food"), ("B", "energy")))
CONSUMERS = LabelSet("consumer_country", ("A", "B"))


def _result():
    effects = {}
    for index, name in enumerate(CANONICAL_EFFECTS, start=1):
        if name == "D_v":
            dimensions = EffectDimensions(EMITTERS)
            data = np.full(2, index, dtype=float)
        elif name == "D_y":
            dimensions = EffectDimensions(EMITTERS, CONSUMERS)
            data = np.full((2, 2), index, dtype=float)
        elif name in {"D_YF", "D_YB"}:
            dimensions = EffectDimensions(EMITTERS, DEMAND)
            data = np.full((2, 2), index, dtype=float)
        else:
            dimensions = EffectDimensions(EMITTERS, STRUCTURAL)
            data = np.full((2, 2), index, dtype=float)
        effects[name] = EffectResult(name, data, dimensions)
    return SDAResult(
        effects,
        np.array([1.0, 2.0]),
        ValidationReport(True, {"canonical_reconciliation": True}),
    )


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_effect_registry_declares_hierarchy_safe_totals():
    assert EFFECT_REGISTRY["D_A_x"].included_in_total
    assert EFFECT_REGISTRY["D_T_coef"].parent == "D_A_x"
    assert not EFFECT_REGISTRY["D_T_coef"].included_in_total
    assert not EFFECT_REGISTRY["D_w_coef"].included_in_total


def test_csv_headers_and_rows_follow_effect_semantics(tmp_path):
    result = _result()
    for name, expected in {
        "D_v": ["Emitter Country", "Emitter Sector", "Value"],
        "D_C": ["Emitter Country", "Emitter Sector", "Driver Country", "Driver Sector", "Value"],
        "D_YF": ["Emitter Country", "Emitter Sector", "Consumer Country", "Demanded Sector", "Value"],
        "D_y": ["Emitter Country", "Emitter Sector", "Consumer Country", "Value"],
    }.items():
        path = export_effect_csv(result.effects[name], tmp_path / f"{name}.csv")
        rows = _read_csv(path)
        assert rows[0] == expected
        assert len(rows) == (3 if name == "D_v" else 5)


def test_canonical_report_has_metadata_and_does_not_double_count_children(tmp_path):
    result = _result()
    root = export_canonical_reports(
        result, tmp_path / "report", run_id="run-42", units="kg CO2e", run_metadata={"year": 2024}
    )
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    expected_names = ["D_v", "D_C", "D_A_x", "D_YF", "D_YB", "D_y"]
    assert report["headline_effects"] == expected_names
    assert report["excluded_nested_effects"] == ["D_T_coef", "D_w_coef"]
    assert report["headline_total"] == sum(report["effect_totals"][name] for name in expected_names)
    assert report["run_id"] == "run-42"
    assert report["units"] == "kg CO2e"
    assert report["run_metadata"] == {"year": 2024}
    assert report["report_schema_version"] == 1
    assert set(report["files"]) == set(CANONICAL_EFFECTS)


def test_single_effect_report_carries_run_schema_and_units(tmp_path):
    root = export_effect_report(
        _result().effects["D_v"], tmp_path / "effect", run_id="run-43", units="t CO2e"
    )
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["run_id"] == "run-43"
    assert report["units"] == "t CO2e"
    assert report["report_schema_version"] == 1
    assert report["storage_schema_version"] == 1
    assert report["headline_effects"] == ["D_v"]
