import json

import numpy as np
import pyarrow.parquet as pq

from sda.model import EffectDimensions, EffectResult, LabelSet, SDAResult, ValidationReport
from sda.output.annual import publish_annual_report, publish_panel


def _result():
    emitter = LabelSet("emitter country-sector", (("A", "s0"), ("B", "s1")))
    driver = LabelSet("structural driver country-sector", (("A", "s0"), ("B", "s1")))
    demand = LabelSet("consumer country and demanded sector", (("A", "s0"), ("B", "s1")))
    consumer = LabelSet("consumer country", ("A", "B"))
    matrices = {
        "D_C": np.array([[2.0, -1.0], [0.0, 1.0]]),
        "D_A_x": np.array([[1.0, 1.0], [-1.0, 1.0]]),
        "D_T_coef": np.array([[0.5, 0.25], [-0.5, 0.75]]),
        "D_w_coef": np.array([[0.5, 0.75], [-0.5, 0.25]]),
        "D_YF": np.array([[1.0, -1.0], [1.0, 0.0]]),
        "D_YB": np.array([[0.0, 1.0], [-1.0, 0.0]]),
        "D_y": np.array([[1.0, 0.0], [0.0, -1.0]]),
    }
    effects = {"D_v": EffectResult("D_v", np.array([1.0, -1.0]), EffectDimensions(emitter))}
    for name, values in matrices.items():
        columns = driver if name in {"D_C", "D_A_x", "D_T_coef", "D_w_coef"} else demand
        if name == "D_y": columns = consumer
        effects[name] = EffectResult(name, values, EffectDimensions(emitter, columns))
    total = sum(effect.baseline if effect.baseline.ndim == 1 else effect.baseline.sum(axis=1)
                for name, effect in effects.items() if name in {"D_v", "D_C", "D_A_x", "D_YF", "D_YB", "D_y"})
    return SDAResult(effects, total, ValidationReport(True, {"canonical": True}))


def test_annual_report_is_atomic_role_safe_and_signed(tmp_path):
    published = publish_annual_report(
        _result(), tmp_path, run_id="annual_2000_2001", year_prev=2000, year=2001,
        resolution="WIOT2013_PYP", units="kg CO2", source_files=[{"path": "state.npz", "sha256": "abc"}],
        country_groups={"A": "Red", "B": "Non-Red"}, sector_groups={"s0": "Goods", "s1": "Services"},
    )
    manifest = json.loads((published / "manifest.v2.json").read_text())
    assert manifest["validation_status"] == "passed"
    assert manifest["year_pair"] == [2000, 2001]
    assert set(manifest["roles"]) == {"emitter", "structural_driver", "consumer", "demanded_sector", "source"}
    hierarchy = pq.read_table(published / "annual_hierarchy.parquet").to_pylist()
    assert {row["node_id"] for row in hierarchy} >= {"delta_g", "emission_intensity", "D_v", "leontief_production_structure"}
    intensity = next(row for row in hierarchy if row["node_id"] == "D_v")
    assert intensity["positive_contribution"] == 1.0
    assert intensity["negative_contribution"] == -1.0
    rankings = pq.read_table(published / "annual_drilldown.parquet").to_pylist()
    yf_roles = {row["grain_id"] for row in rankings if row["effect_id"] == "D_YF"}
    assert "consumer_country" in yf_roles and "demanded_sector" in yf_roles
    assert "driver_country" not in yf_roles
    assert not list(tmp_path.glob(".staging-*"))


def test_panel_combines_validated_annual_runs(tmp_path):
    result = _result()
    a = publish_annual_report(result, tmp_path / "annual", run_id="a", year_prev=2000, year=2001,
        resolution="r", units="kg", source_files=[{"path": "a", "sha256": "1"}])
    b = publish_annual_report(result, tmp_path / "annual", run_id="b", year_prev=2001, year=2002,
        resolution="r", units="kg", source_files=[{"path": "b", "sha256": "2"}])
    panel = publish_panel([a, b], tmp_path / "panel", run_id="full")
    summary = json.loads((panel / "panel_summary.v2.json").read_text())
    assert summary["annual_runs"] == 2
    assert summary["panel_count"] == 11
    try:
        publish_panel([a, a], tmp_path / "panel", run_id="duplicate")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate transition was accepted")


def test_failed_report_does_not_publish(tmp_path):
    result = _result()
    effects = dict(result.effects)
    effects["D_w_coef"] = EffectResult("D_w_coef", effects["D_w_coef"].baseline + 1,
                                        effects["D_w_coef"].dimensions)
    broken = SDAResult(effects, result.total_change, result.validation)
    try:
        publish_annual_report(broken, tmp_path, run_id="broken", year_prev=2000, year=2001,
                              resolution="r", units="kg", source_files=[{"path": "x"}])
    except ValueError:
        pass
    else:
        raise AssertionError("broken hierarchy published")
    assert not (tmp_path / "broken").exists()
    assert not list(tmp_path.glob(".staging-*"))
