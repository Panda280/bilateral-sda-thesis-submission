import csv
import json

import numpy as np

from build_tw_granularity_report import build
from sda.model import EffectDimensions, EffectResult, LabelSet, SDAResult, ValidationReport
from sda.output.storage import publish_result


def _publish(root):
    labels = LabelSet("country-sector", (("A", "s1"), ("A", "s2")))
    dims = EffectDimensions(labels, labels)
    t = np.array([[2.0, -1.0], [0.0, 1.0]])
    w = np.array([[-1.0, 1.0], [1.0, 0.0]])
    effects = {}
    for name in ("D_C", "D_YF", "D_YB"):
        effects[name] = EffectResult(name, np.zeros((2, 2)), dims)
    effects["D_y"] = EffectResult("D_y", np.zeros((2, 2)), dims)
    effects["D_v"] = EffectResult("D_v", np.zeros(2), EffectDimensions(labels))
    effects["D_T_coef"] = EffectResult("D_T_coef", t, dims)
    effects["D_w_coef"] = EffectResult("D_w_coef", w, dims)
    effects["D_A_x"] = EffectResult("D_A_x", t + w, dims)
    result = SDAResult(effects, (t + w).sum(axis=1), ValidationReport(True, {"synthetic": True}))
    return publish_result(result, root, "sda_2000_2001_R3")


def _concordance(path, complete=True):
    rows = [("A", "s1", "A", "all")]
    if complete:
        rows.append(("A", "s2", "A", "all"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(
            ["fine_country", "fine_sector", "coarse_country", "coarse_sector"]
        ); writer.writerows(rows)


def test_builds_machine_readable_provenance_and_passes_gates(tmp_path):
    run = _publish(tmp_path / "canonical")
    concordance = tmp_path / "map.csv"; _concordance(concordance)
    output = build([run], concordance, tmp_path / "report", units="kg CO2",
                   price_basis="CYP/PYP", tolerance=1e-10, provisional=True, expected_runs=1)
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["run_count"] == 1 and manifest["row_count"] == 4
    assert all(manifest["validation"]["gates"].values())
    assert manifest["status"] == "complete"
    assert manifest["validation"]["thesis_status"] == "accepted_provisional"
    rows = list(csv.DictReader((output / "tw_granularity_results.csv").open()))
    assert {row["effect_id"] for row in rows} == {"D_A_x", "D_T_coef", "D_w_coef", "D_T_coef+D_w_coef"}


def test_refuses_incomplete_concordance_without_publishing(tmp_path):
    run = _publish(tmp_path / "canonical")
    concordance = tmp_path / "map.csv"; _concordance(concordance, complete=False)
    try:
        build([run], concordance, tmp_path / "report", units="kg CO2",
              price_basis="CYP/PYP", tolerance=1e-10, provisional=True, expected_runs=1)
    except ValueError as exc:
        assert "concordance misses" in str(exc)
    else:
        raise AssertionError("incomplete concordance accepted")
    assert not (tmp_path / "report").exists()
