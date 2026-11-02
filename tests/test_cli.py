import json

import numpy as np
import pytest

from sda.cli import ConfigurationError, execute_config


def test_cli_refuses_missing_semantic_contract(tmp_path):
    bundle = tmp_path / "input.npz"
    np.savez(bundle, placeholder=np.zeros(1))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema_version": 1, "input_npz": "input.npz", "arrays": {}, "semantics": {}
    }), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="arrays keys mismatch"):
        execute_config(config, tmp_path / "results", "run")


def test_cli_refuses_npz_path_escape(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema_version": 1, "input_npz": "../input.npz", "arrays": {}, "semantics": {}
    }), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="may not escape"):
        execute_config(config, tmp_path / "results", "run")


def test_cli_json_config_accepts_pair_labels_and_reaches_csv_export(tmp_path):
    """Regression test for the JSON/tuple label gap (docs/runbook.md section 5).

    JSON has no tuple type: emitter_labels/structural_driver_labels arrive
    from json.loads as lists of 2-element lists. Before this fix, that made
    execute_config crash inside run_canonical_sda with
    ``TypeError: unhashable type: 'list'`` (LabelSet needs hashable labels)
    for [[country, sector], ...] input, and export_canonical_reports's
    ``_pair`` check reject flattened scalar strings used as a workaround.
    This proves the full JSON-config -> publish -> reload -> CSV-export path
    now works with real [country, sector] pairs, exactly as the contract
    (docs/bilateral_sda_contract.md) describes emitter/driver labels.
    """
    from sda.output import export_canonical_reports, load_result

    eye = np.eye(2)
    zeros_2 = np.zeros((2, 2), dtype=bool)
    zeros_y = np.zeros((2, 1), dtype=bool)
    arrays = {
        "v0": np.array([1.0, 2.0]), "v1": np.array([1.0, 2.0]),
        "L0": eye, "L1": eye, "C0": eye, "C1": eye,
        "A_x0": eye, "A_x1": eye, "T_coef0": eye, "T_coef1": eye,
        "w_coef0": np.ones(2), "w_coef1": np.ones(2),
        "F0": np.ones((2, 1)), "F1": np.ones((2, 1)),
        "B0": np.ones((2, 1)), "B1": np.ones((2, 1)),
        "levels0": np.array([1.0]), "levels1": np.array([2.0]),
        "supplier_sector": np.array([0, 1]),
        "c_domestic": np.ones((2, 2), dtype=bool), "c_red": zeros_2,
        "c_non_red": zeros_2, "c_non_participant": zeros_2,
        "yf_domestic": np.ones((2, 1), dtype=bool), "yf_red": zeros_y,
        "yf_non_red": zeros_y, "yf_non_participant": zeros_y,
    }
    np.savez(tmp_path / "input.npz", **arrays)
    config = {
        "schema_version": 1,
        "input_npz": "input.npz",
        "arrays": {name: name for name in (
            "v0", "v1", "L0", "L1", "C0", "C1", "A_x0", "A_x1",
            "T_coef0", "T_coef1", "w_coef0", "w_coef1", "F0", "F1",
            "B0", "B1", "levels0", "levels1", "supplier_sector",
        )},
        "semantics": {
            "sectors_per_country": 2,
            "emitter_labels": [["A", "s0"], ["A", "s1"]],
            "structural_driver_labels": [["A", "s0"], ["A", "s1"]],
            "consumer_labels": ["A"],
            "demanded_sector_labels": ["s0", "s1"],
            "c_source_masks": {
                "Domestic": "c_domestic", "Red": "c_red",
                "Non-Red": "c_non_red", "Non-Participant": "c_non_participant",
            },
            "yf_source_masks": {
                "Domestic": "yf_domestic", "Red": "yf_red",
                "Non-Red": "yf_non_red", "Non-Participant": "yf_non_participant",
            },
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    published = execute_config(config_path, tmp_path / "results", "pair-label-run")
    loaded = load_result(published)
    assert loaded.validation.passed

    emitter_labels = loaded.effects["D_v"].dimensions.row.labels
    assert emitter_labels == (("A", "s0"), ("A", "s1"))
    assert all(isinstance(label, tuple) and len(label) == 2 for label in emitter_labels)

    report_dir = export_canonical_reports(loaded, tmp_path / "reports", run_id="pair-label-run", units="kg CO2e")
    assert (report_dir / "D_v.csv").exists()
    assert (report_dir / "report.json").exists()


@pytest.mark.parametrize("bad_emitter_labels", [
    [["A"], ["A", "s1"]],
    [["A", ""], ["A", "s1"]],
    [["A", "s0", "extra"], ["A", "s1"]],
    [123, "A:s1"],
    ["", "A:s1"],
])
def test_cli_rejects_malformed_pair_labels(tmp_path, bad_emitter_labels):
    bundle = tmp_path / "input.npz"
    np.savez(bundle, placeholder=np.zeros(1))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema_version": 1, "input_npz": "input.npz",
        "arrays": {name: "placeholder" for name in (
            "v0", "v1", "L0", "L1", "C0", "C1", "A_x0", "A_x1",
            "T_coef0", "T_coef1", "w_coef0", "w_coef1", "F0", "F1",
            "B0", "B1", "levels0", "levels1", "supplier_sector",
        )},
        "semantics": {
            "sectors_per_country": 2,
            "emitter_labels": bad_emitter_labels,
            "structural_driver_labels": [["A", "s0"], ["A", "s1"]],
            "consumer_labels": ["A"],
            "demanded_sector_labels": ["s0", "s1"],
            "c_source_masks": {
                "Domestic": "placeholder", "Red": "placeholder",
                "Non-Red": "placeholder", "Non-Participant": "placeholder",
            },
            "yf_source_masks": {
                "Domestic": "placeholder", "Red": "placeholder",
                "Non-Red": "placeholder", "Non-Participant": "placeholder",
            },
        },
    }), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"country, sector\] pair"):
        execute_config(config, tmp_path / "results", "run")
