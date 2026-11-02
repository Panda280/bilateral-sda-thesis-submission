import json
from collections import OrderedDict

import numpy as np

from sda.cli import execute_config
from sda.output import AxisGrouping, aggregate_effect, load_result


def test_canonical_cli_storage_reload_and_aggregate_round_trip(tmp_path):
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
            "emitter_labels": ["A:s0", "A:s1"],
            "structural_driver_labels": ["A:s0", "A:s1"],
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

    published = execute_config(config_path, tmp_path / "results", "tiny-run")
    loaded = load_result(published)

    assert loaded.validation.passed
    assert all(loaded.validation.checks.values())
    np.testing.assert_allclose(loaded.total_change, [1.0, 2.0])
    grouped = aggregate_effect(
        loaded.effects["D_y"],
        rows=AxisGrouping(
            "emitter_country", OrderedDict({"A": ("A:s0", "A:s1")})
        ),
        columns=AxisGrouping("consumer_country", OrderedDict({"A": ("A",)})),
    )
    np.testing.assert_allclose(grouped.baseline, [[3.0]])
    assert grouped.dimensions.row.role == "emitter_country"
    assert grouped.dimensions.column.role == "consumer_country"
