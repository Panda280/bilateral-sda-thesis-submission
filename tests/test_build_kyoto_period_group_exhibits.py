from pathlib import Path

import numpy as np
import pandas as pd

from build_kyoto_period_group_exhibits import (
    GROUPS,
    aggregate_matrices,
    aggregate_relationship_view,
    select_overview,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "tests" / "fixtures" / "thesis"


def _load():
    effects = pd.read_csv(PRODUCTS / "kyoto_period_group_effects.csv", dtype={"resolution": str})
    interactions = pd.read_csv(PRODUCTS / "kyoto_period_group_interactions.csv", dtype={"resolution": str})
    return effects, interactions


def test_registered_products_pass_display_contract():
    effects, interactions = _load()
    result = validate_inputs(effects, interactions)
    assert result["expected_transition_counts"] == {"P1": 4, "P2": 3, "P3": 5}
    assert result["configurations"] == ["exio_2013", "exio_2016", "exio_R3"]


def test_overview_has_one_role_and_complete_group_period_inventory():
    effects, _ = _load()
    overview = select_overview(effects, "exio_R3")
    assert len(overview) == 3 * 3 * 7
    assert not overview.duplicated(["period", "effect", "group"]).any()


def test_relationship_and_matrix_aggregations_reconstruct_registered_rows():
    _, interactions = _load()
    for config in sorted(interactions.configuration.unique()):
        for effect in ("D_C", "D_YF"):
            source = interactions[(interactions.configuration == config) & (interactions.effect == effect)]
            relationship = aggregate_relationship_view(interactions, config, effect)
            matrices = aggregate_matrices(interactions, config, effect)
            for period in ("P1", "P2", "P3"):
                expected = source[source.period == period][["net", "gross"]].sum()
                assert np.allclose(
                    relationship[relationship.period == period][["net", "gross"]].sum(),
                    expected,
                    rtol=1e-14,
                    atol=0.02,
                )
                assert np.allclose(
                    matrices[matrices.period == period][["net", "gross"]].sum(),
                    expected,
                    rtol=1e-14,
                    atol=0.02,
                )
                assert len(matrices[matrices.period == period]) == len(GROUPS) ** 2
