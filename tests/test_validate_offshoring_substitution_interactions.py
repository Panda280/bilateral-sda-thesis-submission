from itertools import product

import numpy as np
import pandas as pd
import pytest

from validate_offshoring_substitution_interactions import (
    GROUPS,
    PRODUCTS,
    RELATIONSHIPS,
    InteractionAuditError,
    numerical_floor,
    persistence,
    screen_pairs,
    validate_frame,
)


def _dc_frame():
    sectors = ("s0", "s1")
    frame = pd.DataFrame(
        [(*labels, 0.0) for labels in product(
            RELATIONSHIPS, GROUPS, sectors, GROUPS, sectors
        )],
        columns=PRODUCTS["D_C"].columns,
    )

    def assign(relationship, source_group, source_sector,
               destination_group, destination_sector, value):
        frame.loc[
            (frame.Relationship == relationship)
            & (frame.Source_Group == source_group)
            & (frame.Source_Sector == source_sector)
            & (frame.Destination_Group == destination_group)
            & (frame.Destination_Sector == destination_sector),
            "Value",
        ] = value

    assign("Cross-border", "Non-Participant", "s0", "Red", "s0", 10.0)
    assign("Domestic", "Red", "s0", "Red", "s0", -8.0)
    assign("Cross-border", "Non-Participant", "s1", "Red", "s1", 1.0e-5)
    assign("Domestic", "Red", "s1", "Red", "s1", -1.0e-5)
    return frame


def _yf_frame():
    sectors = ("p0", "p1")
    frame = pd.DataFrame(
        [(*labels, 0.0) for labels in product(
            RELATIONSHIPS, GROUPS, GROUPS, sectors
        )],
        columns=PRODUCTS["D_YF"].columns,
    )
    frame.loc[
        (frame.Relationship == "Cross-border")
        & (frame.Source_Group == "Non-Participant")
        & (frame.Destination_Group == "Red")
        & (frame.Product_Sector == "p0"),
        "Value",
    ] = 6.0
    frame.loc[
        (frame.Relationship == "Domestic")
        & (frame.Source_Group == "Red")
        & (frame.Destination_Group == "Red")
        & (frame.Product_Sector == "p0"),
        "Value",
    ] = -4.0
    return frame


def test_dc_preserves_source_and_destination_sector_when_matching():
    frame = _dc_frame()
    assert validate_frame(frame, PRODUCTS["D_C"])["sectors"] == 2
    pairs, summary = screen_pairs(frame, PRODUCTS["D_C"])

    assert PRODUCTS["D_C"].pair_key == (
        "Source_Sector", "Destination_Group", "Destination_Sector"
    )
    assert summary["screens"]["exact"]["count"] == 2
    assert summary["screens"]["tolerance"]["count"] == 1
    assert summary["screens"]["material"]["count"] == 1
    assert summary["screens"]["strong"]["count"] == 1
    assert summary["false_small_sign"]["removed_count"] == 1
    selected = pairs[pairs.strong].iloc[0]
    assert selected.Source_Sector == "s0"
    assert selected.Destination_Sector == "s0"
    assert selected.matched_kg == 8.0
    assert selected.balance == 0.8


def test_dyf_uses_one_product_sector_and_no_destination_sector():
    frame = _yf_frame()
    validate_frame(frame, PRODUCTS["D_YF"])
    pairs, summary = screen_pairs(frame, PRODUCTS["D_YF"])

    assert PRODUCTS["D_YF"].pair_key == ("Destination_Group", "Product_Sector")
    assert "Destination_Sector" not in pairs.columns
    assert summary["screens"]["strong"]["count"] == 1
    assert pairs[pairs.strong].iloc[0].matched_kg == 4.0


def test_domestic_relationship_cannot_cross_absolute_groups():
    frame = _yf_frame()
    frame.loc[
        (frame.Relationship == "Domestic")
        & (frame.Source_Group == "Non-Participant")
        & (frame.Destination_Group == "Red")
        & (frame.Product_Sector == "p1"),
        "Value",
    ] = 1.0
    with pytest.raises(InteractionAuditError, match="crosses absolute groups"):
        validate_frame(frame, PRODUCTS["D_YF"])


def test_tolerance_removes_false_small_sign_and_false_persistence():
    assert numerical_floor(np.array([10.0, -2.0])) > 1.0e-5
    records = []
    for previous in (2000, 2001, 2003):
        records.append({
            "effect": "D_C", "resolution": "2013",
            "previous_year": previous, "current_year": previous + 1,
            "cell_key": ("s0", "Red", "s0"),
            "matched_kg": 1.0, "gross_share": 0.1, "balance": 0.8,
            "exact": True,
            "tolerance": previous != 2001,
            "material": previous != 2001,
            "strong": previous != 2001,
        })

    exact = persistence(records, "exact")
    tolerance = persistence(records, "tolerance")
    assert exact["streak_count"] == 1
    assert exact["max_streak"] == 2
    assert tolerance["streak_count"] == 0
    assert tolerance["max_streak"] == 0
