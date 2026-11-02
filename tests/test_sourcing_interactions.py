import csv
import json

import numpy as np

from sda.attribution import SOURCE_ORIGIN_GROUPS
from sda.attribution.structural import attribute_structural_driver
from sda.effects.structural import two_factor_structural_components
from sda.final_demand_sourcing import YFSourceGroupSectorCube
from sda.model import LabelSet
from sda.output.reports import (
    export_source_group_sector_interaction,
    export_yf_group_product_interaction,
)
from sda.sourcing_interactions import (
    SOURCING_RELATIONSHIPS,
    build_source_group_sector_interaction,
    build_yf_group_product_interaction,
)
from sda.validation import assert_scaled_additive


COUNTRY_GROUPS = {"A": "Red", "B": "Non-Red"}


def _dc_interaction():
    """Return a transparent two-country, one-sector sourcing example."""

    return build_source_group_sector_interaction(
        C0=np.zeros((2, 2)),
        C1=np.array([[1.0, 2.0], [3.0, 4.0]]),
        A_x0=np.ones((2, 2)),
        A_x1=np.ones((2, 2)),
        v0=np.ones(2),
        delta_v=np.zeros(2),
        L0=np.eye(2),
        L1=np.eye(2),
        y0=np.ones(2),
        delta_y=np.zeros(2),
        source_countries=("A", "B"),
        source_sectors=("s0", "s0"),
        destination_countries=("A", "B"),
        destination_sectors=("s0", "s0"),
        country_group_by_country=COUNTRY_GROUPS,
    )


def _yf_cube():
    """Return a valid cube with individually traceable relationship cells."""

    values = np.zeros((2, 2, 4, 2), dtype=np.float64)
    # Consumer A is Red. Consumer B is Non-Red. Only the first emitter is
    # populated so emitter aggregation has an unambiguous reference.
    values[0, 0, 0, 0] = 10.0  # domestic A, product p0
    values[0, 0, 1, 1] = 3.0   # cross-border Red -> Red, product p1
    values[0, 0, 2, 0] = 4.0   # cross-border Non-Red -> Red, product p0
    values[0, 0, 3, 1] = 5.0   # cross-border Non-Participant -> Red, p1
    values[0, 1, 0, 1] = 20.0  # domestic B, product p1
    values[0, 1, 1, 0] = 6.0   # cross-border Red -> Non-Red, product p0
    values[0, 1, 2, 1] = 7.0   # cross-border Non-Red -> Non-Red, product p1
    values[0, 1, 3, 0] = 8.0   # cross-border Non-Participant -> Non-Red, p0
    passed = assert_scaled_additive([np.zeros(1)], np.zeros(1))
    return YFSourceGroupSectorCube(
        values,
        LabelSet("emitter_country_sector", (("A", "s0"), ("B", "s0"))),
        LabelSet("consumer_country", ("A", "B")),
        LabelSet("source_group", SOURCE_ORIGIN_GROUPS),
        LabelSet("source_sector", ("p0", "p1")),
        passed,
        {group: passed for group in SOURCE_ORIGIN_GROUPS},
    )


def test_dc_interaction_schema_additivity_and_relationship_semantics():
    interaction = _dc_interaction()

    assert interaction.baseline.shape == (2, 3, 1, 3, 1)
    assert interaction.relationship.labels == SOURCING_RELATIONSHIPS
    assert interaction.relationship.role == "sourcing_relationship"
    assert interaction.source_group.labels == ("Red", "Non-Red", "Non-Participant")
    assert interaction.source_group.role == "source_country_group"
    assert interaction.destination_group.role == "destination_country_group"
    assert interaction.source_sector.role == "source_sector"
    assert interaction.destination_sector.role == "destination_sector"
    assert interaction.reconciliation.passed

    domestic, cross_border = 0, 1
    red, non_red = 0, 1
    assert interaction.baseline[domestic, red, 0, red, 0] == 1.0
    assert interaction.baseline[domestic, non_red, 0, non_red, 0] == 4.0
    assert interaction.baseline[cross_border, red, 0, non_red, 0] == 2.0
    assert interaction.baseline[cross_border, non_red, 0, red, 0] == 3.0
    # Domestic is a bilateral relationship, not an absolute fourth group.
    assert interaction.baseline[domestic, red, 0, non_red, 0] == 0.0
    assert interaction.baseline[domestic, non_red, 0, red, 0] == 0.0

    expected_by_destination = np.array([[4.0], [6.0], [0.0]])
    np.testing.assert_allclose(
        interaction.baseline.sum(axis=(0, 1, 2)), expected_by_destination
    )
    component = two_factor_structural_components(
        np.zeros((2, 2)), np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.ones((2, 2)), np.ones((2, 2)),
    ).C
    explicit_parent = attribute_structural_driver(
        component,
        v0=np.ones(2), delta_v=np.zeros(2),
        L0=np.eye(2), L1=np.eye(2),
        y0=np.ones(2), delta_y=np.zeros(2),
    )
    np.testing.assert_allclose(interaction.baseline.sum(), explicit_parent.sum())
    assert interaction.baseline.sum() == 10.0


def test_yf_interaction_is_lossless_aggregation_with_one_product_axis():
    cube = _yf_cube()
    interaction = build_yf_group_product_interaction(
        cube, consumer_group_by_country=COUNTRY_GROUPS
    )

    assert interaction.baseline.shape == (2, 3, 3, 2)
    assert interaction.relationship.labels == SOURCING_RELATIONSHIPS
    assert interaction.source_group.role == "source_country_group"
    assert interaction.destination_group.role == "destination_country_group"
    assert interaction.product_sector.role == "demanded_sector"
    assert not hasattr(interaction, "destination_sector")
    assert interaction.reconciliation.passed

    domestic, cross_border = 0, 1
    red, non_red, non_participant = 0, 1, 2
    assert interaction.baseline[domestic, red, red, 0] == 10.0
    assert interaction.baseline[domestic, non_red, non_red, 1] == 20.0
    assert interaction.baseline[cross_border, red, red, 1] == 3.0
    assert interaction.baseline[cross_border, non_red, red, 0] == 4.0
    assert interaction.baseline[cross_border, non_participant, red, 1] == 5.0
    assert interaction.baseline[cross_border, red, non_red, 0] == 6.0
    assert interaction.baseline[cross_border, non_red, non_red, 1] == 7.0
    assert interaction.baseline[cross_border, non_participant, non_red, 0] == 8.0
    np.testing.assert_allclose(interaction.baseline.sum(), cube.baseline.sum())


def test_interaction_exports_have_unambiguous_long_format_schemas(tmp_path):
    dc = _dc_interaction()
    yf = build_yf_group_product_interaction(
        _yf_cube(), consumer_group_by_country=COUNTRY_GROUPS
    )
    dc_root = export_source_group_sector_interaction(
        dc, tmp_path / "dc", run_id="tiny", units="kg CO2", file_format="csv"
    )
    yf_root = export_yf_group_product_interaction(
        yf, tmp_path / "yf", run_id="tiny", units="kg CO2", file_format="csv"
    )

    with (dc_root / "D_C.source_destination_group_sector_interaction.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        dc_rows = list(csv.DictReader(stream))
    with (yf_root / "D_YF.source_destination_group_product_interaction.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        yf_rows = list(csv.DictReader(stream))
    assert tuple(dc_rows[0]) == (
        "Relationship", "Source_Group", "Source_Sector",
        "Destination_Group", "Destination_Sector", "Value",
    )
    assert tuple(yf_rows[0]) == (
        "Relationship", "Source_Group", "Destination_Group", "Product_Sector", "Value",
    )
    assert len(dc_rows) == dc.baseline.size
    assert len(yf_rows) == yf.baseline.size

    dc_report = json.loads(
        (dc_root / "source_group_sector_interaction_report.json").read_text()
    )
    yf_report = json.loads(
        (yf_root / "yf_group_product_interaction_report.json").read_text()
    )
    assert dc_report["effect"] == "D_C"
    assert yf_report["effect"] == "D_YF"
    assert dc_report["axes"]["relationship"] == ["Domestic", "Cross-border"]
    assert dc_report["axes"]["destination_sector"] == ["s0"]
    assert yf_report["axes"]["product_sector"] == ["p0", "p1"]
    assert dc_report["reconciliation"]["failing_index"] is None
    assert yf_report["reconciliation"]["failing_index"] is None
