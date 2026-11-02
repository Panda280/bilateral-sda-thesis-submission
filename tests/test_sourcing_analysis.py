import numpy as np
import pytest

from sda.final_demand_sourcing import YFSourceCountryCube
from sda.model import LabelSet
from sda.offshoring import SourceCountryCube
from sda.sourcing_analysis import analyse_source_country_cube
from sda.validation import ScaleAwareAdditivityReport


PASS = ScaleAwareAdditivityReport(0.0, None, 0.0, 0.0)


def labels(role, values):
    return LabelSet(role, tuple(values))


def test_dc_metrics_retain_driver_role_and_signed_contributions():
    # emitter x driver x source; driver A has domestic=-2 and foreign=+6.
    cube = SourceCountryCube(
        np.array([[[-2.0, 6.0], [1.0, 1.0]]]),
        labels("emitter_country_sector", [("E", "s")]),
        labels("structural_driver_country", ["A", "B"]),
        labels("source_country", ["A", "B"]), PASS,
    )
    a, b = analyse_source_country_cube(cube)
    assert a.focal_role == "structural_driver_country"
    assert a.signed_domestic_contribution == -2.0
    assert a.signed_foreign_contribution == 6.0
    assert a.absolute_foreign_sourcing_intensity == pytest.approx(0.75)
    assert a.source_hhi == pytest.approx(0.625)
    assert a.source_entropy == pytest.approx(-(0.25*np.log(.25) + .75*np.log(.75)))
    assert a.effective_source_count == pytest.approx(np.exp(a.source_entropy))
    assert a.cancellation_ratio == pytest.approx(0.5)
    assert a.evidence == "STRONG"
    assert b.evidence == "NO"


def test_dyf_metrics_retain_consumer_after_sourcing_cube_sector_collapse():
    # emitter x consumer x source; demanded sector was collapsed by the builder.
    values = np.array([[[-4.0, 4.0]]])
    cube = YFSourceCountryCube(
        values, labels("emitter_country_sector", [("E", "s")]),
        labels("consumer_country", ["A"]), labels("source_country", ["A", "B"]), PASS,
    )
    row, = analyse_source_country_cube(cube)
    assert row.effect_id == "D_YF"
    assert row.focal_role == "consumer_country"
    assert row.signed_domestic_contribution == -4.0
    assert row.signed_foreign_contribution == 4.0
    assert row.signed_total == 0.0
    assert row.source_shares[0].signed_share_of_net is None
    assert row.cancellation_ratio == 1.0
    assert row.evidence == "MODERATE"


def test_zero_cube_has_explicit_undefined_ratios():
    cube = SourceCountryCube(
        np.zeros((1, 1, 1)), labels("emitter_country_sector", [("E", "s")]),
        labels("structural_driver_country", ["A"]), labels("source_country", ["A"]), PASS,
    )
    row, = analyse_source_country_cube(cube)
    assert row.absolute_foreign_sourcing_intensity is None
    assert row.source_hhi is None
    assert row.source_entropy is None
    assert row.effective_source_count is None
    assert row.cancellation_ratio is None
    assert row.evidence == "NO"
