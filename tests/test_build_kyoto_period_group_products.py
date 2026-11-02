from pathlib import Path

import pandas as pd
import pytest

from scripts.build_kyoto_period_group_products import (
    GROUPS,
    PERIODS,
    _assert_metric_identities,
    build_interactions,
    build_support_inventory,
    period_for_year,
)


def test_periods_are_exactly_the_three_approved_windows():
    assert PERIODS == {"P1": (2001, 2002, 2003, 2004), "P2": (2005, 2006, 2007), "P3": (2008, 2009, 2010, 2011, 2012)}
    assert period_for_year(2000) is None
    assert period_for_year(2004) == "P1"
    assert period_for_year(2007) == "P2"
    assert period_for_year(2012) == "P3"
    assert period_for_year(2013) is None


def test_signed_pool_identities_are_enforced():
    good = pd.DataFrame({"net": [1.0], "positive": [3.0], "negative": [-2.0], "gross": [5.0]})
    _assert_metric_identities(good, "test")
    bad = good.assign(gross=4.0)
    with pytest.raises(ValueError, match="gross"):
        _assert_metric_identities(bad, "test")


def _write_interaction_campaign(root: Path) -> None:
    rows = []
    for source in GROUPS:
        for destination in GROUPS:
            rows.append({"Relationship": "Cross-border", "Source_Group": source, "Destination_Group": destination, "Value": 2.0})
            rows.append({"Relationship": "Domestic", "Source_Group": source, "Destination_Group": destination,
                         "Value": 1.0 if source == destination else 0.0})
    frame = pd.DataFrame(rows)
    for period, years in PERIODS.items():
        for year in years:
            for resolution in ("2013", "2016", "R3"):
                run = root / "offshoring" / f"{year-1}_{year}_{resolution}"
                dc = run / "source_group_sector_interaction"
                yf = run / "yf_group_product_interaction"
                dc.mkdir(parents=True)
                yf.mkdir(parents=True)
                frame.to_parquet(dc / "D_C.source_destination_group_sector_interaction.parquet", index=False)
                frame.to_parquet(yf / "D_YF.source_destination_group_product_interaction.parquet", index=False)


def test_interactions_preserve_joint_key_and_omit_domestic_cross_group(tmp_path):
    _write_interaction_campaign(tmp_path)
    result, gates = build_interactions(tmp_path)
    assert len(result) == 3 * 3 * 2 * 12
    assert not ((result.relationship == "Domestic") & (result.source_group != result.destination_group)).any()
    counts = result.groupby(["period", "resolution", "effect"]).size()
    assert counts.eq(12).all()
    p1 = result[(result.period == "P1") & (result.resolution == "R3") & (result.effect == "D_C")]
    assert p1.net.sum() == pytest.approx(84.0)  # four transitions * (9*2 foreign + 3*1 domestic)
    assert gates["omitted_zero_domestic_cross_group_source_rows"] == 3 * 2 * 12 * 6


def test_non_bilateral_effect_roles_are_explicitly_unavailable_not_inferred():
    support = build_support_inventory().set_index(["effect", "role"])
    assert support.loc[("D_C", "structural_driver"), "bilateral_status"] == "available"
    assert support.loc[("D_YF", "consumer"), "bilateral_status"] == "available"
    assert support.loc[("D_v", "emitter"), "bilateral_status"] == "unavailable_not_inferred"
    assert support.loc[("D_C", "emitter"), "bilateral_status"] == "unavailable_not_inferred"
