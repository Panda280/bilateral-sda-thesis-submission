import pandas as pd
import pytest

from scripts.analyze_sourcing_group_trends import KG_PER_MT, signed_metrics, trends


def test_signed_metrics_preserve_net_gross_and_exact_mt_conversion():
    frame = pd.DataFrame({"group": ["A", "A", "A"], "Value": [2 * KG_PER_MT, -0.5 * KG_PER_MT, -1.5 * KG_PER_MT]})
    result = signed_metrics(frame, ["group"]).iloc[0]
    assert result.net_mt == 0
    assert result.positive_mt == 2
    assert result.negative_mt == -2
    assert result.gross_mt == 4
    assert result.cancellation_ratio == 1


def test_trends_keep_effects_and_resolutions_separate():
    rows = []
    for effect in ("D_C", "D_YF"):
        for year, value in ((2001, 1.0), (2002, 3.0)):
            rows.append({"effect": effect, "resolution": "2013", "year_previous": year - 1, "year_current": year,
                "dimension_level": "overall", "relationship": "ALL", "source_group": "ALL", "destination_group": "ALL",
                "net_mt": value, "gross_mt": abs(value)})
    result = trends(pd.DataFrame(rows))
    assert len(result) == 2
    assert set(result.effect) == {"D_C", "D_YF"}
    assert result.linear_slope_net_mt_per_year.tolist() == pytest.approx([2.0, 2.0])
