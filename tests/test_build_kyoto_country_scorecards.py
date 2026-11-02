import pandas as pd

import build_kyoto_country_scorecards as scorecards


def test_legacy_mapping_and_mechanism_classification(tmp_path):
    groups_path = tmp_path / "groups.csv"
    countries_path = tmp_path / "countries.csv"
    pd.DataFrame([
        {"country_code": "AUT", "analysis_group": "Red"},
        {"country_code": "ROW", "analysis_group": "Non-Participant"},
    ]).to_csv(groups_path, index=False)
    pd.DataFrame([
        {"EXIO": "AT", "WIOT2016": "AUT", "WIOT2013": "AUT"},
        {"EXIO": "WA", "WIOT2016": "ROW", "WIOT2013": "ROW"},
        {"EXIO": "WL", "WIOT2016": "ROW", "WIOT2013": "ROW"},
    ]).to_csv(countries_path, index=False)
    groups, countries = scorecards.load_mappings(groups_path, countries_path)
    assert set(groups.canonical_country) == {"AUT", "ROW"}
    assert set(countries.resolution) == {"2013", "2016", "R3"}

    summary = pd.DataFrame([
        {"resolution": "2016", "canonical_country": "AUT", "units": "kg CO2", "role": "source",
         "effect": "D_C", "cumulative_net": -10, "cumulative_gross": 14},
        {"resolution": "2016", "canonical_country": "AUT", "units": "kg CO2", "role": "source",
         "effect": "D_YF", "cumulative_net": 2, "cumulative_gross": 4},
    ])
    result = scorecards.mechanism_profiles(summary).iloc[0]
    assert result.descriptive_mechanism_class == "intermediate_sourcing_reduction_only"
    assert result.dominant_source_subeffect_by_absolute_net == "D_C"


def test_summary_counts_signs_reversals_and_cancellation():
    frame = pd.DataFrame([
        {"year_previous": 2000, "year_current": 2001, "effect": "D_C", "resolution": "2016",
         "role": "source", "canonical_country": "AUT", "units": "kg CO2", "net": 3, "gross": 5},
        {"year_previous": 2001, "year_current": 2002, "effect": "D_C", "resolution": "2016",
         "role": "source", "canonical_country": "AUT", "units": "kg CO2", "net": -1, "gross": 3},
    ])
    row = scorecards.summarize(frame, "canonical_country").iloc[0]
    assert row.cumulative_net == 2
    assert row.cumulative_gross == 8
    assert row.sign_reversals == 1
    assert row.cancellation_ratio == 0.75
