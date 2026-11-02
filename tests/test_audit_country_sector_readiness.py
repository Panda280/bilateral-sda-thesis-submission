import json

import pandas as pd

from audit_country_sector_readiness import audit, main


def test_missing_authoritative_inputs_are_explicit_blockers(tmp_path):
    result = audit(tmp_path, None, None, None)
    assert result["status"] == "blocked"
    assert result["inventory_count"] == 0
    assert any("group registry" in item for item in result["blockers"])
    assert any("84 source-country products" in item for item in result["blockers"])
    assert result["provisional_due_to_upstream_validation"] is True
    assert result["analysis_id"] == "country-sector-readiness-v1"
    assert result["validation"]["thesis_status"] == "ineligible"


def test_rejects_invalid_groups_and_ambiguous_concordances(tmp_path):
    groups = tmp_path / "groups.csv"
    countries = tmp_path / "countries.csv"
    sectors = tmp_path / "sectors.csv"
    pd.DataFrame({"canonical_country": ["A", "A"], "analysis_group": ["Red", "Other"]}).to_csv(groups, index=False)
    pd.DataFrame({"resolution": ["R3", "R3"], "observed_country": ["A", "A"], "canonical_country": ["A", "B"]}).to_csv(countries, index=False)
    pd.DataFrame({"resolution": ["R3", "R3"], "observed_sector": ["S", "S"], "canonical_sector": ["S1", "S2"]}).to_csv(sectors, index=False)
    result = audit(tmp_path, groups, countries, sectors)
    assert result["status"] == "blocked"
    assert any("duplicate canonical_country" in item for item in result["blockers"])
    assert any("invalid analysis_group" in item for item in result["blockers"])
    assert sum("ambiguous duplicate" in item for item in result["blockers"]) == 2
    assert result["mapping_inputs"]["groups"]["sha256"]
