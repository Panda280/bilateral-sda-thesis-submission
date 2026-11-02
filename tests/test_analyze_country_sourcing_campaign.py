import json
from pathlib import Path

import pandas as pd

from analyze_country_sourcing_campaign import inventory, analyse_product, longitudinal


def _write(root: Path, run: str, effect: str, frame: pd.DataFrame) -> None:
    if effect == "D_C":
        folder, filename, report_name = "source_country", "D_C.source_country_cube.parquet", "source_country_cube_report.json"
        reconciliation = {"passed": True, "failing_index": None}
    else:
        folder, filename, report_name = "yf_source_country", "D_YF.source_country_cube.parquet", "yf_source_country_cube_report.json"
        reconciliation = {"parent": {"passed": True, "failing_index": None}}
    target = root / "offshoring" / run / folder
    target.mkdir(parents=True)
    frame.to_parquet(target / filename, index=False)
    (target / report_name).write_text(json.dumps({"effect": effect, "units": "kg CO2",
        "total": float(frame.Value.sum()), "reconciliation": reconciliation}))


def test_campaign_metrics_preserve_roles_and_rank_noncausally(tmp_path):
    common = {"Emitter_Country": ["E", "E"], "Emitter_Sector": ["s", "s"],
              "Source_Country": ["A", "B"], "Value": [-2.0, 6.0]}
    _write(tmp_path, "2000_2001_2013", "D_C", pd.DataFrame(common | {"Driver_Country": ["A", "A"]}))
    _write(tmp_path, "2000_2001_2013", "D_YF", pd.DataFrame(common | {"Consumer_Country": ["A", "A"]}))
    inv = inventory(tmp_path)
    indicator_rows, share_rows = [], []
    for row in inv.itertuples(index=False):
        indicators, shares = analyse_product(row)
        indicator_rows.extend(indicators); share_rows.extend(shares)
    annual = pd.DataFrame(indicator_rows)
    assert set(annual.focal_role) == {"structural_driver_country", "consumer_country"}
    assert set(annual.offshoring_consistent_evidence) == {"STRONG"}
    period, cases = longitudinal(annual, pd.DataFrame(share_rows))
    assert len(period) == 2 and len(cases) == 2
    assert cases.result_id.is_unique
    assert cases.interpretation_limit.str.contains("not causal proof").all()
