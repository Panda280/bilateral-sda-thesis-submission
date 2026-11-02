from pathlib import Path

import pandas as pd

from scripts.analyze_2008_2009_divergence import build


def test_real_registered_episode(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parent / "fixtures" / "thesis" / "annual_hierarchy.csv"
    manifest = build(source, tmp_path)
    episode = pd.read_csv(tmp_path / "episode_effects.csv")
    gaps = pd.read_csv(tmp_path / "resolution_effect_gaps.csv")
    assert len(episode) == 6
    assert len(gaps) == 66
    assert set(episode["year_current"]) == {2008, 2009}
    assert episode["residual_relative_to_component_absolute_total"].max() < 2e-6
    assert gaps["gap_kg_co2"].notna().all()
    assert manifest["status"] == "accepted_provisional"
    assert manifest["validation"]["reconstruction_passed"]
