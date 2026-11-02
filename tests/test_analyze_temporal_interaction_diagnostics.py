import pandas as pd

import analyze_temporal_interaction_diagnostics as analysis


def _row(effect, resolution, year, level, relationship, source, destination, gross, net):
    return {
        "effect": effect,
        "resolution": resolution,
        "year_previous": year,
        "year_current": year + 1,
        "dimension_level": level,
        "relationship": relationship,
        "source_group": source,
        "destination_group": destination,
        "net_mt": net,
        "positive_mt": (gross + net) / 2,
        "negative_mt": (net - gross) / 2,
        "gross_mt": gross,
        "cancellation_ratio": 1 - abs(net) / gross,
        "cell_count": 1,
    }


def _complete_metrics():
    rows = []
    for effect in analysis.EFFECTS:
        for resolution in analysis.RESOLUTIONS:
            for year in range(2000, 2014):
                net = 20.0 if year % 2 == 0 else -20.0
                rows.append(_row(effect, resolution, year, "overall", "ALL", "ALL", "ALL", 100.0, net))
                relationship_shares = (0.5, 0.5) if year == 2000 else (0.75, 0.25)
                for category, share in zip(("Domestic", "Cross-border"), relationship_shares):
                    rows.append(_row(
                        effect, resolution, year, "relationship", category, "ALL", "ALL",
                        100 * share, net * share,
                    ))
                for level, field in (("source_group", "source"), ("destination_group", "destination")):
                    for category, share in zip(("Red", "Non-Red", "Non-Participant"), (0.5, 0.3, 0.2)):
                        rows.append(_row(
                            effect, resolution, year, level, "ALL",
                            category if field == "source" else "ALL",
                            category if field == "destination" else "ALL",
                            100 * share, net * share,
                        ))
    return pd.DataFrame(rows)


def test_reversal_materiality_and_composition_calculations():
    frame = _complete_metrics()
    annual = analysis.build_annual_overall(frame)
    summary, events = analysis.build_reversals(annual, (0.0, 0.25))
    exact = summary.loc[
        (summary["effect"] == "D_C")
        & (summary["resolution"] == "2013")
        & (summary["materiality_threshold"] == 0.0)
    ].iloc[0]
    material = summary.loc[
        (summary["effect"] == "D_C")
        & (summary["resolution"] == "2013")
        & (summary["materiality_threshold"] == 0.25)
    ].iloc[0]
    assert exact["adjacent_sign_reversals"] == 13
    assert material["material_transitions"] == 0
    assert len(events.loc[events["materiality_threshold"] == 0.0]) == 13 * 6

    composition, shifts = analysis.build_composition(frame)
    first_relationship_shift = shifts.loc[
        (shifts["effect"] == "D_C")
        & (shifts["resolution"] == "2013")
        & (shifts["dimension"] == "relationship")
    ].sort_values("current_start_year").iloc[0]
    assert first_relationship_shift["gross_tvd"] == 0.25
    assert set(composition["dimension"]) == set(analysis.PARTITIONS)


def test_cli_writes_validated_compact_outputs(tmp_path):
    source = tmp_path / "annual_group_metrics.csv"
    _complete_metrics().to_csv(source, index=False)
    output = tmp_path / "diagnostics"

    result = analysis.main([
        "--annual-metrics", str(source),
        "--output-dir", str(output),
    ])

    assert result == 0
    expected = {
        "annual_overall_metrics.csv",
        "sign_reversal_summary.csv",
        "sign_reversal_events.csv",
        "materiality_summary.csv",
        "composition_metrics.csv",
        "composition_shifts.csv",
        "anomaly_candidates.csv",
        "anomaly_sensitivity.csv",
        "analysis_manifest.json",
    }
    assert expected == {path.name for path in output.iterdir()}
    validated = analysis.load_and_validate(source)
    assert len(validated.loc[validated["dimension_level"] == "overall"]) == 84


def test_zero_mad_scores_are_finite_and_flagged():
    scores, median, mad, zero_mad = analysis._robust_scores(pd.Series([1.0, 1.0, 2.0]))
    assert zero_mad and median == 1.0 and mad == 0.0
    assert scores.eq(0.0).all()
