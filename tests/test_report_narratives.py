from sda.output import (
    BANNED_NARRATIVE_PHRASES,
    REQUIRED_NARRATIVE_PHRASES,
    aggregate_to_group_sector,
    lint_narrative_terminology,
    render_effect_narrative,
    render_group_narratives,
)

from _group_report_fixtures import GROUPS, build_synthetic_result


def test_lint_flags_every_banned_causal_phrase():
    for phrase in BANNED_NARRATIVE_PHRASES:
        text = f"This effect was {phrase} a policy change."
        assert lint_narrative_terminology(text) == (phrase,)


def test_lint_is_clean_for_approved_terminology():
    text = (
        "This total is attributed to structural change, associated with "
        "sourcing shifts, and traced to changes in the driver country's "
        "classification."
    )
    assert lint_narrative_terminology(text) == ()


def test_generated_group_narratives_use_only_approved_terminology_and_cover_every_effect():
    result = build_synthetic_result()
    text = render_group_narratives(result, GROUPS, run_id="narrative-run", units="kg CO2e")

    violations = lint_narrative_terminology(text)
    assert violations == (), f"banned causal language found in generated narrative: {violations}"
    for phrase in REQUIRED_NARRATIVE_PHRASES:
        assert phrase in text
    for name in ("D_v", "D_C", "D_A_x", "D_T_coef", "D_w_coef", "D_YF", "D_YB", "D_y"):
        assert name in text


def test_narrative_states_run_id_and_validation_status():
    result = build_synthetic_result()
    text = render_group_narratives(result, GROUPS, run_id="narrative-run", units="kg CO2e")

    assert "narrative-run" in text
    assert "PASSED" in text
    assert result.validation.passed  # sanity: the synthetic fixture is actually valid


def test_source_split_effect_narrative_names_every_category_without_banned_language():
    result = build_synthetic_result()
    grouped = aggregate_to_group_sector(result.effects["D_C"], GROUPS)
    text = render_effect_narrative("D_C", grouped, run_id="r", units="kg CO2e")

    for category in ("Domestic", "Red", "Non-Red", "Non-Participant"):
        assert category in text
    assert lint_narrative_terminology(text) == ()
