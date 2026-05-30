"""Narrative hints (plateau bootstrap / non-monotonic triangle) steer LLM + fallback."""

from agents.llm_client import (
    _narrative_override_block,
    build_chief_actuary_fallback_narrative,
    detect_response_language,
)


def test_override_block_includes_priority_tail():
    calc = {"narrative_hints": {"plateau_bootstrap_adequacy": True, "triangle_non_monotonic": False}}
    block = _narrative_override_block(calc)
    assert "ПЕРЕВІЗНИЙ БЛОК" in block
    assert "Заборонено" in block


def test_adequacy_fallback_for_plateau_profile():
    calc = {
        "total_reserve": 2962.54,
        "risk_metrics": {
            "risk_analysis_skipped": False,
            "base_reserve": 2962.54,
            "p_def": 0.4896,
            "rm_required_005": 119.89,
            "var_995": 3082.43,
            "var_95": 2800.0,
            "tvar_995": 3100.0,
            "heavy_tail": False,
            "capital_surplus_regime": False,
            "low_p_def_extreme_tail_warning": False,
            "analysis_basis": "gross_before_reinsurance",
        },
        "narrative_hints": {"plateau_bootstrap_adequacy": True, "triangle_non_monotonic": False},
    }
    text = build_chief_actuary_fallback_narrative(calc)
    assert "bootstrap" in text.lower() or "адекват" in text.lower()
    assert "планов" in text.lower()


def test_adequacy_fallback_when_only_non_monotonic():
    calc = {
        "total_reserve": 5000.0,
        "risk_metrics": {
            "risk_analysis_skipped": False,
            "base_reserve": 5000.0,
            "p_def": 0.15,
            "rm_required_005": 800.0,
            "var_995": 5800.0,
            "var_95": 5200.0,
            "tvar_995": 5900.0,
            "heavy_tail": False,
            "capital_surplus_regime": False,
            "low_p_def_extreme_tail_warning": False,
            "analysis_basis": "gross_before_reinsurance",
        },
        "narrative_hints": {"plateau_bootstrap_adequacy": False, "triangle_non_monotonic": True},
    }
    text = build_chief_actuary_fallback_narrative(calc)
    assert "немонотон" in text.lower()


def test_negative_reserve_uses_surplus_closure_even_with_plateau_hints():
    calc = {
        "total_reserve": -614.0,
        "risk_metrics": {"risk_analysis_skipped": True, "base_reserve": -614.0},
        "narrative_hints": {"plateau_bootstrap_adequacy": True, "triangle_non_monotonic": True},
    }
    text = build_chief_actuary_fallback_narrative(calc)
    assert "надлишков" in text.lower()
    assert "симетричним bootstrap" not in text.lower()


def test_positive_plateau_not_surplus_closure():
    calc = {
        "total_reserve": 2962.54,
        "risk_metrics": {
            "risk_analysis_skipped": False,
            "base_reserve": 2962.54,
            "p_def": 0.4896,
            "rm_required_005": 119.89,
            "var_995": 3082.43,
        },
        "narrative_hints": {"plateau_bootstrap_adequacy": True, "triangle_non_monotonic": False},
    }
    text = build_chief_actuary_fallback_narrative(calc)
    assert "надлишков" not in text.lower()
    assert "bootstrap" in text.lower() or "адекват" in text.lower()


def test_response_language_detection():
    assert detect_response_language("Оціни резерв") == "uk"
    assert detect_response_language("Estimate the reserve") == "en"
