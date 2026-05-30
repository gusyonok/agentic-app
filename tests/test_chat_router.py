"""Router heuristics for chat UI."""

from app.chat_router import (
    ChatIntent,
    classify_chat_intent,
    detect_chat_language,
    infer_followup_focus,
    infer_narrative_scope,
    is_trivial_chitchat,
    resolve_chat_language,
    response_no_triangle_loaded,
    should_run_full_orchestration,
)
from app.text_formatting import format_actuarial_notation_html


def test_first_substantive_message_triggers_run():
    assert should_run_full_orchestration(
        "Оціни резерв для портфеля",
        has_valid_triangle=True,
        has_previous_result=False,
    )


def test_followup_question_skips_run():
    assert not should_run_full_orchestration(
        "Чому IBNR для 2024 такий великий?",
        has_valid_triangle=True,
        has_previous_result=True,
    )


def test_recalc_forces_run_even_with_result():
    assert should_run_full_orchestration(
        "Перерахуй з bootstrap",
        has_valid_triangle=True,
        has_previous_result=True,
    )


def test_trivial_message_never_triggers_run():
    assert not should_run_full_orchestration(
        "Привіт!",
        has_valid_triangle=True,
        has_previous_result=False,
    )


def test_thanks_with_prior_result_is_trivial_not_run():
    assert not should_run_full_orchestration(
        "добре дякую!",
        has_valid_triangle=True,
        has_previous_result=True,
    )
    assert (
        classify_chat_intent(
            "добре дякую!",
            has_valid_triangle=True,
            has_previous_result=True,
        )
        == ChatIntent.TRIVIAL
    )


def test_vague_message_with_prior_result_is_followup_not_run():
    assert (
        classify_chat_intent(
            "цікаво",
            has_valid_triangle=True,
            has_previous_result=True,
        )
        == ChatIntent.FOLLOWUP
    )
    assert not should_run_full_orchestration(
        "цікаво",
        has_valid_triangle=True,
        has_previous_result=True,
    )


def test_translate_is_followup_not_run():
    assert (
        classify_chat_intent(
            "переклади попередню відповідь англійською",
            has_valid_triangle=True,
            has_previous_result=True,
        )
        == ChatIntent.TRANSLATE
    )
    assert infer_followup_focus("переклади англійською") == "translate"


def test_is_trivial_chitchat():
    assert is_trivial_chitchat("Дякую")
    assert is_trivial_chitchat("добре дякую!")
    assert is_trivial_chitchat("ok")
    assert not is_trivial_chitchat("Оціни резерв і поясни VaR")


def test_narrative_scope():
    assert infer_narrative_scope("який VaR 99.5%?") == "focused"
    assert infer_narrative_scope("оціни загальний ризик портфеля") == "risk_summary"


def test_canned_response_matches_user_language():
    assert detect_chat_language("Оціни резерв") == "uk"
    assert detect_chat_language("Estimate the reserve") == "en"
    assert resolve_chat_language("hello", "uk") == "uk"
    en_msg = response_no_triangle_loaded("Estimate the reserve").lower()
    assert "attach" in en_msg or "sidebar" in en_msg or "upload" in en_msg
    uk_msg = response_no_triangle_loaded("Оціни резерв").lower()
    assert "оберіть" in uk_msg or "завантажте" in uk_msg


def test_actuarial_notation_renders_with_subscripts():
    text = format_actuarial_notation_html("P_def and VaR 99.5% and TVaR 99.5%")
    assert "P<sub>def</sub>" in text
    assert "VaR<sub>99.5%</sub>" in text
    assert "TVaR<sub>99.5%</sub>" in text


def test_actuarial_notation_strips_orphan_asterisks():
    broken = (
        "approximately **4,090,203.60***. This value indicates that in extreme scenarios, "
        "our payouts could be higher. The *Tail Value at Risk (TVaR)* calculated at the "
        "same confidence level stands at **4,141,093.01**."
    )
    out = format_actuarial_notation_html(broken)
    assert "This value indicates" in out
    assert "Tail Value at Risk" in out
    assert "<strong>4,090,203.60</strong>" in out
    assert "valueindicatesthatinextreme" not in out.lower()
