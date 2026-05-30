"""Heuristics: full reserving run vs conversational follow-up on last results."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

GREETING_ONLY = frozenset(
    {
        "привіт",
        "вітаю",
        "hello",
        "hi",
        "hey",
        "добрий день",
        "добридень",
        "доброго дня",
        "дякую",
        "thanks",
        "thank",
        "thank you",
        "спасибі",
        "ok",
        "okay",
        "зрозуміло",
        "зрозміла",
        "добре",
        "ясно",
        "ага",
        "угу",
        "good",
        "great",
        "perfect",
        "cool",
    }
)

TRIVIAL_ACK = frozenset(
    {
        "ок",
        "ok",
        "okay",
        "дякую",
        "thanks",
        "thank",
        "you",
        "добре",
        "ясно",
        "зрозуміло",
        "зрозміла",
        "ага",
        "угу",
        "дуже",
        "величезно",
        "щиро",
        "so",
        "much",
        "lot",
    }
)

CHITCHAT_WORDS = GREETING_ONLY | TRIVIAL_ACK

RECALC_TRIGGERS = (
    "оцін",
    "розрах",
    "резерв",
    "chain",
    "bootstrap",
    "симуляц",
    "симуляці",
    "monte",
    "перерах",
    "повтор",
    "знову",
    "ще раз",
    "run ",
    " run",
    "estimate",
    "reserve",
    "simulate",
    "simulation",
    "прогін",
    "прогон",
    "аналізуй",
    "проаналіз",
    "запусти",
    "перезапуст",
    "recalc",
    "rerun",
)

RESERVING_REQUEST_TRIGGERS = RECALC_TRIGGERS + (
    "ризик",
    "risk",
    "небезп",
    "ibnr",
    "дефіцит",
    "deficit",
    "дефолт",
    "default",
    "var",
    "tvar",
    "маржа",
    "margin",
    "капітал",
    "capital",
    "stress",
    "стрес",
)

FOLLOWUP_TRIGGERS = (
    "чому",
    "поясни",
    "поясніть",
    "що означає",
    "що таке",
    "як ",
    "які ",
    "який ",
    "яка ",
    "яке ",
    "розкрий",
    "детальніше",
    "уточни",
    "чи можна",
    "чи достатн",
    "why ",
    "why?",
    "what ",
    "what?",
    "how ",
    "explain",
    "interpret",
    "meaning",
    "clarify",
)

TRANSLATE_TRIGGERS = (
    "переклад",
    "перекладі",
    "translate",
    "translation",
    "англійськ",
    "english",
    "ukrainian",
    "українськ",
    "in english",
    "in ukrainian",
    "на англ",
    "на укр",
    "українською",
    "англійською",
)

METRIC_FOCUS_KEYWORDS = (
    "p_def",
    "p def",
    "p_default",
    "p default",
    "дефіцит",
    "deficit",
    "дефолт",
    "default",
    "var",
    "tvar",
    "маржа",
    "margin",
    "ibnr",
    "резерв",
    "reserve",
    "mack",
    "bootstrap",
    "monte",
    "simulation",
    "симуляц",
    "стресс",
    "stress",
    "99.5",
    "99,5",
    "хвіст",
    "tail",
    "ldf",
    "cdf",
    "фактор",
    "factor",
)


class ChatIntent(str, Enum):
    NO_TRIANGLE = "no_triangle"
    TRIVIAL = "trivial"
    TRANSLATE = "translate"
    FULL_RUN = "full_run"
    FOLLOWUP = "followup"
    WAIT_FOR_RUN = "wait_for_run"


def _norm(s: str) -> str:
    return s.strip().lower()


def detect_chat_language(user_text: str) -> str:
    cyrillic = sum(1 for ch in user_text if "\u0400" <= ch <= "\u04ff")
    latin = sum(1 for ch in user_text if "a" <= ch.lower() <= "z")
    if cyrillic > 0:
        return "uk"
    if latin > 0:
        return "en"
    return "uk"


def resolve_chat_language(user_text: str, preference: str = "auto") -> str:
    """Apply sidebar language preference when not auto."""
    pref = (preference or "auto").strip().lower()
    if pref in ("uk", "en"):
        return pref
    return detect_chat_language(user_text)


def _only_greeting_or_trivial(text: str) -> bool:
    t = _norm(text)
    if not t:
        return True
    t = re.sub(r"[!?.…,:;]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    if t in CHITCHAT_WORDS:
        return True
    if re.fullmatch(r"(добре\s+)?(дуже\s+)?дякую(\s+(вам|тебе))?", t):
        return True
    if re.fullmatch(r"(thanks?(\s+you)?(\s+so\s+much)?|thank you)", t):
        return True
    parts = [p for p in t.split() if p]
    if parts and len(parts) <= 4 and all(p in CHITCHAT_WORDS for p in parts):
        return True
    return False


def _wants_recalc(text: str) -> bool:
    low = _norm(text)
    return any(k in low for k in RECALC_TRIGGERS)


def is_reserving_request(text: str) -> bool:
    low = _norm(text)
    return any(k in low for k in RESERVING_REQUEST_TRIGGERS)


def _looks_like_followup_question(text: str) -> bool:
    low = _norm(text)
    if "?" in text:
        return True
    return any(k in low for k in FOLLOWUP_TRIGGERS)


def is_translate_request(text: str) -> bool:
    low = _norm(text)
    return any(k in low for k in TRANSLATE_TRIGGERS)


def is_metric_focused_question(text: str) -> bool:
    low = _norm(text)
    if not any(k in low for k in METRIC_FOCUS_KEYWORDS):
        return False
    broad = ("загал", "overall", "повний", "full report", "усі метрик", "all metric", "everything")
    if any(b in low for b in broad):
        return False
    return True


def is_general_risk_question(text: str) -> bool:
    low = _norm(text)
    return any(w in low for w in ("ризик", "risk", "небезп", "danger", "безпеч"))


def infer_narrative_scope(user_text: str) -> str:
    """Hint for LLM narrative length: focused | risk_summary | standard."""
    if is_metric_focused_question(user_text) and not is_general_risk_question(user_text):
        return "focused"
    if is_general_risk_question(user_text):
        return "risk_summary"
    narrow = ("лише", "тільки", "only", "just", "one metric", "одну метрик", "одна метрик")
    if any(n in _norm(user_text) for n in narrow):
        return "focused"
    return "standard"


def infer_followup_focus(user_text: str) -> str:
    """Hint for follow-up replies: translate | metric | risk_summary | general."""
    if is_translate_request(user_text):
        return "translate"
    if is_metric_focused_question(user_text):
        return "metric"
    if is_general_risk_question(user_text):
        return "risk_summary"
    return "general"


def classify_chat_intent(
    user_text: str,
    *,
    has_valid_triangle: bool,
    has_previous_result: bool,
) -> ChatIntent:
    if not has_valid_triangle:
        return ChatIntent.NO_TRIANGLE
    if _only_greeting_or_trivial(user_text):
        return ChatIntent.TRIVIAL
    if has_previous_result and is_translate_request(user_text):
        return ChatIntent.TRANSLATE
    if _wants_recalc(user_text):
        return ChatIntent.FULL_RUN
    if not has_previous_result:
        return ChatIntent.FULL_RUN if is_reserving_request(user_text) else ChatIntent.WAIT_FOR_RUN
    if _looks_like_followup_question(user_text) or is_metric_focused_question(user_text):
        return ChatIntent.FOLLOWUP
    if is_reserving_request(user_text):
        return ChatIntent.FULL_RUN
    return ChatIntent.FOLLOWUP


def should_run_full_orchestration(
    user_text: str,
    *,
    has_valid_triangle: bool,
    has_previous_result: bool,
) -> bool:
    """True → run full OrchestratorAgent pipeline."""
    return (
        classify_chat_intent(
            user_text,
            has_valid_triangle=has_valid_triangle,
            has_previous_result=has_previous_result,
        )
        == ChatIntent.FULL_RUN
    )


def response_thanks(user_text: str = "", *, language_pref: str = "auto") -> str:
    if resolve_chat_language(user_text, language_pref) == "en":
        return (
            "You're welcome! If you want details on a **specific number**, just ask. "
            "For a **new calculation**, write something like **estimate the reserve**."
        )
    return (
        "Будь ласка! Якщо потрібно пояснення по **конкретній метриці** — напишіть, про яку саме. "
        "Для **нового повного прогону** — наприклад, **«перерахуй резерв»**."
    )


def response_no_triangle_loaded(user_text: str = "", *, language_pref: str = "auto") -> str:
    if resolve_chat_language(user_text, language_pref) == "en":
        return (
            "Please upload a **CSV or Excel** file in the sidebar. "
            "When the data is ready, ask: **Estimate the reserve** or **Explain the risks**."
        )
    return (
        "Щоб я міг оцінити резерв, **завантажте файл** (CSV або Excel) у боковій панелі. "
        "Коли трикутник буде валідний, напишіть у чат, наприклад: «Оціни резерв для цього портфеля»."
    )


def response_wait_for_run(user_text: str = "", *, language_pref: str = "auto") -> str:
    if resolve_chat_language(user_text, language_pref) == "en":
        return (
            "Your file is loaded. Ask for a calculation, for example: "
            "**Estimate the reserve** or **Run chain-ladder and simulation**."
        )
    return (
        "Трикутник уже завантажено. Напишіть запит на розрахунок, наприклад: "
        "**«Оціни резерв»**, **«Запусти метод ланцюгової драбини і бутстреп-симуляцію»** — тоді я виконаю повний прогін агентів."
    )


def response_trivial_without_run(user_text: str = "", *, language_pref: str = "auto") -> str:
    if resolve_chat_language(user_text, language_pref) == "en":
        return (
            "If you need numbers, ask directly: **Estimate the reserve**. "
            "If you already ran a calculation, you can ask a follow-up question about the results."
        )
    return (
        "Якщо потрібен розрахунок — опишіть запит актуарною мовою або натисніть сенс «оціни резерв». "
        "Якщо вже був прогін — можете поставити уточнювальне питання про цифри чи припущення."
    )


def compact_context_for_followup(result_dict: dict[str, Any], max_chars: int = 12000) -> str:
    """Serialize last RunResult (dict) into a bounded JSON block for LLM context."""
    tables = result_dict.get("tables") or {}
    slim: dict[str, Any] = {
        "run_id": result_dict.get("run_id"),
        "narrative": (result_dict.get("narrative") or "")[:4000],
        "risk_metrics": tables.get("risk_metrics"),
        "ultimate_ibnr": (tables.get("ultimate_ibnr") or [])[:30],
        "ldf": tables.get("ldf"),
        "cdf": tables.get("cdf"),
        "validation": tables.get("validation"),
    }
    raw = json.dumps(slim, ensure_ascii=False, indent=2)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 20] + "\n…(truncated)…"


def is_trivial_chitchat(text: str) -> bool:
    """True for greetings / thanks only — no reserving work implied."""
    return _only_greeting_or_trivial(text)
