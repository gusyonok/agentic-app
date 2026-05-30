"""LLM helper for Chief Actuary-style narrative generation."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from app.chat_router import infer_narrative_scope


def _money(v: float | None) -> str:
    if v is None:
        return "н/д"
    return f"{float(v):.2f}"


def detect_response_language(user_text: str) -> str:
    """Return 'uk' for Ukrainian-like prompts, 'en' for English-like prompts."""
    cyrillic = sum(1 for ch in user_text if "\u0400" <= ch <= "\u04ff")
    latin = sum(1 for ch in user_text if ("a" <= ch.lower() <= "z"))
    if cyrillic > 0:
        return "uk"
    if latin > 0:
        return "en"
    return "uk"


def _english_b2_style_block(user_text: str) -> str:
    """Extra prompt block when the user expects English at B1–B2 level."""
    if detect_response_language(user_text) != "en":
        return ""
    return (
        "ENGLISH LEVEL (B1–B2) — mandatory for your answer:\n"
        "- Write in clear, simple English. Short sentences. One idea per sentence.\n"
        "- Use common words. Examples: \"enough\" (not \"sufficient\"), \"extra money\" (not \"additional buffer\"), "
        "\"very bad scenarios\" (not \"adverse tail realisations\").\n"
        "- Avoid idioms, rare words, and long nested clauses.\n"
        "- Prefer active voice: \"The model shows…\" not \"It is indicated that…\".\n"
        "- When you use a technical term the first time, add a brief plain explanation.\n"
        "- Plain wording (vary, do not repeat every time):\n"
        "  • base reserve / IBNR → money set aside for future claim payments;\n"
        "  • reserve deficit probability (Monte Carlo) → chance the reserve is not enough (simulation);\n"
        "  • default probability (Mack) → chance future payments will exceed the reserve (Mack model);\n"
        "  • risk margin → extra safety amount on top of the base reserve;\n"
        "  • stress level / VaR<sub>99.5%</sub> → money needed in very bad scenarios (99.5% level).\n"
        "- Sound like a helpful colleague, not a legal document or academic paper.\n"
        "- For a full report: about 2–3 short paragraphs unless the user asks for more.\n\n"
    )


def _language_instruction(user_text: str) -> str:
    lang = detect_response_language(user_text)
    if lang == "en":
        return (
            "LANGUAGE RULE: the user wrote in English — answer in English at B1–B2 level (upper-intermediate). "
            "Grammar should be correct, but keep words and sentences simple. "
            "Use actuarial terms only when needed, and explain them in plain English. "
            "For notation use HTML subscripts: P<sub>def</sub>, VaR<sub>99.5%</sub>, TVaR<sub>99.5%</sub>; do not write P_def."
        )
    return (
        "ПРАВИЛО МОВИ: відповідай українською, бо користувач пише українською. "
        "Уникай англійських слів у прозі. Допускаються лише необхідні позначення, абревіатури "
        "та власні назви методів у міру: IBNR, VaR, TVaR, P<sub>def</sub>, ODP, Solvency II. "
        "Якщо треба показати технічне позначення, використовуй HTML-індекс: "
        "P<sub>def</sub>, VaR<sub>99.5%</sub>, TVaR<sub>99.5%</sub>; не пиши P_def. "
        "Для method names спочатку використовуй український відповідник: «метод ланцюгової драбини», "
        "«бутстреп-симуляція», «точкова оцінка», «договори перестрахування». "
        "Не пиши англійські слова на кшталт best estimate, bootstrap, chain-ladder, treaties, "
        "якщо їх можна природно замінити українською."
    )


def _rm_pct_of_base(rm: float, base: float) -> float:
    if base <= 1e-9:
        return 0.0
    return (rm / base) * 100.0


def _narrative_scope_block(user_prompt: str) -> str:
    scope = infer_narrative_scope(user_prompt)
    if scope == "focused":
        return (
            "\n\nОБСЯГ ВІДПОВІДІ (вузький запит — найвищий пріоритет над «повним звітом» нижче):\n"
            "- Лише те, про що прямо питає користувач. 1–2 короткі абзаци.\n"
            "- **Не** перелічуй усі метрики підряд.\n"
        )
    if scope == "risk_summary":
        return (
            "\n\nОБСЯГ ВІДПОВІДІ (загальний ризик — найвищий пріоритет над «повним звітом» нижче):\n"
            "- Базовий резерв + ймовірність дефіциту (Monte Carlo) + ймовірність дефолту (Mack).\n"
            "- Максимум ще 1–2 ключові показники (наприклад VaR<sub>99.5%</sub> або маржа), якщо доречно.\n"
            "- **Не** виводь повний перелік усіх метрик.\n"
            "- Заверши одним реченням: запропонуй пояснити інші метрики, якщо користувач захоче.\n"
        )
    return (
        "\n\nОБСЯГ ВІДПОВІДІ (стандартний звіт):\n"
        "- Повний звіт можливий, але без зайвого дублювання цифр.\n"
        "- На кінці можеш одним реченням запропонувати деталі по окремих метриках.\n"
    )


def _followup_focus_block(focus: str, last_narrative: str) -> str:
    if focus == "translate":
        return (
            "РЕЖИМ ПЕРЕКЛАДУ (найвищий пріоритет):\n"
            "- Переклади **лише** попередній текст висновку нижче мовою, яку просить користувач.\n"
            "- **Не** додавай нових цифр, метрик, симуляцій, порад чи повторного аналізу.\n"
            "- JSON — лише для точності термінів і чисел.\n\n"
            f"Текст для перекладу:\n{last_narrative}\n\n"
        )
    if focus == "metric":
        return (
            "ФОКУС ВІДПОВІДІ (конкретна метрика):\n"
            "- Поясни **лише** метрику, про яку питають (1–2 короткі абзаци).\n"
            "- **Заборонено** знову виводити повний перелік усіх метрик і дублювати весь попередній звіт.\n\n"
        )
    if focus == "risk_summary":
        return (
            "ФОКУС ВІДПОВІДІ (загальний ризик):\n"
            "- Коротко: базовий резерв, дефіцит (Monte Carlo) і дефолт (Mack) у %.\n"
            "- Максимум ще 1–2 ключові величини. Запропонуй інші метрики, якщо потрібно.\n"
            "- **Не** перераховуй усі метрики підряд.\n\n"
        )
    return (
        "ФОКУС ВІДПОВІДІ:\n"
        "- Відповідай **прямо на питання**, не дублюй увесь попередній звіт.\n"
        "- **Не** перераховуй усі метрики підряд без потреби.\n\n"
    )


def _deficit_risk_label_ua(p_def: float, rm_pct_of_base: float | None = None) -> str:
    """Qualitative label; near 50% P<sub>def</sub> with tiny margin vs base is not critical by itself."""
    if (
        rm_pct_of_base is not None
        and rm_pct_of_base < 15.0
        and 0.40 <= p_def <= 0.60
    ):
        return "Узгоджено з типовим профілем центральної оцінки резерву при вузькому розподілі (низький абсолютний хвіст)"
    if p_def >= 0.5:
        return "Критично високий ризик"
    if p_def >= 0.25:
        return "Підвищений ризик"
    if p_def >= 0.10:
        return "Помірний ризик"
    return "Обмежений (низький до помірного) ризик"


def _method_comparison_paragraph(risk: dict[str, Any]) -> str:
    """Compare Monte Carlo reserve deficit vs Mack default probability."""
    p_deficit = float(risk.get("p_def_bootstrap", risk.get("p_def")) or 0.0)
    p_default = float(risk.get("p_default_mack", risk.get("p_def_mack_analytical")) or 0.0)
    mack_se = float(risk.get("mack_se_total") or 0.0)
    delta_pp = abs(p_deficit - p_default) * 100.0
    if delta_pp < 3.0:
        agreement = (
            "Обидва підходи дають близькі оцінки — це підсилює довіру до загального висновку про ризик."
        )
    elif delta_pp < 10.0:
        agreement = (
            "Розбіжність помірна: Monte Carlo відображає емпіричний розподіл симуляцій, "
            "тоді як Mack — параметричне аналітичне наближення (логнормальний розподіл)."
        )
    else:
        agreement = (
            "Розбіжність суттєва: варто перевірити припущення (хвіст, якість даних, стабільність факторів) "
            "перед тим, як спиратися лише на одну з оцінок."
        )
    return (
        f"**Порівняння методів.** "
        f"Ймовірність **дефіциту резервів** (Monte Carlo / бутстреп): симульований IBNR перевищує базовий резерв — "
        f"**{p_deficit * 100:.2f}%**. "
        f"Ймовірність **дефолту** (Mack, логнормальний розподіл; σ Mack = **{_money(mack_se)}**): "
        f"майбутні виплати перевищать резерв — **{p_default * 100:.2f}%**. {agreement}"
    )

_ZERO_IBNR_CLOSURE = (
    "Портфель за поточними даними виглядає повністю розвинутим: поточна оцінка очікуваних майбутніх виплат "
    "дорівнює нулю або не є додатною — тож додаткових зобов’язань «на доростання» по суті немає. "
    "Розрахунок додаткової подушки безпеки та перевірка «найгірших» сценаріїв у симуляції не застосовуються. "
    "Усі оцінки тут — на брутто-основі (до перестрахування); реальна стійність залежить також від договорів "
    "перестрахування та лімітів власного утримання."
)

_deficit_method_comparison_paragraph = _method_comparison_paragraph

_NEGATIVE_IBNR_SURPLUS_CLOSURE = (
    "За результатами методу ланцюгової драбини **поточна оцінка майбутніх виплат (базовий резерв)** виходить "
    "**від’ємною**: це означає, що відносно модельної «кінцевої» величини зобов’язань **на балансі вже закладено "
    "більше коштів, ніж модель вважає необхідним** — тобто резерви виглядають **надлишковими**. "
    "У цьому режимі **додаткова маржа ризику для рівня 99,5% не застосовується і дорівнює нулю**; **не потрібно "
    "просити додатковий капітал** саме через цю оцінку. "
    "Також **не варто інтерпретувати симуляційні метрики «ймовірності дефіциту»** — для від’ємної точкової оцінки "
    "бутстреп-хвіст не будується, і розмови про «банкрутство» чи «критичний дефіцит резервів» з цих цифр **були б "
    "некоректними**. Натомість логічний акцент — на **можливому поверненні коштів** (перескладення оцінок, суброгація, "
    "закриття справ) та на тому, що **фінансовий стан з точки зору цієї модельної картини виглядає спокійним**. "
    "Усі оцінки — **брутто до перестрахування**; реальна позиція також залежить від договорів перестрахування та утримання."
)


def _adequacy_profile_fallback_narrative(calc: dict[str, Any], hints: dict[str, Any]) -> str:
    """Deterministic narrative: plateau bootstrap and/or non-monotonic cumulative (no LLM deficit spin)."""
    risk = calc.get("risk_metrics", {})
    total = float(calc.get("total_reserve") or 0.0)
    base = float(risk.get("base_reserve", total))
    p_def = float(risk.get("p_def") or 0.0)
    p_def_mack = float(risk.get("p_def_mack_analytical") or 0.0)
    mack_se = float(risk.get("mack_se_total") or 0.0)
    rm = float(risk.get("rm_required_005") or 0.0)
    var995 = float(risk.get("var_995") or 0.0)
    suffix = os.getenv("REPORT_CURRENCY_SUFFIX", "").strip()
    unit = f" {suffix}" if suffix else ""
    rm_pct = _rm_pct_of_base(rm, base)
    rel = ((var995 - base) / base * 100.0) if base > 1e-9 else 0.0

    if hints.get("plateau_bootstrap_adequacy"):
        p1 = (
            f"Точкова оцінка зобов’язань за методом ланцюгової драбини — близько {_money(total)}{unit}; це **базовий резерв** — орієнтир очікуваних майбутніх виплат за моделлю. "
            f"Бутстреп-симуляція побудована навколо цієї оцінки: частка прогонів, де результат **вищий** за цю базу, — близько {p_def * 100:.2f}%. "
            f"Це **часто буває близько до половини** при симетричній бутстреп-симуляції і **не** означає, що «резервів замало» в операційному сенсі. "
            f"Додаток для орієнтиру 99,5% скромний — {_money(rm)}{unit} (**{rm_pct:.2f}%** від бази), квантиль лише на **{rel:.2f}%** вище за базу — типовий вигляд **адекватного** резерву з вузьким хвостом у грошах."
        )
        p2 = (
            "Цю частку симуляції **не** варто подавати головним меседжем як «ймовірність дефіциту коштів»: це технічна характеристика розподілу навколо бази, а не сигнал капітального голоду. "
            "**Не** випливає з цих цифр заклик нарощувати капітал лише через частку близько до половини. Достатньо **планового** перегляду припущень і даних."
        )
        if hints.get("triangle_non_monotonic"):
            p2 += (
                " Додатково: у даних є **немонотонний кумулятив** по лагах (корекції, повернення, суброгація) — метод ланцюгової драбини тут є спрощеною моделлю; "
                "не варто читати симуляційні відсотки як «замало грошей на балансі»."
            )
        p3 = "Усі суми — брутто до перестрахування; реальна стійкість також залежить від договорів перестрахування та утримання."
        return "\n\n".join([p1, p2, p3, _deficit_method_comparison_paragraph(risk)])

    # Only non-monotonic (no plateau band): short caveat, still neutral on deficit hype.
    p1 = (
        f"Базовий резерв за методом ланцюгової драбини — близько {_money(base)}{unit}. "
        f"Частка симуляцій з IBNR вище за базу — близько {p_def * 100:.2f}%; орієнтир 99,5% — {_money(var995)}{unit} "
        f"(надбавка {_money(rm)}{unit}, **{rm_pct:.2f}%** від бази)."
    )
    p2 = (
        "У даних зафіксовано **немонотонний кумулятив** по лагах (типові корекції, повернення, суброгація). "
        "Метод ланцюгової драбини й бутстреп-симуляція тоді лише орієнтири: **не** варто апокаліптично трактувати метрики як «критичний дефіцит» резерву чи банкрутство. "
        "Рекомендація — плановий перегляд якості даних і припущень моделі."
    )
    p3 = "Усі суми — брутто до перестрахування; реальна стійкість також залежить від договорів перестрахування та утримання."
    return "\n\n".join([p1, p2, p3, _deficit_method_comparison_paragraph(risk)])


def _narrative_override_block(calc: dict[str, Any]) -> str:
    """Hard prompt tail so the LLM cannot 'hallucinate' a deficit story on a plateau bootstrap profile."""
    hints = calc.get("narrative_hints") or {}
    if not (hints.get("plateau_bootstrap_adequacy") or hints.get("triangle_non_monotonic")):
        return ""
    chunks: list[str] = [
        "\n\n>>> ПЕРЕВІЗНИЙ БЛОК (найвищий пріоритет над формулюваннями про «ризик дефіциту») <<<\n",
    ]
    if hints.get("plateau_bootstrap_adequacy"):
        chunks.append(
            "Якщо P<sub>def</sub> близько до 50% **і** відносна надбавка до 99,5% **і** відносне перевищення VaR<sub>99.5%</sub> над базою — "
            "**обидва дуже малі (типово <14% від бази кожне)**, це **не** доказ «помірного ризику недостатності резерву» "
            "і **не** підстава закликати нарощувати капітал чи «терміново підсилювати подушку». "
            "Опиши ситуацію як **адекватний резерв** при симетричній бутстреп-симуляції навколо точкової оцінки. "
            "**Заборонено** лякати менеджмент через сам факт P<sub>def</sub>≈50%.\n"
        )
    if hints.get("triangle_non_monotonic"):
        chunks.append(
            "У вхідних даних — **немонотонний кумулятив** (повернення/суброгація/корекції). "
            "Тоді **заборонено** апокаліптику про банкрутство, «критичний дефіцит» резерву чи крах лише з цих метрик.\n"
        )
    chunks.append(
        "Порада для менеджменту: лише **плановий** перегляд припущень і даних; без мотиву «рятувати резерв».\n"
    )
    return "".join(chunks)


def build_chief_actuary_fallback_narrative(calc: dict[str, Any]) -> str:
    """Coherent prose when LLM is unavailable (Ukrainian, 2–3 paragraphs)."""
    risk = calc.get("risk_metrics", {})
    if risk.get("risk_analysis_skipped"):
        total = float(calc.get("total_reserve") or 0.0)
        if total < -1e-9:
            return _NEGATIVE_IBNR_SURPLUS_CLOSURE
        return _ZERO_IBNR_CLOSURE

    total = float(calc.get("total_reserve") or 0.0)
    hints = calc.get("narrative_hints") or {}
    if total < -1e-9:
        return _NEGATIVE_IBNR_SURPLUS_CLOSURE
    if hints.get("plateau_bootstrap_adequacy") or hints.get("triangle_non_monotonic"):
        return _adequacy_profile_fallback_narrative(calc, hints)

    base = float(risk.get("base_reserve", total))
    var995 = float(risk.get("var_995") or 0.0)
    rm = float(risk.get("rm_required_005") or 0.0)
    p_def = float(risk.get("p_def") or 0.0)
    p_def_mack = float(risk.get("p_def_mack_analytical") or 0.0)
    mack_se = float(risk.get("mack_se_total") or 0.0)
    suffix = os.getenv("REPORT_CURRENCY_SUFFIX", "").strip()
    unit = f" {suffix}" if suffix else ""

    rel = ((var995 - base) / base * 100.0) if base > 1e-9 else 0.0
    rm_pct = _rm_pct_of_base(rm, base)
    heavy = base > 1e-9 and (var995 - base) / base >= 0.20
    tail_note = risk.get("heavy_tail")
    capital_surplus = bool(risk.get("capital_surplus_regime"))
    cat_tail_warn = bool(risk.get("low_p_def_extreme_tail_warning"))
    small_margin = rm_pct < 15.0
    p_def_near_half = 0.40 <= p_def <= 0.60

    p1 = (
        f"За класичною актуарною екстраполяцією по трикутнику виплат поточна оцінка зобов’язань — близько {_money(total)}{unit}; "
        f"це гроші, які ми закладаємо як «базовий резерв» на очікувані майбутні виплати, уже відомі з даних."
    )

    if p_def_near_half and small_margin:
        p_def_line = (
            f"Ймовірність того, що резервів не вистачить порівняно з базою {_money(base)}{unit}, — близько {p_def * 100:.2f}%. "
            f"При цьому необхідна надбавка до надійності регулятора — лише близько {rm_pct:.2f}% від цієї бази ({_money(rm)}{unit}), "
            f"тож це не сигнал «нестабільності»: центральна оцінка проходить близько до середини симульованих сценаріїв, "
            f"а грошовий розмір можливого перевищення резерву залишається невеликим. Портфель виглядає прогнозованим."
        )
    elif 0.45 <= p_def <= 0.55:
        p_def_line = (
            f"Ризик дефіциту відносно бази {_money(base)}{unit} — близько {p_def * 100:.2f}%; "
            f"додаткова подушка безпеки для «найгірших» регуляторних сценаріїв відносно бази — близько {rm_pct:.2f}% ({_money(rm)}{unit}). "
            f"Якщо ця надбавка помірна, образ «підкидання монети» стосується знаку відхилення від центральної оцінки, а не великого удару в грошах."
        )
    else:
        p_def_line = (
            f"Ймовірність нестачі коштів (що фактичні виплати перевищать закладену базу {_money(base)}{unit}) — "
            f"{p_def * 100:.2f}%; орієнтовна кваліфікація: «{_deficit_risk_label_ua(p_def, rm_pct)}»."
        )

    p2 = p_def_line
    if cat_tail_warn:
        p2 += (
            " Навіть коли ризик дефіциту виглядає низьким, відстань до стресового рівня капіталу (найгірші сценарії) велика: "
            "звичайні сценарії виглядають безпечними, але портфель може бути вразливим до поодиноких катастрофічних виплат "
            "у хвості розподілу — це варто поєднувати з аналізом перестрахування (оцінки брутто)."
        )

    tail_sentence = ""
    if (heavy or tail_note) and not (p_def_near_half and small_margin):
        tail_sentence = (
            f" Орієнтир для найгірших сценаріїв — близько {_money(var995)}{unit}, що на {rel:.2f}% вище за базу: "
            f"це натякає на чутливість до екстремальних подій і важчий хвіст — однією «плоскою» сумою ризик не описати."
        )
    elif p_def_near_half and small_margin:
        tail_sentence = (
            f" Капітал для покриття найгірших сценаріїв ({_money(var995)}{unit}) лише на {rel:.2f}% вищий за базу — "
            f"хвіст у грошах стислий, що підсилює висновок про відносну стабільність."
        )
    else:
        tail_sentence = (
            f" Регуляторний стресовий орієнтир ({_money(var995)}{unit}) дає перевищення над базою близько {rel:.2f}%; "
            f"тлумачення має узгоджуватися з масштабом резервного буфера ({rm_pct:.2f}% від бази)."
        )

    if capital_surplus and base > 1e-9:
        p3 = (
            f"Рівень «найгірших» симульованих виплат ({_money(var995)}{unit}) не перевищує базовий резерв "
            f"({_money(base)}{unit}); додаткова подушка безпеки для цього порогу — 0. Поточні резерви перекривають "
            f"навіть обраний екстремальний регуляторний орієнтир (можлива надлишковість). На дуже малих даних "
            f"симуляція інколи дає орієнтир на рівні або нижче бази — варто перевірити обсяг прогонів і стабільність."
        )
    else:
        p3 = (
            f"Для орієнтиру Solvency II додаткова надбавка понад базу — близько {_money(rm)}{unit} "
            f"({rm_pct:.2f}% від бази), тобто скільки ще слід мати «зверху», щоб витримати рідкі великі виплати."
            f"{tail_sentence}"
        )

    if (heavy and rm_pct >= 10.0) or (p_def >= 0.25 and rm_pct >= 15.0) or (heavy and not small_margin):
        advice = (
            "Рекомендація: узгодити з фінансовим блоком або перестрахуванням конкретний обсяг капіталу/третьої лінії "
            "не нижче згаданого буфера та переглянути політику резервування до наступного кварталу."
        )
    elif rm > 0 and rel >= 10.0:
        advice = (
            "Рекомендація: закріпити у внутрішніх лімітах мінімальний додаток до резерву на рівні обчисленого буфера "
            "і відстежувати, як він змінюється при оновленні трикутника."
        )
    else:
        advice = (
            "Рекомендація: зафіксувати поточні припущення моделі екстраполяції та симуляції хвоста в робочій документації "
            "і планово перевірити стійкість висновку при наступному оновленні даних."
        )

    advice += (
        " Усі наведені симуляції — брутто (до перестрахування); для управління капіталом доцільно переглянути "
        "договори перестрахування та ліміти власного утримання."
    )

    comparison = _deficit_method_comparison_paragraph(risk)

    return "\n\n".join([p1, p2, p3, comparison, advice])


def build_llm_narrative(
    user_prompt: str,
    method: str,
    calc: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    currency_suffix = os.getenv("REPORT_CURRENCY_SUFFIX", "").strip()
    language_rule = _language_instruction(user_prompt)
    english_style = _english_b2_style_block(user_prompt)

    if provider != "openai" or not api_key:
        return None, {"llm_used": False, "provider": provider, "reason": "missing_provider_or_key"}

    risk = calc.get("risk_metrics", {})
    total = float(calc.get("total_reserve") or 0.0)

    if risk.get("risk_analysis_skipped"):
        sign = "negative" if total < -1e-9 else "zero"
        surplus_prompt = (
            "Ти — старший актуарій. Напиши 2 короткі абзаци зв’язною прозою, без маркованих списків.\n"
            f"{language_rule}\n\n"
            f"{english_style}"
            f"Факт з моделі ланцюгової драбини: сумарна точкова оцінка резерву (IBNR) = {total:.2f}. "
            "Бутстреп-симуляція хвоста не виконувалася — **не вигадуй** P<sub>def</sub>, VaR, «ймовірність дефіциту» чи інші метрики симуляції.\n\n"
            "СУВОРІ ПРАВИЛА (не порушуй):\n"
        )
        if sign == "negative":
            surplus_prompt += (
                "- Оцінка **від’ємна** ⇒ резерви **надлишкові** відносно модельної кінцевої величини: на балансі вже більше коштів, ніж модель вважає "
                "потрібним для закриття залишкових зобов’язань.\n"
                "- **Маржа ризику для 99,5% = 0**; **не закликай** додавати капітал або «підсилювати буфер» під цю оцінку.\n"
                "- **Заборонено** писати про ризик дефіциту резервів, банкрутство, «критичну нестачу» коштів, загрозу стабільності через нестачу резерву.\n"
                "- Натомість: очікуване **повернення коштів** / перескладення / суброгація / закриття справ; **фінансовий стан** за цією картиною — **спокійний**.\n"
            )
        else:
            surplus_prompt += (
                "- Оцінка **нульова** (або фактично нульова) ⇒ додаткових зобов’язань «на доростання» немає; маржа = 0; хвостовий аналіз не застосовувався.\n"
                "- Не вигадуй симуляційні ризики.\n"
            )
        surplus_prompt += (
            "- Нагадай: усі суми **брутто** (до перестрахування).\n\n"
            f"Запит користувача (тон і акценти, якщо доречно): {user_prompt}\n"
        )
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.responses.create(model=model, input=surplus_prompt)
            text = response.output_text.strip()
            if text:
                return text, {
                    "llm_used": True,
                    "provider": provider,
                    "model": model,
                    "narrative_mode": "reserve_surplus_or_zero",
                    "best_estimate_sign": sign,
                }
        except Exception as exc:  # pragma: no cover
            return None, {"llm_used": False, "provider": provider, "reason": f"surplus_narrative:{exc}"}
        return None, {"llm_used": False, "provider": provider, "reason": "empty_surplus_response"}

    base = float(risk.get("base_reserve", total))
    var995 = float(risk.get("var_995") or 0.0)
    rm = float(risk.get("rm_required_005") or 0.0)
    p_def = float(risk.get("p_def") or 0.0)
    p_def_mack = float(risk.get("p_def_mack_analytical") or 0.0)
    mack_se = float(risk.get("mack_se_total") or 0.0)
    var95 = float(risk.get("var_95") or 0.0)
    tvar995 = float(risk.get("tvar_995") or 0.0)
    rel_excess_pct = ((var995 - base) / base * 100.0) if base > 1e-9 else 0.0
    rm_pct_of_base = _rm_pct_of_base(rm, base)
    p_def_near_half = 0.40 <= p_def <= 0.60
    small_margin_vs_base = rm_pct_of_base < 15.0
    capital_surplus_regime = bool(risk.get("capital_surplus_regime"))
    low_p_def_tail_warn = bool(risk.get("low_p_def_extreme_tail_warning"))
    mandate_heavy_tail = (
        base > 1e-9
        and (var995 - base) / base >= 0.20
        and not (p_def_near_half and small_margin_vs_base)
        and not capital_surplus_regime
    )

    unit_rule = (
        f"Позначення сум: додавай суфікс «{currency_suffix}» після кожної грошової величини."
        if currency_suffix
        else (
            "Позначення сум: використовуй ті самі одиниці, що й вхідні виплати в трикутнику "
            "(без перекладу в мільйони, якщо це дає дроби на кшталт «0.000169 млн»). "
            "Формулюй на кшталт: «… одиниці, узгоджені з вхідними даними» або конкретну одиницю, "
            "якщо користувач її назвав у запиті."
        )
    )

    numeric_block = (
        f"Факти з розрахунку (внутрішні назви; у тексті для людей використовуй людські формулювання з правила живої мови; "
        f"грошові величини з округленням до 2 знаків):\n"
        f"- Сумарний IBNR (метод ланцюгової драбини, точкова оцінка): {total:.2f}\n"
        f"- Базовий резерв (IBNR): {base:.2f}\n"
        f"- Ймовірність дефіциту резервів P<sub>def</sub> (Monte Carlo / бутстреп): {p_def:.6f} ({p_def * 100:.2f}%)\n"
        f"  (симульований IBNR перевищує базовий резерв)\n"
        f"- Ймовірність дефолту P<sub>default</sub> (Mack + логнормальний): {p_def_mack:.6f} ({p_def_mack * 100:.2f}%)\n"
        f"  (майбутні виплати перевищать зарезервований IBNR; σ Mack = {mack_se:.2f})\n"
        f"- VaR<sub>95%</sub> симульованого IBNR: {var95:.2f}\n"
        f"- VaR<sub>99.5%</sub> симульованого IBNR: {var995:.2f}\n"
        f"- TVaR<sub>99.5%</sub>: {tvar995:.2f}\n"
        f"- Додаток до резерву для рівня надійності 99.5% (max(0, VaR<sub>99.5%</sub> − база)): {rm:.2f}\n"
        f"- Цей додаток як частка від базового резерву: {rm_pct_of_base:.2f}% (важливо для узгодженої інтерпретації з P<sub>def</sub>)\n"
        f"- Наскільки VaR<sub>99.5%</sub> вищий за базу у відносних одиницях: {rel_excess_pct:.2f}%\n"
        f"- Діагностика «важкий хвіст» у моделі: {risk.get('heavy_tail')}\n"
        f"- Режим надлишку резерву vs VaR<sub>99.5%</sub> (квантиль ≤ база, маржа = 0): {capital_surplus_regime}\n"
        f"- Попередження: низький P<sub>def</sub>, але велика відстань до VaR<sub>99.5%</sub> (катастрофічний хвіст): {low_p_def_tail_warn}\n"
        f"- База аналізу: {risk.get('analysis_basis', 'gross_before_reinsurance')}\n"
        f"- Чи доречні акценти на «важкому хвості»/екстремальних сценаріях за сукупністю відсотків: "
        f"{'так' if mandate_heavy_tail else 'лише якщо абсолютна та відносна маржа справді великі — інакше не роздмухуй'}\n"
    )

    override_block = _narrative_override_block(calc)
    scope_block = _narrative_scope_block(user_prompt)

    prompt = (
        "Ти — досвідчений актуарій-аналітик: не заповнюй шаблон, а зроби експертний висновок, який читається як від живої людини.\n\n"
        f"{language_rule}\n\n"
        f"{english_style}"
        f"{scope_block}"
        "ПРІОРИТЕТ ЗАПИТУ КОРИСТУВАЧА (User Constraint Override) — перевір ПЕРШИМ:\n"
        "Якщо з промпта випливає, що потрібні лише одна чи дві названі метрики, або явно вимагають бути максимально стислим/лаконічним, "
        "або задані жорсткі обмеження формату (наприклад «тільки число», «одне речення», «без пояснень») — ти ЗОБОВ’ЯЗАНИЙ ігнорувати будь-які нижчі правила "
        "про «обов’язкові» чотири показники, розгорнуту форму звіту та фінальну пораду менеджменту. Відповідай СУВОРО тим, що просить користувач, "
        "без зайвого математичного чи методологічного контексту. Блок «Факти» нижче — довідник: використовуй лише релевантні рядки.\n"
        "Якщо ж користувач просить аналіз, звіт, обговорення ризиків, пояснення або не обмежує обсяг — тоді застосовуй повний набір правил «Повний звіт».\n\n"
        "ПРАВИЛО ЖИВОЇ МОВИ — за замовчуванням для загальних і бізнес-запитів "
        "(не дублюй це, якщо користувач просить суто технічний звіт, явно назви змінні або вузький запит лише на код метрики):\n"
        "- Пояснюй так, щоб зрозуміла людина без технічної освіти. Не «засмічуй» відповідь абревіатурами (IBNR, BE, P<sub>def</sub>, VaR, TVaR, RM тощо), "
        "якщо це не вимагає сам запит.\n"
        "- Узгоджені заміни (варіюй, не копіюй одне й те саме):\n"
        "  • IBNR / точкова оцінка → «базовий резерв», «очікувані майбутні виплати», «поточна оцінка зобов’язань»;\n"
        "  • P<sub>def</sub> (Monte Carlo) → «ймовірність дефіциту резервів», «ризик нестачі IBNR»;\n"
        "  • P<sub>default</sub> (Mack) → «ймовірність дефолту», «ризик, що майбутні виплати перевищать резерв»;\n"
        "  • VaR<sub>99.5%</sub> → «капітал для покриття найгірших сценаріїв», «регуляторні вимоги до надійності», «стресовий рівень капіталу»;\n"
        "  • маржа ризику → «додаткова подушка безпеки», «необхідна надбавка», «резервний буфер».\n"
        "- Цифри з блоку «Факти» вплітай у речення природно, як фінансовий радник у розмові, а не списком «назва — число».\n"
        "У блоці «Факти» нижче залишаються технічні підписи для твоєї опори; у відповіді користувачу перекладай їх людською мовою.\n\n"
        "СТИЛЬ ВІДПОВІДІ (для повного звіту або коли доречно; при вузькому запиті — мінімальний стиль за вимогою):\n"
        "- Якщо запит побудований бізнес-мовою (керівництво, фінанси, стратегія) — пиши мовою фіндиректора: зрозуміло для менеджменту, "
        "без зайвого жаргону, але точно.\n"
        "- Якщо запит формальний, математичний або статистичний — відповідай у відповідному академічно-статистичному тоні (ймовірності, квантилі, розподіл, невизначеність).\n"
        "- Якщо змішаний стиль — збалансуй: спочатку суть для бізнесу, потім коротко технічне обґрунтування.\n\n"
        "ФОРМА ПОВНОГО ЗВІТУ (не застосовуй при вузькому запиті користувача): 2–3 зв’язних абзаци природною мовою (без нумерованого списку 1-2-3-4), "
        "плюс **окремий короткий абзац «Порівняння методів»**: ймовірність **дефіциту резервів** (Monte Carlo) vs **дефолту** (Mack) — обидва % з «Фактів», короткий висновок про узгодженість. "
        "Логіка читання: від точкової оцінки зобов’язань → до ризику та хвоста → до наслідків для капіталу → порівняння двох методів. "
        "Не повторюй однакові технічні підписи щоразу; пояснюй різними словами, що означає кожна величина для прийняття рішень.\n\n"
        "УЗГОДЖЕНИЙ АНАЛІЗ для ПОВНОГО звіту (без логічних суперечностей; при вузькому запиті — пропусти зайве):\n"
        "- Завжди поєднуй P<sub>def</sub> із відносною величиною додатку до резерву (% від бази з блоку «Факти»). Сам по собі P<sub>def</sub> близько 50% "
        "не є доказом «серйозного ризику» чи «нестабільності», якщо додаток для 99.5% становить менше приблизно 10–15% від бази: "
        "у такій комбінації це часто математично нормально для профілю навколо точкової оцінки, абсолютний масштаб можливого дефіциту малий — "
        "портфель можна охарактеризувати як відносно стабільний і прогнозований на рівні резерву. Не використовуй у цьому випадку лякаючі формулювання про критичну небезпеку лише через «підкидання монети».\n"
        "- Якщо ж P<sub>def</sub> високий і водночас відносна маржа велика, або VaR<sub>99.5%</sub> істотно (у %) відривається від бази — тоді обґрунтовано говорити про чутливість до екстремальних сценаріїв, важчий хвіст, потребу в капіталі чи перестрахуванні; масштаб має відповідати цифрам.\n"
        "- Пояснюй сенс цифр, а не лише називай їх; варіюй формулювання між звітами.\n"
        "- Навіть якщо P<sub>def</sub> дуже низький (<10%), обов’язково звір відстань до VaR<sub>99.5%</sub> (у % від бази у «Фактах»). "
        "Якщо прапор «катастрофічний хвіст при низькому P<sub>def</sub>» увімкнено — поясни: базові сценарії виглядають безпечними, "
        "але портфель може залишатися вразливим до поодиноких катастрофічних подій у хвості; це не суперечить низькій P<sub>def</sub>.\n"
        "- Правило надлишку: якщо VaR<sub>99.5%</sub> ≤ базового резерву, додаток (маржа) = 0; поясни, що резерви можуть бути надлишковими "
        "відносно цього регуляторного квантиля (на малих даних бутстреп-симуляція інколи дає квантиль ≤ бази).\n"
        "- БРУТТО ТА ОБЕРЕЖНІСТЬ (для повного звіту або коли обговорюєш ризик; при вузькому запиті «лише метрика X» не розширюй це до лекції): "
        "усі розрахунки — брутто, до перестрахування. КАТЕГОРИЧНО ЗАБОРОНЕНО стверджувати про «неминуче банкрутство», «фінансовий крах», тотальний колапс. "
        "Реальні компанії знімають ризики перестрахуванням. У повному звіті пропонуй перегляд договорів перестрахування та лімітів утримання — без апокаліптики.\n\n"
        "ПОВНИЙ ЗВІТ — додатково: органічно вплети (хоча б по одному разу кожну ідею) чотири змістові опори з «Фактів» — "
        "поточна оцінка зобов’язань, ризик нестачі резервів (у відсотках), стресовий рівень для регуляторної надійності та додаткова подушка безпеки — "
        "усі суми з округленням до двох знаків; формулювання для читача — людською мовою. "
        "Якщо запит користувача вузький — це правило НЕ діє.\n\n"
        f"{unit_rule}\n"
        "ЗАБОРОНЕНО: дрібні нечитабельні «мільйони» (на кшталт 0.000169 млн); тримайся базових одиниць даних або тисяч цілих, якщо це природно.\n\n"
        "Модель: метод ланцюгової драбини; симуляція IBNR — бутстреп Пірсонових залишків (ODP); аналітична невизначеність — Mack SE + логнормальний CDF.\n"
        f"Метод оцінки: {method}\n\n"
        f"Запит користувача: {user_prompt}\n\n"
        f"{numeric_block}\n"
        f"{override_block}"
        "ЗАВЕРШЕННЯ: для ПОВНОГО звіту або коли користувач просить рекомендації — заверши однією конкретною порадою для менеджменту з цих цифр. "
        "Якщо запит вузький і поради не просять — не додавай пораду.\n\n"
        "Не виводь сирих таблиць чи JSON."
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=model, input=prompt)
        text = response.output_text.strip()
        if not text:
            return None, {"llm_used": False, "provider": provider, "reason": "empty_response"}
        return text, {"llm_used": True, "provider": provider, "model": model}
    except Exception as exc:  # pragma: no cover
        return None, {"llm_used": False, "provider": provider, "reason": str(exc)}


def followup_fallback_reply(user_message: str, context_json: str) -> str:
    """Short reply when LLM is unavailable; uses compact JSON from last run."""
    lang = detect_response_language(user_message)
    try:
        data = json.loads(context_json)
    except json.JSONDecodeError:
        if lang == "en":
            return "I could not read the last result. Please run the calculation again."
        return (
            "Не вдалося прочитати контекст останнього прогону. Запустіть повний розрахунок ще раз "
            "або перевірте збережену сесію."
        )
    rm: dict[str, Any] = {}
    for row in data.get("risk_metrics") or []:
        if isinstance(row, dict) and "metric" in row:
            rm[str(row["metric"])] = row.get("value")
    lines: list[str] = []
    if lang == "en":
        lines.append("*(The AI model is off — short answer from the last run.)*\n")
        if rm.get("base_reserve") is not None:
            lines.append(f"- Base reserve: **{_money(float(rm['base_reserve']))}**.")
        if rm.get("var_995") is not None:
            lines.append(f"- Stress level (99.5%): **{_money(float(rm['var_995']))}**.")
        if rm.get("p_def_bootstrap", rm.get("p_def")) is not None:
            p = float(rm.get("p_def_bootstrap", rm["p_def"]))
            lines.append(f"- Chance the reserve is not enough (simulation): **{p * 100:.2f}%**.")
        if rm.get("p_default_mack", rm.get("p_def_mack_analytical")) is not None:
            p = float(rm.get("p_default_mack", rm["p_def_mack_analytical"]))
            lines.append(f"- Default probability (Mack): **{p * 100:.2f}%**.")
        if rm.get("risk_analysis_skipped"):
            lines.append("- We did **not** run the simulation (reserve estimate is zero or negative).")
        lines.append("\nSee the full text above. Tables and charts are below.")
        return "\n".join(lines)

    lines.append("*(Мовна модель вимкнена або недоступна — коротка відповідь із цифр останнього прогону.)*\n")
    if rm.get("base_reserve") is not None:
        lines.append(f"- Базовий резерв: **{_money(float(rm['base_reserve']))}**.")
    if rm.get("var_995") is not None:
        lines.append(f"- VaR<sub>99.5%</sub> (симуляція): **{_money(float(rm['var_995']))}**.")
    if rm.get("p_def_bootstrap", rm.get("p_def")) is not None:
        p = float(rm.get("p_def_bootstrap", rm["p_def"]))
        lines.append(f"- Ймовірність дефіциту резервів (Monte Carlo): **{p * 100:.2f}%**.")
    if rm.get("p_default_mack", rm.get("p_def_mack_analytical")) is not None:
        p = float(rm.get("p_default_mack", rm["p_def_mack_analytical"]))
        lines.append(f"- Ймовірність дефолту (Mack): **{p * 100:.2f}%**.")
    if rm.get("risk_analysis_skipped"):
        lines.append("- Симуляцію хвоста **пропущено** (нульова або від’ємна точкова оцінка).")
    lines.append("\nПовний текст висновку з останнього прогону — у повідомленні вище; таблиці й графіки доступні нижче.")
    return "\n".join(lines)


def build_followup_llm_response(
    user_message: str,
    context_json: str,
    *,
    last_narrative: str = "",
    focus: str = "general",
) -> tuple[str | None, dict[str, Any]]:
    """Answer a follow-up question using last run JSON context (OpenAI Responses API)."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    language_rule = _language_instruction(user_message)
    english_style = _english_b2_style_block(user_message)
    focus_block = _followup_focus_block(focus, last_narrative)
    if provider != "openai" or not api_key:
        return followup_fallback_reply(user_message, context_json), {
            "llm_used": False,
            "provider": provider,
            "reason": "missing_provider_or_key",
            "mode": "followup",
        }

    prompt = (
        "Ти — старший актуарій. Користувач ставить **уточнювальне питання** до вже виконаного прогону методу ланцюгової драбини "
        "з бутстреп-симуляцією та аналітичною оцінкою Mack+lognormal.\n\n"
        f"{language_rule}\n\n"
        f"{english_style}"
        f"{focus_block}"
        "ПРАВИЛА:\n"
        "- Відповідай **лише** на основі JSON-контексту нижче (і тексту для перекладу, якщо режим перекладу). "
        "Не вигадуй нових цифр, нових методів чи нових симуляцій.\n"
        "- Якщо питання стосується **іншого методу резервування** (BF, GLM тощо), якого немає в контексті — чесно скажи, що він ще не підключений. "
        "Для **Monte Carlo / дефіциту резервів** — `p_def_bootstrap` або `p_def`; "
        "для **Mack / дефолту** — `p_default_mack`, `p_def_mack_analytical`, `mack_se_total`.\n"
        "- Якщо даних у JSON бракує для відповіді — скажи, чого саме не вистачає.\n"
        "- Стиль: стільки абзаців, скільки потрібно для фокусу (зазвичай 1–2). **Не** роби повторний повний звіт.\n"
        "- Не дублюй увесь JSON; не використовуй довгі марковані списки.\n\n"
        f"Запит користувача:\n{user_message}\n\n"
        "Контекст останнього прогону (JSON):\n"
        f"{context_json}\n"
    )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=model, input=prompt)
        text = response.output_text.strip()
        if not text:
            return followup_fallback_reply(user_message, context_json), {
                "llm_used": False,
                "provider": provider,
                "reason": "empty_followup_response",
                "mode": "followup",
            }
        return text, {"llm_used": True, "provider": provider, "model": model, "mode": "followup"}
    except Exception as exc:  # pragma: no cover
        return followup_fallback_reply(user_message, context_json), {
            "llm_used": False,
            "provider": provider,
            "reason": f"followup:{exc}",
            "mode": "followup",
        }
