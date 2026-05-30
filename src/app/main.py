"""Streamlit UI: conversational actuarial copilot."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agents.llm_client import build_followup_llm_response
from agents.orchestrator import OrchestratorAgent
from app.chat_router import (
    ChatIntent,
    classify_chat_intent,
    compact_context_for_followup,
    infer_followup_focus,
    response_no_triangle_loaded,
    response_thanks,
    response_trivial_without_run,
    response_wait_for_run,
)
from app.text_formatting import format_actuarial_notation_html
from core.models import RunRequest, RunResult
from tools.charting import (
    build_development_factors_chart,
    build_observed_vs_ultimate_chart,
    build_simulated_ibnr_distribution,
    build_triangle_heatmap,
)
from tools.column_inference import infer_column_mapping
from tools.triangle_from_transactions import load_dataframe_from_upload, parse_transaction_dataframe

_AUTO = "— Auto —"
_UI_VERSION = "welcome-short-v2"
_DEFAULT_RUN_PROMPT = "Estimate the reserve and explain the risks"
_CHAT_INPUT_KEY = "actuarial_chat_input"
_DATA_FILE_KEY = "chat_data_file"


def _col_choice(selectbox_val: str) -> str | None:
    return None if selectbox_val == _AUTO else selectbox_val


def _welcome_message() -> str:
    return (
        "Hi! I'm your **actuarial copilot** for reserve analysis — chain-ladder reserves, "
        "risk metrics, and plain-language explanations.\n\n"
        "Attach a **CSV or Excel** file in the sidebar, then ask your question "
        "(e.g. **Estimate the reserve and explain the risks**)."
    )


def _language_pref() -> str:
    return str(st.session_state.get("chat_language_pref") or "auto")


def _init_session() -> None:
    if st.session_state.get("ui_version") != _UI_VERSION:
        st.session_state.ui_version = _UI_VERSION
        st.session_state.messages = [{"role": "assistant", "content": _welcome_message()}]
        st.session_state.result_panel = None
        st.session_state.last_result = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "triangle_records" not in st.session_state:
        st.session_state.triangle_records = []
    if "upload_report" not in st.session_state:
        st.session_state.upload_report = None
    if "upload_df" not in st.session_state:
        st.session_state.upload_df = None
    if "upload_name" not in st.session_state:
        st.session_state.upload_name = None
    if "simulations" not in st.session_state:
        st.session_state.simulations = 10000
    if "last_run_prompt" not in st.session_state:
        st.session_state.last_run_prompt = None
    if "chat_language_pref" not in st.session_state:
        st.session_state.chat_language_pref = "auto"
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": _welcome_message()}]
    if "result_panel" not in st.session_state:
        st.session_state.result_panel = None


def _triangle_valid() -> bool:
    report = st.session_state.upload_report or {}
    return bool(st.session_state.triangle_records) and bool(report.get("valid"))


def _accident_year_range(records: list[dict]) -> str | None:
    years: list[int] = []
    for row in records:
        ay = row.get("accident_year")
        if ay is not None:
            try:
                years.append(int(ay))
            except (TypeError, ValueError):
                continue
    if not years:
        return None
    return f"{min(years)}–{max(years)}"


def _has_uploaded_file() -> bool:
    return bool(st.session_state.get("upload_name")) or st.session_state.get("upload_df") is not None


def _chat_has_activity() -> bool:
    """True once the user has sent a message or a calculation has completed."""
    if st.session_state.get("last_result") is not None:
        return True
    messages = st.session_state.get("messages") or []
    if any(m.get("role") == "user" for m in messages):
        return True
    return len(messages) > 1


def _execute_full_run(prompt: str) -> RunResult:
    orchestrator = OrchestratorAgent()
    request = RunRequest(
        user_prompt=prompt,
        simulations=int(st.session_state.simulations),
        triangle_records=st.session_state.triangle_records,
    )
    result = orchestrator.run(request)
    st.session_state.last_result = result
    st.session_state.last_run_prompt = prompt
    st.session_state.result_panel = None
    return result


def _render_result_actions() -> None:
    col_tables, col_charts, _ = st.columns([1, 1, 5])
    panel = st.session_state.result_panel
    with col_tables:
        if st.button("Tables", type="secondary", width="stretch"):
            st.session_state.result_panel = None if panel == "tables" else "tables"
    with col_charts:
        if st.button("Charts", type="secondary", width="stretch"):
            st.session_state.result_panel = None if panel == "charts" else "charts"


def _render_tables(result: RunResult) -> None:
    st.subheader("Tables")

    triangle_df = pd.DataFrame(result.tables["triangle"])
    if not triangle_df.empty:
        cumulative = (
            triangle_df.pivot_table(
                index="accident_year",
                columns="development_period",
                values="value",
                aggfunc="sum",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        cumulative.index.name = "Accident Year"
        cumulative.columns = [f"Development Period {c}" for c in cumulative.columns]
        st.write("Historical Cumulative Payments")
        st.table(cumulative)

    if "ultimate_ibnr" in result.tables:
        ibnr_df = pd.DataFrame(result.tables["ultimate_ibnr"])
        if not ibnr_df.empty:
            base_reserve_df = pd.DataFrame(
                {
                    "Accident Year": ibnr_df["accident_year"],
                    "Currently Paid": ibnr_df["latest"],
                    "Growth Factor": ibnr_df.apply(
                        lambda row: round(float(row["ultimate"]) / float(row["latest"]), 6)
                        if float(row["latest"]) != 0
                        else None,
                        axis=1,
                    ),
                    "Expected Final Cost": ibnr_df["ultimate"],
                    "Required Reserve": ibnr_df["ibnr"],
                }
            )
            st.write("Base Reserve Calculation by Year")
            st.table(base_reserve_df)

    if "risk_metrics" in result.tables:
        raw_metrics = {
            str(row.get("metric")): row.get("value")
            for row in result.tables["risk_metrics"]
            if isinstance(row, dict)
        }
        metric_rows = [
            ("Base Reserve (IBNR)", raw_metrics.get("base_reserve")),
            (
                "Reserve deficit probability (Monte Carlo)",
                raw_metrics.get("p_def_bootstrap", raw_metrics.get("p_def")),
            ),
            (
                "Default probability (Mack analytical)",
                raw_metrics.get("p_default_mack", raw_metrics.get("p_def_mack_analytical")),
            ),
            ("Mack Standard Error", raw_metrics.get("mack_se_total")),
            ("Stress Level Capital 99.5%", raw_metrics.get("var_995")),
            ("Additional Risk Margin", raw_metrics.get("rm_required_005")),
        ]
        risk_df = pd.DataFrame(
            [{"Metric": label, "Value": value} for label, value in metric_rows if value is not None]
        )
        if not risk_df.empty:
            st.write("Risk Analysis & Stress Testing")
            st.table(risk_df)


def _render_charts(result: RunResult) -> None:
    st.subheader("Charts")
    payload = result.chart_payload
    if not payload:
        st.info("No chart data available.")
        return
    st.plotly_chart(build_triangle_heatmap(payload["triangle_records"]), use_container_width=True)
    st.plotly_chart(build_development_factors_chart(payload["factors"]), use_container_width=True)
    st.plotly_chart(
        build_observed_vs_ultimate_chart(payload["latest_by_ay"], payload["ultimate_by_ay"]),
        use_container_width=True,
    )
    st.plotly_chart(
        build_simulated_ibnr_distribution(
            payload["simulated_ibnr"],
            payload["base_reserve"],
            payload["var_995"],
            payload["rm_required_005"],
        ),
        use_container_width=True,
    )


def _clear_upload_session() -> None:
    st.session_state.upload_name = None
    st.session_state.upload_df = None
    st.session_state.upload_report = None
    st.session_state.triangle_records = []


def _on_immediate_file_upload() -> None:
    """file_uploader on_change — load on pick, clear session when user removes the file."""
    uploaded = st.session_state.get(_DATA_FILE_KEY)
    if uploaded is not None:
        _handle_uploaded_file(uploaded)
    else:
        _clear_upload_session()


def _handle_uploaded_file(uploaded: object) -> None:
    if uploaded is not None:
        raw = uploaded.getvalue()
        name = uploaded.name
        df_load, load_err = load_dataframe_from_upload(name, raw)
        st.session_state.upload_name = name
        if load_err or df_load is None:
            st.session_state.upload_df = None
            st.session_state.upload_report = {"valid": False, "errors": [load_err or "Load failed"]}
            st.session_state.triangle_records = []
            st.sidebar.error(load_err or "Load failed")
        else:
            st.session_state.upload_df = df_load
            records, report = parse_transaction_dataframe(df_load)
            st.session_state.upload_report = report.model_dump()
            if report.valid:
                st.session_state.triangle_records = records
                st.sidebar.success("File uploaded.")
            else:
                st.session_state.triangle_records = []
                st.sidebar.warning("Automatic import failed. Check column layout in your file.")
                for err in report.errors:
                    st.sidebar.error(err)


def _render_advanced_import_settings() -> None:
    df = st.session_state.upload_df
    if df is None or df.empty:
        return
    with st.expander("Import settings", expanded=False):
        inferred = infer_column_mapping(df)
        col_list = [str(c) for c in df.columns]
        layout = st.selectbox(
            "Table format",
            options=("auto", "long", "wide"),
            format_func=lambda x: {"auto": "Auto", "long": "Long rows", "wide": "Wide triangle"}.get(
                str(x), x
            ),
            index=0,
            key="adv_layout",
        )
        values_mode = st.selectbox(
            "Payment values",
            options=("auto", "incremental", "cumulative"),
            format_func=lambda x: {
                "auto": "Auto",
                "incremental": "Incremental (sum to cumulative)",
                "cumulative": "Already cumulative",
            }.get(str(x), x),
            key="adv_values_mode",
            help="Auto detects whether amounts are cumulative or incremental.",
        )
        col_options = [_AUTO] + col_list
        if layout == "wide":
            default_ay = inferred.get("accident_year") or col_list[0]
            try:
                ay_ix = col_options.index(default_ay) if default_ay in col_list else 1
            except ValueError:
                ay_ix = 1
            ay_pick = st.selectbox(
                "Year column",
                options=col_options,
                index=ay_ix if ay_ix < len(col_options) else 1,
                key="adv_ay_wide",
            )
            column_map = {"accident_year": _col_choice(ay_pick), "development_lag": None, "paid_amount": None}
        else:
            ay_def = inferred.get("accident_year")
            lag_def = inferred.get("development_lag")
            amt_def = inferred.get("paid_amount")

            def _ix(default: str | None) -> int:
                if not default or default not in col_list:
                    return 0
                return col_options.index(default)

            ay_pick = st.selectbox(
                "Accident year column", options=col_options, index=_ix(ay_def), key="adv_ay_long"
            )
            lag_pick = st.selectbox(
                "Development period column", options=col_options, index=_ix(lag_def), key="adv_lag_long"
            )
            amt_pick = st.selectbox("Amount column", options=col_options, index=_ix(amt_def), key="adv_amt_long")
            column_map = {
                "accident_year": _col_choice(ay_pick),
                "development_lag": _col_choice(lag_pick),
                "paid_amount": _col_choice(amt_pick),
            }
        if st.button("Apply and validate", type="secondary", key="adv_apply"):
            records, report = parse_transaction_dataframe(
                df,
                column_map=column_map,
                layout=layout,  # type: ignore[arg-type]
                values_mode=values_mode,  # type: ignore[arg-type]
            )
            st.session_state.upload_report = report.model_dump()
            if report.valid:
                st.session_state.triangle_records = records
                st.success("Settings applied.")
            else:
                st.session_state.triangle_records = []
                for err in report.errors:
                    st.error(err)


def _render_sidebar_session_status() -> None:
    st.subheader("Session")
    name = st.session_state.upload_name or "—"
    st.caption(f"**File:** {name}")
    df = st.session_state.upload_df
    if df is not None and not df.empty:
        st.caption(f"**Sheet size:** {len(df)} rows × {len(df.columns)} columns")
        cols_preview = ", ".join(str(c) for c in list(df.columns)[:6])
        if len(df.columns) > 6:
            cols_preview += ", …"
        st.caption(f"**Columns:** {cols_preview}")
    report = st.session_state.upload_report or {}
    valid = bool(report.get("valid"))
    n_records = len(st.session_state.triangle_records or [])
    st.caption(f"**Triangle valid:** {'Yes' if valid else 'No'}")
    st.caption(f"**Triangle records:** {n_records}")
    ay_range = _accident_year_range(st.session_state.triangle_records or [])
    if ay_range:
        st.caption(f"**Accident years:** {ay_range}")
    for err in (report.get("errors") or [])[:3]:
        st.caption(f"✗ {err}")


def _render_sidebar_data_panels() -> None:
    """Upload, session, and optional clear chat."""
    with st.sidebar:
        st.header("Menu")
        st.caption("Attach your data file here")
        picked_file = st.file_uploader(
            "CSV or Excel",
            type=["csv", "xlsx", "xls"],
            key=_DATA_FILE_KEY,
            on_change=_on_immediate_file_upload,
            label_visibility="collapsed",
        )
        if picked_file is None:
            if _has_uploaded_file():
                _clear_upload_session()
        elif st.session_state.get("upload_name") != getattr(picked_file, "name", None):
            _handle_uploaded_file(picked_file)

        if _has_uploaded_file():
            _render_sidebar_session_status()

        if _chat_has_activity():
            if st.button("Clear chat", type="secondary", width="stretch"):
                st.session_state.messages = [{"role": "assistant", "content": _welcome_message()}]
                st.session_state.last_result = None
                st.session_state.result_panel = None
                st.rerun()


def _process_user_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    lang_pref = _language_pref()

    valid_tri = _triangle_valid()
    has_result = st.session_state.last_result is not None
    intent = classify_chat_intent(
        prompt,
        has_valid_triangle=valid_tri,
        has_previous_result=has_result,
    )

    if intent == ChatIntent.NO_TRIANGLE:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_no_triangle_loaded(prompt, language_pref=lang_pref),
            }
        )
    elif intent == ChatIntent.TRIVIAL:
        if has_result:
            st.session_state.messages.append(
                {"role": "assistant", "content": response_thanks(prompt, language_pref=lang_pref)}
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_trivial_without_run(prompt, language_pref=lang_pref),
                }
            )
    elif intent == ChatIntent.WAIT_FOR_RUN:
        st.session_state.messages.append(
            {"role": "assistant", "content": response_wait_for_run(prompt, language_pref=lang_pref)}
        )
    elif intent == ChatIntent.FULL_RUN:
        with st.spinner("Running the full calculation…"):
            result = _execute_full_run(prompt)
        st.session_state.messages.append({"role": "assistant", "content": result.narrative})
    elif has_result:
        ctx_json = compact_context_for_followup(st.session_state.last_result.model_dump(mode="json"))
        last_narrative = st.session_state.last_result.narrative or ""
        focus = infer_followup_focus(prompt)
        with st.spinner("Preparing the follow-up answer…"):
            reply, _meta = build_followup_llm_response(
                prompt,
                ctx_json,
                last_narrative=last_narrative,
                focus=focus,
            )
        st.session_state.messages.append({"role": "assistant", "content": reply or ""})
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": response_wait_for_run(prompt, language_pref=lang_pref)}
        )


def _parse_chat_value(chat_value: object) -> tuple[str, list[object]]:
    if chat_value is None:
        return "", []
    if isinstance(chat_value, str):
        return chat_value.strip(), []
    text = ""
    files: list[object] = []
    if hasattr(chat_value, "__getitem__"):
        try:
            text = str(chat_value["text"] or "").strip()
        except (KeyError, TypeError):
            text = str(getattr(chat_value, "text", "") or "").strip()
        try:
            if "files" in chat_value:
                files = list(chat_value["files"] or [])
        except (KeyError, TypeError):
            pass
    if not files:
        text = str(getattr(chat_value, "text", text) or "").strip()
        if getattr(chat_value, "files", None):
            files = list(chat_value.files)
    if isinstance(chat_value, dict):
        text = str(chat_value.get("text", text) or "").strip()
        files = list(chat_value.get("files", files) or [])
    return text, files


def _mark_chat_submitted() -> None:
    """Runs before the script body; st.rerun() in callbacks is a no-op in Streamlit."""
    st.session_state["_pending_chat_submit"] = True


def _consume_chat_submission(chat_value: object | None) -> bool:
    """Process text from chat_input Send. Returns True if a full rerun is needed."""
    if not st.session_state.pop("_pending_chat_submit", False) and chat_value is None:
        return False

    raw = chat_value
    if raw is None:
        raw = st.session_state.get(_CHAT_INPUT_KEY)
    if raw is None:
        return False

    prompt, _attached_files = _parse_chat_value(raw)
    if not prompt:
        return False

    _process_user_prompt(prompt)
    return True


st.set_page_config(page_title="Actuarial Copilot", layout="wide")
_init_session()

st.title("Actuarial Copilot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(format_actuarial_notation_html(msg["content"]), unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

if st.session_state.last_result is not None:
    _render_result_actions()
    if st.session_state.result_panel == "tables":
        _render_tables(st.session_state.last_result)
    elif st.session_state.result_panel == "charts":
        _render_charts(st.session_state.last_result)

chat_value = st.chat_input(
    "Ask for a calculation or a follow-up question…",
    key=_CHAT_INPUT_KEY,
    on_submit=_mark_chat_submitted,
)

chat_needs_rerun = _consume_chat_submission(chat_value)

_render_sidebar_data_panels()

if chat_needs_rerun:
    st.rerun()
