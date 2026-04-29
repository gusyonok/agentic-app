"""Streamlit UI for orchestrator/child-agent actuarial workflow."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from agents.orchestrator import OrchestratorAgent
from agents.strands_runtime import check_strands_available
from core.models import RunRequest
from tools.charting import (
    build_development_factors_chart,
    build_ibnr_waterfall,
    build_observed_vs_ultimate_chart,
    build_reserve_by_ay_chart,
    build_triangle_heatmap,
)


st.set_page_config(page_title="Actuarial Agentic App", layout="wide")
st.title("Actuarial Agentic App")
st.caption("Strands-style orchestrator + child agents with lifecycle visibility and charts.")
availability = check_strands_available()
st.info(f"Strands status: {availability.detail}")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.header("Scenario Controls")
    scenario = st.selectbox("Scenario", options=["base", "optimistic", "stress"], index=0)
    triangle_size = st.slider("Years from real example triangle", min_value=4, max_value=8, value=8, step=1)

prompt = st.text_area(
    "Actuarial request",
    value="Estimate reserve for this portfolio and explain key assumptions.",
    height=100,
)

if st.button("Run Orchestration", type="primary"):
    orchestrator = OrchestratorAgent()
    request = RunRequest(user_prompt=prompt, scenario=scenario, triangle_size=triangle_size)
    st.session_state.last_result = orchestrator.run(request)

result = st.session_state.last_result
if result is not None:
    st.success(f"Run complete: {result.run_id}")
    llm_meta = result.artifacts.get("llm", {})
    if llm_meta.get("llm_used"):
        st.info(f"LLM explanation enabled: {llm_meta.get('provider')} / {llm_meta.get('model')}")
    else:
        st.warning("LLM explanation fallback used. Add OPENAI_API_KEY in .env to enable.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Agent lifecycle")
        lifecycle_df = pd.DataFrame(
            [
                {
                    "agent": e.agent_name,
                    "status": e.status.value,
                    "time_utc": e.timestamp.isoformat(),
                    "message": e.message,
                }
                for e in result.traces
            ]
        )
        st.dataframe(lifecycle_df, width="stretch")
    with col2:
        st.subheader("Run artifact")
        artifact_json = result.model_dump_json(indent=2)
        st.download_button(
            "Download JSON",
            data=artifact_json,
            file_name=f"run_{result.run_id}.json",
            mime="application/json",
        )

    tab_narrative, tab_tables, tab_charts, tab_trace = st.tabs(
        ["Narrative", "Tables", "Charts", "Trace"]
    )
    with tab_narrative:
        st.write(result.narrative)
    with tab_tables:
        st.write("Triangle")
        st.dataframe(pd.DataFrame(result.tables["triangle"]), width="stretch", height=220)
        st.write("Reserve by Accident Year")
        st.dataframe(pd.DataFrame(result.tables["reserve_by_ay"]), width="stretch", height=220)
        if "ldf" in result.tables:
            st.write("Selected LDF")
            st.dataframe(pd.DataFrame(result.tables["ldf"]), width="stretch", height=180)
        if "cdf" in result.tables:
            st.write("CDF")
            st.dataframe(pd.DataFrame(result.tables["cdf"]), width="stretch", height=180)
        if "ultimate_ibnr" in result.tables:
            st.write("Ultimate and IBNR")
            st.dataframe(pd.DataFrame(result.tables["ultimate_ibnr"]), width="stretch", height=220)
        st.write("Validation")
        st.dataframe(pd.DataFrame(result.tables["validation"]), width="stretch")
    with tab_charts:
        payload = result.chart_payload
        if payload:
            st.plotly_chart(build_triangle_heatmap(payload["triangle_records"]), use_container_width=True)
            st.plotly_chart(build_development_factors_chart(payload["factors"]), use_container_width=True)
            st.plotly_chart(build_reserve_by_ay_chart(payload["reserve_by_ay"]), use_container_width=True)
            st.plotly_chart(
                build_observed_vs_ultimate_chart(payload["latest_by_ay"], payload["ultimate_by_ay"]),
                use_container_width=True,
            )
            st.plotly_chart(build_ibnr_waterfall(payload["ibnr_by_ay"]), use_container_width=True)
        else:
            st.warning("No chart payload due to validation failure.")
    with tab_trace:
        st.code(json.dumps([e.model_dump(mode="json") for e in result.traces], indent=2), language="json")
