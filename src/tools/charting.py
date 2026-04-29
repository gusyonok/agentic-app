"""Chart tools that convert outputs to Plotly figures."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_triangle_heatmap(triangle_records: list[dict]) -> go.Figure:
    df = pd.DataFrame(triangle_records)
    pivot = df.pivot_table(
        index="accident_year",
        columns="development_period",
        values="value",
        aggfunc="sum",
    )
    fig = px.imshow(pivot, aspect="auto", title="Cumulative Triangle Heatmap")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_development_factors_chart(factors: list[float]) -> go.Figure:
    x_vals = [f"Age-to-Age {i + 1}" for i in range(len(factors))]
    fig = go.Figure(data=[go.Bar(x=x_vals, y=factors)])
    fig.update_layout(title="Development Factors", xaxis_title="Factor", yaxis_title="Value")
    return fig


def build_reserve_by_ay_chart(reserve_by_ay: dict[str, float]) -> go.Figure:
    items = sorted(reserve_by_ay.items(), key=lambda item: item[0])
    years = [k for k, _ in items]
    reserves = [v for _, v in items]
    fig = go.Figure(data=[go.Scatter(x=years, y=reserves, mode="lines+markers")])
    fig.update_layout(
        title="Reserve by Accident Year",
        xaxis_title="Accident Year",
        yaxis_title="Reserve",
    )
    return fig


def build_observed_vs_ultimate_chart(
    latest_by_ay: dict[str, float], ultimate_by_ay: dict[str, float]
) -> go.Figure:
    years = sorted(latest_by_ay.keys())
    latest = [latest_by_ay[y] for y in years]
    ultimates = [ultimate_by_ay[y] for y in years]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Latest Observed", x=years, y=latest))
    fig.add_trace(go.Bar(name="Projected Ultimate", x=years, y=ultimates))
    fig.update_layout(
        barmode="group",
        title="Observed vs Projected Ultimate by AY",
        xaxis_title="Accident Year",
        yaxis_title="Amount",
    )
    return fig


def build_ibnr_waterfall(ibnr_by_ay: dict[str, float]) -> go.Figure:
    items = sorted(ibnr_by_ay.items(), key=lambda item: item[0])
    years = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(data=[go.Bar(x=years, y=values)])
    fig.update_layout(
        title="IBNR Contribution by AY",
        xaxis_title="Accident Year",
        yaxis_title="IBNR",
    )
    return fig

