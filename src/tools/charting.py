"""Chart tools that convert outputs to Plotly figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_triangle_heatmap(triangle_records: list[dict]) -> go.Figure:
    df = pd.DataFrame(triangle_records)
    pivot = df.pivot_table(
        index="accident_year",
        columns="development_period",
        values="value",
        aggfunc="sum",
    )
    # Nested Python floats — Plotly 6 + narwhals may coerce ndarray/Series via a fragile path.
    z = np.asarray(pivot.values, dtype=float).tolist()
    x = [str(c) for c in pivot.columns]
    y = [str(i) for i in pivot.index]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Blues",
            colorbar=dict(title="Cumulative Paid Amount"),
        )
    )
    fig.update_layout(
        title="History of Accumulated Payments",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Development Period",
        yaxis_title="Accident Year",
        yaxis_autorange="reversed",
    )
    return fig


def build_development_factors_chart(factors: list[float]) -> go.Figure:
    x_vals = [f"{i}→{i + 1}" for i in range(len(factors))]
    fig = go.Figure(data=[go.Bar(x=x_vals, y=[float(f) for f in factors])])
    fig.update_layout(
        title="Payout Growth Rate Over Time",
        xaxis_title="Development Period",
        yaxis_title="Development Factor",
        showlegend=False,
    )
    return fig


def build_reserve_by_ay_chart(reserve_by_ay: dict[str, float]) -> go.Figure:
    items = sorted(reserve_by_ay.items(), key=lambda item: item[0])
    years = [k for k, _ in items]
    reserves = [float(v) for _, v in items]
    fig = go.Figure(data=[go.Scatter(x=years, y=reserves, mode="lines+markers")])
    fig.update_layout(
        title="Expected vs. Actual Payments by Accident Year",
        xaxis_title="Accident Year",
        yaxis_title="Amount",
    )
    return fig


def build_observed_vs_ultimate_chart(
    latest_by_ay: dict[str, float], ultimate_by_ay: dict[str, float]
) -> go.Figure:
    years = sorted(latest_by_ay.keys())
    latest = [float(latest_by_ay[y]) for y in years]
    ultimates = [float(ultimate_by_ay[y]) for y in years]
    future_reserve = [float(ultimate_by_ay[y]) - float(latest_by_ay[y]) for y in years]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Already Paid", x=years, y=latest))
    fig.add_trace(go.Bar(name="Future Reserve", x=years, y=future_reserve))
    fig.update_layout(
        barmode="stack",
        title="Expected vs. Actual Payments by Accident Year",
        xaxis_title="Accident Year",
        yaxis_title="Amount",
    )
    return fig


def build_ibnr_waterfall(ibnr_by_ay: dict[str, float]) -> go.Figure:
    items = sorted(ibnr_by_ay.items(), key=lambda item: item[0])
    years = [k for k, _ in items]
    values = [float(v) for _, v in items]
    fig = go.Figure(data=[go.Bar(x=years, y=values)])
    fig.update_layout(
        title="Expected vs. Actual Payments by Accident Year",
        xaxis_title="Accident Year",
        yaxis_title="Amount",
    )
    return fig


def build_simulated_ibnr_distribution(
    simulated_ibnr: list[float], base_reserve: float, var_995: float, rm_required: float
) -> go.Figure:
    if not simulated_ibnr:
        fig = go.Figure()
        fig.update_layout(
            title="Risk Distribution and Stress Scenarios",
            xaxis_title="Simulated Reserve Amount",
            yaxis_title="Frequency",
        )
        return fig
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=[float(x) for x in simulated_ibnr],
            nbinsx=60,
            name="Simulated Reserve Amount",
            opacity=0.75,
            showlegend=False,
        )
    )
    y_top = max(1, len(simulated_ibnr) // 10)
    fig.add_trace(
        go.Scatter(
            x=[base_reserve, base_reserve],
            y=[0, y_top],
            mode="lines",
            name="Base Reserve",
            line=dict(color="#f2c94c", dash="dash", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[var_995, var_995],
            y=[0, y_top],
            mode="lines",
            name="Stress Scenario 99.5%",
            line=dict(color="#eb5757", dash="dot", width=3),
        )
    )
    fig.update_layout(
        title="Risk Distribution and Stress Scenarios",
        xaxis_title="Simulated Reserve Amount",
        yaxis_title="Frequency",
    )
    return fig


def build_simulated_ibnr_cdf(
    simulated_ibnr: list[float], base_reserve: float, threshold: float
) -> go.Figure:
    if not simulated_ibnr:
        fig = go.Figure()
        fig.update_layout(
            title="Risk Distribution and Stress Scenarios",
            xaxis_title="Simulated Reserve Amount",
            yaxis_title="Frequency",
        )
        return fig
    xs = sorted(float(x) for x in simulated_ibnr)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    fig = go.Figure(data=[go.Scatter(x=xs, y=ys, mode="lines", name="Simulated Reserve Amount")])
    fig.add_vline(x=base_reserve, line_dash="dash", annotation_text="Base Reserve")
    fig.add_vline(x=threshold, line_dash="dashdot", annotation_text="Stress Scenario 99.5%")
    fig.update_layout(
        title="Risk Distribution and Stress Scenarios",
        xaxis_title="Simulated Reserve Amount",
        yaxis_title="Frequency",
    )
    return fig

