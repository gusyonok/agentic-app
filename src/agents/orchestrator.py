"""Orchestrator agent coordinating child agents and lifecycle tracking."""

from __future__ import annotations

from typing import Any

from agents.children import (
    CalculationAgent,
    DataPrepAgent,
    ExplanationAgent,
    IntakeAgent,
    MethodSelectionAgent,
)
from core.lifecycle import mark_completed, mark_failed, mark_running
from core.models import AgentContext, RunRequest, RunResult


def _ldf_table_rows(calculation: dict) -> list[dict[str, object]]:
    factors = calculation["factors"]
    diag = calculation.get("diagnostics") or {}
    from_dev = diag.get("ldf_from_dev")
    to_dev = diag.get("ldf_to_dev")
    if from_dev and to_dev and len(from_dev) == len(factors) == len(to_dev):
        return [
            {"from_period": fr, "to_period": to, "ldf": v}
            for fr, to, v in zip(from_dev, to_dev, factors)
        ]
    return [
        {"from_period": i + 1, "to_period": i + 2, "ldf": v}
        for i, v in enumerate(factors)
    ]


class OrchestratorAgent:
    name = "OrchestratorAgent"

    def __init__(self) -> None:
        self.intake = IntakeAgent()
        self.data_prep = DataPrepAgent()
        self.method_select = MethodSelectionAgent()
        self.calculate = CalculationAgent()
        self.explain = ExplanationAgent()

    def _run_child(self, ctx: AgentContext, key: str, child: Any) -> None:
        mark_running(ctx, child.name, "started")
        try:
            ctx.intermediate[key] = child.run(ctx)
            mark_completed(ctx, child.name, "completed")
        except Exception as exc:  # pragma: no cover
            mark_failed(ctx, child.name, str(exc))
            raise

    def run(self, request: RunRequest) -> RunResult:
        ctx = AgentContext(request=request)
        mark_running(ctx, self.name, "orchestration started")

        self._run_child(ctx, "intake", self.intake)
        self._run_child(ctx, "data_prep", self.data_prep)
        if not ctx.intermediate["data_prep"]["validation"]["valid"]:
            mark_failed(
                ctx,
                self.name,
                "validation failed",
                {"errors": ctx.intermediate["data_prep"]["validation"]["errors"]},
            )
            return RunResult(
                run_id=ctx.run_id,
                narrative="Validation failed. Please fix input data and rerun.",
                tables={"validation": [ctx.intermediate["data_prep"]["validation"]]},
                traces=ctx.events,
                chart_payload={},
                artifacts={"intermediate": ctx.intermediate},
            )
        self._run_child(ctx, "method", self.method_select)
        self._run_child(ctx, "calculation", self.calculate)
        self._run_child(ctx, "explanation", self.explain)

        result = RunResult(
            run_id=ctx.run_id,
            narrative=ctx.intermediate["explanation"]["narrative"],
            tables={
                "triangle": ctx.intermediate["calculation"]["triangle_records"],
                "reserve_by_ay": [
                    {"accident_year": k, "reserve": v}
                    for k, v in ctx.intermediate["calculation"]["reserve_by_ay"].items()
                ],
                "ldf": _ldf_table_rows(ctx.intermediate["calculation"]),
                "cdf": [
                    {"development_period": i + 1, "cdf": v}
                    for i, v in enumerate(ctx.intermediate["calculation"]["cdf"])
                ],
                "ultimate_ibnr": [
                    {
                        "accident_year": ay,
                        "latest": ctx.intermediate["calculation"]["latest_by_ay"][ay],
                        "ultimate": ctx.intermediate["calculation"]["ultimate_by_ay"][ay],
                        "ibnr": ctx.intermediate["calculation"]["ibnr_by_ay"][ay],
                    }
                    for ay in ctx.intermediate["calculation"]["latest_by_ay"]
                ],
                "risk_metrics": [
                    {
                        "metric": key,
                        "value": value,
                    }
                    for key, value in ctx.intermediate["calculation"]["risk_metrics"].items()
                    if key
                    not in {
                        "simulated_ibnr",
                        "stability_batch_vars",
                        "mack_se_by_ay",
                        "mack_process_variances",
                        "default_probability_definition",
                        "deficit_probability_definition",
                    }
                ],
                "stability_batch_var995": [
                    {"run": idx + 1, "var_995": value}
                    for idx, value in enumerate(
                        ctx.intermediate["calculation"]["risk_metrics"]["stability_batch_vars"]
                    )
                ],
                "validation": [ctx.intermediate["data_prep"]["validation"]],
            },
            traces=ctx.events,
            chart_payload={
                "triangle_records": ctx.intermediate["calculation"]["triangle_records"],
                "factors": ctx.intermediate["calculation"]["factors"],
                "reserve_by_ay": ctx.intermediate["calculation"]["reserve_by_ay"],
                "ultimate_by_ay": ctx.intermediate["calculation"]["ultimate_by_ay"],
                "latest_by_ay": ctx.intermediate["calculation"]["latest_by_ay"],
                "ibnr_by_ay": ctx.intermediate["calculation"]["ibnr_by_ay"],
                "simulated_ibnr": ctx.intermediate["calculation"]["risk_metrics"]["simulated_ibnr"],
                "base_reserve": ctx.intermediate["calculation"]["risk_metrics"]["base_reserve"],
                "var_995": ctx.intermediate["calculation"]["risk_metrics"]["var_995"],
                "rm_required_005": ctx.intermediate["calculation"]["risk_metrics"]["rm_required_005"],
            },
            artifacts={
                "intermediate": ctx.intermediate,
                "llm": ctx.intermediate["explanation"].get("llm_meta", {}),
            },
        )
        mark_completed(ctx, self.name, "orchestration completed")
        result.traces = ctx.events
        return result

