"""Child agent implementations for actuarial workflow."""

from __future__ import annotations

from agents.llm_client import build_llm_narrative
from core.models import AgentContext
from tools.reserving import ReservingParams, compute_chain_ladder_reserving, load_example_triangle
from tools.validation import validate_triangle_records


class IntakeAgent:
    name = "IntakeAgent"

    def run(self, ctx: AgentContext) -> dict:
        return {
            "goal": "chain_ladder_reserving",
            "prompt": ctx.request.user_prompt.strip(),
            "scenario": ctx.request.scenario,
        }


class DataPrepAgent:
    name = "DataPrepAgent"

    def run(self, ctx: AgentContext) -> dict:
        params = ReservingParams(scenario=ctx.request.scenario, triangle_size=ctx.request.triangle_size)
        triangle = load_example_triangle(params)
        validation = validate_triangle_records(triangle)
        return {"triangle_records": triangle, "validation": validation.model_dump()}


class MethodSelectionAgent:
    name = "MethodSelectionAgent"

    def run(self, ctx: AgentContext) -> dict:
        return {
            "method": "deterministic_chain_ladder",
            "assumptions": [
                "Volume-weighted link ratio selection",
                "No explicit tail factor beyond observed development",
            ],
        }


class CalculationAgent:
    name = "CalculationAgent"

    def run(self, ctx: AgentContext) -> dict:
        params = ReservingParams(scenario=ctx.request.scenario, triangle_size=ctx.request.triangle_size)
        return compute_chain_ladder_reserving(params).model_dump()


class ExplanationAgent:
    name = "ExplanationAgent"

    def run(self, ctx: AgentContext) -> dict:
        calc = ctx.intermediate["calculation"]
        method = ctx.intermediate["method"]
        top_ay, top_ibnr = max(calc["ibnr_by_ay"].items(), key=lambda item: item[1])
        fallback_narrative = (
            f"Using {method['method']} in {ctx.request.scenario} scenario, "
            f"the indicated IBNR reserve is {calc['total_reserve']:.2f}. "
            f"The largest AY contribution is {top_ay} with {top_ibnr:.2f}. "
            "Review diagnostics and selected factors before adopting this estimate."
        )
        llm_narrative, llm_meta = build_llm_narrative(
            user_prompt=ctx.request.user_prompt,
            method=method["method"],
            calc=calc,
        )
        return {"narrative": llm_narrative or fallback_narrative, "llm_meta": llm_meta}

