"""Child agent implementations for actuarial workflow."""

from __future__ import annotations

from agents.llm_client import (
    build_chief_actuary_fallback_narrative,
    build_llm_narrative,
    detect_response_language,
)
from core.models import AgentContext
from tools.reserving import (
    compute_chain_ladder_from_triangle_records,
    compute_risk_metrics,
    risk_metrics_skipped_zero_ibnr,
)
from tools.validation import validate_triangle_records


class IntakeAgent:
    name = "IntakeAgent"

    def run(self, ctx: AgentContext) -> dict:
        return {
            "goal": "chain_ladder_reserving",
            "prompt": ctx.request.user_prompt.strip(),
        }


class DataPrepAgent:
    name = "DataPrepAgent"

    def run(self, ctx: AgentContext) -> dict:
        triangle = list(ctx.request.triangle_records)
        if not triangle:
            return {
                "triangle_records": [],
                "validation": {
                    "valid": False,
                    "errors": [
                        "No transaction data loaded. Upload a CSV or Excel file with columns "
                        "Accident_Year, Development_Lag, Paid_Amount (incremental payments per lag)."
                    ],
                    "warnings": [],
                    "data_quality_flags": [],
                },
            }
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
                "IBNR distribution: Monte Carlo bootstrap of Pearson residuals (ODP) — reserve deficit probability",
                "Default probability: Mack standard error + lognormal CDF (future payments exceed reserve)",
            ],
        }


class CalculationAgent:
    name = "CalculationAgent"

    def run(self, ctx: AgentContext) -> dict:
        triangle = ctx.intermediate["data_prep"]["triangle_records"]
        return compute_chain_ladder_from_triangle_records(triangle_records=triangle).model_dump()


class ExplanationAgent:
    name = "ExplanationAgent"

    def run(self, ctx: AgentContext) -> dict:
        calc = ctx.intermediate["calculation"]
        validation = ctx.intermediate["data_prep"].get("validation", {})
        flags = validation.get("data_quality_flags") or []

        be = float(calc["total_reserve"])
        if be <= 1e-9:
            risk = risk_metrics_skipped_zero_ibnr(be)
        else:
            risk = compute_risk_metrics(
                triangle_records=calc["triangle_records"],
                base_reserve=be,
                simulations=ctx.request.simulations,
            )
        calc["risk_metrics"] = risk

        base = float(risk.get("base_reserve", be))
        p_def = float(risk.get("p_def") or 0.0)
        rm = float(risk.get("rm_required_005") or 0.0)
        var995 = float(risk.get("var_995") or 0.0)
        rm_pct = (rm / base * 100.0) if base > 1e-9 else 0.0
        rel_excess = ((var995 - base) / base * 100.0) if base > 1e-9 else 0.0
        triangle_non_monotonic = any("monotonic" in str(f).lower() for f in flags)
        plateau_bootstrap = (
            not risk.get("risk_analysis_skipped")
            and base > 1e-9
            and 0.40 <= p_def <= 0.58
            and rm_pct < 14.0
            and rel_excess < 14.0
        )
        calc["narrative_hints"] = {
            "plateau_bootstrap_adequacy": plateau_bootstrap,
            "triangle_non_monotonic": triangle_non_monotonic,
        }

        method = ctx.intermediate["method"]
        fallback_narrative = build_chief_actuary_fallback_narrative(calc)
        hints = calc["narrative_hints"]
        prompt_language = detect_response_language(ctx.request.user_prompt)
        if prompt_language == "uk" and be > 1e-9 and (
            hints.get("plateau_bootstrap_adequacy") or hints.get("triangle_non_monotonic")
        ):
            llm_narrative, llm_meta = None, {
                "llm_used": False,
                "reason": "adequacy_profile_deterministic_narrative",
                "narrative_hints": hints,
            }
        else:
            llm_narrative, llm_meta = build_llm_narrative(
                user_prompt=ctx.request.user_prompt,
                method=method["method"],
                calc=calc,
            )
        return {"narrative": llm_narrative or fallback_narrative, "llm_meta": llm_meta}

