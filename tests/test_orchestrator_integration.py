from agents.orchestrator import OrchestratorAgent
from core.models import RunRequest
from tools.reserving import ReservingParams, load_example_triangle


def test_orchestrator_full_mock_run():
    triangle = load_example_triangle(ReservingParams(scenario="base", triangle_size=8))
    orchestrator = OrchestratorAgent()
    result = orchestrator.run(
        RunRequest(
            user_prompt="Estimate reserve and show diagnostics.",
            simulations=4000,
            triangle_records=triangle,
        )
    )

    assert result.run_id
    low = result.narrative.lower()
    assert (
        "reserve" in low
        or "резерв" in low
        or "ibnr" in low
        or "p_def" in low
        or "deficit" in low
        or "дефіцит" in low
        or "зобов" in low
    )
    assert len(result.tables["triangle"]) > 0
    assert "reserve_by_ay" in result.chart_payload
    assert "simulated_ibnr" in result.chart_payload
    assert "ldf" in result.tables
    assert "ultimate_ibnr" in result.tables
    assert "risk_metrics" in result.tables
    assert result.artifacts["intermediate"]["method"]["method"] == "deterministic_chain_ladder"
    assert len(result.traces) > 0
