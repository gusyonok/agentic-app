from agents.orchestrator import OrchestratorAgent
from core.models import RunRequest


def test_orchestrator_full_mock_run():
    orchestrator = OrchestratorAgent()
    result = orchestrator.run(
        RunRequest(
            user_prompt="Estimate reserve and show diagnostics.",
            scenario="base",
            triangle_size=8,
        )
    )

    assert result.run_id
    assert "ibnr reserve" in result.narrative.lower()
    assert len(result.tables["triangle"]) > 0
    assert "reserve_by_ay" in result.chart_payload
    assert "ldf" in result.tables
    assert "ultimate_ibnr" in result.tables
    assert result.artifacts["intermediate"]["method"]["method"] == "deterministic_chain_ladder"
    assert len(result.traces) > 0
