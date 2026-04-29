from core.lifecycle import mark_completed, mark_failed, mark_running
from core.models import AgentContext, AgentStatus, RunRequest


def test_lifecycle_status_transitions_recorded():
    ctx = AgentContext(request=RunRequest(user_prompt="test"))

    mark_running(ctx, "A")
    mark_completed(ctx, "A")
    mark_running(ctx, "B")
    mark_failed(ctx, "B", "error")

    statuses = [event.status for event in ctx.events]
    assert statuses == [
        AgentStatus.running,
        AgentStatus.completed,
        AgentStatus.running,
        AgentStatus.failed,
    ]
