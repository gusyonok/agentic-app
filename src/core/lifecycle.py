"""Agent lifecycle tracking and event creation."""

from __future__ import annotations

from typing import Any

from core.models import AgentContext, AgentStatus, LifecycleEvent


def log_event(
    ctx: AgentContext,
    agent_name: str,
    status: AgentStatus,
    message: str = "",
    payload: dict[str, Any] | None = None,
) -> LifecycleEvent:
    event = LifecycleEvent(
        run_id=ctx.run_id,
        agent_name=agent_name,
        status=status,
        message=message,
        payload=payload or {},
    )
    ctx.events.append(event)
    return event


def mark_running(ctx: AgentContext, agent_name: str, message: str = "") -> LifecycleEvent:
    return log_event(ctx, agent_name, AgentStatus.running, message)


def mark_completed(
    ctx: AgentContext, agent_name: str, message: str = "", payload: dict[str, Any] | None = None
) -> LifecycleEvent:
    return log_event(ctx, agent_name, AgentStatus.completed, message, payload)


def mark_failed(
    ctx: AgentContext, agent_name: str, message: str, payload: dict[str, Any] | None = None
) -> LifecycleEvent:
    return log_event(ctx, agent_name, AgentStatus.failed, message, payload)

