"""Typed models for agent orchestration and outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class LifecycleEvent(BaseModel):
    run_id: str
    agent_name: str
    status: AgentStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    user_prompt: str
    scenario: str = "base"
    triangle_size: int = 8


class ValidationReport(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)


class MockActuarialOutput(BaseModel):
    triangle_records: list[dict[str, Any]]
    factors: list[float]
    cdf: list[float] = Field(default_factory=list)
    latest_by_ay: dict[str, float] = Field(default_factory=dict)
    ultimate_by_ay: dict[str, float] = Field(default_factory=dict)
    ibnr_by_ay: dict[str, float] = Field(default_factory=dict)
    reserve_by_ay: dict[str, float]
    total_reserve: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    run_id: str
    narrative: str
    tables: dict[str, list[dict[str, Any]]]
    traces: list[LifecycleEvent]
    chart_payload: dict[str, Any]
    artifacts: dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    request: RunRequest
    intermediate: dict[str, Any] = Field(default_factory=dict)
    events: list[LifecycleEvent] = Field(default_factory=list)

