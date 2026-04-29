"""LLM helper for narrative generation."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv


def build_llm_narrative(
    user_prompt: str,
    method: str,
    calc: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()

    if provider != "openai" or not api_key:
        return None, {"llm_used": False, "provider": provider, "reason": "missing_provider_or_key"}

    prompt = (
        "You are an actuarial assistant. Write a concise explanation of the reserving output.\n"
        f"User request: {user_prompt}\n"
        f"Method: {method}\n"
        f"Total IBNR: {calc['total_reserve']}\n"
        f"Top AY IBNR contributors: {sorted(calc['ibnr_by_ay'].items(), key=lambda x: x[1], reverse=True)[:3]}\n"
        "Include assumptions caveat in one sentence."
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=model, input=prompt)
        text = response.output_text.strip()
        if not text:
            return None, {"llm_used": False, "provider": provider, "reason": "empty_response"}
        return text, {"llm_used": True, "provider": provider, "model": model}
    except Exception as exc:  # pragma: no cover
        return None, {"llm_used": False, "provider": provider, "reason": str(exc)}
