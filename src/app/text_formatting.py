"""Text formatting helpers for chat display."""

from __future__ import annotations

import re


def format_actuarial_notation_html(text: str) -> str:
    """Render actuarial labels and safe HTML for Streamlit markdown (unsafe_allow_html=True)."""
    formatted = text
    formatted = re.sub(r"\bP[_\s-]?def\b", "P<sub>def</sub>", formatted, flags=re.IGNORECASE)
    formatted = re.sub(r"\bTVaR\s*99[.,]5%?\b", "TVaR<sub>99.5%</sub>", formatted, flags=re.IGNORECASE)
    formatted = re.sub(r"\bVaR\s*99[.,]5%?\b", "VaR<sub>99.5%</sub>", formatted, flags=re.IGNORECASE)
    formatted = re.sub(r"\bVaR\s*95%?\b", "VaR<sub>95%</sub>", formatted, flags=re.IGNORECASE)
    # LLM sometimes emits broken markdown (e.g. "4,090,203.60***" or lone "*" before words).
    formatted = re.sub(r"\*\*([^*\n]+?)\*\*+", r"<strong>\1</strong>", formatted)
    formatted = re.sub(r"(?<!\*)\*(?!\*)", "", formatted)
    return formatted
