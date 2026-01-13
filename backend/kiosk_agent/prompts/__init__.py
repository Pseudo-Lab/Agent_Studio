"""Prompts and schema for VLM interactions."""

from .schema import GUI_OUTPUT, InterruptInfo
from .system import VLM_GEMINI_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# Alias for convenience
SYSTEM_PROMPT = VLM_GEMINI_SYSTEM_PROMPT

__all__ = [
    "GUI_OUTPUT",
    "InterruptInfo",
    "VLM_GEMINI_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
]
