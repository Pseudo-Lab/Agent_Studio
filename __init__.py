from __future__ import annotations

from .src.config import AgentConfig, ModelConfig, ScreenshotConfig, ADBConfig

__all__ = [
    "ADBConfig",
    "AgentConfig",
    "ModelConfig",
    "ScreenshotConfig",
    "KioskAgent",
]


def __getattr__(name: str):
    # Avoid importing heavy runtime deps (langgraph/langsmith/etc.) on package import.
    if name == "KioskAgent":
        from .src.langgraph_kiosk_agent import KioskAgent  # noqa: WPS433 (runtime import)

        return KioskAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
