"""Configuration module for Kiosk Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .prompts.system import VLM_GEMINI_SYSTEM_PROMPT


def _get_screenshots_dir() -> Path:
    """Get screenshots directory from env or default."""
    env_dir = os.getenv("SCREENSHOTS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[2] / "screenshots"


@dataclass
class ScreenshotConfig:
    """Configuration for how screenshots are captured from the kiosk device."""

    adb_path: str = field(default_factory=lambda: os.getenv("ADB_PATH", "adb"))
    device_id: Optional[str] = field(default_factory=lambda: os.getenv("DEVICE_ID"))
    output_dir: Path = field(default_factory=_get_screenshots_dir)
    keep_last_n: int = field(default_factory=lambda: int(os.getenv("KEEP_LAST_N", "10")))


@dataclass
class ModelConfig:
    """Configuration shared by all model providers."""

    provider: Literal["chatgpt", "gemini", "local_vllm"] = field(
        default_factory=lambda: os.getenv("MODEL_PROVIDER", "gemini")
    )
    output_schema: Literal["standard", "planning"] = field(
        default_factory=lambda: os.getenv("MODEL_OUTPUT_SCHEMA", "standard")
    )
    system_prompt: str = VLM_GEMINI_SYSTEM_PROMPT
    temperature: float = field(default_factory=lambda: float(os.getenv("MODEL_TEMPERATURE", "0.1")))
    top_p: float = field(default_factory=lambda: float(os.getenv("MODEL_TOP_P", "0.3")))
    
    # ChatGPT / OpenAI
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_base: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_BASE"))
    
    # Gemini
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    )
    
    # Local vLLM (OpenAI-compatible HTTP server)
    vllm_base_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000"))
    vllm_model_name: str = field(default_factory=lambda: os.getenv("VLLM_MODEL_NAME", "AgentCPM-GUI"))
    vllm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("VLLM_API_KEY"))


@dataclass
class ADBConfig:
    """Configuration for translating structured actions into concrete adb commands."""
    
    adb_path: str = field(default_factory=lambda: os.getenv("ADB_PATH", "adb"))
    device_id: Optional[str] = field(default_factory=lambda: os.getenv("DEVICE_ID"))
    default_swipe_duration_ms: int = field(default_factory=lambda: int(os.getenv("SWIPE_MS", "300")))
    steps: int = 1
    screenshot_abs_path: str = field(
        default_factory=lambda: os.getenv("SCREENSHOTS_DIR") or str(Path(__file__).resolve().parents[2] / "screenshots")
    )


@dataclass
class PlanningConfig:
    """Configuration for Planning Mode (task decomposition and web search)."""
    
    enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_PLANNING_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
    )
    web_search_provider: Literal["tavily", "duckduckgo"] = field(
        default_factory=lambda: os.getenv("PLANNING_SEARCH_PROVIDER", "tavily")
    )
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )
    max_search_results: int = field(
        default_factory=lambda: int(os.getenv("PLANNING_MAX_SEARCH_RESULTS", "5"))
    )
    max_plan_steps: int = field(
        default_factory=lambda: int(os.getenv("PLANNING_MAX_STEPS", "10"))
    )


@dataclass
class AgentConfig:
    """Top level configuration bundle."""
    
    screenshot: ScreenshotConfig = field(default_factory=ScreenshotConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adb: ADBConfig = field(default_factory=ADBConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    
    # Runtime parameters (can be overridden via env)
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS") or os.getenv("AGENT_MAX_ITERATIONS") or "20")
    )
    progress_threshold: float = field(
        default_factory=lambda: float(os.getenv("AGENT_PROGRESS_THRESHOLD") or os.getenv("PROGRESS_THRESHOLD") or "0.02")
    )
    recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("AGENT_RECURSION_LIMIT", "100"))
    )
    tts_keep_last_n: int = field(
        default_factory=lambda: int(os.getenv("AGENT_TTS_KEEP_LAST_N", "5"))
    )
    tts_thought_enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_TTS_THOUGHT", "0").lower() in {"1", "true", "yes", "on"}
    )
    tts_thought_max_chars: int = field(
        default_factory=lambda: int(os.getenv("AGENT_TTS_THOUGHT_MAX_CHARS", "320"))
    )
    tts_output_dir: str = field(
        default_factory=lambda: os.getenv("TTS_OUTPUT_DIR") or str(Path(__file__).resolve().parents[2] / "screenshots" / "tts_output")
    )
