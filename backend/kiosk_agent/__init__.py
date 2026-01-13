"""
Kiosk Agent - GUI Automation Agent

A Vision-Language-Action based agent for automating Android kiosk interactions.
"""

from .config import AgentConfig, ADBConfig, ModelConfig, ScreenshotConfig
from .types import AgentState, AgentStepResult, AgentStreamEvent, HistoryEntry
from .core import ADBController, ActionTranslator, AndroidScreenshotter, CapturedScreen
from .llm import BaseModelClient, GeminiClient, ChatGPTClient, LocalVLLMClient, ModelAction
from .frameworks import KioskAgent, BaseAgent
from .utils import get_logger, setup_logging

__version__ = "0.1.0"

__all__ = [
    # Config
    "AgentConfig",
    "ADBConfig", 
    "ModelConfig",
    "ScreenshotConfig",
    # Types
    "AgentState",
    "AgentStepResult",
    "AgentStreamEvent",
    "HistoryEntry",
    # Core
    "ADBController",
    "ActionTranslator",
    "AndroidScreenshotter",
    "CapturedScreen",
    # LLM
    "BaseModelClient",
    "GeminiClient",
    "ChatGPTClient",
    "LocalVLLMClient",
    "ModelAction",
    # Agent
    "KioskAgent",
    "BaseAgent",
    # Utils
    "get_logger",
    "setup_logging",
]
