"""Type definitions for Kiosk Agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict

from .llm.base import ModelAction


class HistoryEntry(TypedDict, total=False):
    """Single step in agent execution history."""
    
    iteration: int
    payload: Dict[str, Any]
    thought: Optional[str]
    adb_commands: List  # ['adb', 'shell', 'input', 'touchscreen', 'tap', '538', '2233']
    status: str
    pre_action_path: Optional[Path]
    post_action_path: Optional[Path]
    difference: Optional[float]
    progress: bool
    screen_id: Optional[str]  # Visual fingerprint of the screen


class AgentState(TypedDict, total=False):
    """LangGraph workflow state definition."""
    
    instruction: str
    iteration: int
    model_action: Optional[ModelAction]
    raw_response: Optional[str]
    payload: Dict[str, Any]
    thought: Optional[str]
    
    status: str
    route: Literal["loop", "analyze", "human", "end", "backtrack"]
    pre_action_path: Optional[Path]
    post_action_path: Optional[Path]
    analysis: Optional[str]
    history: List[HistoryEntry]
    difference: Optional[float]
    progress: Optional[bool]
    last_adb_commands: List
    
    # Advanced backtracking & structure tracking
    backtrack_target_index: Optional[int]  # Index in history to return to
    last_iteration_id: int  # The 'parent' iteration for the current step
    current_screen_id: Optional[str]
    
    # Planning Mode fields
    plan: List[str]  # Generated plan steps
    plan_step_index: int  # Current step being executed (0-based)
    unknown_entities: List[str]  # Detected unknown concepts/entities
    search_context: str  # Web search results for context enrichment
    planning_complete: bool  # Whether planning phase is done
    original_instruction: str  # Original user instruction before enrichment


@dataclass
class AgentStepResult:
    """Result of a single agent step."""
    
    screenshot_path: Optional[Path]
    raw_response: str
    thought: Optional[str]
    payload: Dict[str, Any]
    adb_commands: List[Sequence[str]]
    status: str
    analysis: Optional[str] = None
    history: List[HistoryEntry] | None = None


@dataclass
class AgentStreamEvent:
    """Event emitted during agent streaming."""
    
    stage: str
    state: AgentState
    message: str
