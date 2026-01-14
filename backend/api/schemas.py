"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StartRequest(BaseModel):
    """Request to start agent execution."""
    
    instruction: str
    thread_id: Optional[str] = None
    model: Optional[str] = None  # 'gemini-flash', 'gemini-3-preview'
    enable_planning: bool = False  # Enable Planning Mode (task decomposition + web search)


class RespondRequest(BaseModel):
    """Request to respond to agent's question."""
    
    thread_id: str
    response: str


class SnapshotResponse(BaseModel):
    """Agent state snapshot."""
    
    status: str
    stage: Optional[str] = None
    iteration: int = 0
    thought: Optional[str] = None
    action: Optional[str] = None
    box_2d: Optional[List[int]] = None
    adb_commands: List[str] = []
    interrupt: Optional[Dict[str, Any]] = None
    progress: Optional[bool] = None
    difference: Optional[float] = None
    # Planning Mode fields
    plan: Optional[List[str]] = None
    plan_step_index: Optional[int] = None
    planning_complete: Optional[bool] = None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    active_sessions: int = 0
    tts_available: bool = False
