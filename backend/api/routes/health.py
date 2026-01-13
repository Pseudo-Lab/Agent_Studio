"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Basic health check."""
    from .agent import agent_instance, sessions
    
    agent = agent_instance()
    
    response = {
        "status": "ok",
        "active_sessions": sessions.count(),
        "tts_available": bool(agent and agent.tts is not None),
    }
    
    return response
