"""Agent execution endpoints with AG-UI SSE streaming."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from kiosk_agent import AgentConfig
from kiosk_agent.utils import get_logger

from ..schemas import RespondRequest, StartRequest
from ..session import SessionStore
from ..streamer import AgentStreamer

logger = get_logger(__name__)

router = APIRouter()

# Singletons
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("API_MAX_WORKERS", "4")))
sessions = SessionStore(
    max_sessions=int(os.getenv("API_MAX_SESSIONS", "100")),
    idle_ttl_sec=int(os.getenv("API_SESSION_TTL", "3600")),
)
_agent: Optional[Any] = None
_current_model: Optional[str] = None
_current_planning: Optional[bool] = None


def get_agent(model_id: Optional[str] = None, enable_planning: bool = False):
    """
    Get or create agent instance with config from environment.
    
    Agent is recreated when:
    - Model changes
    - Planning mode changes (graph structure differs)
    """
    global _agent, _current_model, _current_planning
    
    # Import here to avoid any module-level issues
    from kiosk_agent.frameworks import KioskAgent as Agent
    
    # Model mapping
    model_map = {
        "gemini-3-preview": "gemini-3-pro-preview",
        "gemini-flash": "gemini-3-flash-preview",
    }
    requested = model_id or os.getenv("DEFAULT_MODEL", "gemini-flash")
    target_model = model_map.get(requested, os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    
    # Recreate agent if model or planning mode changes
    needs_rebuild = (
        _agent is None 
        or _current_model != requested 
        or _current_planning != enable_planning
    )
    
    if needs_rebuild:
        # Create config (will read from env automatically)
        config = AgentConfig()
        
        # Override model if specified
        config.model.gemini_model = target_model
        
        # Validate API key
        if config.model.provider == "gemini" and not config.model.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경변수가 필요합니다.")
        if config.model.provider == "chatgpt" and not config.model.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
        
        # Create agent with specified planning mode
        enable_tts = os.getenv("AGENT_TTS_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
        _agent = Agent(config, enable_tts=enable_tts, enable_planning=enable_planning)
        _current_model = requested
        _current_planning = enable_planning
        
        mode_str = "Planning Mode" if enable_planning else "Standard Mode"
        logger.info(f"Agent built: provider={config.model.provider}, model={target_model}, mode={mode_str}")
    
    return _agent


def agent_instance() -> Optional[Any]:
    """Get current agent instance (for health check)."""
    return _agent


# Create streamer
streamer = AgentStreamer(
    executor=_executor,
    sessions=sessions,
    get_agent=get_agent,
)


@router.post("/start")
async def agent_start(req: StartRequest):
    """Start agent execution with SSE streaming."""
    try:
        return streamer.start(req)
    except Exception as e:
        import traceback
        error_msg = str(e)
        logger.error(f"Start failed: {error_msg}\n{traceback.format_exc()}")
        
        # ADB connection error detection
        if "adb" in error_msg.lower() and ("255" in error_msg or "no devices" in error_msg.lower()):
            raise HTTPException(
                status_code=503,
                detail={
                    "error_type": "adb_connection",
                    "message": "Android 디바이스가 연결되어 있지 않습니다",
                    "hint": "USB 케이블로 디바이스를 연결하고 USB 디버깅을 활성화하세요"
                }
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/respond")
async def agent_respond(req: RespondRequest):
    """Respond to agent's question and continue execution."""
    session = sessions.get(req.thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        return streamer.respond(req)
    except Exception as e:
        logger.error(f"Respond failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interrupt")
async def agent_interrupt(req: RespondRequest):
    """Force interrupt running agent."""
    session = sessions.get(req.thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    prev_state = session.get("last_state") or {}
    sessions.update(
        req.thread_id,
        waiting_human=True,
        last_state={
            **prev_state,
            "status": "needs_human",
            "payload": {
                "interrupt": {
                    "reason": "USER_STOP",
                    "question": req.response or "무엇을 도와드릴까요?",
                }
            },
        },
    )
    
    return JSONResponse({
        "status": "interrupted",
        "thread_id": req.thread_id,
        "message": "Agent interrupted. Awaiting human input.",
    })


@router.get("/session/{thread_id}")
async def get_session(thread_id: str):
    """Get session state."""
    session = sessions.get(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = session.get("last_state", {})
    return JSONResponse({
        "thread_id": thread_id,
        "waiting_human": session.get("waiting_human", False),
        "iteration": state.get("iteration", 0),
        "status": state.get("status"),
    })
