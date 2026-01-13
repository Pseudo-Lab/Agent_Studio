"""
FastAPI server for Kiosk Agent.

Entry point:
    uvicorn api.main:app --port 8080 --reload
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kiosk_agent.utils import get_logger
from .routes import agent, health, voice

logger = get_logger(__name__)

# Initialize app
app = FastAPI(
    title="Kiosk Agent API",
    version="0.1.0",
    description="Vision-Language-Action Agent for Kiosk Automation",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(health.router, tags=["health"])
app.include_router(agent.router, prefix="/agent", tags=["agent"])
app.include_router(voice.router, tags=["voice"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Initializing Kiosk Agent API...")
    
    # Pre-initialize agent (optional)
    try:
        from .routes.agent import get_agent
        agent_instance = get_agent(None)
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
    
    # Background cleanup task
    async def cleanup_loop():
        from .routes.agent import sessions
        while True:
            await asyncio.sleep(600)  # 10 minutes
            sessions.cleanup()
    
    asyncio.create_task(cleanup_loop())
    logger.info("Ready to serve requests")
