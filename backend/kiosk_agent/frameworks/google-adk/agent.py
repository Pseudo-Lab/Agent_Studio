"""Google ADK-based Kiosk Agent implementation.

Replaces the LangGraph workflow with Google ADK's Agent/LoopAgent pattern.
Uses the same config, core modules (ADB, perception, translator), and prompts
as the LangGraph implementation for consistency.

Architecture:
  LoopAgent (kiosk_loop, max_iterations from config)
    -> Orchestrator Agent (kiosk_orchestrator)
        -> capture_agent   (screenshot via ADB)
        -> analysis_agent  (VLM screen analysis)
        -> action_agent    (ADB command execution)
            -> adb_agent   (low-level ADB tools)
        -> status_agent    (goal verification)
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterator, Optional

from google.adk.agents import Agent, LoopAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai.types import Content, Part

from .prompt import ORCHESTRATOR_PROMPT
from .tools import load_screenshot, exit_loop
from .callbacks import inject_screenshot_callback
from .shared_libraries import get_model

from .sub_agents.capture_agent import capture_agent
from .sub_agents.analysis_agent import analysis_agent
from .sub_agents.action_agent import action_agent
from .sub_agents.status_agent import status_agent

from ...config import AgentConfig
from ...types import AgentState, AgentStepResult, AgentStreamEvent
from ...utils import get_logger
from ..base import BaseAgent

logger = get_logger(__name__)


def _build_orchestrator() -> Agent:
    """Build the orchestrator agent with all sub-agents as tools."""
    return Agent(
        model=get_model(),
        name="kiosk_orchestrator",
        description="Orchestrates kiosk GUI automation using sub-agents",
        instruction=ORCHESTRATOR_PROMPT,
        tools=[
            AgentTool(agent=capture_agent),
            AgentTool(agent=analysis_agent),
            AgentTool(agent=action_agent),
            AgentTool(agent=status_agent),
            load_screenshot,
            exit_loop,
        ],
        before_model_callback=inject_screenshot_callback,
    )


def _build_loop_agent(max_iterations: int = 20) -> LoopAgent:
    """Build the LoopAgent that wraps the orchestrator."""
    orchestrator = _build_orchestrator()
    return LoopAgent(
        name="kiosk_loop",
        sub_agents=[orchestrator],
        max_iterations=max_iterations,
    )


class ADKKioskAgent(BaseAgent):
    """
    Google ADK-based Kiosk Agent.

    Provides the same BaseAgent interface as the LangGraph KioskAgent,
    reading configuration from AgentConfig (same .env variables).
    """

    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config)
        self.max_iterations = config.max_iterations
        self.loop_agent = _build_loop_agent(max_iterations=self.max_iterations)
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=self.loop_agent,
            app_name="kiosk_agent_adk",
            session_service=self._session_service,
        )

    def init_state(self, instruction: str) -> AgentState:
        """Initialize agent state (ADK manages state internally)."""
        return {
            "instruction": instruction,
            "iteration": 0,
            "status": "init",
            "route": "vlm",
            "history": [],
            "human_decision": None,
            "difference": None,
            "progress": None,
            "last_adb_commands": [],
            "last_iteration_id": -1,
            "current_screen_id": None,
            "plan": [],
            "plan_step_index": 0,
            "unknown_entities": [],
            "search_context": "",
            "planning_complete": True,
            "original_instruction": instruction,
        }

    def prepare_workflow(
        self,
        instruction: str,
        previous_state: Optional[AgentState] = None,
    ) -> tuple[Any, AgentState]:
        """Prepare workflow (returns loop_agent and initial state)."""
        state = previous_state or self.init_state(instruction)
        return self.loop_agent, state

    def run(self, instruction: str) -> AgentStepResult:
        """Run agent synchronously."""
        return asyncio.run(self._run_async(instruction))

    async def _run_async(self, instruction: str) -> AgentStepResult:
        """Internal async runner."""
        logger.info(f"ADK Kiosk Agent starting: {instruction[:80]}...")

        session = await self._session_service.create_session(
            app_name="kiosk_agent_adk",
            user_id="kiosk_user"
        )

        user_message = Content(
            role="user",
            parts=[Part(text=f"Goal: {instruction}\n\nStart by capturing a screenshot from the kiosk device.")]
        )

        history = []
        last_text = ""
        iteration = 0

        try:
            async for event in self._runner.run_async(
                user_id="kiosk_user",
                session_id=session.id,
                new_message=user_message
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text = part.text.strip()
                            if text:
                                last_text = text
                                logger.debug(f"[{event.author}]: {text[:200]}")
                        if hasattr(part, 'function_call') and part.function_call:
                            logger.debug(f"[{event.author}]: Calling {part.function_call.name}")
                        if hasattr(part, 'function_response') and part.function_response:
                            resp = part.function_response
                            if hasattr(resp, 'response') and resp.response:
                                result = resp.response
                                if isinstance(result, dict) and result.get('status') == 'finish':
                                    logger.info(f"Task finished: {result.get('reason', '')}")

        except Exception as e:
            logger.error(f"ADK Agent error: {e}")
            import traceback
            traceback.print_exc()
            return AgentStepResult(
                screenshot_path=None,
                raw_response=str(e),
                thought=None,
                payload={"action": "ERROR"},
                adb_commands=[],
                status="error",
            )

        return AgentStepResult(
            screenshot_path=None,
            raw_response=last_text,
            thought=last_text,
            payload={},
            adb_commands=[],
            status="completed",
            history=history,
        )

    def stream(
        self,
        instruction: str,
        *,
        thread_id: Optional[str] = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute agent with streaming output."""

        async def _stream_inner():
            session = await self._session_service.create_session(
                app_name="kiosk_agent_adk",
                user_id="kiosk_user"
            )

            user_message = Content(
                role="user",
                parts=[Part(text=f"Goal: {instruction}\n\nStart by capturing a screenshot from the kiosk device.")]
            )

            events = []
            async for event in self._runner.run_async(
                user_id="kiosk_user",
                session_id=session.id,
                new_message=user_message
            ):
                stage = "running"
                message = ""

                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            message = part.text.strip()
                        if hasattr(part, 'function_call') and part.function_call:
                            stage = f"calling_{part.function_call.name}"
                        if hasattr(part, 'function_response') and part.function_response:
                            resp = part.function_response
                            if hasattr(resp, 'response') and isinstance(resp.response, dict):
                                if resp.response.get('status') == 'finish':
                                    stage = "completed"

                stream_event = AgentStreamEvent(
                    stage=stage,
                    state={"instruction": instruction, "status": stage},
                    message=message,
                )
                events.append(stream_event)

            return events

        loop = asyncio.new_event_loop()
        try:
            events = loop.run_until_complete(_stream_inner())
            yield from events
        finally:
            loop.close()


# Alias for backwards compatibility
KioskAgentADK = ADKKioskAgent
