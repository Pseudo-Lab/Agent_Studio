"""LangGraph-based Kiosk Agent implementation."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, Optional, cast

from PIL import Image
from langgraph.graph.state import CompiledStateGraph

from ...config import AgentConfig
from ...core import ADBController, ActionTranslator, AndroidScreenshotter, CapturedScreen
from ...llm import GeminiClient, ChatGPTClient, LocalVLLMClient
from ...types import AgentState, AgentStepResult, AgentStreamEvent
from ...utils import get_logger, compute_screen_id
from ...utils.common import build_result
from ..base import BaseAgent
from .nodes import NodeMixin
from .graph import GraphMixin

logger = get_logger(__name__)


class KioskAgent(NodeMixin, GraphMixin, BaseAgent):
    """
    Vision-Language-Action agent using LangGraph for orchestration.
    
    Implements the VLA workflow:
    1. Perception: Capture screenshot from Android device
    2. Reasoning: VLM analyzes screen and proposes action
    3. Action: Execute ADB command on device
    4. Loop: Repeat until task complete or human input needed
    
    Inherits:
    - NodeMixin: Node implementations (vlm, execute, router, human, analyze)
    - GraphMixin: Graph building and routing logic
    - BaseAgent: Base agent interface
    """

    def __init__(
        self, 
        config: AgentConfig, 
        dry_run: bool = False,
        enable_tts: bool = True,
    ):
        super().__init__(config)
        
        self.current_screen_path: Optional[str] = None
        self._current_screen_image: Optional[Image.Image] = None
        
        # Runtime parameters from config (env-backed)
        self.max_iterations = config.max_iterations
        self.progress_threshold = config.progress_threshold
        self.recursion_limit = config.recursion_limit
        self.tts_keep_last_n = config.tts_keep_last_n
        self.enable_tts = enable_tts
        self.graph: Optional[CompiledStateGraph] = None
        
        # Initialize model client
        self._init_model_client(config)
        
        # Initialize device control
        self._init_device_control(config, dry_run)
        
        # Initialize TTS
        self._init_tts()

    def _init_model_client(self, config: AgentConfig) -> None:
        """Initialize VLM model client based on provider."""
        provider = config.model.provider
        if provider == "chatgpt":
            self.model_client = ChatGPTClient(config.model)
        elif provider == "gemini":
            self.model_client = GeminiClient(config.model)
        elif provider == "local_vllm":
            self.model_client = LocalVLLMClient(config.model)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
        
        self.analysis_client = self.model_client

    def _init_device_control(self, config: AgentConfig, dry_run: bool) -> None:
        """Initialize ADB controller and screen capture."""
        adb_controller = ADBController(config.adb, dry_run=dry_run)
        self.translator = ActionTranslator(adb_controller, config.adb)
        self.screenshotter = AndroidScreenshotter(config.screenshot)

    def _init_tts(self) -> None:
        """Initialize TTS (Text-to-Speech) module."""
        self.tts = None
        if self.enable_tts:
            try:
                from ...voice import create_default_tts
                self.tts = create_default_tts()
                logger.info("TTS initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize TTS: {e}")
                self.tts = None

    # ==================== Screen Management ====================

    def _capture_screen(self) -> CapturedScreen:
        """Capture current screen and cache it."""
        screenshot = self.screenshotter.capture(save=True)
        self.current_screen_path = screenshot.path
        self._current_screen_image = screenshot.image
        return screenshot

    def _get_screen(self, *, refresh: bool = False) -> CapturedScreen:
        """Get cached screen or capture new one."""
        if refresh or self._current_screen_image is None or self.current_screen_path is None:
            return self._capture_screen()
        return CapturedScreen(image=self._current_screen_image, path=self.current_screen_path)

    # ==================== State Management ====================

    def init_state(self, instruction: str) -> AgentState:
        """Initialize agent state for new task."""
        self._capture_screen()
        initial_state: AgentState = {
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
        }
        screen_id = compute_screen_id(self.current_screen_path)
        initial_state["current_screen_id"] = screen_id
        return initial_state

    def prepare_workflow(
        self, 
        instruction: str, 
        previous_state: Optional[AgentState] = None
    ) -> tuple[CompiledStateGraph, AgentState]:
        """Prepare graph and initial state for execution."""
        if previous_state:
            initial_state = previous_state
        else:
            initial_state = self.init_state(instruction)

        self.graph = self._build_graph()
        return self.graph, initial_state

    # ==================== Execution Methods ====================

    def run(self, instruction: str) -> AgentStepResult:
        """Run agent synchronously."""
        graph, initial_state = self.prepare_workflow(instruction)
        config = {"recursion_limit": self.recursion_limit}
        final_state = graph.invoke(initial_state, config)
        result = build_result(final_state)
        
        if self.tts and result.get("status") in ["task_complete", "completed"]:
            self._generate_completion_tts(result)
        
        return result

    def run_async(self, instruction: str) -> AgentStepResult:
        """Run agent asynchronously (placeholder)."""
        return self.run(instruction)

    def stream(
        self, 
        instruction: str,
        *,
        thread_id: Optional[str] = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute agent with streaming output."""
        graph, initial_state = self.prepare_workflow(instruction)
        
        config: Dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        
        latest_state = dict(initial_state)
        
        for event in graph.stream(initial_state, config=config, stream_mode="values"):
            if isinstance(event, dict):
                latest_state.update(event)
            
            yield AgentStreamEvent(
                stage=latest_state.get("route", "unknown"),
                state=cast(AgentState, dict(latest_state)),
                message=latest_state.get("thought", ""),
            )

    # ==================== TTS ====================

    def _generate_completion_tts(self, result: Dict[str, Any]) -> None:
        """Generate TTS audio for task completion."""
        try:
            final_thought = None
            if result.get("history"):
                last_entry = result["history"][-1]
                final_thought = last_entry.get("thought")
            
            if not final_thought:
                final_thought = result.get("thought") or "작업이 완료되었습니다."
            
            logger.info(f"TTS generating: {final_thought[:50]}...")
            audio_path = self.tts.synthesize(
                text=final_thought,
                file_prefix=f"agent_completion_{time.time_ns()}",
            )
            logger.info(f"TTS saved: {audio_path}")
            
            if self.tts_keep_last_n > 0:
                self.tts.cleanup_old_files(keep_last_n=self.tts_keep_last_n)
                
        except Exception as e:
            logger.warning(f"TTS generation failed: {e}")


# Backwards compatibility alias
LangGraphKioskAgent = KioskAgent
