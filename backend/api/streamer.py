"""
AG-UI compatible SSE streaming for Kiosk Agent.

Provides streaming endpoints that follow AG-UI event protocol:
- RUN_STARTED
- STATE_SNAPSHOT
- CUSTOM (waiting_human, tts_generated)
- RUN_ERROR
- RUN_FINISHED
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Optional

from fastapi.responses import StreamingResponse

from kiosk_agent.utils import get_logger
from kiosk_agent.characters import (
    get_character_for_session,
    get_character_image_path,
    get_character_ref_audio_path,
    clear_session_character,
    Character,
)
from .schemas import RespondRequest, StartRequest
from .session import SessionStore

logger = get_logger(__name__)


def sse_encode(data: Dict[str, Any]) -> bytes:
    """Encode data as SSE event."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def snapshot_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert agent state to snapshot format including planning fields."""
    payload = state.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    
    # Planning fields
    plan = state.get("plan", [])
    plan_step_index = state.get("plan_step_index", 0)
    planning_complete = state.get("planning_complete", True)
    step_completed = state.get("step_completed", False)
    
    interrupt = payload.get("interrupt")
    action = payload.get("action")
    action_upper = (str(action) if action else "").upper()
    is_interrupt = bool(interrupt) or action_upper == "INTERRUPT"
    
    status = state.get("status")
    if is_interrupt or status == "needs_human":
        status = "waiting_human"
    
    return {
        "status": status,
        "stage": state.get("route"),
        "iteration": int(state.get("iteration") or 0),
        "thought": state.get("thought"),
        "action": action,
        "box_2d": payload.get("box_2d"),
        "adb_commands": [] if is_interrupt else state.get("last_adb_commands", []),
        "interrupt": interrupt,
        "progress": state.get("progress"),
        "difference": state.get("difference"),
        # Planning fields (to-do style)
        "plan": plan if plan else None,
        "plan_step_index": plan_step_index if plan else None,
        "planning_complete": planning_complete,
        "step_completed": step_completed,  # True when a plan step was just completed
    }


def interrupt_question(state: Dict[str, Any]) -> str:
    """Extract interrupt question from state."""
    payload = state.get("payload") or {}
    interrupt = payload.get("interrupt") or {}
    return interrupt.get("question", "추가 정보가 필요합니다.")


def extract_final_thought(state: Dict[str, Any]) -> str:
    """Extract final thought from state."""
    if state.get("status") == "aborted":
        return "사용자 요청으로 작업을 종료합니다."
    if state.get("status") == "max_iterations":
        return "사용자가 설정한 최대 턴 수에 도달해 종료합니다."
    history = state.get("history", [])
    if history:
        last_entry = history[-1]
        if last_entry.get("thought"):
            return last_entry["thought"]
    return state.get("thought") or "작업이 완료되었습니다."


def tts_file_prefix(thread_id: str, run_id: str) -> str:
    """Generate TTS file prefix."""
    return f"tts_{thread_id[:8]}_{run_id[:8]}_{time.time_ns()}"


def validate_file(path: Optional[str]) -> bool:
    """Check if file exists."""
    if not path:
        return False
    return Path(path).exists()


class AgentStreamer:
    """
    AG-UI compatible SSE streaming handler.
    
    Handles:
    - POST /agent/start
    - POST /agent/respond
    """

    def __init__(
        self,
        *,
        executor: ThreadPoolExecutor,
        sessions: SessionStore,
        get_agent: Callable[[Optional[str], bool], Any],
    ):
        self._executor = executor
        self._sessions = sessions
        self._get_agent = get_agent

    def start(self, req: StartRequest) -> StreamingResponse:
        """Start agent execution with SSE streaming."""
        thread_id = req.thread_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        
        enable_planning = getattr(req, 'enable_planning', False)
        logger.info(f"Starting agent: thread={thread_id}, model={req.model or 'default'}, planning={enable_planning}")

        # Assign character for this session
        clear_session_character(thread_id)  # 새 세션이므로 기존 할당 초기화
        character = get_character_for_session(thread_id)
        # NOTE: Frontend expects 'character' key
        character_info = {
            "id": character.id,
            "name": character.name,
            "nickname": character.nickname,
            "imagePath": get_character_image_path(character),
        }
        character_ref_audio = get_character_ref_audio_path(character)
        character_ref_text = character.ref_text

        agent = self._get_agent(req.model, enable_planning)
        graph, initial_state = agent.prepare_workflow(req.instruction)

        # Initialize session
        self._sessions.set(
            thread_id,
            {
                "waiting_human": False,
                "last_state": initial_state,
                "last_iteration": int(initial_state.get("iteration") or 0),
                "instruction": req.instruction,
                "model": req.model or "gemini-flash",
                "character_id": character.id,
                "enable_planning": enable_planning,
            },
        )

        q: Queue[tuple[str, Dict[str, Any]]] = Queue()

        def run_graph():
            latest_state: Dict[str, Any] = dict(initial_state)
            try:
                config = {"recursion_limit": int(os.getenv("AGENT_RECURSION_LIMIT", "100"))}
                last_emitted_iter = -1
                final_action: Optional[str] = None

                for event in graph.stream(initial_state, config=config, stream_mode="values"):
                    if isinstance(event, dict):
                        latest_state.update(event)

                    # Planning phase events (AG-UI CUSTOM)
                    status = latest_state.get("status", "")
                    if status in ["detecting_unknown", "web_search_complete", "web_search_skipped", "planning_complete"]:
                        planning_event = {
                            "phase": "planning",
                            "status": status,
                            "unknown_entities": latest_state.get("unknown_entities", []),
                            "search_context": latest_state.get("search_context", "")[:500] if latest_state.get("search_context") else "",
                            "plan": latest_state.get("plan", []),
                        }
                        q.put(("planning", planning_event))
                        logger.info(f"[Planning] Status: {status}")

                    iteration = int(latest_state.get("iteration") or 0)
                    thought = latest_state.get("thought")
                    payload = latest_state.get("payload") or {}
                    if not isinstance(payload, dict):
                        payload = {}
                    
                    current_action = payload.get("action")
                    action_upper = (str(current_action) if current_action else "").upper()
                    decision = str(payload.get("step_decision") or "").lower()
                    is_abort = (
                        action_upper == "ABORT"
                        or (enable_planning and decision == "abort")
                        or latest_state.get("status") == "aborted"
                    )
                    is_interrupt = False if is_abort else (bool(payload.get("interrupt")) or action_upper == "INTERRUPT")
                    has_result = latest_state.get("progress") is not None
                    ready = bool(is_interrupt or (latest_state.get("post_action_path") and has_result))

                    if thought and ready and iteration != last_emitted_iter:
                        last_emitted_iter = iteration
                        final_action = str(current_action) if current_action else final_action
                        snap = snapshot_from_state(latest_state)
                        logger.debug(f"Snapshot: iter={iteration}, action={current_action}")
                        q.put(("snapshot", snap))
                        
                        # Plan step completion event (to-do style progress)
                        if snap.get("step_completed") and snap.get("plan"):
                            step_event = {
                                "plan": snap["plan"],
                                "plan_step_index": snap["plan_step_index"],
                                "total_steps": len(snap["plan"]),
                                "current_step": snap["plan"][snap["plan_step_index"]] if snap["plan_step_index"] < len(snap["plan"]) else None,
                                "completed_steps": snap["plan"][:snap["plan_step_index"]],
                            }
                            q.put(("plan_step", step_event))
                            logger.info(f"[Plan Mode] Step {snap['plan_step_index']}/{len(snap['plan'])} completed")
                        
                    # HITL check
                    is_needs_human = latest_state.get("status") == "needs_human"
                    if is_interrupt or is_needs_human:
                        # TTS only for HITL question (with character voice)
                        if agent.tts:
                            try:
                                q_text = interrupt_question(latest_state)
                                if q_text:
                                    audio_path = agent.tts.synthesize(
                                        text=q_text,
                                        file_prefix=tts_file_prefix(thread_id, run_id),
                                        ref_audio_override=character_ref_audio,
                                        ref_text_override=character_ref_text,
                                    )
                                    q.put(("tts", {"audio_path": str(audio_path), "final_thought": q_text}))
                            except Exception as e:
                                logger.warning(f"TTS (HITL) failed: {e}")

                        self._sessions.update(
                            thread_id,
                            waiting_human=True,
                            last_state=dict(latest_state),
                            last_iteration=iteration,
                        )
                        q.put(("hitl", {"thread_id": thread_id, "character": character_info}))
                        q.put(("done", {"status": "waiting_human", "final_action": final_action, "character": character_info}))
                        return

                final_thought = extract_final_thought(latest_state)
                # Apply character completion style (use quit tone for abort)
                if latest_state.get("status") == "aborted":
                    styled_thought = f"{final_thought} {character.get_quit_message()}"
                else:
                    styled_thought = character.get_completion_message(final_thought)

                self._sessions.update(
                    thread_id,
                    last_state=dict(latest_state),
                    last_iteration=int(latest_state.get("iteration") or 0),
                )
                q.put(("done", {"status": "completed", "final_action": final_action, "final_thought": styled_thought, "character": character_info}))
                
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Agent execution failed: {e}\n{tb}")
                q.put(("error", {"message": str(e), "error_type": type(e).__name__, "traceback": tb}))

        self._executor.submit(run_graph)

        async def stream():
            # RUN_STARTED (AG-UI) with character info and planning status
            yield sse_encode({
                "type": "RUN_STARTED",
                "threadId": thread_id,
                "runId": run_id,
                "timestamp": int(time.time() * 1000),
                "character": character_info,  # Frontend expects 'character' key
                "planningEnabled": enable_planning,
            })
            await asyncio.sleep(0)

            loop = asyncio.get_running_loop()
            last_event = time.time()
            max_idle = 300

            while True:
                try:
                    item = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: q.get(timeout=0.5)),
                        timeout=1.0,
                    )
                    kind, payload = item
                    last_event = time.time()

                    if kind == "snapshot":
                        yield sse_encode({"type": "STATE_SNAPSHOT", "snapshot": payload})
                    elif kind == "planning":
                        yield sse_encode({"type": "CUSTOM", "name": "planning_update", "value": payload})
                    elif kind == "plan_step":
                        yield sse_encode({"type": "CUSTOM", "name": "plan_step_update", "value": payload})
                    elif kind == "hitl":
                        yield sse_encode({"type": "CUSTOM", "name": "waiting_human", "value": payload})
                    elif kind == "tts":
                        if validate_file(payload.get("audio_path")):
                            yield sse_encode({"type": "CUSTOM", "name": "tts_generated", "value": payload})
                    elif kind == "error":
                        yield sse_encode({
                            "type": "RUN_ERROR",
                            "message": payload.get("message"),
                            "code": payload.get("error_type"),
                            "timestamp": int(time.time() * 1000),
                        })
                        return
                    elif kind == "done":
                        yield sse_encode({
                            "type": "RUN_FINISHED",
                            "threadId": thread_id,
                            "runId": run_id,
                            "result": {
                                "status": payload.get("status"),
                                "finalAction": payload.get("final_action"),
                                "finalThought": payload.get("final_thought"),
                                "character": payload.get("character") or character_info,
                            },
                            "character": payload.get("character") or character_info,
                            "timestamp": int(time.time() * 1000),
                        })
                        return
                    
                    await asyncio.sleep(0)

                except asyncio.TimeoutError:
                    if time.time() - last_event > max_idle:
                        logger.info("SSE idle timeout")
                        break
                    yield b": keepalive\n\n"
                    await asyncio.sleep(0.5)
                except Empty:
                    await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    def respond(self, req: RespondRequest) -> StreamingResponse:
        """Continue agent execution after human response."""
        session = self._sessions.get(req.thread_id)
        if not session:
            raise KeyError("Session not found")

        logger.info(f"Responding: thread={req.thread_id}")

        previous_state = dict(session.get("last_state") or {})
        model_id = session.get("model") or "gemini-flash"
        enable_planning = bool(session.get("enable_planning", False))
        last_iteration = int(session.get("last_iteration") or 0)

        agent = self._get_agent(model_id, enable_planning)
        
        # Get character for this session (same character as start)
        resp_character = get_character_for_session(req.thread_id)
        resp_character_info = {
            "id": resp_character.id,
            "name": resp_character.name,
            "nickname": resp_character.nickname,
            "imagePath": get_character_image_path(resp_character),
        }
        resp_character_ref_audio = get_character_ref_audio_path(resp_character)
        resp_character_ref_text = resp_character.ref_text

        # Build combined instruction
        question = interrupt_question(previous_state)
        combined = f"{previous_state.get('instruction', '')}\n[추가 정보]: {question} -> {req.response}"

        # Reset state
        initial_state = dict(previous_state)
        initial_state["instruction"] = combined
        initial_state["payload"] = {}
        initial_state["model_action"] = None
        initial_state["thought"] = None
        if enable_planning:
            initial_state["original_instruction"] = combined

        graph, _ = agent.prepare_workflow(combined, previous_state=initial_state)
        self._sessions.touch(req.thread_id)

        run_id = str(uuid.uuid4())
        q: Queue[tuple[str, Dict[str, Any]]] = Queue()

        def run_continue():
            latest_state = dict(initial_state)
            try:
                config = {"recursion_limit": int(os.getenv("AGENT_RECURSION_LIMIT", "100"))}
                last_emitted_iter = -1
                final_action = None

                for event in graph.stream(initial_state, config=config, stream_mode="values"):
                    if isinstance(event, dict):
                        latest_state.update(event)

                    iteration = int(latest_state.get("iteration") or 0)
                    if iteration <= last_iteration:
                        continue

                    thought = latest_state.get("thought")
                    payload = latest_state.get("payload") or {}
                    if not isinstance(payload, dict):
                        payload = {}
                    
                    current_action = payload.get("action")
                    action_upper = (str(current_action) if current_action else "").upper()
                    decision = str(payload.get("step_decision") or "").lower()
                    is_abort = (
                        action_upper == "ABORT"
                        or (enable_planning and decision == "abort")
                        or latest_state.get("status") == "aborted"
                    )
                    is_interrupt = False if is_abort else (bool(payload.get("interrupt")) or action_upper == "INTERRUPT")
                    has_result = latest_state.get("progress") is not None
                    ready = bool(is_interrupt or (latest_state.get("post_action_path") and has_result))

                    if thought and ready and iteration != last_emitted_iter:
                        last_emitted_iter = iteration
                        final_action = str(current_action) if current_action else final_action
                        snap = snapshot_from_state(latest_state)
                        q.put(("snapshot", snap))

                    is_needs_human = latest_state.get("status") == "needs_human"
                    if is_interrupt or is_needs_human:
                        if agent.tts:
                            try:
                                q_text = interrupt_question(latest_state)
                                if q_text:
                                    audio_path = agent.tts.synthesize(
                                        text=q_text,
                                        file_prefix=tts_file_prefix(req.thread_id, run_id),
                                        ref_audio_override=resp_character_ref_audio,
                                        ref_text_override=resp_character_ref_text,
                                    )
                                    q.put(("tts", {"audio_path": str(audio_path), "final_thought": q_text}))
                            except Exception:
                                pass

                        self._sessions.update(
                            req.thread_id,
                            waiting_human=True,
                            last_state=dict(latest_state),
                            last_iteration=iteration,
                        )
                        q.put(("hitl", {"thread_id": req.thread_id, "character": resp_character_info}))
                        q.put(("done", {"status": "waiting_human", "final_action": final_action, "character": resp_character_info}))
                        return

                final_thought = extract_final_thought(latest_state)
                if latest_state.get("status") == "aborted":
                    styled_thought = f"{final_thought} {resp_character.get_quit_message()}"
                else:
                    styled_thought = resp_character.get_completion_message(final_thought)

                self._sessions.update(
                    req.thread_id,
                    waiting_human=False,
                    last_state=dict(latest_state),
                    last_iteration=int(latest_state.get("iteration") or 0),
                )
                q.put(("done", {"status": "completed", "final_action": final_action, "final_thought": styled_thought, "character": resp_character_info}))

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Continue failed: {e}\n{tb}")
                q.put(("error", {"message": str(e), "error_type": type(e).__name__}))

        self._executor.submit(run_continue)

        async def stream():
            yield sse_encode({
                "type": "RUN_STARTED",
                "threadId": req.thread_id,
                "runId": run_id,
                "timestamp": int(time.time() * 1000),
                "character": resp_character_info,
            })
            await asyncio.sleep(0)

            loop = asyncio.get_running_loop()
            
            while True:
                try:
                    item = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: q.get(timeout=0.5)),
                        timeout=1.0,
                    )
                    kind, payload = item

                    if kind == "snapshot":
                        yield sse_encode({"type": "STATE_SNAPSHOT", "snapshot": payload})
                    elif kind == "hitl":
                        yield sse_encode({"type": "CUSTOM", "name": "waiting_human", "value": payload})
                    elif kind == "tts":
                        if validate_file(payload.get("audio_path")):
                            yield sse_encode({"type": "CUSTOM", "name": "tts_generated", "value": payload})
                    elif kind == "error":
                        yield sse_encode({
                            "type": "RUN_ERROR",
                            "message": payload.get("message"),
                            "timestamp": int(time.time() * 1000),
                        })
                        return
                    elif kind == "done":
                        yield sse_encode({
                            "type": "RUN_FINISHED",
                            "threadId": req.thread_id,
                            "runId": run_id,
                            "result": {
                                "status": payload.get("status"),
                                "finalAction": payload.get("final_action"),
                                "finalThought": payload.get("final_thought"),
                                "character": payload.get("character") or resp_character_info,
                            },
                            "character": payload.get("character") or resp_character_info,
                            "timestamp": int(time.time() * 1000),
                        })
                        return

                    await asyncio.sleep(0)

                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    await asyncio.sleep(0.5)
                except Empty:
                    await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
