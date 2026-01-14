"""Node implementations for LangGraph Kiosk Agent workflow."""

from __future__ import annotations

from typing import Literal, Optional

from PIL import Image

from ...types import AgentState, HistoryEntry
from ...utils import get_logger, compute_difference, compute_screen_id, parse_analysis_response, build_analysis_prompt
from ...utils.common import format_thought_history
from .prompts import USER_PROMPT_TEMPLATE, USER_PROMPT_WITH_PLAN_TEMPLATE

logger = get_logger(__name__)


class NodeMixin:
    """
    Mixin class providing node implementations for KioskAgent.
    
    Nodes:
    - vlm: Vision-Language Model reasoning
    - execute: ADB command execution
    - router: Route decision based on execution results
    - human: Human-in-the-loop interaction
    - analyze: Situation analysis when stuck
    """

    def _vlm_node(self, state: AgentState) -> AgentState:
        """
        VLM reasoning node.
        
        Captures screen, sends to VLM with history context,
        and receives proposed action.
        
        When Planning Mode is enabled with a plan, uses step-by-step
        to-do style prompting.
        """
        screenshot = self._get_screen()
        thought_history = format_thought_history(state.get("history", []))
        
        # Check if we should use Plan Mode prompt (to-do style)
        plan = state.get("plan", [])
        enable_planning = getattr(self, 'enable_planning', False)
        
        if enable_planning and plan:
            # To-do style: focus on current step
            plan_step_index = state.get("plan_step_index", 0)
            total_steps = len(plan)
            
            # Build plan steps display (with checkboxes)
            plan_steps_display = []
            for idx, step in enumerate(plan):
                if idx < plan_step_index:
                    plan_steps_display.append(f"  ✅ {step}")
                elif idx == plan_step_index:
                    plan_steps_display.append(f"  ➡️ {step} ← 현재")
                else:
                    plan_steps_display.append(f"  ⬜ {step}")
            
            current_step = plan[plan_step_index] if plan_step_index < total_steps else "모든 단계 완료"
            
            full_instruction = USER_PROMPT_WITH_PLAN_TEMPLATE.format(
                thought_history=thought_history,
                plan_steps="\n".join(plan_steps_display),
                current_step_num=plan_step_index + 1,
                total_steps=total_steps,
                current_step=current_step,
                user_instruction=state.get("original_instruction") or state["instruction"],
            )
            logger.info(f"[Plan Mode] Executing step {plan_step_index + 1}/{total_steps}: {current_step[:30]}...")
        else:
            # Standard prompt (no planning)
            full_instruction = USER_PROMPT_TEMPLATE.format(
                thought_history=thought_history,
                user_instruction=state["instruction"]
            )
        
        model_action = self.model_client.propose_action(full_instruction, screenshot.image)
        payload = model_action.payload
        
        logger.debug(f"VLM output: action={payload.get('action')}, thought={payload.get('thought', '')[:50]}")
        
        return {
            "model_action": model_action,
            "payload": payload,
            "thought": payload.get("thought"),
            "status": "thinking",
            "pre_action_path": screenshot.path,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _execute_node(self, state: AgentState) -> AgentState:
        """
        Execute ADB command node.
        
        Translates model action to ADB commands and executes them.
        """
        model_action = state.get("model_action")
        if model_action is None:
            raise RuntimeError("Execute node received no model action")
            
        pre_screen = self._get_screen()
        payload = model_action.payload or {}
        enable_planning = getattr(self, "enable_planning", False)
        if enable_planning and state.get("plan") and payload.get("step_decision") == "abort":
            return {
                "status": "executed",
                "post_action_path": pre_screen.path,
                "last_adb_commands": [],
            }

        adb_result = self.translator.execute(payload, img_size=pre_screen.image.size)
        post_screen = self._capture_screen()
        
        logger.debug(f"Executed ADB commands: {len(adb_result.commands)} commands")
        
        return {
            "status": "executed",
            "post_action_path": post_screen.path,
            "last_adb_commands": adb_result.commands,
        }

    def _router_node(self, state: AgentState) -> AgentState:
        """
        Router node.
        
        Evaluates execution results, computes progress,
        updates history, and determines next route.
        
        In Plan Mode, also detects step completion and advances
        to the next step.
        """
        post_screen_path = state.get("post_action_path")
        pre_screen_path = state.get("pre_action_path")
        
        new_screen_id = compute_screen_id(post_screen_path)
        
        # Compute difference & progress
        diff = compute_difference(pre_screen_path, post_screen_path)
        progress = bool(diff is not None and diff >= self.progress_threshold)
        
        # Update history
        iteration = state.get("iteration", 0)
        new_entry: HistoryEntry = {
            "iteration": iteration,
            "payload": state.get("payload", {}),
            "thought": state.get("thought"),
            "adb_commands": state.get("last_adb_commands", []),
            "progress": progress,
            "screen_id": new_screen_id,
            "pre_action_path": pre_screen_path,
            "post_action_path": post_screen_path,
        }
        history = list(state.get("history", [])) + [new_entry]
        
        # Determine status
        status = "progress_made" if progress else "stagnant"
        payload = state.get("payload", {})
        
        if payload.get("action") == "FINISH":
            status = "task_complete"
        elif payload.get("action") == "ABORT":
            status = "aborted"
        elif payload.get("interrupt") or payload.get("action") == "INTERRUPT":
            status = "needs_human"

        # Plan Mode: use model-guided step decision with fallback heuristics
        plan_step_index = state.get("plan_step_index", 0)
        plan = state.get("plan", [])
        step_completed = False

        if getattr(self, 'enable_planning', False) and plan:
            thought = state.get("thought", "") or ""
            decision_raw = payload.get("step_decision")
            decision = str(decision_raw or "").strip().lower()
            is_interrupt = bool(payload.get("interrupt")) or payload.get("action") == "INTERRUPT"

            if decision in {"repeat", "advance", "abort"}:
                if is_interrupt and decision == "advance":
                    decision = "repeat"
            else:
                # Fallback for older prompts/models
                logger.warning("[Plan Mode] Missing or invalid step_decision; falling back to heuristics")
                if "단계 완료" in thought or "✅" in thought or "step complete" in thought.lower():
                    decision = "advance"

            if decision == "abort":
                status = "aborted"
            elif decision == "advance":
                step_completed = True
                plan_step_index += 1
                logger.info(f"[Plan Mode] ✅ Step completed! Advancing to step {plan_step_index + 1}/{len(plan)}")

                # Check if all steps completed
                if plan_step_index >= len(plan):
                    logger.info("[Plan Mode] 🎉 All steps completed!")
                    status = "task_complete"

        # Delegate route decision
        temp_state: AgentState = {**state, "status": status, "history": history}
        route = self._determine_route(temp_state)
        if (
            route == "end"
            and iteration >= self.max_iterations
            and status not in {"task_complete", "aborted", "needs_human"}
        ):
            status = "max_iterations"
        
        logger.debug(f"Router: status={status}, route={route}, progress={progress}")

        result = {
            "history": history,
            "status": status,
            "route": route,
            "progress": progress,
            "difference": diff,
            "current_screen_id": new_screen_id or state.get("current_screen_id"),
            "last_iteration_id": iteration,
            "step_completed": step_completed,
        }
        
        # Update plan_step_index if step was completed
        if step_completed:
            result["plan_step_index"] = plan_step_index
        
        return result

    def _human_node(self, state: AgentState) -> AgentState:
        """
        Human-in-the-loop node (CLI mode).
        
        Prompts user for input and optionally executes text input.
        """
        logger.info("=" * 50)
        logger.info(" HUMAN ASSISTANCE REQUIRED ")
        logger.info("=" * 50)
        
        payload = state.get("payload", {})
        interrupt_data = payload.get("interrupt", {})
        question = interrupt_data.get("question", "에이전트가 정보를 더 필요로 합니다.")
        requires_text_input = interrupt_data.get("requires_text_input", False)
        
        print(f"\n[Agent]: {question}")
        user_input = input("\n답변 (종료: quit): ").strip()
        
        if user_input.lower() in {"finish", "quit", "exit"}:
            logger.info("User requested abort")
            return {
                "route": "abort",
                "status": "aborted",
            }
        
        # Execute text input if required
        adb_commands = []
        if requires_text_input and user_input:
            logger.info(f"Executing text input: '{user_input}'")
            try:
                screen = self._get_screen()
                input_result = self.translator.execute(
                    {"action": "INPUT", "value": user_input},
                    img_size=screen.image.size
                )
                adb_commands = input_result.commands
                self._capture_screen()
            except Exception as e:
                logger.warning(f"Text input failed: {e}")
        
        combined_instruction = f"{state['instruction']}\n[추가 정보]: {question} -> {user_input}"
        
        logger.info("Resuming with human feedback")
        return {
            "instruction": combined_instruction,
            "route": "resume",
            "status": "human_feedback_applied",
            "last_adb_commands": adb_commands,
            "human_decision": user_input,
        }

    def _analyze_node(self, state: AgentState) -> AgentState:
        """
        Analyze node.
        
        Analyzes current situation when agent is stuck
        and suggests next action.
        """
        prompt = build_analysis_prompt(state)
        screen = self._get_screen(refresh=False)
        raw_analysis = self._invoke_analysis_model(prompt, screen.image)
        analysis_text, suggested_route, target_idx = parse_analysis_response(raw_analysis)
        
        logger.debug(f"Analysis result: route={suggested_route}, target={target_idx}")
        
        if suggested_route == "backtrack":
            return {
                "analysis": analysis_text,
                "route": "backtrack",
                "status": "backtracking",
                "backtrack_target_index": target_idx,
            }
            
        iteration = state.get("iteration", 0)
        next_route = "loop" if suggested_route == "loop" and iteration < self.max_iterations else "end"
        if next_route == "end" and suggested_route == "loop" and iteration >= self.max_iterations:
            status = "max_iterations"
        else:
            status = "needs_retry" if next_route == "loop" else "analyzed"
        
        return {
            "analysis": analysis_text,
            "route": next_route,
            "status": status,
        }

    def _invoke_analysis_model(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Invoke analysis model."""
        raw = self.analysis_client.generate(prompt, image)
        return self.model_client._coerce_raw_text(raw)
