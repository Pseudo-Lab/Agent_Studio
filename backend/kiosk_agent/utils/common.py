"""Common utility functions for Kiosk Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from PIL import Image, ImageChops, ImageStat


def compute_difference(
    pre_path: Optional[Path], 
    post_path: Optional[Path]
) -> Optional[float]:
    """
    Compute visual difference between two screenshots.
    
    Args:
        pre_path: Path to screenshot before action
        post_path: Path to screenshot after action
        
    Returns:
        Normalized difference (0.0 - 1.0) or None if comparison fails
    """
    if not pre_path or not post_path:
        return None
    try:
        with Image.open(pre_path) as before_img, Image.open(post_path) as after_img:
            before = before_img.convert("RGB")
            after = after_img.convert("RGB")
    except (FileNotFoundError, OSError):
        return None
    
    if before.size != after.size:
        after = after.resize(before.size)
    
    diff = ImageChops.difference(before, after)
    stat = ImageStat.Stat(diff)
    if not stat.mean:
        return None
    
    mean = sum(stat.mean) / len(stat.mean)
    return mean / 255


def compute_screen_id(image_path: Optional[Path]) -> Optional[str]:
    """
    Compute a simple visual fingerprint for the screen.
    
    Args:
        image_path: Path to screenshot
        
    Returns:
        Hex string fingerprint or None
    """
    if not image_path or not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            small = img.resize((16, 16)).convert("L")
            pixels = list(small.getdata())
            return "".join(f"{p:02x}" for p in pixels)
    except Exception:
        return None


def parse_analysis_response(
    raw: str
) -> Tuple[str, Literal["loop", "end", "backtrack"], Optional[int]]:
    """
    Parse analysis model response.
    
    Args:
        raw: Raw response text from analysis model
        
    Returns:
        Tuple of (analysis_text, route_hint, target_step_index)
    """
    try:
        payload = _safe_parse_json(raw)
    except ValueError:
        return raw, "end", None
        
    thought = payload.get("thought") or payload.get("analysis") or raw
    next_hint = (payload.get("next") or payload.get("route") or "end").lower()
    target_idx = payload.get("target_step_index")
    
    if next_hint in {"vlm", "loop", "retry"}:
        return thought, "loop", None
    if next_hint in {"backtrack", "back"}:
        return thought, "backtrack", target_idx
    return thought, "end", None


def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
    """Best effort JSON parsing with fallbacks for code fenced output."""
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if "\n" in candidate:
            candidate = candidate.split("\n", 1)[1]
    return json.loads(candidate)


# Analysis prompt template
ANALYSIS_PROMPT_TEMPLATE = """You are analyzing the current state of an Android kiosk agent.

User Instruction: {instruction}

Last VLM Payload:
{serialized_payload}

ADB Commands Executed:
{commands_text}

Application Structure (discovered so far):
{application_structure}

Thought History:
{thought_history}

Based on the above, analyze:
1. Has progress been made towards the goal?
2. Is the agent stuck in a loop or dead end?
3. Should the agent continue (loop), backtrack (back), or terminate (end)?

Respond in JSON:
{{
    "thought": "Your analysis of the current situation",
    "next": "loop" | "backtrack" | "end",
    "target_step_index": <optional integer if backtracking>
}}
"""


def build_analysis_prompt(state: Dict[str, Any]) -> str:
    """
    Build analysis prompt from agent state.
    
    Args:
        state: Current agent state
        
    Returns:
        Formatted analysis prompt
    """
    payload = state.get("payload") or {}
    history = state.get("history", [])
    
    all_adb_commands = []
    for entry in history:
        all_adb_commands.extend(entry.get("adb_commands", []))
    
    commands_text = "\n".join(" ".join(cmd) for cmd in all_adb_commands) or "(no commands executed)"
    app_structure = state.get("application_structure") or "Not yet defined."
    
    # Build thought history from history entries
    thought_items = []
    for entry in history:
        thought = entry.get("thought")
        if thought:
            thought_items.append(f"Step {entry.get('iteration')}: {thought}")
    thought_history = "\n".join(thought_items) or "No previous thoughts."
    
    try:
        serialized_payload = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        serialized_payload = str(payload)
        
    return ANALYSIS_PROMPT_TEMPLATE.format(
        instruction=state.get("instruction"),
        serialized_payload=serialized_payload,
        commands_text=commands_text,
        application_structure=app_structure,
        thought_history=thought_history
    )


def build_result(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build result dictionary from final agent state.
    
    Args:
        state: Final agent state
        
    Returns:
        Result dictionary
    """
    payload = state.get("payload") or {}
    history = state.get("history", [])
    
    all_adb_commands = []
    for entry in history:
        all_adb_commands.extend(entry.get("adb_commands", []))
        
    last_screenshot = state.get("post_action_path")
    if history:
        last_entry = history[-1]
        last_screenshot = last_entry.get("post_action_path") or last_screenshot
    
    return {
        "screenshot_path": last_screenshot,
        "raw_response": state.get("raw_response") or "",
        "thought": state.get("thought"),
        "payload": payload,
        "adb_commands": all_adb_commands,
        "status": state.get("status", "unknown"),
        "analysis": state.get("analysis"),
        "history": history,
    }


def format_thought_history(history: List[Dict[str, Any]]) -> str:
    """
    Format history entries as thought history text.
    
    Args:
        history: List of history entries
        
    Returns:
        Formatted thought history string
    """
    items = []
    for entry in history:
        thought = entry.get("thought")
        if thought:
            items.append(f"Step {entry.get('iteration')}: {thought}")
    return "\n".join(items) or "Initial step."
