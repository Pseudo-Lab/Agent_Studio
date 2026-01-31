"""Tools for the status agent -- screenshot loading and loop control."""

from google.adk.tools import ToolContext

from ...shared_libraries import Status


def load_screenshot(tool_context: ToolContext) -> dict:
    """Load the current screenshot from session state for goal verification."""
    print("TOOL CALLED: load_screenshot()")

    screenshot_part = tool_context.state.get("current_screenshot")

    if screenshot_part and hasattr(screenshot_part, "inline_data") and screenshot_part.inline_data:
        return {
            "status": Status.SUCCESS.value,
            "message": "Screenshot loaded successfully. The image is now available for verification.",
            "image": screenshot_part
        }
    else:
        return {
            "status": Status.FAIL.value,
            "error": "No screenshot found in session state. Please capture a screenshot first."
        }


def exit_loop(reason: str, tool_context: ToolContext) -> dict:
    """Exit the LoopAgent -- call when the goal is achieved or unrecoverable.

    Sets ``actions.escalate = True`` so the LoopAgent stops iterating.
    """
    print(f"TOOL CALLED: exit_loop(reason={reason}) triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {
        "status": Status.FINISH.value,
        "reason": reason
    }


def continue_loop(reason: str, tool_context: ToolContext) -> dict:
    """Signal that the goal is not yet achieved; proceed to the next iteration."""
    print(f"TOOL CALLED: continue_loop(reason={reason}) triggered by {tool_context.agent_name}")
    return {
        "status": Status.CONTINUE.value,
        "reason": reason
    }
