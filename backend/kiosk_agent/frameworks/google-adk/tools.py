"""Shared tools available to the orchestrator agent.

These tools are registered directly on the orchestrator (not via a
sub-agent) so that the orchestrator itself can load screenshots and
terminate the loop.
"""

from google.adk.tools import ToolContext

from .shared_libraries import Status


def load_screenshot(tool_context: ToolContext) -> dict:
    """Load the current screenshot from session state for analysis.

    Must be called after capture_agent has stored a screenshot.
    The returned ``image`` Part is later injected into the LLM context
    by ``inject_screenshot_callback``.
    """
    print("TOOL CALLED: load_screenshot()")

    screenshot_part = tool_context.state.get("current_screenshot")

    if screenshot_part and hasattr(screenshot_part, "inline_data") and screenshot_part.inline_data:
        return {
            "status": Status.SUCCESS.value,
            "message": "Screenshot loaded successfully. The image is now in context for analysis.",
            "image": screenshot_part
        }
    else:
        return {
            "status": Status.FAIL.value,
            "error": "No screenshot found in session state. Please call capture_agent first."
        }


def exit_loop(reason: str, tool_context: ToolContext) -> dict:
    """Exit the automation loop.

    Sets ``actions.escalate = True`` which causes the LoopAgent to stop
    iterating and return control to the caller.
    """
    print(f"TOOL CALLED: exit_loop(reason={reason}) triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {
        "status": Status.FINISH.value,
        "reason": reason
    }
