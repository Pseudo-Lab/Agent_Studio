"""Tools for the analysis agent."""

from google.adk.tools import ToolContext

from ...shared_libraries import Status


def load_screenshot(tool_context: ToolContext) -> dict:
    """Load the current screenshot from session state for VLM analysis.

    The returned ``image`` Part is injected into the LLM context by
    ``inject_screenshot_callback`` so Gemini can see the kiosk screen.
    """
    print("TOOL CALLED: load_screenshot()")

    screenshot_part = tool_context.state.get("current_screenshot")

    if screenshot_part and hasattr(screenshot_part, "inline_data") and screenshot_part.inline_data:
        return {
            "status": Status.SUCCESS.value,
            "message": "Screenshot loaded successfully. The image is now available for analysis.",
            "image": screenshot_part
        }
    else:
        return {
            "status": Status.FAIL.value,
            "error": "No screenshot found in session state. Please capture a screenshot first."
        }
