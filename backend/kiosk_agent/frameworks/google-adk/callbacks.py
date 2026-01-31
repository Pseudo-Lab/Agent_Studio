"""before_model_callback that injects screenshot images into LLM context.

Used by analysis_agent, status_agent, and the orchestrator so that the
Gemini model can *see* the kiosk screen when making decisions.

IMPORTANT (google-adk >=1.x):
    before_model_callback must return:
    - None        -> proceed normally with (possibly mutated) llm_request
    - LlmResponse -> skip the LLM call entirely and use this response

    Returning the LlmRequest itself causes ADK to interpret it as an
    LlmResponse, leading to ``'LlmRequest has no attribute content'``.
"""

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types


async def inject_screenshot_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Inject the screenshot image Part into the LLM request contents.

    Scans ``llm_request.contents`` for any ``load_screenshot`` function
    response that contains an ``image`` Part.  When found, appends a
    text prompt and the raw image Part right after the function response
    so that the model receives the image in its next turn.

    Mutates ``llm_request.contents`` in-place and returns ``None`` so
    that ADK proceeds with the normal LLM call.
    """
    if not llm_request.contents:
        return None

    new_contents = []

    for content in llm_request.contents:
        new_parts = []

        for part in content.parts:
            new_parts.append(part)

            # Look for a successful load_screenshot function response
            if hasattr(part, 'function_response') and part.function_response:
                func_name = part.function_response.name
                response_data = part.function_response.response

                if func_name == 'load_screenshot' and response_data:
                    if response_data.get('status') == 'success' and 'image' in response_data:
                        image_part = response_data.get('image')

                        if image_part and hasattr(image_part, 'inline_data'):
                            # Add a text cue followed by the actual image
                            new_parts.append(types.Part.from_text(
                                text="[Screenshot] The current kiosk screen is shown below. Analyze carefully:"
                            ))
                            new_parts.append(image_part)
                            print("CALLBACK: Injected screenshot image into LLM context")

        new_contents.append(types.Content(role=content.role, parts=new_parts))

    llm_request.contents = new_contents
    return None
