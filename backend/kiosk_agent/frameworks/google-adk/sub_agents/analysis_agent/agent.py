"""Analysis agent -- uses VLM (Gemini) to analyse the kiosk screenshot.

The ``inject_screenshot_callback`` ensures that the actual image bytes
are present in the LLM context when the model processes this agent's turn.
"""

from google.adk.agents import Agent

from .prompt import ANALYSIS_AGENT_PROMPT
from .tools import load_screenshot
from ...shared_libraries import get_model
from ...callbacks import inject_screenshot_callback

analysis_agent = Agent(
    model=get_model(),
    name='analysis_agent',
    description='Analyzes screen captures and determines the next action to take on the kiosk',
    instruction=ANALYSIS_AGENT_PROMPT,
    tools=[load_screenshot],
    before_model_callback=inject_screenshot_callback,
    output_key='analysis_result',
)
