"""Status agent -- verifies goal achievement and controls the loop.

Loads the latest screenshot, compares it against the user's goal, and
calls either ``exit_loop`` (done) or ``continue_loop`` (next iteration).
"""

from google.adk.agents import Agent

from .tools import load_screenshot, exit_loop, continue_loop
from .prompt import STATUS_AGENT_PROMPT
from ...shared_libraries import get_model
from ...callbacks import inject_screenshot_callback

status_agent = Agent(
    model=get_model(),
    name='status_agent',
    description='Evaluates task progress and decides whether to continue or finish the loop',
    instruction=STATUS_AGENT_PROMPT,
    tools=[load_screenshot, exit_loop, continue_loop],
    before_model_callback=inject_screenshot_callback,
    output_key='agent_status',
)
