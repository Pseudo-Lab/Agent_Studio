"""Action agent -- executes UI actions on the kiosk device via ADB.

Two-level structure:
  action_agent  (high-level: parses analysis instructions)
    -> adb_agent  (low-level: click / swipe / type / press / long_click)
"""

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from .tools import adb_click, adb_swipe, adb_type, adb_press, adb_long_click
from .prompt import ACTION_AGENT_PROMPT
from ...shared_libraries import get_model

# Low-level agent that directly calls ADB tool functions
adb_agent = Agent(
    name="adb_agent",
    model=get_model(),
    description="Executes ADB commands on the Android kiosk device",
    instruction="Execute the requested ADB command using the appropriate tool.",
    tools=[adb_click, adb_swipe, adb_type, adb_press, adb_long_click],
    output_key='adb_result',
)

# High-level agent that interprets analysis results and delegates to adb_agent
action_agent = Agent(
    model=get_model(),
    name='action_agent',
    description='Executes UI actions on the kiosk device based on analysis results',
    instruction=ACTION_AGENT_PROMPT,
    tools=[AgentTool(agent=adb_agent)],
    output_key='action_result',
)
