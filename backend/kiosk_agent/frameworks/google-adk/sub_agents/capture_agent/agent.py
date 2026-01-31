"""Capture agent -- takes a screenshot from the Android kiosk via ADB."""

from google.adk.agents import Agent

from .tools import android_capture
from .prompt import CAPTURE_AGENT_PROMPT
from ...shared_libraries import get_model

capture_agent = Agent(
    model=get_model(),
    name='capture_agent',
    description='Captures the current screen state from the Android kiosk device',
    instruction=CAPTURE_AGENT_PROMPT,
    tools=[android_capture],
    output_key='capture_result',
)
