"""Configuration for ADK Kiosk Agent.

Reads from the same environment variables as the existing kiosk_agent config
(AgentConfig) so that a single .env file works for both LangGraph and ADK.
"""

import os


def _ensure_google_api_key():
    """Ensure GOOGLE_API_KEY is set for google-adk SDK.

    google-adk reads GOOGLE_API_KEY by default. If it is missing or a
    placeholder, fall back to GEMINI_API_KEY which the project already uses.
    """
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_key or google_key.startswith("your_"):
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key


# Run once on import so the key is available before any ADK client is created.
_ensure_google_api_key()


def get_model():
    """Get model string for Google ADK Agent.

    Uses the same GEMINI_MODEL env var that AgentConfig.ModelConfig uses,
    ensuring consistent model selection across frameworks.
    """
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return gemini_model
