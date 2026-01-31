"""Agent framework implementations."""

from .base import BaseAgent
from .langgraph.agent import KioskAgent, LangGraphKioskAgent

# Google ADK implementation
# The directory is named "google-adk" (with hyphen), so we use importlib.util
# to load it since Python identifiers cannot contain hyphens.
try:
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _adk_pkg_dir = _Path(__file__).parent / "google-adk"
    _spec = _ilu.spec_from_file_location(
        "kiosk_agent.frameworks.google_adk",
        _adk_pkg_dir / "__init__.py",
        submodule_search_locations=[str(_adk_pkg_dir)],
    )
    _adk_mod = _ilu.module_from_spec(_spec)
    import sys as _sys
    _sys.modules[_spec.name] = _adk_mod
    _spec.loader.exec_module(_adk_mod)
    ADKKioskAgent = _adk_mod.ADKKioskAgent
    KioskAgentADK = _adk_mod.KioskAgentADK
except Exception:
    # google-adk dependencies (google.adk) may not be installed
    ADKKioskAgent = None  # type: ignore[assignment,misc]
    KioskAgentADK = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseAgent",
    "KioskAgent",
    "LangGraphKioskAgent",  # Backwards compatibility
    "ADKKioskAgent",
    "KioskAgentADK",
]
