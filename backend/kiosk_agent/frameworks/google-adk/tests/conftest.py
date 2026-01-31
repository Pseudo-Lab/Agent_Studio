"""Pytest conftest - registers the google-adk package for importability.

venv 에서 실행:
    cd Agent_Studio
    backend/kiosk_agent/frameworks/google-adk/.venv/bin/python -m pytest \
        backend/kiosk_agent/frameworks/google-adk/tests/ -v
"""

import importlib.util
import sys
from pathlib import Path

_ADK_ROOT = Path(__file__).resolve().parent.parent  # google-adk/
_BACKEND_DIR = _ADK_ROOT.parent.parent.parent       # backend/
_PROJECT_ROOT = _BACKEND_DIR.parent                  # Agent_Studio/

# backend 경로를 sys.path 에 추가
for p in [str(_BACKEND_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# google-adk 디렉터리를 Python importable 이름으로 등록
_pkg_name = "kiosk_agent.frameworks.google_adk_pkg"
if _pkg_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _pkg_name,
        _ADK_ROOT / "__init__.py",
        submodule_search_locations=[str(_ADK_ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_pkg_name] = mod
    spec.loader.exec_module(mod)

# .env 로드
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

# Ensure GOOGLE_API_KEY is set from GEMINI_API_KEY
import os
_gk = os.environ.get("GOOGLE_API_KEY", "")
if not _gk or _gk.startswith("your_"):
    _gemk = os.environ.get("GEMINI_API_KEY", "")
    if _gemk:
        os.environ["GOOGLE_API_KEY"] = _gemk
