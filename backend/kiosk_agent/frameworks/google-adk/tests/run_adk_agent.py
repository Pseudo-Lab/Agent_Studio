#!/usr/bin/env python3
"""
ADK Kiosk Agent - Interactive Runner

실제 Android 기기(ADB)에 연결된 상태에서 ADK 기반 키오스크 에이전트를 실행합니다.

Usage:
    # 기본 실행 (프롬프트 입력)
    python run_adk_agent.py

    # 명령어 직접 전달
    python run_adk_agent.py "버거킹 앱에서 와퍼 주문해줘"

    # max iterations 지정
    python run_adk_agent.py --max-iter 10 "맥도날드 앱 열어줘"

Examples:
    python run_adk_agent.py "버거킹 앱에서 와퍼 세트 주문해줘"
    python run_adk_agent.py "맥도날드 앱 열어서 빅맥 주문해줘"
    python run_adk_agent.py "배달의민족에서 짜장면 검색해줘"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ADK_ROOT = _THIS_DIR.parent                          # google-adk/
_FRAMEWORKS_DIR = _ADK_ROOT.parent                     # frameworks/
_KIOSK_AGENT_DIR = _FRAMEWORKS_DIR.parent              # kiosk_agent/
_BACKEND_DIR = _KIOSK_AGENT_DIR.parent                 # backend/
_PROJECT_ROOT = _BACKEND_DIR.parent                    # Agent_Studio/

for p in [str(_BACKEND_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Register google-adk package under importable name
import importlib.util
spec = importlib.util.spec_from_file_location(
    "kiosk_agent.frameworks.google_adk_pkg",
    _ADK_ROOT / "__init__.py",
    submodule_search_locations=[str(_ADK_ROOT)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# Load .env
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

# Ensure GOOGLE_API_KEY is set from GEMINI_API_KEY before any ADK import
import os
_gk = os.environ.get("GOOGLE_API_KEY", "")
if not _gk or _gk.startswith("your_"):
    _gemk = os.environ.get("GEMINI_API_KEY", "")
    if _gemk:
        os.environ["GOOGLE_API_KEY"] = _gemk


async def run_agent(instruction: str, max_iterations: int = 20):
    """Run the ADK Kiosk Agent."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    from kiosk_agent.frameworks.google_adk_pkg.agent import _build_loop_agent

    print(f"\n{'=' * 60}")
    print(f"  ADK Kiosk Agent")
    print(f"  Model: {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Goal: {instruction}")
    print(f"{'=' * 60}\n")

    loop_agent = _build_loop_agent(max_iterations=max_iterations)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=loop_agent,
        app_name="kiosk_agent_adk",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="kiosk_agent_adk",
        user_id="kiosk_user",
    )

    user_message = Content(
        role="user",
        parts=[Part(text=f"Goal: {instruction}\n\nStart by capturing a screenshot from the kiosk device.")],
    )

    print("Running agent...\n")

    try:
        async for event in runner.run_async(
            user_id="kiosk_user",
            session_id=session.id,
            new_message=user_message,
        ):
            if not (event.content and event.content.parts):
                continue

            for part in event.content.parts:
                # Text output
                if hasattr(part, "text") and part.text:
                    text = part.text.strip()
                    if text:
                        # Truncate long text
                        display = text[:300] + "..." if len(text) > 300 else text
                        print(f"  [{event.author}] {display}")

                # Function call
                if hasattr(part, "function_call") and part.function_call:
                    print(f"  [{event.author}] -> {part.function_call.name}()")

                # Function response
                if hasattr(part, "function_response") and part.function_response:
                    resp = part.function_response
                    if hasattr(resp, "response") and isinstance(resp.response, dict):
                        status = resp.response.get("status", "")
                        msg = resp.response.get("message", resp.response.get("reason", ""))
                        if msg:
                            print(f"  [{event.author}] <- {status}: {str(msg)[:200]}")

                        if status == "finish":
                            print(f"\n{'=' * 60}")
                            print(f"  FINISHED: {resp.response.get('reason', 'done')}")
                            print(f"{'=' * 60}\n")
                            return True

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'=' * 60}")
    print(f"  Max iterations reached ({max_iterations})")
    print(f"{'=' * 60}\n")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run ADK Kiosk Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_adk_agent.py "버거킹 앱에서 와퍼 주문해줘"
  python run_adk_agent.py --max-iter 10 "맥도날드 앱 열어줘"
        """,
    )
    parser.add_argument("instruction", nargs="?", help="Goal instruction")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=int(os.getenv("AGENT_MAX_ITERATIONS", "20")),
        help="Maximum loop iterations (default: from env or 20)",
    )
    args = parser.parse_args()

    instruction = args.instruction
    if not instruction:
        print("\n  ADK Kiosk Agent - Interactive Mode")
        print("  " + "-" * 40)
        instruction = input("  Enter your goal: ").strip()
        if not instruction:
            instruction = "버거킹 앱에서 와퍼 주문해줘"
            print(f"  (Using default: {instruction})")

    success = asyncio.run(run_agent(instruction, max_iterations=args.max_iter))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
