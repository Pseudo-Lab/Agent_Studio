"""Unit tests for ADK Kiosk Agent components.

venv 에서 실행:
    cd Agent_Studio
    backend/kiosk_agent/frameworks/google-adk/.venv/bin/python -m pytest \
        backend/kiosk_agent/frameworks/google-adk/tests/test_unit.py -v
"""

import asyncio
import os


# ===========================================================================
# 1. shared_libraries 테스트
# ===========================================================================

class TestSharedLibraries:
    """shared_libraries 모듈 테스트."""

    def test_status_enum_values(self):
        from kiosk_agent.frameworks.google_adk_pkg.shared_libraries.constants import Status

        assert Status.SUCCESS.value == "success"
        assert Status.FAIL.value == "fail"
        assert Status.FINISH.value == "finish"
        assert Status.CONTINUE.value == "continue"
        assert Status.ERROR.value == "error"

    def test_get_model_returns_gemini_string(self):
        from kiosk_agent.frameworks.google_adk_pkg.shared_libraries.config import get_model

        model = get_model()
        # .env 에서 GEMINI_MODEL 을 읽으므로 항상 문자열이어야 함
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_model_reads_env(self):
        """GEMINI_MODEL 환경변수가 반영되는지 확인."""
        original = os.environ.get("GEMINI_MODEL")
        try:
            os.environ["GEMINI_MODEL"] = "gemini-test-model"
            from importlib import reload
            import kiosk_agent.frameworks.google_adk_pkg.shared_libraries.config as cfg
            reload(cfg)
            assert cfg.get_model() == "gemini-test-model"
        finally:
            if original is not None:
                os.environ["GEMINI_MODEL"] = original
            else:
                os.environ.pop("GEMINI_MODEL", None)


# ===========================================================================
# 2. Sub-agent 구성 테스트
# ===========================================================================

class TestSubAgents:
    """각 sub-agent 가 올바르게 생성되는지 테스트."""

    def test_capture_agent(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.capture_agent import capture_agent

        assert capture_agent.name == "capture_agent"
        assert len(capture_agent.tools) >= 1

    def test_analysis_agent(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.analysis_agent import analysis_agent

        assert analysis_agent.name == "analysis_agent"
        assert analysis_agent.before_model_callback is not None

    def test_action_agent(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.action_agent import action_agent

        assert action_agent.name == "action_agent"
        # action_agent 는 AgentTool(adb_agent) 를 도구로 가짐
        assert len(action_agent.tools) >= 1

    def test_action_agent_has_adb_agent(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.action_agent.agent import adb_agent

        assert adb_agent.name == "adb_agent"
        # adb_click, adb_swipe, adb_type, adb_press, adb_long_click
        assert len(adb_agent.tools) == 5

    def test_status_agent(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.status_agent import status_agent

        assert status_agent.name == "status_agent"
        # load_screenshot, exit_loop, continue_loop
        assert len(status_agent.tools) == 3
        assert status_agent.before_model_callback is not None


# ===========================================================================
# 3. Orchestrator & LoopAgent 구성 테스트
# ===========================================================================

class TestAgentConstruction:
    """오케스트레이터와 LoopAgent 빌드 테스트."""

    def test_build_orchestrator(self):
        from kiosk_agent.frameworks.google_adk_pkg.agent import _build_orchestrator

        agent = _build_orchestrator()
        assert agent.name == "kiosk_orchestrator"
        # 6개 도구: capture_agent, analysis_agent, action_agent, status_agent,
        #          load_screenshot, exit_loop
        assert len(agent.tools) == 6

    def test_build_loop_agent_default(self):
        from kiosk_agent.frameworks.google_adk_pkg.agent import _build_loop_agent

        loop = _build_loop_agent()
        assert loop.name == "kiosk_loop"
        assert loop.max_iterations == 20

    def test_build_loop_agent_custom_iterations(self):
        from kiosk_agent.frameworks.google_adk_pkg.agent import _build_loop_agent

        loop = _build_loop_agent(max_iterations=7)
        assert loop.max_iterations == 7

    def test_adk_kiosk_agent_class(self):
        from kiosk_agent.frameworks.google_adk_pkg.agent import ADKKioskAgent
        from kiosk_agent.config import AgentConfig

        config = AgentConfig()
        agent = ADKKioskAgent(config)
        assert agent.max_iterations == config.max_iterations
        assert agent.loop_agent.name == "kiosk_loop"


# ===========================================================================
# 4. Callback 테스트
# ===========================================================================

class TestCallbacks:
    """스크린샷 주입 콜백 테스트."""

    def test_callback_passthrough_empty_contents(self):
        """contents 가 빈 리스트면 None 반환 (정상 진행)."""
        from kiosk_agent.frameworks.google_adk_pkg.callbacks import inject_screenshot_callback
        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_req = MagicMock()
        mock_req.contents = []

        result = asyncio.run(inject_screenshot_callback(mock_ctx, mock_req))
        assert result is None


# ===========================================================================
# 5. Tool 함수 시그니처 테스트
# ===========================================================================

class TestToolSignatures:
    """도구 함수들의 시그니처와 기본 동작 확인."""

    def test_adb_click_signature(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.action_agent.tools import adb_click
        import inspect

        sig = inspect.signature(adb_click)
        params = list(sig.parameters.keys())
        assert params == ["y_min", "x_min", "y_max", "x_max"]

    def test_adb_swipe_signature(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.action_agent.tools import adb_swipe
        import inspect

        sig = inspect.signature(adb_swipe)
        assert len(sig.parameters) == 0

    def test_adb_type_signature(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.action_agent.tools import adb_type
        import inspect

        sig = inspect.signature(adb_type)
        assert "text" in sig.parameters

    def test_adb_press_signature(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.action_agent.tools import adb_press
        import inspect

        sig = inspect.signature(adb_press)
        assert "key" in sig.parameters

    def test_exit_loop_signature(self):
        from kiosk_agent.frameworks.google_adk_pkg.tools import exit_loop
        import inspect

        sig = inspect.signature(exit_loop)
        assert "reason" in sig.parameters
        assert "tool_context" in sig.parameters


# ===========================================================================
# 6. Prompt 내용 테스트
# ===========================================================================

class TestPrompts:
    """프롬프트가 키오스크 도메인 키워드를 포함하는지 확인."""

    def test_orchestrator_prompt_contains_workflow(self):
        from kiosk_agent.frameworks.google_adk_pkg.prompt import ORCHESTRATOR_PROMPT

        assert "capture_agent" in ORCHESTRATOR_PROMPT
        assert "analysis_agent" in ORCHESTRATOR_PROMPT
        assert "action_agent" in ORCHESTRATOR_PROMPT
        assert "status_agent" in ORCHESTRATOR_PROMPT

    def test_analysis_prompt_contains_instructions(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.analysis_agent.prompt import ANALYSIS_AGENT_PROMPT

        assert "load_screenshot" in ANALYSIS_AGENT_PROMPT
        assert "box_2d" in ANALYSIS_AGENT_PROMPT

    def test_status_prompt_contains_verification(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.status_agent.prompt import STATUS_AGENT_PROMPT

        assert "exit_loop" in STATUS_AGENT_PROMPT
        assert "continue_loop" in STATUS_AGENT_PROMPT

    def test_action_prompt_contains_tools(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.action_agent.prompt import ACTION_AGENT_PROMPT

        assert "adb_click" in ACTION_AGENT_PROMPT
        assert "adb_swipe" in ACTION_AGENT_PROMPT

    def test_capture_prompt_mentions_session_state(self):
        from kiosk_agent.frameworks.google_adk_pkg.sub_agents.capture_agent.prompt import CAPTURE_AGENT_PROMPT

        assert "session state" in CAPTURE_AGENT_PROMPT
        assert "android_capture" in CAPTURE_AGENT_PROMPT
