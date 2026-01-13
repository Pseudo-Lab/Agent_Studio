"""Prompt templates for LangGraph Kiosk Agent."""

# User prompt template for VLM reasoning
USER_PROMPT_TEMPLATE = """
<context>
이전 단계 히스토리:
{thought_history}

⚠️ 중요: 이전 히스토리를 반드시 확인하세요!
- 같은 액션을 2번 이상 반복하고 있다면 다른 접근 방식을 시도해야 합니다.
- 특히 CLICK 액션이 반복된다면 화면이 변하지 않았을 가능성이 높으므로 SWIPE(스크롤)를 고려하세요.
- 스크롤로도 원하는 요소를 찾지 못한다면 BACK 또는 다른 경로를 시도하세요.
</context>

<user_instruction>
{user_instruction}
</user_instruction>
"""
