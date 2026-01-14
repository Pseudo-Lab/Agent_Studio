"""VLM (Vision-Language Model) prompt templates."""

# User prompt template for VLM reasoning (no planning)
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

# User prompt template with Plan Mode (step-by-step to-do execution)
USER_PROMPT_WITH_PLAN_TEMPLATE = """
<context>
이전 단계 히스토리:
{thought_history}

⚠️ 중요: 이전 히스토리를 반드시 확인하세요!
- 같은 액션을 2번 이상 반복하고 있다면 다른 접근 방식을 시도해야 합니다.
- 특히 CLICK 액션이 반복된다면 화면이 변하지 않았을 가능성이 높으므로 SWIPE(스크롤)를 고려하세요.
</context>

<plan>
📋 전체 계획:
{plan_steps}

🎯 현재 수행할 단계: [{current_step_num}/{total_steps}] {current_step}
</plan>

<instruction>
현재 단계만 집중해서 수행하세요.

규칙:
- 현재 단계가 완료되었다고 판단되면 thought에 "✅ 단계 완료"를 반드시 포함하세요.
- 아직 진행 중이면 계속 수행하세요.
- 마지막 단계까지 모두 완료되면 FINISH 액션을 수행하세요.

원래 요청: {user_instruction}
</instruction>
"""
