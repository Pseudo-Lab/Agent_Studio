"""Planning Mode prompt templates.

Prompts for task decomposition and web search:
- Unknown entity detection
- Plan generation
- Re-planning during execution
"""

# Unknown Entity Detection Prompt
DETECT_UNKNOWN_PROMPT = """다음 명령에서 일반적으로 알려지지 않은 고유명사, 브랜드명, 또는 특수한 메뉴/상품명이 있는지 확인하세요.

명령: {instruction}

규칙:
- "아메리카노", "카페라떼" 같은 일반적인 메뉴는 알려진 것으로 판단
- "대상혁버거", "민정이네라떼" 같은 특수한 이름은 알려지지 않은 것으로 판단
- 브랜드명이나 매장명이 특이한 경우도 알려지지 않은 것으로 판단

응답 형식 (JSON):
{{"unknown_entities": ["엔티티1", "엔티티2"], "reasoning": "판단 이유"}}

알 수 없는 개념이 없다면:
{{"unknown_entities": [], "reasoning": "모든 개념이 일반적으로 알려진 것입니다."}}
"""

# Plan Generation Prompt
PLAN_GENERATION_PROMPT = """키오스크에서 다음 작업을 완료하기 위한 단계별 계획을 세우세요.

목표: {instruction}
{context_section}

규칙:
1. 각 단계는 구체적이고 실행 가능해야 합니다
2. 키오스크 UI 조작에 필요한 단계만 포함하세요
3. 주관적 선택이 필요한 부분은 "사용자에게 묻기"로 표시
4. 최대 {max_steps}단계 이내로 작성

응답 형식 (JSON):
{{"plan": ["1단계: 메뉴 탭 클릭", "2단계: 카테고리 선택", ...], "reasoning": "계획 수립 이유"}}
"""

# Re-planning Prompt
REPLAN_PROMPT = """현재 진행 상황을 바탕으로 계획을 수정해야 하는지 확인하세요.

원래 목표: {instruction}
원래 계획: {plan}
완료된 단계: {completed_steps}
현재 상태: {current_status}

계획대로 진행 가능하면:
{{"action": "continue", "reasoning": "계획대로 진행 가능"}}

계획 수정이 필요하면:
{{"action": "replan", "new_plan": ["새로운 단계1", ...], "reasoning": "수정 이유"}}
"""
