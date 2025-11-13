## 📋 아키텍처

```
┌─────────────────────────────────────────────┐
│         사용자 요구사항 입력                   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│     Stage 0: Requirements Analyzer           │
│  - 요구사항 분석                              │
│  - 필요한 Stage 결정                         │
│  - required_stages: [1, 2, 4, 5, 8, 9]     │
│  - 불필요: [3, 6, 7] (Tool, Router, DB)     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│     Dynamic Pipeline Builder                 │
│  - 선택된 Stage만으로 그래프 생성             │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│     Execute (필요한 Stage만 실행)            │
│  Stage1 → Stage2 → Stage4 → Stage5 → ...   │
└─────────────────────────────────────────────┘
```

---

## 1️⃣ 개선된 State 정의

```python
from typing import TypedDict, Literal

class WorkflowBuilderState(TypedDict):
    """LangGraph 빌더의 전체 상태"""
    
    # === Stage 0: Requirements Analysis ===
    user_requirements: str
    required_stages: list[int]  # [1, 2, 4, 5, 8, 9] 형태
    stage_skip_reasons: dict    # {3: "도구 사용 없음", 6: "단순 순차"}
    workflow_complexity: Literal["simple", "medium", "complex"]
    
    # === Stage별 출력 (동일) ===
    state_schema: dict
    state_code: str
    nodes_spec: list[dict]
    nodes_code: str
    tools_spec: dict  # Stage 3 스킵 시 None
    tools_code: str
    # ... 나머지 동일
    
    # === 실행 메타데이터 ===
    executed_stages: list[int]  # 실제 실행된 Stage 목록
    total_execution_time: float
```

---

## 2️⃣ Stage 0: Requirements Analyzer

```python
def stage0_requirements_analyzer(
    state: WorkflowBuilderState
) -> WorkflowBuilderState:
    """요구사항 분석 및 필요 Stage 결정"""
    
    requirements = state["user_requirements"]
    
    # RAG로 유사 사례 검색
    retriever = vectorstore.as_retriever()
    similar_cases = retriever.invoke(
        f"LangGraph workflow patterns: {requirements}"
    )
    
    # LLM으로 분석
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    analysis_prompt = f"""
당신은 LangGraph 아키텍처 전문가입니다.

사용자 요구사항:
{requirements}

다음 Stage들이 필요한지 판단하세요:

1. State Designer (항상 필수)
2. Node Designer (항상 필수)
3. Tool Integrator - 외부 API, 검색, 데이터베이스 등 도구 필요 시
4. Graph Assembler (항상 필수)
5. Edge Connector (항상 필수)
6. Conditional Router - 조건부 분기, 동적 라우팅 필요 시
7. Persistence Manager - 대화 기록, 메모리, 세션 유지 필요 시
8. Compiler (항상 필수)
9. Executor (항상 필수)

판단 기준:
- Tool 필요 키워드: "검색", "API", "웹", "크롤링", "데이터베이스", "외부 시스템"
- Router 필요 키워드: "조건", "분기", "판단", "선택", "라우팅", "동적"
- Persistence 필요 키워드: "기억", "대화 기록", "메모리", "세션", "이어서", "저장"

출력 JSON:
{{
  "required_stages": [1, 2, 3, 4, 5, 6, 7, 8, 9],  // 필요한 Stage 번호
  "skip_reasons": {{
    "3": "외부 도구 사용 없음",
    "6": "순차적 실행만 필요"
  }},
  "workflow_complexity": "simple|medium|complex",
  "reasoning": "판단 근거"
}}
"""
    
    response = llm.invoke(analysis_prompt)
    analysis = json.loads(extract_json(response.content))
    
    return {
        "required_stages": analysis["required_stages"],
        "stage_skip_reasons": analysis["skip_reasons"],
        "workflow_complexity": analysis["workflow_complexity"]
    }
```

---

## 3️⃣ Dynamic Pipeline Builder

```python
def build_dynamic_pipeline(required_stages: list[int]) -> StateGraph:
    """필요한 Stage만으로 동적 그래프 생성"""
    
    # 모든 Stage 정의
    all_stages = {
        1: ("stage1_state_designer", stage1_state_designer_node),
        2: ("stage2_node_designer", stage2_node_designer_node),
        3: ("stage3_tool_integrator", stage3_tool_integrator_node),
        4: ("stage4_graph_assembler", stage4_graph_assembler_node),
        5: ("stage5_edge_connector", stage5_edge_connector_node),
        6: ("stage6_conditional_router", stage6_conditional_router_node),
        7: ("stage7_persistence_manager", stage7_persistence_manager_node),
        8: ("stage8_compiler", stage8_compiler_node),
        9: ("stage9_executor", stage9_executor_node),
    }
    
    # 그래프 생성
    builder = StateGraph(WorkflowBuilderState)
    
    # 필요한 노드만 추가
    selected_nodes = []
    for stage_num in sorted(required_stages):
        if stage_num in all_stages:
            node_name, node_func = all_stages[stage_num]
            builder.add_node(node_name, node_func)
            selected_nodes.append(node_name)
    
    # 순차 연결
    builder.add_edge(START, selected_nodes[0])
    for i in range(len(selected_nodes) - 1):
        builder.add_edge(selected_nodes[i], selected_nodes[i + 1])
    builder.add_edge(selected_nodes[-1], END)
    
    return builder.compile()
```

---

## 4️⃣ 통합 Meta-Graph

```python
from langgraph.graph import StateGraph, START, END

def create_adaptive_meta_graph():
    """적응형 메타 그래프 생성"""
    
    builder = StateGraph(WorkflowBuilderState)
    
    # Stage 0: Requirements Analyzer
    builder.add_node("analyzer", stage0_requirements_analyzer)
    
    # Stage 1-9: Dynamic Executor
    builder.add_node("dynamic_executor", dynamic_executor_node)
    
    # 연결
    builder.add_edge(START, "analyzer")
    builder.add_edge("analyzer", "dynamic_executor")
    builder.add_edge("dynamic_executor", END)
    
    return builder.compile(checkpointer=InMemorySaver())


def dynamic_executor_node(
    state: WorkflowBuilderState
) -> WorkflowBuilderState:
    """필요한 Stage만 동적 실행"""
    
    # 동적 파이프라인 생성
    pipeline = build_dynamic_pipeline(state["required_stages"])
    
    # 실행
    result = pipeline.invoke(state)
    
    return {
        **result,
        "executed_stages": state["required_stages"]
    }
```

---

## 5️⃣ 실전 예시

### 예시 1: 간단한 챗봇 (Tool, Router, Persistence 불필요)

```python
meta_graph = create_adaptive_meta_graph()

result = meta_graph.invoke({
    "user_requirements": """
    간단한 Q&A 챗봇 만들어줘.
    - 질문 받고 답변만 함
    - 외부 검색 필요 없음
    - 단순 대화만
    """
})

print("필요한 Stage:", result["required_stages"])
# 출력: [1, 2, 4, 5, 8, 9]

print("스킵된 Stage:", result["stage_skip_reasons"])
# 출력: {
#   "3": "외부 도구 사용 없음",
#   "6": "조건부 라우팅 불필요 (단순 순차)",
#   "7": "메모리 유지 불필요"
# }
```

### 예시 2: 복잡한 RAG 에이전트 (모든 Stage 필요)

```python
result = meta_graph.invoke({
    "user_requirements": """
    고급 RAG 에이전트 시스템:
    - Vector DB 검색
    - 웹 검색 도구
    - 조건부 라우팅 (관련성 판단)
    - 대화 기록 저장
    - PostgreSQL persistence
    """
})

print("필요한 Stage:", result["required_stages"])
# 출력: [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 전부!

print("워크플로우 복잡도:", result["workflow_complexity"])
# 출력: "complex"
```

### 예시 3: 중간 복잡도 (일부 Stage만)

```python
result = meta_graph.invoke({
    "user_requirements": """
    문서 요약 시스템:
    - PDF 업로드 받기
    - 청크로 분할
    - 요약 생성
    - 단순 순차 처리
    - 일회성 실행 (메모리 불필요)
    """
})

print("필요한 Stage:", result["required_stages"])
# 출력: [1, 2, 4, 5, 8, 9]

print("스킵된 Stage:", result["stage_skip_reasons"])
# 출력: {
#   "3": "PDF 처리는 노드 내부에서 처리 가능",
#   "6": "순차 처리만 필요",
#   "7": "일회성 실행"
# }
```

---

## 6️⃣ Stage별 필요 조건 매트릭스

| Stage | 이름 | 필수 여부 | 필요 조건 |
|-------|------|----------|-----------|
| 1 | State Designer | ✅ 항상 필수 | - |
| 2 | Node Designer | ✅ 항상 필수 | - |
| 3 | Tool Integrator | ⚠️ 조건부 | 외부 API, 검색, DB, 도구 사용 시 |
| 4 | Graph Assembler | ✅ 항상 필수 | - |
| 5 | Edge Connector | ✅ 항상 필수 | - |
| 6 | Conditional Router | ⚠️ 조건부 | 조건부 분기, 동적 라우팅 시 |
| 7 | Persistence Manager | ⚠️ 조건부 | 대화 기록, 메모리, 세션 유지 시 |
| 8 | Compiler | ✅ 항상 필수 | - |
| 9 | Executor | ✅ 항상 필수 | - |

---

## 7️⃣ Requirements Analyzer의 판단 로직

```python
# 키워드 기반 자동 판단
STAGE_DETECTION_RULES = {
    3: {  # Tool Integrator
        "keywords": ["검색", "API", "웹", "크롤링", "데이터베이스", "외부", "tool", "function call"],
        "required_if": "any_keyword_present"
    },
    6: {  # Conditional Router
        "keywords": ["조건", "분기", "판단", "선택", "라우팅", "동적", "if", "routing"],
        "required_if": "any_keyword_present"
    },
    7: {  # Persistence Manager
        "keywords": ["기억", "대화 기록", "메모리", "세션", "이어서", "저장", "persistence", "checkpoint"],
        "required_if": "any_keyword_present"
    }
}

def auto_detect_required_stages(requirements: str) -> list[int]:
    """키워드 기반 자동 Stage 감지"""
    
    required = [1, 2, 4, 5, 8, 9]  # 항상 필수
    
    requirements_lower = requirements.lower()
    
    for stage_num, rules in STAGE_DETECTION_RULES.items():
        keywords = rules["keywords"]
        if any(kw in requirements_lower for kw in keywords):
            required.append(stage_num)
    
    return sorted(required)
```

---

## 8️⃣ 실행 시간 비교

```python
# 전체 Stage 실행 (9개)
start = time.time()
result_full = meta_graph.invoke({"user_requirements": "..."})
time_full = time.time() - start
print(f"전체 실행: {time_full:.2f}초")

# 필요한 Stage만 실행 (6개)
start = time.time()
result_optimized = meta_graph.invoke({"user_requirements": "..."})
time_optimized = time.time() - start
print(f"최적화 실행: {time_optimized:.2f}초")

print(f"시간 절약: {((time_full - time_optimized) / time_full * 100):.1f}%")
```

**예상 결과:**
```
전체 실행: 45.2초
최적화 실행: 28.7초
시간 절약: 36.5%
```

---

## 9️⃣ 사용자 피드백 루프

```python
def stage0_with_confirmation(
    state: WorkflowBuilderState
) -> WorkflowBuilderState:
    """Stage 선택 결과를 사용자에게 확인"""
    
    analysis = stage0_requirements_analyzer(state)
    
    print("\n" + "="*50)
    print("📋 워크플로우 분석 결과")
    print("="*50)
    print(f"복잡도: {analysis['workflow_complexity']}")
    print(f"\n실행할 Stage: {analysis['required_stages']}")
    print(f"\n스킵할 Stage:")
    for stage, reason in analysis['stage_skip_reasons'].items():
        print(f"  - Stage {stage}: {reason}")
    
    # 사용자 확인
    confirm = input("\n이대로 진행하시겠습니까? (y/n): ")
    
    if confirm.lower() == 'y':
        return analysis
    else:
        # 수동 조정
        print("\n어떤 Stage를 추가/제거하시겠습니까?")
        # ... 인터랙티브 조정 로직
```
