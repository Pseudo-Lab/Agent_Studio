## 📋 전체 파이프라인 구조

```
[요구사항] 
    ↓
[1. State Designer] → State Schema JSON
    ↓
[2. Node Designer] → Node Functions Code
    ↓
[3. Tool Integrator] → Tool Bindings
    ↓
[4. Graph Assembler] → Graph Structure
    ↓
[5. Edge Connector] → Static Edges
    ↓
[6. Conditional Router] → Dynamic Routing
    ↓
[7. Persistence Manager] → Checkpointer Config
    ↓
[8. Compiler Agent] → Compiled Graph
    ↓
[9. Executor Agent] → Execution Code
    ↓
[완성된 LangGraph 애플리케이션]
```

---

## 🎯 Stage 1: State Designer Agent

**역할**: State 스키마 정의 및 데이터 구조 설계

**입력**: 
- 사용자 요구사항 (자연어)
- 필요한 데이터 필드 목록

**출력**: 
```json
{
  "state_schema": {
    "name": "AgentState",
    "fields": {
      "messages": {
        "type": "Annotated[list, add_messages]",
        "description": "대화 히스토리"
      },
      "current_plan": {
        "type": "str",
        "description": "현재 계획"
      },
      "iterations": {
        "type": "int",
        "description": "반복 횟수"
      }
    }
  }
}
```

**참조 문서 카테고리**:
- State management
- TypedDict definitions
- Annotated types & reducers
- MessagesState

**생성 코드 예시**:
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_plan: str
    iterations: int
```

---

## 🎯 Stage 2: Node Designer Agent

**역할**: State를 변환하는 노드 함수 작성

**입력**: 
- Stage 1의 State Schema JSON
- 각 노드의 비즈니스 로직 요구사항

**출력**:
```json
{
  "nodes": [
    {
      "name": "call_model",
      "function_signature": "def call_model(state: AgentState) -> AgentState",
      "logic_description": "LLM을 호출하여 응답 생성",
      "dependencies": ["model"],
      "state_updates": ["messages"]
    },
    {
      "name": "plan",
      "function_signature": "def plan(state: AgentState) -> AgentState",
      "logic_description": "계획 수립",
      "state_updates": ["current_plan", "iterations"]
    }
  ]
}
```

**참조 문서 카테고리**:
- Node functions
- State transformation
- Return value patterns

**생성 코드 예시**:
```python
def call_model(state: AgentState) -> AgentState:
    """LLM을 호출하여 응답 생성"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def plan(state: AgentState) -> AgentState:
    """계획 수립"""
    plan_response = planner.invoke(state["messages"])
    return {
        "current_plan": plan_response.content,
        "iterations": state["iterations"] + 1
    }
```

---

## 🎯 Stage 3: Tool Integrator Agent

**역할**: 외부 도구 통합 및 ToolNode 생성

**입력**:
- Stage 2의 Node 정의
- 필요한 도구 목록

**출력**:
```json
{
  "tools": [
    {
      "name": "tavily_search",
      "type": "TavilySearch",
      "config": {"max_results": 2}
    }
  ],
  "tool_bindings": {
    "model_node": ["tavily_search"]
  },
  "tool_node": {
    "name": "tools",
    "tools": ["tavily_search"]
  }
}
```

**참조 문서 카테고리**:
- ToolNode
- bind_tools
- tools_condition
- Tool execution

**생성 코드 예시**:
```python
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

# 도구 정의
tools = [TavilySearch(max_results=2)]

# 모델에 도구 바인딩
llm_with_tools = model.bind_tools(tools)

# ToolNode 생성
tool_node = ToolNode(tools=tools)
```

---

## 🎯 Stage 4: Graph Assembler Agent

**역할**: StateGraph 객체 생성 및 노드 추가

**입력**:
- Stage 1의 State Schema
- Stage 2의 Node 목록
- Stage 3의 Tool Node

**출력**:
```json
{
  "graph_config": {
    "state_class": "AgentState",
    "nodes": [
      {"name": "call_model", "function": "call_model"},
      {"name": "plan", "function": "plan"},
      {"name": "tools", "function": "tool_node"}
    ]
  }
}
```

**참조 문서 카테고리**:
- StateGraph initialization
- add_node
- Graph structure

**생성 코드 예시**:
```python
from langgraph.graph import StateGraph

# Graph 생성
builder = StateGraph(AgentState)

# 노드 추가
builder.add_node("call_model", call_model)
builder.add_node("plan", plan)
builder.add_node("tools", tool_node)
```

---

## 🎯 Stage 5: Edge Connector Agent

**역할**: 정적 엣지로 노드 연결

**입력**:
- Stage 4의 Graph Structure
- 워크플로우 순서 정의

**출력**:
```json
{
  "static_edges": [
    {"from": "START", "to": "plan"},
    {"from": "plan", "to": "call_model"},
    {"from": "tools", "to": "call_model"}
  ]
}
```

**참조 문서 카테고리**:
- add_edge
- START, END constants
- Edge patterns

**생성 코드 예시**:
```python
from langgraph.graph import START, END

# Entry point 설정
builder.add_edge(START, "plan")

# 정적 연결
builder.add_edge("plan", "call_model")
builder.add_edge("tools", "call_model")
```

---

## 🎯 Stage 6: Conditional Router Agent

**역할**: 조건부 분기 로직 구현

**입력**:
- Stage 5의 Graph with Static Edges
- 분기 조건 정의

**출력**:
```json
{
  "conditional_edges": [
    {
      "source": "call_model",
      "condition_function": "route_after_model",
      "paths": {
        "tools": "tools",
        "end": "END"
      }
    }
  ],
  "routing_functions": [
    {
      "name": "route_after_model",
      "logic": "도구 호출 필요 시 tools, 아니면 END"
    }
  ]
}
```

**참조 문서 카테고리**:
- add_conditional_edges
- Routing functions
- tools_condition
- Dynamic control flow

**생성 코드 예시**:
```python
from langgraph.prebuilt import tools_condition

# 조건 함수 정의
def route_after_model(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 조건부 엣지 추가
builder.add_conditional_edges(
    "call_model",
    route_after_model,
    {
        "tools": "tools",
        END: END
    }
)

# 또는 prebuilt 사용
builder.add_conditional_edges(
    "call_model",
    tools_condition,
)
```

---

## 🎯 Stage 7: Persistence Manager Agent

**역할**: 메모리 및 상태 영속화 설정

**입력**:
- Stage 6의 Complete Graph Structure
- 영속성 요구사항 (메모리 타입, DB 설정)

**출력**:
```json
{
  "persistence_config": {
    "checkpointer_type": "postgres",
    "connection": {
      "db_uri": "postgresql://...",
      "pool_size": 10
    },
    "thread_management": {
      "thread_id_key": "thread_id",
      "namespace": "user_sessions"
    }
  }
}
```

**참조 문서 카테고리**:
- Checkpointers (InMemory, Postgres, MongoDB)
- Persistence patterns
- Cross-thread state
- Store management

**생성 코드 예시**:
```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver

# 개발 환경: InMemory
checkpointer = InMemorySaver()

# 프로덕션 환경: Postgres
DB_URI = "postgresql://postgres:postgres@localhost:5432/db"
checkpointer = PostgresSaver.from_conn_string(DB_URI)
```

---

## 🎯 Stage 8: Compiler Agent

**역할**: 그래프 컴파일 및 최적화

**입력**:
- Stage 7의 Graph + Checkpointer
- 컴파일 옵션 (interrupt, debug)

**출력**:
```json
{
  "compile_config": {
    "checkpointer": "postgres_saver",
    "interrupt_before": ["tools"],
    "interrupt_after": [],
    "debug": false
  },
  "compiled_graph": "<CompiledStateGraph object>"
}
```

**참조 문서 카테고리**:
- compile()
- interrupt_before/after
- Debug mode
- Graph optimization

**생성 코드 예시**:
```python
# 기본 컴파일
graph = builder.compile(checkpointer=checkpointer)

# Human-in-the-loop
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"],
)

# 디버그 모드
graph = builder.compile(
    checkpointer=checkpointer,
    debug=True
)
```

---

## 🎯 Stage 9: Executor Agent

**역할**: 실행 코드 생성 및 최적화

**입력**:
- Stage 8의 Compiled Graph
- 실행 모드 요구사항 (동기/비동기, 스트리밍)

**출력**:
```json
{
  "execution_config": {
    "mode": "stream",
    "stream_mode": "values",
    "config": {
      "configurable": {
        "thread_id": "user-123"
      }
    }
  },
  "execution_code": "<Python code>"
}
```

**참조 문서 카테고리**:
- invoke vs stream vs astream
- stream_mode options
- RunnableConfig
- Async execution

**생성 코드 예시**:
```python
# 1. 단순 실행 (invoke)
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"messages": [{"role": "user", "content": "Hello"}]}, config)

# 2. 스트리밍 (stream)
for chunk in graph.stream(inputs, config, stream_mode="values"):
    print(chunk["messages"][-1])

# 3. 비동기 스트리밍 (astream)
async for chunk in graph.astream(inputs, config, stream_mode="updates"):
    print(chunk)

# 4. 디버깅용 stream_mode
for chunk in graph.stream(inputs, config, stream_mode="debug"):
    print(chunk)
```

---

## 🔗 Sequential Pipeline 통합 예시

```python
# ====== STAGE 1: State Designer ======
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    iterations: int

# ====== STAGE 2: Node Designer ======
def planner(state: AgentState):
    plan_response = planning_llm.invoke(state["messages"])
    return {"plan": plan_response.content, "iterations": state["iterations"] + 1}

def agent(state: AgentState):
    response = agent_llm.invoke(state["messages"])
    return {"messages": [response]}

# ====== STAGE 3: Tool Integrator ======
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

tools = [TavilySearch(max_results=2)]
agent_llm_with_tools = agent_llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)

# ====== STAGE 4: Graph Assembler ======
from langgraph.graph import StateGraph

builder = StateGraph(AgentState)
builder.add_node("planner", planner)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

# ====== STAGE 5: Edge Connector ======
from langgraph.graph import START, END

builder.add_edge(START, "planner")
builder.add_edge("planner", "agent")
builder.add_edge("tools", "agent")

# ====== STAGE 6: Conditional Router ======
def route_agent(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

builder.add_conditional_edges("agent", route_agent)

# ====== STAGE 7: Persistence Manager ======
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/db"
checkpointer = PostgresSaver.from_conn_string(DB_URI)

# ====== STAGE 8: Compiler ======
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]
)

# ====== STAGE 9: Executor ======
config = {"configurable": {"thread_id": "user-123"}}

for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "안녕하세요"}]},
    config,
    stream_mode="values"
):
    print(chunk["messages"][-1])
```

---

## 📊 각 Stage별 LangGraph 문서 매핑

| Stage | Agent | 필요한 문서 카테고리 | Context7 검색 키워드 |
|-------|-------|---------------------|---------------------|
| 1 | State Designer | State, TypedDict, Annotated | `state management schema annotated` |
| 2 | Node Designer | Node functions, transformations | `node functions state transformation` |
| 3 | Tool Integrator | ToolNode, bind_tools | `tool execution toolnode bind_tools` |
| 4 | Graph Assembler | StateGraph, add_node | `stategraph initialization add_node` |
| 5 | Edge Connector | add_edge, START, END | `edges start end connections` |
| 6 | Conditional Router | add_conditional_edges | `conditional routing branching` |
| 7 | Persistence Manager | Checkpointers, memory | `checkpointer persistence memory` |
| 8 | Compiler | compile, interrupt | `compile interrupt debugging` |
| 9 | Executor | invoke, stream, config | `execution streaming invoke` |

---

## 🎯 Pipeline 실행 전략

### Option 1: Full Sequential (완전 순차)
```
Stage 1 완료 → Stage 2 시작 → ... → Stage 9 완료
```
- **장점**: 단순, 디버깅 용이
- **단점**: 느림, 병렬화 불가

### Option 2: Phased Parallel (단계별 병렬)
```
[Stage 1] → [Stage 2 + Stage 3 병렬] → [Stage 4-6 순차] → [Stage 7-9 순차]
```
- **장점**: 일부 병렬화 가능
- **단점**: 의존성 관리 필요

### Option 3: Micro-Pipeline (마이크로 파이프라인)
```
각 노드별로 Stage 1-6 반복 → 통합 → Stage 7-9
```
- **장점**: 노드별 독립 개발
- **단점**: 통합 복잡도 증가

---

## 💡 각 Agent의 Context7 활용 전략

```python
# Stage 1: State Designer
context = get_library_docs(
    "/websites/langchain-ai_github_io_langgraph",
    topic="state management annotated typeddict"
)

# Stage 2: Node Designer
context = get_library_docs(
    "/websites/langchain-ai_github_io_langgraph",
    topic="node functions transformation"
)

# Stage 3: Tool Integrator
context = get_library_docs(
    "/websites/langchain-ai_github_io_langgraph",
    topic="toolnode tool execution bind_tools"
)

# ... 각 Stage마다 필요한 문서만 focused 검색
```
