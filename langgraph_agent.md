# 🎯 LangGraph Meta-Agent 시스템 설계

## 📐 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                  사용자 요구사항 입력                          │
│  "결제 시스템을 위한 멀티 에이전트 워크플로우 만들어줘"          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Meta-Agent System                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           WorkflowBuilderState                        │  │
│  │  - user_requirements                                  │  │
│  │  - state_schema, state_code                          │  │
│  │  - nodes_spec, nodes_code                            │  │
│  │  - tools_spec, tools_code                            │  │
│  │  - graph_structure, edges, routing                   │  │
│  │  - complete_code                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               RAG System (Tools)                      │  │
│  │  ┌─────────────────────────────────────────────┐    │  │
│  │  │  Vector Store (Chroma/FAISS)                │    │  │
│  │  │  - LangGraph 공식 문서                       │    │  │
│  │  │  - 카테고리별 인덱싱                          │    │  │
│  │  │  - 메타데이터 필터링                          │    │  │
│  │  └─────────────────────────────────────────────┘    │  │
│  │                                                        │  │
│  │  ┌─────────────────────────────────────────────┐    │  │
│  │  │  Retriever                                   │    │  │
│  │  │  - Semantic Search                           │    │  │
│  │  │  - Metadata Filtering                        │    │  │
│  │  │  - Hybrid Search                             │    │  │
│  │  └─────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Sequential Pipeline (Nodes)                 │  │
│  │                                                        │  │
│  │  [Stage1] → [Stage2] → [Stage3] → ... → [Stage9]    │  │
│  │  StateD.    NodeD.     ToolI.           Executor     │  │
│  │                                                        │  │
│  │  각 노드는:                                            │  │
│  │  1. 이전 State 읽기                                    │  │
│  │  2. RAG로 관련 문서 검색                               │  │
│  │  3. LLM으로 코드 생성                                  │  │
│  │  4. 검증 및 State 업데이트                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              완성된 LangGraph 코드 출력                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ State 정의

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class WorkflowBuilderState(TypedDict):
    """LangGraph 빌더의 전체 상태"""
    
    # === 입력 ===
    user_requirements: str  # 사용자 요구사항
    
    # === Stage 1: State Designer ===
    state_schema: dict  # {"name": "AgentState", "fields": {...}}
    state_code: str     # 생성된 State 클래스 코드
    
    # === Stage 2: Node Designer ===
    nodes_spec: list[dict]  # [{"name": "call_model", "logic": "..."}]
    nodes_code: str         # 생성된 노드 함수들
    
    # === Stage 3: Tool Integrator ===
    tools_spec: dict    # {"tools": [...], "bindings": {...}}
    tools_code: str     # Tool 설정 코드
    
    # === Stage 4: Graph Assembler ===
    graph_structure: dict   # 그래프 구조 명세
    graph_init_code: str    # StateGraph 초기화 코드
    
    # === Stage 5: Edge Connector ===
    edges_spec: list[dict]  # [{"from": "A", "to": "B"}]
    edges_code: str         # add_edge 코드
    
    # === Stage 6: Conditional Router ===
    routing_spec: dict      # 라우팅 로직 명세
    routing_code: str       # 조건부 엣지 코드
    
    # === Stage 7: Persistence Manager ===
    persistence_config: dict    # Checkpointer 설정
    persistence_code: str       # 영속성 코드
    
    # === Stage 8: Compiler ===
    compile_config: dict    # 컴파일 옵션
    compile_code: str       # compile() 코드
    
    # === Stage 9: Executor ===
    execution_code: str     # 실행 예시 코드
    
    # === 최종 출력 ===
    complete_code: str      # 전체 통합 코드
    errors: list[str]       # 에러 목록
    
    # === RAG 컨텍스트 ===
    retrieved_docs: dict    # {"stage_name": ["doc1", "doc2"]}
```

---

## 2️⃣ RAG 시스템 구축

### Step 1: 문서 수집 및 전처리

```python
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def crawl_langgraph_docs():
    """LangGraph 공식 문서 크롤링"""
    base_url = "https://langchain-ai.github.io/langgraph/"
    
    # 크롤링할 카테고리
    categories = [
        "concepts/low_level",      # State, Nodes, Edges
        "concepts/agentic_concepts",  # ReAct, Router
        "how-tos/persistence",      # Checkpointer
        "how-tos/tool-calling",     # Tools
        "tutorials/introduction",   # 기본 튜토리얼
    ]
    
    docs = []
    for category in categories:
        url = f"{base_url}{category}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 코드 블록 추출
        code_blocks = soup.find_all('pre')
        text_content = soup.get_text()
        
        doc = Document(
            page_content=text_content,
            metadata={
                "source": url,
                "category": category.split('/')[0],
                "subcategory": category.split('/')[-1],
            }
        )
        docs.append(doc)
    
    return docs

# 문서 청크 분할
def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "```", " "]
    )
    
    chunks = text_splitter.split_documents(docs)
    
    # 메타데이터 강화
    for chunk in chunks:
        # 코드 예제 포함 여부
        chunk.metadata["has_code"] = "```python" in chunk.page_content
        
        # 카테고리 태깅
        content_lower = chunk.page_content.lower()
        if "stategraph" in content_lower:
            chunk.metadata["tags"] = ["state", "graph"]
        elif "toolnode" in content_lower:
            chunk.metadata["tags"] = ["tool", "execution"]
        elif "checkpointer" in content_lower:
            chunk.metadata["tags"] = ["persistence", "memory"]
        # ... 추가 태깅
    
    return chunks
```

### Step 2: Vector Store 생성

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vector_store():
    """Vector Store 생성"""
    
    # 1. 문서 수집
    docs = crawl_langgraph_docs()
    chunks = chunk_documents(docs)
    
    # 2. Embedding
    embeddings = OpenAIEmbeddings()
    
    # 3. Vector Store 생성
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./langgraph_vectorstore",
        collection_name="langgraph_docs"
    )
    
    return vectorstore

# Retriever 생성
def create_stage_retriever(vectorstore, stage_name):
    """각 Stage별 특화 Retriever"""
    
    stage_filters = {
        "state_designer": {
            "category": "concepts",
            "tags": ["state", "graph"]
        },
        "node_designer": {
            "category": "concepts",
            "tags": ["node", "function"]
        },
        "tool_integrator": {
            "category": "how-tos",
            "subcategory": "tool-calling"
        },
        "persistence_manager": {
            "category": "how-tos",
            "subcategory": "persistence"
        }
        # ... 각 Stage별 필터
    }
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 5,
            "filter": stage_filters.get(stage_name, {})
        }
    )
    
    return retriever
```

---

## 3️⃣ Sequential Pipeline 구현

### Graph 구조

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Graph Builder
builder = StateGraph(WorkflowBuilderState)

# === Nodes 추가 ===
builder.add_node("stage1_state_designer", stage1_state_designer_node)
builder.add_node("stage2_node_designer", stage2_node_designer_node)
builder.add_node("stage3_tool_integrator", stage3_tool_integrator_node)
builder.add_node("stage4_graph_assembler", stage4_graph_assembler_node)
builder.add_node("stage5_edge_connector", stage5_edge_connector_node)
builder.add_node("stage6_conditional_router", stage6_conditional_router_node)
builder.add_node("stage7_persistence_manager", stage7_persistence_manager_node)
builder.add_node("stage8_compiler", stage8_compiler_node)
builder.add_node("stage9_executor", stage9_executor_node)

# === Sequential Edges ===
builder.add_edge(START, "stage1_state_designer")
builder.add_edge("stage1_state_designer", "stage2_node_designer")
builder.add_edge("stage2_node_designer", "stage3_tool_integrator")
builder.add_edge("stage3_tool_integrator", "stage4_graph_assembler")
builder.add_edge("stage4_graph_assembler", "stage5_edge_connector")
builder.add_edge("stage5_edge_connector", "stage6_conditional_router")
builder.add_edge("stage6_conditional_router", "stage7_persistence_manager")
builder.add_edge("stage7_persistence_manager", "stage8_compiler")
builder.add_edge("stage8_compiler", "stage9_executor")
builder.add_edge("stage9_executor", END)

# Compile
checkpointer = InMemorySaver()
meta_graph = builder.compile(checkpointer=checkpointer)
```

---

## 4️⃣ 각 Stage 노드 구현 예시

### Stage 1: State Designer Node

```python
from langchain_openai import ChatOpenAI

def stage1_state_designer_node(state: WorkflowBuilderState) -> WorkflowBuilderState:
    """State 스키마 설계"""
    
    # 1. RAG로 관련 문서 검색
    retriever = create_stage_retriever(vectorstore, "state_designer")
    docs = retriever.invoke(
        f"State schema design for: {state['user_requirements']}"
    )
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. LLM으로 State 설계
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"""
당신은 LangGraph State 설계 전문가입니다.

사용자 요구사항:
{state['user_requirements']}

참고 문서:
{context}

다음을 생성하세요:
1. State 스키마 (JSON 형식)
2. Python 코드

출력 형식:
```json
{{
  "name": "AgentState",
  "fields": {{
    "messages": {{"type": "Annotated[list, add_messages]", "description": "..."}},
    ...
  }}
}}
```

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    ...
```
"""
    
    response = llm.invoke(prompt)
    
    # 3. 파싱
    state_schema = extract_json(response.content)
    state_code = extract_python_code(response.content)
    
    # 4. 검증
    errors = validate_state_code(state_code)
    
    return {
        "state_schema": state_schema,
        "state_code": state_code,
        "errors": errors,
        "retrieved_docs": {
            "stage1": [doc.metadata["source"] for doc in docs]
        }
    }
```

### Stage 2: Node Designer Node

```python
def stage2_node_designer_node(state: WorkflowBuilderState) -> WorkflowBuilderState:
    """Node 함수 설계"""
    
    # 1. RAG 검색
    retriever = create_stage_retriever(vectorstore, "node_designer")
    docs = retriever.invoke(
        f"Node functions for: {state['user_requirements']}"
    )
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. LLM 호출
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"""
당신은 LangGraph Node 설계 전문가입니다.

이전 단계에서 생성된 State:
{state['state_code']}

사용자 요구사항:
{state['user_requirements']}

참고 문서:
{context}

필요한 노드 함수들을 설계하고 구현하세요.

출력 형식:
1. 노드 명세 (JSON)
2. 노드 함수 코드 (Python)
"""
    
    response = llm.invoke(prompt)
    
    nodes_spec = extract_json(response.content)
    nodes_code = extract_python_code(response.content)
    
    return {
        "nodes_spec": nodes_spec,
        "nodes_code": nodes_code,
        "retrieved_docs": {
            **state["retrieved_docs"],
            "stage2": [doc.metadata["source"] for doc in docs]
        }
    }
```

### Stage 3-9: 동일한 패턴

각 노드는:
1. **이전 State 읽기**
2. **RAG로 관련 문서 검색** (Stage별 특화 필터)
3. **LLM으로 코드 생성**
4. **검증**
5. **State 업데이트**

---

## 5️⃣ 실행 예시

```python
# Vector Store 초기화 (한 번만)
vectorstore = create_vector_store()

# Meta-Graph 실행
config = {"configurable": {"thread_id": "workflow-123"}}

initial_state = {
    "user_requirements": """
    결제 시스템을 위한 멀티 에이전트 워크플로우를 만들어줘.
    
    요구사항:
    - 결제 요청 검증
    - 재고 확인
    - 결제 처리
    - 알림 발송
    - 각 단계에서 실패 시 롤백
    - PostgreSQL로 상태 저장
    """,
    "errors": []
}

# 실행
result = meta_graph.invoke(initial_state, config)

# 결과 출력
print("=== 생성된 LangGraph 코드 ===")
print(result["complete_code"])

# 파일로 저장
with open("generated_workflow.py", "w") as f:
    f.write(result["complete_code"])

print("\n✅ 완성된 워크플로우가 generated_workflow.py에 저장되었습니다!")
```

---

## 6️⃣ 스트리밍으로 중간 과정 확인

```python
# 각 Stage별 진행 상황 모니터링
for chunk in meta_graph.stream(initial_state, config, stream_mode="updates"):
    stage_name = list(chunk.keys())[0]
    stage_data = chunk[stage_name]
    
    print(f"\n{'='*50}")
    print(f"✅ {stage_name} 완료")
    print(f"{'='*50}")
    
    if "state_code" in stage_data:
        print("State 코드 생성됨:")
        print(stage_data["state_code"][:200] + "...")
    
    if "nodes_code" in stage_data:
        print("노드 코드 생성됨:")
        print(stage_data["nodes_code"][:200] + "...")
    
    if "errors" in stage_data and stage_data["errors"]:
        print("⚠️ 에러:", stage_data["errors"])
```

---

## 7️⃣ 장점

### Context7 대비

| 항목 | Context7 | 자체 RAG 시스템 |
|------|----------|----------------|
| 문서 업데이트 | 외부 의존 | 직접 제어 |
| 검색 정확도 | 일반적 | Stage별 특화 가능 |
| 메타데이터 필터링 | 제한적 | 완전 커스터마이징 |
| 오프라인 실행 | 불가 | 가능 |
| 커스텀 청킹 | 불가 | 가능 |
| 비용 | API 호출 | 초기 구축 후 무료 |

### 추가 기능 확장 가능

```python
# 1. 코드 검증 노드 추가
builder.add_node("code_validator", validate_generated_code)
builder.add_edge("stage9_executor", "code_validator")

# 2. 에러 발생 시 재생성
def should_retry(state):
    if state["errors"]:
        return "stage1_state_designer"  # 처음부터 재시작
    return END

builder.add_conditional_edges("code_validator", should_retry)

# 3. 사람 승인 추가 (Human-in-the-loop)
builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["stage8_compiler"]  # 컴파일 전 확인
)
```

---

## 8️⃣ 실제 사용 시나리오

```python
# 시나리오 1: 간단한 챗봇
meta_graph.invoke({
    "user_requirements": "간단한 챗봇. OpenAI GPT-4 사용. 메모리 유지."
})

# 시나리오 2: RAG 에이전트
meta_graph.invoke({
    "user_requirements": "문서 검색 에이전트. Vector store 연동. 소스 출처 표시."
})

# 시나리오 3: 멀티 에이전트 시스템
meta_graph.invoke({
    "user_requirements": """
    리서치 에이전트 시스템:
    - Supervisor가 작업 분배
    - Researcher가 웹 검색
    - Writer가 보고서 작성
    - Reviewer가 검토
    """
})
```
