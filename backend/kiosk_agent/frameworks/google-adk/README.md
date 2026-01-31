# Google ADK Kiosk Agent

LangGraph 기반 키오스크 에이전트를 [Google ADK (Agent Development Kit)](https://google.github.io/adk-docs/) 로 재구현한 모듈입니다.

> **Note:** 웹 UI 시연은 지원하지 않습니다. CLI 전용으로 `run_adk_agent.py` 를 통해 실행합니다.

---

## 아키텍처

```
LoopAgent (kiosk_loop, max_iterations=20)
  └── Orchestrator Agent (kiosk_orchestrator)
        ├── capture_agent   ── ADB 스크린샷 캡처
        ├── analysis_agent  ── VLM 화면 분석 (Gemini)
        ├── action_agent    ── ADB 명령 실행
        │     └── adb_agent ── 저수준 ADB 도구 (click, swipe, type, press, long_click)
        └── status_agent    ── 목표 달성 여부 판단
```

### 반복 워크플로우

```
1. capture_agent  → ADB screencap 으로 스크린샷 캡처
2. load_screenshot → 이미지를 LLM 컨텍스트에 로드
3. analysis_agent → 화면 분석 후 다음 액션 결정 (box_2d 좌표 포함)
4. action_agent   → ADB 명령 실행 (tap, swipe, type 등)
5. status_agent   → 스크린샷 확인 후 목표 달성 여부 판단
   └── 달성 → exit_loop / 미달성 → 다음 반복
```

---

## 폴더 구조

```
google-adk/
├── __init__.py              # 패키지 진입점 (ADKKioskAgent 내보내기)
├── agent.py                 # LoopAgent + Orchestrator 빌드, ADKKioskAgent 클래스
├── callbacks.py             # before_model_callback (스크린샷 이미지 주입)
├── prompt.py                # 오케스트레이터 시스템 프롬프트
├── tools.py                 # 공유 도구 (load_screenshot, exit_loop)
│
├── shared_libraries/
│   ├── __init__.py
│   ├── config.py            # API 키 / 모델명 환경변수 관리
│   └── constants.py         # Status enum (SUCCESS, FAIL, FINISH, ...)
│
├── sub_agents/
│   ├── capture_agent/       # ADB 스크린샷 캡처 + JPEG 압축
│   │   ├── agent.py
│   │   ├── prompt.py
│   │   └── tools.py         # android_capture()
│   │
│   ├── analysis_agent/      # VLM 화면 분석
│   │   ├── agent.py
│   │   ├── prompt.py
│   │   └── tools.py         # load_screenshot()
│   │
│   ├── action_agent/        # ADB 명령 실행
│   │   ├── agent.py         # action_agent + 내부 adb_agent
│   │   ├── prompt.py
│   │   └── tools.py         # adb_click, adb_swipe, adb_type, adb_press, adb_long_click
│   │
│   └── status_agent/        # 목표 달성 여부 판단
│       ├── agent.py
│       ├── prompt.py
│       └── tools.py         # load_screenshot, exit_loop, continue_loop
│
├── tests/
│   ├── conftest.py          # pytest 설정 (임포트 경로, .env 로드)
│   ├── pytest.ini
│   ├── test_unit.py         # 단위 테스트 (23개)
│   └── run_adk_agent.py     # 실제 실행 스크립트
│
└── .venv/                   # Python 3.11 가상환경 (uv)
```

---

## 실행 방법

### 1. 가상환경 세팅 (최초 1회)

```bash
cd Agent_Studio
# uv 로 .venv 생성
uv venv backend/kiosk_agent/frameworks/google-adk/.venv --python 3.11
# 의존성 설치
uv pip install -r backend/requirements.txt --python backend/kiosk_agent/frameworks/google-adk/.venv/bin/python
uv pip install google-adk --python backend/kiosk_agent/frameworks/google-adk/.venv/bin/python
```

### 2. 환경변수 (.env)

프로젝트 루트 `Agent_Studio/.env` 에 다음 키를 설정합니다:

```env
GEMINI_API_KEY=AIzaSy...         # Google AI Studio API 키
GEMINI_MODEL=gemini-2.5-flash    # 사용할 모델명
```

> `GOOGLE_API_KEY` 가 없거나 placeholder 인 경우 `GEMINI_API_KEY` 를 자동으로 사용합니다.

### 3. 에이전트 실행

```bash
cd backend/kiosk_agent/frameworks/google-adk/tests

# 기본 실행 (인터랙티브 프롬프트)
../.venv/bin/python run_adk_agent.py

# 명령어 직접 전달
../.venv/bin/python run_adk_agent.py "버거킹 앱에서 와퍼 주문해줘"

# max iterations 지정
../.venv/bin/python run_adk_agent.py --max-iter 10 "맥도날드 앱 열어줘"
```

### 4. 단위 테스트

```bash
cd backend/kiosk_agent/frameworks/google-adk/tests
../.venv/bin/python -m pytest test_unit.py -v
```

---

## 환경변수 목록

| 변수 | 기본값 | 설명 |
|---|---|---|
| `GEMINI_API_KEY` | (필수) | Google AI Studio API 키 |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini 모델명 |
| `GOOGLE_API_KEY` | `GEMINI_API_KEY` 폴백 | google-adk SDK 가 읽는 키 |
| `SCREENSHOT_JPEG_QUALITY` | `75` | 스크린샷 JPEG 압축 품질 (0-100) |
| `SCREENSHOT_MAX_DIM` | `1280` | 스크린샷 최대 해상도 (px, 0=리사이즈 안 함) |
| `AGENT_MAX_ITERATIONS` | `20` | LoopAgent 최대 반복 횟수 |

---

## 기존 LangGraph 와의 차이

| 항목 | LangGraph | Google ADK |
|---|---|---|
| 프레임워크 | LangGraph StateGraph | ADK Agent / LoopAgent |
| 워크플로우 제어 | 노드 + 엣지 그래프 | 오케스트레이터가 도구 호출로 라우팅 |
| 이미지 전달 | base64 인코딩 | `types.Part.from_bytes()` → session state |
| 루프 관리 | 커스텀 루프 노드 | `LoopAgent(max_iterations=N)` |
| 설정/환경변수 | 동일 (`AgentConfig`, `.env`) | 동일 |
| 코어 모듈 | `ADBController`, `AndroidScreenshotter` | 동일 (재사용) |

---

## 주의사항

- ADB 연결이 필요합니다 (`adb devices` 로 기기가 보여야 함)
- 웹 UI 시연 (Agent Studio 대시보드)은 ADK 에이전트를 지원하지 않습니다. `run_adk_agent.py` CLI 로만 실행 가능합니다.
- 프리뷰 모델 (`gemini-3-flash-preview` 등) 사용 시 응답 시간이 길어질 수 있습니다.
