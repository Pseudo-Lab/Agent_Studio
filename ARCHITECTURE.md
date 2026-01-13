# Kiosk Agent Architecture

## Overview

키오스크 에이전트는 Vision-Language-Action (VLA) 패러다임을 기반으로 Android 키오스크를 자동으로 제어하는 AI 에이전트입니다.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend (Next.js)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Chat Panel  │  │ Thought     │  │ State       │  │ HITL Card   │    │
│  │             │  │ Panel       │  │ Panel       │  │             │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ SSE
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API Server (FastAPI)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ /agent/*    │  │ /stt/*      │  │ /tts/*      │  │ Session     │    │
│  │ endpoints   │  │ endpoints   │  │ endpoints   │  │ Store       │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Agent Core                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Framework Layer                             │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │   │
│  │  │ LangGraph │  │ CrewAI    │  │ MS Agent  │  │ Google    │    │   │
│  │  │ (Default) │  │ (TBD)     │  │ (TBD)     │  │ ADK (TBD) │    │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         Core Modules                              │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │  │
│  │  │ LLM       │  │ Perception│  │ Control   │  │ Translator│     │  │
│  │  │ Clients   │  │ (Screenshot)│ │ (ADB)     │  │           │     │  │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         Voice Modules                             │  │
│  │  ┌───────────────────┐  ┌───────────────────┐                    │  │
│  │  │ TTS (CosyVoice3)  │  │ STT (Google Cloud)│                    │  │
│  │  └───────────────────┘  └───────────────────┘                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Android Device (ADB)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ screencap   │  │ input tap   │  │ input swipe │  │ input text  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## VLA Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VLA Loop                                      │
│                                                                         │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐      │
│   │ Screen  │ ───► │  VLM    │ ───► │ Action  │ ───► │ Execute │      │
│   │ Capture │      │ Reason  │      │ Decode  │      │  ADB    │      │
│   └─────────┘      └─────────┘      └─────────┘      └─────────┘      │
│        ▲                                                    │          │
│        │                                                    │          │
│        └────────────────────────────────────────────────────┘          │
│                                                                         │
│   Termination Conditions:                                              │
│   - FINISH action (payment screen reached)                             │
│   - INTERRUPT action (human input required)                            │
│   - Max iterations reached                                             │
│   - User abort                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## LangGraph State Machine

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
             ┌──────│     VLM     │◄────────┐
             │      └──────┬──────┘         │
             │             │                │
             │             ▼                │
             │      ┌─────────────┐         │
             │      │   EXECUTE   │         │
             │      └──────┬──────┘         │
             │             │                │
             │             ▼                │
             │      ┌─────────────┐         │
             │      │   ROUTER    │─────────┤
             │      └──────┬──────┘         │
             │             │                │
             │    ┌────────┼────────┐       │
             │    │        │        │       │
             │    ▼        ▼        ▼       │
             │ ┌─────┐ ┌─────────┐ ┌───┐   │
             │ │LOOP │ │  HUMAN  │ │END│   │
             │ └──┬──┘ └────┬────┘ └───┘   │
             │    │         │              │
             │    │    ┌────┴────┐         │
             │    │    │         │         │
             │    │    ▼         ▼         │
             │    │ ┌──────┐ ┌──────┐      │
             └────┼─│RESUME│ │ABORT │      │
                  │ └──┬───┘ └──┬───┘      │
                  │    │        │          │
                  └────┘        │          │
                               ▼          │
                            ┌─────┐       │
                            │ END │◄──────┘
                            └─────┘
```

## Key Components

### 1. LLM Clients

| Client | Provider | Model |
|--------|----------|-------|
| Gemini | Google | gemini-3-flash-preview |
| OpenAI | OpenAI | gpt-4o-mini |
| Local | Self-hosted | AgentCPM-GUI |

### 2. Action Types

| Action | Description | box_2d |
|--------|-------------|--------|
| CLICK | 탭 | Required |
| LONG_CLICK | 길게 누르기 | Required |
| SWIPE | 스크롤 | [0,0,0,0] |
| INPUT | 텍스트 입력 | Optional |
| BACK | 뒤로가기 | [0,0,0,0] |
| HOME | 홈 화면 | [0,0,0,0] |
| INTERRUPT | HITL | [0,0,0,0] |
| FINISH | 완료 | [0,0,0,0] |

### 3. Human-in-the-Loop

INTERRUPT 액션이 발생하면:

1. 에이전트가 `interrupt` 필드와 함께 질문
2. API가 `waiting_human` 상태로 전환
3. 프론트엔드에서 선택지 표시
4. 사용자 응답 후 `/agent/respond` 호출
5. 에이전트 실행 재개

### 4. Voice Pipeline

```
[Microphone] → [STT API] → [Text] → [Agent] → [Response] → [TTS] → [Speaker]
                  ▲                                           │
            Google Cloud                               CosyVoice3 (MLX)
```

## Data Flow

1. **User Input**: 프론트엔드에서 명령 입력
2. **API Request**: `/agent/start` POST 요청
3. **Agent Init**: LangGraph 워크플로우 초기화
4. **VLA Loop**: 스크린샷 → VLM 추론 → ADB 실행
5. **SSE Stream**: 각 스텝마다 `STATE_SNAPSHOT` 이벤트
6. **Completion**: `RUN_FINISHED` 이벤트 및 TTS 생성

## Extensibility

### Adding New LLM Provider

```python
# backend/kiosk_agent/llm/new_provider.py
from .base import BaseLLM

class NewProvider(BaseLLM):
    def generate(self, instruction: str, image: Image.Image) -> str:
        # Implementation
        pass
```

### Adding New Framework

```python
# backend/kiosk_agent/frameworks/new_framework/agent.py
from ..base import BaseAgent

class NewFrameworkAgent(BaseAgent):
    def run(self, instruction: str) -> AgentResult:
        pass
    
    def stream(self, instruction: str, thread_id=None):
        pass
    
    def resume(self, state, human_response: str):
        pass
```
