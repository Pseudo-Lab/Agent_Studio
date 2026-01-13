# 📋 Release Notes

## v1.0.0 (2026-01-13)

### 🚀 Initial Release - Kiosk Agent

**Vision-Language-Action (VLA) Agent for Automated Kiosk Interaction**

가짜연구소 11기 Agent Studio 팀이 개발한 키오스크 자동화 AI 에이전트입니다.

---

### ✨ Features

#### Core Agent
- **VLA Workflow**: Vision → Language → Action 패러다임 기반 GUI 자동화
- **LangGraph Integration**: 상태 기반 멀티스텝 워크플로우 오케스트레이션
- **Human-in-the-Loop (HITL)**: 주관적 선택이 필요할 때 사용자에게 질문
- **Multi-LLM Support**: Gemini, OpenAI GPT-4V, Local vLLM 지원

#### Voice Interface
- **TTS (Text-to-Speech)**: CosyVoice3 기반 자연스러운 음성 합성
- **STT (Speech-to-Text)**: Google Cloud Speech-to-Text API 연동
- **Character System**: YAML 기반 캐릭터 설정 (음성, 이미지, 메시지 커스터마이징)

#### Device Control
- **ADB Integration**: Android 디바이스 제어 (tap, swipe, text input, screenshot)
- **Action Types**: CLICK, LONG_CLICK, SWIPE, INPUT, BACK, HOME, INTERRUPT, FINISH

#### Frontend Dashboard
- **Real-time Monitoring**: SSE 기반 에이전트 상태 실시간 스트리밍
- **Interactive UI**: Next.js + Tailwind CSS 기반 모던 대시보드
- **HITL Interface**: 사용자 선택 카드 및 음성 녹음 UI

---

### 📁 Project Structure

```
├── backend/                 # Python FastAPI 서버
│   ├── kiosk_agent/         # 코어 에이전트 라이브러리
│   │   ├── core/            # ADB, Screenshot, Translator
│   │   ├── llm/             # Gemini, OpenAI, Local
│   │   ├── frameworks/      # LangGraph (+ 확장 예정)
│   │   ├── prompts/         # 시스템 프롬프트
│   │   └── voice/           # TTS, STT
│   └── api/                 # FastAPI 엔드포인트
├── web/                     # Next.js 프론트엔드
└── run.sh                   # 통합 실행 스크립트
```

---

### 🛠️ Technical Stack

| Category | Technology |
|----------|------------|
| Backend | Python 3.10+, FastAPI, LangGraph |
| Frontend | Next.js 14, React 18, Tailwind CSS |
| LLM | Google Gemini, OpenAI GPT-4V |
| TTS | CosyVoice3 (MLX) |
| STT | Google Cloud Speech-to-Text |
| Device | Android ADB |

---

### 👥 Contributors

| Name | Role | Company |
|------|------|---------|
| 김재현 | 빌더 | KTDS |
| 김승혁 | 러너 | namu |
| 이규민 | 러너 | KT |
| 전민정 | 러너 | AICESS |

---

### 📄 License

Apache License 2.0

---

## 🗓️ Upcoming Releases

### v1.1.0 (2026-01 예정)

#### 🔜 Coming Soon

- **Microsoft Agent Framework 지원**
  - Azure AI Agent Service 연동
  - Semantic Kernel 기반 에이전트 구현

- **Google ADK (Agent Development Kit) 지원**
  - Google AI Studio 연동
  - Gemini 네이티브 에이전트 프레임워크

- **CrewAI 지원**
  - 멀티 에이전트 협업 워크플로우
  - Role-based 에이전트 시스템

#### 🎯 Planned Features

- Planning Mode: 복잡한 태스크 분해 및 계획 수립
- Context Management: 장기 메모리 및 컨텍스트 관리
- On-device Model: 경량화 모델 로컬 실행
- Service Architecture: 마이크로서비스 아키텍처 전환

---

## 📚 Links

- **GitHub**: https://github.com/Pseudo-Lab/Agent_Studio
- **PseudoLab**: https://pseudo-lab.com/
- **Discord**: https://discord.gg/EPurkHVtp2
