# Kiosk Agent Backend

Vision-Language-Action (VLA) Agent for Kiosk Automation.

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .

# With optional TTS (Apple Silicon only)
pip install -e ".[tts]"

# With all extras
pip install -e ".[all]"
```

### 2. Set Environment Variables

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"  # For STT

# Optional
export ADB_PATH="/usr/local/bin/adb"
export DEVICE_ID="emulator-5554"
```

### 3. Run Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

## Project Structure

```
backend/
├── kiosk_agent/              # Core agent library
│   ├── config.py             # Configuration
│   ├── types.py              # Type definitions
│   ├── core/                 # Device control modules
│   │   ├── perception.py     # Screenshot capture
│   │   ├── control.py        # ADB commands
│   │   └── translator.py     # Action translation
│   ├── llm/                  # LLM clients
│   │   ├── base.py           # Base interface
│   │   ├── gemini.py         # Google Gemini
│   │   ├── openai.py         # OpenAI GPT-4V
│   │   └── local.py          # Local vLLM
│   ├── frameworks/           # Agent frameworks
│   │   ├── base.py           # Base agent interface
│   │   ├── langgraph/        # LangGraph implementation
│   │   ├── msagent/          # MS Agent Framework (placeholder)
│   │   ├── crewai/           # CrewAI (placeholder)
│   │   └── google_adk/       # Google ADK (placeholder)
│   ├── prompts/              # System prompts
│   └── voice/                # TTS/STT modules
├── api/                      # FastAPI server
│   ├── main.py               # Entry point
│   ├── schemas.py            # Request/Response models
│   ├── session.py            # Session management
│   └── routes/               # API endpoints
│       ├── agent.py          # Agent execution
│       ├── health.py         # Health check
│       └── voice.py          # TTS/STT
├── requirements.txt
└── pyproject.toml
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/agent/start` | Start agent execution |
| POST | `/agent/respond` | Respond to agent question |
| POST | `/agent/interrupt` | Force interrupt agent |
| POST | `/stt/transcribe` | Speech to text |
| GET | `/tts/audio/{filename}` | Get TTS audio file |

## Supported Frameworks

- **LangGraph** (default): Stateful multi-step workflow orchestration
- **Microsoft Agent Framework**: (Coming soon)
- **CrewAI**: (Coming soon)
- **Google ADK**: (Coming soon)

## Environment Variables

### API Keys
| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Gemini API key | Required for Gemini |
| `GEMINI_API_KEY` | Alternative to GOOGLE_API_KEY | - |
| `OPENAI_API_KEY` | OpenAI API key | Required for ChatGPT |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud credentials for STT | Optional |

### Model Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PROVIDER` | LLM provider (gemini/chatgpt/local_vllm) | `gemini` |
| `GEMINI_MODEL` | Gemini model name | `gemini-3-flash-preview` |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `VLLM_BASE_URL` | Local vLLM server URL | `http://localhost:8000` |
| `VLLM_MODEL_NAME` | Local vLLM model name | `AgentCPM-GUI` |
| `MODEL_TEMPERATURE` | LLM temperature | `0.1` |
| `MODEL_TOP_P` | LLM top_p | `0.3` |

### ADB Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `ADB_PATH` | Path to adb binary | `adb` |
| `DEVICE_ID` | Android device serial | Auto-detect |
| `SCREENSHOTS_DIR` | Screenshots directory | `./screenshots` |
| `KEEP_LAST_N` | Keep last N screenshots | `10` |
| `SWIPE_MS` | Swipe duration (ms) | `300` |

### Agent Runtime
| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_ITERATIONS` | Max agent steps | `20` |
| `PROGRESS_THRESHOLD` | Screen change threshold | `0.02` |
| `AGENT_RECURSION_LIMIT` | LangGraph recursion limit | `100` |

### TTS Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_TTS_ENABLED` | Enable TTS | `1` |
| `AGENT_TTS_KEEP_LAST_N` | Keep last N TTS files | `5` |
| `AGENT_TTS_THOUGHT` | Enable TTS for thoughts | `0` |
| `AGENT_TTS_THOUGHT_MAX_CHARS` | Max chars for thought TTS | `320` |

### API Server
| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Server host | `0.0.0.0` |
| `API_PORT` | Server port | `8000` |
| `API_MAX_WORKERS` | Thread pool workers | `4` |
| `API_MAX_SESSIONS` | Max concurrent sessions | `100` |
| `API_SESSION_TTL` | Session TTL (seconds) | `3600` |
| `DEFAULT_MODEL` | Default model ID | `gemini-flash` |
| `LOG_LEVEL` | Logging level | `INFO` |