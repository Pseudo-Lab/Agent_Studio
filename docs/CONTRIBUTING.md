# 협업 가이드

Self-Evolving Agent Framework 프로젝트에 기여해주셔서 감사합니다!

## 개발 환경 설정

### 1. 저장소 클론
```bash
git clone <repository-url>
cd agnet_studio
```

### 2. UV 가상환경 설정
```bash
# UV가 설치되어 있지 않다면
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성 및 의존성 설치
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### 3. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 열어 API 키 설정
```

### 4. 개발 도구 설치
```bash
uv sync --extra dev
```

## 프로젝트 구조

```
src/
├── layers/                    # 6개 핵심 레이어
│   ├── model_selection/       # Layer 1
│   ├── prompt_processing/     # Layer 2
│   ├── code_generation/       # Layer 3
│   ├── testing/               # Layer 4
│   ├── evaluation/            # Layer 5
│   └── reporting/             # Layer 6
├── evolution/                 # 진화 메커니즘
├── knowledge_base/            # 지식 베이스
└── utils/                     # 유틸리티

tests/                         # 테스트
docs/                          # 문서
examples/                      # 예제
```

## 개발 워크플로우

### 브랜치 전략
- `main`: 안정 버전
- `develop`: 개발 브랜치
- `feature/*`: 기능 개발
- `fix/*`: 버그 수정

### 작업 흐름
1. 이슈 생성 또는 할당
2. 브랜치 생성: `git checkout -b feature/your-feature`
3. 코드 작성 및 커밋
4. 테스트 작성 및 실행
5. Pull Request 생성
6. 코드 리뷰
7. 머지

## 코드 스타일

### Python 코딩 컨벤션
```bash
# 코드 포맷팅
uv run black src/ tests/

# 린팅
uv run ruff check src/ tests/

# 타입 체크
uv run mypy src/
```

### 스타일 가이드
- PEP 8 준수
- 타입 힌트 사용 (Python 3.11+ 스타일)
- Docstring 작성 (Google 스타일)
- 최대 줄 길이: 100자

### 예시
```python
from typing import Optional

def process_task(
    task_id: str,
    complexity: str,
    timeout: Optional[int] = None
) -> dict:
    """
    Process a task with given parameters.

    Args:
        task_id: Unique identifier for the task
        complexity: Task complexity level (simple, medium, complex)
        timeout: Optional timeout in seconds

    Returns:
        Dictionary containing processing results

    Raises:
        ValueError: If task_id is invalid
        TimeoutError: If processing exceeds timeout
    """
    pass
```

## 테스트

### 테스트 작성
```bash
# 단위 테스트
uv run pytest tests/unit/

# 통합 테스트
uv run pytest tests/integration/

# 전체 테스트 + 커버리지
uv run pytest --cov=src tests/
```

### 테스트 요구사항
- 모든 새로운 기능에 대한 테스트 작성
- 최소 80% 코드 커버리지 유지
- 테스트 이름은 명확하고 설명적으로

## 커밋 메시지

### 형식
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 코드 포맷팅
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드, 설정 등

### 예시
```
feat(model_selection): add GPT-4 Turbo support

- Add GPT-4 Turbo to model pool
- Update cost calculations
- Add performance benchmarks

Closes #123
```

## Pull Request

### PR 제목
- 커밋 메시지 형식과 동일

### PR 설명 템플릿
```markdown
## 변경 사항
-

## 관련 이슈
Closes #

## 테스트
- [ ] 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 수동 테스트 완료

## 체크리스트
- [ ] 코드 스타일 준수
- [ ] 문서 업데이트
- [ ] 테스트 추가
- [ ] Breaking changes 문서화
```

## 레이어별 개발 가이드

### Layer 1: Model Selection
- `src/layers/model_selection/`
- 새로운 모델 추가 시 ModelConfig 정의
- 성능 메트릭 업데이트 로직 구현

### Layer 2: Prompt Preprocessing
- `src/layers/prompt_processing/`
- 새로운 프롬프트 전략 추가
- 전략별 효과 측정

### Layer 3: Code Generation
- `src/layers/code_generation/`
- LangGraph 워크플로우 생성
- MCTS 알고리즘 개선

### Layer 4-6
- 각 레이어별 README 참조

## 이슈 리포팅

### 버그 리포트
- 재현 가능한 최소 예제 제공
- 환경 정보 (OS, Python 버전, 패키지 버전)
- 예상 동작 vs 실제 동작

### 기능 제안
- 구체적인 사용 사례
- 예상되는 이점
- 가능한 구현 방법

## 문서화

### 문서 작성
- `docs/` 디렉토리에 마크다운 파일 추가
- API 문서는 Docstring으로 작성
- 예제 코드 포함

### 문서 빌드
```bash
# 추후 Sphinx 등으로 문서 빌드 예정
```

## 코드 리뷰

### 리뷰어 체크리스트
- [ ] 코드가 요구사항을 충족하는가?
- [ ] 테스트가 충분한가?
- [ ] 성능 이슈가 없는가?
- [ ] 보안 문제가 없는가?
- [ ] 문서가 업데이트되었는가?

### 리뷰이 체크리스트
- [ ] Self-review 완료
- [ ] 모든 테스트 통과
- [ ] CI/CD 통과
- [ ] 충돌 해결

## 질문 및 지원

- GitHub Issues: 버그 리포트, 기능 제안
- Discussions: 일반 질문, 아이디어 공유
- Pull Request: 코드 기여

## 라이센스

MIT License (예정)

기여하신 코드는 프로젝트 라이센스 하에 배포됩니다.
