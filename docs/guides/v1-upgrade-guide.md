# DocScanner AI v1 Upgrade Guide

이 문서는 기존 개발 환경에서 v1으로 업그레이드하는 방법을 설명합니다.
v1에서는 AI 분석 파이프라인, LangGraph 기반 채팅, 프론트엔드 UI가 전면 개편되었습니다.

---

## 목차

1. [변경 사항 요약](#변경-사항-요약)
2. [사전 준비](#사전-준비)
3. [업그레이드 절차](#업그레이드-절차)
4. [환경 변수 설정](#환경-변수-설정)
5. [실행 방법](#실행-방법)
6. [문제 해결](#문제-해결)

---

## 변경 사항 요약

### Backend (Python)
- AI 분석 파이프라인 전면 개편 (`backend/app/ai/`)
  - LLM 기반 조항 분석 (clause_analyzer.py)
  - PII 마스킹, HyDE, CRAG, RAPTOR 등 고급 RAG 기법
  - Constitutional AI, LLM-as-a-Judge 평가 시스템
- LangGraph 기반 채팅 에이전트 (`langgraph_agent.py`)
  - SSE 스트리밍 응답
  - Vector DB, Graph DB, Web Search 도구 통합
  - Tavily API 기반 웹 검색
- 새로운 API 엔드포인트
  - `/api/v1/agent/*` - LangGraph 채팅
  - `/api/v1/analysis/*` - 분석 결과 조회
  - `/api/v1/search/*` - 검색 API

### Frontend (Next.js)
- 분석 페이지 UI 전면 개편
  - 2컬럼 레이아웃 (계약서 뷰어 + 분석 결과)
  - 위험 조항 하이라이팅 (클릭 시 계약서에서 해당 부분 표시)
  - AI 채팅 패널 (SSE 스트리밍, 마크다운 렌더링)
- 사용하지 않는 컴포넌트 정리
- 새로운 패키지 추가 (react-markdown, remark-gfm 등)

### Database
- Contract 모델에 `analysis_result` (JSONB) 필드 추가
- Alembic 마이그레이션 필요

---

## 사전 준비

### 필수 요구사항
- Python 3.10+
- Node.js 18+
- Docker Desktop (실행 중이어야 함)
- Git

### API 키 준비
다음 API 키가 필요합니다:
- **OpenAI API Key** - [platform.openai.com](https://platform.openai.com)
- **Gemini API Key** - [aistudio.google.com](https://aistudio.google.com)
- **Tavily API Key** (선택) - [tavily.com](https://tavily.com) - 에이전트 웹 검색용

---

## 업그레이드 절차

### Step 1: 코드 업데이트

```bash
cd docscanner-ai
git fetch origin
git checkout feature/advanced-ai-pipeline
git pull origin feature/advanced-ai-pipeline
```

### Step 2: Docker 서비스 확인

Docker Desktop이 실행 중인지 확인하고, 인프라 서비스를 시작합니다.

```bash
docker-compose up -d
```

실행 중인 서비스 확인:
```bash
docker ps
```

다음 서비스가 실행되어야 합니다:
- Pgadmin (5050)
- PostgreSQL (5435)
- Redis (6379)
- Elasticsearch (9200)
- Neo4j (7474, 7687)

### Step 3: Python 패키지 설치

#### Mac
```bash
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Windows (CMD)
```cmd
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**주요 추가 패키지:**
- `langgraph`, `langchain-core`, `langchain-google-genai` - LangGraph 에이전트
- `tavily-python` - 웹 검색
- `sse-starlette` - SSE 스트리밍
- `olefile`, `python-docx`, `chardet` - 문서 파싱

### Step 4: Frontend 패키지 설치

```bash
cd frontend
npm install
```

**주요 추가 패키지:**
- `react-markdown`, `remark-gfm` - 마크다운 렌더링
- `mammoth` - DOCX 파싱
- `pdfjs-dist`, `react-pdf` - PDF 뷰어

### Step 5: 환경 변수 설정

프로젝트 루트의 `.env` 파일을 수정합니다. 아래 [환경 변수 설정](#환경-변수-설정) 섹션 참조.

### Step 6: 데이터베이스 마이그레이션

```bash
cd backend
alembic upgrade head
```

**중요:** 기존 데이터베이스가 있는 경우, 마이그레이션이 실패할 수 있습니다.
이 경우 아래 해결 방법을 참조하세요.

---

## 환경 변수 설정

프로젝트 루트의 `.env` 파일에 다음 변수들을 설정합니다:

```bash
# ========================================
# Database (Docker)
# ========================================
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=docscanner
POSTGRES_HOST=localhost
POSTGRES_PORT=5435

# ========================================
# Redis (Docker)
# ========================================
REDIS_HOST=localhost
REDIS_PORT=6379

# ========================================
# Neo4j (Docker)
# ========================================
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# ========================================
# Elasticsearch (Docker)
# ========================================
ELASTICSEARCH_URL=http://localhost:9200

# ========================================
# JWT 인증
# ========================================
SECRET_KEY=your_super_secret_key_here

# ========================================
# LLM API Keys (필수 - 최소 하나)
# ========================================
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# ========================================
# Tavily (선택 - 웹 검색용)
# ========================================
TAVILY_API_KEY=tvly-...

# ========================================
# LLM 모델 설정 (선택)
# ========================================
LLM_RETRIEVAL_MODEL=gemini-2.5-flash-lite
LLM_REASONING_MODEL=gpt-5-mini
LLM_HYDE_MODEL=gpt-4o
```

---

## 실행 방법

### 전체 실행 순서

4개의 터미널이 필요합니다.

#### 터미널 1: Docker 서비스
```bash
cd docscanner-ai
docker-compose up -d
docker ps  # 서비스 확인
```

#### 터미널 2: Backend 서버

**Mac:**
```bash
cd docscanner-ai/backend
source ../venv/bin/activate
python main.py
```

**Windows:**
```powershell
cd docscanner-ai\backend
..\venv\Scripts\Activate.ps1
python main.py
```

서버 실행 확인: http://localhost:8000/docs

#### 터미널 3: Celery Worker

**Mac:**
```bash
cd docscanner-ai/backend
source ../venv/bin/activate
celery -A celery_worker worker --loglevel=info --pool=solo
```

**Windows:**
```powershell
cd docscanner-ai\backend
..\venv\Scripts\Activate.ps1
celery -A celery_worker worker --loglevel=info --pool=solo
```

#### 터미널 4: Frontend

```bash
cd docscanner-ai/frontend
npm run dev
```

접속: http://localhost:3000

---

## 문제 해결

### 1. pip install 실패 (Windows)

일부 패키지가 Windows에서 빌드 실패할 수 있습니다.

```powershell
# Visual C++ Build Tools 설치 필요
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 또는 pre-built wheel 사용
pip install --only-binary :all: -r requirements.txt
```

### 2. Alembic 마이그레이션 실패

```bash
# 현재 상태 확인
alembic current

# 기존 마이그레이션 히스토리와 충돌 시
alembic stamp head  # 현재 상태를 최신으로 표시
alembic upgrade head
```

데이터베이스를 완전히 초기화하려면:
```bash
docker-compose down -v  # 볼륨 삭제
docker-compose up -d
cd backend
alembic upgrade head
```

### 3. Celery Worker가 Task를 받지 않음

1. Redis가 실행 중인지 확인:
```bash
docker ps | grep redis
```

2. Worker 로그에서 `[tasks]` 섹션 확인:
```
[tasks]
  . analyze_contract
  . analyze_contract_quick
```

3. Worker 재시작

### 4. LangGraph 채팅이 작동하지 않음

1. GEMINI_API_KEY가 설정되어 있는지 확인
2. 백엔드 로그에서 에러 확인
3. Agent 헬스체크:
```bash
curl http://localhost:8000/api/v1/agent/health
```

### 5. 웹 검색이 작동하지 않음

TAVILY_API_KEY가 설정되지 않으면 정적 fallback 정보가 표시됩니다.
실제 웹 검색을 원하면 [tavily.com](https://tavily.com)에서 API 키를 발급받으세요.

### 6. Frontend 빌드 에러

```bash
cd frontend
rm -rf node_modules package-lock.json  # Windows: rmdir /s /q node_modules
npm install
npm run dev
```

### 7. Windows에서 localhost 연결 실패

Windows에서 localhost가 IPv6로 해석될 수 있습니다.

`.env` 파일에서:
```bash
POSTGRES_HOST=127.0.0.1  # localhost 대신
```

---

## 서비스 URL 요약

| 서비스 | URL | 설명 |
|--------|-----|------|
| Frontend | http://localhost:3000 | 웹 애플리케이션 |
| FastAPI Docs | http://localhost:8000/docs | API 문서 |
| pgAdmin | http://localhost:5050 | PostgreSQL 관리 |
| Neo4j Browser | http://localhost:7474 | Graph DB 관리 |
| Kibana | http://localhost:5601 | Elasticsearch 시각화 |

---

## 새로운 API 엔드포인트

### LangGraph Agent Chat
- `POST /api/v1/agent/{contract_id}/stream` - SSE 스트리밍 채팅
- `POST /api/v1/agent/{contract_id}/stream/history` - 히스토리 포함 채팅
- `GET /api/v1/agent/health` - 에이전트 상태 확인

### Analysis
- `GET /api/v1/analysis/{contract_id}` - 분석 결과 조회

### Search
- `POST /api/v1/search/vector` - Vector DB 검색
- `POST /api/v1/search/graph` - Graph DB 검색

---

## 주의사항

1. **API 키 보안**: `.env` 파일을 절대 커밋하지 마세요.
2. **Docker 리소스**: Elasticsearch와 Neo4j는 메모리를 많이 사용합니다. Docker Desktop에서 충분한 메모리를 할당하세요 (권장: 8GB 이상).
3. **첫 분석 시 지연**: 첫 계약서 분석 시 모델 로딩으로 인해 시간이 걸릴 수 있습니다.

---

문제가 해결되지 않으면 팀 Slack 채널에 문의하세요.
