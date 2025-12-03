# Development Environment Setup Guide

DocScanner AI 개발 환경 설정 및 실행 가이드입니다.

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [프로젝트 클론 및 설정](#프로젝트-클론-및-설정)
3. [Docker 서비스 실행](#docker-서비스-실행)
4. [Backend 서버 실행](#backend-서버-실행)
5. [Celery Worker 실행](#celery-worker-실행)
6. [Frontend 실행](#frontend-실행)
7. [API 테스트](#api-테스트)
8. [문제 해결](#문제-해결)

---

## 사전 요구사항

### 공통
- Git
- Docker Desktop
- Python 3.10+
- Node.js 18+

### Mac
```bash
# Homebrew 설치 (없는 경우)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 필수 도구 설치
brew install python@3.10 node docker
```

### Windows
- [Python 3.10+](https://www.python.org/downloads/) 설치
- [Node.js 18+](https://nodejs.org/) 설치
- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) 설치
- WSL2 활성화 권장

---

## 프로젝트 클론 및 설정

### 1. 프로젝트 클론
```bash
git clone https://github.com/your-org/docscanner-ai.git
cd docscanner-ai
```

### 2. Python 가상환경 설정

#### Mac / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Windows (CMD)
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 3. 환경 변수 설정
```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env

# .env 파일 편집하여 API 키 설정
# - OPENAI_API_KEY
# - GEMINI_API_KEY
# - 기타 필요한 설정
```

---

## Docker 서비스 실행

Docker로 인프라 서비스(PostgreSQL, Redis, Elasticsearch, Neo4j)를 실행합니다.

### 서비스 시작

#### Mac / Linux / Windows
```bash
docker-compose up -d
```

### 실행 중인 서비스 확인
```bash
docker ps
```

예상 출력:
```
CONTAINER ID   IMAGE                                                  STATUS         PORTS
xxxx           postgres:15                                            Up             0.0.0.0:5435->5432/tcp
xxxx           redis:latest                                           Up             0.0.0.0:6379->6379/tcp
xxxx           docker.elastic.co/elasticsearch/elasticsearch:8.11.1   Up             0.0.0.0:9200->9200/tcp
xxxx           neo4j:latest                                           Up             0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
xxxx           dpage/pgadmin4                                         Up             0.0.0.0:5050->80/tcp
xxxx           docker.elastic.co/kibana/kibana:8.11.1                 Up             0.0.0.0:5601->5601/tcp
```

### 서비스 중지
```bash
docker-compose down
```

### 데이터 포함 완전 삭제
```bash
docker-compose down -v
```

---

## Backend 서버 실행

### 1. 데이터베이스 마이그레이션

처음 실행 시 또는 모델 변경 후 마이그레이션이 필요합니다.

#### Mac / Linux
```bash
cd backend
source ../venv/bin/activate
alembic upgrade head
```

#### Windows
```powershell
cd backend
..\venv\Scripts\Activate.ps1
alembic upgrade head
```

### 2. FastAPI 서버 실행

#### Mac / Linux
```bash
cd backend
source ../venv/bin/activate
python main.py
```

#### Windows
```powershell
cd backend
..\venv\Scripts\Activate.ps1
python main.py
```

서버가 실행되면:
- API 문서: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Celery Worker 실행

Celery Worker는 비동기 작업(PDF 분석 등)을 처리합니다.
**새 터미널**에서 실행해야 합니다.

### Mac

```bash
cd backend
source ../venv/bin/activate
celery -A celery_worker worker --loglevel=info --pool=solo
```

> **Note**: Apple Silicon(M1/M2/M3)에서는 `--pool=solo` 옵션 필수 (MPS/PyTorch 호환 문제)

### Windows

```powershell
cd backend
..\venv\Scripts\Activate.ps1
celery -A celery_worker worker --loglevel=info --pool=solo
```

> **Note**: Windows에서는 `--pool=solo` 또는 `--pool=threads` 사용 권장

### Celery 실행 확인

정상 실행 시 출력:
```
 -------------- celery@your-machine v5.5.3 (immunity)
--- ***** -----
-- ******* ----
- *** --- * ---
- ** ---------- [config]
- ** ---------- .> app:         worker:0x...
- ** ---------- .> transport:   redis://localhost:6379/0
- ** ---------- .> results:     redis://localhost:6379/0
- *** --- * --- .> concurrency: 12 (solo)
-- ******* ---- .> task events: OFF
--- ***** -----
 -------------- [queues]
                .> celery           exchange=celery(direct) key=celery

[tasks]
  . analyze_contract
  . analyze_contract_quick

[... INFO/MainProcess] celery@your-machine ready.
```

---

## Frontend 실행

### 1. 의존성 설치
```bash
cd frontend
npm install
```

### 2. 개발 서버 실행
```bash
npm run dev
```

접속: http://localhost:3000

---

## API 테스트

### Swagger UI 사용 (권장)

1. http://localhost:8000/docs 접속
2. 회원가입: `POST /api/v1/auth/signup`
3. 로그인: `POST /api/v1/auth/login`
4. 토큰 복사 (access_token)
5. 우측 상단 `Authorize` 버튼 클릭
6. 토큰만 입력 (Bearer 제외)
7. API 테스트

### cURL 사용

```bash
# 회원가입
curl -X POST http://localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test1234","username":"tester"}'

# 로그인
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test1234"}'

# 계약서 업로드 (TOKEN을 실제 토큰으로 교체)
curl -X POST http://localhost:8000/api/v1/contracts/ \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@/path/to/contract.pdf"
```

---

## 문제 해결

### Docker 관련

#### 포트 충돌
```bash
# 사용 중인 포트 확인
lsof -i :5435  # Mac/Linux
netstat -ano | findstr :5435  # Windows

# 해당 프로세스 종료 후 재실행
```

#### 볼륨 초기화
```bash
docker-compose down -v
docker-compose up -d
```

### Celery 관련

#### Worker가 Task를 받지 않음
- Redis가 실행 중인지 확인: `docker ps | grep redis`
- 새 터미널에서 Worker 재시작

#### Apple Silicon SIGABRT 에러
```bash
# --pool=solo 옵션으로 실행
celery -A celery_worker worker --loglevel=info --pool=solo
```

#### Windows에서 Worker 충돌
```powershell
# eventlet 또는 gevent 설치 후 사용
pip install eventlet
celery -A celery_worker worker --loglevel=info --pool=eventlet
```

### Database 관련

#### "relation does not exist" 에러
```bash
# 마이그레이션 실행
cd backend
alembic upgrade head
```

#### 마이그레이션 충돌
```bash
# 현재 상태 확인
alembic current

# 마이그레이션 히스토리 확인
alembic history

# 특정 버전으로 다운그레이드
alembic downgrade -1
```

### OpenAI API 관련

#### "max_tokens is not supported" 에러
- 새 모델(gpt-5-mini 등)은 `max_completion_tokens` 파라미터 사용
- 코드에서 `max_tokens` -> `max_completion_tokens` 변경 필요

---

## 전체 실행 순서 요약

### 터미널 1: Docker 서비스
```bash
cd docscanner-ai
docker-compose up -d
```

### 터미널 2: Backend 서버
```bash
cd docscanner-ai/backend
source ../venv/bin/activate  # Windows: ..\venv\Scripts\Activate.ps1
alembic upgrade head  # 최초 1회
python main.py
```

### 터미널 3: Celery Worker
```bash
cd docscanner-ai/backend
source ../venv/bin/activate  # Windows: ..\venv\Scripts\Activate.ps1
celery -A celery_worker worker --loglevel=info --pool=solo
```

### 터미널 4: Frontend (선택)
```bash
cd docscanner-ai/frontend
npm install  # 최초 1회
npm run dev
```

---

## 서비스 접속 URL

| 서비스 | URL | 설명 |
|--------|-----|------|
| FastAPI Docs | http://localhost:8000/docs | API 문서 (Swagger) |
| Frontend | http://localhost:3000 | 웹 애플리케이션 |
| pgAdmin | http://localhost:5050 | PostgreSQL 관리 |
| Kibana | http://localhost:5601 | Elasticsearch 시각화 |
| Neo4j Browser | http://localhost:7474 | 그래프 DB 관리 |
