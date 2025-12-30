# Railway Deployment Troubleshooting Guide

이 문서는 Railway 배포 과정에서 발생한 에러와 해결 방법을 정리합니다.

---

## 1. PyTorch 다운로드 타임아웃

### 증상
```
BrokenPipeError: [Errno 32] Broken pipe
```
빌드 중 PyTorch (~900MB) 다운로드 시 연결이 끊어지며 실패.

### 원인
PyTorch 기본 패키지가 너무 커서 Railway 빌드 환경에서 다운로드 타임아웃 발생.

### 해결
`requirements.txt`에 CPU-only PyTorch 사용:
```
# PyTorch CPU-only (must be before sentence-transformers)
--extra-index-url https://download.pytorch.org/whl/cpu
torch
sentence-transformers>=2.2.0
transformers>=4.30.0
```

CPU-only 버전은 약 200MB로 훨씬 가볍고 빌드가 안정적으로 완료됨.

---

## 2. Alembic 데이터베이스 연결 실패

### 증상
```
ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 5435)
```
Railway에서 마이그레이션 실행 시 localhost로 연결 시도.

### 원인
`alembic.ini`에 로컬 개발용 DATABASE_URL이 하드코딩되어 있음:
```ini
sqlalchemy.url = postgresql+asyncpg://doc_db_admin:DocScannerDBPass2025@127.0.0.1:5435/docscanner_db
```

### 해결
`backend/alembic/env.py`에서 환경변수를 우선 사용하도록 수정:
```python
import os

# DATABASE_URL 환경변수에서 동적으로 읽기 (Railway 등 PaaS 지원)
database_url = os.getenv("DATABASE_URL")
if database_url:
    # postgres:// -> postgresql+asyncpg://
    database_url = database_url.replace(
        "postgres://", "postgresql+asyncpg://"
    ).replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    config.set_main_option("sqlalchemy.url", database_url)
```

이 방식은 로컬과 배포 환경 모두 호환:
- 로컬: DATABASE_URL 환경변수 없음 -> alembic.ini 기본값 사용
- Railway: DATABASE_URL 환경변수 있음 -> Railway Postgres URL 사용

---

## 3. Railway 환경변수 참조 형식

### 증상
환경변수가 Railway 서비스 참조로 설정되어 있는데도 localhost로 연결 시도.

### 원인
Railway 변수 참조에 따옴표를 사용하면 문자열 그대로 저장됨:
```
# 잘못된 형식
DATABASE_URL="${{Postgres.DATABASE_URL}}"

# 올바른 형식
DATABASE_URL=${{Postgres.DATABASE_URL}}
```

### 해결
Railway Variables 탭에서 참조 변수 값을 **따옴표 없이** 입력:
```
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
CELERY_BROKER_URL=${{Redis.REDIS_URL}}
```

---

## Railway 서비스 환경변수 체크리스트

### Backend (main-backend)
```
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
CELERY_BROKER_URL=${{Redis.REDIS_URL}}
CORS_ORIGINS=https://your-vercel-domain.vercel.app
SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
# ... 기타 LLM 모델 설정
```

### Celery Worker
```
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
CELERY_BROKER_URL=${{Redis.REDIS_URL}}
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
# ... Backend와 동일한 API 키 설정
```

---

## Railway 배포 설정

### Backend
- **Builder**: Dockerfile
- **Start Command**: (Dockerfile CMD 사용)
- **Public Networking**: Enable (HTTP)

### Celery Worker
- **Builder**: Dockerfile
- **Start Command**: `/app/start-worker.sh`
- **Public Networking**: 불필요 (내부 서비스)
