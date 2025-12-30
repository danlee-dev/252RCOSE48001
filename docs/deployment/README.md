# DocScanner.ai Deployment Guide

이 문서는 DocScanner.ai 프로젝트의 배포 구성을 설명합니다.

---

## 아키텍처 개요

```
[Vercel]                    [Railway]
Frontend (Next.js)  <--->   Backend (FastAPI)
                            Celery Worker
                            PostgreSQL
                            Redis
```

---

## 서비스 구성

### Frontend (Vercel)
- **Framework**: Next.js 14
- **배포 브랜치**: main (Production), develop (Preview)
- **빌드 명령**: `npm run build`

### Backend (Railway)
- **Framework**: FastAPI
- **Docker**: Dockerfile 기반 빌드
- **시작 스크립트**: `/app/start.sh`
- **포트**: 8000

### Celery Worker (Railway)
- **역할**: 비동기 작업 처리 (계약서 분석)
- **시작 스크립트**: `/app/start-worker.sh`
- **브로커**: Redis

### Database (Railway)
- **PostgreSQL**: 메인 데이터 저장소
- **Redis**: Celery 브로커 및 캐시

---

## 환경변수 요약

### Vercel (Frontend)
| 변수명 | 설명 |
|--------|------|
| NEXT_PUBLIC_API_URL | Railway Backend URL |

### Railway (Backend/Celery)
| 변수명 | 설명 |
|--------|------|
| DATABASE_URL | PostgreSQL 연결 URL |
| REDIS_URL | Redis 연결 URL |
| CELERY_BROKER_URL | Celery 브로커 URL (보통 Redis) |
| CORS_ORIGINS | 허용할 프론트엔드 도메인 |
| SECRET_KEY | JWT 서명 키 |
| OPENAI_API_KEY | OpenAI API 키 |
| GEMINI_API_KEY | Google Gemini API 키 |

---

## 트러블슈팅 문서

- [Railway 배포 문제 해결](./RAILWAY_TROUBLESHOOTING.md)
- [Vercel 배포 문제 해결](./VERCEL_TROUBLESHOOTING.md)

---

## 배포 플로우

### 1. Feature 개발
```
feature/* 브랜치에서 개발 -> PR 생성 -> develop 머지
```

### 2. Vercel Preview 배포
develop 브랜치 머지 시 자동으로 Preview 환경에 배포

### 3. Railway 배포
develop 브랜치 머지 시 자동으로 Railway 서비스 재배포

### 4. Production 배포
```
develop -> main PR 생성 -> 머지 -> Production 배포
```

---

## 로컬 개발 환경

### Backend
```bash
cd backend
pip install -r ../requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Celery Worker
```bash
cd backend
celery -A celery_worker worker -l info
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 주의사항

1. **환경변수 동기화**: Backend와 Celery Worker는 동일한 API 키 환경변수 필요
2. **Railway 참조 변수**: 따옴표 없이 `${{Service.VAR}}` 형식 사용
3. **CORS 설정**: 프론트엔드 도메인 변경 시 Railway CORS_ORIGINS 업데이트 필요
4. **PyTorch**: CPU-only 버전 사용하여 빌드 시간 단축
