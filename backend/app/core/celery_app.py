import os
from celery import Celery
from dotenv import load_dotenv
from pathlib import Path
import sys

# 🔴 [CRITICAL FIX] Task 모듈 로딩을 위한 Python Path 설정
# backend 폴더 경로를 Python Path에 추가 (Worker가 contracts 모듈을 찾도록)
backend_dir = Path(__file__).parent.parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# .env 파일 경로 명시 및 로드
# celery_app.py (backend/app/core/) -> 3단계 위로 이동하여 프로젝트 루트의 .env를 찾음
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / ".env") 

# Redis URL 설정 (Railway REDIS_URL 우선, 그 다음 CELERY_BROKER_URL)
CELERY_BROKER_URL = os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Celery 인스턴스 생성
celery_app = Celery(
    "worker", # Celery App 이름
    broker=CELERY_BROKER_URL,
    include=['app.tasks.analysis_tasks']
)
# Celery 설정
celery_app.conf.update(
    # 작업 결과를 저장할 백엔드
    result_backend=CELERY_BROKER_URL, 
    # 작업 직렬화 방식
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Seoul',
    enable_utc=False,
    # Celery Timeouts 설정
    task_soft_time_limit=300,  # 5분 soft limit
    task_time_limit=360,       # 6분 hard limit
)