from app.core.celery_app import celery_app

if __name__ == '__main__':
    # Celery 워커 시작 명령어
    # app.worker는 app/core/celery_app.py에서 정의된 celery_app 인스턴스의 worker 이름
    # -l info는 로그 레벨을 정보로 설정
    print("💡 Starting Celery Worker. Run with 'celery -A celery_worker worker -l info'")
    # Note: 실제 실행은 이 스크립트를 직접 실행하는 것이 아니라 Celery 명령어를 사용합니다.
    # 예: celery -A celery_worker worker -l info