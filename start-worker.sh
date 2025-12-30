#!/bin/bash
set -e

cd /app/backend

echo "Starting Celery Worker..."
exec celery -A celery_worker worker -l info --concurrency=2
