#!/bin/bash
set -e

echo ">>> Running database migrations..."
cd /app/backend
alembic upgrade head

echo ">>> Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
