#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

if [ "$1" = "celery" ]; then
    echo "Starting Celery worker..."
    exec "$@"
fi

echo "Starting ML Serving API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000