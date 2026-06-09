#!/bin/bash

# Local backend development environment.
# Clears Docker-specific environment variables that may be injected
# by VS Code or the root .env file.

unset DATABASE_URL
unset REDIS_HOST
unset REDIS_PORT
unset CELERY_BROKER_URL
unset CELERY_RESULT_BACKEND

source venv/bin/activate

echo "Backend local development environment activated."

echo "DATABASE_URL=$(python -c "from app.config.settings import settings; print(settings.database_url)")"

echo "CELERY_BROKER_URL=$(python -c "from app.config.settings import settings; print(settings.celery_broker_url)")"

echo "Python: $(which python)"