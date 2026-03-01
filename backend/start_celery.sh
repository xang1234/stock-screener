#!/bin/bash
#
# Start Celery workers for stock scanner background tasks
#
# Usage: ./start_celery.sh
#

cd "$(dirname "$0")"

# macOS fork() safety: required for Python deps that initialize Objective-C after fork.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

echo "Starting Celery workers..."

# On macOS, use solo pool to avoid fork() crashes with Objective-C runtime safety checks.
# Solo pool runs tasks sequentially in the main process (no forking)
# On Linux, you can change this to 'prefork' for parallel execution
POOL="${CELERY_POOL:-solo}"

echo "  Pool: $POOL"

# Worker 1: General tasks queue
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q celery \
    -n general@%h &

# Worker 2: Data-fetch queue (serialized execution for API rate limiting)
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q data_fetch \
    -n datafetch@%h &

# Worker 3: User scan queue (isolated from maintenance tasks)
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q user_scans \
    -n userscans@%h &

echo "Workers started. Use 'pkill -f celery' to stop."
wait
