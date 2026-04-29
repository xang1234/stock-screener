#!/bin/bash
#
# Start Celery workers for stock scanner background tasks.
#
# Coordinated queue topology:
#   - One global datafetch worker subscribed to all data_fetch_* queues with
#     concurrency 1 so external-provider fetches are serialized across markets.
#   - One marketjobs-<market> worker per enabled market for breadth, group
#     rankings, and feature snapshots.
#   - One userscans-<market> worker per enabled market plus a shared safety-net
#     worker for any scan task dispatched without an explicit market.
#
# Usage: ./start_celery.sh
# Override enabled markets: ENABLED_MARKETS="US,HK" ./start_celery.sh
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

# Enabled markets (comma-separated). Override via env to skip markets locally.
ENABLED_MARKETS="${ENABLED_MARKETS:-US,HK,IN,JP,KR,TW,CN}"

echo "  Pool: $POOL"
echo "  Enabled markets: $ENABLED_MARKETS"

# General compute queue (unchanged from pre-9.1).
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q celery \
    -n general@%h &

# Global data-fetch worker: handles all external fetch queues under a single
# concurrency-1 worker so yfinance-bound jobs never overlap across markets.
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    --concurrency=1 \
    -Q data_fetch_shared,data_fetch_us,data_fetch_hk,data_fetch_in,data_fetch_jp,data_fetch_kr,data_fetch_tw,data_fetch_cn \
    -n datafetch-global@%h &

# Shared user-scans worker — same safety-net pattern for user-initiated scans.
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q user_scans_shared \
    -n userscans-shared@%h &

# One worker per enabled market for market compute/write jobs and market scans.
IFS=',' read -ra MARKET_ARRAY <<< "$ENABLED_MARKETS"
for RAW_MARKET in "${MARKET_ARRAY[@]}"; do
    MARKET_UPPER="$(echo "$RAW_MARKET" | tr '[:lower:]' '[:upper:]' | xargs)"
    MARKET_LOWER="$(echo "$MARKET_UPPER" | tr '[:upper:]' '[:lower:]')"

    case "$MARKET_UPPER" in
        US|HK|IN|JP|KR|TW|CN) ;;
        *)
            echo "  Skipping unknown market: $MARKET_UPPER"
            continue
            ;;
    esac

    echo "  Starting marketjobs-$MARKET_LOWER and userscans-$MARKET_LOWER workers"

    ./venv/bin/celery -A app.celery_app worker \
        --loglevel=info \
        --pool="$POOL" \
        -Q "market_jobs_${MARKET_LOWER}" \
        -n "marketjobs-${MARKET_LOWER}@%h" &

    ./venv/bin/celery -A app.celery_app worker \
        --loglevel=info \
        --pool="$POOL" \
        -Q "user_scans_${MARKET_LOWER}" \
        -n "userscans-${MARKET_LOWER}@%h" &
done

echo "Workers started. Use 'pkill -f celery' to stop."
wait
