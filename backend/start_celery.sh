#!/bin/bash
#
# Start Celery workers for stock scanner background tasks.
#
# Per-market queue topology (bead StockScreenClaude-asia.9.1):
#   - One dedicated datafetch-<market> worker per enabled market. Each holds
#     its own Redis lock key so one market's refresh can't stall another's.
#   - One datafetch-shared worker for market-agnostic jobs and backward-compat
#     safety-net routing.
#   - Same split for user_scans queues.
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
ENABLED_MARKETS="${ENABLED_MARKETS:-US,HK,JP,TW}"

echo "  Pool: $POOL"
echo "  Enabled markets: $ENABLED_MARKETS"

# General compute queue (unchanged from pre-9.1).
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q celery \
    -n general@%h &

# Shared data-fetch worker: handles market-agnostic jobs (theme tasks,
# orphan cleanup) and acts as safety-net for any task enqueued without an
# explicit queue.
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q data_fetch_shared \
    -n datafetch-shared@%h &

# Shared user-scans worker — same safety-net pattern for user-initiated scans.
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q user_scans_shared \
    -n userscans-shared@%h &

# One worker per enabled market. Concurrency stays at 1 per market (solo pool)
# so rate limits hold intra-market; cross-market parallelism comes from having
# separate worker processes on separate lock keys.
IFS=',' read -ra MARKET_ARRAY <<< "$ENABLED_MARKETS"
for RAW_MARKET in "${MARKET_ARRAY[@]}"; do
    MARKET_UPPER="$(echo "$RAW_MARKET" | tr '[:lower:]' '[:upper:]' | xargs)"
    MARKET_LOWER="$(echo "$MARKET_UPPER" | tr '[:upper:]' '[:lower:]')"

    case "$MARKET_UPPER" in
        US|HK|JP|TW) ;;
        *)
            echo "  Skipping unknown market: $MARKET_UPPER"
            continue
            ;;
    esac

    echo "  Starting datafetch-$MARKET_LOWER and userscans-$MARKET_LOWER workers"

    ./venv/bin/celery -A app.celery_app worker \
        --loglevel=info \
        --pool="$POOL" \
        -Q "data_fetch_${MARKET_LOWER}" \
        -n "datafetch-${MARKET_LOWER}@%h" &

    ./venv/bin/celery -A app.celery_app worker \
        --loglevel=info \
        --pool="$POOL" \
        -Q "user_scans_${MARKET_LOWER}" \
        -n "userscans-${MARKET_LOWER}@%h" &
done

echo "Workers started. Use 'pkill -f celery' to stop."
wait
