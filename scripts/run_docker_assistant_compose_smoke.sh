#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
ENV_FILE="$TMP_DIR/.env.docker"
COOKIE_JAR="$TMP_DIR/cookies.txt"
FRONTEND_PORT="${FRONTEND_PORT:-18080}"
SERVER_AUTH_PASSWORD="${SERVER_AUTH_PASSWORD:-smoke-password}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-ssc_assistant_smoke}"
ASSISTANT_HEALTH_MAX_ATTEMPTS="${ASSISTANT_HEALTH_MAX_ATTEMPTS:-150}"
export COMPOSE_PROJECT_NAME

cleanup() {
  echo "--- Backend container logs (last 100 lines) ---"
  docker logs --tail=100 "${COMPOSE_PROJECT_NAME}-backend-1" 2>&1 || true
  echo "--- End backend logs ---"
  echo "--- Hermes container logs (last 100 lines) ---"
  docker logs --tail=100 "${COMPOSE_PROJECT_NAME}-hermes-1" 2>&1 || true
  echo "--- End hermes logs ---"
  docker compose \
    --env-file "$ENV_FILE" \
    --profile assistant \
    down -v --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cat > "$ENV_FILE" <<EOF
SERVER_AUTH_PASSWORD=${SERVER_AUTH_PASSWORD}
SERVER_AUTH_SESSION_SECRET=smoke-session-secret
SERVER_AUTH_SECURE_COOKIE=false
SERVER_EXPOSE_API_DOCS=false
POSTGRES_DB=stockscanner
POSTGRES_USER=stockscanner
POSTGRES_PASSWORD=stockscanner
HERMES_API_BASE=http://hermes:8642/v1
HERMES_API_KEY=smoke-hermes-key
HERMES_MODEL=hermes-agent
HERMES_PLATFORM=linux/amd64
FRONTEND_PORT=${FRONTEND_PORT}
MINIMAX_API_KEY=smoke-minimax-key
SERPER_API_KEY=smoke-serper-key
EOF

echo "Starting assistant profile compose stack on frontend port ${FRONTEND_PORT}..."
bash "$ROOT_DIR/scripts/start_docker_assistant_stack.sh" "$ENV_FILE"

wait_for_url() {
  local url="$1"
  local name="$2"
  local headers=("${@:3}")
  local attempt
  for attempt in $(seq 1 60); do
    if curl -fsS "${headers[@]}" "$url" >/dev/null 2>&1; then
      echo "${name} is ready"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${name}: ${url}" >&2
  return 1
}

assert_json_field_equals() {
  local json_payload="$1"
  local field_name="$2"
  local expected="$3"
  python3 - "$field_name" "$expected" "$json_payload" <<'PY'
import json
import sys

field_name = sys.argv[1]
expected = sys.argv[2]
payload = json.loads(sys.argv[3])
actual = payload
for part in field_name.split("."):
    actual = actual[part]
if str(actual).lower() != expected.lower():
    raise SystemExit(f"Expected {field_name}={expected!r}, got {actual!r}")
PY
}

wait_for_assistant_available() {
  local url="http://127.0.0.1:${FRONTEND_PORT}/api/v1/assistant/health"
  local attempt
  local last_payload=""
  for attempt in $(seq 1 "${ASSISTANT_HEALTH_MAX_ATTEMPTS}"); do
    local payload
    payload="$(curl -fsS -H "x-server-auth: ${SERVER_AUTH_PASSWORD}" "$url" 2>/dev/null || true)"
    if [[ -n "$payload" ]]; then
      last_payload="$payload"
      if python3 - "$payload" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
raise SystemExit(0 if payload.get("available") is True and payload.get("status") == "healthy" else 1)
PY
      then
        echo "assistant health is ready"
        return 0
      fi
    fi
    sleep 2
  done
  echo "Timed out waiting for assistant health to become healthy" >&2
  if [[ -n "$last_payload" ]]; then
    echo "Last assistant health payload: $last_payload" >&2
  fi
  return 1
}

wait_for_url "http://127.0.0.1:${FRONTEND_PORT}/nginx-health" "frontend"
wait_for_assistant_available

docker compose --env-file "$ENV_FILE" --profile assistant exec -T hermes \
  sh -lc 'test -f /opt/data/config.yaml && test -f /opt/data/.env'

HEALTH_PAYLOAD="$(curl -fsS \
  -H "x-server-auth: ${SERVER_AUTH_PASSWORD}" \
  "http://127.0.0.1:${FRONTEND_PORT}/api/v1/assistant/health")"
assert_json_field_equals "$HEALTH_PAYLOAD" "available" "true"
assert_json_field_equals "$HEALTH_PAYLOAD" "status" "healthy"

LOGIN_PAYLOAD="$(curl -fsS \
  -c "$COOKIE_JAR" \
  -H "Content-Type: application/json" \
  -d "{\"password\":\"${SERVER_AUTH_PASSWORD}\"}" \
  "http://127.0.0.1:${FRONTEND_PORT}/api/v1/auth/login")"
assert_json_field_equals "$LOGIN_PAYLOAD" "authenticated" "true"

CONVERSATION_PAYLOAD="$(curl -fsS \
  -b "$COOKIE_JAR" \
  -H "Content-Type: application/json" \
  -d '{"title":"Compose smoke"}' \
  "http://127.0.0.1:${FRONTEND_PORT}/api/v1/assistant/conversations")"
python3 - "$CONVERSATION_PAYLOAD" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
conversation_id = payload.get("conversation_id", "")
if not conversation_id:
    raise SystemExit("conversation_id missing from assistant conversation payload")
PY

echo "Assistant compose smoke check passed."
