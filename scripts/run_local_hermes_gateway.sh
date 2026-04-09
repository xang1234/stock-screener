#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME_DIR="${HERMES_HOME:-$ROOT_DIR/data/hermes}"

load_env_file() {
  local file_path="$1"
  [[ -f "$file_path" ]] || return 0
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local line="$raw_line"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    [[ "$line" == *=* ]] || continue

    local key="${line%%=*}"
    local value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    export "$key=$value"
  done < "$file_path"
}

load_env_file "$ROOT_DIR/.env"
load_env_file "$ROOT_DIR/backend/.env"

resolve_hermes_provider() {
  if [[ -n "${HERMES_INFERENCE_PROVIDER:-}" ]]; then
    printf '%s' "${HERMES_INFERENCE_PROVIDER}"
    return
  fi
  if [[ -n "${MINIMAX_API_KEY:-}" ]]; then
    printf '%s' "minimax"
    return
  fi
  if [[ -n "${GLM_API_KEY:-}" || -n "${ZAI_API_KEY:-}" || -n "${Z_AI_API_KEY:-}" ]]; then
    printf '%s' "zai"
    return
  fi
  if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    printf '%s' "openrouter"
    return
  fi
  if [[ -n "${OPENAI_API_KEY:-}" || -n "${OPENAI_BASE_URL:-}" ]]; then
    printf '%s' "custom"
    return
  fi
  printf '%s' "auto"
}

resolve_hermes_model() {
  local provider="$1"
  if [[ -n "${HERMES_DEFAULT_MODEL:-}" ]]; then
    printf '%s' "${HERMES_DEFAULT_MODEL}"
    return
  fi
  case "$provider" in
    minimax)
      printf '%s' "MiniMax-M2.7"
      ;;
    zai)
      printf '%s' "glm-4.5-flash"
      ;;
    openrouter)
      printf '%s' "anthropic/claude-sonnet-4"
      ;;
    custom)
      printf '%s' "gpt-4o-mini"
      ;;
    *)
      printf '%s' "MiniMax-M2.7"
      ;;
  esac
}

if [[ -z "${SERVER_AUTH_PASSWORD:-}" ]]; then
  echo "SERVER_AUTH_PASSWORD must be set in .env or backend/.env before starting Hermes." >&2
  exit 1
fi

mkdir -p \
  "$HERMES_HOME_DIR/cron" \
  "$HERMES_HOME_DIR/hooks" \
  "$HERMES_HOME_DIR/logs" \
  "$HERMES_HOME_DIR/memories" \
  "$HERMES_HOME_DIR/sessions" \
  "$HERMES_HOME_DIR/skills"

export HERMES_HOME="$HERMES_HOME_DIR"
export API_SERVER_ENABLED="${API_SERVER_ENABLED:-true}"
export API_SERVER_HOST="${API_SERVER_HOST:-127.0.0.1}"
export API_SERVER_PORT="${API_SERVER_PORT:-8642}"
if [[ -n "${HERMES_API_KEY:-}" ]]; then
  export API_SERVER_KEY="${API_SERVER_KEY:-$HERMES_API_KEY}"
fi

HERMES_PROVIDER_VALUE="$(resolve_hermes_provider)"
HERMES_DEFAULT_MODEL_VALUE="$(resolve_hermes_model "$HERMES_PROVIDER_VALUE")"

cat > "$HERMES_HOME_DIR/config.yaml" <<EOF
model:
  provider: ${HERMES_PROVIDER_VALUE}
  default: ${HERMES_DEFAULT_MODEL_VALUE}

mcp_servers:
  stockscreen_market:
    url: "http://127.0.0.1:8000/mcp/"
    headers:
      x-server-auth: "${SERVER_AUTH_PASSWORD}"
    tools:
      include:
        - market_overview
        - compare_feature_runs
        - find_candidates
        - explain_symbol
        - watchlist_snapshot
        - theme_state
        - task_status
        - group_rankings
        - stock_lookup
        - stock_snapshot
        - breadth_snapshot
        - daily_digest
      prompts: false
      resources: false

skills:
  external_dirs:
    - ${ROOT_DIR}/integrations/hermes/skills
EOF

echo "Starting Hermes with HERMES_HOME=$HERMES_HOME_DIR on http://${API_SERVER_HOST}:${API_SERVER_PORT}/v1 using ${HERMES_PROVIDER_VALUE}/${HERMES_DEFAULT_MODEL_VALUE}"
exec hermes gateway
