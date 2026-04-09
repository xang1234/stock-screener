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

yaml_escape() {
  local escaped="${1//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  escaped="${escaped//$'\n'/\\n}"
  printf '%s' "$escaped"
}

validate_provider_credentials() {
  local provider="$1"
  case "$provider" in
    minimax)
      [[ -n "${MINIMAX_API_KEY:-}" ]] || {
        echo "MINIMAX_API_KEY is required when HERMES_INFERENCE_PROVIDER=minimax." >&2
        exit 1
      }
      ;;
    zai)
      [[ -n "${GLM_API_KEY:-}" || -n "${ZAI_API_KEY:-}" || -n "${Z_AI_API_KEY:-}" ]] || {
        echo "GLM_API_KEY, ZAI_API_KEY, or Z_AI_API_KEY is required when HERMES_INFERENCE_PROVIDER=zai." >&2
        exit 1
      }
      ;;
    openrouter)
      [[ -n "${OPENROUTER_API_KEY:-}" ]] || {
        echo "OPENROUTER_API_KEY is required when HERMES_INFERENCE_PROVIDER=openrouter." >&2
        exit 1
      }
      ;;
    custom)
      [[ -n "${OPENAI_API_KEY:-}" && -n "${OPENAI_BASE_URL:-}" ]] || {
        echo "OPENAI_API_KEY and OPENAI_BASE_URL are required when HERMES_INFERENCE_PROVIDER=custom." >&2
        exit 1
      }
      ;;
    auto)
      ;;
    *)
      echo "Invalid HERMES_INFERENCE_PROVIDER: ${provider}. Allowed values: minimax|zai|openrouter|custom|auto." >&2
      exit 1
      ;;
  esac
}

has_provider_credentials() {
  local provider_key
  for provider_key in OPENROUTER_API_KEY OPENAI_API_KEY MINIMAX_API_KEY GLM_API_KEY ZAI_API_KEY Z_AI_API_KEY; do
    if [[ "$provider_key" == "OPENAI_API_KEY" ]]; then
      if [[ -n "${OPENAI_API_KEY:-}" && -n "${OPENAI_BASE_URL:-}" ]]; then
        return 0
      fi
      continue
    fi
    if [[ -n "${!provider_key:-}" ]]; then
      return 0
    fi
  done
  return 1
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

if [[ -n "${OPENAI_API_KEY:-}" && -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "OPENAI_BASE_URL must be set when OPENAI_API_KEY is provided." >&2
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
if ! has_provider_credentials; then
  echo "Set at least one Hermes inference key before starting Hermes locally." >&2
  exit 1
fi
validate_provider_credentials "$HERMES_PROVIDER_VALUE"
ESCAPED_SERVER_AUTH_PASSWORD="$(yaml_escape "$SERVER_AUTH_PASSWORD")"

cat > "$HERMES_HOME_DIR/config.yaml" <<EOF
model:
  provider: ${HERMES_PROVIDER_VALUE}
  default: ${HERMES_DEFAULT_MODEL_VALUE}

mcp_servers:
  stockscreen_market:
    url: "http://127.0.0.1:8000/mcp/"
    headers:
      x-server-auth: "${ESCAPED_SERVER_AUTH_PASSWORD}"
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
chmod 600 "$HERMES_HOME_DIR/config.yaml"

echo "Starting Hermes with HERMES_HOME=$HERMES_HOME_DIR on http://${API_SERVER_HOST}:${API_SERVER_PORT}/v1 using ${HERMES_PROVIDER_VALUE}/${HERMES_DEFAULT_MODEL_VALUE}"
exec hermes gateway
