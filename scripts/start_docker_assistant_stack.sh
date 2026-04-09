#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=false

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

usage() {
  cat <<'EOF'
Usage:
  bash scripts/start_docker_assistant_stack.sh [--dry-run] [ENV_FILE]

Defaults:
  ENV_FILE=.env.docker

What it does:
  - loads Docker env values from ENV_FILE
  - writes data/hermes/config.yaml for Docker MCP access
  - writes data/hermes/.env for Hermes provider/api-server settings
  - starts postgres, redis, backend, frontend, and hermes via Docker Compose
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

ENV_FILE="${1:-$ROOT_DIR/.env.docker}"
if [[ ! -f "$ENV_FILE" ]]; then
  cat >&2 <<EOF
Missing env file: $ENV_FILE

Create one first, for example:
  cp $ROOT_DIR/.env.docker.example $ROOT_DIR/.env.docker
EOF
  exit 1
fi

load_env_file "$ENV_FILE"

: "${SERVER_AUTH_PASSWORD:?Set SERVER_AUTH_PASSWORD in $ENV_FILE}"

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

HERMES_HAS_PROVIDER_KEY=false
for provider_key in OPENROUTER_API_KEY OPENAI_API_KEY MINIMAX_API_KEY GLM_API_KEY ZAI_API_KEY Z_AI_API_KEY; do
  if [[ "$provider_key" == "OPENAI_API_KEY" ]]; then
    if [[ -n "${OPENAI_API_KEY:-}" && -n "${OPENAI_BASE_URL:-}" ]]; then
      HERMES_HAS_PROVIDER_KEY=true
      break
    fi
    continue
  fi
  if [[ -n "${!provider_key:-}" ]]; then
    HERMES_HAS_PROVIDER_KEY=true
    break
  fi
done

if [[ "${HERMES_HAS_PROVIDER_KEY}" != "true" ]]; then
  cat >&2 <<EOF
Set at least one Hermes inference key in $ENV_FILE before starting the assistant stack.

Supported Docker-first options:
  - MINIMAX_API_KEY
  - ZAI_API_KEY or GLM_API_KEY
  - OPENROUTER_API_KEY
  - OPENAI_API_KEY together with OPENAI_BASE_URL for a custom OpenAI-compatible endpoint
EOF
  exit 1
fi

if [[ -n "${OPENAI_API_KEY:-}" && -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "OPENAI_BASE_URL must be set when OPENAI_API_KEY is provided in $ENV_FILE." >&2
  exit 1
fi

if [[ -n "${HERMES_API_BASE:-}" && "${HERMES_API_BASE}" != "http://hermes:8642/v1" ]]; then
  cat >&2 <<EOF
HERMES_API_BASE in $ENV_FILE must be http://hermes:8642/v1 for Docker assistant runs.
Current value: ${HERMES_API_BASE}
EOF
  exit 1
fi

GLM_API_KEY_VALUE="${GLM_API_KEY:-${ZAI_API_KEY:-${Z_AI_API_KEY:-}}}"
ZAI_API_KEY_VALUE="${ZAI_API_KEY:-${GLM_API_KEY:-${Z_AI_API_KEY:-}}}"
Z_AI_API_KEY_VALUE="${Z_AI_API_KEY:-${ZAI_API_KEY:-${GLM_API_KEY:-}}}"
GLM_BASE_URL_VALUE="${GLM_BASE_URL:-${ZAI_API_BASE:-https://api.z.ai/api/paas/v4}}"
HERMES_PROVIDER_VALUE="$(resolve_hermes_provider)"
HERMES_DEFAULT_MODEL_VALUE="$(resolve_hermes_model "$HERMES_PROVIDER_VALUE")"
MINIMAX_BASE_URL_VALUE="${MINIMAX_BASE_URL:-${MINIMAX_API_BASE:-https://api.minimax.io/v1}}"

validate_provider_credentials "$HERMES_PROVIDER_VALUE"

render_hermes_config() {
  cat <<EOF
model:
  provider: ${HERMES_PROVIDER_VALUE}
  default: ${HERMES_DEFAULT_MODEL_VALUE}

mcp_servers:
  stockscreen_market:
    url: "http://backend:8000/mcp/"
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
    - /opt/external-skills
EOF
}

render_hermes_env() {
  cat <<EOF
API_SERVER_ENABLED=true
API_SERVER_HOST=0.0.0.0
API_SERVER_PORT=8642
API_SERVER_KEY=${HERMES_API_KEY:-}
HERMES_INFERENCE_PROVIDER=${HERMES_PROVIDER_VALUE}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-}
GLM_API_KEY=${GLM_API_KEY_VALUE}
ZAI_API_KEY=${ZAI_API_KEY_VALUE}
Z_AI_API_KEY=${Z_AI_API_KEY_VALUE}
GLM_BASE_URL=${GLM_BASE_URL_VALUE}
MINIMAX_API_KEY=${MINIMAX_API_KEY:-}
MINIMAX_BASE_URL=${MINIMAX_BASE_URL_VALUE}
TAVILY_API_KEY=${TAVILY_API_KEY:-}
SERPER_API_KEY=${SERPER_API_KEY:-}
EOF
}

export HERMES_PLATFORM="${HERMES_PLATFORM:-linux/amd64}"
COMPOSE_CMD=(
  docker compose
  --env-file "$ENV_FILE"
  --profile assistant
  up -d
  postgres redis backend frontend hermes
)

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Using env file: $ENV_FILE"
  echo "Using Hermes platform: $HERMES_PLATFORM"
  echo "Using Hermes provider/model: ${HERMES_PROVIDER_VALUE}/${HERMES_DEFAULT_MODEL_VALUE}"
  printf 'Would run:'
  printf ' %q' "${COMPOSE_CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$ROOT_DIR/data/hermes"
render_hermes_config > "$ROOT_DIR/data/hermes/config.yaml"
render_hermes_env > "$ROOT_DIR/data/hermes/.env"
chmod 600 "$ROOT_DIR/data/hermes/config.yaml" "$ROOT_DIR/data/hermes/.env"

echo "Prepared data/hermes/config.yaml and data/hermes/.env"
echo "Using env file: $ENV_FILE"
echo "Using Hermes platform: $HERMES_PLATFORM"
echo "Using Hermes provider/model: ${HERMES_PROVIDER_VALUE}/${HERMES_DEFAULT_MODEL_VALUE}"

"${COMPOSE_CMD[@]}"

cat <<EOF

Assistant stack started.

Open:
  http://localhost/chatbot

Check health:
  docker compose --env-file "$ENV_FILE" --profile assistant ps
  docker compose --env-file "$ENV_FILE" --profile assistant logs -f backend hermes frontend
EOF
