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

HERMES_HAS_PROVIDER_KEY=false
for provider_key in OPENROUTER_API_KEY OPENAI_API_KEY MINIMAX_API_KEY GLM_API_KEY ZAI_API_KEY Z_AI_API_KEY; do
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

if [[ -n "${HERMES_API_BASE:-}" && "${HERMES_API_BASE}" != "http://hermes:8642/v1" ]]; then
  cat >&2 <<EOF
HERMES_API_BASE in $ENV_FILE must be http://hermes:8642/v1 for Docker assistant runs.
Current value: ${HERMES_API_BASE}
EOF
  exit 1
fi

mkdir -p "$ROOT_DIR/data/hermes"

GLM_API_KEY_VALUE="${GLM_API_KEY:-${ZAI_API_KEY:-${Z_AI_API_KEY:-}}}"
ZAI_API_KEY_VALUE="${ZAI_API_KEY:-${GLM_API_KEY:-${Z_AI_API_KEY:-}}}"
Z_AI_API_KEY_VALUE="${Z_AI_API_KEY:-${ZAI_API_KEY:-${GLM_API_KEY:-}}}"
GLM_BASE_URL_VALUE="${GLM_BASE_URL:-${ZAI_API_BASE:-https://api.z.ai/api/paas/v4}}"

cat > "$ROOT_DIR/data/hermes/config.yaml" <<EOF
model:
  provider: ${HERMES_INFERENCE_PROVIDER:-auto}

mcp_servers:
  stockscreen_market:
    url: "http://backend:8000/mcp"
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

cat > "$ROOT_DIR/data/hermes/.env" <<EOF
API_SERVER_ENABLED=true
API_SERVER_HOST=0.0.0.0
API_SERVER_PORT=8642
API_SERVER_KEY=${HERMES_API_KEY:-}
HERMES_INFERENCE_PROVIDER=${HERMES_INFERENCE_PROVIDER:-}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-}
GLM_API_KEY=${GLM_API_KEY_VALUE}
ZAI_API_KEY=${ZAI_API_KEY_VALUE}
Z_AI_API_KEY=${Z_AI_API_KEY_VALUE}
GLM_BASE_URL=${GLM_BASE_URL_VALUE}
MINIMAX_API_KEY=${MINIMAX_API_KEY:-}
MINIMAX_BASE_URL=${MINIMAX_API_BASE:-https://api.minimax.io/v1}
TAVILY_API_KEY=${TAVILY_API_KEY:-}
SERPER_API_KEY=${SERPER_API_KEY:-}
EOF

export HERMES_PLATFORM="${HERMES_PLATFORM:-linux/amd64}"
COMPOSE_CMD=(
  docker compose
  --env-file "$ENV_FILE"
  --profile assistant
  up -d
  postgres redis backend frontend hermes
)

echo "Prepared data/hermes/config.yaml and data/hermes/.env"
echo "Using env file: $ENV_FILE"
echo "Using Hermes platform: $HERMES_PLATFORM"

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'Would run:'
  printf ' %q' "${COMPOSE_CMD[@]}"
  printf '\n'
  exit 0
fi

"${COMPOSE_CMD[@]}"

cat <<EOF

Assistant stack started.

Open:
  http://localhost/chatbot

Check health:
  docker compose --env-file "$ENV_FILE" --profile assistant ps
  docker compose --env-file "$ENV_FILE" --profile assistant logs -f backend hermes frontend
EOF
