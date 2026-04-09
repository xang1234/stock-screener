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

if [[ -n "${HERMES_API_BASE:-}" && "${HERMES_API_BASE}" != "http://hermes:8642/v1" ]]; then
  cat >&2 <<EOF
HERMES_API_BASE in $ENV_FILE must be http://hermes:8642/v1 for Docker assistant runs.
Current value: ${HERMES_API_BASE}
EOF
  exit 1
fi

mkdir -p "$ROOT_DIR/data/hermes"

cat > "$ROOT_DIR/data/hermes/config.yaml" <<EOF
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
MINIMAX_API_KEY=${MINIMAX_API_KEY:-}
MINIMAX_BASE_URL=${MINIMAX_API_BASE:-https://api.minimax.io/v1}
GROQ_API_KEY=${GROQ_API_KEY:-}
GROQ_API_KEYS=${GROQ_API_KEYS:-}
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
