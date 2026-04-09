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

cat > "$HERMES_HOME_DIR/config.yaml" <<EOF
mcp_servers:
  stockscreen_market:
    url: "http://127.0.0.1:8000/mcp"
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

echo "Starting Hermes with HERMES_HOME=$HERMES_HOME_DIR on http://${API_SERVER_HOST}:${API_SERVER_PORT}/v1"
exec hermes gateway
