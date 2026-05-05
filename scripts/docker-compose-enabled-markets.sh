#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

read_env_value() {
  local file="$1"
  local key="$2"

  if [[ ! -f "$file" ]]; then
    return 1
  fi

  python3 - "$file" "$key" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]

for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    name, value = line.split("=", 1)
    if name.strip() != key:
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    print(value)
    raise SystemExit(0)

raise SystemExit(1)
PY
}

env_file_from_args() {
  local previous=""

  for arg in "$@"; do
    if [[ "$previous" == "--env-file" ]]; then
      printf '%s\n' "$arg"
      return 0
    fi
    case "$arg" in
      --env-file=*)
        printf '%s\n' "${arg#--env-file=}"
        return 0
        ;;
    esac
    previous="$arg"
  done

  return 1
}

env_file_path() {
  local file="$1"
  if [[ "$file" = /* ]]; then
    printf '%s\n' "$file"
  else
    printf '%s\n' "$ROOT_DIR/$file"
  fi
}

MARKETS="${ENABLED_MARKETS:-}"
if [[ -z "$MARKETS" ]]; then
  if ENV_FILE="$(env_file_from_args "$@")" && VALUE="$(read_env_value "$(env_file_path "$ENV_FILE")" ENABLED_MARKETS)"; then
    MARKETS="$VALUE"
  elif VALUE="$(read_env_value "$ROOT_DIR/.env" ENABLED_MARKETS)"; then
    MARKETS="$VALUE"
  elif VALUE="$(read_env_value "$ROOT_DIR/.env.docker" ENABLED_MARKETS)"; then
    MARKETS="$VALUE"
  else
    MARKETS="US"
  fi
fi

MARKET_PROFILES="$(python3 "$ROOT_DIR/backend/scripts/compose_enabled_markets.py" profiles --markets "$MARKETS")"
if [[ -n "${COMPOSE_PROFILES:-}" ]]; then
  PROFILES="$COMPOSE_PROFILES,$MARKET_PROFILES"
else
  PROFILES="$MARKET_PROFILES"
fi

export ENABLED_MARKETS="$MARKETS"
export COMPOSE_PROFILES="$PROFILES"

echo "ENABLED_MARKETS=$ENABLED_MARKETS"
echo "COMPOSE_PROFILES=$COMPOSE_PROFILES"

exec docker compose "$@"
