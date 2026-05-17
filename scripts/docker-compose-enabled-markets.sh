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

env_files_from_args() {
  local next_is_env_file=0

  for arg in "$@"; do
    if [[ "$next_is_env_file" == "1" ]]; then
      printf '%s\n' "$arg"
      next_is_env_file=0
      continue
    fi
    case "$arg" in
      --env-file)
        next_is_env_file=1
        ;;
      --env-file=*)
        printf '%s\n' "${arg#--env-file=}"
        ;;
    esac
  done
}

env_file_path() {
  local file="$1"
  if [[ "$file" = /* ]]; then
    printf '%s\n' "$file"
  else
    printf '%s\n' "$ROOT_DIR/$file"
  fi
}

read_env_value_from_files() {
  local key="$1"
  shift
  local file
  local value
  local found=0

  for file in "$@"; do
    if value="$(read_env_value "$file" "$key")"; then
      printf -v RESULT '%s' "$value"
      found=1
    fi
  done

  [[ "$found" == "1" ]]
}

has_arg() {
  local needle="$1"
  shift
  local arg

  for arg in "$@"; do
    if [[ "$arg" == "$needle" ]]; then
      return 0
    fi
  done

  return 1
}

is_down_command() {
  local arg

  for arg in "$@"; do
    if [[ "$arg" == "down" ]]; then
      return 0
    fi
  done

  return 1
}

mapfile -t PROVIDED_ENV_FILES < <(env_files_from_args "$@")
RESOLVED_ENV_FILES=()
for ENV_FILE in "${PROVIDED_ENV_FILES[@]}"; do
  RESOLVED_ENV_FILES+=("$(env_file_path "$ENV_FILE")")
done

ENV_FILE_TO_FORWARD=""
if [[ "${#RESOLVED_ENV_FILES[@]}" -eq 0 ]]; then
  if [[ -f "$ROOT_DIR/.env" ]]; then
    ENV_FILE_TO_FORWARD="$ROOT_DIR/.env"
    RESOLVED_ENV_FILES+=("$ENV_FILE_TO_FORWARD")
  elif [[ -f "$ROOT_DIR/.env.docker" ]]; then
    ENV_FILE_TO_FORWARD="$ROOT_DIR/.env.docker"
    RESOLVED_ENV_FILES+=("$ENV_FILE_TO_FORWARD")
  fi
fi

MARKETS="${ENABLED_MARKETS:-}"
if [[ -z "$MARKETS" ]]; then
  if [[ "${#RESOLVED_ENV_FILES[@]}" -gt 0 ]] && read_env_value_from_files ENABLED_MARKETS "${RESOLVED_ENV_FILES[@]}"; then
    MARKETS="$RESULT"
  else
    MARKETS="US"
  fi
fi

MARKET_PROFILES="$(python3 "$ROOT_DIR/backend/scripts/compose_enabled_markets.py" profiles --markets "$MARKETS")"
if is_down_command "$@"; then
  ALL_MARKET_PROFILES="$(python3 "$ROOT_DIR/backend/scripts/compose_enabled_markets.py" profiles --markets "$(python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1])
backend_root = root / "backend"
sys.path.insert(0, str(backend_root))

from app.domain.markets import market_registry

print(",".join(market_registry.supported_market_codes()))
PY
)")"
  MARKET_PROFILES="$ALL_MARKET_PROFILES"
fi

if [[ -n "${COMPOSE_PROFILES:-}" ]]; then
  PROFILES="$COMPOSE_PROFILES,$MARKET_PROFILES"
elif [[ "${#RESOLVED_ENV_FILES[@]}" -gt 0 ]] && read_env_value_from_files COMPOSE_PROFILES "${RESOLVED_ENV_FILES[@]}"; then
  PROFILES="$RESULT,$MARKET_PROFILES"
else
  PROFILES="$MARKET_PROFILES"
fi

export ENABLED_MARKETS="$MARKETS"
export COMPOSE_PROFILES="$PROFILES"

echo "ENABLED_MARKETS=$ENABLED_MARKETS"
echo "COMPOSE_PROFILES=$COMPOSE_PROFILES"

COMPOSE_ARGS=()
if [[ -n "$ENV_FILE_TO_FORWARD" ]]; then
  COMPOSE_ARGS+=(--env-file "$ENV_FILE_TO_FORWARD")
fi
COMPOSE_ARGS+=("$@")

if is_down_command "${COMPOSE_ARGS[@]}" && ! has_arg "--remove-orphans" "${COMPOSE_ARGS[@]}"; then
  COMPOSE_ARGS+=("--remove-orphans")
fi

exec docker compose "${COMPOSE_ARGS[@]}"
