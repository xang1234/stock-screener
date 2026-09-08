#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_COMPOSE="$ROOT_DIR/docker-compose.yml"
SOCIAL_COMPOSE="$ROOT_DIR/docker-compose.social.yml"
XUI_COMPOSE="$ROOT_DIR/docker-compose.social-xui.yml"
STACK_WRAPPER="$ROOT_DIR/scripts/docker-compose-enabled-markets.sh"

die() {
  printf 'Social Signal stack: %s\n' "$1" >&2
  exit 1
}

select_env_file() {
  if [[ -n "${SOCIAL_STACK_ENV_FILE:-}" ]]; then
    if [[ "$SOCIAL_STACK_ENV_FILE" = /* ]]; then
      printf '%s\n' "$SOCIAL_STACK_ENV_FILE"
    else
      printf '%s/%s\n' "$ROOT_DIR" "$SOCIAL_STACK_ENV_FILE"
    fi
  elif [[ -f "$ROOT_DIR/.env" ]]; then
    printf '%s\n' "$ROOT_DIR/.env"
  elif [[ -f "$ROOT_DIR/.env.docker" ]]; then
    printf '%s\n' "$ROOT_DIR/.env.docker"
  else
    die "missing .env or .env.docker; copy .env.docker.example first"
  fi
}

read_env_value() {
  local key="$1"
  local line value=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      "$key"=*) value="${line#*=}" ;;
    esac
  done < "$ENV_FILE"
  if [[ ${#value} -ge 2 ]]; then
    if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] \
      || [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi
  fi
  printf '%s\n' "$value"
}

setting() {
  local key="$1"
  local value="${!key:-}"
  if [[ -z "$value" ]]; then
    value="$(read_env_value "$key")"
  fi
  printf '%s\n' "$value"
}

compose_social() {
  docker compose --env-file "$ENV_FILE" \
    -f "$BASE_COMPOSE" \
    -f "$SOCIAL_COMPOSE" \
    -f "$XUI_COMPOSE" \
    --profile social "$@"
}

validate_profile() {
  XUI_PROFILE_DIR_VALUE="$(setting XUI_PROFILE_DIR)"
  [[ -n "$XUI_PROFILE_DIR_VALUE" ]] || die "XUI_PROFILE_DIR is not configured"
  [[ -f "$XUI_PROFILE_DIR_VALUE/config.toml" ]] \
    || die "xui profile config not found at $XUI_PROFILE_DIR_VALUE/config.toml"
}

local_up() {
  local ref known_hosts image
  ref="$(setting XUI_READER_REF)"
  known_hosts="$(setting GITHUB_KNOWN_HOSTS_FILE)"
  image="$(setting SOCIAL_WORKER_IMAGE)"
  image="${image:-stock-screener-social-xui:dev}"

  [[ "$ref" =~ ^[0-9a-fA-F]{40}$ ]] \
    || die "XUI_READER_REF must be a full 40-character commit SHA"
  [[ -n "$known_hosts" && -s "$known_hosts" ]] \
    || die "GITHUB_KNOWN_HOSTS_FILE must point to a non-empty verified file"
  validate_profile
  ssh-add -l >/dev/null 2>&1 \
    || die "no SSH key is loaded; run ssh-add before building"

  DOCKER_BUILDKIT=1 docker build \
    --ssh default \
    --secret "id=github_known_hosts,src=$known_hosts" \
    --target social-xui \
    --build-arg "XUI_READER_REF=$ref" \
    -t "$image" \
    -f "$ROOT_DIR/backend/Dockerfile" \
    "$ROOT_DIR"

  "$STACK_WRAPPER" --env-file "$ENV_FILE" up -d
  compose_social up -d --no-build celery-social
  printf 'Social Signal stack is running with local image %s.\n' "$image"
  printf 'Open http://localhost and activate validation from Operations.\n'
}

ghcr_up() {
  local image
  image="$(setting SOCIAL_WORKER_IMAGE)"
  [[ "$image" =~ ^ghcr\.io/[^/]+/[^:@]+@sha256:[0-9a-fA-F]{64}$ \
    || "$image" =~ ^ghcr\.io/[^/]+/[^:@]+:sha-[0-9a-fA-F]{40}$ \
    || "$image" =~ ^ghcr\.io/[^/]+/[^:@]+:v[0-9][A-Za-z0-9._-]*$ ]] \
    || die "SOCIAL_WORKER_IMAGE must use a GHCR digest, full sha-* tag, or v* release tag; latest is forbidden"
  validate_profile

  if ! compose_social pull celery-social; then
    printf 'GHCR pull failed. Authenticate with: docker login ghcr.io -u <github-username>\n' >&2
    exit 1
  fi
  "$STACK_WRAPPER" --env-file "$ENV_FILE" up -d
  compose_social up -d --no-build celery-social
  printf 'Social Signal stack is running with GHCR image %s.\n' "$image"
  printf 'Open http://localhost and activate validation from Operations.\n'
}

usage() {
  printf 'Usage:\n' >&2
  printf '  %s {local|ghcr} up\n' "${0##*/}" >&2
  printf '  %s {status|logs|stop}\n' "${0##*/}" >&2
  exit 2
}

ENV_FILE="$(select_env_file)"
[[ -f "$ENV_FILE" ]] || die "environment file not found: $ENV_FILE"
cd "$ROOT_DIR"

if [[ "${1:-}" == "local" && "${2:-}" == "up" && $# -eq 2 ]]; then
  local_up
elif [[ "${1:-}" == "ghcr" && "${2:-}" == "up" && $# -eq 2 ]]; then
  ghcr_up
elif [[ "${1:-}" == "status" && $# -eq 1 ]]; then
  compose_social ps celery-social
elif [[ "${1:-}" == "logs" && $# -eq 1 ]]; then
  compose_social logs -f celery-social
elif [[ "${1:-}" == "stop" && $# -eq 1 ]]; then
  compose_social stop celery-social
else
  usage
fi
