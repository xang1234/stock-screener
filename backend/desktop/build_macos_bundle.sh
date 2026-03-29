#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"

if [ ! -d "frontend/dist" ]; then
  echo "frontend/dist is missing. Build the frontend first with VITE_API_URL=/api."
  exit 1
fi

pyinstaller backend/desktop/StockScanner.macos.spec --noconfirm --clean
