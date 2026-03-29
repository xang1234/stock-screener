#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PATH="$ROOT_DIR/dist/StockScanner.app"
DMG_PATH="$ROOT_DIR/dist/StockScanner.dmg"

if [ ! -d "$APP_PATH" ]; then
  echo "Missing $APP_PATH. Build the app bundle first."
  exit 1
fi

rm -f "$DMG_PATH"
hdiutil create \
  -volname "StockScanner" \
  -srcfolder "$APP_PATH" \
  -ov \
  -format UDZO \
  "$DMG_PATH"
