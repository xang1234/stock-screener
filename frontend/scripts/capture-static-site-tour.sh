#!/usr/bin/env bash
# Capture a hero GIF tour of the static site at https://xang1234.github.io/stock-screener/.
# Pipeline: Playwright (record WebM) → ffmpeg (extract frames) → gifski (palette + quantize).
#
# Requirements (already installed on this machine):
#   - node 18+ with frontend/node_modules/playwright present
#   - ffmpeg, gifski on PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP="$ROOT/.tmp/hero-capture"
OUT="$ROOT/docs/gifs/static-site-tour.gif"

mkdir -p "$TMP/frames" "$ROOT/docs/gifs"
rm -f "$TMP"/*.webm "$TMP"/frames/*.png 2>/dev/null || true

echo "→ capturing video via Playwright"
cd "$ROOT/frontend"
node scripts/capture-static-site-tour.mjs

echo "→ extracting frames at 12 fps"
ffmpeg -y -loglevel warning -i "$TMP/video.webm" \
  -vf "fps=12,scale=1280:-1:flags=lanczos" \
  "$TMP/frames/frame_%04d.png"

echo "→ encoding GIF with gifski"
# width 1200 + quality 80 tuned to keep output under the 5 MB SCREENSHOT_GUIDE budget
gifski --fps 12 --width 1200 --quality 80 \
  -o "$OUT" "$TMP"/frames/frame_*.png

ls -lh "$OUT"
echo "✓ done"
