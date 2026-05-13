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

echo "→ extracting frames at 10 fps"
ffmpeg -y -loglevel warning -i "$TMP/video.webm" \
  -vf "fps=10,scale=1100:-1:flags=lanczos" \
  "$TMP/frames/frame_%04d.png"

echo "→ encoding GIF with gifski"
# fps=10, width=1000, quality=70 tuned for the long storyboard (~40s) to stay near the 5 MB budget
gifski --fps 10 --width 1000 --quality 70 \
  -o "$OUT" "$TMP"/frames/frame_*.png

ls -lh "$OUT"
echo "✓ done"
