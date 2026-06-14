#!/usr/bin/env bash
# Capture a hero GIF tour of the static site at https://xang1234.github.io/stock-screener/.
# Storyboard mirrors the live-app tour: Daily Snapshot → Scan (drill into a stock
# detail) → Breadth → Groups (ranked table + RRG rotation).
# Pipeline: Playwright (record WebM) → ffmpeg (trim + speed + frames) →
#           ffmpeg palettegen/paletteuse (dither=none) → gifsicle.
#
# Why dither=none + gifsicle (not gifski): the dark UI background is a near-black
# gradient. Error-diffusion dithering (gifski's default, no opt-out) scatters it
# into patchy speckle. A flat global palette with dither=none keeps the darks
# uniform; gifsicle -O3 --lossy then claws back the size.
#
# Requirements:
#   - node 18+ with frontend/node_modules/playwright present
#   - ffmpeg, gifsicle on PATH  (brew install ffmpeg gifsicle)

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

echo "→ extracting frames (trim 2.6s head, 1.5x speed, 13 fps)"
ffmpeg -y -loglevel warning -ss 2.6 -i "$TMP/video.webm" \
  -vf "setpts=PTS/1.5,fps=13,scale=1200:-1:flags=lanczos" \
  "$TMP/frames/f_%04d.png"

echo "→ building flat global palette (full stats, 256 colors)"
ffmpeg -y -loglevel error -framerate 13 -i "$TMP/frames/f_%04d.png" \
  -vf "fps=11,scale=960:-1:flags=lanczos,palettegen=max_colors=256:stats_mode=full" \
  "$TMP/palette.png"

echo "→ encoding GIF with dither=none (uniform darks)"
ffmpeg -y -loglevel error -framerate 13 -i "$TMP/frames/f_%04d.png" -i "$TMP/palette.png" \
  -lavfi "fps=11,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle" \
  "$TMP/raw.gif"

echo "→ optimizing with gifsicle (-O3 --lossy=30)"
gifsicle -O3 --lossy=30 "$TMP/raw.gif" -o "$OUT"

ls -lh "$OUT"
echo "✓ done"
