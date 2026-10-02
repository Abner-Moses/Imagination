#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

RGB_COUNT=$(find dataset -path '*/rgb/*.jpg' -type f 2>/dev/null | wc -l | tr -d ' ')
COMPLETE_EPISODES=$(find dataset -name metadata.json -type f 2>/dev/null | wc -l | tr -d ' ')
SIZE=$(du -sh dataset 2>/dev/null | awk '{print $1}')
SIZE=${SIZE:-0}
echo "RGB frames: $RGB_COUNT / 20000"
echo "Complete episodes: $COMPLETE_EPISODES / 250"
echo "Dataset size: $SIZE"
if launchctl print "gui/$(id -u)/com.imagination.generate20k" >/dev/null 2>&1; then
  echo "Background job: running"
else
  echo "Background job: not registered"
fi
tail -n 5 logs/generate_20k.log 2>/dev/null || true
