#!/bin/sh
set -u

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR" || exit 1

LOCK_DIR="$ROOT_DIR/.runtime/generate_20k.lock"
mkdir -p "$ROOT_DIR/.runtime"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [ -f "$LOCK_DIR/pid" ]; then
    EXISTING_PID=$(sed -n '1p' "$LOCK_DIR/pid")
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
      echo "20k generator is already running as PID $EXISTING_PID" >&2
      exit 2
    fi
  fi
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || exit 2
  mkdir "$LOCK_DIR" || exit 2
fi
echo $$ > "$LOCK_DIR/pid"
cleanup() {
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ATTEMPT=0
while :; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 20k generation attempt $ATTEMPT"
  "$ROOT_DIR/.venv/bin/python" generator/generate_dataset.py \
    --config configs/field.yaml \
    --output dataset \
    --num-episodes 250 \
    --trajectories-per-layout 1 \
    --frames-per-trajectory 80 \
    --base-seed 24051991 \
    --visual-approved \
    --resume
  STATUS=$?
  if [ "$STATUS" -eq 0 ]; then
    break
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] generator stopped with status $STATUS; resuming in 5 seconds"
  sleep 5
done

"$ROOT_DIR/.venv/bin/python" tests/validate_dataset.py dataset \
  --report dataset/validation_report.json || exit $?

FRAME_COUNT=$(find dataset -path '*/rgb/*.jpg' -type f | wc -l | tr -d ' ')
if [ "$FRAME_COUNT" != "20000" ]; then
  echo "Expected exactly 20000 RGB frames, found $FRAME_COUNT" >&2
  exit 3
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETE: 20000 validated frames"
# A launchctl-submitted job is kept alive by macOS. Remove the registration
# only after successful count + validation so it cannot restart needlessly.
launchctl remove com.imagination.generate20k 2>/dev/null || true
