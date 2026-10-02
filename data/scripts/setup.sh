#!/bin/sh
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BLENDER_APP=${BLENDER_APP:-/Applications/Blender.app}
BLENDER_BIN="$BLENDER_APP/Contents/MacOS/Blender"
if [ ! -x "$BLENDER_BIN" ]; then
  echo "Blender executable not found: $BLENDER_BIN" >&2
  exit 1
fi
"$BLENDER_BIN" --version | head -n 1
cd "$ROOT_DIR"
python3 -m venv .venv
"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT_DIR/.venv/bin/pip" install -r requirements.txt
"$ROOT_DIR/.venv/bin/blenderproc" --version
echo "Setup complete. Run: ./scripts/run_smoke_test.sh"

