#!/bin/sh
# Convenience only; all orchestration lives in the cross-platform Python runner.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python "$SCRIPT_DIR/run.py" "$@"
