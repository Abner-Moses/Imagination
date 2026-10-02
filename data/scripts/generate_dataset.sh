#!/bin/sh
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" generator/generate_dataset.py --config configs/field.yaml "$@"

