#!/bin/sh
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
"$ROOT_DIR/.venv/bin/python" scripts/download_assets.py
"$ROOT_DIR/.venv/bin/python" generator/generate_dataset.py \
  --config configs/field.yaml --output visual_review --visual-review --overwrite
"$ROOT_DIR/.venv/bin/python" tests/validate_dataset.py visual_review
"$ROOT_DIR/.venv/bin/python" scripts/create_contact_sheet.py visual_review
"$ROOT_DIR/.venv/bin/python" scripts/create_landing_contact_sheet.py visual_review
