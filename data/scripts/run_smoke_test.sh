#!/bin/sh
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
if [ "${IMAGINATION_VISUAL_APPROVED:-0}" != "1" ]; then
  echo "Visual-approval gate is active. Review visual_review/contact_sheet.png first." >&2
  echo "After explicit approval, run: IMAGINATION_VISUAL_APPROVED=1 ./scripts/run_smoke_test.sh --overwrite" >&2
  exit 2
fi
"$ROOT_DIR/.venv/bin/python" generator/generate_dataset.py \
  --config configs/field.yaml --output dataset \
  --num-episodes 5 --trajectories-per-layout 1 --frames-per-trajectory 20 --visual-approved "$@"
"$ROOT_DIR/.venv/bin/python" tests/validate_dataset.py dataset
