#!/usr/bin/env bash

set -euo pipefail
export CUDA_VISIBLE_DEVICES="0"
# Usage:
#   bash update_submission_new.sh
#
# Edit the paths below before running.

OUTPUT_DIR="outputs/test"

# Choose one image input source:
IMAGE_PATH=""
IMAGE_DIR="DATASET_DDL/DDL_X_test/image"
IMAGE_SIZE=512
BATCH_SIZE=8
DEVICE="cuda"
FAKE_THRESHOLD=0.5
MASK_THRESHOLD=0.5
MIN_BOX_AREA=16

# DashScope / OpenAI-compatible explanation API settings.
EXPLAIN_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
EXPLAIN_API_KEY=""  ## fix it before run
EXPLAIN_MODEL="qwen3.5-flash"
EXPLAIN_TIMEOUT=60

# Set to true to reuse "Visible forgery traces" from an existing JSON file if present.
REUSE_EXISTING_TRACES="false"

cd "$(dirname "$0")"
  # --image-dir "${IMAGE_DIR}"  --image-path "${IMAGE_PATH}"
CMD=(
  python update_json_traces.py
  --image-dir "${IMAGE_DIR}"
  --image-path "${IMAGE_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --explain-api-url "${EXPLAIN_API_URL}"
  --explain-api-key "${EXPLAIN_API_KEY}"
  --explain-model "${EXPLAIN_MODEL}"
  --explain-timeout "${EXPLAIN_TIMEOUT}"
  --explain-workers 4
)

if [[ "${REUSE_EXISTING_TRACES}" == "true" ]]; then
  CMD+=(--reuse-existing-traces)
fi

"${CMD[@]}"
