#!/usr/bin/env bash

set -euo pipefail
export CUDA_VISIBLE_DEVICES="0"
# Usage:
#   bash infer_submission.sh
#
# Edit the paths below before running.

CHECKPOINT="weights/model_best.ckpt"
OUTPUT_DIR="outputs/test"

# Choose one image input source:
IMAGE_PATH=""
# or the test image folder
IMAGE_DIR="DATASET_DDL/DDL_X_test/image"

IMAGE_SIZE=768
BATCH_SIZE=4
DEVICE="cuda"
FAKE_THRESHOLD=0.5
MASK_THRESHOLD=0.4
MIN_BOX_AREA=8

# DashScope / OpenAI-compatible explanation API settings.
EXPLAIN_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
EXPLAIN_API_KEY=""  # fix your api-key
EXPLAIN_MODEL="qwen3.5-flash"
EXPLAIN_TIMEOUT=60

# Set to true to reuse "Visible forgery traces" from an existing JSON file if present.
REUSE_EXISTING_TRACES="false"

cd "$(dirname "$0")"

CMD=(
  python infer_submission.py
  --checkpoint "${CHECKPOINT}"
  --image-path "${IMAGE_PATH}"
  --image-dir "${IMAGE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --image-size "${IMAGE_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --fake-threshold "${FAKE_THRESHOLD}"
  --mask-threshold "${MASK_THRESHOLD}"
  --min-box-area "${MIN_BOX_AREA}"
  --explain-api-url "${EXPLAIN_API_URL}"
  --explain-api-key "${EXPLAIN_API_KEY}"
  --explain-model "${EXPLAIN_MODEL}"
  --explain-timeout "${EXPLAIN_TIMEOUT}"
  --save-mask-png
  # --backbone-path "weights/dinov3-l16"  # optional; you can replace it to match the target machine.
  --explain-workers 4
  --reuse-existing-traces
  --max-api-calls 0
)

if [[ "${REUSE_EXISTING_TRACES}" == "true" ]]; then
  CMD+=(--reuse-existing-traces)
fi

"${CMD[@]}"
