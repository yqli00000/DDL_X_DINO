#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_PATH="cfgs/train/gps_dino_mask_mixed_phase1_phase2_wo_maskloss.yaml"

# Fill this in if you do not want to run `wandb login` first.
export WANDB_API_KEY=""

export WANDB_MODE="online"
WANDB_PROJECT="DDL——MIX"
WANDB_RUN_NAME="gps_dino_mask_track1_tarck2_ddp"
WANDB_NUM_VAL_IMAGES=4

NUM_GPUS=3  # you can change the num_gpus to match your machine
# export CUDA_VISIBLE_DEVICES="4,5,6"
export MASTER_PORT=$((10000 + RANDOM % 50000))
cd "${PROJECT_DIR}"

python -m torch.distributed.run \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --cfg "${CFG_PATH}" \
  --logdir gps_dino_mask_mixed_test \
  train.wandb.enabled=true \
  train.wandb.project="${WANDB_PROJECT}" \
  train.wandb.name="${WANDB_RUN_NAME}" \
  train.wandb.mode="${WANDB_MODE}" \
  train.wandb.num_val_images="${WANDB_NUM_VAL_IMAGES}" \
  train.accelerator=gpu \
  train.gpu_ids="[0,1,2]"
