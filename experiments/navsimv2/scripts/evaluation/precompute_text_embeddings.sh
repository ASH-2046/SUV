#!/usr/bin/env bash
# Build one shared text-embedding cache for NAVSIM v2 navtest and navhard.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
NAVSIM_ROOT="${REPO_ROOT}/experiments/navsimv2/vendor"

# ======================== User configuration ========================
# Edit the paths below before running this script.

# Local checkpoint root
# Expected Wan layout:
#   ${CKPT_LOCAL_DIR}/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
#   ${CKPT_LOCAL_DIR}/Wan2.2-TI2V-5B/google/umt5-xxl/
CKPT_LOCAL_DIR="/path/to/checkpoints"

# Shared text-embedding output directory for navtest and navhard
TEXT_EMBEDDING_CACHE_DIR="/path/to/suv_text_embeds_cache_test/navsim_v2"

# NAVSIM v2 data
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/navsim}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
ORIGINAL_SENSOR_PATH="${ORIGINAL_SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles}"

# Text encoder settings: embeddings are sharded across four GPUs by default
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OVERWRITE="${OVERWRITE:-false}"
# ====================================================================

export CKPT_LOCAL_DIR
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${NAVSIM_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  "${REPO_ROOT}/experiments/navsimv2/precompute_text_embeddings.py" \
  --train-test-split all \
  --navsim-log-path "${NAVSIM_LOG_PATH}" \
  --original-sensor-path "${ORIGINAL_SENSOR_PATH}" \
  --synthetic-sensor-path "${SYNTHETIC_SENSOR_PATH}" \
  --synthetic-scenes-path "${SYNTHETIC_SCENES_PATH}" \
  --text-embedding-cache-dir "${TEXT_EMBEDDING_CACHE_DIR}" \
  --num-history-frames 4 \
  --num-future-frames 8 \
  --slot-inference true \
  --model-id Wan2.2-TI2V-5B \
  --tokenizer-model-id Wan2.2-TI2V-5B \
  --batch-size "${BATCH_SIZE}" \
  --overwrite "${OVERWRITE}" \
  "$@"
