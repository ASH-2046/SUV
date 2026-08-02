#!/usr/bin/env bash
# Prepare the dynamic-prompt text embeddings used by NAVSIM v1 PDM evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${NAVSIM_DEVKIT_ROOT:-}" && -d "${NAVSIM_DEVKIT_ROOT}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
fi
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-${REPO_ROOT}/checkpoints}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${CKPT_LOCAL_DIR}}"

OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${REPO_ROOT}/data/navsim}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${CKPT_LOCAL_DIR}/suv_text_embeds_cache_test/navsim_v1}"
SUV_MODEL_ID="${SUV_MODEL_ID:-Wan2.2-TI2V-5B}"
SUV_TOKENIZER_MODEL_ID="${SUV_TOKENIZER_MODEL_ID:-Wan2.2-TI2V-5B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for required_path in "${NAVSIM_LOG_PATH}" "${SENSOR_BLOBS_PATH}"; do
  if [[ ! -d "${required_path}" ]]; then
    echo "ERROR: required NAVSIM path not found: ${required_path}" >&2
    exit 1
  fi
done

echo "SUV NAVSIM v1 PDM text embedding preparation"
echo "  logs:         ${NAVSIM_LOG_PATH}"
echo "  sensor blobs: ${SENSOR_BLOBS_PATH}"
echo "  output:       ${TEXT_EMBEDDING_CACHE_DIR}"
echo "  cuda:         ${CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u experiments/navsimv1/precompute_text_embeds.py \
  --train-test-split navtest \
  --navsim-log-path "${NAVSIM_LOG_PATH}" \
  --sensor-blobs-path "${SENSOR_BLOBS_PATH}" \
  --text-embedding-cache-dir "${TEXT_EMBEDDING_CACHE_DIR}" \
  --model-id "${SUV_MODEL_ID}" \
  --tokenizer-model-id "${SUV_TOKENIZER_MODEL_ID}" \
  --num-history-frames "${SUV_PDM_HISTORY_FRAMES:-4}" \
  --num-future-frames "${SUV_NUM_FUTURE_FRAMES:-8}" \
  --frame-interval "${SUV_FRAME_INTERVAL:-1}" \
  --fps "${SUV_FPS:-2.0}" \
  --prompt-mode "${SUV_PROMPT_MODE:-dynamic}" \
  --prompt-velocity-quantization "${SUV_PROMPT_VELOCITY_QUANTIZATION:-0.5}" \
  --prompt-acceleration-quantization "${SUV_PROMPT_ACCELERATION_QUANTIZATION:-0.5}" \
  --overwrite "${OVERWRITE:-false}" \
  "$@"
