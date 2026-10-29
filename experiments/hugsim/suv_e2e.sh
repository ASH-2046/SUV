#!/bin/zsh
# HUGSIM launches this as: zsh suv_e2e.sh ${CUDA_ID} ${output_dir}
# Point configs/sim/*_base.yaml `suv_path` at this file and pass `--ad suv`
# after adding the `suv` branch in HUGSIM closed_loop.py.
set -eu

CUDA_ID="${1:?cuda id}"
OUTPUT="${2:?output dir}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)

# Edit these, or export them before launching closed_loop.py.
CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-/path/to/checkpoints}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CKPT_LOCAL_DIR}/suv_navsim.pt}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${CKPT_LOCAL_DIR}/suv_text_embeds_cache/hugsim}"
NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-}"
SUV_NUM_INFERENCE_STEPS="${SUV_NUM_INFERENCE_STEPS:-10}"
SUV_HISTORY_STRIDE="${SUV_HISTORY_STRIDE:-2}"

if [ -n "${SUV_CONDA_ENV:-}" ]; then
  if [ -z "${SUV_CONDA_SH:-}" ]; then
    echo "SUV_CONDA_SH must point at conda.sh when SUV_CONDA_ENV is set." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  . "${SUV_CONDA_SH}"
  conda activate "${SUV_CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export CKPT_LOCAL_DIR CHECKPOINT_PATH TEXT_EMBEDDING_CACHE_DIR
export SUV_NUM_INFERENCE_STEPS SUV_HISTORY_STRIDE
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${CKPT_LOCAL_DIR}}"
export SUV_MODEL_ID="${SUV_MODEL_ID:-Wan2.2-TI2V-5B}"
export SUV_TOKENIZER_MODEL_ID="${SUV_TOKENIZER_MODEL_ID:-Wan2.2-TI2V-5B}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export PYTHONUNBUFFERED=1

INCOMING_PYTHONPATH="${PYTHONPATH:-}"
PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
if [ -n "${NAVSIM_DEVKIT_ROOT}" ]; then
  PYTHONPATH="${PYTHONPATH}:${NAVSIM_DEVKIT_ROOT}"
fi
if [ -n "${INCOMING_PYTHONPATH}" ]; then
  PYTHONPATH="${PYTHONPATH}:${INCOMING_PYTHONPATH}"
fi
export PYTHONPATH

cd "${REPO_ROOT}"
exec python "${REPO_ROOT}/experiments/hugsim/suv_e2e.py" --output "${OUTPUT}"
