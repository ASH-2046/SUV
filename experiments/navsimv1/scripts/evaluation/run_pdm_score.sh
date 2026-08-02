#!/usr/bin/env bash
# Evaluate SUV NAVSIM v1 with the official NAVSIM PDM scorer.
#
# This script delegates scoring to the NAVSIM devkit's run_pdm_score.py and
# instantiates experiments.navsimv1.pdm_agent.SUVNavsimV1Agent.
#
# Usage:
#   CHECKPOINT_PATH=/path/to/checkpoints/weights/step_XXXXXX.pt \
#   METRIC_CACHE_PATH=/path/to/navsim/metric_cache \
#   bash experiments/navsimv1/scripts/evaluation/run_pdm_score.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${NAVSIM_DEVKIT_ROOT:-}" && -d "${NAVSIM_DEVKIT_ROOT}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
fi
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-${REPO_ROOT}/checkpoints}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${CKPT_LOCAL_DIR}}"
export SUV_MODEL_ID="${SUV_MODEL_ID:-Wan2.2-TI2V-5B}"
export SUV_TOKENIZER_MODEL_ID="${SUV_TOKENIZER_MODEL_ID:-Wan2.2-TI2V-5B}"
SUV_NAVSIM_MODEL_CONFIG="${SUV_NAVSIM_MODEL_CONFIG:-${REPO_ROOT}/experiments/navsimv1/config/model/suv_navsim.yaml}"

if [[ $# -gt 0 && "$1" != *=* && "$1" != +* ]]; then
  echo "ERROR: positional checkpoint args are disabled. Set CHECKPOINT_PATH as an env var." >&2
  exit 1
fi

# ==================== User config ====================
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-suv_navsimv1_pdm_score}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORKER_MAX_WORKERS="${WORKER_MAX_WORKERS:-1}"
WORKER_USE_PROCESS_POOL="${WORKER_USE_PROCESS_POOL:-false}"
IFS=',' read -ra PDM_VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
PDM_AUTO_NUM_GPUS=0
for gpu_id in "${PDM_VISIBLE_GPU_LIST[@]}"; do
  if [[ -n "${gpu_id//[[:space:]]/}" ]]; then
    PDM_AUTO_NUM_GPUS=$((PDM_AUTO_NUM_GPUS + 1))
  fi
done
PDM_NUM_GPUS="${PDM_NUM_GPUS:-${PDM_AUTO_NUM_GPUS}}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: CHECKPOINT_PATH is empty." >&2
  exit 1
fi
if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint path not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
NAVSIM_RUN_PDM_SCORE="$(python - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("navsim")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(1)
script = Path(next(iter(spec.submodule_search_locations))) / "planning" / "script" / "run_pdm_score.py"
print(script)
PY
)"
if [[ ! -f "${NAVSIM_RUN_PDM_SCORE}" ]]; then
  echo "ERROR: NAVSIM run_pdm_score.py not found. Set NAVSIM_DEVKIT_ROOT or install navsim in the current env." >&2
  exit 1
fi

# ==================== Data paths ====================
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.2}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${REPO_ROOT}/data/navsim/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${REPO_ROOT}/data/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/runs}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${REPO_ROOT}/data/navsim/metric_cache}"

NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${CKPT_LOCAL_DIR}/suv_text_embeds_cache_test/navsim_v1}"

if [[ ! -d "${NUPLAN_MAPS_ROOT}" ]]; then
  echo "ERROR: NUPLAN_MAPS_ROOT not found: ${NUPLAN_MAPS_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${NAVSIM_LOG_PATH}" ]]; then
  echo "ERROR: NAVSIM_LOG_PATH not found: ${NAVSIM_LOG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${SENSOR_BLOBS_PATH}" ]]; then
  echo "ERROR: SENSOR_BLOBS_PATH not found: ${SENSOR_BLOBS_PATH}" >&2
  exit 1
fi
if [[ ! -d "${METRIC_CACHE_PATH}" ]]; then
  echo "ERROR: METRIC_CACHE_PATH not found: ${METRIC_CACHE_PATH}" >&2
  exit 1
fi
if [[ ! -d "${TEXT_EMBEDDING_CACHE_DIR}" ]]; then
  echo "WARNING: text embedding cache directory does not exist yet: ${TEXT_EMBEDDING_CACHE_DIR}" >&2
fi
if [[ ! -f "${SUV_NAVSIM_MODEL_CONFIG}" ]]; then
  echo "ERROR: SUV_NAVSIM_MODEL_CONFIG does not exist: ${SUV_NAVSIM_MODEL_CONFIG}" >&2
  exit 1
fi

# ==================== SUV setting ====================
SUV_VIDEO_HEIGHT="${SUV_VIDEO_HEIGHT:-384}"
SUV_VIDEO_WIDTH="${SUV_VIDEO_WIDTH:-640}"
SUV_CROP_TOP_BOTTOM="${SUV_CROP_TOP_BOTTOM:-28}"
SUV_VISUAL_CONDITIONING="${SUV_VISUAL_CONDITIONING:-current}"
SUV_NUM_FUTURE_FRAMES="${SUV_NUM_FUTURE_FRAMES:-8}"
SUV_FRAME_INTERVAL="${SUV_FRAME_INTERVAL:-1}"
SUV_FPS="${SUV_FPS:-2.0}"
case "${SUV_VISUAL_CONDITIONING}" in
  current|single|single_frame|current_frame)
    SUV_PDM_HISTORY_FRAMES=4
    ;;
  history_4|history4|4_history)
    SUV_PDM_HISTORY_FRAMES=4
    ;;
  history_5|history5|5_history)
    SUV_PDM_HISTORY_FRAMES=5
    ;;
  *)
    echo "ERROR: Unsupported SUV_VISUAL_CONDITIONING=${SUV_VISUAL_CONDITIONING}" >&2
    echo "  Expected one of: current, history_4, history_5" >&2
    exit 1
    ;;
esac
SUV_PROMPT="${SUV_PROMPT:-A high-quality, photorealistic ego-centric driving video captured by a camera rigidly mounted on the ego vehicle, always facing forward.}"
SUV_PROMPT_MODE="${SUV_PROMPT_MODE:-dynamic}"
SUV_PROMPT_FUTURE_INSTRUCTION="${SUV_PROMPT_FUTURE_INSTRUCTION:-null}"
SUV_PROMPT_QUALITY_INSTRUCTION="${SUV_PROMPT_QUALITY_INSTRUCTION:-null}"
SUV_PROMPT_VELOCITY_QUANTIZATION="${SUV_PROMPT_VELOCITY_QUANTIZATION:-0.5}"
SUV_PROMPT_ACCELERATION_QUANTIZATION="${SUV_PROMPT_ACCELERATION_QUANTIZATION:-0.5}"
PRECISION="${PRECISION:-bf16}"
DEVICE="${DEVICE:-cuda}"
RAND_DEVICE="${RAND_DEVICE:-cpu}"
EVAL_NUM_INFERENCE_STEPS="${EVAL_NUM_INFERENCE_STEPS:-10}"
SEED="${SEED:-42}"
TILED="${TILED:-false}"

SUV_EVAL_VISUALIZE="${SUV_EVAL_VISUALIZE:-false}"
SUV_EVAL_VISUALIZATION_DIR="${SUV_EVAL_VISUALIZATION_DIR:-suv_navsimv1_pdm_visualizations}"
SUV_EVAL_VISUALIZATION_MAX_SAMPLES="${SUV_EVAL_VISUALIZATION_MAX_SAMPLES:-8}"
SUV_EVAL_SCORE_GROUND_TRUTH="${SUV_EVAL_SCORE_GROUND_TRUTH:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/$(date +%Y-%m-%d_%H-%M-%S)}"

echo "SUV NAVSIM v1 PDM evaluation"
echo "  checkpoint:     ${CHECKPOINT_PATH}"
echo "  navsim scorer:  ${NAVSIM_RUN_PDM_SCORE}"
echo "  experiment:     ${EXPERIMENT_NAME}"
echo "  output:         ${OUTPUT_DIR}"
echo "  split:          ${TRAIN_TEST_SPLIT}"
echo "  logs:           ${NAVSIM_LOG_PATH}"
echo "  sensor blobs:   ${SENSOR_BLOBS_PATH}"
echo "  metric cache:   ${METRIC_CACHE_PATH}"
echo "  maps:           ${NUPLAN_MAPS_ROOT}"
echo "  text embeds:    ${TEXT_EMBEDDING_CACHE_DIR}"
echo "  video window:   visual_conditioning=${SUV_VISUAL_CONDITIONING}, future=${SUV_NUM_FUTURE_FRAMES} @ ${SUV_VIDEO_HEIGHT}x${SUV_VIDEO_WIDTH}"
echo "  prompt mode:    ${SUV_PROMPT_MODE}, vq=${SUV_PROMPT_VELOCITY_QUANTIZATION}, aq=${SUV_PROMPT_ACCELERATION_QUANTIZATION}"
echo "  wan ckpts:      base=${DIFFSYNTH_MODEL_BASE_PATH}, model=${SUV_MODEL_ID}, tokenizer=${SUV_TOKENIZER_MODEL_ID}, offline=true"
echo "  model config:   ${SUV_NAVSIM_MODEL_CONFIG}"
echo "  infer:          steps=${EVAL_NUM_INFERENCE_STEPS}, precision=${PRECISION}, device=${DEVICE}, rand_device=${RAND_DEVICE}, seed=${SEED}"
echo "  visualization:  ${SUV_EVAL_VISUALIZE}, dir=${SUV_EVAL_VISUALIZATION_DIR}, samples=${SUV_EVAL_VISUALIZATION_MAX_SAMPLES}"
echo "  GT sanity:      ${SUV_EVAL_SCORE_GROUND_TRUTH}"
echo "  worker:         max_workers=${WORKER_MAX_WORKERS}, process_pool=${WORKER_USE_PROCESS_POOL}"
echo "  cuda:           ${CUDA_VISIBLE_DEVICES}, pdm_num_gpus=${PDM_NUM_GPUS}"

PDM_OVERRIDES=(
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "train_test_split.scene_filter.num_history_frames=${SUV_PDM_HISTORY_FRAMES}"
  "train_test_split.scene_filter.num_future_frames=${SUV_NUM_FUTURE_FRAMES}"
  "metric_cache_path=${METRIC_CACHE_PATH}"
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "output_dir=${OUTPUT_DIR}"
  "experiment_name=${EXPERIMENT_NAME}"
  "worker=single_machine_thread_pool"
  "worker.max_workers=${WORKER_MAX_WORKERS}"
  "worker.use_process_pool=${WORKER_USE_PROCESS_POOL}"
  "agent._target_=experiments.navsimv1.pdm_agent.SUVNavsimV1Agent"
  "agent.checkpoint_path=${CHECKPOINT_PATH}"
  "++agent.model_config_path=${SUV_NAVSIM_MODEL_CONFIG}"
  "++agent.text_embedding_cache_dir=${TEXT_EMBEDDING_CACHE_DIR}"
  "++agent.video_height=${SUV_VIDEO_HEIGHT}"
  "++agent.video_width=${SUV_VIDEO_WIDTH}"
  "++agent.crop_top_bottom=${SUV_CROP_TOP_BOTTOM}"
  "++agent.visual_conditioning=${SUV_VISUAL_CONDITIONING}"
  "++agent.num_future_frames=${SUV_NUM_FUTURE_FRAMES}"
  "++agent.frame_interval=${SUV_FRAME_INTERVAL}"
  "++agent.fps=${SUV_FPS}"
  "++agent.prompt='${SUV_PROMPT}'"
  "++agent.prompt_mode=${SUV_PROMPT_MODE}"
  "++agent.prompt_future_instruction=${SUV_PROMPT_FUTURE_INSTRUCTION}"
  "++agent.prompt_quality_instruction=${SUV_PROMPT_QUALITY_INSTRUCTION}"
  "++agent.prompt_velocity_quantization=${SUV_PROMPT_VELOCITY_QUANTIZATION}"
  "++agent.prompt_acceleration_quantization=${SUV_PROMPT_ACCELERATION_QUANTIZATION}"
  "++agent.mixed_precision=${PRECISION}"
  "++agent.device=${DEVICE}"
  "++agent.rand_device=${RAND_DEVICE}"
  "++agent.num_inference_steps=${EVAL_NUM_INFERENCE_STEPS}"
  "++agent.seed=${SEED}"
  "++agent.tiled=${TILED}"
  "++agent.eval_visualize=${SUV_EVAL_VISUALIZE}"
  "++agent.eval_visualization_dir=${SUV_EVAL_VISUALIZATION_DIR}"
  "++agent.eval_visualization_max_samples=${SUV_EVAL_VISUALIZATION_MAX_SAMPLES}"
  "++agent.eval_score_ground_truth=${SUV_EVAL_SCORE_GROUND_TRUTH}"
)

# GT sanity scoring is implemented in the local runner. Route single-GPU debug
# runs through it as well, while preserving the official single-GPU path by
# default.
if [[ "${PDM_NUM_GPUS}" -gt 1 || "${SUV_EVAL_SCORE_GROUND_TRUTH}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u experiments/navsimv1/run_pdm_score_multigpu.py \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --num-gpus "${PDM_NUM_GPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --overrides "${PDM_OVERRIDES[@]}" \
    "$@"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u "${NAVSIM_RUN_PDM_SCORE}" \
    "${PDM_OVERRIDES[@]}" \
    "$@"
fi
