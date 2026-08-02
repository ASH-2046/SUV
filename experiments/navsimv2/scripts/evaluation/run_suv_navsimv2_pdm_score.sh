#!/usr/bin/env bash
# Evaluate SUV NAVSIM v2 with the OFFICIAL NAVSIM PDM scorers.
# Produces the official EPDMS:
#   navtest            -> single-stage open-loop EPDMS
#   navhard_two_stage  -> two-stage reactive (IDM) pseudo-closed-loop EC
#
# Usage: TRAIN_TEST_SPLIT=navhard_two_stage CHECKPOINT_PATH=... \
#        bash experiments/navsimv2/scripts/evaluation/run_suv_navsimv2_pdm_score.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Point at the intended navsim devkit. find_spec can grab the WRONG navsim
# (e.g. an unrelated navsim_v1 already importable), so prefer an explicit devkit
# and put it FIRST on PYTHONPATH so `import navsim` resolves to it.
NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${NAVSIM_V2_DEVKIT_ROOT:-${REPO_ROOT}/experiments/navsimv2/vendor}}"
if [[ -n "${NAVSIM_DEVKIT_ROOT}" && -d "${NAVSIM_DEVKIT_ROOT}/navsim" ]]; then
  export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
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
SUV_NAVSIM_MODEL_CONFIG="${SUV_NAVSIM_MODEL_CONFIG:-${REPO_ROOT}/experiments/navsimv2/config/model/suv_navsim.yaml}"

if [[ $# -gt 0 && "$1" != *=* && "$1" != +* ]]; then
  echo "ERROR: positional args are disabled. Set CHECKPOINT_PATH as an env var." >&2
  exit 1
fi

# ==================== User config ====================
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
# navtest and navhard_two_stage are different official NAVSIM v2 leaderboards; require an explicit choice.
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-}"
if [[ -z "${TRAIN_TEST_SPLIT}" ]]; then
  echo "ERROR: set TRAIN_TEST_SPLIT explicitly (navtest = single-stage EPDMS, navhard_two_stage = two-stage reactive EC)." >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: CHECKPOINT_PATH is empty." >&2
  exit 1
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-suv_navsimv2_pdm_score}"
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

case "${TRAIN_TEST_SPLIT}" in
  navtest|navhard_two_stage) ;;
  *)
    echo "ERROR: TRAIN_TEST_SPLIT must be navtest or navhard_two_stage, got: ${TRAIN_TEST_SPLIT}" >&2
    exit 1
    ;;
esac

resolve_navsim_script() {
  local script_name="$1"
  if [[ -n "${NAVSIM_DEVKIT_ROOT}" && -f "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/${script_name}" ]]; then
    printf '%s\n' "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/${script_name}"
    return
  fi
  python - "${script_name}" <<'PY'
import importlib.util
import sys
from pathlib import Path

script_name = sys.argv[1]
spec = importlib.util.find_spec("navsim")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(1)
print(Path(next(iter(spec.submodule_search_locations))) / "planning" / "script" / script_name)
PY
}

NAVSIM_RUN_PDM_SCORE="$(resolve_navsim_script run_pdm_score.py || true)"
NAVSIM_RUN_PDM_SCORE_ONE_STAGE="$(resolve_navsim_script run_pdm_score_one_stage.py || true)"
if [[ "${TRAIN_TEST_SPLIT}" == "navtest" && ! -f "${NAVSIM_RUN_PDM_SCORE_ONE_STAGE}" ]]; then
  echo "ERROR: navsim run_pdm_score_one_stage.py not found; set NAVSIM_DEVKIT_ROOT to the official NAVSIM devkit." >&2
  exit 1
fi
if [[ "${TRAIN_TEST_SPLIT}" == "navhard_two_stage" && ! -f "${NAVSIM_RUN_PDM_SCORE}" ]]; then
  echo "ERROR: navsim run_pdm_score.py not found; set NAVSIM_DEVKIT_ROOT to the official NAVSIM devkit." >&2
  exit 1
fi
echo "Using navsim two-stage scorer: ${NAVSIM_RUN_PDM_SCORE}"
echo "Using navsim one-stage scorer: ${NAVSIM_RUN_PDM_SCORE_ONE_STAGE:-<missing>}"
python -c "import navsim, pathlib; print('Importing navsim from:', pathlib.Path(navsim.__file__).parent)"

# ==================== Data paths ====================
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.2}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${REPO_ROOT}/data/navsim/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${REPO_ROOT}/data/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/runs}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/Drive-JEPA-cache/metric_cache_v2}"

NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
ORIGINAL_SENSOR_PATH="${ORIGINAL_SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles}"
TEXT_EMBEDDING_CACHE_DIR="${SUV_NAVSIM_V2_TEXT_EMBEDDING_CACHE_DIR:-${CKPT_LOCAL_DIR}/suv_text_embeds_cache_test/navsim_v2}"

# ==================== SUV setting ====================
SUV_VIDEO_HEIGHT="${SUV_VIDEO_HEIGHT:-384}"
SUV_VIDEO_WIDTH="${SUV_VIDEO_WIDTH:-640}"
SUV_CROP_TOP_BOTTOM="${SUV_CROP_TOP_BOTTOM:-28}"
SUV_VISUAL_CONDITIONING="${SUV_VISUAL_CONDITIONING:-current}"
SUV_NUM_FUTURE_FRAMES="${SUV_NUM_FUTURE_FRAMES:-8}"
SUV_FRAME_INTERVAL="${SUV_FRAME_INTERVAL:-1}"
SUV_FPS="${SUV_FPS:-2.0}"
case "${SUV_VISUAL_CONDITIONING}" in
  history_5|history5|5_history) SUV_PDM_HISTORY_FRAMES=5 ;;
  *)                            SUV_PDM_HISTORY_FRAMES=4 ;;
esac
SUV_PROMPT="${SUV_PROMPT:-A high-quality, photorealistic ego-centric driving video captured by a camera rigidly mounted on the ego vehicle, always facing forward.}"
SUV_PROMPT_MODE="${SUV_PROMPT_MODE:-dynamic}"
SUV_PROMPT_FUTURE_INSTRUCTION="${SUV_PROMPT_FUTURE_INSTRUCTION:-null}"
SUV_PROMPT_QUALITY_INSTRUCTION="${SUV_PROMPT_QUALITY_INSTRUCTION:-null}"
SUV_PROMPT_VELOCITY_QUANTIZATION="${SUV_PROMPT_VELOCITY_QUANTIZATION:-0.5}"
SUV_PROMPT_ACCELERATION_QUANTIZATION="${SUV_PROMPT_ACCELERATION_QUANTIZATION:-0.5}"
SUV_ACTION_CONTEXT_MODE="${SUV_ACTION_CONTEXT_MODE:-base}"
SUV_ACTION_PROMPT="${SUV_ACTION_PROMPT:-null}"
SUV_ACTION_PROMPT_FUTURE_INSTRUCTION="${SUV_ACTION_PROMPT_FUTURE_INSTRUCTION:-null}"
SUV_ACTION_PROMPT_QUALITY_INSTRUCTION="${SUV_ACTION_PROMPT_QUALITY_INSTRUCTION:-null}"
# Comma-separated active inference slots, without brackets.
SUV_SLOT_MODALITIES="${SUV_SLOT_MODALITIES:-rgb,depth,seg,instance}"
PRECISION="${PRECISION:-bf16}"
DEVICE="${DEVICE:-cuda}"
RAND_DEVICE="${RAND_DEVICE:-cpu}"
EVAL_NUM_INFERENCE_STEPS="${EVAL_NUM_INFERENCE_STEPS:-10}"
SEED="${SEED:-42}"
TILED="${TILED:-false}"
SUV_EVAL_VISUALIZE="${SUV_EVAL_VISUALIZE:-false}"
SUV_EVAL_VISUALIZATION_DIR="${SUV_EVAL_VISUALIZATION_DIR:-suv_navsimv2_pdm_visualizations}"
SUV_EVAL_VISUALIZATION_MAX_SAMPLES="${SUV_EVAL_VISUALIZATION_MAX_SAMPLES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/$(date +%Y-%m-%d_%H-%M-%S)}"

PDM_OVERRIDES=(
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "train_test_split.scene_filter.num_history_frames=${SUV_PDM_HISTORY_FRAMES}"
  "train_test_split.scene_filter.num_future_frames=${SUV_NUM_FUTURE_FRAMES}"
  "metric_cache_path=${METRIC_CACHE_PATH}"
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "++original_sensor_path=${ORIGINAL_SENSOR_PATH}"
  "output_dir=${OUTPUT_DIR}"
  "experiment_name=${EXPERIMENT_NAME}"
  "worker=single_machine_thread_pool"
  "worker.max_workers=${WORKER_MAX_WORKERS}"
  "worker.use_process_pool=${WORKER_USE_PROCESS_POOL}"
  "agent._target_=experiments.navsimv2.pdm_agent.SUVNavsimV2Agent"
  "++agent.checkpoint_path=${CHECKPOINT_PATH}"
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
  "++agent.action_context_mode=${SUV_ACTION_CONTEXT_MODE}"
  "++agent.action_prompt='${SUV_ACTION_PROMPT}'"
  "++agent.action_prompt_future_instruction=${SUV_ACTION_PROMPT_FUTURE_INSTRUCTION}"
  "++agent.action_prompt_quality_instruction=${SUV_ACTION_PROMPT_QUALITY_INSTRUCTION}"
  "++agent.slot_modalities=[${SUV_SLOT_MODALITIES}]"
  "++agent.mixed_precision=${PRECISION}"
  "++agent.device=${DEVICE}"
  "++agent.rand_device=${RAND_DEVICE}"
  "++agent.num_inference_steps=${EVAL_NUM_INFERENCE_STEPS}"
  "++agent.seed=${SEED}"
  "++agent.tiled=${TILED}"
  "++agent.eval_visualize=${SUV_EVAL_VISUALIZE}"
  "++agent.eval_visualization_dir=${SUV_EVAL_VISUALIZATION_DIR}"
  "++agent.eval_visualization_max_samples=${SUV_EVAL_VISUALIZATION_MAX_SAMPLES}"
)

echo "SUV NAVSIM v2 PDM evaluation"
echo "  checkpoint:      ${CHECKPOINT_PATH}"
echo "  experiment:      ${EXPERIMENT_NAME}"
echo "  output:          ${OUTPUT_DIR}"
echo "  split:           ${TRAIN_TEST_SPLIT}"
echo "  logs:            ${NAVSIM_LOG_PATH}"
echo "  original sensor: ${ORIGINAL_SENSOR_PATH}"
echo "  metric cache:    ${METRIC_CACHE_PATH}"
echo "  maps:            ${NUPLAN_MAPS_ROOT}"
echo "  text embeds:     ${TEXT_EMBEDDING_CACHE_DIR}"
echo "  model config:    ${SUV_NAVSIM_MODEL_CONFIG}"
echo "  slot modalities: ${SUV_SLOT_MODALITIES}"
echo "  worker:          max_workers=${WORKER_MAX_WORKERS}, process_pool=${WORKER_USE_PROCESS_POOL}"
echo "  bev visualize:   ${SUV_EVAL_VISUALIZE} -> ${SUV_EVAL_VISUALIZATION_DIR} (max=${SUV_EVAL_VISUALIZATION_MAX_SAMPLES}; 0=all)"
echo "  cuda:            ${CUDA_VISIBLE_DEVICES}, pdm_num_gpus=${PDM_NUM_GPUS}"

if [[ "${TRAIN_TEST_SPLIT}" == "navhard_two_stage" ]]; then
  NAVHARD_OVERRIDES=(
    "++synthetic_sensor_path=${SYNTHETIC_SENSOR_PATH}"
    "++synthetic_scenes_path=${SYNTHETIC_SCENES_PATH}"
  )
  echo "  synthetic sensor:${SYNTHETIC_SENSOR_PATH}"
  echo "  synthetic scenes:${SYNTHETIC_SCENES_PATH}"
  echo "  traffic policy:  NavsimIDMTrafficAgents (official reactive IDM; not overridden)"
  if [[ "${PDM_NUM_GPUS}" -gt 1 ]]; then
    echo "  scorer:          experiments/navsimv2/run_pdm_score_navhard_multigpu.py"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u experiments/navsimv2/run_pdm_score_navhard_multigpu.py \
      --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
      --num-gpus "${PDM_NUM_GPUS}" \
      --output-dir "${OUTPUT_DIR}" \
      --overrides "${PDM_OVERRIDES[@]}" \
      "${NAVHARD_OVERRIDES[@]}" \
      "$@"
  else
    echo "  scorer:          ${NAVSIM_RUN_PDM_SCORE}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u "${NAVSIM_RUN_PDM_SCORE}" \
      "${PDM_OVERRIDES[@]}" \
      "${NAVHARD_OVERRIDES[@]}" \
      "$@"
  fi
else
  echo "  scorer:          ${NAVSIM_RUN_PDM_SCORE_ONE_STAGE}"
  echo "  synthetic paths: <not passed for one_stage>"
  if [[ "${PDM_NUM_GPUS}" -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u experiments/navsimv2/run_pdm_score_multigpu.py \
      --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
      --num-gpus "${PDM_NUM_GPUS}" \
      --output-dir "${OUTPUT_DIR}" \
      --overrides "${PDM_OVERRIDES[@]}" \
      "$@"
  else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u "${NAVSIM_RUN_PDM_SCORE_ONE_STAGE}" \
      "${PDM_OVERRIDES[@]}" \
      "$@"
  fi
fi
