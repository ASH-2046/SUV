#!/usr/bin/env bash
# Precompute SUV text embeddings for official NAVSIM v2 PDM evaluation.
# This follows the official SceneLoader used by run_pdm_score.py /
# run_pdm_score_one_stage.py, so
# navhard_two_stage includes reactive synthetic stage-2 scenes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${NAVSIM_V2_DEVKIT_ROOT:-${REPO_ROOT}/experiments/navsimv2/vendor}}"
if [[ -n "${NAVSIM_DEVKIT_ROOT}" && -d "${NAVSIM_DEVKIT_ROOT}/navsim" ]]; then
  export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
fi

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-${REPO_ROOT}/checkpoints}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${CKPT_LOCAL_DIR}}"
export SUV_MODEL_ID="${SUV_MODEL_ID:-Wan2.2-TI2V-5B}"
export SUV_TOKENIZER_MODEL_ID="${SUV_TOKENIZER_MODEL_ID:-Wan2.2-TI2V-5B}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-}"
if [[ -z "${TRAIN_TEST_SPLIT}" ]]; then
  echo "ERROR: set TRAIN_TEST_SPLIT explicitly (navtest or navhard_two_stage)." >&2
  exit 1
fi
case "${TRAIN_TEST_SPLIT}" in
  navtest|navhard_two_stage) ;;
  *)
    echo "ERROR: TRAIN_TEST_SPLIT must be navtest or navhard_two_stage, got: ${TRAIN_TEST_SPLIT}" >&2
    exit 1
    ;;
esac

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.2}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${REPO_ROOT}/data/navsim/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${REPO_ROOT}/data/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/runs}"

NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
ORIGINAL_SENSOR_PATH="${ORIGINAL_SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles}"
TEXT_EMBEDDING_CACHE_DIR="${SUV_NAVSIM_V2_TEXT_EMBEDDING_CACHE_DIR:-${CKPT_LOCAL_DIR}/suv_text_embeds_cache_test/navsim_v2}"

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
SUV_PROMPT_HISTORY_SECONDS="${SUV_PROMPT_HISTORY_SECONDS:-2.5}"
SUV_PROMPT_VELOCITY_QUANTIZATION="${SUV_PROMPT_VELOCITY_QUANTIZATION:-0.5}"
SUV_PROMPT_ACCELERATION_QUANTIZATION="${SUV_PROMPT_ACCELERATION_QUANTIZATION:-0.5}"
SUV_ACTION_CONTEXT_MODE="${SUV_ACTION_CONTEXT_MODE:-base}"
SUV_ACTION_PROMPT="${SUV_ACTION_PROMPT:-null}"
SUV_ACTION_PROMPT_FUTURE_INSTRUCTION="${SUV_ACTION_PROMPT_FUTURE_INSTRUCTION:-null}"
SUV_ACTION_PROMPT_QUALITY_INSTRUCTION="${SUV_ACTION_PROMPT_QUALITY_INSTRUCTION:-null}"
SUV_SLOT_INFERENCE="${SUV_SLOT_INFERENCE:-false}"

CONTEXT_LEN="${CONTEXT_LEN:-512}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OVERWRITE="${OVERWRITE:-false}"
DRY_RUN="${DRY_RUN:-false}"
VALIDATE_ONLY="${VALIDATE_ONLY:-false}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "Precomputing SUV NAVSIM v2 PDM text embeddings"
python -c "import navsim, pathlib; print('  importing navsim:', pathlib.Path(navsim.__file__).parent)"
echo "  split:           ${TRAIN_TEST_SPLIT}"
echo "  logs:            ${NAVSIM_LOG_PATH}"
echo "  original sensor: ${ORIGINAL_SENSOR_PATH}"
if [[ "${TRAIN_TEST_SPLIT}" == "navhard_two_stage" ]]; then
  echo "  synthetic sensor:${SYNTHETIC_SENSOR_PATH}"
  echo "  synthetic scenes:${SYNTHETIC_SCENES_PATH}"
fi
echo "  text embeds:     ${TEXT_EMBEDDING_CACHE_DIR}"
echo "  history/future:  ${SUV_PDM_HISTORY_FRAMES}/${SUV_NUM_FUTURE_FRAMES}"
echo "  prompt:          mode=${SUV_PROMPT_MODE}, action_context=${SUV_ACTION_CONTEXT_MODE}, slot_inference=${SUV_SLOT_INFERENCE}"
echo "  wan ckpts:       base=${DIFFSYNTH_MODEL_BASE_PATH}, model=${SUV_MODEL_ID}, tokenizer=${SUV_TOKENIZER_MODEL_ID}"
echo "  overwrite:       ${OVERWRITE}, dry_run=${DRY_RUN}, validate_only=${VALIDATE_ONLY}"
echo "  cuda:            ${CUDA_VISIBLE_DEVICES}"

ARGS=(
  "--train-test-split" "${TRAIN_TEST_SPLIT}"
  "--navsim-log-path" "${NAVSIM_LOG_PATH}"
  "--original-sensor-path" "${ORIGINAL_SENSOR_PATH}"
  "--text-embedding-cache-dir" "${TEXT_EMBEDDING_CACHE_DIR}"
  "--num-history-frames" "${SUV_PDM_HISTORY_FRAMES}"
  "--num-future-frames" "${SUV_NUM_FUTURE_FRAMES}"
  "--frame-interval" "${SUV_FRAME_INTERVAL}"
  "--fps" "${SUV_FPS}"
  "--prompt" "${SUV_PROMPT}"
  "--prompt-mode" "${SUV_PROMPT_MODE}"
  "--prompt-future-instruction" "${SUV_PROMPT_FUTURE_INSTRUCTION}"
  "--prompt-quality-instruction" "${SUV_PROMPT_QUALITY_INSTRUCTION}"
  "--prompt-history-seconds" "${SUV_PROMPT_HISTORY_SECONDS}"
  "--prompt-velocity-quantization" "${SUV_PROMPT_VELOCITY_QUANTIZATION}"
  "--prompt-acceleration-quantization" "${SUV_PROMPT_ACCELERATION_QUANTIZATION}"
  "--action-context-mode" "${SUV_ACTION_CONTEXT_MODE}"
  "--action-prompt" "${SUV_ACTION_PROMPT}"
  "--action-prompt-future-instruction" "${SUV_ACTION_PROMPT_FUTURE_INSTRUCTION}"
  "--action-prompt-quality-instruction" "${SUV_ACTION_PROMPT_QUALITY_INSTRUCTION}"
  "--slot-inference" "${SUV_SLOT_INFERENCE}"
  "--context-len" "${CONTEXT_LEN}"
  "--model-id" "${SUV_MODEL_ID}"
  "--tokenizer-model-id" "${SUV_TOKENIZER_MODEL_ID}"
  "--batch-size" "${BATCH_SIZE}"
  "--overwrite" "${OVERWRITE}"
)

if [[ "${TRAIN_TEST_SPLIT}" == "navhard_two_stage" ]]; then
  ARGS+=(
    "--synthetic-sensor-path" "${SYNTHETIC_SENSOR_PATH}"
    "--synthetic-scenes-path" "${SYNTHETIC_SCENES_PATH}"
  )
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  ARGS+=("--dry-run")
fi
if [[ "${VALIDATE_ONLY}" == "true" ]]; then
  ARGS+=("--validate-only")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u experiments/navsimv2/precompute_pdm_text_embeds.py "${ARGS[@]}" "$@"
