#!/usr/bin/env bash
# Build the official NAVSIM v2 metric cache required by navhard_two_stage.
#
# This cache is separate from navtest by default because the official
# MetricCacheLoader reads one metadata CSV, so mixing split metadata in the same
# directory can silently hide tokens.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${NAVSIM_V2_DEVKIT_ROOT:-${REPO_ROOT}/experiments/navsimv2/vendor}}"
if [[ -n "${NAVSIM_DEVKIT_ROOT}" && -d "${NAVSIM_DEVKIT_ROOT}/navsim" ]]; then
  export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
fi

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.2}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${REPO_ROOT}/data/navsim/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${REPO_ROOT}/data/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/runs}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"
if [[ "${TRAIN_TEST_SPLIT}" != "navhard_two_stage" ]]; then
  echo "ERROR: this script only caches TRAIN_TEST_SPLIT=navhard_two_stage, got: ${TRAIN_TEST_SPLIT}" >&2
  exit 1
fi

NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/Drive-JEPA-cache/metric_cache_v2_navhard}"
WORKER_MAX_WORKERS="${WORKER_MAX_WORKERS:-32}"
WORKER_USE_PROCESS_POOL="${WORKER_USE_PROCESS_POOL:-false}"
FORCE_FEATURE_COMPUTATION="${FORCE_FEATURE_COMPUTATION:-true}"
CHECK_ONLY="${CHECK_ONLY:-false}"
ALLOW_EXISTING_METADATA="${ALLOW_EXISTING_METADATA:-false}"

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

NAVSIM_RUN_METRIC_CACHING="$(resolve_navsim_script run_metric_caching.py || true)"
if [[ ! -f "${NAVSIM_RUN_METRIC_CACHING}" ]]; then
  echo "ERROR: navsim run_metric_caching.py not found; set NAVSIM_DEVKIT_ROOT to the official NAVSIM devkit." >&2
  exit 1
fi

echo "Official NAVSIM navhard metric caching"
echo "  scorer/devkit:     ${NAVSIM_RUN_METRIC_CACHING}"
python -c "import navsim, pathlib; print('  importing navsim:', pathlib.Path(navsim.__file__).parent)"
echo "  split:             ${TRAIN_TEST_SPLIT}"
echo "  logs:              ${NAVSIM_LOG_PATH}"
echo "  synthetic scenes:  ${SYNTHETIC_SCENES_PATH}"
echo "  metric cache:      ${METRIC_CACHE_PATH}"
echo "  maps:              ${NUPLAN_MAPS_ROOT}"
echo "  worker:            max_workers=${WORKER_MAX_WORKERS}, process_pool=${WORKER_USE_PROCESS_POOL}"
echo "  force recompute:   ${FORCE_FEATURE_COMPUTATION}"

METADATA_DIR="${METRIC_CACHE_PATH}/metadata"
if [[ "${CHECK_ONLY}" != "true" && -d "${METADATA_DIR}" ]]; then
  shopt -s nullglob
  EXISTING_METADATA=("${METADATA_DIR}"/*.csv)
  shopt -u nullglob
  if [[ ${#EXISTING_METADATA[@]} -gt 0 && "${ALLOW_EXISTING_METADATA}" != "true" ]]; then
    echo "ERROR: ${METADATA_DIR} already contains metadata CSV files." >&2
    echo "Use a fresh METRIC_CACHE_PATH, or set ALLOW_EXISTING_METADATA=true if you know the loader should read this directory." >&2
    printf 'Existing metadata:\n' >&2
    printf '  %s\n' "${EXISTING_METADATA[@]}" >&2
    exit 1
  fi
fi

if [[ "${CHECK_ONLY}" != "true" ]]; then
  python -u "${NAVSIM_RUN_METRIC_CACHING}" \
    "train_test_split=${TRAIN_TEST_SPLIT}" \
    "navsim_log_path=${NAVSIM_LOG_PATH}" \
    "metric_cache_path=${METRIC_CACHE_PATH}" \
    "++synthetic_scenes_path=${SYNTHETIC_SCENES_PATH}" \
    "worker=single_machine_thread_pool" \
    "worker.max_workers=${WORKER_MAX_WORKERS}" \
    "worker.use_process_pool=${WORKER_USE_PROCESS_POOL}" \
    "force_feature_computation=${FORCE_FEATURE_COMPUTATION}" \
    "$@"
else
  echo "CHECK_ONLY=true: skipping metric cache generation."
fi

echo "Checking navhard_two_stage metric cache coverage..."
python - "${NAVSIM_LOG_PATH}" "${SYNTHETIC_SCENES_PATH}" "${METRIC_CACHE_PATH}" <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import navsim
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader

navsim_log_path = Path(sys.argv[1])
synthetic_scenes_path = Path(sys.argv[2])
metric_cache_path = Path(sys.argv[3])
config_dir = Path(navsim.__file__).resolve().parent / "planning" / "script" / "config" / "pdm_scoring"

with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(
        config_name="default_run_pdm_score",
        overrides=[
            "train_test_split=navhard_two_stage",
            f"navsim_log_path={navsim_log_path}",
            f"metric_cache_path={metric_cache_path}",
            f"++synthetic_scenes_path={synthetic_scenes_path}",
        ],
    )

scene_loader = SceneLoader(
    synthetic_sensor_path=None,
    original_sensor_path=None,
    data_path=Path(cfg.navsim_log_path),
    synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
    scene_filter=instantiate(cfg.train_test_split.scene_filter),
    sensor_config=SensorConfig.build_no_sensors(),
)
metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

scene_tokens = set(scene_loader.tokens)
cache_tokens = set(metric_cache_loader.tokens)
missing_tokens = sorted(scene_tokens - cache_tokens)
unused_tokens = sorted(cache_tokens - scene_tokens)

print(f"  scene tokens:          {len(scene_tokens)}")
print(f"  stage1 original:       {len(scene_loader.tokens_stage_one)}")
print(f"  stage2 reactive:       {len(scene_loader.reactive_tokens_stage_two)}")
print(f"  cache tokens:          {len(cache_tokens)}")
print(f"  cached scene overlap:  {len(scene_tokens & cache_tokens)}")
print(f"  missing scene tokens:  {len(missing_tokens)}")
print(f"  unused cache tokens:   {len(unused_tokens)}")

if missing_tokens:
    print("  first missing tokens:")
    for token in missing_tokens[:20]:
        print(f"    {token}")
    raise SystemExit(2)
PY

echo "navhard_two_stage metric cache coverage is complete."
