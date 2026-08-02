# SUV on NAVSIM v1

This directory contains only the NAVSIM v1 inference adapter, text-embedding
preparation, and official PDM evaluation entrypoints.

## Required paths

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/navsim_v1
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan/checkpoints
export OPENSCENE_DATA_ROOT=/path/to/navsim
export NUPLAN_MAPS_ROOT=/path/to/navsim/maps
export METRIC_CACHE_PATH=/path/to/navsim_v1/metric_cache
export NAVSIM_EXP_ROOT=/path/to/evaluation_outputs
```

The default layout is:

```text
${OPENSCENE_DATA_ROOT}/navsim_logs/test
${OPENSCENE_DATA_ROOT}/sensor_blobs/test
${NUPLAN_MAPS_ROOT}/
```

## Evaluate

First cache the dynamic-prompt embeddings used by the evaluation agent:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv1/scripts/evaluation/precompute_suv_navsimv1_pdm_text_embeds.sh
```

Then run PDM scoring:

```bash
CHECKPOINT_PATH=/path/to/suv_navsim.pt \
CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv1/scripts/evaluation/run_pdm_score.sh
```

Set `CUDA_VISIBLE_DEVICES=0,1,...` to shard logs across GPUs. Common overrides
include `NAVSIM_LOG_PATH`, `SENSOR_BLOBS_PATH`, `TEXT_EMBEDDING_CACHE_DIR`,
`OUTPUT_DIR`, and `EVAL_NUM_INFERENCE_STEPS`.
