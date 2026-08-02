# SUV on NAVSIM v1

This directory contains only the NAVSIM v1 inference adapter, text-embedding
preparation, and official PDM evaluation entrypoints.

## Required paths

Set `CKPT_LOCAL_DIR="/path/to/checkpoints"` once in each evaluation script.
The SUV checkpoint and Wan components live under this root. Set
`TEXT_EMBEDDING_CACHE_DIR` separately for the generated text embeddings.

```bash
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
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash experiments/navsimv1/scripts/evaluation/precompute_suv_navsimv1_pdm_text_embeds.sh
```

The script caches the RGB, depth, semantic-segmentation, and instance-track
prompts used by SUV's slot inference path.

Then run PDM scoring:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash experiments/navsimv1/scripts/evaluation/run_pdm_score.sh
```

Both preparation and scoring default to four-GPU sharding. Set
`CUDA_VISIBLE_DEVICES` to another comma-separated GPU list when needed. All
required paths are collected in the user-configuration block at the top of the
script.
