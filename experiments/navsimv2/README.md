# SUV on NAVSIM v2

This directory contains the NAVSIM v2 inference adapter, a vendored evaluation
devkit, text-embedding preparation, and official EPDMS evaluation paths.

## Required paths

Set `CKPT_LOCAL_DIR="/path/to/checkpoints"` once in each evaluation script.
The SUV checkpoint and Wan components live under this root. Set the shared
`TEXT_EMBEDDING_CACHE_DIR` separately for the navtest and navhard embeddings.

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim
export NUPLAN_MAPS_ROOT=/path/to/navsim/maps
export NAVSIM_EXP_ROOT=/path/to/evaluation_outputs
```

Embedding preparation and scoring default to `CUDA_VISIBLE_DEVICES=0,1,2,3`
and shard work across all listed GPUs.

## Standard navtest evaluation

Build the shared text embedding cache once. It contains prompts for both
`navtest` and `navhard_two_stage`, including all four visual slots:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash experiments/navsimv2/scripts/evaluation/precompute_text_embeddings.sh

METRIC_CACHE_PATH=/path/to/navsim_v2/metric_cache \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash experiments/navsimv2/scripts/evaluation/run_epdm_score.sh
```

## NavHard two-stage evaluation

Required data paths are:

```bash
export SYNTHETIC_SENSOR_PATH=/path/to/navhard_two_stage/sensor_blobs
export SYNTHETIC_SCENES_PATH=/path/to/navhard_two_stage/synthetic_scene_pickles
export METRIC_CACHE_PATH=/path/to/navsim_v2/navhard_metric_cache
```

Create the metric cache when needed, then score. The shared cache prepared
above already contains the `navhard_two_stage` embeddings:

```bash
bash experiments/navsimv2/scripts/run_navhard_metric_caching.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash experiments/navsimv2/scripts/evaluation/run_navhard_epdm_score.sh
```
