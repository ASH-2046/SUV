# SUV on NAVSIM v2

This directory contains the NAVSIM v2 inference adapter, a vendored evaluation
devkit, text-embedding preparation, and official PDM scoring paths.

## Required paths

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan/checkpoints
export OPENSCENE_DATA_ROOT=/path/to/navsim
export NUPLAN_MAPS_ROOT=/path/to/navsim/maps
export NAVSIM_EXP_ROOT=/path/to/evaluation_outputs
```

Set `NAVSIM_DEVKIT_ROOT` only when using another compatible NAVSIM v2 checkout.

## Standard navtest evaluation

```bash
TRAIN_TEST_SPLIT=navtest \
bash experiments/navsimv2/scripts/evaluation/precompute_suv_navsimv2_pdm_text_embeds.sh

CHECKPOINT_PATH=/path/to/suv_navsim.pt \
TRAIN_TEST_SPLIT=navtest \
METRIC_CACHE_PATH=/path/to/navsim_v2/metric_cache \
CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv2/scripts/evaluation/run_suv_navsimv2_pdm_score.sh
```

The scorer automatically uses the multi-GPU launcher when more than one GPU ID
is supplied.

## NavHard two-stage evaluation

Required data paths are:

```bash
export SYNTHETIC_SENSOR_PATH=/path/to/navhard_two_stage/sensor_blobs
export SYNTHETIC_SCENES_PATH=/path/to/navhard_two_stage/synthetic_scene_pickles
export METRIC_CACHE_PATH=/path/to/navsim_v2/navhard_metric_cache
```

Create the metric cache when needed, then prepare embeddings and score:

```bash
bash experiments/navsimv2/scripts/run_navhard_metric_caching.sh

TRAIN_TEST_SPLIT=navhard_two_stage \
bash experiments/navsimv2/scripts/evaluation/precompute_suv_navsimv2_pdm_text_embeds.sh

CHECKPOINT_PATH=/path/to/suv_navsim.pt \
TRAIN_TEST_SPLIT=navhard_two_stage \
bash experiments/navsimv2/scripts/evaluation/run_suv_navsimv2_pdm_score.sh
```
