<div align="center">

# SUV: Future Scene Understanding as Video Generation for End-to-End Driving

<p>
  <a href="#citation"><img src="assets/arxiv_logo.png" alt="arXiv" height="18"> arXiv</a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#model-zoo">🤗 Model Zoo</a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#evaluation">🚗 Evaluation</a>
</p>

</div>

<p align="center">
  <img src="assets/suv_teaser.png" alt="SUV overview" width="82%">
</p>

SUV formulates future scene understanding as video generation for end-to-end
driving. A shared video expert generates future RGB, segmentation, relative
depth, and instance-track streams, while an action expert attends to their
latent tokens to predict the ego trajectory. The model uses one front camera
and does not rely on candidate-trajectory selection.

## Authors

> - [Yibo Yuan](https://scholar.google.com/citations?user=ZimfkbYAAAAJ), [Jiacheng Fu](https://scholar.google.com/citations?user=-xnJpxsAAAAJ), [Jiangtong Zhu](https://scholar.google.com/citations?user=23Sy-mAAAAAJ), [Yi Li](https://scholar.google.com/citations?user=qGsK180AAAAJ), [Jianhua Han](https://scholar.google.com/citations?user=OEPMQEMAAAAJ), [Meng Tian](https://scholar.google.com/citations?user=btzoGAQAAAAJ), Zhuohan Liu, [Zhiwei Xiong](https://scholar.google.com/citations?user=Snl0HPEAAAAJ), [Hang Xu](https://scholar.google.com/citations?user=J_8TX6sAAAAJ), [Jianwu Fang](https://scholar.google.com/citations?user=hr8eDYsAAAAJ), and [Jianru Xue](https://scholar.google.com/citations?user=-qenJysAAAAJ)
> - **Paper:** arXiv link coming soon
> - If you have any questions, please feel free to contact: *Yibo Yuan* ([Yyb_XJTU@stu.xjtu.edu.cn](mailto:Yyb_XJTU@stu.xjtu.edu.cn)).

## TODO

- [x] Release the pretrained checkpoint, inference code, and
  NAVSIM v1/v2 evaluation.
- [ ] Release the training code and configurations.
- [ ] Release the structured-target preparation pipeline for
  segmentation, relative depth, and instance tracks.
- [ ] Release the WOD-E2E training and evaluation pipeline.

## Highlights

- **One video expert, four future streams.** SUV represents future appearance,
  semantic layout, scene geometry, and instance dynamics as native video
  streams without stream-specific visual prediction heads.
- **Future-aware trajectory planning.** Masked joint video-action attention
  lets the action expert read every future stream in latent space while
  preserving directed information flow.
- **Strong NAVSIM-v2 performance.** SUV reaches **91.0 EPDMS** on `navtest` and
  **36.9 EPDMS** on `navhard` with a single front camera.

## Method

<p align="center">
  <img src="assets/suv_framework.png" alt="SUV framework" width="95%">
</p>

SUV initializes its shared video expert from Wan2.2-5B. Given a front-camera
observation history, navigation command, and ego state, it jointly denoises
four future video streams and the ego trajectory over a 4-second horizon. The
action expert accesses the evolving future-stream tokens directly; video
decoding is not required for planning.

## Results

### NAVSIM v2

The following results use the corrected official EPDMS implementation. SUV
uses one front camera and ten solver steps. It does not use top-k trajectory
selection.

#### `navtest`

| Method | Sensors | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DriveVLA-W0 | 1×C | 98.4 | 95.2 | 99.4 | 99.9 | 86.6 | 97.9 | 97.8 | 98.3 | 82.7 | 86.9 |
| EponaV2 | 1×C | 98.5 | 97.4 | 99.5 | 99.9 | 87.9 | 98.1 | 97.7 | 98.2 | 77.4 | 88.9 |
| Metis | 1×C | 98.4 | 97.2 | 99.6 | 99.8 | 87.8 | 97.7 | 97.8 | 98.4 | 88.0 | 89.5 |
| Metis (Top 6) | 1×C | 98.5 | 97.5 | 99.6 | 99.8 | 87.9 | 97.8 | 98.0 | 98.4 | 90.0 | 90.3 |
| **SUV** | **1×C** | 99.1 | 97.8 | 99.7 | 99.8 | 87.8 | 98.7 | 98.1 | 98.4 | 88.3 | **91.0** |

#### `navhard`

All listed methods use one front camera. S1 and S2 are the two official
`navhard` stages; EPDMS is the combined score.

| Method | Stage | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | S. ↑ | EPDMS ↑ |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DriveVLA-W0 | S1 | 96.8 | 83.3 | 99.0 | 99.6 | 84.6 | 95.3 | 96.4 | 97.6 | 78.2 | - | 24.4 |
|  | S2 | 76.8 | 64.3 | 79.9 | 98.3 | 89.2 | 75.0 | 46.8 | 95.8 | 53.1 | - |  |
| EponaV2 | S1 | 97.3 | 90.7 | 99.4 | 100 | 83.3 | 97.3 | 97.3 | 97.6 | 60.9 | - | 36.1 |
|  | S2 | 83.6 | 78.0 | 88.0 | 98.9 | 86.0 | 80.3 | 50.1 | 96.1 | 52.0 | - |  |
| Metis | S1 | 96.6 | 87.8 | 99.0 | 99.3 | 84.5 | 95.6 | 97.8 | 97.8 | 77.8 | 75.8 | 32.2 |
|  | S2 | 79.6 | 73.3 | 84.9 | 97.8 | 85.8 | 76.6 | 47.7 | 95.4 | 75.3 | 41.7 |  |
| **SUV** | S1 | 96.9 | 94.2 | 99.3 | 99.6 | 84.1 | 95.6 | 96.7 | 97.8 | 79.6 | 82.3 | **36.9** |
|  | S2 | 82.7 | 74.2 | 85.9 | 98.4 | 86.0 | 78.9 | 47.2 | 95.9 | 69.1 | 43.9 |  |

## Model Zoo

The same SUV checkpoint supports NAVSIM v1 and NAVSIM v2 evaluation.

| Model | Video backbone | Input | Checkpoint |
| --- | --- | --- | --- |
| SUV | Wan2.2-TI2V-5B | Single front camera | Coming soon |

The public checkpoint URL and checksum will be added before release. Until
then, the commands below use `/path/to/suv_navsim.pt` as a placeholder.

## Installation

```bash
# Install the SUV environment
conda create -n suv python=3.10 -y
conda activate suv
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Install NAVSIM
git clone https://github.com/autonomousvision/navsim.git /path/to/navsim
pip install -e /path/to/navsim
```

## Data Preparation

Download the NAVSIM logs, camera sensor blobs, NuPlan maps, and official metric
caches by following the corresponding NAVSIM instructions. The expected common
paths are:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan/checkpoints
export OPENSCENE_DATA_ROOT=/path/to/navsim
export NUPLAN_MAPS_ROOT=/path/to/navsim/maps
export NAVSIM_EXP_ROOT=/path/to/evaluation_outputs
```

`DIFFSYNTH_MODEL_BASE_PATH` must contain the public Wan2.2-TI2V-5B components.
The SUV checkpoint contains the video/action expert weights, so a separate
ActionDiT checkpoint is not required.

The default dataset layout is:

```text
${OPENSCENE_DATA_ROOT}/
├── navsim_logs/test
├── sensor_blobs/test
└── navhard_two_stage/
    ├── sensor_blobs
    └── synthetic_scene_pickles
```

## Evaluation

Text embeddings depend on the dynamic driving prompts and must be prepared once
before scoring. Set `CUDA_VISIBLE_DEVICES=0,1,...` to shard evaluation logs
across multiple GPUs.

### NAVSIM v1

Point `NAVSIM_DEVKIT_ROOT` to a compatible NAVSIM v1 checkout:

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/navsim_v1

CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv1/scripts/evaluation/precompute_suv_navsimv1_pdm_text_embeds.sh

CHECKPOINT_PATH=/path/to/suv_navsim.pt \
METRIC_CACHE_PATH=/path/to/navsim_v1/metric_cache \
CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv1/scripts/evaluation/run_pdm_score.sh
```

See the [NAVSIM v1 evaluation guide](experiments/navsimv1/README.md) for path
overrides and multi-GPU options.

### NAVSIM v2 `navtest`

NAVSIM v2 uses the vendored evaluation devkit by default:

```bash
TRAIN_TEST_SPLIT=navtest \
CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv2/scripts/evaluation/precompute_suv_navsimv2_pdm_text_embeds.sh

CHECKPOINT_PATH=/path/to/suv_navsim.pt \
TRAIN_TEST_SPLIT=navtest \
METRIC_CACHE_PATH=/path/to/navsim_v2/metric_cache \
CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv2/scripts/evaluation/run_suv_navsimv2_pdm_score.sh
```

### NAVSIM v2 `navhard`

Configure the reactive two-stage assets first:

```bash
export SYNTHETIC_SENSOR_PATH=/path/to/navhard_two_stage/sensor_blobs
export SYNTHETIC_SCENES_PATH=/path/to/navhard_two_stage/synthetic_scene_pickles
export METRIC_CACHE_PATH=/path/to/navsim_v2/navhard_metric_cache
```

Create the metric cache if it is not already available, then prepare prompts
and run the official two-stage scorer:

```bash
bash experiments/navsimv2/scripts/run_navhard_metric_caching.sh

TRAIN_TEST_SPLIT=navhard_two_stage \
CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv2/scripts/evaluation/precompute_suv_navsimv2_pdm_text_embeds.sh

CHECKPOINT_PATH=/path/to/suv_navsim.pt \
TRAIN_TEST_SPLIT=navhard_two_stage \
CUDA_VISIBLE_DEVICES=0 \
bash experiments/navsimv2/scripts/evaluation/run_suv_navsimv2_pdm_score.sh
```

See the [NAVSIM v2 evaluation guide](experiments/navsimv2/README.md) for the
complete path configuration and multi-GPU behavior.

## Repository Structure

```text
src/suv/               SUV model loading and inference
experiments/navsimv1/  NAVSIM v1 adapter and official PDM evaluation
experiments/navsimv2/  NAVSIM v2 adapter, devkit, and official EPDMS evaluation
```

## Citation

The public paper link and BibTeX entry will be added when available.

## Acknowledgements

SUV builds on Wan2.2 and the NAVSIM/NuPlan evaluation ecosystem. We thank the
authors and maintainers of these projects for releasing their code, models,
datasets, and benchmarks. Please follow their licenses and citation requests.

## License

SUV is released under the [MIT License](LICENSE). Vendored third-party files
retain their original license headers and terms.
