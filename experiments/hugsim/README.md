# HUGSIM closed-loop client

SUV is the planning client. HUGSIM stays in its own environment and exchanges
observations and waypoints through the existing `obs_pipe` / `plan_pipe` pair.
The client uses only `CAM_FRONT`, keeps four history frames at 2 Hz (every
second HUGSIM step when `dt` is 0.25 s), and returns lidar-frame waypoints at
0.5 s spacing, which matches HUGSIM's LQR discretization.

## Wire it into HUGSIM

In `closed_loop.py`, next to the `ltf` branch:

```python
elif args.ad == 'suv':
    ad_path = cfg.base.suv_path
```

In `configs/sim/*_base.yaml`:

```yaml
suv_path: /path/to/SUV/experiments/hugsim/suv_e2e.sh
```

Launch one scenario with the simulator on GPU 0 and SUV on GPU 1:

```bash
export CKPT_LOCAL_DIR=/path/to/checkpoints
export NAVSIM_DEVKIT_ROOT=/path/to/navsim
export SUV_CONDA_SH=/path/to/conda.sh
export SUV_CONDA_ENV=suv

CUDA_VISIBLE_DEVICES=0 python closed_loop.py \
  --scenario_path ./configs/benchmark/nuscenes/scene-0013-easy-00.yaml \
  --base_path ./configs/sim/nuscenes_base.yaml \
  --camera_path ./configs/sim/nuscenes_camera.yaml \
  --kinematic_path ./configs/sim/kinematic.yaml \
  --ad suv \
  --ad_cuda 1
```

`CKPT_LOCAL_DIR` must contain `suv_navsim.pt` and `Wan2.2-TI2V-5B/`. Prompts
that are missing from `TEXT_EMBEDDING_CACHE_DIR` are encoded once with the Wan
text encoder and then reused. HD-Score is written by HUGSIM to `eval.json`.

Check the coordinate conversion without loading the model:

```bash
python experiments/hugsim/suv_e2e.py --self-test
```
