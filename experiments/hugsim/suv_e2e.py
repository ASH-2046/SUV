"""SUV client for the HUGSIM closed-loop benchmark.

HUGSIM writes ``(observation, info)`` to ``obs_pipe`` and reads a lidar-frame
waypoint array from ``plan_pipe``. This client keeps a 2 Hz front-camera
history, runs the released NAVSIM SUV agent, and returns waypoints in the same
lidar frame as the official LTF client.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

def _find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "experiments" / "navsimv1" / "pdm_agent.py").is_file():
            return parent
    return path.parent


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# OpenCV camera axes (x right, y down, z forward) to the ego IMU used by
# HUGSIM's LTF client (x forward, y left, z up).
OPENCV_TO_IMU = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
COMMAND_LABELS = ("turn left", "go straight", "turn right", "follow the lane")


def history_indices(length: int, count: int, stride: int) -> list[int]:
    """Oldest-to-newest indices sampled every ``stride`` steps, clamped at 0."""
    if length < 1:
        raise ValueError("history buffer is empty")
    if count < 1 or stride < 1:
        raise ValueError(f"count and stride must be positive, got {count}, {stride}")
    last = length - 1
    return [max(0, last - stride * (count - 1 - i)) for i in range(count)]


def imu_pose(info: dict) -> np.ndarray:
    """Current ego SE2 pose in the IMU frame: x forward, y left, yaw."""
    translation = OPENCV_TO_IMU @ np.asarray(info["ego_pos"], dtype=np.float64)
    yaw = -float(np.asarray(info["ego_rot"], dtype=np.float64).reshape(-1)[1])
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def relative_se2(origin: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Express ``poses`` in the frame of ``origin``. Both are [x, y, yaw]."""
    origin = np.asarray(origin, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    dx = poses[:, 0] - origin[0]
    dy = poses[:, 1] - origin[1]
    cos_yaw = np.cos(origin[2])
    sin_yaw = np.sin(origin[2])
    rel_x = cos_yaw * dx + sin_yaw * dy
    rel_y = -sin_yaw * dx + cos_yaw * dy
    rel_yaw = (poses[:, 2] - origin[2] + np.pi) % (2.0 * np.pi) - np.pi
    return np.stack([rel_x, rel_y, rel_yaw], axis=-1).astype(np.float32)


def driving_command(info: dict) -> np.ndarray:
    """NAVSIM one-hot command. HUGSIM's route label is used when it is 0..3."""
    command = np.zeros(4, dtype=np.float32)
    raw = info.get("command", 1)
    if isinstance(raw, (list, tuple)) or (isinstance(raw, np.ndarray) and raw.shape != ()):
        values = np.asarray(raw, dtype=np.float32).reshape(-1)
        if values.shape[0] == 4:
            command[:] = values
            return command
        raw = values[0] if values.shape[0] else 1
    try:
        index = int(raw)
    except (TypeError, ValueError):
        index = 1
    if not 0 <= index < 4:
        index = 1
    command[index] = 1.0
    return command


def body_velocity(info: dict) -> np.ndarray:
    """Bicycle-model velocity in the current ego frame, [forward, left] m/s."""
    return np.array([float(info.get("ego_velo", 0.0)), 0.0], dtype=np.float32)


def body_acceleration(info: dict) -> np.ndarray:
    """Longitudinal acceleration command in the current ego frame, m/s^2."""
    return np.array([float(info.get("accelerate", 0.0)), 0.0], dtype=np.float32)


def front_image(obs: dict) -> np.ndarray:
    rgb = obs.get("rgb", obs)
    if "CAM_FRONT" not in rgb:
        raise KeyError(f"HUGSIM observation has no CAM_FRONT. Keys: {sorted(rgb)}")
    image = np.asarray(rgb["CAM_FRONT"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"CAM_FRONT must be HxWx3, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def imu_waypoints_to_lidar(poses: np.ndarray) -> np.ndarray:
    """NAVSIM ego waypoints to HUGSIM lidar xy. x is right, y is forward."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 2:
        raise ValueError(f"expected waypoints [N, >=2], got {poses.shape}")
    lidar = np.stack([-poses[:, 1], poses[:, 0]], axis=-1)
    return lidar.astype(np.float32)


class HistoryBuffer:
    """Front images and ego states sampled at the SUV training rate."""

    def __init__(self, num_frames: int, stride: int):
        self.num_frames = int(num_frames)
        self.stride = int(stride)
        self.images: list[np.ndarray] = []
        self.poses: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.accelerations: list[np.ndarray] = []
        self.commands: list[np.ndarray] = []

    def append(self, obs: dict, info: dict) -> None:
        self.images.append(front_image(obs))
        self.poses.append(imu_pose(info))
        self.velocities.append(body_velocity(info))
        self.accelerations.append(body_acceleration(info))
        self.commands.append(driving_command(info))

    def agent_input(self):
        indices = history_indices(len(self.images), self.num_frames, self.stride)
        origin = self.poses[indices[-1]]
        relative = relative_se2(origin, np.stack([self.poses[i] for i in indices], axis=0))
        cameras = []
        ego_statuses = []
        for row, index in enumerate(indices):
            cameras.append(
                SimpleNamespace(
                    cam_f0=SimpleNamespace(
                        image=self.images[index],
                        sensor2lidar_rotation=None,
                        sensor2lidar_translation=None,
                        intrinsics=None,
                    )
                )
            )
            ego_statuses.append(
                SimpleNamespace(
                    ego_pose=relative[row],
                    ego_velocity=self.velocities[index],
                    ego_acceleration=self.accelerations[index],
                    driving_command=self.commands[index],
                )
            )
        return SimpleNamespace(cameras=cameras, ego_statuses=ego_statuses)


def _self_test() -> None:
    indices = history_indices(1, 4, 2)
    assert indices == [0, 0, 0, 0], indices
    indices = history_indices(5, 4, 2)
    assert indices == [0, 0, 2, 4], indices

    info = {
        "ego_pos": [0.0, 1.5, 0.0],
        "ego_rot": [0.0, 0.0, 0.0],
        "ego_velo": 2.0,
        "accelerate": 0.5,
        "command": 1,
    }
    pose = imu_pose(info)
    assert np.allclose(pose, [0.0, 0.0, 0.0]), pose
    assert np.allclose(body_velocity(info), [2.0, 0.0])
    assert driving_command({"command": 0})[0] == 1.0
    assert driving_command({"command": "straight"})[1] == 1.0

    # theta = 0 moves the HUGSIM ego toward +z in OpenCV, which is +x in IMU.
    moved = dict(info)
    moved["ego_pos"] = [0.0, 1.5, 4.0]
    moved_pose = imu_pose(moved)
    relative = relative_se2(moved_pose, np.stack([pose, moved_pose], axis=0))
    assert np.allclose(relative[-1], 0.0, atol=1e-6), relative
    assert np.allclose(relative[0], [-4.0, 0.0, 0.0], atol=1e-5), relative

    lidar = imu_waypoints_to_lidar(np.array([[4.0, 1.0, 0.1]], dtype=np.float32))
    assert np.allclose(lidar, [[-1.0, 4.0]]), lidar
    print("hugsim client self-test passed")


def _build_agent(args: argparse.Namespace):
    from omegaconf import OmegaConf

    from experiments.navsimv1.pdm_agent import SUVNavsimV1Agent

    class _HugsimAgent(SUVNavsimV1Agent):
        def initialize(self) -> None:
            if self.model is not None:
                return
            cfg = OmegaConf.load(self.model_config_path)
            if not bool(cfg.load_text_encoder):
                cfg.load_text_encoder = True
                dumped = Path(self.text_embedding_cache_dir).expanduser() / "_hugsim_model.yaml"
                dumped.parent.mkdir(parents=True, exist_ok=True)
                OmegaConf.save(cfg, dumped)
                self.model_config_path = str(dumped)
            super().initialize()
            encoder = getattr(self.model, "text_encoder", None)
            if encoder is not None:
                encoder.to("cpu")

        def _get_cached_text_context(self, prompt: str):
            try:
                return super()._get_cached_text_context(prompt)
            except FileNotFoundError:
                return self._encode_and_cache_prompt(prompt)

        def _encode_and_cache_prompt(self, prompt: str):
            import hashlib

            import torch

            encoder = getattr(self.model, "text_encoder", None)
            if encoder is None:
                raise RuntimeError(
                    "Closed-loop prompt is not in the text-embedding cache and "
                    "the Wan text encoder was not loaded."
                )
            encoder.to(self.model.device)
            embedding, _mask = self.model.encode_prompt(prompt)
            encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            context = embedding[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            mask = torch.ones(context.shape[0], dtype=torch.bool)
            cache_dir = Path(self.text_embedding_cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache_path = cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
            torch.save({"context": context, "mask": mask}, cache_path)
            return context, mask

    agent = _HugsimAgent(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        text_embedding_cache_dir=args.text_cache,
        num_inference_steps=args.steps,
        visual_conditioning="history_4",
        device="cuda",
        seed=args.seed,
    )
    agent.initialize()
    return agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SUV HUGSIM closed-loop client")
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH", ""))
    parser.add_argument(
        "--model-config",
        default=os.environ.get(
            "SUV_NAVSIM_MODEL_CONFIG",
            str(REPO_ROOT / "experiments" / "navsimv1" / "config" / "model" / "suv_navsim.yaml"),
        ),
    )
    parser.add_argument(
        "--text-cache",
        default=os.environ.get("TEXT_EMBEDDING_CACHE_DIR", ""),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("SUV_NUM_INFERENCE_STEPS", "10")),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SUV_SEED", "42")))
    parser.add_argument(
        "--history-stride",
        type=int,
        default=int(os.environ.get("SUV_HISTORY_STRIDE", "2")),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.output:
        raise SystemExit("--output is required")
    if not args.checkpoint:
        raise SystemExit("Set CHECKPOINT_PATH to the SUV checkpoint.")
    if not args.text_cache:
        raise SystemExit("Set TEXT_EMBEDDING_CACHE_DIR for prompt embeddings.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    obs_pipe = output / "obs_pipe"
    plan_pipe = output / "plan_pipe"
    if not obs_pipe.exists():
        os.mkfifo(obs_pipe)
    if not plan_pipe.exists():
        os.mkfifo(plan_pipe)

    agent = _build_agent(args)
    history = HistoryBuffer(agent.num_history_frames, args.history_stride)
    print("Ready for receiving", flush=True)
    step = 0
    while True:
        with open(obs_pipe, "rb") as pipe:
            raw = pickle.loads(pipe.read())
        print("received", flush=True)
        if raw == "Done":
            return
        obs, info = raw
        history.append(obs, info)
        try:
            trajectory = agent.compute_trajectory(history.agent_input())
            waypoints = imu_waypoints_to_lidar(trajectory.poses)
        except Exception as exc:
            print(f"SUV inference failed: {exc}", flush=True)
            waypoints = None
        with open(plan_pipe, "wb") as pipe:
            pipe.write(pickle.dumps(waypoints))
        if waypoints is None:
            print("sent empty plan", flush=True)
            return
        print(
            f"sent step={step} speed={float(info.get('ego_velo', 0.0)):.2f} "
            f"command={COMMAND_LABELS[int(np.argmax(history.commands[-1]))]} "
            f"waypoints={waypoints.shape[0]} first={waypoints[0].tolist()}",
            flush=True,
        )
        step += 1


if __name__ == "__main__":
    main()
