"""Odyssey planner adapter for the released NAVSIM SUV agent.

Odyssey's generic adapter expects ``agent.forward(features)``. SUV plans from a
NAVSIM ``AgentInput`` through ``compute_trajectory``, so this subclass builds
that input and returns ego-frame poses at 0.5 s spacing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from odyssey_bridge.planners.base import NavsimPlanner, narrow_sensor_config

try:
    from .. import ipc_common as ipc
except ImportError:  # launched as a script, not a package
    import ipc_common as ipc  # type: ignore[no-redef]


class SUVPlanner(NavsimPlanner):
    """Front-camera SUV. Reads the NAVSIM driving command, not the SD route."""

    TAG = "suv"
    PLAN_DT = 0.5

    def build(self) -> None:
        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            raise SystemExit(f"[{self.TAG}] checkpoint not found: {self.ckpt_path!r}")
        repo = self.repo
        for root in (os.path.join(repo, "src"), repo):
            if root not in sys.path:
                sys.path.insert(0, root)
        os.chdir(repo)
        os.environ.setdefault("CKPT_LOCAL_DIR", "/mnt/workspace/aaai_rebuttal/ckpt")
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.environ["CKPT_LOCAL_DIR"])
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        os.environ.setdefault("SUV_MODEL_ID", "Wan2.2-TI2V-5B")
        os.environ.setdefault("SUV_TOKENIZER_MODEL_ID", "Wan2.2-TI2V-5B")
        os.environ.setdefault(
            "TEXT_EMBEDDING_CACHE_DIR",
            os.path.join(os.environ["CKPT_LOCAL_DIR"], "suv_text_embeds_cache"),
        )
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._build_agent()
        self.agent.eval()
        self._sensor_config = self.agent.get_sensor_config()
        if self.cameras is not None:
            self._sensor_config, self.dropped_sensors = narrow_sensor_config(
                self._sensor_config, self.cameras, self.history_frames
            )
            if self.dropped_sensors:
                print(
                    f"[{self.TAG}] sensors narrowed; not requested: {' '.join(self.dropped_sensors)}",
                    flush=True,
                )
        print(
            f"[{self.TAG}] loaded ckpt={os.path.basename(self.ckpt_path)} "
            f"steps={self.agent.num_inference_steps} repo={repo}",
            flush=True,
        )

    def _build_agent(self) -> None:
        from omegaconf import OmegaConf

        from experiments.navsimv1.pdm_agent import SUVNavsimV1Agent

        class _OdysseyAgent(SUVNavsimV1Agent):
            def get_sensor_config(self):
                from navsim.common.dataclasses import SensorConfig

                if hasattr(SensorConfig, "build_front_only_sensors"):
                    return SensorConfig.build_front_only_sensors(include=True)
                # The NAVSIM revision in the SUV env only has build_all_sensors.
                return SensorConfig(
                    cam_f0=True,
                    cam_l0=False,
                    cam_l1=False,
                    cam_l2=False,
                    cam_r0=False,
                    cam_r1=False,
                    cam_r2=False,
                    cam_b0=False,
                    lidar_pc=False,
                )

            def initialize(self) -> None:
                if self.model is not None:
                    return
                cfg = OmegaConf.load(self.model_config_path)
                if not bool(cfg.load_text_encoder):
                    cfg.load_text_encoder = True
                    dumped = Path(self.text_embedding_cache_dir).expanduser() / "_odyssey_model.yaml"
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
                        "Odyssey prompt is not in the text-embedding cache and "
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

        agent = _OdysseyAgent(
            checkpoint_path=self.ckpt_path,
            text_embedding_cache_dir=os.environ["TEXT_EMBEDDING_CACHE_DIR"],
            num_inference_steps=int(os.environ.get("SUV_NUM_INFERENCE_STEPS", "2")),
            visual_conditioning="history_4",
            device=self.device,
            seed=int(os.environ.get("SUV_SEED", "42")),
        )
        agent.initialize()
        self.agent = agent

    def infer(self, frames):
        scene_list = (
            self.history_sampler(frames)
            if self.history_sampler is not None
            else ipc.pad_history(frames, history_stride=self.history_stride)
        )
        if ipc.inject_ay_enabled():
            import copy

            scene_list = ipc.inject_ay([copy.deepcopy(frame) for frame in scene_list])
        agent_input = self._build_agent_input(scene_list)
        trajectory = self.agent.compute_trajectory(agent_input)
        poses = np.asarray(trajectory.poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] < 3:
            raise RuntimeError(f"[{self.TAG}] unexpected trajectory shape {poses.shape}")
        return poses[:, :3]
