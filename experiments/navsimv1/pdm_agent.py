from __future__ import annotations

import hashlib
import inspect
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from experiments.navsimv1.data.conditioning import resolve_visual_conditioning
from experiments.navsimv1.data.features import NavsimV1FeatureBuilder, NavsimV1FeatureConfig
from experiments.navsimv1.data.prompts import (
    DEFAULT_NAVSIM_PROMPT,
    build_navsim_prompts,
    normalize_stagea_modalities,
    stagea_prompt_overrides,
)
from suv.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
logger = logging.getLogger(__name__)


try:
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling as NuPlanTrajectorySampling
except ImportError as exc:  # pragma: no cover - only needed when NAVSIM devkit is installed.
    AbstractAgent = torch.nn.Module
    AgentInput = Any
    Scene = Any
    SensorConfig = Any
    Trajectory = Any
    NuPlanTrajectorySampling = Any
    _NAVSIM_IMPORT_ERROR = exc
else:
    _NAVSIM_IMPORT_ERROR = None


def _resolve_checkpoint(path_like: str) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    search_roots = []
    if (path / "checkpoints" / "weights").is_dir():
        search_roots.append(path / "checkpoints" / "weights")
    if (path / "weights").is_dir():
        search_roots.append(path / "weights")
    search_roots.append(path)
    candidates = [candidate for root in search_roots for candidate in sorted(root.glob("*.pt"))]
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found under directory: {path}")

    def _step_key(candidate: Path) -> tuple[int, float, str]:
        match = re.search(r"step[_-](\d+)", candidate.stem)
        step = int(match.group(1)) if match else -1
        return step, candidate.stat().st_mtime, candidate.name

    return max(candidates, key=_step_key)


def _resolve_repo_model_config() -> Path:
    return Path(__file__).resolve().parent / "config" / "model" / "suv_navsim.yaml"


def _build_history_condition_video(
    history_video: torch.Tensor,
    *,
    frame_mode: str = "history_plus_future",
    padding_mode: str = "repeat_first",
) -> torch.Tensor:
    frame_mode = str(frame_mode).strip().lower()
    padding_mode = str(padding_mode).strip().lower()
    if history_video.ndim != 4:
        raise ValueError(f"`history_video` must be [T, C, H, W], got {tuple(history_video.shape)}")
    if history_video.shape[0] < 1:
        raise ValueError("`history_video` must contain at least one frame.")
    if frame_mode == "current_plus_future":
        return history_video[-1:].contiguous()
    if frame_mode != "history_plus_future":
        raise ValueError(f"Unsupported video_frame_mode: {frame_mode!r}")
    if padding_mode == "none":
        return history_video.contiguous()
    if padding_mode == "repeat_first":
        return torch.cat([history_video[:1], history_video], dim=0).contiguous()
    raise ValueError(f"Unsupported history padding mode: {padding_mode!r}")


class SUVNavsimV1Agent(AbstractAgent):
    """NAVSIM `AbstractAgent` wrapper around a SUV NAVSIM v1 checkpoint."""

    requires_scene = False
    SLOT_JOINT_MODALITIES = ("rgb", "depth", "seg", "instance")

    def __init__(
        self,
        checkpoint_path: str,
        model_config_path: str | None = None,
        text_embedding_cache_dir: str = "./data/text_embeds_cache/navsim_v1",
        video_height: int = 384,
        video_width: int = 640,
        crop_top_bottom: int = 28,
        num_history_frames: int | None = None,
        num_future_frames: int = 8,
        frame_interval: int = 1,
        visual_conditioning: str = "history_4",
        video_frame_mode: str | None = None,
        history_padding_mode: str | None = None,
        fps: float = 2.0,
        action_dim: int = 3,
        proprio_dim: int = 11,
        prompt: str = DEFAULT_NAVSIM_PROMPT,
        prompt_mode: str = "dynamic",
        prompt_future_instruction: str | None = None,
        prompt_quality_instruction: str | None = None,
        prompt_history_seconds: float | None = 2.5,
        prompt_velocity_quantization: float = 0.5,
        prompt_acceleration_quantization: float = 0.5,
        stagea_modality: str | None = None,
        slot_inference: bool = True,
        context_len: int = 512,
        text_encoder_id: str = "wan22ti2v5b",
        mixed_precision: str = "bf16",
        device: str = "cuda",
        rand_device: str = "cpu",
        num_inference_steps: int = 10,
        seed: int = 42,
        tiled: bool = False,
        eval_visualize: bool = False,
        eval_visualization_dir: str = "suv_navsimv1_pdm_visualizations",
        eval_visualization_max_samples: int = 8,
        eval_score_ground_truth: bool = False,
        config: Any = None,
        lr: float | None = None,
        **_: Any,
    ):
        if _NAVSIM_IMPORT_ERROR is not None:
            raise ImportError(
                "SUVNavsimV1Agent requires the NAVSIM devkit on PYTHONPATH."
            ) from _NAVSIM_IMPORT_ERROR
        super().__init__()
        del config, lr, frame_interval
        self.checkpoint_path = str(checkpoint_path)
        self.model_config_path = str(model_config_path or _resolve_repo_model_config())
        self.text_embedding_cache_dir = str(text_embedding_cache_dir)
        self.context_len = int(context_len)
        self.text_encoder_id = str(text_encoder_id)
        self.prompt = str(prompt)
        self.prompt_mode = str(prompt_mode).strip().lower()
        self.prompt_future_instruction = prompt_future_instruction
        self.prompt_quality_instruction = prompt_quality_instruction
        self.prompt_history_seconds = None if prompt_history_seconds is None else float(prompt_history_seconds)
        self.prompt_velocity_quantization = float(prompt_velocity_quantization)
        self.prompt_acceleration_quantization = float(prompt_acceleration_quantization)
        self.stagea_modality = (
            None
            if stagea_modality is None or str(stagea_modality).strip().lower() in {"", "none", "null", "shared"}
            else normalize_stagea_modalities([stagea_modality])[0]
        )
        self.slot_inference = bool(slot_inference)
        resolved_conditioning = resolve_visual_conditioning(
            visual_conditioning,
            num_history_frames=num_history_frames,
            video_frame_mode=video_frame_mode,
            history_padding_mode=history_padding_mode,
        )
        self.visual_conditioning = resolved_conditioning.name
        self.num_history_frames = int(resolved_conditioning.num_history_frames)
        self.num_future_frames = int(num_future_frames)
        self.video_frame_mode = str(resolved_conditioning.video_frame_mode)
        self.history_padding_mode = str(resolved_conditioning.history_padding_mode)
        self.fps = float(fps)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.device_name = str(device)
        self.rand_device = str(rand_device)
        self.mixed_precision = str(mixed_precision)
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        self.tiled = bool(tiled)
        self.eval_visualization_enabled = bool(eval_visualize)
        self.eval_visualization_dir = str(eval_visualization_dir)
        self.eval_visualization_max_samples = int(eval_visualization_max_samples)
        self.eval_score_ground_truth = bool(eval_score_ground_truth)
        self._eval_visualization_saved_samples = 0
        self.model = None

        video_frame_mode_norm = self.video_frame_mode.strip().lower()
        history_padding_mode_norm = self.history_padding_mode.strip().lower()
        if video_frame_mode_norm == "current_plus_future":
            internal_history_frames = 1
        elif video_frame_mode_norm == "history_plus_future":
            internal_history_frames = self.num_history_frames
            if history_padding_mode_norm == "repeat_first":
                internal_history_frames += 1
            elif history_padding_mode_norm != "none":
                raise ValueError(f"Unsupported history_padding_mode: {self.history_padding_mode!r}")
        else:
            raise ValueError(f"Unsupported video_frame_mode: {self.video_frame_mode!r}")
        if internal_history_frames % 4 != 1:
            raise ValueError(
                "SUV NAVSIM agent condition history must satisfy T % 4 == 1 after padding, "
                f"got num_history_frames={self.num_history_frames}, "
                f"history_padding_mode={self.history_padding_mode!r}, "
                f"internal_history_frames={internal_history_frames}."
            )
        total_video_frames = internal_history_frames + self.num_future_frames
        if total_video_frames % 4 != 1:
            raise ValueError(
                "SUV NAVSIM agent video length must satisfy T % 4 == 1, "
                f"got total_video_frames={total_video_frames}."
            )

        self.feature_config = NavsimV1FeatureConfig(
            video_height=int(video_height),
            video_width=int(video_width),
            crop_top_bottom=int(crop_top_bottom),
            num_history_frames=self.num_history_frames,
        )
        self.feature_builder = NavsimV1FeatureBuilder(self.feature_config)

    def name(self) -> str:
        return "suv_navsimv1_agent"

    def get_sensor_config(self):
        return SensorConfig.build_front_only_sensors(include=True)

    def initialize(self) -> None:
        if self.model is not None:
            return
        if self.visual_conditioning != "history_4":
            raise RuntimeError(
                "The released SUV checkpoint requires visual_conditioning=history_4."
            )
        if not self.slot_inference:
            raise RuntimeError(
                "SUV NAVSIM v1 evaluation requires slot_inference=true for joint denoising."
            )
        device = self.device_name
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU for SUV NAVSIM agent.")
            device = "cpu"
        elif device == "cuda":
            device = "cuda:0"

        raw_model_cfg = OmegaConf.load(self.model_config_path)
        if not isinstance(raw_model_cfg, DictConfig):
            raw_model_cfg = OmegaConf.create(raw_model_cfg)
        model_root = OmegaConf.create({"model": raw_model_cfg})
        model_cfg = model_root.model
        model_dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(self.mixed_precision))
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        if not bool(getattr(self.model, "joint_future_access", False)):
            raise RuntimeError(
                "SUV NAVSIM v1 evaluation requires SUVJoint with joint future-scene access."
            )
        logger.info("SUV joint future-scene access enabled for NAVSIM v1 evaluation.")

        checkpoint = _resolve_checkpoint(self.checkpoint_path)
        logger.info("Loading SUV NAVSIM v1 checkpoint for PDM eval: %s", checkpoint)
        self.model.load_checkpoint(str(checkpoint))
        self.model.eval()

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_dir = Path(self.text_embedding_cache_dir).expanduser()
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache for prompt {prompt!r}: {cache_path}. "
                "Run experiments/navsimv1/scripts/evaluation/"
                "precompute_suv_navsimv1_pdm_text_embeds.sh first."
            )
        payload = torch.load(str(cache_path), map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.shape[0] != self.context_len or context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Text context length mismatch in {cache_path}: "
                f"context={tuple(context.shape)}, mask={tuple(context_mask.shape)}, expected L={self.context_len}."
            )
        context = context.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return context, context_mask

    def _build_prompt(self, ego_status: torch.Tensor, stagea_modality: str | None = None) -> str:
        prompt_prefix = self.prompt
        future_instruction = self.prompt_future_instruction
        quality_instruction = self.prompt_quality_instruction
        if stagea_modality is not None:
            prompt_prefix, future_instruction, quality_instruction = stagea_prompt_overrides(stagea_modality)
        return build_navsim_prompts(
            prompt_prefix=prompt_prefix,
            ego_status=ego_status.unsqueeze(0),
            batch_size=1,
            history_seconds=(
                float(ego_status.shape[0]) / self.fps
                if self.prompt_history_seconds is None
                else self.prompt_history_seconds
            ),
            future_seconds=float(self.num_future_frames) / self.fps,
            mode=self.prompt_mode,
            future_instruction=future_instruction,
            quality_instruction=quality_instruction,
            velocity_quantization=self.prompt_velocity_quantization,
            acceleration_quantization=self.prompt_acceleration_quantization,
        )[0]

    def _uses_slot_inference(self) -> bool:
        # Route to the packed multi-slot path only when explicitly requested.
        if self.model is None or not callable(getattr(self.model, "infer_action_slot", None)):
            return False
        return self.slot_inference

    def _build_slot_text_contexts(self, ego_status: torch.Tensor) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
        contexts = []
        context_masks = []
        for modality in self.SLOT_JOINT_MODALITIES:
            prompt = self._build_prompt(ego_status, stagea_modality=modality)
            context, context_mask = self._get_cached_text_context(prompt)
            contexts.append(context)
            context_masks.append(context_mask)
        return (
            self.SLOT_JOINT_MODALITIES,
            torch.stack(contexts, dim=0),
            torch.stack(context_masks, dim=0).bool(),
        )

    @torch.no_grad()
    def compute_trajectory_for_modality(
        self,
        agent_input: AgentInput,
        stagea_modality: str | None,
    ) -> Trajectory:
        self.initialize()
        assert self.model is not None
        modality = (
            None
            if stagea_modality is None or str(stagea_modality).strip().lower() in {"", "none", "null", "shared"}
            else normalize_stagea_modalities([stagea_modality])[0]
        )

        features = self.feature_builder.compute_features(agent_input)
        history_video = torch.as_tensor(features["video_front"], dtype=torch.float32)
        ego_status = torch.as_tensor(features["ego_status"], dtype=torch.float32)
        if history_video.shape[0] < 1:
            raise RuntimeError("SUV NAVSIM agent did not receive any history frames.")

        condition_video = _build_history_condition_video(
            history_video,
            frame_mode=self.video_frame_mode,
            padding_mode=self.history_padding_mode,
        )
        input_image = condition_video.mul(2.0).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        current_proprio = ego_status[-1]

        if self._uses_slot_inference():
            slot_names, slot_contexts, slot_context_masks = self._build_slot_text_contexts(ego_status)
            pred = self.model.infer_action_slot(
                input_image=input_image,
                action_horizon=self.num_future_frames,
                num_video_frames=int(condition_video.shape[0]) + self.num_future_frames,
                proprio=current_proprio,
                slot_contexts=slot_contexts,
                slot_context_masks=slot_context_masks,
                slot_names=slot_names,
                num_inference_steps=self.num_inference_steps,
                seed=self.seed,
                rand_device=self.rand_device,
                tiled=self.tiled,
            )
        else:
            prompt = self._build_prompt(ego_status, stagea_modality=modality)
            context, context_mask = self._get_cached_text_context(prompt)

            infer_kwargs = {
                "prompt": None,
                "input_image": input_image,
                "action_horizon": self.num_future_frames,
                "proprio": current_proprio,
                "context": context,
                "context_mask": context_mask,
                "num_inference_steps": self.num_inference_steps,
                "seed": self.seed,
                "rand_device": self.rand_device,
                "tiled": self.tiled,
            }
            if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
                infer_kwargs["num_video_frames"] = int(condition_video.shape[0]) + self.num_future_frames

            pred = self.model.infer_action(**infer_kwargs)
        poses = pred["action"].detach().to(device="cpu", dtype=torch.float32).numpy()[:, :3]
        return Trajectory(
            poses.astype(np.float32),
            NuPlanTrajectorySampling(num_poses=int(poses.shape[0]), interval_length=1.0 / self.fps),
        )

    @torch.no_grad()
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        return self.compute_trajectory_for_modality(agent_input, self.stagea_modality)

    def should_visualize_pdm_score_sample(self) -> bool:
        if self.eval_visualization_max_samples <= 0:
            return True
        return self._eval_visualization_saved_samples < self.eval_visualization_max_samples

    def maybe_visualize_pdm_score_sample(
        self,
        agent_input: AgentInput,
        scene: Scene,
        trajectory: Trajectory,
        output_root: Path,
        token: str,
        log_name: str,
        sample_idx: int,
        metrics: dict[str, float] | None = None,
        stage_name: str | None = None,
        frame_type: str | None = None,
    ) -> None:
        del agent_input
        if not self.should_visualize_pdm_score_sample():
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from navsim.visualization.bev import add_trajectory_to_bev_ax
        from navsim.visualization.config import TRAJECTORY_CONFIG
        from navsim.visualization.plots import plot_bev_frame

        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage_name or "sample"))[:80]
        output_dir = Path(output_root) / self.eval_visualization_dir / safe_stage
        output_dir.mkdir(parents=True, exist_ok=True)
        current_idx = int(scene.scene_metadata.num_history_frames) - 1
        fig, ax = plot_bev_frame(scene, current_idx)

        available_human_future = max(0, len(scene.frames) - current_idx - 1)
        human_future_frames = min(
            int(trajectory.trajectory_sampling.num_poses),
            available_human_future,
        )
        human_note = "  red=prediction"
        if human_future_frames <= 0:
            human_note += "  human_unavailable"
        else:
            human_note = "  green=human  red=prediction"
            if human_future_frames < int(trajectory.trajectory_sampling.num_poses):
                human_note += (
                    f"  human_frames={human_future_frames}/"
                    f"{int(trajectory.trajectory_sampling.num_poses)}"
                )
            human_trajectory = scene.get_future_trajectory(
                num_trajectory_frames=human_future_frames
            )
            add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
        add_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["agent"])

        metric_text = ""
        if metrics:
            metric_text = "\n" + "  ".join(
                f"{name}={float(value):.3f}" for name, value in metrics.items()
            )
        ax.set_title(
            f"{safe_stage} token={token}{metric_text}\n"
            f"frame_type={frame_type or 'n/a'}{human_note}",
            fontsize=9,
        )
        safe_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(token))[:120] or f"sample_{sample_idx}"
        output_path = output_dir / (
            f"{safe_stage}_{self._eval_visualization_saved_samples:04d}_{log_name}_{safe_token}.png"
        )
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved official NAVSIM BEV visualization: %s", output_path)
        self._eval_visualization_saved_samples += 1
