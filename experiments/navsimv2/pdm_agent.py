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

from experiments.navsimv2.data.conditioning import resolve_visual_conditioning
from experiments.navsimv2.data.features import (
    NavsimV2FeatureBuilder,
    NavsimV2FeatureConfig,
)
from experiments.navsimv2.data.prompts import (
    DEFAULT_NAVSIM_PROMPT,
    action_prompt_components,
    build_navsim_prompts,
    normalize_action_context_mode,
    normalize_stagea_modalities,
    stagea_prompt_overrides,
)
from suv.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
logger = logging.getLogger(__name__)

DEFAULT_SLOT_MODALITIES = ("rgb", "depth", "seg", "instance")


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


def _resolve_slot_modalities(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "all", "default"}):
        return DEFAULT_SLOT_MODALITIES
    selected = set(normalize_stagea_modalities(value))
    modalities = tuple(modality for modality in DEFAULT_SLOT_MODALITIES if modality in selected)
    if not modalities or modalities[0] != "rgb":
        raise ValueError("slot_modalities must include rgb as the first modality.")
    return modalities


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
    raise ValueError(f"Unsupported history_padding_mode: {padding_mode!r}")


class SUVNavsimV2Agent(AbstractAgent):
    """NAVSIM `AbstractAgent` wrapper around a SUV NAVSIM v2 checkpoint."""

    requires_scene = False
    SLOT_JOINT_MODALITIES = DEFAULT_SLOT_MODALITIES

    def __init__(
        self,
        checkpoint_path: str,
        model_config_path: str | None = None,
        text_embedding_cache_dir: str = "./data/text_embeds_cache/navsim_v2",
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
        action_context_mode: str = "base",
        action_prompt: str | None = None,
        action_prompt_future_instruction: str | None = None,
        action_prompt_quality_instruction: str | None = None,
        stagea_modality: str | None = None,
        slot_modalities: str | list[str] | tuple[str, ...] | None = None,
        slot_inference: bool = False,
        context_len: int = 512,
        text_encoder_id: str = "wan22ti2v5b",
        mixed_precision: str = "bf16",
        device: str = "cuda",
        rand_device: str = "cpu",
        num_inference_steps: int = 10,
        seed: int = 42,
        tiled: bool = False,
        eval_visualize: bool = False,
        eval_visualization_dir: str = "suv_navsimv2_pdm_visualizations",
        eval_visualization_max_samples: int = 8,
        config: Any = None,
        lr: float | None = None,
        **_: Any,
    ):
        if _NAVSIM_IMPORT_ERROR is not None:
            raise ImportError(
                "SUVNavsimV2Agent requires the NAVSIM devkit on PYTHONPATH."
            ) from _NAVSIM_IMPORT_ERROR
        trajectory_sampling = NuPlanTrajectorySampling(
            num_poses=int(num_future_frames),
            interval_length=1.0 / float(fps),
        )
        if "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters:
            super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=False)
        else:
            super().__init__(requires_scene=False)
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
        raw_action_context_mode = "base" if action_context_mode is None else str(action_context_mode).strip().lower()
        self.use_base_prompt_context = raw_action_context_mode in {
            "base",
            "shared",
            "default",
            "video_prompt",
        }
        self.action_context_mode = (
            "base"
            if self.use_base_prompt_context
            else normalize_action_context_mode(action_context_mode)
        )
        self.action_prompt = action_prompt
        self.action_prompt_future_instruction = action_prompt_future_instruction
        self.action_prompt_quality_instruction = action_prompt_quality_instruction
        self.stagea_modality = (
            None
            if stagea_modality is None or str(stagea_modality).strip().lower() in {"", "none", "null", "shared"}
            else normalize_stagea_modalities([stagea_modality])[0]
        )
        self.slot_modalities = _resolve_slot_modalities(slot_modalities)
        self.slot_inference = bool(slot_inference)
        if (
            self.action_context_mode == "custom"
            and str(self.action_prompt or "").strip().lower() in {"", "none", "null"}
        ):
            raise ValueError("action_context_mode='custom' requires a non-empty action_prompt.")
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
        self._eval_visualization_saved_samples = 0
        self._prompt_debug_logged = False
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

        self.feature_config = NavsimV2FeatureConfig(
            video_height=int(video_height),
            video_width=int(video_width),
            crop_top_bottom=int(crop_top_bottom),
            num_history_frames=self.num_history_frames,
        )
        self.feature_builder = NavsimV2FeatureBuilder(self.feature_config)

    def name(self) -> str:
        return "suv_navsimv2_agent"

    def get_sensor_config(self):
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

        checkpoint = _resolve_checkpoint(self.checkpoint_path)
        logger.info("Loading SUV NAVSIM v2 checkpoint for PDM eval: %s", checkpoint)
        self.model.load_checkpoint(str(checkpoint))
        self.model.eval()

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_dir = Path(self.text_embedding_cache_dir).expanduser()
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache for prompt {prompt!r}: {cache_path}. "
                f"hash={hashed}, cache_dir={cache_dir}. "
                "Run experiments/navsimv2/scripts/evaluation/precompute_suv_navsimv2_pdm_text_embeds.sh first."
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
        elif not self.use_base_prompt_context:
            prompt_prefix, future_instruction, quality_instruction = action_prompt_components(
                self.action_context_mode,
                base_prompt=self.prompt,
                custom_prompt=self.action_prompt,
                future_instruction=self.action_prompt_future_instruction,
                quality_instruction=self.action_prompt_quality_instruction,
            )
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
        for modality in self.slot_modalities:
            prompt = self._build_prompt(ego_status, stagea_modality=modality)
            context, context_mask = self._get_cached_text_context(prompt)
            contexts.append(context)
            context_masks.append(context_mask)
        return (
            self.slot_modalities,
            torch.stack(contexts, dim=0),
            torch.stack(context_masks, dim=0).bool(),
        )

    @torch.no_grad()
    def compute_trajectory(self, agent_input: AgentInput, *_: Any) -> Trajectory:
        self.initialize()
        assert self.model is not None

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
            if not self._prompt_debug_logged:
                logger.info(
                    "SUV NAVSIM v2 PDM slot prompt debug: model=%s slot_inference_arg=%s slot_names=%s context_norm=%.4f",
                    type(self.model).__name__,
                    self.slot_inference,
                    slot_names,
                    float(slot_contexts.detach().float().norm().item()),
                )
                self._prompt_debug_logged = True
            slot_infer_kwargs = dict(
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
            pred = self.model.infer_action_slot(**slot_infer_kwargs)
        else:
            prompt = self._build_prompt(ego_status, stagea_modality=self.stagea_modality)
            context, context_mask = self._get_cached_text_context(prompt)
            if not self._prompt_debug_logged:
                logger.info(
                    "SUV NAVSIM v2 PDM prompt debug: action_context_mode=%s stagea_modality=%s prompt_hash=%s context_norm=%.4f",
                    self.action_context_mode,
                    self.stagea_modality,
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
                    float(context.detach().float().norm().item()),
                )
                self._prompt_debug_logged = True

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
        if not self.should_visualize_pdm_score_sample():
            return
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage_name or "sample"))[:80]
        stage_key = safe_stage.lower()
        render_errors: list[str] = []
        human_note = "  red=prediction"
        human_trajectory = None
        current_idx = int(scene.scene_metadata.num_history_frames) - 1
        if stage_key != "stage_two":
            try:
                available_human_future = max(0, len(scene.frames) - current_idx - 1)
                human_future_frames = min(
                    int(trajectory.trajectory_sampling.num_poses),
                    available_human_future,
                )
                if human_future_frames <= 0:
                    human_note = "  red=prediction  human_unavailable"
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
            except Exception as exc:
                human_note = "  red=prediction  human_unavailable"
                render_errors.append(f"human trajectory export failed: {type(exc).__name__}: {exc}")
        else:
            human_note = "  red=prediction  human_not_available_for_stage_two"

        fig = None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from navsim.visualization.bev import add_trajectory_to_bev_ax
            from navsim.visualization.config import TRAJECTORY_CONFIG
            from navsim.visualization.plots import plot_bev_frame

            fig, ax = plot_bev_frame(scene, current_idx)
            if human_trajectory is not None:
                add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
            add_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["agent"])
            metric_items = list((metrics or {}).items())[:6]
            metric_text = ""
            if metric_items:
                metric_text = "\n" + "  ".join(
                    f"{name}={float(value):.3f}" for name, value in metric_items
                )
            ax.set_title(
                f"{safe_stage} token={token}{metric_text}\n"
                f"frame_type={frame_type or 'n/a'}{human_note}",
                fontsize=9,
            )
            output_dir = Path(output_root) / self.eval_visualization_dir / safe_stage
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_token = (
                re.sub(r"[^A-Za-z0-9_.-]+", "_", str(token))[:120]
                or f"sample_{sample_idx}"
            )
            safe_log_name = (
                re.sub(r"[^A-Za-z0-9_.-]+", "_", str(log_name))[:100]
                or "log"
            )
            output_path = output_dir / (
                f"{safe_stage}_{self._eval_visualization_saved_samples:04d}_"
                f"{safe_log_name}_{safe_token}.png"
            )
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            logger.info("Saved official NAVSIM BEV visualization: %s", output_path)
        except Exception as exc:
            render_errors.append(f"BEV rendering failed: {type(exc).__name__}: {exc}")
            logger.exception("Failed to render NAVSIM BEV for token %s", token)
        finally:
            if fig is not None:
                plt.close(fig)

        self._eval_visualization_saved_samples += 1
