from typing import Any, Optional

import torch

from suv.utils.logging_config import get_logger

from .slot_layout import SlotLayout
from .suv import SUV

logger = get_logger(__name__)


class SUVJoint(SUV):
    """SUV inference with joint future-scene and action denoising."""

    joint_future_access = True

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config")
        if not isinstance(video_dit_config, dict):
            raise ValueError("`video_dit_config` must be provided as a dict for SUVJoint.")
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError("SUVJoint requires `video_dit_config['action_conditioned']=false`.")
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        condition_video_frames: int = 1,
        slot_layout: Optional[SlotLayout] = None,
    ) -> torch.Tensor:
        condition_video_frames = max(int(condition_video_frames), 1)
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # Each future stream attends to the shared condition and its own tokens.
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            condition_video_frames=condition_video_frames,
            slot_layout=slot_layout,
        )
        # Action tokens attend to one another and to the complete packed video:
        # condition + future RGB, depth, segmentation, and instance tracks.
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def infer_action_slot(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        slot_contexts: torch.Tensor,
        slot_context_masks: torch.Tensor,
        slot_names: tuple[str, ...] | list[str] = ("rgb", "depth", "seg", "instance"),
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        return_debug: bool = False,
        return_slot_videos: bool = False,
    ) -> dict[str, Any]:
        """Jointly denoise the packed future streams and ego action."""
        self.eval()

        prep = self._prepare_slot_action(
            input_image=input_image,
            action_horizon=action_horizon,
            num_video_frames=num_video_frames,
            slot_contexts=slot_contexts,
            slot_context_masks=slot_context_masks,
            slot_names=slot_names,
            proprio=proprio,
            tiled=tiled,
        )
        condition_latents = prep["condition_latents"]
        condition_latent_frames = prep["condition_latent_frames"]
        slot_layout = prep["slot_layout"]
        action_horizon = prep["action_horizon"]
        context = prep["action_context"]
        context_mask = prep["action_context_mask"]

        z_dim = int(condition_latents.shape[1])
        latent_h = int(condition_latents.shape[3])
        latent_w = int(condition_latents.shape[4])
        packed_video_frames = slot_layout.total_frames

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, z_dim, packed_video_frames, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        latents_video[:, :, :condition_latent_frames] = condition_latents.clone()
        initial_latents_video = latents_video.detach().clone() if return_debug else None
        condition_after_steps = [] if return_debug else None

        video_timesteps, video_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for video_t, video_delta, action_t, action_delta in zip(
            video_timesteps,
            video_deltas,
            action_timesteps,
            action_deltas,
        ):
            timestep_video = video_t.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = action_t.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=True,
                condition_latent_frames=condition_latent_frames,
                gt_action=None,
                slot_layout=slot_layout,
            )

            latents_video = self.infer_video_scheduler.step(pred_video, video_delta, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, action_delta, latents_action)
            latents_video[:, :, :condition_latent_frames] = condition_latents.clone()
            if condition_after_steps is not None:
                condition_after_steps.append(
                    latents_video[:, :, :condition_latent_frames]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if return_slot_videos:
            try:
                slot_videos = {}
                for slot_idx, slot_name in enumerate(slot_layout.slot_names):
                    start, end = slot_layout.slot_frame_range(slot_idx)
                    clip_latents = torch.cat(
                        [condition_latents, latents_video[:, :, start:end]],
                        dim=2,
                    )
                    slot_videos[str(slot_name)] = self._decode_latents(clip_latents, tiled=tiled)
                result.update(
                    {
                        "slot_videos": slot_videos,
                        "slot_names": tuple(str(name) for name in slot_layout.slot_names),
                        "num_condition_frames": (
                            int(input_image.shape[2]) if input_image.ndim == 5 else 1
                        ),
                    }
                )
            except Exception as exc:
                logger.exception("Failed to decode joint slot videos for qualitative export.")
                result["slot_video_error"] = f"{type(exc).__name__}: {exc}"
        if return_debug:
            result.update(
                {
                    "slot_layout": slot_layout,
                    "condition_latents": condition_latents.detach().to(
                        device="cpu", dtype=torch.float32
                    ),
                    "initial_latents_video": initial_latents_video.to(
                        device="cpu", dtype=torch.float32
                    ),
                    "final_latents_video": latents_video.detach().to(
                        device="cpu", dtype=torch.float32
                    ),
                    "condition_after_steps": condition_after_steps,
                }
            )
        return result
