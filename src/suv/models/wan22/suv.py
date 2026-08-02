import inspect
import os
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
from PIL import Image

from suv.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .slot_layout import SlotLayout

logger = get_logger(__name__)


class SUV(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_shift: float = 5.0,
        video_num_timesteps: int = 1000,
        action_shift: float = 5.0,
        action_num_timesteps: int = 1000,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_timesteps=video_num_timesteps,
            shift=video_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_timesteps=action_num_timesteps,
            shift=action_shift,
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        mot_checkpoint_mixed_attn: bool = True,
        video_shift: float = 5.0,
        video_num_timesteps: int = 1000,
        action_shift: float = 5.0,
        action_num_timesteps: int = 1000,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for SUV.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for SUV.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT(**action_dit_config).to(device=device, dtype=torch_dtype)
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_shift=video_shift,
            video_num_timesteps=video_num_timesteps,
            action_shift=action_shift,
            action_num_timesteps=action_num_timesteps,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit": "checkpoint",
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )


    @staticmethod
    def _get_uniform_int(value: Any, default: int, name: str) -> int:
        if value is None:
            return int(default)
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().reshape(-1)
            if flat.numel() == 0:
                raise ValueError(f"`{name}` cannot be an empty tensor.")
            first = int(flat[0].item())
            if not torch.all(flat == flat[0]):
                raise ValueError(f"`{name}` must be uniform across the batch, got {flat.tolist()}.")
            return first
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError(f"`{name}` cannot be empty.")
            first = int(value[0])
            if any(int(item) != first for item in value):
                raise ValueError(f"`{name}` must be uniform across the batch, got {value}.")
            return first
        return int(value)


    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim == 4:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,H,W], [3,H,W], or [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
            video = input_image.to(device=self.device).unsqueeze(2)
        elif input_image.ndim == 5:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
            if input_image.shape[2] < 1:
                raise ValueError("`input_image` history length must be >= 1.")
            if input_image.shape[2] % 4 != 1:
                raise ValueError(
                    "`input_image` history length must satisfy T % 4 == 1 for VAE encoding, "
                    f"got T={input_image.shape[2]}."
                )
            video = input_image.to(device=self.device)
        else:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W], [3,H,W], or [1,3,T,H,W], got {tuple(input_image.shape)}"
            )
        z = self.vae.encode(
            video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames


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

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            condition_video_frames=condition_video_frames,
            slot_layout=slot_layout,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> known video prefix only
        condition_tokens = min(condition_video_frames * video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :condition_tokens] = True
        return mask


    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        condition_latent_frames: int = 1,
        gt_action: Optional[torch.Tensor] = None,
        slot_layout: Optional[SlotLayout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            condition_latent_frames=condition_latent_frames,
            slot_layout=slot_layout,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            condition_video_frames=condition_latent_frames,
            slot_layout=slot_layout,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        condition_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        condition_latent_frames = int(condition_latents.shape[2])
        timestep_video = torch.zeros_like(timestep_action, dtype=condition_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=condition_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            condition_latent_frames=condition_latent_frames,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            condition_video_frames=condition_latent_frames,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            infer_action_kwargs = {
                "prompt": prompt,
                "input_image": input_image.clone(),
                "action_horizon": action_horizon,
                "context": context.clone() if context is not None else None,
                "context_mask": context_mask.clone() if context_mask is not None else None,
                "num_inference_steps": num_inference_steps,
                "sigma_shift": sigma_shift,
                "seed": seed,
                "rand_device": rand_device,
                "tiled": tiled,
                "proprio": proprio.clone() if proprio is not None else None,
            }
            if "num_video_frames" in inspect.signature(self.infer_action).parameters:
                infer_action_kwargs["num_video_frames"] = num_video_frames
            action_only_out = self.infer_action(
                **infer_action_kwargs,
            )["action"]

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim == 4:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,H,W], [3,H,W], or [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
            _, _, height, width = input_image.shape
            input_num_frames = 1
        elif input_image.ndim == 5:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
            _, _, input_num_frames, height, width = input_image.shape
        else:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W], [3,H,W], or [1,3,T,H,W], got {tuple(input_image.shape)}"
            )
        if input_num_frames > num_video_frames:
            raise ValueError(
                f"`input_image` history frames cannot exceed num_video_frames: {input_num_frames} > {num_video_frames}"
            )
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
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

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        condition_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        condition_latent_frames = int(condition_latents.shape[2])
        if condition_latent_frames > latent_t:
            raise ValueError(
                f"Condition latent frames cannot exceed video latent frames: {condition_latent_frames} > {latent_t}"
            )
        latents_video[:, :, :condition_latent_frames] = condition_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                condition_latent_frames=condition_latent_frames,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, :condition_latent_frames] = condition_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim == 4:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,H,W], [3,H,W], or [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
            _, _, height, width = input_image.shape
            input_num_frames = 1
        elif input_image.ndim == 5:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
            _, _, input_num_frames, height, width = input_image.shape
        else:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W], [3,H,W], or [1,3,T,H,W], got {tuple(input_image.shape)}"
            )
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if num_video_frames is not None and input_num_frames > int(num_video_frames):
            raise ValueError(
                f"`input_image` history frames cannot exceed num_video_frames: {input_num_frames} > {num_video_frames}"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        condition_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        condition_latent_frames = int(condition_latents.shape[2])
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (condition_latents.shape[0],),
            dtype=condition_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=condition_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            condition_latent_frames=condition_latent_frames,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            condition_video_frames=condition_latent_frames,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def _prepare_slot_action(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        slot_contexts: torch.Tensor,
        slot_context_masks: torch.Tensor,
        slot_names: tuple[str, ...] | list[str],
        proprio: Optional[torch.Tensor],
        tiled: bool,
    ) -> dict[str, Any]:
        """Shared setup for slot-aware action inference (base + joint).

        Validates inputs, encodes the RGB condition latents, builds the
        `SlotLayout`, and packs the multi-slot text context: each slot segment
        is ``[slot_context | proprio]`` and the
        segments are concatenated along the token axis (``action_context``). It
        also returns the slot-0 (rgb) segment (``condition_video_context``), which
        is the only text the condition video frames cross-attend to under
        `build_slot_text_context_mask(condition_segment=0)`.
        """
        action_horizon = self._get_uniform_int(action_horizon, default=0, name="action_horizon")
        num_video_frames = self._get_uniform_int(num_video_frames, default=0, name="num_video_frames")
        if action_horizon < 1:
            raise ValueError(f"`action_horizon` must be >= 1, got {action_horizon}.")
        if num_video_frames < 2:
            raise ValueError(f"`num_video_frames` must be >= 2, got {num_video_frames}.")

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim == 4:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    "`input_image` must have shape [1,3,H,W], [3,H,W], "
                    f"or [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
            _, _, height, width = input_image.shape
            input_num_frames = 1
        elif input_image.ndim == 5:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
            _, _, input_num_frames, height, width = input_image.shape
        else:
            raise ValueError(
                "`input_image` must have shape [1,3,H,W], [3,H,W], "
                f"or [1,3,T,H,W], got {tuple(input_image.shape)}"
            )
        if input_num_frames > num_video_frames:
            raise ValueError(
                "`input_image` history frames cannot exceed num_video_frames: "
                f"{input_num_frames} > {num_video_frames}"
            )
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                "`input_image` must be resized before infer, expected multiples "
                f"of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        slot_names = tuple(str(name) for name in slot_names)
        if len(slot_names) < 1:
            raise ValueError("`slot_names` must contain at least one slot name.")
        if len(set(slot_names)) != len(slot_names):
            raise ValueError(f"`slot_names` must be unique, got {slot_names}.")
        num_slots = len(slot_names)

        if slot_contexts.ndim == 3:
            slot_contexts = slot_contexts.unsqueeze(0)
        if slot_context_masks.ndim == 2:
            slot_context_masks = slot_context_masks.unsqueeze(0)
        if slot_contexts.ndim != 4 or slot_context_masks.ndim != 3:
            raise ValueError(
                "`slot_contexts/slot_context_masks` must be [N,L,D]/[N,L] "
                "or [1,N,L,D]/[1,N,L], got "
                f"{tuple(slot_contexts.shape)} and {tuple(slot_context_masks.shape)}"
            )
        if slot_contexts.shape[0] != 1 or slot_context_masks.shape[0] != 1:
            raise ValueError(
                "`infer_action_slot` currently supports batch size 1, got "
                f"{slot_contexts.shape[0]} and {slot_context_masks.shape[0]}."
            )
        if slot_contexts.shape[:3] != slot_context_masks.shape:
            raise ValueError(
                "`slot_contexts` leading dims must match `slot_context_masks`, "
                f"got {tuple(slot_contexts.shape)} and {tuple(slot_context_masks.shape)}"
            )
        if int(slot_contexts.shape[1]) != num_slots:
            raise ValueError(
                "`slot_contexts` slot dim must match `slot_names`, got "
                f"{slot_contexts.shape[1]} vs {num_slots}."
            )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        condition_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        ).to(device=self.device, dtype=self.torch_dtype)
        condition_latent_frames = int(condition_latents.shape[2])

        temporal_factor = int(self.vae.temporal_downsample_factor)
        latent_t = (num_video_frames - 1) // temporal_factor + 1
        if condition_latent_frames >= latent_t:
            raise ValueError(
                "`input_image` must leave at least one future latent frame: "
                f"{condition_latent_frames} >= {latent_t}."
            )
        slot_frames = int(latent_t) - int(condition_latent_frames)
        slot_layout = SlotLayout(
            num_condition_frames=condition_latent_frames,
            slot_frames=slot_frames,
            slot_names=slot_names,
        )

        context = slot_contexts.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = slot_context_masks.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio_token = self.proprio_encoder(
                proprio.to(device=self.device, dtype=self.torch_dtype).unsqueeze(1)
            ).to(dtype=context.dtype)
            proprio_token = proprio_token.unsqueeze(1).expand(-1, num_slots, -1, -1)
            proprio_mask = torch.ones((1, num_slots, 1), dtype=torch.bool, device=context_mask.device)
            context = torch.cat([context, proprio_token], dim=2)
            context_mask = torch.cat([context_mask, proprio_mask], dim=2)
        elif proprio is not None:
            raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")

        _, _, context_len, context_dim = context.shape
        # slot-0 (rgb) segment that the condition video frames cross-attend to.
        condition_video_context = context[:, 0]
        condition_video_context_mask = context_mask[:, 0]
        action_context = context.reshape(1, num_slots * context_len, context_dim)
        action_context_mask = context_mask.reshape(1, num_slots * context_len)

        if not bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)):
            raise ValueError("Slot-aware inference requires video_expert.fuse_vae_embedding_in_latents=true.")

        return {
            "condition_latents": condition_latents,
            "condition_latent_frames": condition_latent_frames,
            "slot_layout": slot_layout,
            "num_slots": num_slots,
            "slot_names": slot_names,
            "action_horizon": action_horizon,
            "num_video_frames": num_video_frames,
            "action_context": action_context,
            "action_context_mask": action_context_mask,
            "condition_video_context": condition_video_context,
            "condition_video_context_mask": condition_video_context_mask,
        }

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
    ) -> dict[str, Any]:
        """Slot-aware action inference for base `SUV`.

        The action conditions only on the RGB history (its `_build_mot_attention_mask`
        restricts action -> video to the condition prefix), so we feed the action
        expert the same packed multi-slot text context expected by slot inference but
        SKIP generating the slot futures entirely: only the RGB condition is
        prefilled into the video KV cache and only the action scheduler iterates.
        This is numerically equivalent to the full packed forward for the action
        output (condition frames are blind to futures and cross-attend only to the
        slot-0 segment) at a fraction of the cost.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action_slot` requires `video_attention_mask_mode='first_frame_causal'`."
            )

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
        action_context = prep["action_context"]
        action_context_mask = prep["action_context_mask"]
        video_context = prep["condition_video_context"]
        video_context_mask = prep["condition_video_context_mask"]

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        timestep_video = torch.zeros((condition_latents.shape[0],), dtype=condition_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=condition_latents,
            timestep=timestep_video,
            context=video_context,
            context_mask=video_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            condition_latent_frames=condition_latent_frames,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            condition_video_frames=condition_latent_frames,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=action_context,
                context_mask=action_context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if return_debug:
            result.update(
                {
                    "slot_layout": slot_layout,
                    "condition_latents": condition_latents.detach().to(device="cpu", dtype=torch.float32),
                }
            )
        return result

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=test_action_with_infer_action,
        )

    @staticmethod
    def _report_load(tag, result):
        """Log the missing/unexpected keys that strict=False would otherwise swallow."""
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if missing:
            logger.warning(
                "load_checkpoint[%s]: %d MISSING keys (in model, absent from ckpt) e.g. %s",
                tag, len(missing), missing[:5],
            )
        if unexpected:
            logger.warning(
                "load_checkpoint[%s]: %d UNEXPECTED keys (in ckpt, absent from model) e.g. %s",
                tag, len(unexpected), unexpected[:5],
            )
        if not missing and not unexpected:
            logger.info("load_checkpoint[%s]: all keys matched.", tag)

    def _assert_expert_loaded(self, path, result, prefix, label):
        """Fail loudly if a whole expert received zero weights (silent strict=False skip)."""
        missing = set(getattr(result, "missing_keys", []) or [])
        expected = [n for n, _ in self.mot.named_parameters() if n.startswith(prefix)]
        if not expected or any(n not in missing for n in expected):
            return
        msg = (
            f"load_checkpoint: the {label} received ZERO of its {len(expected)} weights "
            f"from {path} (all missing) -- the checkpoint lacks the trained {label} or its "
            f"keys are renamed. Refusing to continue with uninitialized weights."
        )
        if os.environ.get("SUV_ALLOW_PARTIAL_LOAD", "").lower() in {"1", "true", "yes"}:
            logger.warning("%s (continuing: SUV_ALLOW_PARTIAL_LOAD is set)", msg)
            return
        raise ValueError(msg + " Set SUV_ALLOW_PARTIAL_LOAD=1 to override.")

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu")
        logger.info(
            "load_checkpoint: %s | ckpt step=%s | top-level keys=%s",
            path, payload.get("step"), sorted(str(k) for k in payload.keys()),
        )
        if "mot" in payload:
            result = self.mot.load_state_dict(payload["mot"], strict=False)
            self._report_load("mot", result)
            # Stop if an entire expert is missing instead of silently evaluating
            # randomly initialized weights.
            self._assert_expert_loaded(path, result, "mixtures.action.", "action expert")
            self._assert_expert_loaded(path, result, "mixtures.video.", "video expert")
        elif "dit" in payload:
            logger.warning(
                "Checkpoint %s has a legacy `dit` key (video-only) and NO `mot` key: loading "
                "the VIDEO expert ONLY. The ACTION expert keeps its INITIALIZED weights and is "
                "NOT restored -- action predictions will be at cold-start quality. Use a "
                "checkpoint with an `mot` key (trained action) if that is not intended.",
                path,
            )
            result = self.video_expert.load_state_dict(payload["dit"], strict=False)
            self._report_load("dit(video-only)", result)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        return payload
