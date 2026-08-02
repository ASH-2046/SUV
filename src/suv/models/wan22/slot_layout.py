"""Slot layout utilities for packed multi-target video inference.

A "slot" layout packs N modality-specific future clips along the latent
temporal axis behind a shared condition prefix:

    [ cond(c) | slot_0(m) | slot_1(m) | ... | slot_{N-1}(m) ]    F = c + N * m

Design invariants:

1. Every slot reuses the *same* virtual temporal RoPE coordinates
   ``c .. c+m-1`` (see :func:`SlotLayout.virtual_frame_index`).
2. Slots never attend to each other. A slot token attends only to the shared
   condition prefix and to its own slot, following the configured single-clip
   video attention mask mode
   (see :func:`build_slot_causal_frame_mask`).
3. Each slot cross-attends only to its own text-prompt segment
   (see :func:`build_slot_text_context_mask`).

Under these rules the receptive field of any slot token is indistinguishable
from a standard single-clip forward of ``c + m`` frames, i.e. a slot forward
is computationally equivalent to batching N of today's single-modality
samples while sharing the condition tokens' KV.

This module is intentionally free of model code: pure functions over plain
tensors, unit-testable on CPU, imported by ``wan_video_dit.py`` /
``suv.py`` through thin hooks.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class SlotLayout:
    """Static description of a slot layout on the latent temporal axis.

    Args:
        num_condition_frames: number of shared condition latent frames ``c`` (>= 1).
        slot_frames: number of future latent frames per slot ``m`` (>= 1).
        slot_names: one name per slot, e.g. ``("rgb", "depth", "seg", "instance")``.
            Order is fixed; slot 0 is also the segment condition frames cross-attend to
            by default (recommended: "rgb").
    """

    num_condition_frames: int
    slot_frames: int
    slot_names: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.num_condition_frames < 1:
            raise ValueError(
                f"`num_condition_frames` must be >= 1, got {self.num_condition_frames}"
            )
        if self.slot_frames < 1:
            raise ValueError(f"`slot_frames` must be >= 1, got {self.slot_frames}")
        if len(self.slot_names) < 1:
            raise ValueError("`slot_names` must contain at least one slot name.")
        if len(set(self.slot_names)) != len(self.slot_names):
            raise ValueError(f"`slot_names` must be unique, got {self.slot_names}")
        # frozen dataclass: normalize via object.__setattr__
        object.__setattr__(self, "slot_names", tuple(str(n) for n in self.slot_names))

    # ------------------------------------------------------------------ basic
    @property
    def num_slots(self) -> int:
        return len(self.slot_names)

    @property
    def total_frames(self) -> int:
        """F = c + N * m (latent frames in the packed sequence)."""
        return self.num_condition_frames + self.num_slots * self.slot_frames

    @property
    def virtual_frames(self) -> int:
        """Number of frames of the equivalent single-clip forward (c + m)."""
        return self.num_condition_frames + self.slot_frames

    def slot_index(self, name: str) -> int:
        try:
            return self.slot_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown slot name {name!r}; have {self.slot_names}") from None

    # ----------------------------------------------------------------- ranges
    def slot_frame_range(self, slot: int) -> Tuple[int, int]:
        """[start, end) latent-frame range of `slot` in the packed sequence."""
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range [0, {self.num_slots})")
        start = self.num_condition_frames + slot * self.slot_frames
        return start, start + self.slot_frames

    def slot_token_range(self, slot: int, tokens_per_frame: int) -> Tuple[int, int]:
        """[start, end) token range of `slot` (frames x tokens_per_frame)."""
        start, end = self.slot_frame_range(slot)
        return start * tokens_per_frame, end * tokens_per_frame

    def condition_token_count(self, tokens_per_frame: int) -> int:
        return self.num_condition_frames * tokens_per_frame

    # ----------------------------------------------------------- frame labels
    def virtual_frame_index(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Virtual temporal coordinate per packed frame, shape ``[F]`` (long).

        ``[0..c-1] + [c..c+m-1] * N`` - every slot reuses the same temporal
        RoPE coordinates, so within any slot's receptive field the coordinates
        look exactly like a normal ``c+m``-frame clip.
        """
        cond = torch.arange(self.num_condition_frames, device=device)
        slot = torch.arange(
            self.num_condition_frames,
            self.num_condition_frames + self.slot_frames,
            device=device,
        )
        return torch.cat([cond, slot.repeat(self.num_slots)], dim=0)

    def slot_id_per_frame(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Slot id per packed frame, shape ``[F]`` (long). Condition frames = -1."""
        cond = torch.full((self.num_condition_frames,), -1, dtype=torch.long, device=device)
        slot = torch.arange(self.num_slots, device=device).repeat_interleave(self.slot_frames)
        return torch.cat([cond, slot], dim=0)


# ---------------------------------------------------------------------- masks
def build_slot_causal_frame_mask(
    layout: SlotLayout,
    device: Optional[torch.device] = None,
    mask_mode: str = "first_frame_causal",
) -> torch.Tensor:
    """Frame-level self-attention mask, shape ``[F, F]`` (bool, True = allowed).

    Supported modes mirror ``WanVideoDiT.build_video_to_video_mask``:

    - ``first_frame_causal``: condition frames see only the condition prefix;
      slot frames see the condition prefix + all frames in their own slot.
    - ``per_frame_causal``: same slot isolation, plus virtual-frame causal order
      ``v_key <= v_query`` where ``v`` is the virtual temporal coordinate.

    Consequences:

    - condition frames are blind to all slots;
    - slot frames never see other slots;
    - for ``num_slots == 1`` this reduces exactly to the corresponding
      single-clip mask mode of a ``c+m``-frame clip.
    """
    mask_mode = str(mask_mode)
    sid = layout.slot_id_per_frame(device=device)
    same_slot = sid.unsqueeze(1) == sid.unsqueeze(0)
    key_is_cond = (sid < 0).unsqueeze(0).expand(layout.total_frames, -1)
    slot_isolated = same_slot | key_is_cond
    if mask_mode == "first_frame_causal":
        return slot_isolated
    if mask_mode == "per_frame_causal":
        v = layout.virtual_frame_index(device=device)
        causal = v.unsqueeze(1) >= v.unsqueeze(0)  # [F, F]: v_query >= v_key
        return causal & slot_isolated
    raise ValueError(
        "slot mask_mode must be one of ['first_frame_causal', 'per_frame_causal'], "
        f"got {mask_mode!r}."
    )


def expand_frame_mask_to_tokens(frame_mask: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    """Expand a frame-level ``[F, F]`` mask to token level ``[F*tpf, F*tpf]``."""
    if frame_mask.ndim != 2 or frame_mask.shape[0] != frame_mask.shape[1]:
        raise ValueError(f"`frame_mask` must be square 2D, got {tuple(frame_mask.shape)}")
    if tokens_per_frame < 1:
        raise ValueError(f"`tokens_per_frame` must be >= 1, got {tokens_per_frame}")
    return frame_mask.repeat_interleave(tokens_per_frame, dim=0).repeat_interleave(
        tokens_per_frame, dim=1
    )


def build_slot_text_context_mask(
    layout: SlotLayout,
    tokens_per_frame: int,
    segment_masks: torch.Tensor,
    condition_segment: int = 0,
) -> torch.Tensor:
    """Token -> text cross-attention routing mask, shape ``[B, F*tpf, N*L]``.

    Args:
        segment_masks: per-slot text padding masks, shape ``[B, N, L]`` (bool).
            Segment order must match ``layout.slot_names``; the packed text
            context is assumed to be the concatenation along L in that order.
        condition_segment: which prompt segment the condition frames attend to
            (default 0, i.e. the first slot / "rgb").

    Each slot's tokens attend only to their own prompt segment, so per-slot
    prompts are the sole modality-routing signal (no modality embeddings).
    """
    if segment_masks.ndim != 3:
        raise ValueError(f"`segment_masks` must be [B, N, L], got {tuple(segment_masks.shape)}")
    batch_size, num_segments, seg_len = segment_masks.shape
    if num_segments != layout.num_slots:
        raise ValueError(
            f"`segment_masks` has {num_segments} segments, layout has {layout.num_slots} slots."
        )
    if not 0 <= condition_segment < num_segments:
        raise IndexError(f"`condition_segment` {condition_segment} out of range.")

    seq_len = layout.total_frames * tokens_per_frame
    mask = torch.zeros(
        (batch_size, seq_len, num_segments * seg_len),
        dtype=torch.bool,
        device=segment_masks.device,
    )
    cond_tokens = layout.condition_token_count(tokens_per_frame)
    cond_seg = slice(condition_segment * seg_len, (condition_segment + 1) * seg_len)
    mask[:, :cond_tokens, cond_seg] = segment_masks[:, condition_segment].unsqueeze(1)
    for slot in range(layout.num_slots):
        tok_start, tok_end = layout.slot_token_range(slot, tokens_per_frame)
        seg = slice(slot * seg_len, (slot + 1) * seg_len)
        mask[:, tok_start:tok_end, seg] = segment_masks[:, slot].unsqueeze(1)
    return mask


def tile_action_group_mask(
    virtual_action_mask: torch.Tensor,
    layout: SlotLayout,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Tile a virtual-clip action group mask onto the packed slot sequence.

    Args:
        virtual_action_mask: mask built by ``create_group_causal_attn_mask``
            for the *virtual* clip of ``c+m`` frames, i.e. shape
            ``[(c+m-1) * tpf, action_len]`` - rows are virtual frames
            ``1..c+m-1`` (frame 0 never attends actions, mirroring `pre_dit`).

    Returns:
        Mask of shape ``[(F-1) * tpf, action_len]`` for the packed sequence:
        condition rows (virtual frames ``1..c-1``) appear once, the slot rows
        (virtual frames ``c..c+m-1``) are tiled N times, so every slot's
        frame ``t`` attends to exactly the same action-token group as the
        equivalent single-clip forward.
    """
    expected_rows = (layout.virtual_frames - 1) * tokens_per_frame
    if virtual_action_mask.ndim != 2 or virtual_action_mask.shape[0] != expected_rows:
        raise ValueError(
            "`virtual_action_mask` must have shape [(c+m-1)*tokens_per_frame, action_len] "
            f"= [{expected_rows}, *], got {tuple(virtual_action_mask.shape)}"
        )
    cond_rows = (layout.num_condition_frames - 1) * tokens_per_frame
    cond_part = virtual_action_mask[:cond_rows]
    slot_part = virtual_action_mask[cond_rows:]
    return torch.cat([cond_part] + [slot_part] * layout.num_slots, dim=0)
