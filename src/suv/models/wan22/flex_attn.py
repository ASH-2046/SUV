"""FlexAttention path for slot/causal masked self-attention (opt-in).

Why this exists
---------------
The slot attention mask is block-sparse: each slot future attends only to the
shared condition prefix and to its own slot (slots never attend to each other),
so the intended cost is ~O(N * (c + m)^2), not O((c + N*m)^2). But the existing
code hands a dense ``(S, S)`` boolean mask to ``F.scaled_dot_product_attention``.
A boolean ``attn_mask`` forces SDPA off the flash kernel onto the math backend,
which materializes the full ``(B, H, S, S)`` score matrix and only then zeroes
the forbidden (e.g. cross-slot) blocks -- it computes ~4x too much for a 4-slot
layout and keeps the corresponding activations.

This module converts that *same* dense mask into a FlexAttention ``BlockMask``.
``create_block_mask`` analyses the mask at block granularity and marks fully
masked blocks as skippable, so ``flex_attention`` never computes or stores them.
The mask values are identical to the dense path, so the result is numerically
equivalent (validate with ``scripts/validate_flex_attention.py``).

Safety / scope
--------------
* Opt-in: only active when ``SUV_ATTN_IMPL=flex``. Default is ``dense`` ->
  zero behaviour change.
* Only square 2D boolean masks (self-attention: video->video and the MoT mixed
  attention) are routed to FlexAttention. Cross-attention key-padding masks
  (4D) keep using SDPA.
* If FlexAttention is unavailable or any call fails (e.g. ``torch.compile`` not
  cooperating with DeepSpeed / gradient checkpointing), it falls back to the
  dense / eager path automatically.
"""

from __future__ import annotations

import os
import weakref
from typing import Optional

import torch

_AVAILABLE: Optional[bool] = None
_create_block_mask = None
_flex_attention_fn = None  # may be torch.compile'd


def attn_impl() -> str:
    return os.environ.get("SUV_ATTN_IMPL", "dense").strip().lower()


def _compile_enabled() -> bool:
    return os.environ.get("SUV_FLEX_COMPILE", "1").strip().lower() in {"1", "true", "yes"}


def _ensure_loaded() -> bool:
    global _AVAILABLE, _create_block_mask, _flex_attention_fn
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception:
        _AVAILABLE = False
        return False
    _create_block_mask = create_block_mask
    fn = flex_attention
    if _compile_enabled():
        try:
            fn = torch.compile(flex_attention, dynamic=False)
        except Exception:
            fn = flex_attention
    _flex_attention_fn = fn
    _AVAILABLE = True
    return True


def _is_self_attn_mask(ctx_mask) -> bool:
    return (
        isinstance(ctx_mask, torch.Tensor)
        and ctx_mask.dtype == torch.bool
        and ctx_mask.dim() == 2
        and ctx_mask.shape[0] == ctx_mask.shape[1]
    )


def use_flex(ctx_mask) -> bool:
    """True if this mask should be routed through FlexAttention."""
    return attn_impl() == "flex" and _is_self_attn_mask(ctx_mask) and _ensure_loaded()


# Cache the BlockMask per dense-mask object so it is built once per forward and
# reused across all transformer layers (the same mask tensor is passed to every
# layer). Keyed by id() with a weakref identity guard to avoid id-reuse hazards;
# capped so it never grows unbounded across forwards.
_CACHE: dict = {}
_CACHE_CAP = 8


def _block_mask_from_dense(dense_mask: torch.Tensor):
    key = id(dense_mask)
    entry = _CACHE.get(key)
    if entry is not None and entry[0]() is dense_mask:
        return entry[1]

    seq = int(dense_mask.shape[0])

    def mask_mod(b, h, q_idx, kv_idx):  # True == attend, matching SDPA bool mask
        return dense_mask[q_idx, kv_idx]

    block_mask = _create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seq,
        KV_LEN=seq,
        device=dense_mask.device,
    )
    _CACHE[key] = (weakref.ref(dense_mask), block_mask)
    while len(_CACHE) > _CACHE_CAP:
        _CACHE.pop(next(iter(_CACHE)))
    return block_mask


def flex_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
    """FlexAttention counterpart of ``F.scaled_dot_product_attention``.

    Args:
        q, k, v: ``[B, H, S, D]`` (same layout SDPA receives in compatibility mode).
        ctx_mask: ``[S, S]`` boolean self-attention mask, ``True`` == attend.
    Returns:
        ``[B, H, S, D]`` attention output.
    """
    global _flex_attention_fn
    block_mask = _block_mask_from_dense(ctx_mask)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    try:
        return _flex_attention_fn(q, k, v, block_mask=block_mask)
    except Exception:
        # Compiled path unhappy (e.g. under checkpoint/DeepSpeed) -> eager flex.
        from torch.nn.attention.flex_attention import flex_attention as _eager

        _flex_attention_fn = _eager
        return _eager(q, k, v, block_mask=block_mask)
