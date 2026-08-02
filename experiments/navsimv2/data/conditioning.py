from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VisualConditioningConfig:
    name: str
    num_history_frames: int
    video_frame_mode: str
    history_padding_mode: str
    cache_history_policy: str


_ALIASES = {
    "single": "current",
    "single_frame": "current",
    "current_frame": "current",
    "history4": "history_4",
    "4_history": "history_4",
    "history5": "history_5",
    "5_history": "history_5",
}

_PRESETS = {
    "current": VisualConditioningConfig(
        name="current",
        num_history_frames=4,
        video_frame_mode="current_plus_future",
        history_padding_mode="none",
        cache_history_policy="tail",
    ),
    "history_4": VisualConditioningConfig(
        name="history_4",
        num_history_frames=4,
        video_frame_mode="history_plus_future",
        history_padding_mode="repeat_first",
        cache_history_policy="tail",
    ),
    "history_5": VisualConditioningConfig(
        name="history_5",
        num_history_frames=5,
        video_frame_mode="history_plus_future",
        history_padding_mode="none",
        cache_history_policy="tail",
    ),
}


def resolve_visual_conditioning(
    visual_conditioning: str = "history_4",
    *,
    num_history_frames: Optional[int] = None,
    video_frame_mode: Optional[str] = None,
    history_padding_mode: Optional[str] = None,
    cache_history_policy: Optional[str] = None,
) -> VisualConditioningConfig:
    name = str(visual_conditioning).strip().lower()
    name = _ALIASES.get(name, name)
    if name not in _PRESETS:
        raise ValueError(
            "visual_conditioning must be one of "
            f"{sorted(_PRESETS.keys())}, got {visual_conditioning!r}."
        )
    preset = _PRESETS[name]
    return VisualConditioningConfig(
        name=name,
        num_history_frames=(
            preset.num_history_frames if num_history_frames is None else int(num_history_frames)
        ),
        video_frame_mode=(
            preset.video_frame_mode
            if video_frame_mode is None
            else str(video_frame_mode).strip().lower()
        ),
        history_padding_mode=(
            preset.history_padding_mode
            if history_padding_mode is None
            else str(history_padding_mode).strip().lower()
        ),
        cache_history_policy=(
            preset.cache_history_policy
            if cache_history_policy is None
            else str(cache_history_policy).strip().lower()
        ),
    )
