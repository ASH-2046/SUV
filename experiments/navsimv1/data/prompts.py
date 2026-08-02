from __future__ import annotations

from typing import List, Optional

import torch


DEFAULT_NAVSIM_PROMPT = (
    "A high-quality, photorealistic ego-centric driving video captured by "
    "a camera rigidly mounted on the ego vehicle, always facing forward."
)

STAGEA_RGB_PROMPT = (
    "Generate a natural photorealistic RGB camera video from the forward-facing "
    "ego vehicle camera. Preserve realistic lighting, textures, colors, object "
    "appearance, and camera imaging. Do not generate depth maps, segmentation "
    "masks, instance colors, labels, overlays, or false-color colormaps."
)

STAGEA_DENSE_PROMPTS = {
    "rgb": STAGEA_RGB_PROMPT,
    "depth": (
        "Generate a relative-depth RGB visualization video for the future "
        "ego-centric driving scene. Use the dense-label turbo colormap with "
        "clip-level normalization, consistently across all future frames."
    ),
    "seg": (
        "Generate a semantic segmentation RGB visualization video for the "
        "future ego-centric driving scene. Use this exact RGB palette: "
        "background=(0,0,0), road=(96,96,96), car=(0,0,142), "
        "truck=(0,0,70), bus=(0,60,100), pedestrian=(220,20,60), "
        "bicycle=(119,11,32), motorcycle=(0,0,230), "
        "traffic light=(250,170,30), traffic sign=(220,220,0), "
        "traffic cone=(255,80,0), barrier=(102,102,156)."
    ),
    "instance": (
        "Generate an instance segmentation RGB visualization video for the "
        "future ego-centric driving scene. Track only cars, trucks, buses, "
        "pedestrians, bicycles, and motorcycles. Use black background and "
        "temporally stable colors for each object instance across frames."
    ),
}

STAGEA_DENSE_FUTURE_INSTRUCTIONS = {
    "depth": (
        "predict and generate the next {future_seconds:.1f} seconds as a "
        "relative-depth RGB visualization video, not a photorealistic RGB video."
    ),
    "seg": (
        "predict and generate the next {future_seconds:.1f} seconds as a "
        "semantic segmentation RGB visualization video, not a photorealistic RGB video."
    ),
    "instance": (
        "predict and generate the next {future_seconds:.1f} seconds as an "
        "instance segmentation RGB visualization video, not a photorealistic RGB video."
    ),
}

STAGEA_DENSE_QUALITY_INSTRUCTIONS = {
    "depth": (
        "Keep the depth colormap stable across time, preserve object boundaries, "
        "and encode near and far scene geometry consistently."
    ),
    "seg": (
        "Keep class colors stable across time, preserve object and lane boundaries, "
        "and leave unlabeled background regions black."
    ),
    "instance": (
        "Keep instance colors stable across time, preserve object boundaries, "
        "and assign separate colors to separate tracked objects."
    ),
}

STAGEA_DENSE_TARGET_KEYS = {
    "rgb": "future_video_front",
    "depth": "future_depth_rgb",
    "seg": "future_seg_rgb",
    "instance": "future_instance_rgb",
}

_DEFAULT_NAV_HINTS = {
    "turn left": "the route ahead goes left",
    "go straight": "the route continues straight",
    "turn right": "the route ahead goes right",
    "follow the lane": "the route follows the current lane",
}
_DEFAULT_COMMAND_LABELS = ("turn left", "go straight", "turn right", "follow the lane")


def _get_nav_hint(command_vector: torch.Tensor) -> str:
    if command_vector.numel() == 0 or float(command_vector.abs().sum().item()) < 1e-6:
        return _DEFAULT_NAV_HINTS["follow the lane"]
    idx = int(torch.argmax(command_vector).item())
    if 0 <= idx < len(_DEFAULT_COMMAND_LABELS):
        return _DEFAULT_NAV_HINTS[_DEFAULT_COMMAND_LABELS[idx]]
    return _DEFAULT_NAV_HINTS["follow the lane"]


def _quantize(value: float, step: float) -> float:
    step = float(step)
    if step <= 0:
        return float(value)
    return round(float(value) / step) * step


def build_navsim_prompts(
    prompt_prefix: str,
    ego_status: Optional[torch.Tensor],
    batch_size: int,
    history_seconds: float,
    future_seconds: float,
    mode: str = "static",
    future_instruction: Optional[str] = None,
    quality_instruction: Optional[str] = None,
    velocity_quantization: float = 0.5,
    acceleration_quantization: float = 0.5,
) -> List[str]:
    """Build the static/dynamic NAVSIM prompt schema used by Drive-JEPA CosmosAction."""
    mode = str(mode).strip().lower()
    if mode == "static" or ego_status is None:
        return [prompt_prefix] * int(batch_size)
    if mode != "dynamic":
        raise ValueError(f"Unknown NAVSIM prompt mode: {mode!r}")

    if ego_status.ndim == 2:
        ego_status = ego_status.unsqueeze(0)
    if ego_status.ndim != 3 or ego_status.shape[0] != batch_size or ego_status.shape[-1] < 11:
        return [prompt_prefix] * int(batch_size)

    future_instruction = future_instruction or (
        "predict and generate the next {future_seconds:.1f} seconds of realistic driving continuation."
    )
    quality_instruction = quality_instruction or (
        "Maintain temporal consistency, natural motion flow, clear details, and realistic physics."
    )

    prompts: List[str] = []
    for sample in ego_status.detach().float().cpu():
        latest = sample[-1]
        vx = _quantize(float(latest[3].item()), velocity_quantization)
        vy = _quantize(float(latest[4].item()), velocity_quantization)
        ax = _quantize(float(latest[5].item()), acceleration_quantization)
        ay = _quantize(float(latest[6].item()), acceleration_quantization)
        nav_hint = _get_nav_hint(latest[7:])
        prompts.append(
            f"{prompt_prefix} "
            f"Current ego state: velocity ({vx:.1f}, {vy:.1f}) m/s, "
            f"acceleration ({ax:.1f}, {ay:.1f}) m/s². "
            f"High-level navigation: {nav_hint}. "
            f"Based on the past {history_seconds:.1f} seconds of driving footage, "
            f"{future_instruction.format(history_seconds=history_seconds, future_seconds=future_seconds)} "
            f"The camera is rigidly attached to the moving vehicle and always faces forward. "
            f"{quality_instruction.format(history_seconds=history_seconds, future_seconds=future_seconds)}"
        )
    return prompts


def normalize_stagea_modalities(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return ["rgb"]
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    else:
        raw_items = [str(item) for item in value]

    modalities: list[str] = []
    aliases = {
        "video": "rgb",
        "image": "rgb",
        "semantic": "seg",
        "segmentation": "seg",
        "inst": "instance",
        "instances": "instance",
    }
    for item in raw_items:
        modality = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if not modality:
            continue
        if modality not in STAGEA_DENSE_TARGET_KEYS:
            raise ValueError(
                "stagea_target_modalities must contain only "
                f"{sorted(STAGEA_DENSE_TARGET_KEYS)}, got {item!r}."
            )
        if modality not in modalities:
            modalities.append(modality)
    if not modalities:
        raise ValueError("stagea_target_modalities resolved to an empty list.")
    return modalities


def stagea_prompt_overrides(modality: str) -> tuple[str, Optional[str], Optional[str]]:
    modality = normalize_stagea_modalities([modality])[0]
    return (
        STAGEA_DENSE_PROMPTS[modality],
        STAGEA_DENSE_FUTURE_INSTRUCTIONS.get(modality),
        STAGEA_DENSE_QUALITY_INSTRUCTIONS.get(modality),
    )
