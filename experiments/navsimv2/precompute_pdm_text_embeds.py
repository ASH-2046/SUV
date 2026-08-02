#!/usr/bin/env python3
"""Precompute SUV text embeddings for official NAVSIM v2 PDM evaluation splits."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

logger = logging.getLogger(__name__)

DEFAULT_NAVSIM_PROMPT = (
    "A high-quality, photorealistic ego-centric driving video captured by "
    "a camera rigidly mounted on the ego vehicle, always facing forward."
)
DEFAULT_MODEL_ID = "Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan2.2-TI2V-5B"
DEFAULT_CONTEXT_LEN = 512
DEFAULT_BATCH_SIZE = 16

BASE_ACTION_CONTEXT_MODES = {
    "base",
    "shared",
    "default",
    "video_prompt",
}
SLOT_JOINT_MODALITIES = ("rgb", "depth", "seg", "instance")


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _optional_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return float(text)


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _init_distributed() -> tuple[bool, int, int, int]:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _atomic_torch_save(payload: dict[str, Any], output_path: Path) -> None:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _resolve_navsim_config_dir() -> Path:
    import navsim

    return Path(navsim.__file__).resolve().parent / "planning" / "script" / "config" / "pdm_scoring"


def _compose_navsim_cfg(args: argparse.Namespace):
    import hydra

    config_dir = _resolve_navsim_config_dir()
    overrides = [
        f"train_test_split={args.train_test_split}",
        f"train_test_split.scene_filter.num_history_frames={args.num_history_frames}",
        f"train_test_split.scene_filter.num_future_frames={args.num_future_frames}",
        f"train_test_split.scene_filter.frame_interval={args.frame_interval}",
        f"navsim_log_path={args.navsim_log_path}",
        f"++original_sensor_path={args.original_sensor_path}",
    ]
    if args.train_test_split == "navhard_two_stage":
        overrides.extend(
            [
                f"++synthetic_sensor_path={args.synthetic_sensor_path}",
                f"++synthetic_scenes_path={args.synthetic_scenes_path}",
            ]
        )

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return hydra.compose(config_name="default_run_pdm_score", overrides=overrides)


def _build_scene_loader(args: argparse.Namespace):
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    cfg = _compose_navsim_cfg(args)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    return SceneLoader(
        synthetic_sensor_path=Path(args.synthetic_sensor_path) if args.train_test_split == "navhard_two_stage" else None,
        original_sensor_path=Path(args.original_sensor_path),
        data_path=Path(args.navsim_log_path),
        synthetic_scenes_path=(
            Path(args.synthetic_scenes_path) if args.train_test_split == "navhard_two_stage" else None
        ),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )


def _tokens_for_split(scene_loader, train_test_split: str) -> list[str]:
    if train_test_split == "navhard_two_stage":
        stage_two_tokens = scene_loader.reactive_tokens_stage_two or []
        return list(dict.fromkeys(list(scene_loader.tokens_stage_one) + list(stage_two_tokens)))
    return list(scene_loader.tokens)


def _prompt_components(args: argparse.Namespace) -> tuple[str, str | None, str | None]:
    from experiments.navsimv2.data.prompts import action_prompt_components

    action_context_mode = str(args.action_context_mode).strip().lower()
    if action_context_mode in BASE_ACTION_CONTEXT_MODES:
        return args.prompt, _optional_text(args.prompt_future_instruction), _optional_text(args.prompt_quality_instruction)
    return action_prompt_components(
        action_context_mode,
        base_prompt=args.prompt,
        custom_prompt=_optional_text(args.action_prompt),
        future_instruction=_optional_text(args.action_prompt_future_instruction),
        quality_instruction=_optional_text(args.action_prompt_quality_instruction),
    )


def _build_prompt_from_agent_input(
    agent_input,
    args: argparse.Namespace,
    *,
    stagea_modality: str | None = None,
) -> str:
    from experiments.navsimv2.data.features import NavsimV2FeatureBuilder
    from experiments.navsimv2.data.prompts import build_navsim_prompts, stagea_prompt_overrides

    ego_status = NavsimV2FeatureBuilder._ego_status_tensor(agent_input)
    if stagea_modality is None:
        prompt_prefix, future_instruction, quality_instruction = _prompt_components(args)
    else:
        prompt_prefix, future_instruction, quality_instruction = stagea_prompt_overrides(stagea_modality)
    history_seconds = (
        float(ego_status.shape[0]) / float(args.fps)
        if args.prompt_history_seconds is None
        else float(args.prompt_history_seconds)
    )
    return build_navsim_prompts(
        prompt_prefix=prompt_prefix,
        ego_status=ego_status.unsqueeze(0),
        batch_size=1,
        history_seconds=history_seconds,
        future_seconds=float(args.num_future_frames) / float(args.fps),
        mode=args.prompt_mode,
        future_instruction=future_instruction,
        quality_instruction=quality_instruction,
        velocity_quantization=args.prompt_velocity_quantization,
        acceleration_quantization=args.prompt_acceleration_quantization,
    )[0]


def _collect_unique_prompts(args: argparse.Namespace) -> tuple[list[str], dict[str, int]]:
    scene_loader = _build_scene_loader(args)
    tokens = _tokens_for_split(scene_loader, args.train_test_split)
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]

    prompts: list[str] = []
    seen: set[str] = set()
    stage_one_count = len(getattr(scene_loader, "tokens_stage_one", []))
    stage_two_count = len(scene_loader.reactive_tokens_stage_two or []) if args.train_test_split == "navhard_two_stage" else 0

    for token in tqdm(tokens, desc="Scanning PDM prompts", dynamic_ncols=True):
        agent_input = scene_loader.get_agent_input_from_token(token)
        if args.slot_inference:
            sample_prompts = [
                _build_prompt_from_agent_input(agent_input, args, stagea_modality=modality)
                for modality in SLOT_JOINT_MODALITIES
            ]
        else:
            sample_prompts = [_build_prompt_from_agent_input(agent_input, args)]
        for prompt in sample_prompts:
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)

    stats = {
        "tokens": len(tokens),
        "stage_one_tokens": stage_one_count,
        "stage_two_reactive_tokens": stage_two_count,
        "unique_prompts": len(prompts),
    }
    return prompts, stats


def _cache_path(cache_dir: Path, prompt: str, context_len: int, enc_id: str) -> Path:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"


def _print_prompt_cache_stats(prompts: list[str], cache_dir: Path, context_len: int, enc_id: str) -> tuple[int, int]:
    existing = sum(1 for prompt in prompts if _cache_path(cache_dir, prompt, context_len, enc_id).exists())
    missing = len(prompts) - existing
    print(f"text embedding cache: {cache_dir}")
    print(f"unique prompts:       {len(prompts)}")
    print(f"already cached:       {existing}")
    print(f"missing:              {missing}")
    return existing, missing


def _encode_missing_prompts(prompts: list[str], args: argparse.Namespace, *, rank: int, world_size: int, local_rank: int) -> None:
    import torch

    from suv.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from suv.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    cache_dir = Path(args.text_embedding_cache_dir).expanduser()
    context_len = int(args.context_len)
    enc_id = _model_id_to_enc_id(args.model_id)

    prompts_to_encode: list[str] = []
    skipped = 0
    for prompt in prompts:
        cache_path = _cache_path(cache_dir, prompt, context_len, enc_id)
        if cache_path.exists() and not args.overwrite:
            skipped += 1
            continue
        prompts_to_encode.append(prompt)

    local_prompts = prompts_to_encode[rank::world_size]
    if rank == 0:
        logger.info(
            "Encoding %d / %d prompts, skipped=%d, overwrite=%s.",
            len(prompts_to_encode),
            len(prompts),
            skipped,
            args.overwrite,
        )

    if not local_prompts:
        return

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=args.redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    with tqdm(
        total=len(local_prompts),
        desc=f"Encoding prompts rank {rank}/{world_size}",
        unit="prompt",
        dynamic_ncols=True,
        disable=rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(local_prompts), int(args.batch_size)):
                batch_prompts = local_prompts[start : start + int(args.batch_size)]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                context = text_encoder(ids, mask)
                for idx, prompt in enumerate(batch_prompts):
                    payload = {
                        "context": context[idx].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                        "mask": mask[idx].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                    }
                    _atomic_torch_save(payload, _cache_path(cache_dir, prompt, context_len, enc_id))
                pbar.update(len(batch_prompts))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute text embeddings for SUV official NAVSIM v2 PDM evaluation."
    )
    parser.add_argument("--train-test-split", choices=["navtest", "navhard_two_stage"], required=True)
    parser.add_argument("--navsim-log-path", required=True)
    parser.add_argument("--original-sensor-path", required=True)
    parser.add_argument("--synthetic-sensor-path", default="")
    parser.add_argument("--synthetic-scenes-path", default="")
    parser.add_argument("--text-embedding-cache-dir", required=True)
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=8)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--prompt", default=DEFAULT_NAVSIM_PROMPT)
    parser.add_argument("--prompt-mode", default="dynamic")
    parser.add_argument("--prompt-future-instruction", default=None)
    parser.add_argument("--prompt-quality-instruction", default=None)
    parser.add_argument("--prompt-history-seconds", type=_optional_float, default=2.5)
    parser.add_argument("--prompt-velocity-quantization", type=float, default=0.5)
    parser.add_argument("--prompt-acceleration-quantization", type=float, default=0.5)
    parser.add_argument("--action-context-mode", default="base")
    parser.add_argument("--action-prompt", default=None)
    parser.add_argument("--action-prompt-future-instruction", default=None)
    parser.add_argument("--action-prompt-quality-instruction", default=None)
    parser.add_argument("--slot-inference", type=_str_to_bool, default=False)
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model-id", default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", type=_str_to_bool, default=False)
    parser.add_argument("--redirect-common-files", type=_str_to_bool, default=False)
    parser.add_argument("--dry-run", action="store_true", help="Only collect prompts and report missing caches.")
    parser.add_argument("--validate-only", action="store_true", help="Exit non-zero if any prompt cache is missing.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Debug helper to scan only the first N tokens.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    from suv.utils.logging_config import setup_logging

    setup_logging(log_level=logging.INFO)
    if args.train_test_split == "navhard_two_stage":
        if not args.synthetic_scenes_path:
            raise SystemExit("--synthetic-scenes-path is required for navhard_two_stage")
        if not args.synthetic_sensor_path:
            raise SystemExit("--synthetic-sensor-path is required for navhard_two_stage")

    import navsim

    logger.info("Importing navsim from: %s", Path(navsim.__file__).resolve().parent)
    logger.info("PDM text embedding split: %s", args.train_test_split)

    prompts, scan_stats = _collect_unique_prompts(args)
    cache_dir = Path(args.text_embedding_cache_dir).expanduser()
    enc_id = _model_id_to_enc_id(args.model_id)

    print("PDM prompt scan:")
    for key, value in scan_stats.items():
        print(f"  {key}: {value}")
    _, missing = _print_prompt_cache_stats(prompts, cache_dir, int(args.context_len), enc_id)
    if args.validate_only and missing > 0:
        return 2

    if args.dry_run or args.validate_only:
        return 0

    is_distributed, rank, world_size, local_rank = _init_distributed()
    _encode_missing_prompts(prompts, args, rank=rank, world_size=world_size, local_rank=local_rank)
    if is_distributed:
        import torch.distributed as dist

        dist.barrier()

    if rank == 0:
        _, missing = _print_prompt_cache_stats(prompts, cache_dir, int(args.context_len), enc_id)
        if missing > 0:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
