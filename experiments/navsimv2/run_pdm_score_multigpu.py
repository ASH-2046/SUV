from __future__ import annotations

import argparse
import inspect
import logging
import os
import pickle
import subprocess
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

_NAVSIM_DEVKIT_ROOT = (
    os.environ.get("NAVSIM_DEVKIT_ROOT")
    or os.environ.get("NAVSIM_V2_DEVKIT_ROOT")
    or str(Path(__file__).resolve().parent / "vendor")
)
if _NAVSIM_DEVKIT_ROOT and (Path(_NAVSIM_DEVKIT_ROOT) / "navsim").is_dir():
    sys.path.insert(0, _NAVSIM_DEVKIT_ROOT)

import numpy as np

np.bool = np.bool_

import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

import navsim
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.planning.script import run_pdm_score_one_stage as official_one_stage


logger = logging.getLogger(__name__)


def _parse_cuda_devices(value: str) -> list[str]:
    devices = [item.strip() for item in str(value).split(",") if item.strip()]
    return devices or ["0"]


def _resolve_navsim_config_dir() -> Path:
    navsim_root = Path(navsim.__file__).resolve().parent
    config_dir = navsim_root / "planning" / "script" / "config" / "pdm_scoring"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"NAVSIM PDM config directory not found: {config_dir}")
    return config_dir


def _build_cfg(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(_resolve_navsim_config_dir()), version_base=None):
        return compose(config_name="default_run_pdm_score", overrides=overrides)


def _discover_data_points(cfg: DictConfig) -> list[dict[str, Any]]:
    if cfg.train_test_split.get("reactive_all_mapping", None) is not None:
        raise ValueError(
            "experiments/navsimv2/run_pdm_score_multigpu.py is only for navtest one-stage scoring. "
            "Use official run_pdm_score.py in a single process for navhard_two_stage."
        )

    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    num_missing = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing > 0:
        logger.warning("Missing metric cache for %d tokens. Skipping these tokens.", num_missing)
    if num_unused > 0:
        logger.warning("Unused metric cache for %d tokens. Skipping these tokens.", num_unused)

    data_points: list[dict[str, Any]] = []
    for log_file, tokens in scene_loader.get_tokens_list_per_log().items():
        filtered_tokens = [token for token in tokens if token in tokens_to_evaluate]
        if filtered_tokens:
            data_points.append({"log_file": log_file, "tokens": filtered_tokens})

    logger.info(
        "Discovered %d one-stage PDM scenarios across %d logs.",
        len(tokens_to_evaluate),
        len(data_points),
    )
    return data_points


def _split_data_points_by_token(data_points: list[dict[str, Any]], world_size: int) -> list[list[dict[str, Any]]]:
    shards_by_log: list[dict[str, dict[str, Any]]] = [dict() for _ in range(world_size)]
    flat_points: list[tuple[str, str]] = [
        (str(point["log_file"]), str(token))
        for point in data_points
        for token in point["tokens"]
    ]

    for index, (log_file, token) in enumerate(flat_points):
        rank = index % world_size
        log_bucket = shards_by_log[rank].setdefault(log_file, {"log_file": log_file, "tokens": []})
        log_bucket["tokens"].append(token)

    return [list(shard.values()) for shard in shards_by_log]


def _run_official_one_stage_worker(cfg: DictConfig, data_points: list[dict[str, Any]]) -> pd.DataFrame:
    if not data_points:
        return pd.DataFrame(columns=["token", "valid"])

    score_args = [
        {
            "cfg": cfg,
            "log_file": point["log_file"],
            "tokens": point["tokens"],
        }
        for point in data_points
    ]
    score_rows = official_one_stage.run_pdm_score(score_args)
    if not score_rows:
        return pd.DataFrame(columns=["token", "valid"])
    return pd.concat(score_rows, ignore_index=True)


def _call_infer_ste_scene_aggregators(
    pdm_score_df: pd.DataFrame,
    proposal_sampling: Any,
) -> pd.DataFrame:
    infer_ste = getattr(official_one_stage, "infer_ste_scene_aggregators", None)
    if infer_ste is None:
        start_adjacent_mapping = official_one_stage.infer_start_adjacent_mapping(pdm_score_df)
        return official_one_stage.create_scene_aggregators(
            start_adjacent_mapping,
            pdm_score_df,
            proposal_sampling,
        )

    signature = inspect.signature(infer_ste)
    if len(signature.parameters) == 2:
        return infer_ste(pdm_score_df, proposal_sampling)
    if len(signature.parameters) == 1:
        return infer_ste(pdm_score_df)

    start_adjacent_mapping = official_one_stage.infer_start_adjacent_mapping(pdm_score_df)
    return infer_ste(start_adjacent_mapping, pdm_score_df, proposal_sampling)


def _score_columns(pdm_score_df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]


def _finalize_official_one_stage_scores(
    cfg: DictConfig, score_rows: list[pd.DataFrame]
) -> tuple[float, Path]:
    if not score_rows:
        raise RuntimeError("No PDM score shards were produced.")

    pdm_score_df = pd.concat(score_rows, ignore_index=True)
    proposal_sampling = instantiate(cfg.simulator.proposal_sampling)
    pdm_score_df = _call_infer_ste_scene_aggregators(pdm_score_df, proposal_sampling)
    pdm_score_df = official_one_stage.compute_final_scores(pdm_score_df)

    num_successful = int(pdm_score_df["valid"].sum())
    num_failed = int(len(pdm_score_df) - num_successful)
    failed_tokens = pdm_score_df.loc[~pdm_score_df["valid"], "token"].to_list() if num_failed > 0 else []

    score_cols = _score_columns(pdm_score_df)
    average_row = pdm_score_df[score_cols].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = pdm_score_df["valid"].all()

    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    final_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(final_path)

    average_score = pdm_score_df.loc[pdm_score_df["token"] == "average_all_frames", "score"].iloc[0]
    logger.info(
        "Finished multi-GPU one-stage evaluation. successful=%d failed=%d average_all_frames=%s results=%s",
        num_successful,
        num_failed,
        average_score,
        final_path,
    )
    if num_failed > 0:
        logger.info("Failed tokens: %s", failed_tokens)
    return float(average_score), final_path


def _run_parent(args: argparse.Namespace) -> None:
    devices = _parse_cuda_devices(args.cuda_visible_devices)
    if args.num_gpus > 0:
        devices = devices[: args.num_gpus]
    world_size = len(devices)
    if world_size < 1:
        raise ValueError("No CUDA devices selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(args.overrides)
    data_points = _discover_data_points(cfg)
    shards = _split_data_points_by_token(data_points, world_size)
    data_points_path = output_dir / "_data_points_by_rank.pkl"
    with open(data_points_path, "wb") as f:
        pickle.dump(shards, f)

    logger.info(
        "Starting NAVSIM v2 one-stage multi-GPU PDM eval: world_size=%d tokens=%d logs=%d",
        world_size,
        sum(len(point["tokens"]) for point in data_points),
        len(data_points),
    )
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for rank, device in enumerate(devices):
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--child",
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--data-points-file",
            str(data_points_path),
            "--output-dir",
            str(output_dir),
            "--overrides",
            *args.overrides,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["PDM_RANK"] = str(rank)
        env["PDM_WORLD_SIZE"] = str(world_size)
        processes.append((rank, subprocess.Popen(cmd, env=env, text=True)))

    failed = False
    for rank, process in processes:
        return_code = process.wait()
        if return_code != 0:
            logger.error("PDM rank %d failed with return code %d", rank, return_code)
            failed = True
    if failed:
        raise RuntimeError("At least one PDM worker failed.")

    shard_paths = [output_dir / f"_rank_{rank:03d}.pkl" for rank in range(world_size)]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PDM shard files: {missing}")
    score_rows = [pd.read_pickle(path) for path in shard_paths]
    model_score, model_path = _finalize_official_one_stage_scores(cfg, score_rows)

    for path in shard_paths:
        path.unlink(missing_ok=True)
    data_points_path.unlink(missing_ok=True)

    if not getattr(official_one_stage, "EVALUATE_HUMAN_GT", False):
        logger.info("Skipping NAVSIM v2 navtest GT evaluation.")
        return

    gt_score, gt_path = official_one_stage.run_human_gt_evaluation(cfg)
    logger.info(
        "NAVSIM v2 navtest model/GT comparison: model_score=%.12f gt_score=%.12f model_csv=%s gt_csv=%s",
        model_score,
        gt_score,
        model_path,
        gt_path,
    )


def _run_child(args: argparse.Namespace) -> None:
    rank = int(args.rank)
    world_size = int(args.world_size)
    cfg = _build_cfg(args.overrides)
    with open(args.data_points_file, "rb") as f:
        shards = pickle.load(f)
    if len(shards) != world_size:
        raise ValueError(f"Expected {world_size} shards, found {len(shards)}")

    my_data_points = shards[rank]
    logger.info(
        "[rank=%d] evaluating %d logs / %d tokens",
        rank,
        len(my_data_points),
        sum(len(point["tokens"]) for point in my_data_points),
    )
    if rank != 0 and cfg.agent.get("eval_visualize", None) is not None:
        cfg.agent.eval_visualize = False

    df = _run_official_one_stage_worker(cfg, my_data_points)
    shard_path = Path(args.output_dir) / f"_rank_{rank:03d}.pkl"
    df.to_pickle(shard_path)
    logger.info("[rank=%d] saved pickle shard: %s", rank, shard_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU NAVSIM v2 official one-stage PDM score runner.")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--data-points-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cuda-visible-devices", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rank_prefix = f"[rank={args.rank}] " if args.child else ""
    logging.basicConfig(
        level=logging.INFO,
        format=rank_prefix + "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Importing navsim from: %s", Path(navsim.__file__).resolve().parent)
    if args.child:
        _run_child(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
