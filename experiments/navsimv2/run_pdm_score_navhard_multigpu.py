from __future__ import annotations

# ruff: noqa: E402

import argparse
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
from experiments.navsimv2.pdm_multigpu_sharding import split_data_points_by_log
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.planning.script import run_pdm_score as official_two_stage


logger = logging.getLogger(__name__)


def _parse_cuda_devices(value: str) -> list[str]:
    devices = [item.strip() for item in str(value).split(",") if item.strip()]
    devices = devices or ["0"]
    if len(devices) != len(set(devices)):
        raise ValueError(f"CUDA device list contains duplicates: {devices}")
    return devices


def _resolve_navsim_config_dir() -> Path:
    navsim_root = Path(navsim.__file__).resolve().parent
    config_dir = navsim_root / "planning" / "script" / "config" / "pdm_scoring"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"NAVSIM PDM config directory not found: {config_dir}")
    return config_dir


def _build_cfg(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(_resolve_navsim_config_dir()), version_base=None
    ):
        return compose(config_name="default_run_pdm_score", overrides=overrides)


def _discover_data_points(
    cfg: DictConfig,
) -> tuple[list[dict[str, Any]], set[str]]:
    if cfg.train_test_split.get("reactive_all_mapping", None) is None:
        raise ValueError(
            "run_pdm_score_navhard_multigpu.py requires a two-stage split with "
            "train_test_split.reactive_all_mapping."
        )

    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_tokens = set(scene_loader.tokens)
    stage_two_tokens = set(scene_loader.reactive_tokens_stage_two or [])
    scorable_scene_tokens = set(scene_loader.tokens_stage_one) | stage_two_tokens
    metric_tokens = set(metric_cache_loader.tokens)
    tokens_to_evaluate = scorable_scene_tokens & metric_tokens
    num_missing = len(scene_tokens - metric_tokens)
    num_unused = len(metric_tokens - scene_tokens)
    if num_missing > 0:
        logger.warning(
            "Missing metric cache for %d tokens. Skipping these tokens.", num_missing
        )
    if num_unused > 0:
        logger.warning(
            "Unused metric cache for %d tokens. Skipping these tokens.", num_unused
        )

    # Preserve the official scorer's complete per-log token lists. Filtering
    # the lists before worker startup can remove an original scene that is
    # still needed to discover its associated synthetic scenes.
    data_points: list[dict[str, Any]] = []
    for log_file, tokens in scene_loader.get_tokens_list_per_log().items():
        num_evaluable = sum(token in tokens_to_evaluate for token in tokens)
        if num_evaluable > 0:
            data_points.append(
                {
                    "log_file": str(log_file),
                    "tokens": list(tokens),
                    "num_evaluable": int(num_evaluable),
                }
            )

    logger.info(
        "Discovered %d navhard scenarios across %d logs.",
        len(tokens_to_evaluate),
        len(data_points),
    )
    return data_points, scene_tokens


def _run_official_two_stage_worker(
    cfg: DictConfig,
    data_points: list[dict[str, Any]],
) -> pd.DataFrame:
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
    score_rows = official_two_stage.run_pdm_score(score_args)
    if not score_rows:
        return pd.DataFrame(columns=["token", "valid"])
    return pd.concat(score_rows, ignore_index=True)


def _active_mappings(
    cfg: DictConfig,
    scene_tokens: set[str],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    all_mappings: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (
        orig_token,
        prev_token,
        two_stage_pairs,
    ) in cfg.train_test_split.reactive_all_mapping:
        orig_token = str(orig_token)
        prev_token = str(prev_token)
        if prev_token in scene_tokens or orig_token in scene_tokens:
            all_mappings[(orig_token, prev_token)] = [
                (str(pair[0]), str(pair[1])) for pair in two_stage_pairs
            ]
    return all_mappings


def _score_columns(pdm_score_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in pdm_score_df.columns
        if (
            (
                any(score.name in column for score in fields(PDMResults))
                or column == "two_frame_extended_comfort"
                or column == "score"
            )
            and column != "pdm_score"
        )
    ]


def _finalize_official_two_stage_scores(
    cfg: DictConfig,
    score_rows: list[pd.DataFrame],
    scene_tokens: set[str],
) -> tuple[float, Path]:
    nonempty_rows = [frame for frame in score_rows if not frame.empty]
    if not nonempty_rows:
        raise RuntimeError("No navhard PDM score shards were produced.")

    pdm_score_df = pd.concat(nonempty_rows, ignore_index=True)
    duplicate_tokens = sorted(
        pdm_score_df.loc[pdm_score_df["token"].duplicated(), "token"]
        .astype(str)
        .unique()
        .tolist()
    )
    if duplicate_tokens:
        raise RuntimeError(
            "Multi-GPU navhard sharding produced duplicate tokens: "
            f"{duplicate_tokens[:20]}"
        )

    all_mappings = _active_mappings(cfg, scene_tokens)
    if not all_mappings:
        raise RuntimeError("No active navhard reactive mappings were found.")

    try:
        pdm_score_df = official_two_stage.create_scene_aggregators(
            all_mappings,
            pdm_score_df,
            instantiate(cfg.simulator.proposal_sampling),
        )
        pdm_score_df = official_two_stage.compute_final_scores(pdm_score_df)
        pseudo_closed_loop_valid = True
    except Exception:
        logger.exception("Failed to calculate pseudo-closed-loop weights or comfort.")
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = False

    num_successful = int(pdm_score_df["valid"].sum())
    num_failed = int(len(pdm_score_df) - num_successful)
    failed_tokens = (
        pdm_score_df.loc[~pdm_score_df["valid"], "token"].to_list()
        if num_failed > 0
        else []
    )

    score_cols = _score_columns(pdm_score_df)
    pcl_group_score, pcl_stage1_score, pcl_stage2_score = (
        official_two_stage.calculate_individual_mapping_scores(
            pdm_score_df[score_cols + ["token", "weight"]], all_mappings
        )
    )

    for column in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC
        pdm_score_df.loc[stage_one_mask, f"{column}_stage_one"] = pdm_score_df.loc[
            stage_one_mask, column
        ]
        pdm_score_df.loc[stage_two_mask, f"{column}_stage_two"] = pdm_score_df.loc[
            stage_two_mask, column
        ]

    pdm_score_df.drop(columns=score_cols, inplace=True)
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(
        pdm_score_df["score_stage_two"]
    )
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)

    stage1_cols = [f"{column}_stage_one" for column in score_cols if column != "score"]
    stage2_cols = [f"{column}_stage_two" for column in score_cols if column != "score"]
    output_score_cols = stage1_cols + stage2_cols + ["score"]
    pdm_score_df = pdm_score_df[["token", "valid"] + output_score_cols]

    summary_rows: list[pd.Series] = []

    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = pseudo_closed_loop_valid
    stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
    for column in pcl_stage1_score.index:
        if column not in ["token", "valid", "score"]:
            stage1_row[f"{column}_stage_one"] = pcl_stage1_score[column]
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = pseudo_closed_loop_valid
    stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
    for column in pcl_stage2_score.index:
        if column not in ["token", "valid", "score"]:
            stage2_row[f"{column}_stage_two"] = pcl_stage2_score[column]
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = pseudo_closed_loop_valid
    combined_row["score"] = pcl_group_score["score"]
    for column in pcl_stage1_score.index:
        if column not in ["token", "valid", "score"]:
            combined_row[f"{column}_stage_one"] = pcl_stage1_score[column]
    for column in pcl_stage2_score.index:
        if column not in ["token", "valid", "score"]:
            combined_row[f"{column}_stage_two"] = pcl_stage2_score[column]
    summary_rows.append(combined_row)

    pdm_score_df = pd.concat(
        [pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    final_path = output_dir / f"{timestamp}.csv"
    pdm_score_df.to_csv(final_path)

    final_score = float(
        pdm_score_df.loc[
            pdm_score_df["token"] == "extended_pdm_score_combined", "score"
        ].iloc[0]
    )
    logger.info(
        "Finished multi-GPU navhard evaluation. successful=%d failed=%d "
        "extended_pdm_score_combined=%s results=%s",
        num_successful,
        num_failed,
        final_score,
        final_path,
    )
    if failed_tokens:
        logger.info("Failed tokens: %s", failed_tokens)
    return final_score, final_path


def _run_parent(args: argparse.Namespace) -> None:
    devices = _parse_cuda_devices(args.cuda_visible_devices)
    if args.num_gpus > len(devices):
        raise ValueError(
            f"Requested {args.num_gpus} GPUs, but CUDA_VISIBLE_DEVICES only exposes "
            f"{len(devices)}: {devices}"
        )
    if args.num_gpus > 0:
        devices = devices[: args.num_gpus]
    world_size = len(devices)
    if world_size < 1:
        raise ValueError("No CUDA devices selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(args.overrides)
    data_points, scene_tokens = _discover_data_points(cfg)
    if not data_points:
        raise RuntimeError("No evaluable navhard logs were discovered.")
    shards = split_data_points_by_log(data_points, world_size)
    shard_manifest_path = output_dir / "_navhard_data_points_by_rank.pkl"
    with open(shard_manifest_path, "wb") as file:
        pickle.dump(shards, file)

    rank_loads = [
        sum(int(point["num_evaluable"]) for point in shard) for shard in shards
    ]
    logger.info(
        "Starting NAVSIM v2 navhard multi-GPU PDM eval: world_size=%d "
        "scenarios=%d logs=%d rank_loads=%s",
        world_size,
        sum(rank_loads),
        len(data_points),
        rank_loads,
    )

    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for rank, device in enumerate(devices):
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--child",
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--data-points-file",
            str(shard_manifest_path),
            "--output-dir",
            str(output_dir),
            "--overrides",
            *args.overrides,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["PDM_RANK"] = str(rank)
        env["PDM_WORLD_SIZE"] = str(world_size)
        processes.append((rank, subprocess.Popen(command, env=env, text=True)))

    failed = False
    for rank, process in processes:
        return_code = process.wait()
        if return_code != 0:
            logger.error("PDM rank %d failed with return code %d", rank, return_code)
            failed = True
    if failed:
        raise RuntimeError(
            "At least one navhard PDM worker failed. Intermediate shards were kept "
            f"in {output_dir}."
        )

    score_shard_paths = [
        output_dir / f"_navhard_rank_{rank:03d}.pkl" for rank in range(world_size)
    ]
    missing = [str(path) for path in score_shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing navhard PDM shard files: {missing}")

    score_rows = [pd.read_pickle(path) for path in score_shard_paths]
    _finalize_official_two_stage_scores(cfg, score_rows, scene_tokens)

    for path in score_shard_paths:
        path.unlink(missing_ok=True)
    shard_manifest_path.unlink(missing_ok=True)


def _run_child(args: argparse.Namespace) -> None:
    rank = int(args.rank)
    world_size = int(args.world_size)
    cfg = _build_cfg(args.overrides)
    with open(args.data_points_file, "rb") as file:
        shards = pickle.load(file)
    if len(shards) != world_size:
        raise ValueError(f"Expected {world_size} shards, found {len(shards)}")

    my_data_points = shards[rank]
    rank_load = sum(int(point["num_evaluable"]) for point in my_data_points)
    logger.info(
        "[rank=%d] evaluating %d logs / approximately %d scenarios",
        rank,
        len(my_data_points),
        rank_load,
    )
    if rank != 0 and cfg.agent.get("eval_visualize", None) is not None:
        cfg.agent.eval_visualize = False

    pdm_score_df = _run_official_two_stage_worker(cfg, my_data_points)
    score_shard_path = Path(args.output_dir) / f"_navhard_rank_{rank:03d}.pkl"
    pdm_score_df.to_pickle(score_shard_path)
    logger.info("[rank=%d] saved pickle shard: %s", rank, score_shard_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU NAVSIM v2 official navhard two-stage PDM scorer."
    )
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--data-points-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
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
