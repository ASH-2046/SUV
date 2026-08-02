from __future__ import annotations

import argparse
import logging
import lzma
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


logger = logging.getLogger(__name__)


def _save_trajectory_debug_csv(
    *,
    output_dir: Path,
    token: str,
    pred_trajectory: Any,
    gt_trajectory: Any,
) -> Path:
    """Save the complete predicted/GT [x, y, yaw] trajectories for one token."""
    pred = np.asarray(pred_trajectory.poses, dtype=np.float32)
    gt = np.asarray(gt_trajectory.poses, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[1] < 3:
        raise ValueError(f"Predicted trajectory must be [T, >=3], got {pred.shape}.")
    if gt.ndim != 2 or gt.shape[1] < 3:
        raise ValueError(f"GT trajectory must be [T, >=3], got {gt.shape}.")

    num_rows = max(int(pred.shape[0]), int(gt.shape[0]))
    values = np.full((num_rows, 6), np.nan, dtype=np.float32)
    values[: pred.shape[0], :3] = pred[:, :3]
    values[: gt.shape[0], 3:] = gt[:, :3]
    frame = pd.DataFrame(
        values,
        columns=["pred_x", "pred_y", "pred_yaw", "gt_x", "gt_y", "gt_yaw"],
    )
    frame.insert(0, "step", np.arange(num_rows, dtype=np.int64))

    debug_dir = output_dir / "trajectory_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(token))[:160] or "unknown_token"
    path = debug_dir / f"{safe_token}.csv"
    frame.to_csv(path, index=False)
    return path


def _trajectory_poses_array(trajectory: Any) -> np.ndarray:
    poses = getattr(trajectory, "poses", trajectory)
    if hasattr(poses, "detach"):
        poses = poses.detach().to(device="cpu").numpy()
    array = np.asarray(poses, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"Trajectory poses must be [T, >=2], got {array.shape}.")
    return np.ascontiguousarray(array[:, :2], dtype=np.float32)


def _trajectory_error_metrics(pred_trajectory: Any, gt_trajectory: Any) -> dict[str, float]:
    pred_xy = _trajectory_poses_array(pred_trajectory)
    gt_xy = _trajectory_poses_array(gt_trajectory)
    num_steps = min(int(pred_xy.shape[0]), int(gt_xy.shape[0]))
    if num_steps <= 0:
        raise ValueError("Cannot compute trajectory error with zero overlapping future steps.")

    distances = np.linalg.norm(pred_xy[:num_steps] - gt_xy[:num_steps], axis=1)
    return {
        "average_displacement_error_m": float(distances.mean()),
        "final_displacement_error_m": float(distances[-1]),
        "trajectory_error_num_steps": float(num_steps),
    }


def _parse_cuda_devices(value: str) -> list[str]:
    devices = [item.strip() for item in str(value).split(",") if item.strip()]
    return devices or ["0"]


def _resolve_navsim_config_dir() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("navsim")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot find importable navsim package.")
    navsim_root = Path(next(iter(spec.submodule_search_locations)))
    config_dir = navsim_root / "planning" / "script" / "config" / "pdm_scoring"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"NAVSIM PDM config directory not found: {config_dir}")
    return config_dir


def _move_all_modules_to_gpu(instance: Any) -> None:
    for attr_name in dir(instance):
        attr = getattr(instance, attr_name)
        if isinstance(attr, nn.Module):
            setattr(instance, attr_name, attr.cuda())


def _build_cfg(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(_resolve_navsim_config_dir()), version_base=None):
        return compose(config_name="default_run_pdm_score", overrides=overrides)


def _discover_tokens(cfg: DictConfig) -> list[str]:
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing > 0:
        logger.warning("Missing metric cache for %d tokens. Skipping these tokens.", num_missing)
    if num_unused > 0:
        logger.warning("Unused metric cache for %d tokens. Skipping these tokens.", num_unused)
    return tokens


def _evaluate_tokens(cfg: DictConfig, tokens: list[str], rank: int) -> pd.DataFrame:
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, (
        "Simulator and scorer proposal sampling has to be identical"
    )

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    if rank != 0 and hasattr(agent, "eval_visualization_enabled"):
        agent.eval_visualization_enabled = False
    _move_all_modules_to_gpu(agent)

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))

    results: list[dict[str, Any]] = []
    if not tokens_to_evaluate:
        return pd.DataFrame(columns=["token", "valid"])

    start_time = time.time()
    for idx, token in enumerate(tokens_to_evaluate):
        score_row: dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            log_name = scene_loader.scene_frames_dicts[token][0]["log_name"]
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                scene = None
                trajectory = agent.compute_trajectory(agent_input)

            gt_trajectory = None
            try:
                if scene is None:
                    scene = scene_loader.get_scene_from_token(token)
                gt_trajectory = scene.get_future_trajectory(
                    num_trajectory_frames=int(getattr(agent, "num_future_frames", 8))
                )
                score_row.update(_trajectory_error_metrics(trajectory, gt_trajectory))
            except Exception:
                score_row["average_displacement_error_m"] = np.nan
                score_row["final_displacement_error_m"] = np.nan
                score_row["trajectory_error_num_steps"] = np.nan
                logger.warning("[rank=%d] Trajectory error metrics failed for token %s", rank, token)
                traceback.print_exc()

            visualize_hook = getattr(agent, "maybe_visualize_pdm_score_sample", None)
            should_visualize_hook = getattr(agent, "should_visualize_pdm_score_sample", None)
            should_visualize = (
                visualize_hook is not None
                and getattr(agent, "eval_visualization_enabled", False)
                and (should_visualize_hook is None or bool(should_visualize_hook()))
            )
            if should_visualize:
                try:
                    if scene is None:
                        scene = scene_loader.get_scene_from_token(token)
                    visualize_hook(
                        agent_input=agent_input,
                        scene=scene,
                        trajectory=trajectory,
                        output_root=Path(cfg.output_dir),
                        token=token,
                        log_name=log_name,
                        sample_idx=idx,
                    )
                except Exception:
                    logger.warning("[rank=%d] Visualization failed for token %s", rank, token)
                    traceback.print_exc()

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))

            if bool(getattr(agent, "eval_score_ground_truth", False)):
                try:
                    if scene is None:
                        scene = scene_loader.get_scene_from_token(token)
                    if gt_trajectory is None:
                        gt_trajectory = scene.get_future_trajectory(
                            num_trajectory_frames=int(getattr(agent, "num_future_frames", 8))
                        )
                    gt_pdm_result = pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=gt_trajectory,
                        future_sampling=simulator.proposal_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                    score_row.update(
                        {f"gt_{name}": value for name, value in asdict(gt_pdm_result).items()}
                    )
                    score_row["gt_debug_valid"] = True
                    debug_path = _save_trajectory_debug_csv(
                        output_dir=Path(cfg.output_dir),
                        token=token,
                        pred_trajectory=trajectory,
                        gt_trajectory=gt_trajectory,
                    )
                    logger.info("[rank=%d] saved trajectory debug: %s", rank, debug_path)
                except Exception:
                    score_row["gt_debug_valid"] = False
                    logger.warning("[rank=%d] GT sanity scoring failed for token %s", rank, token)
                    traceback.print_exc()
        except Exception:
            logger.warning("[rank=%d] Agent failed for token %s", rank, token)
            traceback.print_exc()
            score_row["valid"] = False

        results.append(score_row)
        if (idx + 1) % 50 == 0 or idx == len(tokens_to_evaluate) - 1:
            elapsed = time.time() - start_time
            speed = (idx + 1) / max(elapsed, 1e-6)
            remaining = (len(tokens_to_evaluate) - idx - 1) / max(speed, 1e-6)
            logger.info(
                "[rank=%d] progress: %d/%d samples, %.2f samples/s, eta %.0fs",
                rank,
                idx + 1,
                len(tokens_to_evaluate),
                speed,
                remaining,
            )

    return pd.DataFrame(results)


def _merge_shards(output_dir: Path, world_size: int) -> Path:
    shard_paths = [output_dir / f"_rank_{rank:03d}.csv" for rank in range(world_size)]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PDM shard files: {missing}")

    pdm_score_df = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
    num_successful = int(pdm_score_df["valid"].sum())
    num_failed = int(len(pdm_score_df) - num_successful)

    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    final_path = output_dir / f"{timestamp}.csv"
    pdm_score_df.to_csv(final_path)
    for path in shard_paths:
        path.unlink(missing_ok=True)
    (output_dir / "_tokens.txt").unlink(missing_ok=True)

    logger.info(
        "Finished multi-GPU evaluation. successful=%d failed=%d average_score=%s results=%s",
        num_successful,
        num_failed,
        pdm_score_df["score"].mean(),
        final_path,
    )
    return final_path


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
    tokens = _discover_tokens(cfg)
    token_file = output_dir / "_tokens.txt"
    token_file.write_text("\n".join(tokens) + "\n")

    logger.info("Starting NAVSIM v1 multi-GPU PDM eval: world_size=%d tokens=%d", world_size, len(tokens))
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
            "--token-file",
            str(token_file),
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

    _merge_shards(output_dir, world_size)


def _run_child(args: argparse.Namespace) -> None:
    rank = int(args.rank)
    world_size = int(args.world_size)
    cfg = _build_cfg(args.overrides)
    with open(args.token_file) as f:
        tokens = [line.strip() for line in f if line.strip()]
    my_tokens = tokens[rank::world_size]
    logger.info("[rank=%d] evaluating %d/%d tokens", rank, len(my_tokens), len(tokens))
    df = _evaluate_tokens(cfg, my_tokens, rank)
    shard_path = Path(args.output_dir) / f"_rank_{rank:03d}.csv"
    df.to_csv(shard_path, index=False)
    logger.info("[rank=%d] saved shard: %s", rank, shard_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU NAVSIM v1 PDM score runner.")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--token-file", type=str, default="")
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
    if args.child:
        _run_child(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
