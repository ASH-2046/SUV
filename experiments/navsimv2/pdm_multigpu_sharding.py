from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def split_data_points_by_log(
    data_points: Sequence[Mapping[str, Any]],
    world_size: int,
) -> list[list[dict[str, Any]]]:
    """Greedily balance complete NAVSIM logs without splitting a log.

    NAVSIM v2 two-stage evaluation discovers synthetic scenes through their
    corresponding original scenes. Keeping a complete log on one worker
    preserves that relationship while still providing enough parallelism for
    navhard, which contains many logs.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}.")

    normalized: list[dict[str, Any]] = []
    seen_logs: set[str] = set()
    for point in data_points:
        if "log_file" not in point or "tokens" not in point:
            raise ValueError("Each data point must contain log_file and tokens.")

        log_file = str(point["log_file"])
        if log_file in seen_logs:
            raise ValueError(f"Duplicate log_file in data points: {log_file}")
        seen_logs.add(log_file)

        tokens = list(point["tokens"])
        cost = int(point.get("num_evaluable", len(tokens)))
        if cost < 0:
            raise ValueError(f"num_evaluable must be non-negative for {log_file}.")

        normalized_point = dict(point)
        normalized_point["log_file"] = log_file
        normalized_point["tokens"] = tokens
        normalized_point["num_evaluable"] = cost
        normalized.append(normalized_point)

    # Largest-processing-time-first scheduling gives a deterministic, well
    # balanced assignment while retaining complete logs as atomic units.
    normalized.sort(
        key=lambda point: (-int(point["num_evaluable"]), str(point["log_file"]))
    )
    shards: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    for point in normalized:
        rank = min(range(world_size), key=lambda index: (loads[index], index))
        shards[rank].append(point)
        loads[rank] += int(point["num_evaluable"])

    return shards
