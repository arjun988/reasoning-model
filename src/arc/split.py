"""Frozen holdout split. Never train on these task ids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def task_hash(task_id: str, salt: str = "spark-holdout-v1") -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()


def choose_holdout(task_ids: list[str], fraction: float = 0.25, seed: str = "spark-holdout-v1") -> list[str]:
    ranked = sorted(task_ids, key=lambda tid: task_hash(tid, seed))
    n = max(1, int(round(len(ranked) * fraction)))
    return ranked[:n]


def write_holdout(path: str | Path, ids: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "task_ids": sorted(ids)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_holdout(path: str | Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data["task_ids"])
