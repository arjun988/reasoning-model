"""ARC JSON loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class ArcPair:
    input: np.ndarray
    output: np.ndarray | None = None


@dataclass
class ArcTask:
    task_id: str
    train: list[ArcPair] = field(default_factory=list)
    test: list[ArcPair] = field(default_factory=list)
    split: str = "unknown"


def _as_grid(obj) -> np.ndarray:
    arr = np.array(obj, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"grid must be 2D, got {arr.shape}")
    if arr.size == 0 or arr.shape[0] > 30 or arr.shape[1] > 30:
        raise ValueError(f"bad grid shape {arr.shape}")
    if arr.min() < 0 or arr.max() > 9:
        raise ValueError("grid values must be in 0..9")
    return arr


def load_task_json(path: str | Path, split: str = "unknown") -> ArcTask:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    task_id = path.stem
    train = [ArcPair(_as_grid(p["input"]), _as_grid(p["output"])) for p in raw["train"]]
    test = []
    for p in raw.get("test", []):
        out = _as_grid(p["output"]) if "output" in p and p["output"] is not None else None
        test.append(ArcPair(_as_grid(p["input"]), out))
    return ArcTask(task_id=task_id, train=train, test=test, split=split)


def load_arc_dir(root: str | Path, split: str | None = None) -> list[ArcTask]:
    """Load all task JSON files under a directory (recurses one level of split folders)."""
    root = Path(root)
    tasks: list[ArcTask] = []
    if not root.exists():
        return tasks
    json_files = sorted(root.glob("*.json"))
    inferred = split or root.name
    for path in json_files:
        if path.name in {"dataset.json", "identifiers.json", "test_puzzles.json"}:
            continue
        try:
            tasks.append(load_task_json(path, split=inferred))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    if not json_files:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            tasks.extend(load_arc_dir(sub, split=sub.name))
    return tasks


def iter_demo_pairs(task: ArcTask) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for pair in task.train:
        if pair.output is None:
            continue
        yield pair.input, pair.output
