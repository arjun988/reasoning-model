"""PyTorch dataset of encoded ARC input→output pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.arc.augment import augment_pair
from src.arc.encode import encode_pair
from src.arc.io import ArcTask, iter_demo_pairs


@dataclass
class EncodedExample:
    tokens: np.ndarray
    labels: np.ndarray
    puzzle_id: int
    task_id: str


class ArcPairDataset(Dataset):
    def __init__(
        self,
        tasks: list[ArcTask],
        puzzle_id_map: dict[str, int],
        *,
        augment: bool = True,
        translational_aug: bool = True,
        seed: int = 0,
    ) -> None:
        self.rows: list[tuple[np.ndarray, np.ndarray, int, str]] = []
        for task in tasks:
            pid = puzzle_id_map[task.task_id]
            for inp, out in iter_demo_pairs(task):
                self.rows.append((inp, out, pid, task.task_id))
        self.augment = augment
        self.translational_aug = translational_aug
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        inp, out, pid, task_id = self.rows[idx]
        rng = np.random.default_rng(self.seed + idx * 1_000_003)
        if self.augment:
            inp, out, _ = augment_pair(inp, out, rng)
        x, y = encode_pair(inp, out, random_translation=self.translational_aug and self.augment, rng=rng)
        return {
            "tokens": torch.from_numpy(x.astype(np.int64)),
            "labels": torch.from_numpy(y.astype(np.int64)),
            "puzzle_id": torch.tensor(pid, dtype=torch.long),
            "task_id": task_id,
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.stack([b["tokens"] for b in batch], 0),
        "labels": torch.stack([b["labels"] for b in batch], 0),
        "puzzle_id": torch.stack([b["puzzle_id"] for b in batch], 0),
    }


def build_puzzle_id_map(tasks: list[ArcTask]) -> dict[str, int]:
    ids = sorted({t.task_id for t in tasks})
    return {tid: i + 1 for i, tid in enumerate(ids)}  # 0 reserved
