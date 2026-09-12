"""Render a few ARC tasks to PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arc.io import load_arc_dir
from src.arc.viz import save_task_strip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/arc-agi/data/training")
    p.add_argument("--out", default="artifacts/viz")
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()
    tasks = load_arc_dir(args.data_dir, split="training")[: args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        pairs = [(pr.input, pr.output) for pr in task.train if pr.output is not None]
        save_task_strip(pairs, out / f"{task.task_id}.png")
        print(task.task_id, len(pairs))


if __name__ == "__main__":
    main()
