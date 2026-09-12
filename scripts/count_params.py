"""Print parameter counts for Spark configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spark.config import merge_configs
from src.spark.recursive import SparkTRM, count_parameters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="*", default=["configs/spark7.yaml", "configs/spark15.yaml"])
    args = p.parse_args()
    for path in args.configs:
        arch, _ = merge_configs(path)
        model = SparkTRM(arch)
        c = count_parameters(model)
        print(f"{path:40s}  name={arch.name:18s}  core={c['millions_core']:6.3f}M  total={c['millions_total']:6.3f}M  puzzle={c['puzzle_emb']:,}")


if __name__ == "__main__":
    main()
