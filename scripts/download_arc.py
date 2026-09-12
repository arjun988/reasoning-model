"""Clone ARC-AGI-1 public data into data/arc-agi."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DST = ROOT / "data" / "arc-agi"
REPO = "https://github.com/fchollet/ARC-AGI.git"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dst", default=str(DEFAULT_DST))
    args = p.parse_args()
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if (dst / "data" / "training").exists() or (dst / "training").exists():
        print(f"already present: {dst}")
        return
    cmd = ["git", "clone", "--depth", "1", REPO, str(dst)]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    train = dst / "data" / "training"
    if train.exists():
        print(f"training tasks: {len(list(train.glob('*.json')))}")
    else:
        print("cloned, but training folder not where expected; inspect", dst)
        sys.exit(1)


if __name__ == "__main__":
    main()
