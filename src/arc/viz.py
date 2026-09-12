"""Grid visualization for sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
]


def grid_to_image(grid: np.ndarray, cell: int = 16) -> Image.Image:
    h, w = grid.shape
    img = Image.new("RGB", (w * cell, h * cell), (20, 20, 20))
    px = img.load()
    for r in range(h):
        for c in range(w):
            color = PALETTE[int(grid[r, c]) % 10]
            for dr in range(cell - 1):
                for dc in range(cell - 1):
                    px[c * cell + dc, r * cell + dr] = color
    return img


def save_task_strip(pairs: list[tuple[np.ndarray, np.ndarray]], path: str | Path, cell: int = 12) -> None:
    images = []
    for inp, out in pairs:
        left = grid_to_image(inp, cell)
        right = grid_to_image(out, cell)
        gap = Image.new("RGB", (8, max(left.height, right.height)), (40, 40, 40))
        strip = Image.new("RGB", (left.width + gap.width + right.width, max(left.height, right.height)), (20, 20, 20))
        strip.paste(left, (0, 0))
        strip.paste(gap, (left.width, 0))
        strip.paste(right, (left.width + gap.width, 0))
        images.append(strip)
    if not images:
        return
    height = sum(im.height + 6 for im in images)
    width = max(im.width for im in images)
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height + 6
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
