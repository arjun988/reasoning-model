"""Tiny synthetic ARC-like tasks for the Phase 1 overfit test."""

from __future__ import annotations

import numpy as np

from src.arc.io import ArcPair, ArcTask


def _box(h: int, w: int, fill: int, border: int) -> np.ndarray:
    g = np.full((h, w), fill, dtype=np.uint8)
    g[0, :] = border
    g[-1, :] = border
    g[:, 0] = border
    g[:, -1] = border
    return g


def make_overfit_tasks() -> list[ArcTask]:
    rng = np.random.default_rng(0)
    tasks = []

    # identity
    trains = []
    for i in range(3):
        g = rng.integers(0, 5, size=(4 + i, 4 + i), dtype=np.uint8)
        trains.append(ArcPair(g.copy(), g.copy()))
    test_g = rng.integers(0, 5, size=(5, 5), dtype=np.uint8)
    tasks.append(ArcTask("syn-identity", trains, [ArcPair(test_g, test_g.copy())], "synthetic"))

    # rotate 90 CW (np.rot90 is CCW, so 3)
    trains = []
    for i in range(3):
        g = rng.integers(1, 6, size=(3, 3), dtype=np.uint8)
        trains.append(ArcPair(g, np.rot90(g, 3).copy()))
    g = rng.integers(1, 6, size=(3, 3), dtype=np.uint8)
    tasks.append(ArcTask("syn-rot90", trains, [ArcPair(g, np.rot90(g, 3).copy())], "synthetic"))

    # recolor 1 -> 2
    trains = []
    for i in range(3):
        g = rng.integers(0, 3, size=(4, 4), dtype=np.uint8)
        out = g.copy()
        out[out == 1] = 2
        trains.append(ArcPair(g, out))
    g = rng.integers(0, 3, size=(4, 4), dtype=np.uint8)
    out = g.copy()
    out[out == 1] = 2
    tasks.append(ArcTask("syn-recolor", trains, [ArcPair(g, out)], "synthetic"))

    # fill interior
    trains = []
    for i in range(3):
        inp = _box(5, 5, fill=0, border=1 + i)
        out = inp.copy()
        out[1:-1, 1:-1] = 4
        trains.append(ArcPair(inp, out))
    inp = _box(5, 5, fill=0, border=2)
    out = inp.copy()
    out[1:-1, 1:-1] = 4
    tasks.append(ArcTask("syn-fill", trains, [ArcPair(inp, out)], "synthetic"))
    return tasks
