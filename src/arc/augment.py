"""Dihedral (D4) and color-permutation augmentations. Invertible for voting later."""

from __future__ import annotations

import numpy as np


def dihedral_transform(grid: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0:
        return grid
    if tid == 1:
        return np.rot90(grid, 1)
    if tid == 2:
        return np.rot90(grid, 2)
    if tid == 3:
        return np.rot90(grid, 3)
    if tid == 4:
        return np.fliplr(grid)
    if tid == 5:
        return np.flipud(grid)
    if tid == 6:
        return np.fliplr(np.rot90(grid, 1))
    if tid == 7:
        return np.flipud(np.rot90(grid, 1))
    raise ValueError(f"tid must be 0..7, got {tid}")


def inverse_dihedral_transform(grid: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0:
        return grid
    if tid == 1:
        return np.rot90(grid, 3)
    if tid == 2:
        return np.rot90(grid, 2)
    if tid == 3:
        return np.rot90(grid, 1)
    if tid == 4:
        return np.fliplr(grid)
    if tid == 5:
        return np.flipud(grid)
    if tid == 6:
        return np.rot90(np.fliplr(grid), 3)
    if tid == 7:
        return np.rot90(np.flipud(grid), 3)
    raise ValueError(f"tid must be 0..7, got {tid}")


def color_permutation(rng: np.random.Generator) -> np.ndarray:
    mapping = np.arange(10, dtype=np.uint8)
    mapping[1:] = rng.permutation(np.arange(1, 10, dtype=np.uint8))
    return mapping


def apply_color_map(grid: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return mapping[grid]


def invert_color_map(mapping: np.ndarray) -> np.ndarray:
    inv = np.empty_like(mapping)
    inv[mapping] = np.arange(len(mapping), dtype=mapping.dtype)
    return inv


def augment_pair(
    inp: np.ndarray,
    out: np.ndarray,
    rng: np.random.Generator,
    *,
    dihedral: bool = True,
    color: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    tid = int(rng.integers(0, 8)) if dihedral else 0
    mapping = color_permutation(rng) if color else np.arange(10, dtype=np.uint8)
    inp2 = apply_color_map(dihedral_transform(inp, tid), mapping)
    out2 = apply_color_map(dihedral_transform(out, tid), mapping)
    meta = {"tid": tid, "color_map": mapping}
    return inp2, out2, meta


def invert_grid(grid: np.ndarray, meta: dict) -> np.ndarray:
    inv_map = invert_color_map(meta["color_map"])
    return inverse_dihedral_transform(apply_color_map(grid, inv_map), meta["tid"])
