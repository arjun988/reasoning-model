"""30×30 ARC grid packing matching TRM/HRM.

Token ids:
  0 = PAD
  1 = EOS (L-shaped size marker)
  2..11 = colors 0..9
"""

from __future__ import annotations

import numpy as np

ARC_MAX = 30
PAD_ID = 0
EOS_ID = 1
VOCAB_SIZE = 12  # pad + eos + 10 colors


def color_to_token(color: np.ndarray) -> np.ndarray:
    return (color.astype(np.int16) + 2).astype(np.int64)


def token_to_color(tokens: np.ndarray) -> np.ndarray:
    return np.clip(tokens.astype(np.int16) - 2, 0, 9).astype(np.uint8)


def _place(grid: np.ndarray, pad_r: int, pad_c: int) -> np.ndarray:
    h, w = grid.shape
    canvas = np.full((ARC_MAX, ARC_MAX), PAD_ID, dtype=np.int64)
    tokens = color_to_token(grid)
    canvas[pad_r : pad_r + h, pad_c : pad_c + w] = tokens
    eos_r, eos_c = pad_r + h, pad_c + w
    if eos_r < ARC_MAX:
        canvas[eos_r, pad_c:eos_c] = EOS_ID
    if eos_c < ARC_MAX:
        canvas[pad_r:eos_r, eos_c] = EOS_ID
    return canvas


def encode_grid(
    grid: np.ndarray,
    *,
    pad_r: int = 0,
    pad_c: int = 0,
    random_translation: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = grid.shape
    max_r = ARC_MAX - h
    max_c = ARC_MAX - w
    if random_translation:
        rng = rng or np.random.default_rng()
        pad_r = int(rng.integers(0, max_r + 1))
        pad_c = int(rng.integers(0, max_c + 1))
    else:
        pad_r = min(max(pad_r, 0), max_r)
        pad_c = min(max(pad_c, 0), max_c)
    seq = _place(grid, pad_r, pad_c).reshape(-1)
    return seq, (pad_r, pad_c)


def encode_pair(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    random_translation: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode input/output with the same translation offset."""
    h = max(inp.shape[0], out.shape[0])
    w = max(inp.shape[1], out.shape[1])
    max_r = ARC_MAX - h
    max_c = ARC_MAX - w
    if random_translation:
        rng = rng or np.random.default_rng()
        pad_r = int(rng.integers(0, max(max_r, 0) + 1))
        pad_c = int(rng.integers(0, max(max_c, 0) + 1))
    else:
        pad_r = pad_c = 0
    x, _ = encode_grid(inp, pad_r=pad_r, pad_c=pad_c)
    y, _ = encode_grid(out, pad_r=pad_r, pad_c=pad_c)
    return x, y


def decode_grid(seq: np.ndarray) -> np.ndarray:
    g = np.asarray(seq).reshape(ARC_MAX, ARC_MAX)
    occupied = g != PAD_ID
    if not occupied.any():
        return np.zeros((1, 1), dtype=np.uint8)
    rows = np.any(occupied, axis=1)
    cols = np.any(occupied, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    if crop.shape[0] > 1 and np.all((crop[-1] == EOS_ID) | (crop[-1] == PAD_ID)):
        crop = crop[:-1]
    if crop.shape[1] > 1 and np.all((crop[:, -1] == EOS_ID) | (crop[:, -1] == PAD_ID)):
        crop = crop[:, :-1]
    crop = np.where(crop == EOS_ID, PAD_ID, crop)
    return token_to_color(crop)
