import numpy as np

from src.arc.augment import augment_pair, dihedral_transform, inverse_dihedral_transform
from src.arc.encode import decode_grid, encode_grid, encode_pair
from src.arc.metrics import cell_accuracy, exact_match, pass_at_2, score_tasks


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        h, w = int(rng.integers(1, 12)), int(rng.integers(1, 12))
        grid = rng.integers(0, 10, size=(h, w), dtype=np.uint8)
        seq, _ = encode_grid(grid)
        got = decode_grid(seq)
        assert got.shape == grid.shape
        assert np.array_equal(got, grid)


def test_encode_pair_same_offset():
    inp = np.arange(9, dtype=np.uint8).reshape(3, 3)
    out = np.rot90(inp, 1).copy()
    x, y = encode_pair(inp, out, random_translation=False)
    assert np.array_equal(decode_grid(x), inp)
    assert np.array_equal(decode_grid(y), out)


def test_dihedral_inverses():
    g = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for tid in range(8):
        rec = inverse_dihedral_transform(dihedral_transform(g, tid), tid)
        assert np.array_equal(rec, g), tid


def test_augment_invertible():
    rng = np.random.default_rng(2)
    inp = rng.integers(0, 10, size=(5, 6), dtype=np.uint8)
    out = rng.integers(0, 10, size=(5, 6), dtype=np.uint8)
    a, b, meta = augment_pair(inp, out, rng)
    from src.arc.augment import invert_grid

    assert np.array_equal(invert_grid(a, meta), inp)
    assert np.array_equal(invert_grid(b, meta), out)


def test_metrics_pass_at_2():
    gold = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    wrong = np.zeros_like(gold)
    assert exact_match(gold, gold)
    assert not exact_match(wrong, gold)
    assert pass_at_2([wrong, gold], gold)
    assert not pass_at_2([wrong, wrong], gold)
    assert cell_accuracy(gold, gold) == 1.0
    scores = score_tasks({"t": [[wrong, gold]]}, {"t": [gold]})
    assert scores["pass_at_2"] == 1.0
