import torch

from src.spark.config import SparkConfig, merge_configs
from src.spark.losses import act_loss
from src.spark.recursive import SparkTRM, count_parameters


def test_spark7_core_in_band():
    arch, _ = merge_configs("configs/spark7.yaml")
    counts = count_parameters(SparkTRM(arch))
    assert 6.0 <= counts["millions_core"] <= 9.0, counts


def test_spark15_core_in_band():
    arch, _ = merge_configs("configs/spark15.yaml")
    counts = count_parameters(SparkTRM(arch))
    assert 12.0 <= counts["millions_core"] <= 16.0, counts


def _fwd(arch: SparkConfig):
    model = SparkTRM(arch)
    b = 2
    tokens = torch.randint(0, arch.vocab_size, (b, arch.seq_len))
    labels = tokens.clone()
    pid = torch.tensor([1, 2]) if arch.use_puzzle_emb else None
    carry, logits, halt = model(tokens, pid)
    assert logits.shape == (b, arch.seq_len, arch.vocab_size)
    assert halt.shape == (b,)
    loss, _ = act_loss(logits, labels, halt, kind="softmax_cross_entropy")
    loss.backward()


def test_debug_forward_backward():
    arch, _ = merge_configs("configs/spark_debug.yaml")
    _fwd(arch)


def test_rope2d_forward():
    arch, _ = merge_configs("configs/spark_debug.yaml")
    arch.pos_encodings = "rope_2d"
    _fwd(arch)


def test_no_puzzle_emb_forward():
    arch, _ = merge_configs("configs/spark_debug.yaml")
    arch.use_puzzle_emb = False
    arch.puzzle_emb_len = 0
    _fwd(arch)
