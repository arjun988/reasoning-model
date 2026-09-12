"""YAML-backed configs for Spark TRM (Phase 1–2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SparkConfig:
    name: str = "spark-7"
    hidden_size: int = 512
    num_heads: int = 8
    L_layers: int = 2
    H_cycles: int = 3  # T in the paper: outer recursion cycles
    L_cycles: int = 6  # n in the paper: inner latent steps
    expansion: float = 4.0
    seq_len: int = 900
    vocab_size: int = 12
    pos_encodings: str = "rope"  # rope | rope_2d | learned | none
    mlp_t: bool = False
    use_puzzle_emb: bool = True
    puzzle_emb_len: int = 16
    puzzle_emb_ndim: int = 0  # 0 = hidden_size
    num_puzzle_identifiers: int = 2048
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    no_ACT_continue: bool = True
    use_halt: bool = True
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    grid_size: int = 30

    def __post_init__(self) -> None:
        if self.puzzle_emb_ndim <= 0:
            self.puzzle_emb_ndim = self.hidden_size
        if not self.use_puzzle_emb:
            self.puzzle_emb_len = 0
            self.puzzle_emb_ndim = 0
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.pos_encodings not in {"rope", "rope_2d", "learned", "none"}:
            raise ValueError(f"unknown pos_encodings: {self.pos_encodings}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def total_seq_len(self) -> int:
        return self.seq_len + self.puzzle_emb_len


@dataclass
class TrainConfig:
    seed: int = 42
    lr: float = 1e-4
    puzzle_emb_lr: float = 1e-4
    weight_decay: float = 0.1
    puzzle_emb_weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 2000
    max_steps: int = 50_000
    batch_size: int = 8
    grad_clip: float = 1.0
    ema: bool = True
    ema_decay: float = 0.999
    log_every: int = 20
    eval_every: int = 200
    save_every: int = 1000
    num_aug: int = 300
    translational_aug: bool = True
    train_on_eval_demos: bool = True
    holdout_fraction: float = 0.25
    loss_type: str = "stablemax_cross_entropy"  # or softmax_cross_entropy
    out_dir: str = "artifacts/spark7"
    data_dir: str = "data/arc-agi"
    device: str = "cuda"
    amp: bool = True


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def _from_dict(cls, data: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def merge_configs(
    arch_path: str | Path,
    train_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[SparkConfig, TrainConfig]:
    raw = load_yaml(arch_path)
    train_raw = raw.pop("train", {}) if "train" in raw else {}
    if train_path is not None:
        extra = load_yaml(train_path)
        train_raw = {**train_raw, **extra}
    if overrides:
        arch_over = overrides.get("arch", {})
        train_over = overrides.get("train", {})
        raw = {**raw, **arch_over}
        train_raw = {**train_raw, **train_over}
        for k, v in overrides.items():
            if k not in {"arch", "train"}:
                if k in {f.name for f in fields(SparkConfig)}:
                    raw[k] = v
                elif k in {f.name for f in fields(TrainConfig)}:
                    train_raw[k] = v
    return _from_dict(SparkConfig, raw), _from_dict(TrainConfig, train_raw)


def dump_config(arch: SparkConfig, train: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"arch": asdict(arch), "train": asdict(train)}, f, sort_keys=False)
