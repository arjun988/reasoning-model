"""Tiny Recursive Model (TRM) core — Phase 1 faithful reproduction.

Algorithm (Jolicoeur-Martineau, 2025):
  latent z and answer y, a single tiny net applied recursively.
  For T-1 cycles, recurse without grad; then one cycle with full BPTT
  through n latent updates + 1 answer update. Deep supervision + halt head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.spark.config import SparkConfig
from src.spark.layers import (
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding,
    RotaryEmbedding2D,
    TransformerBlock,
)


@dataclass
class Carry:
    y: torch.Tensor
    z: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor


class SparkTRM(nn.Module):
    def __init__(self, config: SparkConfig) -> None:
        super().__init__()
        self.config = config
        self.fwd_dtype = getattr(torch, config.forward_dtype)
        if self.fwd_dtype == torch.bfloat16 and not torch.cuda.is_available():
            self.fwd_dtype = torch.float32

        scale = config.hidden_size**0.5
        self.embed_scale = scale
        self.embed = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=1.0 / scale)
        self.out_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.halt_head = CastedLinear(config.hidden_size, 1, bias=True)
        with torch.no_grad():
            self.halt_head.weight.zero_()
            self.halt_head.bias.fill_(-5.0)

        if config.use_puzzle_emb:
            self.puzzle_emb = nn.Embedding(config.num_puzzle_identifiers, config.hidden_size)
            nn.init.zeros_(self.puzzle_emb.weight)
        else:
            self.puzzle_emb = None

        pos = config.pos_encodings
        total = config.total_seq_len
        if pos == "rope":
            self.rotary = RotaryEmbedding(config.head_dim, total, config.rope_theta)
        elif pos == "rope_2d":
            self.rotary = RotaryEmbedding2D(
                config.head_dim, config.puzzle_emb_len, config.grid_size, config.rope_theta
            )
        else:
            self.rotary = None
        if pos == "learned":
            self.pos_emb = CastedEmbedding(total, config.hidden_size, init_std=1.0 / scale)
        else:
            self.pos_emb = None

        mixer_len = total
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    expansion=config.expansion,
                    rms_eps=config.rms_norm_eps,
                    mlp_t=config.mlp_t,
                    mixer_seq_len=mixer_len,
                    axial_rope=config.pos_encodings == "rope_2d",
                )
                for _ in range(config.L_layers)
            ]
        )

        y0 = torch.empty(config.hidden_size)
        z0 = torch.empty(config.hidden_size)
        nn.init.trunc_normal_(y0, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(z0, std=1.0, a=-2.0, b=2.0)
        self.register_buffer("y_init", y0, persistent=True)
        self.register_buffer("z_init", z0, persistent=True)

    def net(self, hidden: torch.Tensor, inject: torch.Tensor, cos_sin) -> torch.Tensor:
        hidden = hidden + inject
        for block in self.blocks:
            hidden = block(hidden, cos_sin)
        return hidden

    def _embed(self, tokens: torch.Tensor, puzzle_ids: torch.Tensor | None) -> torch.Tensor:
        x = self.embed(tokens.long(), self.fwd_dtype)
        if self.puzzle_emb is not None:
            if puzzle_ids is None:
                raise ValueError("puzzle_ids required when use_puzzle_emb=True")
            p = self.puzzle_emb(puzzle_ids.long()).to(dtype=self.fwd_dtype)
            p = p.unsqueeze(1)
            if self.config.puzzle_emb_len > 1:
                zeros = p.new_zeros(p.size(0), self.config.puzzle_emb_len - 1, p.size(-1))
                p = torch.cat([p, zeros], dim=1)
            x = torch.cat([p, x], dim=1)
        if self.pos_emb is not None:
            idx = torch.arange(x.size(1), device=x.device)
            x = 0.707106781 * (x + self.pos_emb(idx, self.fwd_dtype).unsqueeze(0))
        return x * self.embed_scale

    def _cos_sin(self):
        if self.rotary is None:
            return None
        return self.rotary()

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cos_sin,
        n: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.config.L_cycles if n is None else n
        for _ in range(n):
            z = self.net(z, y + x, cos_sin)
        y = self.net(y, z, cos_sin)
        return y, z

    def deep_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_sin = self._cos_sin()
        t = self.config.H_cycles
        n = self.config.L_cycles
        with torch.no_grad():
            for _ in range(max(t - 1, 0)):
                y, z = self.latent_recursion(x, y, z, cos_sin, n)
        y, z = self.latent_recursion(x, y, z, cos_sin, n)
        prefix = self.config.puzzle_emb_len
        logits = self.out_head(y[:, prefix:, :])
        halt_logit = self.halt_head(y[:, 0, :]).squeeze(-1).float()
        return y.detach(), z.detach(), logits, halt_logit

    def initial_carry(self, batch_size: int, device: torch.device) -> Carry:
        s = self.config.total_seq_len
        d = self.config.hidden_size
        y = self.y_init.to(device=device, dtype=self.fwd_dtype).view(1, 1, d).expand(batch_size, s, d).contiguous()
        z = self.z_init.to(device=device, dtype=self.fwd_dtype).view(1, 1, d).expand(batch_size, s, d).contiguous()
        return Carry(
            y=y,
            z=z,
            steps=torch.zeros(batch_size, dtype=torch.long, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
        )

    def reset_halted(self, carry: Carry) -> Carry:
        b, s, d = carry.y.shape
        reset = carry.halted.view(b, 1, 1)
        y0 = self.y_init.to(dtype=carry.y.dtype, device=carry.y.device).view(1, 1, d).expand(b, s, d)
        z0 = self.z_init.to(dtype=carry.z.dtype, device=carry.z.device).view(1, 1, d).expand(b, s, d)
        return Carry(
            y=torch.where(reset, y0, carry.y),
            z=torch.where(reset, z0, carry.z),
            steps=torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps),
            halted=carry.halted,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        puzzle_ids: torch.Tensor | None = None,
        carry: Carry | None = None,
    ) -> tuple[Carry, torch.Tensor, torch.Tensor]:
        if carry is None:
            carry = self.initial_carry(tokens.size(0), tokens.device)
        carry = self.reset_halted(carry)
        x = self._embed(tokens, puzzle_ids)
        y, z, logits, halt_logit = self.deep_recursion(x, carry.y, carry.z)
        steps = carry.steps + 1
        last = steps >= self.config.halt_max_steps
        halted = last
        if self.training and self.config.halt_max_steps > 1 and self.config.use_halt:
            halted = halted | (halt_logit > 0)
            explore = torch.rand_like(halt_logit) < self.config.halt_exploration_prob
            min_steps = torch.randint(
                low=2,
                high=self.config.halt_max_steps + 1,
                size=steps.shape,
                device=steps.device,
            )
            min_steps = torch.where(explore, min_steps, torch.ones_like(min_steps))
            halted = halted & (steps >= min_steps)
        new_carry = Carry(y=y, z=z, steps=steps, halted=halted)
        return new_carry, logits, halt_logit


def count_parameters(model: nn.Module, exclude_puzzle: bool = True) -> dict[str, int]:
    total = 0
    core = 0
    puzzle = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if "puzzle_emb" in name:
            puzzle += n
        else:
            core += n
    return {
        "total": total,
        "core": core if exclude_puzzle else total,
        "puzzle_emb": puzzle,
        "millions_core": round(core / 1e6, 3),
        "millions_total": round(total / 1e6, 3),
    }
