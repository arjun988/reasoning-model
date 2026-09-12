"""Transformer blocks for Spark TRM: RMSNorm, RoPE, attention, SwiGLU, MLP-mixer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def find_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.weight, std=1.0 / math.sqrt(in_features), a=-2.0, b=2.0)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, std=init_std, a=-2.0, b=2.0)

    def forward(self, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return F.embedding(idx, self.weight.to(dtype=dtype))


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)).to(orig)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B, S, H, D], cos/sin: [S, D]
    orig_q, orig_k = q.dtype, k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q.to(orig_q), k.to(orig_k)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int, base: float = 10000.0) -> None:
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached


class RotaryEmbedding2D(nn.Module):
    """Axial 2D RoPE for a prefix + H×W grid (Phase 2 probe)."""

    def __init__(
        self,
        dim: int,
        prefix_len: int,
        grid_size: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        half = dim // 2
        inv = 1.0 / (base ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
        rows = torch.arange(grid_size, dtype=torch.float32)
        cols = torch.arange(grid_size, dtype=torch.float32)
        row_freq = torch.outer(rows, inv)
        col_freq = torch.outer(cols, inv)
        row_emb = torch.cat((row_freq, row_freq), dim=-1)
        col_emb = torch.cat((col_freq, col_freq), dim=-1)

        seq = prefix_len + grid_size * grid_size
        cos = torch.zeros(seq, dim)
        sin = torch.zeros(seq, dim)
        if prefix_len > 0:
            t = torch.arange(prefix_len, dtype=torch.float32)
            pref = torch.outer(t, 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))
            pref = torch.cat((pref, pref), dim=-1)
            cos[:prefix_len] = pref.cos()
            sin[:prefix_len] = pref.sin()
        idx = prefix_len
        for r in range(grid_size):
            for c in range(grid_size):
                cos[idx, :half] = row_emb[r].cos()
                sin[idx, :half] = row_emb[r].sin()
                cos[idx, half:] = col_emb[c].cos()
                sin[idx, half:] = col_emb[c].sin()
                idx += 1
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached


def apply_rope_axial(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Independent RoPE on each half of the head dim (row | col)."""
    half = q.shape[-1] // 2
    qr, kr = apply_rope(q[..., :half], k[..., :half], cos[:, :half], sin[:, :half])
    qc, kc = apply_rope(q[..., half:], k[..., half:], cos[:, half:], sin[:, half:])
    return torch.cat([qr, qc], dim=-1), torch.cat([kr, kc], dim=-1)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, axial_rope: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.axial_rope = axial_rope
        self.qkv = CastedLinear(hidden_size, 3 * hidden_size, bias=False)
        self.out = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if cos_sin is not None:
            cos, sin = cos_sin[0][:s], cos_sin[1][:s]
            if self.axial_rope:
                q, k = apply_rope_axial(q, k, cos, sin)
            else:
                q, k = apply_rope(q, k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(b, s, -1)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float) -> None:
        super().__init__()
        inner = find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up = CastedLinear(hidden_size, inner * 2, bias=False)
        self.down = CastedLinear(inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        rms_eps: float,
        mlp_t: bool,
        mixer_seq_len: int,
        axial_rope: bool = False,
    ) -> None:
        super().__init__()
        self.mlp_t = mlp_t
        self.rms_eps = rms_eps
        if mlp_t:
            self.mixer = SwiGLU(mixer_seq_len, expansion)
            self.attn = None
            self.mlp = None
        else:
            self.attn = Attention(hidden_size, num_heads, axial_rope=axial_rope)
            self.mlp = SwiGLU(hidden_size, expansion)
            self.mixer = None

    def forward(self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        if self.mlp_t:
            xt = x.transpose(1, 2)
            return rms_norm(xt + self.mixer(xt), self.rms_eps).transpose(1, 2)
        x = rms_norm(x + self.attn(x, cos_sin), self.rms_eps)
        return rms_norm(x + self.mlp(x), self.rms_eps)
