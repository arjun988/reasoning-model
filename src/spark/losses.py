"""Losses for TRM: stablemax CE + halt BCE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PAD_IGNORE = 0


def stablemax_logprobs(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = logits.double()
    s = torch.where(x < 0, 1.0 / (1.0 - x + 1e-30), x + 1.0)
    return torch.log(s / s.sum(dim=dim, keepdim=True))


def token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = PAD_IGNORE,
    kind: str = "stablemax_cross_entropy",
) -> torch.Tensor:
    """Mean per-sequence token loss, PAD ignored. Shape: [B]."""
    mask = labels != ignore_index
    if kind == "softmax_cross_entropy":
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.long().reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view(labels.shape)
        denom = mask.sum(dim=-1).clamp_min(1)
        return (loss * mask).sum(dim=-1) / denom

    logp = stablemax_logprobs(logits)
    gathered = logp.gather(-1, labels.long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
    nll = torch.where(mask, -gathered, torch.zeros_like(gathered))
    denom = mask.sum(dim=-1).clamp_min(1)
    return nll.sum(dim=-1) / denom


def halt_loss(halt_logit: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(halt_logit, exact.float(), reduction="none")


def act_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    halt_logit: torch.Tensor,
    ignore_index: int = PAD_IGNORE,
    kind: str = "stablemax_cross_entropy",
) -> tuple[torch.Tensor, dict[str, float]]:
    lm = token_loss(logits, labels, ignore_index=ignore_index, kind=kind)
    pred = logits.argmax(dim=-1)
    mask = labels != ignore_index
    correct = (pred == labels) | ~mask
    exact = correct.all(dim=-1)
    q = halt_loss(halt_logit, exact)
    loss = (lm + 0.5 * q).mean()
    cell = ((pred == labels) & mask).float().sum() / mask.float().sum().clamp_min(1)
    metrics = {
        "loss": float(loss.detach()),
        "lm_loss": float(lm.mean().detach()),
        "halt_loss": float(q.mean().detach()),
        "exact": float(exact.float().mean().detach()),
        "cell_acc": float(cell.detach()),
    }
    return loss, metrics
