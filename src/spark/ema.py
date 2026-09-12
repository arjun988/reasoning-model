"""EMA of model weights (TRM uses 0.999)."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(p.data, alpha=1.0 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state)
