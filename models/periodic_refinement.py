"""History-only refinement operators; no IDs, labels or directed-road claims."""
from __future__ import annotations

import torch
from torch import nn


def daily_block_mask(device=None):
    """Same 88 tokens, isolated recent/day/seven aligned-day/summary groups.

    These seven groups are 96-step same-phase snippets, NOT seven full days.
    Lower BERT layers share weights but cannot mix groups. Upper layers perform
    cross-day comparison. This changes the attention support, not token budget.
    """
    groups = torch.repeat_interleave(torch.arange(11, device=device),
        torch.tensor([16, 8]+[8]*7+[4, 4], device=device))
    return groups[:, None] == groups[None, :]


def relation_change_features(target, source):
    """[R,48], [R,K,48] -> [R,K,4,8], using observed changes only.

    Positive lag compares an earlier source with a later target, without claiming
    direction or causality. Source K/V content is never replaced by target data;
    these features are a separate target-conditioned attention bias.
    """
    target, source = target.float(), source.float()
    if target.shape[-1] != 48 or source.shape[-1] != 48:
        raise ValueError("relations require 48 observed steps")
    dt, ds = target.diff(dim=-1), source.diff(dim=-1)
    def correlation(a, b):
        a, b = a-a.mean(-1, keepdim=True), b-b.mean(-1, keepdim=True)
        return (a*b).mean(-1)/(a.square().mean(-1)*b.square().mean(-1)+1e-6).sqrt()
    lag = torch.stack([correlation(dt[:, None, lag:], ds[..., :47-lag])
                       for lag in (0, 3, 6, 12)], -1)
    patches = source.reshape(*source.shape[:-1], 4, 12)
    current = target[:, None, None, -12:]
    change_corr = correlation(current.diff(dim=-1), patches.diff(dim=-1))
    level_delta = (patches.mean(-1)-current.mean(-1)).tanh()
    scale_ratio = ((patches.var(-1, unbiased=False)+1e-4).sqrt()/
                   (current.var(-1, unbiased=False)+1e-4).sqrt()).log().clamp(-4, 4)/4
    phase = torch.linspace(-1, 0, 4, device=source.device).expand_as(level_delta)
    return torch.cat((lag[:, :, None].expand(-1, -1, 4, -1),
        torch.stack((change_corr, level_delta, scale_ratio, phase), -1)), -1)


class RelationChangeBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1))
        # Neutral at initialization, with a trainable path to the final layer.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, target, source):
        feature = relation_change_features(target, source)
        return 2*self.network(feature).squeeze(-1).tanh()


def fine_shape_correction(raw, bound=.5):
    """Bounded within-hour correction in train-standardized units.

    It cannot move an hourly mean: level/trend remains the numerical/retrieval
    route's job. Center before and after bounding to preserve zero initialization
    and exact zero hourly means (up to rounding). Maximum magnitude <=2*bound.
    """
    if raw.shape[-1] != 48 or bound <= 0:
        raise ValueError("fine correction requires 48 steps and positive bound")
    shape = raw.float().reshape(*raw.shape[:-1], 4, 12)
    shape = shape-shape.mean(-1, keepdim=True)
    shape = bound*torch.tanh(shape/bound)
    return (shape-shape.mean(-1, keepdim=True)).flatten(-2)
