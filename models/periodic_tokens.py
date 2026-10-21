"""Ordered content patches and causal multi-period tokenization for V4."""
from __future__ import annotations
from collections.abc import Iterable
import torch
from torch import nn
from torch.nn import functional as F
from models.periodic_memory_policy import mask_numeric_features


def periodic_numeric_features(history, short_state, baseline_prediction, *, anchor_shift_steps=0,
                              component_ablation=None, memory_policy="full"):
    """The 252 causal numeric features used by the checked ridge reference.

Baseline predictions are model outputs from history, never future observations.
Feature order: short128/recent48/day8/base4/anchors56/repeat4/deviation4.
"""
    b, length, n = history.shape
    if length != 2064 or short_state.shape != (b, n, 128) or baseline_prediction.shape != (b, 48, n):
        raise ValueError("invalid periodic numeric interface")
    if not -12 <= anchor_shift_steps <= 12:
        raise ValueError("anchor shift exceeds supported causal range")
    values = history.float().transpose(1, 2)
    recent = values[..., -48:]
    day = values[..., -288:].reshape(b, n, 8, 36).mean(-1)
    base = baseline_prediction.float().transpose(1, 2).reshape(b, n, 4, 12).mean(-1)
    anchors = []
    for lag in range(1, 8):
        aligned = length-288*lag+anchor_shift_steps
        if aligned-48 < 0 or aligned+48 > length:
            raise ValueError("anchor shift lacks causal context; use masked pattern candidates instead")
        anchors.append(values[..., aligned-48:aligned+48].reshape(b, n, 8, 12).mean(-1))
    anchors = torch.stack(anchors, 2)
    context = recent.reshape(b, n, 4, 12).mean(-1)
    distance = (anchors[..., :4]-context[:, :, None]).square().mean(-1)
    weights = (-distance/.1).softmax(-1)
    repeat = (weights[..., None]*anchors[..., 4:]).sum(2)
    deviation = context-(weights[..., None]*anchors[..., :4]).sum(2)
    if component_ablation == "zero_repeat": repeat = torch.zeros_like(repeat)
    if component_ablation == "zero_deviation": deviation = torch.zeros_like(deviation)
    return mask_numeric_features(torch.cat((short_state.float(), recent, day, base,
        anchors.flatten(2), repeat, deviation), -1), memory_policy)


class FixedPeriodicCalibration(nn.Module):
    """Train-only ridge numerical route; not credited to the PLM branch."""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(252))
        self.register_buffer("scale", torch.ones(252))
        self.register_buffer("weight", torch.zeros(4, 252))
        self.register_buffer("bias", torch.zeros(4))
        self.ready = False

    def install(self, state):
        with torch.no_grad():
            for name in ("mean", "scale", "weight", "bias"):
                value = torch.as_tensor(state[name], dtype=torch.float32, device=self.mean.device)
                if value.shape != getattr(self, name).shape or not torch.isfinite(value).all():
                    raise ValueError("invalid numeric calibration "+name)
                getattr(self, name).copy_(value)
            if not (self.scale > 0).all(): raise ValueError("numeric scales must be positive")
        self.ready = True

    def export(self):
        if not self.ready: raise RuntimeError("numeric calibration not installed")
        return {name: value.detach().cpu().clone() for name, value in self.named_buffers()}

    def forward(self, features):
        if not self.ready: raise RuntimeError("install train-only numeric calibration before forecasting")
        # Do not let outer BF16 autocast quantize the known numerical solution.
        with torch.autocast(features.device.type, enabled=False):
            return torch.nn.functional.linear((features.float()-self.mean)/self.scale, self.weight, self.bias)


class AlignedPeriodicPattern(nn.Module):
    """Retrieve old trajectories by context, then align level, scale and phase.

    All 35 candidate context/future pairs are strictly historical. Negative
    shifts with unavailable seventh-day context are masked, never wrapped.
    The controller sees a neural representation, not present forecast targets.
    """
    shifts = (-12, -6, 0, 6, 12)

    def __init__(self, width=32, *, context_only=False):
        super().__init__()
        self.context_only = context_only
        if context_only:
            self.query = nn.Linear(width, width)
            self.key = nn.Linear(width, width)
            self.amplitude = nn.Linear(width, 2)
            nn.init.zeros_(self.key.weight); nn.init.zeros_(self.key.bias)
            nn.init.zeros_(self.amplitude.weight); nn.init.zeros_(self.amplitude.bias)
        else:
            # Retained only for exact historical-checkpoint compatibility.
            self.control = nn.Linear(width, 7*len(self.shifts)+2)
            nn.init.zeros_(self.control.weight)
            nn.init.zeros_(self.control.bias)

    @classmethod
    def candidates(cls, history):
        if history.ndim != 2 or history.shape[1] != 2064:
            raise ValueError("pattern history must be [roads,2064]")
        start = (2064-288*torch.arange(1, 8, device=history.device))[:, None]
        start = start+torch.tensor(cls.shifts, device=history.device)[None]-48
        indices = start[..., None]+torch.arange(96, device=history.device)
        valid = (indices[..., 0] >= 0) & (indices[..., -1] < 2064)
        # Clipping only makes masked candidates safe to gather. They receive
        # exactly zero retrieval weight; the opposite history edge is not used.
        values = history.float()[:, indices.clamp(0, 2063)].flatten(1, 2)
        return values, valid.flatten(), indices

    def forward(self, history, state, *, candidate_keys=None, mode="normal"):
        if mode not in ("normal", "no_phase", "no_amplitude", "no_deviation"):
            raise ValueError("unknown pattern operator intervention")
        candidates, valid, _ = self.candidates(history)
        context, future = candidates[..., :48], candidates[..., 48:]
        current = history.float()[:, -48:]
        current_mean = current.mean(-1, keepdim=True)
        source_mean = context.mean(-1, keepdim=True)
        source_center = context-source_mean
        current_center = current-current_mean
        source_std = context.std(-1, unbiased=False, keepdim=True).clamp_min(.05)
        current_std = current.std(-1, unbiased=False, keepdim=True).clamp_min(.05)
        shape_error = ((source_center/source_std-current_center[:, None]/current_std[:, None])**2).mean(-1)
        if self.context_only:
            if candidate_keys is None or candidate_keys.shape != (len(history), 35, state.shape[-1]):
                raise ValueError("strict retrieval requires isolated context keys")
            query, key = self.query(state), self.key(candidate_keys)
            learned = torch.einsum("bd,bkd->bk", query.float(), key.float())/state.shape[-1]**.5
            amplitude = self.amplitude(state).float()
        else:
            controls = self.control(state).float()
            learned, amplitude = controls[:, :35], controls[:, 35:]
        if mode == "no_phase":
            valid = valid & (torch.arange(35, device=history.device) % 5 == 2)
        scores = learned-shape_error/.5
        weights = scores.masked_fill(~valid[None], -torch.inf).softmax(-1)
        affine_gain = ((source_center*current_center[:, None]).mean(-1, keepdim=True)
                       /source_center.square().mean(-1, keepdim=True).clamp_min(.0025)).clamp(.25, 4.)
        matched_context = affine_gain*source_center+current_mean[:, None]
        matched_future = affine_gain*(future-source_mean)+current_mean[:, None]
        if mode == "no_amplitude":
            matched_context, matched_future = context, future
        context_mix = (weights[..., None]*matched_context).sum(1)
        future_mix = (weights[..., None]*matched_future).sum(1)
        deviation = (current-context_mix)[:, -12:].mean(-1, keepdim=True)
        gain = (.5*amplitude[:, :1].tanh()).exp()
        offset = .5*amplitude[:, 1:2].tanh()
        if mode == "no_amplitude":
            gain, offset = torch.ones_like(gain), torch.zeros_like(offset)
        if mode == "no_deviation":
            deviation = torch.zeros_like(deviation)
        decay = (-torch.arange(48, device=history.device).float()/24).exp()[None]
        template = current_mean+gain*(future_mix-current_mean)+offset+deviation*decay
        return template, {"pattern_weights": weights, "pattern_gain": gain,
                          "pattern_offset": offset, "pattern_deviation": deviation}


class UtilityPeriodicRetrieval(nn.Module):
    """Raw historical outcomes; learned hourly choice includes an abstain slot.

    Keys are independently encoded contexts. Labels are never forward inputs.
    Phase alignment is candidate selection, not extrapolated amplitude fitting.
    """
    candidates = AlignedPeriodicPattern.candidates
    shifts = AlignedPeriodicPattern.shifts

    def __init__(self, width=32):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.horizons = nn.Parameter(torch.empty(4, width))
        self.abstain = nn.Linear(width, 4)
        nn.init.normal_(self.horizons, std=.02)
        nn.init.zeros_(self.abstain.weight)
        nn.init.constant_(self.abstain.bias, 2.)

    def forward(self, history, state, *, candidate_keys, reference_prediction, mode="normal"):
        if mode not in ("normal", "no_phase"):
            raise ValueError("utility retrieval has no amplitude/deviation operator")
        candidates, valid, _ = self.candidates(history)
        if candidate_keys.shape != (len(history), 35, state.shape[-1]):
            raise ValueError("utility retrieval requires independent context keys")
        if mode == "no_phase":
            valid = valid & (torch.arange(35, device=history.device) % 5 == 2)
        q = F.layer_norm(self.query(state), (state.shape[-1],))[:, None]+self.horizons[None]
        k = F.layer_norm(self.key(candidate_keys), (state.shape[-1],))
        scores = torch.einsum("bhd,bkd->bhk", q.float(), k.float())/state.shape[-1]**.5
        scores = scores.masked_fill(~valid[None,None], -1e4)
        logits = torch.cat((scores, self.abstain(state).float()[...,None]), -1)
        weights = logits.softmax(-1)
        # Keep all 48 original samples of every historical continuation.
        future = candidates[...,48:].reshape(len(history),35,4,12).permute(0,2,1,3)
        reference = reference_prediction.float().reshape(len(history),4,12)
        difference = (weights[...,:35,None]*(future-reference[:,:,None])).sum(2)
        template = reference+difference
        return template.flatten(1), {"utility_logits": logits,
            "pattern_weights": weights[...,:35], "abstain_probability": weights[...,-1]}


@torch.no_grad()
def utility_teacher(history, target, reference, node_std, *, loss_scale,
                    temperature=.025, improvement_margin=.005):
    """Training labels only: raw-error utility versus the fixed reference.

    Every historical continuation is available before the prediction origin.
    Current targets appear ONLY here, never in retrieval/model forward.
    Costs and margin use the same train-fitted global scale as forecast loss.
    """
    if loss_scale <= 0 or temperature <= 0 or improvement_margin < 0:
        raise ValueError("invalid utility teacher calibration")
    if target.shape != reference.shape or target.shape != (len(history),48):
        raise ValueError("utility target/reference shape mismatch")
    candidates, valid, _ = AlignedPeriodicPattern.candidates(history)
    future = candidates[...,48:].reshape(len(history),35,4,12).permute(0,2,1,3)
    truth = target.float().reshape(len(history),4,12)
    ref = reference.float().reshape(len(history),4,12)
    errors = (future-truth[:,:,None])*node_std[:,None,None,None]/loss_scale
    costs = errors.abs().mean(-1)+.1*errors.square().mean(-1)
    ref_errors = (ref-truth)*node_std[:,None,None]/loss_scale
    ref_cost = ref_errors.abs().mean(-1)+.1*ref_errors.square().mean(-1)
    improvement = ref_cost[...,None]-costs
    eligible = valid[None,None] & (improvement > improvement_margin)
    scores = (improvement/temperature).masked_fill(~eligible, -1e4)
    scores = torch.cat((scores, torch.zeros_like(scores[...,:1])), -1)
    return scores.softmax(-1).detach()


def utility_cross_entropy(logits, labels):
    if logits.shape != labels.shape or logits.shape[-2:] != (4,36):
        raise ValueError("invalid utility supervision shapes")
    return -(labels.detach()*logits.float().log_softmax(-1)).sum(-1).mean()


class OrderedPatchEncoder(nn.Module):
    """Shared length-independent encoder that retains four ordered shape bins."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.shape_conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv1d(16, 16, 3, padding=1), nn.GELU(),
        )
        self.shape_projection = nn.Linear(16 * 4, width)
        self.statistics_projection = nn.Linear(4, width)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # [..., L] -> [..., D]; keep content free of type, day, or position IDs.
        shape = patches.shape[:-1]
        flat = patches.reshape(-1, 1, patches.shape[-1])
        # Explicit adaptive bins avoid CUDA's nondeterministic adaptive-pool
        # backward while preserving the same ordered pooling boundaries.
        encoded = self.shape_conv(flat)
        length = encoded.shape[-1]
        bins = torch.stack([
            encoded[..., (i * length) // 4 : ((i + 1) * length + 3) // 4].mean(-1)
            for i in range(4)
        ], -1).flatten(1)
        stats = torch.stack((patches.mean(-1),
                             patches.float().std(-1, unbiased=False).to(patches.dtype),
                             patches[..., -1] - patches[..., 0], patches[..., -1]), -1)
        return self.shape_projection(bins).reshape(*shape, -1) + self.statistics_projection(stats)


class PeriodicTokenization:
    """A stateless mixin: every historical analogue ends before the origin."""

    history_length = 2064
    horizon = 48
    d_model = 32
    memory_tokens = 88

    @staticmethod
    def _patches(values: torch.Tensor, pieces: int) -> torch.Tensor:
        b, t, n = values.shape
        if t % pieces:
            raise ValueError("patch range must divide evenly")
        return values.reshape(b, pieces, t // pieces, n).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _stats(values: torch.Tensor, pieces: int) -> torch.Tensor:
        chunks = PeriodicTokenization._patches(values, pieces)
        return torch.stack((chunks.mean(-1),
                            chunks.float().std(-1, unbiased=False).to(chunks.dtype),
                            chunks[..., -1] - chunks[..., 0], chunks[..., -1]), -1)

    def tokenize(self, history: torch.Tensor, *, anchor_shift_steps: int = 0,
                 day_permutation: Iterable[int] | None = None):
        if history.ndim != 3 or history.shape[1] != self.history_length:
            raise ValueError(f"history must be [B,{self.history_length},N]")
        if not history.is_floating_point() or history.shape[0] < 1 or history.shape[2] < 1:
            raise ValueError("history must be a nonempty floating tensor")
        if not -12 <= anchor_shift_steps <= 12:
            raise ValueError("anchor shift must be at most one hour")
        recent = self._patches(history[:, -48:], 16)
        day = self._patches(history[:, -288:], 8)
        current_hour = self._patches(history[:, -48:], 4)
        anchors, anchor_stats = [], []
        for lag in range(1, 8):
            aligned = self.history_length - 288 * lag + anchor_shift_steps
            start, stop = aligned - 48, aligned + 48
            if start < 0 or stop > self.history_length:
                raise ValueError("anchor shift exceeds causal history")
            values = history[:, start:stop]
            anchors.append(self._patches(values, 8))
            anchor_stats.append(self._stats(values, 8))
        anchors, anchor_stats = torch.stack(anchors, 2), torch.stack(anchor_stats, 2)
        if day_permutation is not None:
            order = torch.as_tensor(list(day_permutation), dtype=torch.long, device=history.device)
            if order.shape != (7,) or not torch.equal(order.sort().values, torch.arange(7, device=history.device)):
                raise ValueError("day_permutation must contain each day once")
            anchors, anchor_stats = anchors[:, :, order], anchor_stats[:, :, order]
        current_stats = self._stats(history[:, -48:], 4)
        distance = (anchor_stats[:, :, :, :4].float() - current_stats[:, :, None].float()).square().mean((-1, -2))
        weights = torch.softmax(-distance / self.similarity_temperature, 2)
        return recent, day, current_hour, anchors, weights

