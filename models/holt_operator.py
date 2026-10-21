"""Lightweight HOLT operator correction for spatio-temporal forecasting.

The module implements the method contracts needed by the HOLT pilots without
depending on a language model or on :mod:`models.ST_SepNet`:

* a history-conditioned, differentiable soft hypergraph;
* optional physical candidate support plus unrestricted ``escape`` slots;
* normalized node-to-edge (``P``) and edge-to-node (``Q``) maps;
* multi-source ordered low-rank operators in hyperedge space; and
* an exactly zero-initialized residual correction in node space.

There are deliberately no node-ID embeddings.  Every operation over nodes is
shared or a permutation-invariant reduction.  Consequently, permuting the
history's node axis and the rows of the physical candidate mask permutes the
node correction in exactly the same way.

``forward`` accepts historical observations only.  Static topology may be
registered at construction time or supplied as a candidate mask at inference;
neither path accepts forecast targets or future observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


HOLT_OPERATOR_VARIANTS = (
    "directed",
    "nodewise_temporal",
    "nodewise_dual_local",
    "nodewise_dual_offdiag",
)
HOLT_DUAL_VARIANTS = frozenset(
    {"nodewise_dual_local", "nodewise_dual_offdiag"}
)


def build_binary_undirected_offdiag_aggregation(
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Return a row-normalized one-hop neighbour map with a zero diagonal.

    The input's non-zero pattern is interpreted as physical connectivity.  It
    is symmetrized because the PEMS adjacency distributed with this project is
    an undirected distance graph, and its positive distance values must not be
    mistaken for message strengths.  Row ``n`` of the result averages only
    nodes other than ``n`` that are physically adjacent to it.
    """
    adjacency_tensor = torch.as_tensor(adjacency)
    if (
        adjacency_tensor.ndim != 2
        or adjacency_tensor.shape[0] != adjacency_tensor.shape[1]
    ):
        raise ValueError("adjacency must be square")
    if adjacency_tensor.shape[0] < 2:
        raise ValueError("off-diagonal aggregation requires at least two nodes")
    support = (adjacency_tensor != 0) | (adjacency_tensor.T != 0)
    support = support.clone()
    support.fill_diagonal_(False)
    degree = support.sum(dim=1)
    if (degree == 0).any():
        isolated = (degree == 0).nonzero(as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            "physical off-diagonal aggregation does not support isolated nodes; "
            f"found {isolated}"
        )
    weights = support.to(dtype=torch.float32)
    return weights / degree.to(dtype=weights.dtype).unsqueeze(1)


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return int(value)


def _signed_unit_norm(values: torch.Tensor, eps: float) -> torch.Tensor:
    """Return bounded signed factors with unit L2 norm along the last axis."""
    bounded = torch.tanh(values)
    return bounded / bounded.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)


@dataclass(frozen=True)
class SoftHypergraphState:
    """One window's differentiable hypergraph and its diagnostics.

    Attributes
    ----------
    membership:
        Node-hyperedge incidence ``[B,N,E]``.  In ``learned`` mode it is soft
        but preserves the fixed incidence's node mass; in ``fixed`` mode it is
        the original binary physical incidence.
    P:
        Degree-normalized node-to-edge pooling map ``[B,E,N]``.
    Q:
        Degree-normalized edge-to-node readout map ``[B,N,E]``.
    symmetric_incidence:
        ``D_v^-1/2 H D_e^-1/2`` with shape ``[B,N,E]``.  This is the
        normalized incidence expected by a symmetric HGCN propagation layer.
    membership_entropy:
        Mean (unnormalized) incidence entropy for every batch item ``[B]``.
    normalized_entropy:
        Entropy divided by the log number of candidates available to each
        node, averaged per batch item.  It lies in ``[0,1]`` up to round-off.
    edge_diversity:
        One minus the mean off-diagonal cosine similarity between membership
        columns.  Larger values indicate less redundant hyperedges.
    edge_mass:
        Fraction of total membership assigned to every hyperedge ``[B,E]``.
    blend_rho:
        Effective fixed/dynamic probability-mixture coefficient ``[B]``.
    node_mass_error:
        Mean absolute deviation from the fixed incidence's per-node mass.
    edge_mass_drift:
        Total-variation distance from the fixed incidence's edge-mass profile.
    membership_l1_drift:
        Total absolute membership change from the fixed incidence, normalized
        by twice the fixed total membership mass.  This distinguishes an
        actually changing H from a learned scorer whose aggregate edge masses
        happen to remain unchanged.
    """

    membership: torch.Tensor
    P: torch.Tensor
    Q: torch.Tensor
    symmetric_incidence: torch.Tensor
    membership_entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    edge_diversity: torch.Tensor
    edge_mass: torch.Tensor
    physical_mass: torch.Tensor
    escape_mass: torch.Tensor
    blend_rho: torch.Tensor
    node_mass_error: torch.Tensor
    edge_mass_drift: torch.Tensor
    membership_l1_drift: torch.Tensor


@dataclass(frozen=True)
class HOLTEncodedState:
    """Reusable history encoding and its single source-of-truth hypergraph."""

    patch_node_features: torch.Tensor
    pooled_node_features: torch.Tensor
    hypergraph: SoftHypergraphState


class SharedHistoryNodeEncoder(nn.Module):
    """Encode each node history with one shared temporal network.

    The encoder has no parameter whose size depends on the number of nodes.
    It combines a small temporal convolution with scale-aware window
    statistics.  All convolutions see only the supplied historical window.
    """

    def __init__(self, d_model: int, num_temporal_patches: int = 8):
        super().__init__()
        self.d_model = _positive_int(d_model, "d_model")
        self.num_temporal_patches = _positive_int(
            num_temporal_patches, "num_temporal_patches"
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(1, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stat_projection = nn.Linear(5, self.d_model)
        self.patch_normalization = nn.LayerNorm(self.d_model)
        self.pooled_normalization = nn.LayerNorm(self.d_model)
        position = torch.zeros(self.num_temporal_patches, self.d_model)
        patch_index = torch.arange(self.num_temporal_patches, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / self.d_model)
        )
        position[:, 0::2] = torch.sin(patch_index[:, None] * frequencies)
        if self.d_model > 1:
            position[:, 1::2] = torch.cos(
                patch_index[:, None] * frequencies[: position[:, 1::2].shape[1]]
            )
        self.register_buffer("patch_position", position, persistent=False)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return patch ``[B,P,N,D]`` and pooled ``[B,N,D]`` features."""
        if history.ndim != 3:
            raise ValueError("history must have shape [batch,time,nodes]")
        if history.shape[1] < 1 or history.shape[2] < 1:
            raise ValueError("history must contain at least one time step and node")
        if not history.is_floating_point():
            raise TypeError("history must be a floating-point tensor")

        batch, steps, nodes = history.shape
        sequences = history.transpose(1, 2).reshape(batch * nodes, 1, steps)
        temporal_sequence = self.temporal(sequences)
        patch_temporal = F.adaptive_avg_pool1d(
            temporal_sequence, self.num_temporal_patches
        ).reshape(
            batch, nodes, self.d_model, self.num_temporal_patches
        )
        patch_temporal = patch_temporal.permute(0, 3, 1, 2)
        patch_position = self.patch_position.to(dtype=history.dtype).view(
            1, self.num_temporal_patches, 1, self.d_model
        )
        patch_features = self.patch_normalization(patch_temporal + patch_position)
        pooled_temporal = temporal_sequence.mean(dim=-1).reshape(
            batch, nodes, self.d_model
        )

        node_history = history.transpose(1, 2)
        mean = node_history.mean(dim=-1)
        std = node_history.std(dim=-1, unbiased=False)
        first = node_history[..., 0]
        last = node_history[..., -1]
        if steps == 1:
            slope = torch.zeros_like(mean)
        else:
            coordinates = torch.linspace(
                -1.0,
                1.0,
                steps,
                dtype=history.dtype,
                device=history.device,
            )
            denominator = coordinates.square().sum().clamp_min(
                torch.finfo(history.dtype).eps
            )
            slope = torch.einsum(
                "bnt,t->bn", node_history - mean.unsqueeze(-1), coordinates
            ) / denominator
        statistics = torch.stack([mean, std, last, last - first, slope], dim=-1)
        pooled_features = self.pooled_normalization(
            pooled_temporal + self.stat_projection(statistics)
        )
        return patch_features, pooled_features


class WindowConditionedSoftHypergraph(nn.Module):
    """Construct a differentiable hypergraph from one history window.

    In ``learned`` mode, physical hyperedges are represented only by their
    candidate memberships; there are no learned node or physical-edge ID
    embeddings.  Escape slots are global latent slots that let the forecasting
    loss discover a grouping outside physical support.  ``fixed`` mode instead
    returns the exact binary physical incidence, has no membership logits and
    requires zero escape slots, making it a controlled fixed-H baseline.

    To prevent a normalization confound, learned memberships are mixed in
    row-probability space and then rescaled to the fixed incidence's node mass:
    ``Qw=(1-rho)Q0+rho*Qdynamic`` and ``Hw=Dv0*Qw``.  Therefore ``rho=0``
    exactly recovers fixed ``H/P/Q/symmetric_incidence`` rather than merely
    recovering the same edge-to-node row probabilities.
    """

    def __init__(
        self,
        d_model: int,
        num_physical_hyperedges: int,
        num_escape_slots: int | None = None,
        temperature: float = 1.0,
        eps: float = 1e-6,
        physical_candidate_mask: torch.Tensor | None = None,
        hypergraph_mode: str = "learned",
        initial_blend_rho: float = 0.1,
    ):
        super().__init__()
        self.d_model = _positive_int(d_model, "d_model")
        self.num_physical_hyperedges = _positive_int(
            num_physical_hyperedges,
            "num_physical_hyperedges",
            allow_zero=True,
        )
        if hypergraph_mode not in {"learned", "fixed"}:
            raise ValueError("hypergraph_mode must be 'learned' or 'fixed'")
        if num_escape_slots is None:
            num_escape_slots = 0 if hypergraph_mode == "fixed" else 2
        self.num_escape_slots = _positive_int(
            num_escape_slots, "num_escape_slots", allow_zero=True
        )
        if hypergraph_mode == "fixed" and self.num_escape_slots:
            raise ValueError("fixed hypergraph_mode requires num_escape_slots=0")
        self.hypergraph_mode = hypergraph_mode
        if self.num_physical_hyperedges + self.num_escape_slots < 1:
            raise ValueError("at least one physical hyperedge or escape slot is required")
        if not isinstance(temperature, (float, int)) or temperature <= 0:
            raise ValueError("temperature must be positive")
        if not isinstance(eps, (float, int)) or eps <= 0:
            raise ValueError("eps must be positive")
        if (
            not isinstance(initial_blend_rho, (float, int))
            or not 0.0 < float(initial_blend_rho) < 1.0
        ):
            raise ValueError("initial_blend_rho must lie strictly inside (0,1)")
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.initial_blend_rho = float(initial_blend_rho)

        if self.hypergraph_mode == "learned":
            self.node_key = nn.Linear(self.d_model, self.d_model, bias=False)
            self.edge_key = nn.Linear(self.d_model, self.d_model, bias=False)
            self.edge_bias = nn.Linear(self.d_model, 1, bias=False)
            self.logit_scale = nn.Parameter(torch.zeros(()))
            # Optimize unconstrained log odds.  Unlike a directly clamped rho,
            # sigmoid naturally attenuates its gradient near the boundaries
            # and cannot hit 0/1 at a finite mathematical logit.
            initial_logit = math.log(
                self.initial_blend_rho / (1.0 - self.initial_blend_rho)
            )
            self.blend_rho_logit = nn.Parameter(
                torch.tensor(initial_logit, dtype=torch.float32)
            )
        else:
            self.node_key = None
            self.edge_key = None
            self.edge_bias = None
            self.register_parameter("logit_scale", None)
            self.register_parameter("blend_rho_logit", None)
        if self.hypergraph_mode == "learned" and self.num_escape_slots:
            self.escape_queries = nn.Parameter(
                torch.empty(self.num_escape_slots, self.d_model)
            )
            nn.init.normal_(self.escape_queries, std=self.d_model**-0.5)
        else:
            self.register_parameter("escape_queries", None)

        # NaN means "use the trainable sigmoid(logit)".  A finite override is
        # deliberately non-persistent and exists only for controlled nulls or
        # ablations; optimizers never update it.
        self.register_buffer(
            "_blend_rho_override", torch.tensor(float("nan")), persistent=False
        )

        if physical_candidate_mask is None:
            default_mask = torch.empty(
                0, self.num_physical_hyperedges, dtype=torch.bool
            )
        else:
            default_mask = self._validate_candidate_mask(
                physical_candidate_mask, expected_nodes=None
            )
        self.register_buffer("physical_candidate_mask", default_mask)

    @property
    def num_hyperedges(self) -> int:
        return self.num_physical_hyperedges + self.num_escape_slots

    @property
    def uses_absolute_node_ids(self) -> bool:
        return False

    @property
    def effective_blend_rho(self) -> torch.Tensor:
        """Return the trainable mixture coefficient constrained to ``[0,1]``."""
        if self.blend_rho_logit is None:
            return torch.zeros(
                (), dtype=torch.float32, device=self.physical_candidate_mask.device
            )
        learned_rho = torch.sigmoid(self.blend_rho_logit)
        override = self._blend_rho_override.to(dtype=learned_rho.dtype)
        return torch.where(torch.isnan(override), learned_rho, override)

    @property
    def blend_rho_override_active(self) -> bool:
        return bool(torch.isfinite(self._blend_rho_override).item())

    def set_blend_rho_override(self, value: float | None) -> None:
        """Force an exact rho for a controlled null; ``None`` resumes learning."""
        if self.hypergraph_mode != "learned":
            raise RuntimeError("blend-rho overrides are available only in learned mode")
        if value is None:
            self._blend_rho_override.fill_(float("nan"))
            return
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("blend-rho override must lie in [0,1] or be None")
        self._blend_rho_override.fill_(float(value))

    def _validate_candidate_mask(
        self,
        mask: torch.Tensor,
        expected_nodes: int | None,
    ) -> torch.Tensor:
        candidate = torch.as_tensor(mask)
        if candidate.ndim != 2:
            raise ValueError("physical_candidate_mask must have shape [nodes,edges]")
        if candidate.shape[1] != self.num_physical_hyperedges:
            raise ValueError(
                "physical_candidate_mask has the wrong number of physical hyperedges"
            )
        if expected_nodes is not None and candidate.shape[0] != expected_nodes:
            raise ValueError("physical_candidate_mask has the wrong number of nodes")
        if candidate.is_floating_point() and not torch.isfinite(candidate).all():
            raise ValueError("physical_candidate_mask must be finite")
        candidate = candidate > 0
        if self.num_physical_hyperedges and not candidate.any(dim=0).all():
            raise ValueError("every physical hyperedge needs at least one candidate node")
        if not self.num_escape_slots and not candidate.any(dim=1).all():
            raise ValueError(
                "every node needs a physical candidate when escape slots are disabled"
            )
        return candidate

    def _resolve_candidate_mask(
        self,
        nodes: int,
        device: torch.device,
        physical_candidate_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if physical_candidate_mask is not None:
            candidate = self._validate_candidate_mask(
                physical_candidate_mask, expected_nodes=nodes
            )
        elif self.physical_candidate_mask.numel():
            candidate = self._validate_candidate_mask(
                self.physical_candidate_mask, expected_nodes=nodes
            )
        else:
            candidate = torch.ones(
                nodes,
                self.num_physical_hyperedges,
                dtype=torch.bool,
                device=device,
            )
        return candidate.to(device=device)

    def forward(
        self,
        node_features: torch.Tensor,
        physical_candidate_mask: torch.Tensor | None = None,
    ) -> SoftHypergraphState:
        """Build ``H``, ``P`` and ``Q`` from history-derived node features."""
        if node_features.ndim != 3 or node_features.shape[-1] != self.d_model:
            raise ValueError(
                f"node_features must have shape [batch,nodes,{self.d_model}]"
            )
        batch, nodes, _ = node_features.shape
        candidate = self._resolve_candidate_mask(
            nodes, node_features.device, physical_candidate_mask
        )
        mask_values = candidate.to(dtype=node_features.dtype)

        if self.num_escape_slots:
            allowed = torch.cat(
                [
                    candidate,
                    torch.ones(
                        nodes,
                        self.num_escape_slots,
                        dtype=torch.bool,
                        device=node_features.device,
                    ),
                ],
                dim=1,
            )
        else:
            allowed = candidate

        if self.num_escape_slots:
            base_membership = torch.cat(
                [
                    mask_values,
                    torch.zeros(
                        nodes,
                        self.num_escape_slots,
                        dtype=node_features.dtype,
                        device=node_features.device,
                    ),
                ],
                dim=1,
            )
        else:
            base_membership = mask_values
        base_node_degree = base_membership.sum(dim=1)
        orphan = base_node_degree == 0
        reference_node_degree = torch.where(
            orphan, torch.ones_like(base_node_degree), base_node_degree
        )
        base_Q = base_membership / reference_node_degree.unsqueeze(-1)
        if orphan.any():
            # Only learned mode with escape slots can reach this branch.  An
            # orphan receives a deterministic uniform escape-only base state.
            fallback_Q = torch.zeros_like(base_Q)
            fallback_Q[:, self.num_physical_hyperedges :] = (
                1.0 / self.num_escape_slots
            )
            base_Q = torch.where(orphan.unsqueeze(-1), fallback_Q, base_Q)
        reference_membership = reference_node_degree.unsqueeze(-1) * base_Q

        if self.hypergraph_mode == "fixed":
            # The fixed control is the original binary physical incidence H.
            # It deliberately has no window-dependent or trainable logits.
            membership = reference_membership.unsqueeze(0).expand(batch, -1, -1)
            blend_rho_scalar = node_features.new_zeros(())
        else:
            edge_contexts: list[torch.Tensor] = []
            if self.num_physical_hyperedges:
                counts = mask_values.sum(dim=0).clamp_min(1.0)
                physical_context = torch.einsum(
                    "ne,bnd->bed", mask_values, node_features
                ) / counts.view(1, -1, 1)
                edge_contexts.append(physical_context)
            if self.num_escape_slots:
                global_context = node_features.mean(dim=1, keepdim=True)
                escape_context = global_context + self.escape_queries.to(
                    dtype=node_features.dtype
                ).unsqueeze(0)
                edge_contexts.append(escape_context.expand(batch, -1, -1))
            edge_context = torch.cat(edge_contexts, dim=1)

            node_key = F.normalize(
                self.node_key(node_features), dim=-1, eps=self.eps
            )
            edge_key = F.normalize(
                self.edge_key(edge_context), dim=-1, eps=self.eps
            )
            scale = self.logit_scale.exp().clamp(max=100.0)
            logits = scale * torch.einsum("bnd,bed->bne", node_key, edge_key)
            logits = logits + self.edge_bias(edge_context).transpose(1, 2)
            masked_logits = logits.masked_fill(
                ~allowed.unsqueeze(0), torch.finfo(logits.dtype).min
            )
            dynamic_Q = torch.softmax(masked_logits / self.temperature, dim=-1)
            blend_rho_scalar = self.effective_blend_rho.to(
                dtype=node_features.dtype
            )
            mixed_Q = (
                (1.0 - blend_rho_scalar) * base_Q.unsqueeze(0)
                + blend_rho_scalar * dynamic_Q
            )
            # Restore the binary H0 node mass after mixing in probability
            # space.  This makes rho=0 exactly identical to fixed H0 for H,
            # P, Q and symmetric incidence, rather than only for Q.
            membership = reference_node_degree.view(1, nodes, 1) * mixed_Q

        raw_edge_degree = membership.sum(dim=1)
        raw_node_degree = membership.sum(dim=2)
        edge_degree = raw_edge_degree.clamp_min(self.eps)
        node_degree = raw_node_degree.clamp_min(self.eps)
        P = membership.transpose(1, 2) / edge_degree.unsqueeze(-1)
        Q = membership / node_degree.unsqueeze(-1)
        symmetric_incidence = membership / (
            node_degree.sqrt().unsqueeze(-1) * edge_degree.sqrt().unsqueeze(1)
        )

        entropy_per_node = -(
            Q * Q.clamp_min(self.eps).log()
        ).sum(dim=-1)
        membership_entropy = entropy_per_node.mean(dim=1)
        candidate_count = allowed.sum(dim=1).to(dtype=node_features.dtype)
        entropy_denominator = candidate_count.clamp_min(2.0).log()
        normalized_per_node = torch.where(
            candidate_count > 1,
            entropy_per_node / entropy_denominator,
            torch.zeros_like(entropy_per_node),
        )
        normalized_entropy = normalized_per_node.mean(dim=1)

        profiles = F.normalize(
            membership.transpose(1, 2), dim=-1, eps=self.eps
        )
        if self.num_hyperedges > 1:
            similarities = torch.bmm(profiles, profiles.transpose(1, 2))
            active = membership.sum(dim=1) > self.eps
            pair_mask = active.unsqueeze(2) & active.unsqueeze(1)
            pair_mask = pair_mask & ~torch.eye(
                self.num_hyperedges,
                dtype=torch.bool,
                device=node_features.device,
            ).unsqueeze(0)
            off_diagonal_sum = (similarities * pair_mask).sum(dim=(1, 2))
            pairs = pair_mask.sum(dim=(1, 2))
            edge_diversity = torch.where(
                pairs > 0,
                1.0 - off_diagonal_sum / pairs.clamp_min(1),
                torch.zeros_like(off_diagonal_sum),
            )
        else:
            edge_diversity = torch.zeros(
                batch, dtype=node_features.dtype, device=node_features.device
            )

        edge_mass = raw_edge_degree / raw_edge_degree.sum(
            dim=1, keepdim=True
        ).clamp_min(
            self.eps
        )
        base_edge_degree = reference_membership.sum(dim=0)
        base_edge_mass = base_edge_degree / base_edge_degree.sum().clamp_min(self.eps)
        node_mass_error = (
            raw_node_degree - reference_node_degree.unsqueeze(0)
        ).abs().mean(dim=1)
        edge_mass_drift = 0.5 * (
            edge_mass - base_edge_mass.view(1, -1)
        ).abs().sum(dim=1)
        membership_l1_drift = 0.5 * (
            membership - reference_membership.unsqueeze(0)
        ).abs().sum(dim=(1, 2)) / reference_membership.sum().clamp_min(self.eps)
        blend_rho = blend_rho_scalar.expand(batch)
        physical_mass = edge_mass[:, : self.num_physical_hyperedges].sum(dim=1)
        escape_mass = edge_mass[:, self.num_physical_hyperedges :].sum(dim=1)
        return SoftHypergraphState(
            membership=membership,
            P=P,
            Q=Q,
            symmetric_incidence=symmetric_incidence,
            membership_entropy=membership_entropy,
            normalized_entropy=normalized_entropy,
            edge_diversity=edge_diversity,
            edge_mass=edge_mass,
            physical_mass=physical_mass,
            escape_mass=escape_mass,
            blend_rho=blend_rho,
            node_mass_error=node_mass_error,
            edge_mass_drift=edge_mass_drift,
            membership_l1_drift=membership_l1_drift,
        )


class MultiSourceLowRankOperatorHead(nn.Module):
    """Read from multiple source modes and write through response modes.

    For operator ``r``, source head ``q`` and historical patch ``p``, the
    induced hyperedge-space mapping has the ordered rank-one form
    ``u_r v_{r,q,p}^T``.  Source footprints and activations are computed before
    temporal aggregation, so ``Q`` heads can represent different spatial
    sources across lags while sharing the operator's response footprint.
    """

    def __init__(
        self,
        d_model: int,
        forecast_horizon: int,
        num_operators: int = 8,
        num_source_heads: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = _positive_int(d_model, "d_model")
        self.forecast_horizon = _positive_int(
            forecast_horizon, "forecast_horizon"
        )
        self.num_operators = _positive_int(num_operators, "num_operators")
        self.num_source_heads = _positive_int(
            num_source_heads, "num_source_heads"
        )
        self.eps = float(eps)

        self.source_key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.response_key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.source_queries = nn.Parameter(
            torch.empty(self.num_operators, self.num_source_heads, self.d_model)
        )
        self.response_queries = nn.Parameter(
            torch.empty(self.num_operators, self.d_model)
        )
        nn.init.normal_(self.source_queries, std=self.d_model**-0.5)
        nn.init.normal_(self.response_queries, std=self.d_model**-0.5)
        self.source_decoder = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )
        self.temporal_key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.horizon_queries = nn.Parameter(
            torch.empty(self.forecast_horizon, self.d_model)
        )
        nn.init.normal_(self.horizon_queries, std=self.d_model**-0.5)
        self.source_head_logits = nn.Parameter(
            torch.zeros(self.num_operators, self.num_source_heads)
        )
        self.residual_gate = nn.Parameter(torch.zeros(self.forecast_horizon))

    @property
    def uses_absolute_node_ids(self) -> bool:
        return False

    def forward(
        self,
        pooled_node_features: torch.Tensor,
        state: SoftHypergraphState,
        patch_node_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return node correction ``[B,H,N]`` and operator diagnostics."""
        if (
            pooled_node_features.ndim != 3
            or pooled_node_features.shape[-1] != self.d_model
        ):
            raise ValueError(
                f"pooled_node_features must have shape [batch,nodes,{self.d_model}]"
            )
        if patch_node_features is None:
            patch_node_features = pooled_node_features.unsqueeze(1)
        if (
            patch_node_features.ndim != 4
            or patch_node_features.shape[0] != pooled_node_features.shape[0]
            or patch_node_features.shape[2:] != pooled_node_features.shape[1:]
        ):
            raise ValueError(
                "patch_node_features must have shape [batch,patches,nodes,d_model]"
            )
        if state.P.shape[0] != pooled_node_features.shape[0]:
            raise ValueError(
                "hypergraph state and pooled_node_features batch sizes differ"
            )
        if state.P.shape[2] != pooled_node_features.shape[1]:
            raise ValueError(
                "hypergraph state and pooled_node_features node counts differ"
            )

        edge_features = torch.einsum(
            "ben,bnd->bed", state.P, pooled_node_features
        )
        edge_patch_states = torch.einsum(
            "ben,bpnd->bped", state.P, patch_node_features
        )
        source_keys = F.normalize(
            self.source_key(edge_patch_states), dim=-1, eps=self.eps
        )
        response_keys = F.normalize(
            self.response_key(edge_features), dim=-1, eps=self.eps
        )
        source_queries = F.normalize(
            self.source_queries, dim=-1, eps=self.eps
        )
        response_queries = F.normalize(
            self.response_queries, dim=-1, eps=self.eps
        )

        source_logits = math.sqrt(self.d_model) * torch.einsum(
            "bped,rqd->brqpe", source_keys, source_queries
        )
        response_logits = math.sqrt(self.d_model) * torch.einsum(
            "bed,rd->bre", response_keys, response_queries
        )
        source_weights = _signed_unit_norm(source_logits, self.eps)
        response_weights = _signed_unit_norm(response_logits, self.eps)

        source_context = torch.einsum(
            "brqpe,bped->brqpd", source_weights, edge_patch_states
        )
        source_activation = self.source_decoder(source_context).squeeze(-1)
        temporal_keys = F.normalize(
            self.temporal_key(source_context), dim=-1, eps=self.eps
        )
        horizon_queries = F.normalize(
            self.horizon_queries, dim=-1, eps=self.eps
        )
        temporal_logits = math.sqrt(self.d_model) * torch.einsum(
            "brqpd,hd->brqhp", temporal_keys, horizon_queries
        )
        temporal_weights = torch.softmax(temporal_logits, dim=-1)
        amplitudes = torch.einsum(
            "brqhp,brqp->brqh", temporal_weights, source_activation
        )
        head_mix = torch.softmax(self.source_head_logits, dim=-1)
        amplitudes = amplitudes * head_mix.view(
            1, self.num_operators, self.num_source_heads, 1
        )
        edge_delta = torch.einsum(
            "brqh,bre->bhe", amplitudes, response_weights
        ) / self.num_operators
        raw_correction = torch.einsum("bne,bhe->bhn", state.Q, edge_delta)
        gate = torch.tanh(self.residual_gate).view(1, -1, 1)
        correction = gate * raw_correction

        diagnostics = {
            "edge_features": edge_features,
            "edge_patch_states": edge_patch_states,
            "source_weights": source_weights,
            "response_weights": response_weights,
            "source_activation": source_activation,
            "temporal_weights": temporal_weights,
            "source_amplitudes": amplitudes,
            "raw_correction": raw_correction,
            "residual_gate": torch.tanh(self.residual_gate),
        }
        return correction, diagnostics


class NodeWiseTemporalControlHead(MultiSourceLowRankOperatorHead):
    """Parameter-exact temporal control with no cross-node value mixing.

    This qualification control deliberately inherits the complete directed
    operator parameterization.  Every trainable tensor is used, but all
    reductions are over temporal patches, source heads, or operator atoms --
    never over nodes.  Consequently, conditional on the history supplied to
    this head, changing node ``j`` cannot change another node's correction.

    The control separates two explanations for an observed HOLT gain:

    * access to a shared long-history encoder and a flexible residual head;
    * genuine source-to-response mixing between different nodes.

    It is parameter matched, not FLOP matched.  Runtime and memory therefore
    remain measurements rather than assumed equalities.
    """

    @property
    def mixes_values_across_nodes(self) -> bool:
        return False

    def forward(
        self,
        pooled_node_features: torch.Tensor,
        state: SoftHypergraphState,
        patch_node_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return an independent temporal correction for every node.

        ``state`` is accepted to preserve the directed head API and artifact
        contract, but its pooling/readout maps are intentionally not consumed.
        The returned edge-axis diagnostics use the node axis directly.
        """
        if (
            pooled_node_features.ndim != 3
            or pooled_node_features.shape[-1] != self.d_model
        ):
            raise ValueError(
                f"pooled_node_features must have shape [batch,nodes,{self.d_model}]"
            )
        if patch_node_features is None:
            patch_node_features = pooled_node_features.unsqueeze(1)
        if (
            patch_node_features.ndim != 4
            or patch_node_features.shape[0] != pooled_node_features.shape[0]
            or patch_node_features.shape[2:] != pooled_node_features.shape[1:]
        ):
            raise ValueError(
                "patch_node_features must have shape [batch,patches,nodes,d_model]"
            )
        if state.P.shape[0] != pooled_node_features.shape[0]:
            raise ValueError(
                "hypergraph state and pooled_node_features batch sizes differ"
            )
        if state.P.shape[2] != pooled_node_features.shape[1]:
            raise ValueError(
                "hypergraph state and pooled_node_features node counts differ"
            )

        # All feature transforms below are shared across nodes and act on the
        # final embedding axis only.  In particular, do not unit-normalize over
        # the node axis: that would let one node change every other node's
        # denominator and would invalidate this control.
        source_keys = F.normalize(
            self.source_key(patch_node_features), dim=-1, eps=self.eps
        )
        response_keys = F.normalize(
            self.response_key(pooled_node_features), dim=-1, eps=self.eps
        )
        source_queries = F.normalize(
            self.source_queries, dim=-1, eps=self.eps
        )
        response_queries = F.normalize(
            self.response_queries, dim=-1, eps=self.eps
        )
        source_logits = math.sqrt(self.d_model) * torch.einsum(
            "bpnd,rqd->brqpn", source_keys, source_queries
        )
        response_logits = math.sqrt(self.d_model) * torch.einsum(
            "bnd,rd->brn", response_keys, response_queries
        )
        source_weights = torch.tanh(source_logits)
        response_weights = torch.tanh(response_logits)

        local_patch_activation = self.source_decoder(
            patch_node_features
        ).squeeze(-1)
        source_activation = (
            source_weights * local_patch_activation[:, None, None, :, :]
        )
        temporal_keys = F.normalize(
            self.temporal_key(patch_node_features), dim=-1, eps=self.eps
        )
        horizon_queries = F.normalize(
            self.horizon_queries, dim=-1, eps=self.eps
        )
        base_temporal_logits = math.sqrt(self.d_model) * torch.einsum(
            "bpnd,hd->bhpn", temporal_keys, horizon_queries
        )
        # Source-role evidence also conditions which patches are useful, but
        # only for that same node.  Softmax is exclusively over patch index p.
        temporal_logits = (
            base_temporal_logits[:, None, None, :, :, :]
            + source_logits[:, :, :, None, :, :]
        )
        temporal_weights = torch.softmax(temporal_logits, dim=-2)
        amplitudes = torch.einsum(
            "brqhpn,brqpn->brqhn", temporal_weights, source_activation
        )
        head_mix = torch.softmax(self.source_head_logits, dim=-1).view(
            1, self.num_operators, self.num_source_heads, 1, 1
        )
        raw_correction = (
            amplitudes
            * head_mix
            * response_weights[:, :, None, None, :]
        ).sum(dim=(1, 2)) / self.num_operators
        gate = torch.tanh(self.residual_gate).view(1, -1, 1)
        correction = gate * raw_correction

        diagnostics = {
            # These aliases keep downstream diagnostic code schema-compatible;
            # their final axis is a node, not a pooled hyperedge.
            "edge_features": pooled_node_features,
            "edge_patch_states": patch_node_features,
            "source_weights": source_weights,
            "response_weights": response_weights,
            "source_activation": source_activation,
            "temporal_weights": temporal_weights,
            "source_amplitudes": amplitudes,
            "raw_correction": raw_correction,
            "residual_gate": torch.tanh(self.residual_gate),
        }
        return correction, diagnostics


class HOLTCorrectionHead(nn.Module):
    """End-to-end history-only HOLT residual head.

    Parameters are independent of the number of nodes.  A physical candidate
    mask may be registered once or supplied to :meth:`forward`; in either case
    it is structural support, not a learned node identity.
    """

    def __init__(
        self,
        forecast_horizon: int,
        num_physical_hyperedges: int | None = None,
        *,
        physical_candidate_mask: torch.Tensor | None = None,
        physical_offdiag_aggregation: torch.Tensor | None = None,
        num_escape_slots: int | None = None,
        d_model: int = 32,
        num_operators: int = 8,
        num_source_heads: int = 4,
        num_temporal_patches: int = 8,
        hypergraph_mode: str = "learned",
        operator_variant: str = "directed",
        initial_blend_rho: float = 0.1,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if physical_candidate_mask is not None:
            mask = torch.as_tensor(physical_candidate_mask)
            if mask.ndim != 2:
                raise ValueError(
                    "physical_candidate_mask must have shape [nodes,edges]"
                )
            inferred_edges = int(mask.shape[1])
            if (
                num_physical_hyperedges is not None
                and num_physical_hyperedges != inferred_edges
            ):
                raise ValueError(
                    "num_physical_hyperedges disagrees with physical_candidate_mask"
                )
            num_physical_hyperedges = inferred_edges
        if num_physical_hyperedges is None:
            raise ValueError(
                "provide num_physical_hyperedges or physical_candidate_mask"
            )
        if hypergraph_mode not in {"learned", "fixed"}:
            raise ValueError("hypergraph_mode must be 'learned' or 'fixed'")
        if operator_variant not in HOLT_OPERATOR_VARIANTS:
            raise ValueError(
                f"operator_variant must be one of {HOLT_OPERATOR_VARIANTS}"
            )
        if (
            operator_variant != "directed"
            and hypergraph_mode != "fixed"
        ):
            raise ValueError(
                "nodewise qualification controls require "
                "hypergraph_mode='fixed'"
            )
        if num_escape_slots is None:
            num_escape_slots = 0 if hypergraph_mode == "fixed" else 2
        if hypergraph_mode == "fixed" and num_escape_slots != 0:
            raise ValueError("fixed hypergraph_mode requires num_escape_slots=0")

        self.forecast_horizon = _positive_int(
            forecast_horizon, "forecast_horizon"
        )
        self.d_model = _positive_int(d_model, "d_model")
        self.hypergraph_mode = hypergraph_mode
        self.operator_variant = operator_variant
        node_count = (
            int(torch.as_tensor(physical_candidate_mask).shape[0])
            if physical_candidate_mask is not None
            else None
        )
        if operator_variant in HOLT_DUAL_VARIANTS:
            if physical_offdiag_aggregation is None:
                raise ValueError(
                    "dual nodewise qualification controls require "
                    "physical_offdiag_aggregation"
                )
            offdiag_aggregation = torch.as_tensor(
                physical_offdiag_aggregation, dtype=torch.float32
            )
            if node_count is None or offdiag_aggregation.shape != (
                node_count, node_count
            ):
                raise ValueError(
                    "physical_offdiag_aggregation must have shape [nodes,nodes]"
                )
            if not torch.isfinite(offdiag_aggregation).all():
                raise ValueError("physical_offdiag_aggregation must be finite")
            if (offdiag_aggregation < 0).any():
                raise ValueError(
                    "physical_offdiag_aggregation must be non-negative"
                )
            if not torch.equal(
                offdiag_aggregation.diagonal(),
                torch.zeros_like(offdiag_aggregation.diagonal()),
            ):
                raise ValueError(
                    "physical_offdiag_aggregation must have an exact zero diagonal"
                )
            if not torch.allclose(
                offdiag_aggregation.sum(dim=1),
                torch.ones(node_count, dtype=offdiag_aggregation.dtype),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "physical_offdiag_aggregation rows must sum to one"
                )
        else:
            offdiag_aggregation = torch.empty((0, 0), dtype=torch.float32)
        self.register_buffer(
            "physical_offdiag_aggregation",
            offdiag_aggregation.detach().clone(),
        )
        self.history_encoder = SharedHistoryNodeEncoder(
            self.d_model, num_temporal_patches=num_temporal_patches
        )
        # Construct the shared operator before the mode-specific hypergraph
        # scorer.  With the same random seed, fixed-H and learned-H arms then
        # receive identical history/operator initialization; only the learned-H
        # arm owns the additional membership scorer parameters.
        operator_head_type = (
            MultiSourceLowRankOperatorHead
            if operator_variant == "directed"
            else NodeWiseTemporalControlHead
        )
        self.operator_head = operator_head_type(
            d_model=self.d_model,
            forecast_horizon=self.forecast_horizon,
            num_operators=num_operators,
            num_source_heads=num_source_heads,
            eps=eps,
        )
        self.secondary_operator_head = None
        if operator_variant in HOLT_DUAL_VARIANTS:
            self.secondary_operator_head = NodeWiseTemporalControlHead(
                d_model=self.d_model,
                forecast_horizon=self.forecast_horizon,
                num_operators=num_operators,
                num_source_heads=num_source_heads,
                eps=eps,
            )
        self.soft_hypergraph = WindowConditionedSoftHypergraph(
            d_model=self.d_model,
            num_physical_hyperedges=num_physical_hyperedges,
            num_escape_slots=num_escape_slots,
            temperature=temperature,
            eps=eps,
            physical_candidate_mask=physical_candidate_mask,
            hypergraph_mode=hypergraph_mode,
            initial_blend_rho=initial_blend_rho,
        )

    @property
    def uses_absolute_node_ids(self) -> bool:
        return False

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "input": "history[B,T,N] only",
            "forecast_target_access": False,
            "uses_absolute_node_ids": False,
            "node_permutation_equivariant": True,
            "num_physical_hyperedges": self.soft_hypergraph.num_physical_hyperedges,
            "num_escape_slots": self.soft_hypergraph.num_escape_slots,
            "num_operators": self.operator_head.num_operators,
            "num_source_heads": self.operator_head.num_source_heads,
            "num_temporal_patches": self.history_encoder.num_temporal_patches,
            "hypergraph_mode": self.hypergraph_mode,
            "operator_variant": self.operator_variant,
            "cross_node_value_mixing": self.operator_variant in {
                "directed", "nodewise_dual_offdiag"
            },
            "primary_path": (
                "directed_source_response"
                if self.operator_variant == "directed"
                else "nodewise_long_history"
            ),
            "secondary_path": {
                "nodewise_dual_local": "parameter_matched_nodewise_capacity",
                "nodewise_dual_offdiag": (
                    "binary_undirected_physical_1hop_other_nodes_only"
                ),
            }.get(self.operator_variant),
            "secondary_target_self_history_access": (
                False
                if self.operator_variant == "nodewise_dual_offdiag"
                else None
            ),
            "secondary_operator_parameter_count": (
                sum(
                    parameter.numel()
                    for parameter in self.secondary_operator_head.parameters()
                )
                if self.secondary_operator_head is not None
                else 0
            ),
            "parameter_matching_reference": (
                {
                    "nodewise_temporal": "directed",
                    "nodewise_dual_local": "nodewise_dual_offdiag",
                    "nodewise_dual_offdiag": "nodewise_dual_local",
                }.get(self.operator_variant)
            ),
            "blend_rho": float(
                self.soft_hypergraph.effective_blend_rho.detach().cpu().item()
            ),
            "blend_rho_trainable": self.hypergraph_mode == "learned",
            "blend_rho_parameterization": (
                "sigmoid(logit)" if self.hypergraph_mode == "learned" else "fixed_zero"
            ),
            "blend_rho_override_active": (
                self.soft_hypergraph.blend_rho_override_active
            ),
            "node_mass_reference": "binary physical incidence degree",
            "node_mass_error_diagnostic": True,
            "edge_mass_drift_diagnostic": True,
            "membership_l1_drift_diagnostic": True,
            "hypergraph_diagnostic_names": (
                "blend_rho",
                "node_mass_error",
                "edge_mass_drift",
                "membership_l1_drift",
            ),
            "forecast_horizon": self.forecast_horizon,
            "zero_initialized_correction": True,
            "strict_baseline_requires_holt_off": True,
        }

    def encode_and_build_state(
        self,
        history: torch.Tensor,
        physical_candidate_mask: torch.Tensor | None = None,
        hypergraph_history: torch.Tensor | None = None,
    ) -> HOLTEncodedState:
        """Encode history and build the one shared HOLT hypergraph state.

        ``hypergraph_history`` is a diagnostic-only intervention: operator
        features still come from ``history``, while H is constructed from a
        different observed history.  Normal forecasting leaves it ``None``.
        This makes the window-shuffled-H counterfactual explicit without ever
        passing a target or future value into the model.
        """
        patch_node_features, pooled_node_features = self.history_encoder(history)
        if hypergraph_history is None:
            hypergraph_node_features = pooled_node_features
        else:
            if hypergraph_history.shape != history.shape:
                raise ValueError("hypergraph_history must have the same shape as history")
            _, hypergraph_node_features = self.history_encoder(hypergraph_history)
        hypergraph = self.soft_hypergraph(
            hypergraph_node_features, physical_candidate_mask
        )
        return HOLTEncodedState(
            patch_node_features=patch_node_features,
            pooled_node_features=pooled_node_features,
            hypergraph=hypergraph,
        )

    def predict_from_state(
        self,
        encoded_state: HOLTEncodedState,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict once from an already constructed state, without rebuilding H."""
        if not isinstance(encoded_state, HOLTEncodedState):
            raise TypeError("encoded_state must be HOLTEncodedState")
        state = encoded_state.hypergraph
        primary_correction, operator_diagnostics = self.operator_head(
            encoded_state.pooled_node_features,
            state,
            encoded_state.patch_node_features,
        )
        correction = primary_correction
        if self.secondary_operator_head is not None:
            secondary_pooled = encoded_state.pooled_node_features
            secondary_patches = encoded_state.patch_node_features
            if self.operator_variant == "nodewise_dual_offdiag":
                aggregation = self.physical_offdiag_aggregation.to(
                    device=secondary_pooled.device,
                    dtype=secondary_pooled.dtype,
                )
                secondary_pooled = torch.einsum(
                    "nm,bmd->bnd", aggregation, secondary_pooled
                )
                secondary_patches = torch.einsum(
                    "nm,bpmd->bpnd", aggregation, secondary_patches
                )
            secondary_correction, secondary_diagnostics = (
                self.secondary_operator_head(
                    secondary_pooled,
                    state,
                    secondary_patches,
                )
            )
            correction = primary_correction + secondary_correction
            operator_diagnostics = {
                **operator_diagnostics,
                "primary_correction": primary_correction,
                "secondary_correction": secondary_correction,
                "secondary_offdiag_active": correction.new_tensor(
                    float(self.operator_variant == "nodewise_dual_offdiag")
                ),
                **{
                    f"secondary_{name}": value
                    for name, value in secondary_diagnostics.items()
                },
            }
        if not return_diagnostics:
            return correction
        diagnostics = {
            "membership": state.membership,
            "P": state.P,
            "Q": state.Q,
            "symmetric_incidence": state.symmetric_incidence,
            "membership_entropy": state.membership_entropy,
            "normalized_entropy": state.normalized_entropy,
            "edge_diversity": state.edge_diversity,
            "edge_mass": state.edge_mass,
            "physical_mass": state.physical_mass,
            "escape_mass": state.escape_mass,
            "blend_rho": state.blend_rho,
            "node_mass_error": state.node_mass_error,
            "edge_mass_drift": state.edge_mass_drift,
            "membership_l1_drift": state.membership_l1_drift,
            **operator_diagnostics,
        }
        return correction, diagnostics

    def forward(
        self,
        history: torch.Tensor,
        physical_candidate_mask: torch.Tensor | None = None,
        hypergraph_history: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict a residual correction using the observed history only."""
        encoded_state = self.encode_and_build_state(
            history,
            physical_candidate_mask=physical_candidate_mask,
            hypergraph_history=hypergraph_history,
        )
        return self.predict_from_state(
            encoded_state, return_diagnostics=return_diagnostics
        )


__all__ = [
    "HOLT_DUAL_VARIANTS",
    "HOLT_OPERATOR_VARIANTS",
    "HOLTCorrectionHead",
    "HOLTEncodedState",
    "MultiSourceLowRankOperatorHead",
    "NodeWiseTemporalControlHead",
    "SharedHistoryNodeEncoder",
    "SoftHypergraphState",
    "WindowConditionedSoftHypergraph",
    "build_binary_undirected_offdiag_aggregation",
]
