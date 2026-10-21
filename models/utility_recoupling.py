"""Utility-calibrated hyperedge recoupling building blocks.

All routing features and token selection in this module use history only. Future
targets are used solely by ``utility_targets`` while producing training labels.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_INTERACTION_RANKS = (0, 2, 4, 8, 16, 32)
SPATIAL_LLM_MODES = (
    "separate_identity",
    "joint_identity",
    "joint_structural",
)
STRUCTURAL_DESCRIPTOR_VERSION = "line_graph_lappe_stats_v1"


def build_k_hop_topology_incidence(
    adjacency: torch.Tensor, hops: int = 1
) -> torch.Tensor:
    """Build one radius-``hops`` closed-neighbourhood edge per graph node.

    The physical graph is treated as an undirected support graph.  Column
    ``j`` contains every node whose shortest-path distance from anchor node
    ``j`` is at most ``hops``.  This keeps the existing ``[node, hyperedge]``
    convention while allowing a wider, still topology-defined token
    dictionary for Pilot-2 fallback experiments.
    """
    adjacency = torch.as_tensor(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got {tuple(adjacency.shape)}")
    if isinstance(hops, bool) or not isinstance(hops, int) or hops < 1:
        raise ValueError("hops must be a positive integer")

    support = (adjacency != 0) | (adjacency.T != 0)
    node_count = support.shape[0]
    identity = torch.eye(node_count, dtype=torch.bool, device=support.device)
    reach = identity.clone()
    frontier = identity.to(torch.float32)
    support_float = support.to(torch.float32)
    for _ in range(hops):
        frontier = ((frontier @ support_float) > 0).to(torch.float32)
        reach |= frontier.bool()
    return reach.to(torch.float32)


def build_topology_incidence(adjacency: torch.Tensor) -> torch.Tensor:
    """Build the existing one-hop closed-neighbourhood token incidence."""
    return build_k_hop_topology_incidence(adjacency, hops=1)


def build_randomized_incidence(
    incidence: torch.Tensor, seed: int = 2021
) -> torch.Tensor:
    """Node-label-permuted null preserving edge sizes and edge overlaps.

    This is not a per-node-degree-matched null: permuting rows deliberately
    breaks the association between physical road identity and membership.
    When edge adjacency also uses a node graph, call
    :func:`build_geometry_matched_random_null` instead so the graph is
    permuted by the same labels.
    """
    incidence_tensor = torch.as_tensor(incidence, dtype=torch.float32)
    permutation = _seeded_node_permutation(
        incidence_tensor.shape[0], seed, incidence_tensor.device
    )
    return incidence_tensor[permutation]


def _seeded_node_permutation(
    node_count: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(node_count, generator=generator).to(device=device)


def build_geometry_matched_random_null(
    incidence: torch.Tensor,
    node_adjacency: torch.Tensor,
    seed: int = 2021,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly permute token membership and its tokenizer-only graph support.

    For a seeded node-label permutation ``P``, this returns
    ``H' = P H`` and ``A' = P A P^T``.  Consequently both ``H'^T H'`` and
    ``H'^T A' H'`` exactly match the original dictionary.  The traffic data
    are *not* permuted, so the association between a token and physical road
    identity is broken while token geometry and sparse-recoupler compute stay
    matched.  The forecasting graph branch must continue to use the real
    adjacency rather than the returned tokenizer-only support.
    """
    incidence_tensor = torch.as_tensor(incidence, dtype=torch.float32)
    if incidence_tensor.ndim != 2:
        raise ValueError("incidence must have shape [nodes, hyperedges]")
    adjacency_tensor = torch.as_tensor(node_adjacency, dtype=torch.float32).to(
        device=incidence_tensor.device
    )
    node_count = incidence_tensor.shape[0]
    if adjacency_tensor.shape != (node_count, node_count):
        raise ValueError("node_adjacency shape must match the incidence node axis")
    permutation = _seeded_node_permutation(node_count, seed, incidence_tensor.device)
    randomized_incidence = incidence_tensor[permutation]
    randomized_adjacency = adjacency_tensor[permutation][:, permutation]
    return randomized_incidence, randomized_adjacency


def build_metis_incidence(adjacency: torch.Tensor, num_parts: int = 8) -> torch.Tensor:
    """Construct non-overlapping fixed regions with METIS graph partitioning."""
    try:
        import pymetis
    except ImportError as error:
        raise RuntimeError(
            "METIS grouping requested but pymetis is unavailable. Install pymetis "
            "or select topology/random grouping."
        ) from error
    support = ((adjacency != 0) | (adjacency.T != 0)).cpu()
    support.fill_diagonal_(False)
    adjacency_list = [
        torch.where(support[node])[0].tolist() for node in range(support.shape[0])
    ]
    _, partition = pymetis.part_graph(num_parts, adjacency=adjacency_list)
    labels = torch.tensor(partition, dtype=torch.long)
    incidence = F.one_hot(labels, num_classes=num_parts).to(torch.float32)
    if (incidence.sum(0) == 0).any():
        raise RuntimeError("METIS returned an empty region")
    return incidence


def canonicalize_eigenvector_signs(eigenvectors: torch.Tensor) -> torch.Tensor:
    """Choose a deterministic sign for each (otherwise sign-ambiguous) vector.

    The entry with the largest absolute magnitude is made non-negative.  This
    resolves sign flips, but deliberately does not claim to resolve rotations
    inside repeated-eigenvalue eigenspaces.
    """
    if eigenvectors.ndim != 2:
        raise ValueError("eigenvectors must have shape [items, dimensions]")
    if eigenvectors.shape[1] == 0:
        return eigenvectors
    pivots = eigenvectors.abs().argmax(dim=0)
    columns = torch.arange(eigenvectors.shape[1], device=eigenvectors.device)
    signs = torch.sign(eigenvectors[pivots, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return eigenvectors * signs.unsqueeze(0)


def build_hyperedge_structural_descriptors(
    incidence: torch.Tensor,
    node_adjacency: torch.Tensor,
    pe_dim: int = 8,
) -> torch.Tensor:
    """Describe each hyperedge using graph geometry rather than an ID lookup.

    The first six columns contain bounded group/degree/connectivity statistics.
    The remaining ``pe_dim`` columns are sign-canonicalized low-frequency
    eigenvectors of the binary hyperedge relation graph induced by
    ``H^T H + H^T A H``.  Computing in CPU float64 makes the descriptor stable
    across the mixed-precision forecasting path.

    This descriptor is a low-cost structural baseline, not a canonical graph
    identifier: eigenvectors in a repeated-eigenvalue subspace may rotate.
    """
    if isinstance(pe_dim, bool) or not isinstance(pe_dim, int) or pe_dim < 1:
        raise ValueError("pe_dim must be a positive integer")
    incidence_tensor = torch.as_tensor(incidence, dtype=torch.float64)
    if incidence_tensor.ndim != 2:
        raise ValueError("incidence must have shape [nodes, hyperedges]")
    nodes, edges = incidence_tensor.shape
    if nodes < 1 or edges < 1 or (incidence_tensor.sum(dim=0) == 0).any():
        raise ValueError("incidence must contain non-empty nodes and hyperedges")
    adjacency_tensor = torch.as_tensor(node_adjacency, dtype=torch.float64)
    if adjacency_tensor.shape != (nodes, nodes):
        raise ValueError("node_adjacency shape must match the incidence node axis")

    output_device = torch.as_tensor(incidence).device
    h = (incidence_tensor != 0).to(device="cpu", dtype=torch.float64)
    support = (
        (adjacency_tensor != 0) | (adjacency_tensor.T != 0)
    ).to(device="cpu", dtype=torch.float64)
    support.fill_diagonal_(0.0)

    group_size = h.sum(dim=0).clamp_min(1.0)
    node_degree = support.sum(dim=1)
    group_degree_mean = (h.T @ node_degree) / group_size
    group_degree_second = (h.T @ node_degree.square()) / group_size
    group_degree_std = (
        group_degree_second - group_degree_mean.square()
    ).clamp_min(0.0).sqrt()

    overlap = h.T @ h
    graph_links = h.T @ support @ h
    relation_support = (overlap + graph_links) > 0
    relation_support.fill_diagonal_(False)
    relation_weight = relation_support.to(torch.float64)
    relation_degree = relation_weight.sum(dim=1)

    internal_links = torch.diagonal(graph_links)
    incident_links = (h.T @ support).sum(dim=1)
    boundary_links = (incident_links - internal_links).clamp_min(0.0)
    maximum_internal = (group_size * (group_size - 1.0)).clamp_min(1.0)
    node_degree_scale = float(max(nodes - 1, 1))
    edge_degree_scale = float(max(edges - 1, 1))
    statistics = torch.stack(
        [
            group_size / float(nodes),
            group_degree_mean / node_degree_scale,
            group_degree_std / node_degree_scale,
            internal_links / maximum_internal,
            boundary_links / incident_links.clamp_min(1.0),
            relation_degree / edge_degree_scale,
        ],
        dim=1,
    )

    safe_degree = relation_degree.clamp_min(1.0)
    inv_sqrt = safe_degree.rsqrt()
    normalized_adjacency = inv_sqrt[:, None] * relation_weight * inv_sqrt[None, :]
    laplacian = torch.eye(edges, dtype=torch.float64) - normalized_adjacency
    _, eigenvectors = torch.linalg.eigh(laplacian)
    available = max(0, min(pe_dim, edges - 1))
    positional = eigenvectors[:, 1 : 1 + available]
    positional = canonicalize_eigenvector_signs(positional)
    if available < pe_dim:
        positional = F.pad(positional, (0, pe_dim - available))
    # Re-apply the sign rule after casting: tied pivot magnitudes can round to
    # equality in FP32 and otherwise make the apparent pivot change rows.
    positional = canonicalize_eigenvector_signs(positional.to(torch.float32))
    descriptors = torch.cat([statistics.to(torch.float32), positional], dim=1)
    return descriptors.to(device=output_device)


def hyperedge_structural_descriptor_metadata(pe_dim: int) -> dict[str, object]:
    """Machine-readable semantics shared by checkpoints and result artifacts."""
    return {
        "version": STRUCTURAL_DESCRIPTOR_VERSION,
        "dimensions": int(6 + pe_dim),
        "statistics": [
            "group_size_fraction",
            "mean_node_degree",
            "std_node_degree",
            "internal_link_density",
            "boundary_link_fraction",
            "hyperedge_relation_degree",
        ],
        "positional_encoding": {
            "type": "normalized_laplacian_eigenvectors_on_hyperedge_relation_graph",
            "dimensions": int(pe_dim),
            "relation": "binary support of H^T H + H^T A H",
            "sign_rule": "largest-absolute entry is non-negative",
            "limitation": (
                "sign canonicalization does not remove arbitrary rotations within "
                "repeated-eigenvalue eigenspaces"
            ),
        },
    }


def build_joint_graph_time_attention_mask(
    edge_ids: torch.Tensor,
    edge_adjacency: torch.Tensor,
    patch_count: int,
    temporal_radius: int = 1,
) -> torch.Tensor:
    """Return the BERT mask for a joint, edge-major graph-time sequence.

    A patch can attend to every time patch of the same hyperedge.  Across two
    different hyperedges, attention is allowed only when the hyperedges are
    graph-adjacent and their patch indices differ by at most
    ``temporal_radius``.  The result has shape ``[B, K*P, K*P]`` and is accepted
    by Hugging Face BERT as a per-example 3-D attention mask.
    """
    if edge_ids.ndim != 2:
        raise ValueError("edge_ids must have shape [batch, selected_edges]")
    if isinstance(patch_count, bool) or not isinstance(patch_count, int) or patch_count < 1:
        raise ValueError("patch_count must be a positive integer")
    if (
        isinstance(temporal_radius, bool)
        or not isinstance(temporal_radius, int)
        or temporal_radius < 0
    ):
        raise ValueError("temporal_radius must be a non-negative integer")
    adjacency = torch.as_tensor(edge_adjacency, dtype=torch.bool, device=edge_ids.device)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("edge_adjacency must be square")
    if edge_ids.numel() and (
        edge_ids.min().item() < 0 or edge_ids.max().item() >= adjacency.shape[0]
    ):
        raise ValueError("edge_ids contain an out-of-range hyperedge")

    batch, selected_edges = edge_ids.shape
    if selected_edges == 0:
        return torch.empty(
            (batch, 0, 0), dtype=torch.bool, device=edge_ids.device
        )
    selected_adjacency = adjacency[
        edge_ids[:, :, None], edge_ids[:, None, :]
    ]
    same_edge = torch.eye(
        selected_edges, dtype=torch.bool, device=edge_ids.device
    )[None, :, :, None, None]
    patch_index = torch.arange(patch_count, device=edge_ids.device)
    nearby_time = (
        (patch_index[:, None] - patch_index[None, :]).abs() <= temporal_radius
    )[None, None, None, :, :]
    cross_edge = selected_adjacency[:, :, :, None, None] & nearby_time
    allowed = same_edge | cross_edge
    return allowed.permute(0, 1, 3, 2, 4).reshape(
        batch, selected_edges * patch_count, selected_edges * patch_count
    )


def build_repeated_time_position_ids(
    batch_size: int,
    selected_edges: int,
    patch_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Position IDs for edge-major tokens with no edge-order positional cue."""
    if batch_size < 0 or selected_edges < 0 or patch_count < 1:
        raise ValueError("invalid batch, edge, or patch count")
    per_edge = torch.arange(patch_count, dtype=torch.long, device=device)
    return per_edge.repeat(selected_edges).unsqueeze(0).expand(batch_size, -1)


class HyperedgeStructuralEncoder(nn.Module):
    """Project deterministic descriptors into the LLM token space.

    Keeping this interface descriptor-based leaves room for future relation or
    path descriptors without coupling the forecasting model to one graph
    encoding implementation.
    """

    def __init__(self, descriptors: torch.Tensor, output_dim: int):
        super().__init__()
        descriptors = torch.as_tensor(descriptors, dtype=torch.float32)
        if descriptors.ndim != 2 or descriptors.shape[1] < 1:
            raise ValueError("descriptors must have shape [hyperedges, features]")
        self.register_buffer("descriptors", descriptors, persistent=False)
        descriptor_dim = descriptors.shape[1]
        hidden_dim = min(max(32, 2 * descriptor_dim), output_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, edge_ids: torch.Tensor) -> torch.Tensor:
        if edge_ids.ndim != 2:
            raise ValueError("edge_ids must have shape [batch, selected_edges]")
        return self.projection(self.descriptors[edge_ids])


class HyperedgeInnovationTokenizer(nn.Module):
    """Turn node histories into global and topology-local innovation series."""

    def __init__(
        self,
        incidence: torch.Tensor,
        selection_policy: str = "coherent_diverse",
        node_adjacency: torch.Tensor | None = None,
    ):
        super().__init__()
        incidence = torch.as_tensor(incidence, dtype=torch.float32)
        if incidence.ndim != 2:
            raise ValueError("incidence must have shape [nodes, hyperedges]")
        if (incidence.sum(0) == 0).any() or (incidence.sum(1) == 0).any():
            raise ValueError("every node and hyperedge must have non-zero degree")
        self.register_buffer("incidence", incidence)
        self.register_buffer("edge_degree", incidence.sum(0).clamp_min(1.0))
        edge_adjacency = (incidence.T @ incidence) > 0
        if node_adjacency is not None:
            node_support = torch.as_tensor(node_adjacency, dtype=torch.float32)
            if node_support.shape != (incidence.shape[0], incidence.shape[0]):
                raise ValueError("node_adjacency shape must match the incidence node axis")
            node_support = ((node_support != 0) | (node_support.T != 0)).float()
            edge_adjacency |= (incidence.T @ node_support @ incidence) > 0
        edge_adjacency.fill_diagonal_(True)
        self.register_buffer("edge_adjacency", edge_adjacency)
        intersection = incidence.T @ incidence
        union = self.edge_degree[:, None] + self.edge_degree[None, :] - intersection
        self.register_buffer("edge_jaccard", intersection / union.clamp_min(1.0))
        if selection_policy not in {
            "raw_energy",
            "coherent_energy",
            "coherent_diverse",
        }:
            raise ValueError(f"unknown selection_policy: {selection_policy}")
        self.selection_policy = selection_policy

    @property
    def num_edges(self) -> int:
        return self.incidence.shape[1]

    def all_innovations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global mean [B,T] and all hyperedge innovations [B,T,M]."""
        if x.ndim != 3 or x.shape[-1] != self.incidence.shape[0]:
            raise ValueError(
                f"x must be [B,T,{self.incidence.shape[0]}], got {tuple(x.shape)}"
            )
        # Selection IDs must not depend on whether inference happens under
        # BF16 autocast.  Keep this relatively small reduction explicitly FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            global_series = x_fp32.mean(dim=2)
            residual = x_fp32 - global_series.unsqueeze(-1)
            incidence = self.incidence.to(device=x.device, dtype=torch.float32)
            innovations = torch.einsum("btn,nm->btm", residual, incidence)
            innovations = innovations / self.edge_degree.to(
                device=x.device, dtype=torch.float32
            ).view(1, 1, -1)
        return global_series, innovations

    def forward(
        self, x: torch.Tensor, interaction_rank: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if interaction_rank < 0 or interaction_rank > self.num_edges:
            raise ValueError(
                f"interaction_rank must be in [0,{self.num_edges}], got {interaction_rank}"
            )
        if interaction_rank == 0:
            # The separation baseline must actually be cheap: do not construct
            # every hyperedge innovation merely to discard it.
            with torch.autocast(device_type=x.device.type, enabled=False):
                global_series = x.float().mean(dim=2)
            batch, time = global_series.shape
            ids = torch.empty((batch, 0), dtype=torch.long, device=x.device)
            local = x.new_empty((batch, 0, time))
            membership = x.new_empty((batch, 0, x.shape[-1]))
            return global_series.unsqueeze(1), local, ids, membership

        global_series, innovations = self.all_innovations(x)
        return self.select_precomputed(global_series, innovations, interaction_rank)

    def select_precomputed(
        self,
        global_series: torch.Tensor,
        innovations: torch.Tensor,
        interaction_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select local series from a history-only cache used by the router."""
        if interaction_rank < 0 or interaction_rank > self.num_edges:
            raise ValueError(
                f"interaction_rank must be in [0,{self.num_edges}], got {interaction_rank}"
            )
        batch, time, _ = innovations.shape
        if interaction_rank == 0:
            ids = torch.empty((batch, 0), dtype=torch.long, device=innovations.device)
            local = innovations.new_empty((batch, 0, time))
            membership = innovations.new_empty(
                (batch, 0, self.incidence.shape[0])
            )
            return global_series.unsqueeze(1), local, ids, membership

        # This history-only score ranks *which* hyperedges to represent.  It
        # never decides K.  Multiplying the squared regional mean by group size
        # removes the raw-mean statistic's bias toward noisy tiny groups.
        scores = innovations.square().mean(dim=1)
        if self.selection_policy != "raw_energy":
            scores = scores * self.edge_degree.to(scores).unsqueeze(0)
        if interaction_rank == self.num_edges:
            ids = torch.arange(self.num_edges, device=innovations.device).expand(batch, -1)
        elif self.selection_policy == "coherent_diverse":
            ids = self._diverse_topk(scores, interaction_rank)
        else:
            ids = scores.topk(interaction_rank, dim=1, sorted=True).indices
        local = innovations.transpose(1, 2).gather(
            1, ids.unsqueeze(-1).expand(-1, -1, time)
        )
        edge_membership = self.incidence.T.to(
            device=innovations.device, dtype=innovations.dtype
        )
        membership = edge_membership[ids]
        return global_series.unsqueeze(1), local, ids, membership

    def _diverse_topk(self, scores: torch.Tensor, count: int) -> torch.Tensor:
        """Greedy relevance-times-novelty selection for small interaction K."""
        batch, candidates = scores.shape
        selected = torch.empty((batch, count), dtype=torch.long, device=scores.device)
        unavailable = torch.zeros_like(scores, dtype=torch.bool)
        maximum_overlap = torch.zeros_like(scores)
        jaccard = self.edge_jaccard.to(scores)
        for position in range(count):
            adjusted = scores * (1.0 - maximum_overlap)
            adjusted = adjusted.masked_fill(unavailable, -torch.inf)
            chosen = adjusted.argmax(dim=1)
            selected[:, position] = chosen
            unavailable.scatter_(1, chosen[:, None], True)
            overlap = jaccard[chosen]
            maximum_overlap = torch.maximum(maximum_overlap, overlap)
        return selected


class SparseHyperedgeTemporalRecoupler(nn.Module):
    """Exchange messages only across overlapping edges and adjacent patches.

    The local operator is implemented from the selected hyperedge adjacency
    list.  It therefore scales with the number of allowed space-time links,
    rather than materialising dense attention over ``(edges * patches)^2``.
    This distinction matters for the all-hyperedge Pilot-2 control.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.global_cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_query = nn.Linear(d_model, d_model)
        self.local_key = nn.Linear(d_model, d_model)
        self.local_value = nn.Linear(d_model, d_model)
        self.local_output = nn.Linear(d_model, d_model)
        self.local_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def _sparse_local_attention(
        self,
        hidden: torch.Tensor,
        selected_edge_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-list attention for one sample, with temporal offsets -1/0/+1."""
        edges, patches, dim = hidden.shape
        query = self.local_query(hidden).view(edges, patches, self.n_heads, self.head_dim)
        key = self.local_key(hidden).view(edges, patches, self.n_heads, self.head_dim)
        value = self.local_value(hidden).view(edges, patches, self.n_heads, self.head_dim)

        # ``destination`` receives a message from ``source``.  Diagonal links
        # are forced so every output token has at least one incoming message.
        selected_edge_adjacency = selected_edge_adjacency.clone()
        selected_edge_adjacency.fill_diagonal_(True)
        destination, source = selected_edge_adjacency.nonzero(as_tuple=True)
        destination_parts = []
        score_parts = []
        value_parts = []
        scale = self.head_dim ** -0.5
        for offset in (-1, 0, 1):
            if offset < 0:
                destination_time = torch.arange(-offset, patches, device=hidden.device)
            elif offset > 0:
                destination_time = torch.arange(0, patches - offset, device=hidden.device)
            else:
                destination_time = torch.arange(patches, device=hidden.device)
            if destination_time.numel() == 0:
                continue
            source_time = destination_time + offset
            pair_count = destination.numel()
            time_count = destination_time.numel()
            dst_edge = destination[:, None].expand(pair_count, time_count).reshape(-1)
            src_edge = source[:, None].expand(pair_count, time_count).reshape(-1)
            dst_time = destination_time[None].expand(pair_count, time_count).reshape(-1)
            src_time = source_time[None].expand(pair_count, time_count).reshape(-1)
            q = query[dst_edge, dst_time]
            k = key[src_edge, src_time]
            destination_parts.append(dst_edge * patches + dst_time)
            score_parts.append((q * k).sum(dim=-1) * scale)
            value_parts.append(value[src_edge, src_time])

        destination_index = torch.cat(destination_parts, dim=0)
        scores = torch.cat(score_parts, dim=0)
        messages = torch.cat(value_parts, dim=0)
        expanded_index = destination_index[:, None].expand(-1, self.n_heads)

        # Stable segment softmax and accumulation stay in FP32 under BF16
        # autocast.  Apart from avoiding dtype-mismatch in ``index_add_``, this
        # prevents small neighbourhood weights from underflowing.
        scores_fp32 = scores.float()
        messages_fp32 = messages.float()
        maxima = scores_fp32.new_full((edges * patches, self.n_heads), -torch.inf)
        maxima.scatter_reduce_(
            0, expanded_index, scores_fp32, reduce="amax", include_self=True
        )
        weights = (scores_fp32 - maxima[destination_index]).exp()
        denominator = scores_fp32.new_zeros((edges * patches, self.n_heads))
        denominator.scatter_add_(0, expanded_index, weights)
        weighted = weights.unsqueeze(-1) * messages_fp32
        numerator = messages_fp32.new_zeros(
            (edges * patches, self.n_heads, self.head_dim)
        )
        numerator.index_add_(0, destination_index, weighted)
        attended = numerator / denominator.clamp_min(1e-9).unsqueeze(-1)
        attended = attended.to(messages.dtype).reshape(edges, patches, dim)
        return self.local_output(self.local_dropout(attended))

    def forward(
        self,
        global_hidden: torch.Tensor,
        local_hidden: torch.Tensor,
        edge_ids: torch.Tensor,
        edge_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        if local_hidden.shape[1] == 0:
            return local_hidden
        batch, edges, patches, dim = local_hidden.shape
        flat = local_hidden.reshape(batch, edges * patches, dim)
        global_context = global_hidden[:, 0]
        cross, _ = self.global_cross_attention(
            flat, global_context, global_context, need_weights=False
        )
        flat = self.norm1(flat + cross)

        outputs = []
        for sample in range(batch):
            selected = edge_adjacency[edge_ids[sample]][:, edge_ids[sample]]
            outputs.append(
                self._sparse_local_attention(
                    flat[sample].reshape(edges, patches, dim), selected
                ).reshape(edges * patches, dim)
            )
        coupled = torch.stack(outputs, dim=0)
        flat = self.norm2(flat + coupled)
        flat = self.norm3(flat + self.ffn(flat))
        return flat.reshape(batch, edges, patches, dim)


def history_complexity_features(
    x: torch.Tensor,
    tokenizer: HyperedgeInnovationTokenizer,
    precomputed: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute compact, past-only descriptors for interaction-rank routing."""
    if precomputed is None:
        global_series, innovations = tokenizer.all_innovations(x)
    else:
        global_series, innovations = precomputed
    temporal_diff = global_series.diff(dim=1)
    spatial_std = (x - global_series.unsqueeze(-1)).std(dim=2, unbiased=False)
    edge_rms = innovations.square().mean(dim=1).sqrt()
    lag = min(12, max(1, global_series.shape[1] // 4))
    left = global_series[:, :-lag]
    right = global_series[:, lag:]
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    autocorr = (left * right).mean(dim=1) / (
        left.square().mean(dim=1).sqrt()
        * right.square().mean(dim=1).sqrt()
        + 1e-6
    )
    quantiles = torch.quantile(edge_rms, torch.tensor([0.5, 0.9], device=x.device), dim=1).T
    return torch.stack(
        [
            temporal_diff.abs().mean(1),
            temporal_diff.std(1, unbiased=False),
            temporal_diff.abs().amax(1),
            spatial_std.mean(1),
            spatial_std.std(1, unbiased=False),
            spatial_std[:, -1],
            spatial_std.amax(1),
            edge_rms.mean(1),
            edge_rms.std(1, unbiased=False),
            edge_rms.amax(1),
            quantiles[:, 0],
            quantiles[:, 1],
            (global_series[:, -1] - global_series[:, 0]).abs(),
            autocorr,
        ],
        dim=1,
    )


class UtilityRouter(nn.Module):
    """Predict the useful interaction rank using history-only features."""

    def __init__(
        self,
        tokenizer: HyperedgeInnovationTokenizer,
        ranks: Sequence[int] = DEFAULT_INTERACTION_RANKS,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.register_buffer("rank_values", torch.tensor(tuple(ranks), dtype=torch.long))
        self.network = nn.Sequential(
            nn.LayerNorm(14),
            nn.Linear(14, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ranks)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(history_complexity_features(x, self.tokenizer))

    def from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def choose(self, x: torch.Tensor, confidence_threshold: float = 0.5):
        probabilities = self(x).softmax(dim=-1)
        confidence, indices = probabilities.max(dim=-1)
        ranks = self.rank_values[indices]
        ranks = torch.where(
            confidence >= confidence_threshold, ranks, torch.zeros_like(ranks)
        )
        return ranks, probabilities


def utility_targets(
    per_rank_error: torch.Tensor,
    ranks: Sequence[int] = DEFAULT_INTERACTION_RANKS,
    cost_weight: float = 0.0,
    per_rank_cost: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    """Create oracle labels from error plus a measured, normalized cost.

    ``per_rank_cost`` should normally contain validation-time latency or FLOPs
    for each rank.  The former rank/max-rank proxy is retained only as an
    explicit fallback for unit-level use where measurements are unavailable.
    """
    if per_rank_error.ndim != 2 or per_rank_error.shape[1] != len(ranks):
        raise ValueError("per_rank_error must have shape [windows, number_of_ranks]")
    if per_rank_cost is None:
        costs = torch.tensor(ranks, dtype=per_rank_error.dtype, device=per_rank_error.device)
    else:
        costs = torch.as_tensor(
            per_rank_cost, dtype=per_rank_error.dtype, device=per_rank_error.device
        )
        if costs.ndim != 1 or costs.numel() != len(ranks):
            raise ValueError("per_rank_cost must provide one value for every rank")
        if not torch.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("per_rank_cost must contain finite non-negative values")
    cost_span = costs.max() - costs.min()
    costs = (costs - costs.min()) / cost_span.clamp_min(torch.finfo(costs.dtype).eps)
    return (per_rank_error + cost_weight * costs.unsqueeze(0)).argmin(dim=1)
