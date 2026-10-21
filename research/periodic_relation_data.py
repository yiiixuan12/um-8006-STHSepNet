"""Explicit, target-excluding sources for V4; no data/calendar mutation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from models.periodic_memory_policy import validate_policy, history_from_bundle


@dataclass
class NeighborTable:
    indices: np.ndarray
    valid: np.ndarray
    address: np.ndarray
    metadata: dict


def physical_neighbors(path: str | Path, node_count: int, max_neighbors=4, *, order="legacy"):
    if order not in ("legacy","nearest"): raise ValueError("unknown neighbor ordering")
    graph = pd.read_csv(path).to_numpy(dtype=np.float32)
    if graph.shape != (node_count, node_count) or not np.isfinite(graph).all():
        raise ValueError("adjacency must be a finite N x N matrix in data-column order")
    support = graph > 0
    np.fill_diagonal(support, False)
    degree = support.sum(1)
    indices = np.full((node_count, max_neighbors), -1, dtype=np.int64)
    valid = np.zeros_like(indices, dtype=bool)
    address = np.zeros((node_count, max_neighbors, 4), dtype=np.float32)
    for target in range(node_count):
        candidates = np.flatnonzero(support[target])
        # PEMS03 weights were verified against its original distance edge CSV.
        # Legacy order remains explicit for historical checkpoint reproduction.
        score=graph[target,candidates] if order=="nearest" else -graph[target,candidates]
        chosen = candidates[np.argsort(score, kind="stable")[:max_neighbors]]
        indices[target, :len(chosen)] = chosen; valid[target, :len(chosen)] = True
        for slot, source in enumerate(chosen):
            weight = graph[target, source]
            address[target, slot] = (weight/(1+abs(weight)), degree[target]/(1+degree[target]),
                                      degree[source]/(1+degree[source]), 1.)
    return NeighborTable(indices, valid, address, {
        "path": str(path), "source_excludes_target": True, "max_neighbors": max_neighbors,
        "selection": "ascending verified distance, stable ties" if order=="nearest" else "descending provided positive weight, stable ties",
        "order":order,
        "semantics": "one-hop support, bounded provided weight and degrees; no flow direction",
        "symmetric_support": bool(np.array_equal(support, support.T)),
        "mapping_boundary": "N x N matrix assumed to use the original flow column order; no sensor-ID map is present in this file",
        "isolated_nodes": int((degree == 0).sum()),
    })


def randomized_neighbor_control(table,graph,seed=2021):
    """Source reassignment outside FULL physical support; match degrees if possible.

    Retain edge-distance slots as a weighted-graph null, update source degree.
    These reassigned edges are not asserted to have physical distances.
    """
    support=np.asarray(graph)>0; np.fill_diagonal(support,False)
    if support.shape!=(len(table.indices),)*2: raise ValueError("random control graph mismatch")
    degree=support.sum(1); rng=np.random.default_rng(seed)
    indices=table.indices.copy(); addresses=table.address.copy()
    for target in range(len(indices)):
        banned={target,*np.flatnonzero(support[target]).tolist()}
        for slot in np.flatnonzero(table.valid[target]):
            choices=np.array([node for node in range(len(indices)) if node not in banned],dtype=int)
            if not len(choices): raise ValueError("no nonphysical source available for control")
            difference=np.abs(degree[choices]-degree[table.indices[target,slot]])
            chosen=int(rng.choice(choices[difference==difference.min()]))
            indices[target,slot]=chosen; banned.add(chosen)
            addresses[target,slot,2]=degree[chosen]/(1+degree[chosen])
    return NeighborTable(indices,table.valid.copy(),addresses,{**table.metadata,
        "control":"degree-matched outside-support source reassignment; retained distance slots are null attributes",
        "seed":seed})


class RelationWindows(Dataset):
    def __init__(self, bundle, origins, cache, neighbors: NeighborTable, *, train_nodes=0, seed=2021,
                 memory_policy="full", include_neighbor_long_history=False):
        validate_policy(memory_policy)
        self.memory_policy = memory_policy
        self.include_neighbor_long_history=include_neighbor_long_history
        self.global_series = bundle.raw.mean(1, dtype=np.float64) if memory_policy == "global_long" else None
        self.bundle, self.origins, self.cache, self.neighbors = bundle, np.asarray(origins), cache, neighbors
        count = bundle.raw.shape[1]
        if not len(self.origins) or self.origins.min() < 2064:
            raise ValueError("seven-day causal history required")
        if not np.array_equal(cache["origin"], self.origins): raise ValueError("cache origin mismatch")
        if neighbors.indices.shape[0] != count: raise ValueError("graph node count mismatch")
        self.node_indices = np.stack([
            np.sort(np.random.default_rng(seed+int(origin)).choice(count, train_nodes, replace=False))
            if train_nodes else np.arange(count) for origin in self.origins
        ])

    def __len__(self): return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index]); nodes = self.node_indices[index]
        table = self.neighbors
        source_indices = table.indices[nodes]
        valid = table.valid[nodes]
        # Gather source history from the full original road set, not the subset
        # of target roads chosen for loss. Thus sampling cannot erase neighbors.
        sources = self.bundle.normalized[origin-48:origin][:, source_indices.clip(0)].transpose(1, 2, 0).copy()
        sources[~valid] = 0
        result = {
            "history": torch.from_numpy(self.bundle.normalized[origin-2064:origin][:, nodes].copy()
                if self.memory_policy == "full" else history_from_bundle(self.bundle,origin,nodes,
                    self.memory_policy,global_series=self.global_series)),
            "target": torch.from_numpy(self.bundle.normalized[origin:origin+48][:, nodes].copy()),
            "base": torch.from_numpy(self.cache["base"][index][:, nodes].astype(np.float32)),
            "short_state": torch.from_numpy(self.cache["short_state"][index][nodes].astype(np.float32)),
            "node_std": torch.from_numpy(self.bundle.std[nodes]),
            "neighbor_history": torch.from_numpy(sources),
            "neighbor_valid": torch.from_numpy(valid.copy()),
            "relation_address": torch.from_numpy(table.address[nodes].copy()),
            "origin_step": torch.tensor(origin, dtype=torch.long),
        }
        if self.include_neighbor_long_history:
            long=self.bundle.normalized[origin-2064:origin][:,source_indices.clip(0)].transpose(1,2,0).copy()
            long[~valid]=0
            result["neighbor_long_history"]=torch.from_numpy(long)
        return result
