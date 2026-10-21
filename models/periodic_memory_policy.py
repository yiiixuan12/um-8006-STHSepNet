"""Retrained information ablations, shared by neural and numerical routes.

The frozen baseline/short graph state and source roads are held fixed. These
policies remove information from the ADDED periodic interface, not the baseline.
No policy manufactures historical observations by tiling the recent window.
"""
from __future__ import annotations

import numpy as np
import torch


MEMORY_POLICIES = ("full", "global_long", "recent_only", "day_only", "no_repeat", "no_deviation")


def validate_policy(policy):
    if policy not in MEMORY_POLICIES:
        raise ValueError("unknown periodic memory policy: "+str(policy))


def transform_history(history, policy, *, global_history=None):
    """[B,2064,N]; all policies preserve each target's recent48 observations.

    For global_long the caller must supply the ALL-road raw average converted
    using each target's training-only mean/std; averaging a sampled mini-batch
    of normalized target roads is not a valid substitute.
    """
    validate_policy(policy)
    if history.ndim != 3 or history.shape[1] != 2064:
        raise ValueError("memory policies require [B,2064,N]")
    if policy not in ("recent_only", "day_only", "global_long"):
        return history
    changed = history.clone()
    if policy == "global_long":
        if global_history is None or global_history.shape != history.shape:
            raise ValueError("global_long requires an explicit full-network history")
        changed[:, :-48] = global_history[:, :-48]
    else:
        retained = 48 if policy == "recent_only" else 288
        changed[:, :-retained] = 0
    return changed


def candidate_memory_enabled(policy):
    validate_policy(policy)
    return policy not in ("recent_only", "day_only")


def memory_token_mask(policy, *, device=None):
    validate_policy(policy)
    valid = torch.ones(88, dtype=torch.bool, device=device)
    if policy == "recent_only": valid[16:] = False
    if policy == "day_only": valid[24:] = False
    if policy == "no_repeat": valid[80:84] = False
    if policy == "no_deviation": valid[84:88] = False
    return valid


def mask_numeric_features(features, policy):
    """Same252 feature budget; unavailable features are zeroed before ridge fit.

    short128/recent48/day8/base4/anchors56/repeat4/deviation4. Baseline predictions
    and short state are common controls, not credited as new long-memory gains.
    """
    validate_policy(policy)
    if features.shape[-1] != 252:
        raise ValueError("expected252 numerical features")
    if policy in ("full", "global_long"):
        return features
    changed = features.clone() if isinstance(features, torch.Tensor) else features.copy()
    if policy == "recent_only":
        changed[..., 176:184] = 0
        changed[..., 188:] = 0
    if policy == "day_only": changed[..., 188:] = 0
    if policy == "no_repeat": changed[..., 244:248] = 0
    if policy == "no_deviation": changed[..., 248:252] = 0
    return changed


def history_from_bundle(bundle, origin, nodes, policy, *, global_series=None):
    """Numpy data helper: history slice ends at origin, never at target end."""
    history = torch.from_numpy(bundle.normalized[origin-2064:origin][:, nodes].copy())[None]
    global_history = None
    if policy == "global_long":
        series = bundle.raw.mean(1, dtype=np.float64) if global_series is None else global_series
        average = series[origin-2064:origin, None]
        global_history = torch.from_numpy(((average-bundle.mean[nodes])/bundle.std[nodes]).astype(np.float32))[None]
    return transform_history(history, policy, global_history=global_history)[0].numpy()


def metadata(policy):
    validate_policy(policy)
    return {
        "policy": policy,
        "scope": "added periodic interface, including refitted numeric route and history retrieval; frozen baseline/short state/sources unchanged",
        "recent_target_history_preserved": True,
        "same_memory_token_slots": 88,
        "active_memory_tokens": int(memory_token_mask(policy).sum()),
        "historical_candidates_enabled": candidate_memory_enabled(policy),
        "numeric_route": "fit from training rows under this policy; alpha chosen on Val-A only",
        "global_long_definition": "all-road raw mean, normalized using target training statistics; own recent48 retained",
        "component_controls": "no_repeat/no_deviation remove explicit derived tokens/features, not all information from which they could be reconstructed"
    }
