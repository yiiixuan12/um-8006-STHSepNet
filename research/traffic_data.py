"""Shared chronological, train-normalized traffic data and cache alignment."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataBundle:
    raw: np.ndarray
    normalized: np.ndarray
    marks: np.ndarray
    timestamps_ns: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_min: np.ndarray
    train_max: np.ndarray
    split_bounds: dict[str, tuple[int, int]]
    node_names: tuple[str, ...]


def load_pems03(
    csv_path: Path,
    train_ratio: float,
    val_ratio: float,
) -> DataBundle:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected date plus sensor columns in {csv_path}")
    timestamps = pd.to_datetime(frame.iloc[:, 0], errors="raise")
    raw = frame.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(raw).all():
        locations = np.argwhere(~np.isfinite(raw))
        raise ValueError(f"PEMS03 contains non-finite values, first at {locations[0].tolist()}")

    total = raw.shape[0]
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if not 0 < train_end < val_end < total:
        raise ValueError("train/validation ratios leave an empty split")

    training = raw[:train_end]
    mean = training.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = training.std(axis=0, dtype=np.float64).astype(np.float32)
    constant_nodes = std < 1e-6
    std[constant_nodes] = 1.0
    normalized = ((raw - mean) / std).astype(np.float32, copy=False)
    train_norm = normalized[:train_end]

    marks = np.stack(
        [
            timestamps.dt.month.to_numpy(),
            timestamps.dt.day.to_numpy(),
            timestamps.dt.weekday.to_numpy(),
            timestamps.dt.hour.to_numpy(),
            timestamps.dt.minute.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)
    return DataBundle(
        raw=raw,
        normalized=normalized,
        marks=marks,
        timestamps_ns=timestamps.astype("int64").to_numpy(),
        mean=mean,
        std=std,
        train_min=train_norm.min(axis=0),
        train_max=train_norm.max(axis=0),
        split_bounds={
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, total),
        },
        node_names=tuple(str(name) for name in frame.columns[1:]),
    )


def limit_origins(origins: np.ndarray, maximum: int | None) -> np.ndarray:
    """Choose an ordered, deterministic, full-span screening subset."""

    origins = np.asarray(origins, dtype=np.int64).reshape(-1)
    if maximum is None or maximum >= origins.size:
        return origins.copy()
    if maximum < 1:
        raise ValueError("maximum must be positive when provided")
    indices = np.linspace(0, origins.size - 1, num=maximum, dtype=np.int64)
    return origins[np.unique(indices)]


def select_cache(cache, origins):
    """Never assume a cache row matches a window just because lengths agree."""
    stored = np.asarray(cache["origin"])
    if len(stored) < 1 or np.any(np.diff(stored) <= 0):
        raise ValueError("cache origins must be unique and increasing")
    indices = np.searchsorted(stored, origins)
    if np.any(indices >= len(stored)) or not np.array_equal(stored[indices], origins):
        raise ValueError("requested origins do not match cached inputs")
    return {key: np.asarray(value)[indices] for key, value in cache.items()}


def validate_origin_roles(bundle, origins, *, history=2064, horizon=48, target_gap=48):
    """One authoritative target-boundary convention for all cache consumers."""
    if min(history, horizon) < 1 or target_gap < 0:
        raise ValueError("invalid history/horizon/target gap")
    for role in ("train", "val_a", "val_b"):
        values = np.asarray(origins[role], dtype=np.int64)
        if values.ndim != 1 or not len(values) or np.any(np.diff(values) <= 0):
            raise ValueError(f"{role} origins must be nonempty, unique and increasing")
        lower, upper = bundle.split_bounds["train" if role == "train" else "val"]
        if values.min() < max(history, lower) or values.max()+horizon > upper:
            raise ValueError(f"{role} forecast target crosses chronological split/history boundary")
    gap = int(origins["val_b"][0]-origins["val_a"][-1]-horizon)
    if gap < target_gap:
        raise ValueError(f"validation TARGET gap {gap} is smaller than {target_gap}")
    return gap


def valid_pilot_origins(bundle, candidates, *, train_windows, val_windows, target_gap=48):
    """Explicit legacy-cache migration: filter, record, then validate selections.

    Existing cache rows/predictions are never recomputed or reordered in place.
    The gap counts unused target timestamps, NOT removed forecast origins.
    """
    selected, removed = {}, {}
    for role in ("train", "val_a", "val_b"):
        values = np.asarray(candidates[role], dtype=np.int64)
        if values.ndim != 1 or not len(values) or np.any(np.diff(values) <= 0):
            raise ValueError(f"invalid cached {role} origins")
        lower, upper = bundle.split_bounds["train" if role == "train" else "val"]
        valid = (values >= max(2064, lower)) & (values+48 <= upper)
        if role == "val_b":
            valid &= values >= selected["val_a"][-1]+48+target_gap
        removed[role] = values[~valid].tolist()
        if not valid.any():
            raise ValueError(f"no valid {role} origins after explicit boundary migration")
        selected[role] = limit_origins(values[valid], train_windows if role == "train" else val_windows)
    gap = validate_origin_roles(bundle, selected, target_gap=target_gap)
    return selected, {"split_bounds": bundle.split_bounds,
                      "removed_cached_origins": removed,
                      "target_gap_steps": gap, "required_target_gap_steps": target_gap,
                      "selection": "filter cached origins before deterministic full-span subsampling",
                      "cache_predictions_recomputed": False}

