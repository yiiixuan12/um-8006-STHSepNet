"""History-only regime definitions used by the V4 exploratory pilot."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

REGIME_NAMES = ("stationary_homogeneous", "stationary_heterogeneous", "nonstationary_homogeneous", "nonstationary_heterogeneous")


def _as_float_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _history_array(history: np.ndarray, *, name: str = "history") -> np.ndarray:
    history = _as_float_array(history, name=name)
    if history.ndim == 4 and history.shape[-1] == 1:
        history = history[..., 0]
    if history.ndim != 3:
        raise ValueError(f"{name} must have shape [windows,time,nodes]")
    return history


def fit_history_node_scales(
    train_history: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Fit per-node RMS scales using training histories only.

    RMS is deliberately simple and positively homogeneous: multiplying one
    sensor by a positive constant multiplies only that sensor's fitted scale
    by the same constant.  This removes fixed road-capacity magnitude from the
    dynamic regime score.  Constant-zero sensors receive a unit fallback.
    """

    train_history = _history_array(train_history, name="train_history")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    node_scales = np.sqrt(np.mean(np.square(train_history), axis=(0, 1)))
    return np.where(node_scales > eps, node_scales, 1.0)


def history_regime_scores(
    history: np.ndarray,
    *,
    node_scales: np.ndarray,
    recent_steps: int = 48,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute history-only dynamic asynchrony and temporal-change scores.

    ``history`` is ``[windows,time,nodes]`` and ``node_scales`` must have been
    fitted on training histories.  The spatial score first differences every
    node after its own scale normalization, removes the network-wide change at
    each timestamp, and reports the residual share of change energy.  Thus a
    synchronous proportional rise across roads scores near zero even when road
    capacities differ, while diverging node changes score high.  Means rather
    than sums make the score usable across different node counts.

    The nonstationarity score is the RMS early-to-late shift after the same
    train-fitted node normalization.  No forecast value is consumed.
    """

    history = _history_array(history)
    if recent_steps < 4 or recent_steps > history.shape[1]:
        raise ValueError("recent_steps must be in [4, history length]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    node_scales = _as_float_array(node_scales, name="node_scales").reshape(-1)
    if node_scales.shape[0] != history.shape[2]:
        raise ValueError("node_scales length must match history node count")
    if np.any(node_scales <= 0.0):
        raise ValueError("node_scales must be strictly positive")

    recent = history[:, -recent_steps:, :] / node_scales[None, None, :]
    change = np.diff(recent, axis=1)
    global_change = np.mean(change, axis=2, keepdims=True)
    local_innovation = change - global_change
    total_change_energy = np.mean(np.square(change), axis=(1, 2))
    local_change_energy = np.mean(np.square(local_innovation), axis=(1, 2))
    heterogeneity = np.sqrt(
        local_change_energy / np.maximum(total_change_energy, eps**2)
    )

    midpoint = recent_steps // 2
    earlier = np.mean(recent[:, :midpoint, :], axis=1)
    later = np.mean(recent[:, midpoint:, :], axis=1)
    nonstationarity = np.sqrt(np.mean(np.square(later - earlier), axis=1))
    return heterogeneity, nonstationarity


@dataclass(frozen=True)
class RegimeThresholds:
    heterogeneity: float
    nonstationarity: float
    node_scales: tuple[float, ...]
    quantile: float = 0.5
    scale_estimator: str = "train_history_node_rms"


def fit_regime_thresholds(
    train_history: np.ndarray,
    *,
    recent_steps: int = 48,
    quantile: float = 0.5,
) -> RegimeThresholds:
    """Fit both regime thresholds on training histories only."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    node_scales = fit_history_node_scales(train_history)
    heterogeneity, nonstationarity = history_regime_scores(
        train_history,
        node_scales=node_scales,
        recent_steps=recent_steps,
    )
    return RegimeThresholds(
        heterogeneity=float(np.quantile(heterogeneity, quantile)),
        nonstationarity=float(np.quantile(nonstationarity, quantile)),
        node_scales=tuple(float(value) for value in node_scales),
        quantile=float(quantile),
    )


def classify_history_regimes(
    history: np.ndarray,
    thresholds: RegimeThresholds,
    *,
    recent_steps: int = 48,
) -> np.ndarray:
    """Return integer labels 0..3 using fixed train-derived thresholds."""

    heterogeneity, nonstationarity = history_regime_scores(
        history,
        node_scales=np.asarray(thresholds.node_scales, dtype=np.float64),
        recent_steps=recent_steps,
    )
    spatial = heterogeneity > thresholds.heterogeneity
    temporal = nonstationarity > thresholds.nonstationarity
    return spatial.astype(np.int8) + 2 * temporal.astype(np.int8)

