"""Small, testable information controls and predeclared BIKE case selection.

No changes to the frozen scientific model. The global-long control replaces
ONLY observations older than the target's recent48 hours, in every added route.
The target's recent48h (including the contained daily24h) stays unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from research.coursework_data import CourseworkWindows


class BikeMemoryWindows(CourseworkWindows):
    def __init__(self, bundle, role, arm="full"):
        if arm not in ("full", "global_long", "no_utility"):
            raise ValueError("unknown BIKE ablation arm")
        super().__init__(bundle, role, 240)
        self.arm = arm
        # Spatial mean BEFORE node normalization, over ALL stations, never a
        # target chunk. Only the slice ending strictly before origin is read.
        self.global_series = (bundle.raw.mean(1, dtype=np.float64)
                              if arm == "global_long" else None)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        if self.arm == "global_long":
            origin = int(self.origins[index])
            shared = self.global_series[origin-240:origin-48, None]
            changed = ((shared-self.bundle.mean)/self.bundle.std).astype(np.float32)
            batch["history"][:-48] = torch.from_numpy(changed)
        return batch


def zero_utility_loss(model, batch, logits, *, loss_scale):
    """Remove ONLY auxiliary CE. Retrieval still learns from forecast loss.

    Keep the auxiliary output interface and its zero-valued graph connection.
    The last logit is the always-valid abstention option. No labels are built.
    """
    return logits[..., -1].float().sum()*0., float(logits.new_zeros(()))


def restore_raw(prediction, mean, std):
    prediction = np.asarray(prediction)
    if prediction.shape[-1] != len(mean) or len(mean) != len(std):
        raise ValueError("node normalization dimensions do not match")
    return prediction.astype(np.float64)*np.asarray(std)+np.asarray(mean)


def choose_cases(bundle, test_origins):
    """Three fixed timestamp quantiles and train-only flow-rank stations.

    Selection never looks at predictions or test flow magnitudes. Require a
    real contiguous48h horizon: row offsets must equal actual hours here.
    """
    hour = np.int64(3_600_000_000_000)
    times = bundle.timestamps_ns
    eligible = []
    for origin in test_origins:
        origin = int(origin)
        if times[origin] % (24*hour) != 0:
            continue
        if np.all(np.diff(times[origin:origin+48]) == hour):
            eligible.append(origin)
    if len(eligible) < 3:
        raise ValueError("need three test midnight origins with contiguous48h")
    indices = [0, (len(eligible)-1)//2, len(eligible)-1]
    origins = [eligible[i] for i in indices]
    train_start, train_end = bundle.split_bounds["train"]
    means = bundle.raw[train_start:train_end].mean(0, dtype=np.float64)
    order = np.lexsort((np.arange(len(means)), means))
    nodes = [int(order[int(q*(len(order)-1))]) for q in (.25, .5, .75)]
    return {"origins": origins, "nodes": nodes,
            "station_names": [bundle.node_names[i] for i in nodes],
            "station_train_means": [float(means[i]) for i in nodes],
            "selection": "first/middle/last eligible test midnight; train-only station mean quartiles25/50/75",
            "no_prediction_or_test_flow_used_for_selection": True,
            "forecast_hours_exported": 24, "model_output_hours": 48,
            "forecast_type": "single-origin direct48h; report first24h, no rolling observations",
            "interval_minutes": 60, "unit": "original BIKE-Inflow hourly station inflow units",
            "timestamps": [str(np.datetime64(int(times[i]), 'ns')) for i in origins]}
