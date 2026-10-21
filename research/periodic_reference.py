"""Matched-data ridge reference; regularization is selected on Val-A only."""
import numpy as np
from models.periodic_memory_policy import validate_policy, history_from_bundle, mask_numeric_features


def fit_numeric_calibration(datasets, alpha):
    """A frozen numerical route, alpha already selected using Val-A only."""
    roles = {}
    policies = {getattr(dataset,"memory_policy","full") for dataset in datasets.values()}
    if len(policies) != 1: raise ValueError("numeric fit/evaluation must share one memory policy")
    policy = policies.pop()
    for role, dataset in datasets.items():
        values = probe_arrays(dataset.bundle, dataset.cache, dataset.origins, dataset.node_indices,
                              memory_policy=policy)
        values.append(np.concatenate([dataset.bundle.normalized[o:o+48][:, dataset.node_indices[index]].T
                                      for index, o in enumerate(dataset.origins)]).astype(float))
        roles[role] = values
    train = roles["train"]
    x = np.concatenate((train[0], train[1]), 1)
    mean, scale = x.mean(0), x.std(0).clip(.01)
    matrix = np.column_stack(((x-mean)/scale, np.ones(len(x))))
    penalty = np.eye(matrix.shape[1])*alpha
    penalty[-1, -1] = 0
    coefficient = np.linalg.solve(matrix.T@matrix+penalty, matrix.T@train[2])
    state = {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
             "weight": coefficient[:-1].T.astype(np.float32), "bias": coefficient[-1].astype(np.float32)}
    result = {"alpha_selected_on_val_a": alpha, "parameters": int(coefficient.size), "metrics": {},
              "memory_policy":policy}
    for role, data in roles.items():
        inputs = np.concatenate((data[0], data[1]), 1)
        delta = np.repeat(np.column_stack(((inputs-mean)/scale, np.ones(len(inputs))))@coefficient, 12, 1)
        error = (data[3]+delta-data[5])*data[4][:, None]
        result["metrics"][role] = {**metrics(error), "mse": float(np.square(error).mean()),
                                    "correction_horizon_std_raw": float((delta*data[4][:, None]).std(1).mean())}
    return state, result


def metrics(error):
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def probe_arrays(bundle, cache, origins, node_indices, *, memory_policy="full"):
    validate_policy(memory_policy)
    global_series = bundle.raw.mean(1,dtype=np.float64) if memory_policy == "global_long" else None
    features, periodic, targets, bases, scales = [], [], [], [], []
    for k, origin in enumerate(origins):
        nodes = node_indices[k]
        if memory_policy == "full":
            recent = bundle.normalized[origin-48:origin, nodes].T
            day = bundle.normalized[origin-288:origin, nodes].T.reshape(len(nodes), 8, 36).mean(-1)
            anchors = np.stack([bundle.normalized[origin-288*d-48:origin-288*d+48, nodes].T
                                for d in range(1, 8)], 1).reshape(len(nodes), 7, 8, 12)
        else:
            history=history_from_bundle(bundle,origin,nodes,memory_policy,global_series=global_series)
            recent=history[-48:].T
            day=history[-288:].T.reshape(len(nodes),8,36).mean(-1)
            anchors=np.stack([history[2064-288*d-48:2064-288*d+48].T
                for d in range(1,8)],1).reshape(len(nodes),7,8,12)
        anchor_hour = anchors.mean(-1)
        context = recent.reshape(len(nodes), 4, 12).mean(-1)
        distance = ((anchor_hour[:, :, :4]-context[:, None])**2).mean(-1)
        logits = -distance/.1; logits -= logits.max(1, keepdims=True)
        weights = np.exp(logits); weights /= weights.sum(1, keepdims=True)
        repeat = (weights[:, :, None]*anchor_hour[:, :, 4:]).sum(1)
        deviation = context-(weights[:, :, None]*anchor_hour[:, :, :4]).sum(1)
        base = cache["base"][k][:, nodes].T
        target = bundle.normalized[origin:origin+48, nodes].T
        short = cache["short_state"][k][nodes].astype(np.float32)
        numerical=mask_numeric_features(np.concatenate((short,recent,day,
            base.reshape(len(nodes),4,12).mean(-1),anchor_hour.reshape(len(nodes),-1),repeat,deviation),1),memory_policy)
        features.append(numerical[:,:188])
        periodic.append(numerical[:,188:])
        targets.append((target-base).reshape(len(nodes), 4, 12).mean(-1))
        bases.append(base); scales.append(bundle.std[nodes])
    return [np.concatenate(x).astype(np.float64) for x in (features, periodic, targets, bases, scales)]


def fit_probe(train, validation, decision):
    result = {}
    for mode in ("short_recent", "short_recent_plus_noise", "short_recent_plus_periodic"):
        matrices = []
        for role, data in enumerate((train, validation, decision)):
            if mode == "short_recent": extra = None
            elif mode == "short_recent_plus_periodic": extra = data[1]
            else: extra = np.random.default_rng(7021+role).standard_normal(data[1].shape)
            matrices.append(np.concatenate((data[0], extra), 1) if extra is not None else data[0])
        mean, scale = matrices[0].mean(0), matrices[0].std(0).clip(.01)
        matrices = [np.column_stack(((x-mean)/scale, np.ones(len(x)))) for x in matrices]
        xtx = matrices[0].T@matrices[0]; xty = matrices[0].T@train[2]
        candidates = []
        for alpha in (.1, 1., 10., 100., 1000., 10000.):
            penalty = np.eye(len(xtx))*alpha; penalty[-1, -1] = 0
            coefficient = np.linalg.solve(xtx+penalty, xty)
            prediction = np.repeat(matrices[1]@coefficient, 12, 1)
            # target = base + residual; reconstruct the original 48-step targets
            # from explicit arrays attached by the caller, never block means.
            error = (validation[3]+prediction-validation[5])*validation[4][:, None]
            candidates.append((metrics(error)["mae"], alpha, coefficient))
        _, alpha, coefficient = min(candidates, key=lambda value: value[0])
        entries = {"alpha_selected_on_val_a": alpha, "parameters": int(coefficient.size)}
        for role, matrix, data in zip(("train", "val_a", "val_b"), matrices, (train, validation, decision)):
            prediction = np.repeat(matrix@coefficient, 12, 1)
            error = (data[3]+prediction-data[5])*data[4][:, None]
            entries[role] = metrics(error)
        result[mode] = entries
    return result

