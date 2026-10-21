"""One-seed V4 repair pilot, with common windows/targets/base and staged fitting.

Preserves V3 evidence. Defaults: 1024 x 16 training window-road pairs, 64
Val-A/B windows across ALL roads. No official test. Numeric/pretrained/random
arms share the repaired interface. Additional switches isolate each repair.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from models.periodic_bert_v4 import PeriodicBertV4
from models.periodic_tokens import periodic_numeric_features, utility_teacher, utility_cross_entropy
from models.periodic_memory_policy import MEMORY_POLICIES, candidate_memory_enabled, metadata as memory_policy_metadata
from research.periodic_relation_data import RelationWindows, physical_neighbors, randomized_neighbor_control
from research.traffic_data import load_pems03, limit_origins, select_cache, valid_pilot_origins
from research.traffic_evaluation import REGIME_NAMES, fit_regime_thresholds, classify_history_regimes
from research.common import autocast, loader, move, seed_all, write_json


SETTINGS = {
    "numeric": {"numeric": True}, "repaired": {}, "random": {"pretrained": False},
    "shortcut_gate": {"late_shortcut": True, "gated_output": True, "retrieval": "fixed", "use_neighbors": False, "adapt_layer_norm": False},
    "no_shortcut_gate": {"gated_output": True, "retrieval": "fixed", "use_neighbors": False, "adapt_layer_norm": False},
    "zero_head_fixed": {"retrieval": "fixed", "use_neighbors": False, "adapt_layer_norm": False},
    "learned_no_ln": {"use_neighbors": False, "adapt_layer_norm": False},
    "no_neighbors": {"use_neighbors": False},
    "no_layer_norm": {"adapt_layer_norm": False},
    "fixed_retrieval": {"retrieval": "fixed"},
    "uniform_retrieval": {"retrieval": "uniform"}, "day7_retrieval": {"retrieval": "day7"},
    "forecast_fix": {"repair": "forecast"},
    "query_fix": {"repair": "query"},
    "full_fix": {"repair": "full"},
    "full_random": {"repair": "full", "pretrained": False},
    "full_numeric": {"repair": "full", "numeric": True},
    "fine_fix": {"repair": "fine"},
    "pattern_fix": {"repair": "pattern"},
    "pattern_random": {"repair": "pattern", "pretrained": False},
    "pattern_numeric": {"repair": "pattern", "numeric": True},
    "strict_fix": {"repair": "strict"},
    "strict_random": {"repair": "strict", "pretrained": False},
    "strict_numeric": {"repair": "strict", "numeric": True},
    "strict_no_pattern": {"repair": "strict", "pattern_mode": "off"},
    "strict_no_phase": {"repair": "strict", "pattern_mode": "no_phase"},
    "strict_no_amplitude": {"repair": "strict", "pattern_mode": "no_amplitude"},
    "strict_no_deviation": {"repair": "strict", "pattern_mode": "no_deviation"},
    "utility_fix": {"repair": "utility", "lora_rank": 16},
    "utility_random": {"repair": "utility", "pretrained": False, "lora_rank": 16},
    "utility_numeric": {"repair": "utility", "numeric": True},
    "structured_control": {"repair":"utility", "lora_rank":16, "refinement_interface":True},
    "structured_fix": {"repair":"utility", "lora_rank":16, "hierarchical_memory":True,
        "relation_dynamics":True, "shape_only_correction":True},
    "structured_random": {"repair":"utility", "lora_rank":16, "pretrained":False,
        "hierarchical_memory":True, "relation_dynamics":True, "shape_only_correction":True},
    "structured_no_hierarchy": {"repair":"utility", "lora_rank":16,
        "relation_dynamics":True, "shape_only_correction":True},
    "structured_no_relation": {"repair":"utility", "lora_rank":16,
        "hierarchical_memory":True, "shape_only_correction":True},
    "structured_no_shape": {"repair":"utility", "lora_rank":16,
        "hierarchical_memory":True, "relation_dynamics":True},
    # Complete the 2x2 hierarchy/relation controls WITHOUT the rejected shape
    # constraint. Do not borrow constrained-variant gains as final-model gains.
    "final_no_hierarchy": {"repair":"utility", "lora_rank":16, "relation_dynamics":True},
    "final_no_relation": {"repair":"utility", "lora_rank":16, "hierarchical_memory":True},
    "final_random": {"repair":"utility", "lora_rank":16, "pretrained":False,
        "hierarchical_memory":True, "relation_dynamics":True},
}
_SPATIAL_COMMON={"repair":"utility","lora_rank":16,"hierarchical_memory":True,
    "refinement_interface":True,"spatial_repair":True}
SETTINGS.update({
    "spatial_reference":{**_SPATIAL_COMMON,"spatial_context_mode":"off"},
    "spatial_full":{**_SPATIAL_COMMON,"spatial_context_mode":"aligned","retain_spatial_content":True},
    "spatial_no_alignment":{**_SPATIAL_COMMON,"spatial_context_mode":"current_only","retain_spatial_content":True},
    "spatial_no_content_skip":{**_SPATIAL_COMMON,"spatial_context_mode":"aligned"},
    "spatial_no_neighbors":{**_SPATIAL_COMMON,"spatial_context_mode":"aligned","retain_spatial_content":True,"use_neighbors":False},
    "spatial_far_neighbors":{**_SPATIAL_COMMON,"spatial_context_mode":"aligned","retain_spatial_content":True},
    "spatial_random_neighbors":{**_SPATIAL_COMMON,"spatial_context_mode":"aligned","retain_spatial_content":True},
    # Factorial isolation of existing paths; ALL use the same nearest sources.
    "isolation_control":{**_SPATIAL_COMMON,"spatial_context_mode":"off",
        "retain_context_content":False,"retain_source_values":False},
    "isolation_qk":{**_SPATIAL_COMMON,"spatial_context_mode":"aligned",
        "retain_context_content":True,"retain_source_values":False},
    "isolation_value":{**_SPATIAL_COMMON,"spatial_context_mode":"off",
        "retain_context_content":False,"retain_source_values":True},
    "isolation_both":{**_SPATIAL_COMMON,"spatial_context_mode":"aligned",
        "retain_context_content":True,"retain_source_values":True},
})
INPUT_NAMES = ("history", "short_state", "neighbor_history", "neighbor_valid", "relation_address", "origin_step")


def make_model(arm, args):
    model = PeriodicBertV4(args.bert_path, node_chunk_size=args.node_chunk,
                          gradient_checkpointing=getattr(args, "gradient_checkpointing", True),
                          memory_policy=getattr(args,"memory_policy","full"), **SETTINGS[arm])
    model.evaluation_node_chunk = getattr(args, "eval_node_chunk", 0) or args.node_chunk
    return model


@contextmanager
def evaluation_chunk(model):
    previous = getattr(model, "node_chunk_size", None)
    if previous is not None: model.node_chunk_size = getattr(model, "evaluation_node_chunk", previous)
    try: yield
    finally:
        if previous is not None: model.node_chunk_size = previous


def validate_execution_update(previous, current):
    """Execution changes may not silently modify the scientific protocol."""
    mutable = {"batch_size", "accumulation", "eval_batch_size", "node_chunk", "eval_node_chunk", "gradient_checkpointing"}
    before = {key: value for key, value in previous.items() if key not in mutable}
    after = {key: value for key, value in current.items() if key not in mutable}
    if before != after: raise ValueError("execution update cannot change scientific configuration")
    if previous["batch_size"]*previous["accumulation"] != current["batch_size"]*current["accumulation"]:
        raise ValueError("execution update must preserve effective batch")


def adapter_state(model):
    names = set(model.adapter_parameter_names)
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in names}


def restore_state(model, state):
    if set(state) != set(model.adapter_parameter_names): raise ValueError("adapter state membership mismatch")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state: parameter.copy_(state[name].to(parameter.device))


def correction(model, batch, **intervention):
    inputs = {name: batch[name] for name in INPUT_NAMES}
    if "neighbor_long_history" in batch:
        inputs["neighbor_long_history"]=batch["neighbor_long_history"]
    if getattr(model, "repair", None) is not None:
        inputs["baseline_prediction"] = batch["base"]
    if intervention.pop("zero_long_history", False): inputs["history"] = torch.zeros_like(inputs["history"])
    return model(**inputs, **intervention)


def raw_error(batch, delta):
    return (batch["base"].float()+delta.float()-batch["target"])*batch["node_std"][:, None]


def training_loss(batch, delta, mode="raw_mae", scale=1.):
    if mode == "raw_mae":
        return raw_error(batch, delta).abs().mean()
    if mode == "hourly_normalized_mse":
        error = batch["base"].float()+delta.float()-batch["target"]
        return error.reshape(error.shape[0], 4, 12, error.shape[2]).mean(2).square().mean()
    if mode == "step_normalized_mae_mse":
        error = batch["base"].float()+delta.float()-batch["target"]
        return error.abs().mean()+.1*error.square().mean()
    if mode == "step_raw_scaled_mae_mse":
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("raw-unit loss requires a positive train-only global scale")
        error = raw_error(batch, delta)/scale
        return error.abs().mean()+.1*error.square().mean()
    raise ValueError("unknown training objective")


def utility_training_loss(model, batch, logits, *, loss_scale):
    # Called ONLY inside the training loop, not evaluation/model.forward.
    # The target defines teacher utility, not a retrieval input or cache feature.
    policy=getattr(model,"memory_policy","full")
    if not candidate_memory_enabled(policy):
        return logits.sum()*0,1.
    with torch.no_grad():
        numeric = model.numeric_calibration(periodic_numeric_features(
            batch["history"], batch["short_state"], batch["base"],memory_policy=policy))
        reference = batch["base"].float()+numeric.repeat_interleave(12,-1).transpose(1,2)
        history = batch["history"].transpose(1,2).reshape(-1,2064)
        truth = batch["target"].transpose(1,2).reshape(-1,48)
        reference = reference.transpose(1,2).reshape(-1,48)
        labels = utility_teacher(history,truth,reference,batch["node_std"].reshape(-1),loss_scale=loss_scale)
    return utility_cross_entropy(logits,labels), float(labels[...,-1].mean())


def eligible_checkpoint(metrics, mode="mae", rmse_limit=None):
    if mode == "mae":
        return True
    if mode != "mae_rmse_guard" or rmse_limit is None or not np.isfinite(rmse_limit):
        raise ValueError("guarded checkpoint selection requires a fixed Val-A reference RMSE")
    # Numerical tolerance only: no data-tuned degradation margin.
    return metrics["rmse"] <= rmse_limit+max(1e-6, 1e-6*rmse_limit)


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, regimes, **intervention):
    model.eval(); absolute, squared, elapsed, correction_std = [], [], [], []
    horizon_abs, horizon_sq = np.zeros(48), np.zeros(48); points = 0
    for cpu in loader(dataset, batch_size, False, 2021):
        batch = move(cpu, device)
        if device.type == "cuda": torch.cuda.synchronize()
        tick = time.perf_counter()
        with evaluation_chunk(model), autocast(device): delta = correction(model, batch, **intervention)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed.append((time.perf_counter()-tick)*1000)
        error = raw_error(batch, delta)
        correction_std.extend((delta.float()*batch["node_std"][:, None]).std(1, unbiased=False).mean(1).cpu().tolist())
        if not torch.isfinite(error).all(): raise RuntimeError("non-finite V4 evaluation")
        absolute.append(error.abs().mean((1, 2)).cpu().numpy())
        squared.append(error.square().mean((1, 2)).cpu().numpy())
        horizon_abs += error.abs().sum((0, 2)).cpu().numpy()
        horizon_sq += error.square().sum((0, 2)).cpu().numpy()
        points += error.shape[0]*error.shape[2]
    mae, mse = np.concatenate(absolute).astype(float), np.concatenate(squared).astype(float)
    metrics = {"mae": float(mae.mean()), "mse": float(mse.mean()), "rmse": float(np.sqrt(mse.mean())),
               "horizon_mae": (horizon_abs/points).tolist(), "horizon_rmse": np.sqrt(horizon_sq/points).tolist(),
               "adapter_ms_per_batch": float(np.mean(elapsed[1:] or elapsed)),
               "adapter_ms_per_window": float(sum(elapsed)/len(mae)),
               "correction_horizon_std_raw": float(np.mean(correction_std)), "regimes": {}}
    for label, name in enumerate(REGIME_NAMES):
        chosen = regimes == label
        if chosen.any(): metrics["regimes"][name] = {"mae": float(mae[chosen].mean()),
                                                      "rmse": float(np.sqrt(mse[chosen].mean())), "windows": int(chosen.sum())}
    return metrics, {"origin": dataset.origins, "window_mae": mae, "window_mse": mse, "regime": regimes}


def cached_metrics(dataset):
    target = np.stack([dataset.bundle.normalized[o:o+48] for o in dataset.origins])
    error = (dataset.cache["base"].astype(np.float32)-target)*dataset.bundle.std[None, None]
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "mse": float(np.square(error).mean())}


def linear_reference(datasets):
    # Reuse the checked closed-form diagnostic on the NEW exact window/node
    # budget. This is a strong numeric reference, not a language-model result.
    from research.periodic_reference import probe_arrays, fit_probe
    roles = {}
    for role, dataset in datasets.items():
        arrays = probe_arrays(dataset.bundle, dataset.cache, dataset.origins, dataset.node_indices,
            memory_policy=getattr(dataset,"memory_policy","full"))
        arrays.append(np.concatenate([
            dataset.bundle.normalized[o:o+48][:, dataset.node_indices[index]].T
            for index, o in enumerate(dataset.origins)
        ]).astype(float))
        roles[role] = arrays
    return fit_probe(roles["train"], roles["val_a"], roles["val_b"])


def make_optimizer(model, args):
    groups = []
    for is_backbone in (False, True):
        values = [p for name, p in model.named_parameters()
                  if name in model.adapter_parameter_names and name.startswith("bert.") == is_backbone]
        if values:
            groups.append({"params": values, "lr": args.backbone_lr if is_backbone and not model.numeric else args.interface_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def checkpoint_payload(model, epoch):
    state = {"adapter_state": adapter_state(model), "epoch": epoch}
    if getattr(model,"spatial_repair",False):
        state["spatial_contract"]={key:getattr(model,key) for key in
            ("spatial_context_mode","retain_spatial_content","retain_context_content",
             "retain_source_values","use_neighbors","memory_policy")}
    if getattr(model, "uses_numeric_route", False):
        state["numeric_calibration"] = model.numeric_calibration.export()
    return state


def restore_checkpoint(model, state):
    if "spatial_contract" in state:
        contract=dict(state["spatial_contract"])
        for key in ("retain_context_content","retain_source_values"):
            contract.setdefault(key,contract["retain_spatial_content"])
        expected={key:getattr(model,key,None) for key in contract}
        if expected!=contract: raise ValueError("spatial checkpoint interface/policy mismatch")
    if getattr(model, "uses_numeric_route", False):
        if "numeric_calibration" not in state: raise ValueError("full repair checkpoint lacks numeric route")
        model.numeric_calibration.install(state["numeric_calibration"])
    restore_state(model, state["adapter_state"])


def save_checkpoint(path, model, optimizer, epoch, best, best_epoch, stale, log):
    torch.save({**checkpoint_payload(model, epoch), "optimizer_state": optimizer.state_dict(),
                "best_value": best, "best_epoch": best_epoch, "stale": stale,
                "log": log, "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def train(model, dataset, validation, regimes, device, args, output, arm, report, deadline):
    optimizer = make_optimizer(model, args)
    selection_mode = getattr(args, "selection_mode", "mae")
    rmse_limit = getattr(args, "selection_rmse_limit", None)
    last, best_path = output/f"{arm}_last.pt", output/f"{arm}_best.pt"
    epoch = 0; best_epoch = 0; stale = 0; log = []; gradient_audit = {}
    if args.resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        restore_checkpoint(model, state); optimizer.load_state_dict(state["optimizer_state"])
        epoch, best, best_epoch, stale, log = [state[key] for key in ("epoch", "best_value", "best_epoch", "stale", "log")]
        torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        # Include the exact zero-correction baseline as checkpoint epoch zero.
        initial, _ = evaluate(model, validation, device, args.eval_batch_size, regimes)
        if not eligible_checkpoint(initial, selection_mode, rmse_limit):
            raise ValueError("epoch-zero reference violates the declared Val-A RMSE constraint")
        best = initial["mae"]
        torch.save(checkpoint_payload(model, 0), best_path)
        save_checkpoint(last, model, optimizer, 0, best, 0, 0, [])
    stop_reason = "epoch_cap"
    while epoch < args.epochs:
        if not getattr(args,"matched_updates",False) and stale >= args.patience and epoch >= args.min_epochs:
            stop_reason = "early_stop"; break
        if time.time() >= deadline: stop_reason = "time_budget"; break
        model.set_training_stage(epoch < args.warmup_epochs); model.train()
        batches = loader(dataset, args.batch_size, True, args.seed*100+epoch)
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter(); loss_sum = 0.; objective_sum = 0.; seen = 0; completed = True
        utility_sum = 0.; abstain_sum = 0.
        for step, cpu in enumerate(batches):
            if time.time() >= deadline: completed = False; stop_reason = "time_budget_mid_epoch"; break
            batch = move(cpu, device)
            utility = getattr(model,"uses_utility_supervision",False)
            with autocast(device):
                result = correction(model,batch,return_auxiliary=True) if utility else correction(model,batch)
            delta, auxiliary = result if utility else (result,None)
            loss = training_loss(batch, delta, getattr(args, "loss_mode", "raw_mae"),
                                 scale=getattr(args, "raw_loss_scale", 1.))
            if utility and getattr(args,"utility_weight",.1)>0:
                auxiliary_loss, abstain_fraction = utility_training_loss(model,batch,auxiliary["utility_logits"],
                    loss_scale=getattr(args,"raw_loss_scale",1.))
                loss = loss+getattr(args,"utility_weight",.1)*auxiliary_loss
                utility_sum += float(auxiliary_loss.detach())*len(batch["history"])
                abstain_sum += abstain_fraction*len(batch["history"])
            if not torch.isfinite(loss): raise RuntimeError("non-finite V4 training loss")
            group = min(args.accumulation, len(batches)-(step//args.accumulation)*args.accumulation)
            (loss/group).backward()
            # Audit after zero-head startup AND after backbone unfreezing.
            if epoch == args.warmup_epochs and step == min(7, len(batches)-1):
                gradient_audit = {name: None if p.grad is None else float(p.grad.abs().sum())
                                  for name, p in model.named_parameters() if name.endswith("lora_B")
                                  or name in ("retrieval_query.weight", "retrieval_key.weight", "relation_projection.weight",
                                              "patch_encoder.statistics_projection.weight", "bert.embeddings.LayerNorm.weight",
                                              "pattern_operator.control.weight", "pattern_operator.query.weight",
                                              "pattern_operator.key.weight", "pattern_operator.amplitude.weight", "pattern_gate",
                                              "pattern_operator.abstain.weight","utility_context_pool.2.weight",
                                              "relation_change_bias.network.2.weight",
                                              "spatial_context.attention.in_proj_weight",
                                              "spatial_context.relation_key.weight", "spatial_context.time_key",
                                              "spatial_context.spatial_gate", "spatial_context.bert_gate")}
            if (step+1) % args.accumulation == 0 or step+1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            loss_sum += float(raw_error(batch, delta).abs().mean().detach())*len(batch["history"]); seen += len(batch["history"])
            objective_sum += float(loss.detach())*len(batch["history"])
            if (step+1) % 64 == 0:
                progress = {"arm": arm, "epoch_in_progress": epoch+1, "batch": step+1,
                            "batches": len(batches), "train_mae_so_far": loss_sum/seen,
                            "stage": "interface_only" if epoch < args.warmup_epochs else "joint"}
                write_json(output/"progress.json", progress)
                print(json.dumps(progress), flush=True)
        if not completed: break
        epoch += 1
        metrics, _ = evaluate(model, validation, device, args.eval_batch_size, regimes)
        eligible = eligible_checkpoint(metrics, selection_mode, rmse_limit)
        if eligible and metrics["mae"] < best-1e-4:
            best, best_epoch, stale = metrics["mae"], epoch, 0
            torch.save(checkpoint_payload(model, epoch), best_path)
        elif epoch > args.warmup_epochs:
            stale += 1
        row = {"epoch": epoch, "train_mae": loss_sum/seen, "val_a_mae": metrics["mae"],
               "train_objective": objective_sum/seen,
               "eligible_under_rmse_guard": eligible,
               "val_a_rmse": metrics["rmse"], "seconds": time.perf_counter()-tick,
               "stage": "interface_only" if epoch <= args.warmup_epochs else "joint", "best_epoch": best_epoch}
        if getattr(model,"uses_utility_supervision",False):
            enabled=getattr(args,"utility_weight",.1)>0 and candidate_memory_enabled(getattr(model,"memory_policy","full"))
            row.update(utility_cross_entropy=utility_sum/seen if enabled else None,
                teacher_abstain_fraction=abstain_sum/seen if enabled else None,
                utility_weight=getattr(args,"utility_weight",.1),teacher_supervision_enabled=enabled)
        log.append(row); save_checkpoint(last, model, optimizer, epoch, best, best_epoch, stale, log)
        if epoch in getattr(args,"comparison_epochs",[]):
            torch.save(checkpoint_payload(model,epoch),output/f"{arm}_epoch_{epoch:03d}.pt")
        write_json(output/"progress.json", {"arm": arm, **row})
        report["active_run"] = {"arm": arm, **row}; write_json(output/"report.json", report)
        print(json.dumps({"arm": arm, **row}), flush=True)
    model.set_training_stage(False)
    restore_checkpoint(model, torch.load(best_path, map_location="cpu", weights_only=False))
    return {"epochs_completed": epoch, "best_epoch": best_epoch, "stop_reason": stop_reason,
            "optimizer_updates_completed_epochs": epoch*((((len(dataset)+args.batch_size-1)//args.batch_size)+args.accumulation-1)//args.accumulation),
            "selection_mode": selection_mode, "selection_rmse_limit": rmse_limit,
            "converged": stop_reason == "early_stop", "training_log": log,
            "gradient_audit": gradient_audit, "best_checkpoint": str(best_path),
            "peak_vram_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0}


def run(args):
    output = args.output_dir.resolve(); cache_root = args.cache_root.resolve()
    if output == cache_root or cache_root in output.parents: raise ValueError("cannot overwrite original evidence")
    if output.exists() and any(output.iterdir()) and not args.resume: raise FileExistsError("output exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True); args.failure_output = output
    configuration = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
                     if key not in ("resume", "failure_output", "budget_hours", "execution_update")}
    sources = ("models/periodic_bert_v4.py", "models/periodic_tokens.py", "models/bert_backbone.py",
               "models/periodic_refinement.py", "models/periodic_memory_policy.py", "models/spatial_context.py",
               "research/periodic_relation_data.py", "research/run_periodic_bert_repair.py",
               "research/traffic_data.py", "research/traffic_evaluation.py", "research/common.py",
               "research/periodic_reference.py")
    snapshot = output/"source_snapshot"; snapshot.mkdir(exist_ok=True)
    if args.resume:
        report = json.loads((output/"report.json").read_text())
        active_snapshot = Path(report.get("active_source_snapshot", snapshot))
        if any(not (active_snapshot/Path(name).name).is_file() for name in sources):
            raise ValueError("historical source layout: do not resume in place after consolidation; "
                             "use a new output directory; existing V4 adapter checkpoint loading remains supported")
        if getattr(args, "execution_update", False):
            validate_execution_update(report["configuration"], configuration)
            changed = [name for name in sources if (ROOT/name).read_bytes() != (active_snapshot/Path(name).name).read_bytes()]
            if any(name != "research/run_periodic_bert_repair.py" for name in changed):
                raise ValueError("execution migration cannot change model, data or other scientific sources")
            events = report.setdefault("execution_changes", [])
            updated_snapshot = output/"execution_snapshots"/f"update_{len(events)+1:03d}"
            updated_snapshot.mkdir(parents=True, exist_ok=False)
            for name in sources: shutil.copy2(ROOT/name, updated_snapshot/Path(name).name)
            events.append({"at": time.time(), "previous_configuration": report["configuration"],
                           "new_configuration": configuration, "source_changes": changed,
                           "source_snapshot": str(updated_snapshot), "effective_batch_preserved": True,
                           "scope": "batch packing, chunk sizes, checkpoint recomputation; no sample/loss/optimizer change",
                           "numerics": "dropout RNG consumption and floating point accumulation order may differ; not bitwise replay"})
            report["configuration"] = configuration
            report["active_source_snapshot"] = str(updated_snapshot)
        else:
            if report["configuration"] != configuration: raise ValueError("resume configuration mismatch")
            for name in sources:
                if (ROOT/name).read_bytes() != (active_snapshot/Path(name).name).read_bytes(): raise ValueError("source changed since run snapshot")
    else:
        for name in sources: shutil.copy2(ROOT/name, snapshot/Path(name).name)
        report = {"status": "preparing", "started_at": time.time(), "configuration": configuration,
                  "runs": [], "protocol": {"seed": args.seed, "test_evaluated": False,
                    "checkpoint_selection": "Val-A MAE, including unchanged baseline at epoch zero",
                    "decision_split": "exploratory Val-B, not an untouched publication test",
                    "long_histories_overlap": True, "loss": args.loss_mode,
                    "same_window_target_nodes_and_base_for_all_arms": True,
                    "warmup": "interface only, then LoRA and optional LayerNorm; numeric backbone trainable throughout",
                    "calendar": "relative index phases only; CSV real-calendar origin not trusted",
                    "source_identity_boundary": "provided graph row order assumed to match flow columns",
                    "counterfactual_scope": "adapter only, frozen common base and short state held fixed"}}
    write_json(output/"report.json", report)
    report["protocol"]["memory_ablation"]=memory_policy_metadata(getattr(args,"memory_policy","full"))
    bundle = load_pems03(ROOT/"dataset/PEMS03/PEMS03.csv", .7, .1)
    contract = json.loads((cache_root/"contract.json").read_text())
    with np.load(cache_root/"origins.npz") as stored:
        if not np.array_equal(stored["mean"], bundle.mean) or not np.array_equal(stored["std"], bundle.std):
            raise ValueError("train-only normalization does not match frozen cache")
        selected, boundary_migration = valid_pilot_origins(bundle,
            {role: stored[role] for role in ("train", "val_a", "val_b")},
            train_windows=args.train_windows, val_windows=args.val_windows,
            target_gap=getattr(args, "target_gap_steps", 48))
    report["protocol"]["boundary_migration"] = boundary_migration
    args.raw_loss_scale = float(np.median(bundle.std))
    report["protocol"]["raw_loss_scale"] = {"value": args.raw_loss_scale,
        "source": "median across per-node standard deviations fitted on original training rows only",
        "formula": "raw MAE/s +0.1 raw MSE/s^2", "applies_to_loss_mode": "step_raw_scaled_mae_mse"}
    graph_path=ROOT/"dataset/PEMS03/PEMS03_adj.csv"
    neighbors = physical_neighbors(graph_path, bundle.raw.shape[1], args.max_neighbors,
        order=getattr(args,"neighbor_order","legacy"))
    if getattr(args,"neighbor_control","physical")=="matched_random":
        import pandas as pd
        neighbors=randomized_neighbor_control(neighbors,pd.read_csv(graph_path).to_numpy(np.float32),args.seed)
    include_long_sources=any(SETTINGS[arm].get("spatial_repair",False) for arm in args.arms)
    datasets = {}
    for role, origins in selected.items():
        meta = json.loads((cache_root/f"cache_{role}.json").read_text())
        if meta["base_checkpoint_sha256"] != contract["baseline_sha256"] or meta["role"] != role:
            raise ValueError("base cache identity mismatch")
        with np.load(cache_root/f"cache_{role}.npz") as cached:
            values = select_cache({key: cached[key] for key in cached.files}, origins)
        datasets[role] = RelationWindows(bundle, origins, values, neighbors,
                                        train_nodes=args.train_nodes if role == "train" else 0, seed=args.seed,
                                        memory_policy=getattr(args,"memory_policy","full"),
                                        include_neighbor_long_history=include_long_sources)
    np.savez(output/"selection.npz", **selected, train_node_indices=datasets["train"].node_indices,
             neighbor_indices=neighbors.indices, neighbor_valid=neighbors.valid)
    history_indices = np.arange(-48, 0)
    thresholds = fit_regime_thresholds(bundle.raw[selected["train"][:, None]+history_indices], recent_steps=48)
    regimes = {role: classify_history_regimes(bundle.raw[selected[role][:, None]+history_indices], thresholds, recent_steps=48)
               for role in ("val_a", "val_b")}
    report["baseline"] = {role: cached_metrics(datasets[role]) for role in ("val_a", "val_b")}
    report["topology"] = neighbors.metadata
    report["protocol"]["spatial_repair"]={"include_neighbor_long_history":include_long_sources,
        "neighbor_order":getattr(args,"neighbor_order","legacy"),
        "neighbor_control":getattr(args,"neighbor_control","physical"),
        "aligned_keys":"current and historical source CONTEXT only; no candidate continuation enters keys",
        "content_retention":"shared32d observed-content residual plus gated BERT feature; not a language-pretraining claim",
        "claim_test":"must improve no-neighbor and matched-random controls, not just legacy representation"}
    report["training_pairs_per_epoch"] = len(datasets["train"])*args.train_nodes
    report["base_checkpoint_identity"] = contract["baseline_sha256"]
    if not args.smoke and "linear_reference" not in report:
        print("Fitting same-budget numeric reference; no official test", flush=True)
        report["linear_reference"] = linear_reference(datasets)
    if args.repair_pilot:
        from research.periodic_reference import fit_numeric_calibration
        if "linear_reference" not in report:
            report["linear_reference"] = linear_reference(datasets)
        calibration_path = output/"numeric_calibration.pt"
        if args.resume and calibration_path.exists():
            calibration = torch.load(calibration_path, weights_only=False)
            numeric_state = calibration["state"]
        else:
            alpha = report["linear_reference"]["short_recent_plus_periodic"]["alpha_selected_on_val_a"]
            numeric_state, numeric_report = fit_numeric_calibration(datasets, alpha)
            numeric_state = {name: torch.from_numpy(value) for name, value in numeric_state.items()}
            torch.save({"state": numeric_state, "metrics": numeric_report}, calibration_path)
            report["numeric_control"] = numeric_report
        report["protocol"]["repair_comparison"] = {
            "forecast_fix": "direct baseline forecast input and hourly affine calibration",
            "query_fix": "plus balanced queries, four horizon-specific heads and native query residual",
            "full_fix": "plus fixed numeric periodic route; neural branch starts at zero",
            "full_random": "same complete interface, random BERT + LoRA/LayerNorm",
            "full_numeric": "same complete interface, small numerical Transformer",
            "fine_fix": "48 distinct step corrections and per-step supervision; fixed numeric route",
            "pattern_fix": "fine interface plus context-only day/phase retrieval, amplitude and deviation correction",
            "pattern_random": "same pattern interface with random frozen BERT plus adaptation",
            "pattern_numeric": "same pattern interface with small trainable numerical Transformer, not FLOPs matched",
            "strict_fix": "isolated context-only Q/K, historical outcomes as V, 48-step prediction",
            "strict_random": "same strict interface, random frozen BERT plus adaptation",
            "strict_numeric": "same strict interface, small trainable Transformer; not FLOPs matched",
            "strict_no_pattern": "same architecture, added neural pattern operator disabled; numeric route unchanged",
            "strict_ablation_scope": "no_phase/no_amplitude/no_deviation affect additional neural operator only",
            "utility_fix": "joint repair: raw templates, train-only predictive utility with abstention, matched context tokens, no dynamic native-query head bypass; rank16 LoRA",
            "utility_random": "identical joint repair and rank16 LoRA with random frozen BERT",
            "utility_numeric": "same joint interface with small trainable numeric Transformer, not FLOPs matched",
            "claim_boundary": "gains shared with numeric-only control are not PLM gains"}
    args.selection_rmse_limit = (report["numeric_control"]["metrics"]["val_a"]["rmse"]
        if getattr(args, "selection_mode", "mae") == "mae_rmse_guard" else None)
    report["protocol"]["checkpoint_selection"] = {
        "mode": getattr(args, "selection_mode", "mae"), "split": "Val-A only",
        "rmse_ceiling": args.selection_rmse_limit, "includes_epoch_zero": True,
        "constraint_scope": "validation only; does not guarantee held-out improvement"}
    report["protocol"]["matched_updates"] = {
        "enabled": getattr(args,"matched_updates",False),
        "early_stopping_disabled": getattr(args,"matched_updates",False),
        "primary_comparison": "common final epoch, with Val-A-selected checkpoint separately reported",
        "budget_interruption_invalidates_equal_update_claim": True,
        "utility_teacher": "training targets only; never model inputs, validation fitting or inference labels"}
    report["protocol"]["comparison_epochs"]={
        "epochs":getattr(args,"comparison_epochs",[]),
        "evaluation":"Saved at epoch boundary, evaluated only AFTER all requested training epochs; never used for training decisions",
        "value_scope":"Changing source Values also affects the existing utility_source_read query-conditioning path; direct context Q/K encoder remains unchanged in Value-only arm"}
    report["protocol"]["utility_supervision"] = {
        "weight":getattr(args,"utility_weight",.1),
        "enabled_for_utility_arms":getattr(args,"utility_weight",.1)>0 and candidate_memory_enabled(getattr(args,"memory_policy","full")),
        "zero_weight_ablation":"identical architecture/forward/optimizer; forecast loss only; teacher is not computed"}
    report["status"] = "running"; report["process_id"] = os.getpid()
    write_json(output/"report.json", report)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    deadline = time.time()+args.budget_hours*3600
    remaining = [arm for arm in args.arms if not any(row["arm"] == arm for row in report["runs"])]
    per_arm_seconds = args.budget_hours*3600/max(1, len(remaining))
    for arm in remaining:
        if time.time() >= deadline: break
        seed_all(args.seed); model = make_model(arm, args).to(device)
        if getattr(model, "uses_numeric_route", False):
            model.numeric_calibration.install(numeric_state)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        metadata = train(model, datasets["train"], datasets["val_a"], regimes["val_a"], device, args,
                         output, arm, report, min(deadline, time.time()+per_arm_seconds))
        metrics, arrays = evaluate(model, datasets["val_b"], device, args.eval_batch_size, regimes["val_b"])
        np.savez(output/f"{arm}_val_b.npz", **arrays)
        row = {"arm": arm, **metadata, "metadata": model.metadata, "metrics": metrics,
               "gain_over_frozen_base_percent": {key: 100*(1-metrics[key]/report["baseline"]["val_b"][key]) for key in ("mae", "rmse", "mse")}}
        if getattr(model,"uses_utility_supervision",False):
            row["metadata"]["utility_supervision_enabled_in_training"]=getattr(args,"utility_weight",.1)>0 and candidate_memory_enabled(getattr(model,"memory_policy","full"))
        if getattr(args,"matched_updates",False):
            restore_checkpoint(model,torch.load(output/f"{arm}_last.pt",map_location="cpu",weights_only=False))
            final_metrics, final_arrays = evaluate(model,datasets["val_b"],device,args.eval_batch_size,regimes["val_b"])
            np.savez(output/f"{arm}_matched_final_val_b.npz",**final_arrays)
            row["matched_final_metrics"] = final_metrics
            row["matched_final_epoch"] = metadata["epochs_completed"]
            row["matched_update_budget_complete"] = metadata["epochs_completed"] == args.epochs and metadata["stop_reason"] == "epoch_cap"
        if getattr(args,"comparison_epochs",[]):
            row["matched_epoch_comparisons"]={}
            for epoch in args.comparison_epochs:
                path=output/f"{arm}_epoch_{epoch:03d}.pt"
                if not path.exists(): continue
                if epoch==metadata["epochs_completed"] and getattr(args,"matched_updates",False):
                    epoch_metrics,epoch_arrays=final_metrics,final_arrays
                else:
                    restore_checkpoint(model,torch.load(path,map_location="cpu",weights_only=False))
                    epoch_metrics,epoch_arrays=evaluate(model,datasets["val_b"],device,args.eval_batch_size,regimes["val_b"])
                np.savez(output/f"{arm}_epoch_{epoch:03d}_val_b.npz",**epoch_arrays)
                row["matched_epoch_comparisons"][str(epoch)]={"epoch":epoch,"metrics":epoch_metrics,
                    "optimizer_updates":epoch*(metadata["optimizer_updates_completed_epochs"]//max(1,metadata["epochs_completed"]))}
            if getattr(args,"matched_updates",False):
                restore_checkpoint(model,torch.load(output/f"{arm}_last.pt",map_location="cpu",weights_only=False))
        if "linear_reference" in report:
            reference = report["linear_reference"]["short_recent_plus_periodic"]["val_b"]
            row["gain_over_linear_periodic_percent"] = {key: 100*(1-metrics[key]/reference[key]) for key in ("mae", "rmse")}
        if "numeric_control" in report:
            reference = report["numeric_control"]["metrics"]["val_b"]
            row["gain_over_numeric_control_percent"] = {key: 100*(1-metrics[key]/reference[key]) for key in ("mae", "rmse", "mse")}
            if getattr(args,"matched_updates",False):
                row["matched_final_gain_over_numeric_control_percent"] = {
                    key:100*(1-final_metrics[key]/reference[key]) for key in ("mae","rmse","mse")}
        report["runs"].append(row); write_json(output/"report.json", report)
        print(json.dumps({"completed": arm, "mae": metrics["mae"], "rmse": metrics["rmse"],
            "stop_reason": metadata["stop_reason"], "matched_final_metrics": row.get("matched_final_metrics"),
            "optimizer_updates_completed_epochs": metadata["optimizer_updates_completed_epochs"]}), flush=True)
        if arm == "spatial_full" and not args.smoke and time.time() < deadline:
            # Common-final checkpoint; fixed weights, supplementary evidence only.
            diagnostic_origins=limit_origins(selected["val_a"],8)
            diagnostic=RelationWindows(bundle,diagnostic_origins,
                select_cache(datasets["val_a"].cache,diagnostic_origins),neighbors,
                memory_policy=args.memory_policy,include_neighbor_long_history=True)
            diagnostic_regimes=classify_history_regimes(
                bundle.raw[diagnostic_origins[:,None]+history_indices],thresholds,recent_steps=48)
            row["spatial_interventions_val_a"]={"checkpoint":"common final epoch",
                "scope":"added adapter only; frozen baseline spatial state is unchanged"}
            for mode in ("correct","zero","source_shuffle","history_shuffle","time_shuffle","address_shuffle"):
                if time.time() >= deadline: break
                value,_=evaluate(model,diagnostic,device,args.eval_batch_size,diagnostic_regimes,source_mode=mode)
                row["spatial_interventions_val_a"][mode]=value
            write_json(output/"report.json",report)
        # Mechanism diagnostics use a small Val-A subset, never select from Val-B.
        if arm == "repaired" and not args.smoke and time.time() < deadline:
            diagnostic_origins = limit_origins(selected["val_a"], 8)
            diagnostic = RelationWindows(bundle, diagnostic_origins,
                select_cache(datasets["val_a"].cache, diagnostic_origins), neighbors)
            diagnostic_regimes = classify_history_regimes(bundle.raw[diagnostic_origins[:, None]+history_indices], thresholds, recent_steps=48)
            row["val_a_adapter_interventions"] = {}
            for name, kwargs in {"zero_long": {"zero_long_history": True}, "wrong_hour": {"anchor_shift_steps": 12},
                                 "zero_sources": {"source_mode": "zero"}, "shuffle_sources": {"source_mode": "source_shuffle"},
                                 "shuffle_source_time": {"source_mode": "time_shuffle"}, "shuffle_addresses": {"source_mode": "address_shuffle"}}.items():
                if time.time() >= deadline: break
                value, _ = evaluate(model, diagnostic, device, args.eval_batch_size, diagnostic_regimes, **kwargs)
                row["val_a_adapter_interventions"][name] = value
            normal, _ = evaluate(model, diagnostic, device, args.eval_batch_size, diagnostic_regimes)
            row["val_a_adapter_interventions"]["normal"] = normal
            write_json(output/"report.json", report)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    if args.smoke: report["status"] = "smoke_completed_not_accuracy_evidence"
    elif len(report["runs"]) != len(args.arms): report["status"] = "budget_exhausted_partial"
    elif any(row["stop_reason"].startswith("time_budget") for row in report["runs"]):
        report["status"] = "all_arms_evaluated_budget_limited"
    else: report["status"] = "completed_pilot"
    report["ended_at"] = time.time(); report.pop("active_run", None)
    write_json(output/"report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"pilot_results/periodic_bert_v4")
    parser.add_argument("--cache-root", type=Path, default=ROOT/"pilot_results/periodic_pattern_v1")
    parser.add_argument("--bert-path", type=Path, default=ROOT/"huggingface/google_bert")
    parser.add_argument("--arms", nargs="+", choices=tuple(SETTINGS))
    parser.add_argument("--repair-pilot", action="store_true")
    parser.add_argument("--loss-mode", choices=("raw_mae", "hourly_normalized_mse", "step_normalized_mae_mse", "step_raw_scaled_mae_mse"), default="raw_mae")
    parser.add_argument("--selection-mode", choices=("mae", "mae_rmse_guard"), default="mae")
    parser.add_argument("--target-gap-steps", type=int, default=48)
    parser.add_argument("--matched-updates", action="store_true")
    parser.add_argument("--comparison-epochs",nargs="*",type=int,default=[])
    parser.add_argument("--utility-weight", type=float, default=.1)
    parser.add_argument("--memory-policy", choices=MEMORY_POLICIES, default="full")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--train-windows", type=int, default=1024)
    parser.add_argument("--train-nodes", type=int, default=16)
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--node-chunk", type=int, default=64)
    parser.add_argument("--eval-node-chunk", type=int, default=256)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--neighbor-order",choices=("legacy","nearest"),default="legacy")
    parser.add_argument("--neighbor-control",choices=("physical","matched_random"),default="physical")
    parser.add_argument("--interface-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--budget-hours", type=float, default=4.)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execution-update", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.arms is None:
        args.arms = ["forecast_fix", "query_fix", "full_fix", "full_random", "full_numeric"] if args.repair_pilot else ["numeric", "repaired", "random"]
    if any(SETTINGS[arm].get("repair") for arm in args.arms) and not args.repair_pilot:
        parser.error("repair arms require --repair-pilot")
    if args.selection_mode == "mae_rmse_guard" and not args.repair_pilot:
        parser.error("RMSE-guarded selection requires the fixed numeric repair reference")
    if any(SETTINGS[arm].get("repair") in ("fine", "pattern", "strict", "utility") for arm in args.arms) and args.loss_mode == "hourly_normalized_mse":
        parser.error("fine/pattern repairs require per-step supervision, not hourly-mean-only loss")
    if any(SETTINGS[arm].get("repair") in ("strict","utility") for arm in args.arms):
        if args.loss_mode != "step_raw_scaled_mae_mse" or args.selection_mode != "mae_rmse_guard":
            parser.error("strict repair requires raw scaled step loss and Val-A RMSE-guarded checkpoint selection")
    if any(SETTINGS[arm].get("repair") == "utility" for arm in args.arms):
        if not args.matched_updates or not np.isfinite(args.utility_weight) or args.utility_weight < 0:
            parser.error("joint utility repair requires matched update budgets and a nonnegative finite utility weight; zero is the no-teacher ablation")
    if args.memory_policy != "full" and any(SETTINGS[arm].get("repair") != "utility" for arm in args.arms):
        parser.error("memory ablations use the utility interface and refitted numerical route")
    if include_spatial := any(SETTINGS[arm].get("spatial_repair",False) for arm in args.arms):
        if any(not SETTINGS[arm].get("spatial_repair",False) for arm in args.arms):
            parser.error("spatial repair controls run separately from historical arms")
    if "spatial_far_neighbors" in args.arms and args.neighbor_order!="legacy":
        parser.error("spatial_far_neighbors is explicitly the legacy descending-distance control")
    if "spatial_random_neighbors" in args.arms and args.neighbor_control!="matched_random":
        parser.error("spatial_random_neighbors requires matched_random source reassignment")
    if args.target_gap_steps < 48:
        parser.error("new pilots require at least48 unused target timestamps between validation roles")
    if args.smoke:
        args.train_windows, args.train_nodes, args.val_windows = 8, 4, 2
        args.epochs, args.min_epochs, args.warmup_epochs = 2, 2, 1
        if args.comparison_epochs: args.comparison_epochs=[1,2]
    if args.comparison_epochs and (not args.matched_updates or
        len(set(args.comparison_epochs))!=len(args.comparison_epochs) or
        any(epoch<1 or epoch>args.epochs for epoch in args.comparison_epochs)):
        parser.error("comparison epochs require matched updates and unique valid epoch numbers")
    if any(arm.startswith("isolation_") for arm in args.arms) and (
        args.neighbor_order!="nearest" or args.neighbor_control!="physical"):
        parser.error("isolation factorial requires common nearest physical sources")
    if min(args.train_windows, args.train_nodes, args.val_windows, args.epochs, args.min_epochs,
           args.batch_size, args.eval_batch_size, args.accumulation, args.node_chunk, args.max_neighbors) < 1:
        parser.error("invalid data or compute budget")
    if not 0 <= args.warmup_epochs < args.epochs or not args.min_epochs <= args.epochs:
        parser.error("invalid stage lengths")
    if min(args.budget_hours, args.patience, args.interface_lr, args.backbone_lr) <= 0:
        parser.error("budget, patience and learning rates must be positive")
    if len(set(args.arms)) != len(args.arms): parser.error("duplicate arms")
    if args.execution_update and not args.resume: parser.error("execution update requires resume")
    torch.set_num_threads(4)
    try: run(args)
    except Exception:
        if hasattr(args, "failure_output"):
            write_json(args.failure_output/"failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__": main()
