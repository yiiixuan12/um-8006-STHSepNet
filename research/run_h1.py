"""Portable H1 training/evaluation; no historical queue, pilot or gate required.

The published fast model and original shared scientific functions are unchanged.
Fresh baselines default to seed 2021 and use at most 20 epochs; released
historical baselines retain their own recorded budgets. Adapters use at most
5 epochs. Independent paired-case baseline seeds 2022/2023 are also supported.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASETS = ("pems03", "metr_la", "bj500", "bike_inflow", "bike_outflow")
VERSION = "h1_portable_fast_v1"
MODEL_VERSION = "fast_periodic_v1"
BASELINE_SEED = 2021
SOURCES = (
    "research/run_h1.py", "research/run_coursework.py", "research/run_coursework_extra.py",
    "research/coursework_config.json", "research/common.py", "research/coursework_data.py",
    "research/traffic_data.py", "research/run_periodic_bert_repair.py",
    "research/periodic_relation_data.py", "research/traffic_evaluation.py",
    "research/fast_execution.py", "research/bike_validation_support.py",
    "models/sthsepnet_h1.py", "models/periodic_bert_fast.py", "models/periodic_fast_hourly.py",
    "models/periodic_hourly.py", "models/periodic_bert_v4.py", "models/periodic_tokens.py",
    "models/periodic_memory_policy.py", "models/bert_backbone.py", "models/spatial_context.py",
    "models/periodic_refinement.py", "models/utility_recoupling.py", "models/holt_operator.py",
    "layers/h1/__init__.py", "layers/h1/STHGNN.py", "layers/h1/AdaGNN.py", "layers/h1/HyperGNN.py",
    "layers/Embed.py", "layers/StandardNorm.py", "layers/FusionGate.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def history_length(dataset):
    return 240 if dataset.startswith("bike_") else 2064


def checkpoint_path(directory, stage):
    path = Path(directory) / f"{stage}_best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing final checkpoint: {path}")
    return path


def validate_arguments(args):
    if args.arm != "full" and (args.dataset != "bike_inflow" or args.stage == "baseline"):
        raise ValueError("Non-full ablation arms apply only to BIKE-Inflow improved/evaluate")
    if args.stage == "improved" and args.baseline is None:
        raise ValueError("--baseline must point to a completed baseline directory")
    if args.stage == "evaluate" and args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required for evaluate")
    if args.max_windows is not None and (args.stage != "evaluate" or args.max_windows < 1):
        raise ValueError("--max-windows is a positive evaluate-only smoke limit")
    if args.resume and args.stage == "evaluate":
        raise ValueError("--resume is for baseline/improved training only")
    if (args.replay_training_rng or args.rng_checkpoint) and (args.stage != "evaluate" or args.max_windows is not None or args.origins):
        raise ValueError("Training-RNG replay requires full, unlimited evaluation")
    if args.origins and (args.stage != "evaluate" or args.max_windows is not None):
        raise ValueError("--origins requires evaluate and cannot combine with --max-windows")
    if args.origins and (len(set(args.origins)) != len(args.origins) or min(args.origins) < 0):
        raise ValueError("--origins must contain distinct nonnegative observed-row origins")
    if args.save_predictions and args.stage != "evaluate":
        raise ValueError("--save-predictions is evaluate-only")
    if args.stage == "evaluate" and args.output.resolve() == args.checkpoint_dir.resolve():
        raise ValueError("Evaluation output must differ from the source checkpoint directory")


def baseline_seed(report):
    configuration = report.get("configuration", {})
    return configuration.get("baseline_seed", configuration.get("seed", BASELINE_SEED))


def report_arm(report):
    return report.get("configuration", {}).get("arm", report.get("protocol", {}).get("ablation", "full"))


def check_baseline_record(report, dataset):
    configuration = report.get("configuration", {})
    if configuration.get("dataset") != dataset:
        raise ValueError("Frozen baseline dataset differs from --dataset")
    if baseline_seed(report) not in (2021, 2022, 2023) or configuration.get("smoke", False):
        raise ValueError("A completed full-data frozen baseline with a supported seed is required")
    baseline = report.get("stages", {}).get("baseline", {})
    if baseline.get("status") != "completed_training" or "test" not in baseline:
        raise ValueError("Baseline training and final test must be complete")
    return baseline


def same_model_config(left, right):
    ignored = {"root_path", "llm_path"}
    return ({k: v for k, v in left.items() if k not in ignored}
            == {k: v for k, v in right.items() if k not in ignored})


def assert_same_compact(expected, candidate, tensor_equal):
    """Exact trainable/buffer comparison; frozen construction uses seed/config."""
    for group in ("parameters", "buffers"):
        if group not in expected or group not in candidate or set(expected[group]) != set(candidate[group]):
            raise ValueError("Frozen baseline compact checkpoint tensors differ")
        if any(not tensor_equal(value, candidate[group][name]) for name, value in expected[group].items()):
            raise ValueError("Frozen baseline compact checkpoint tensors differ")


def compare_baseline_files(expected_path, candidate_path, d):
    expected = d.torch.load(expected_path, map_location="cpu", weights_only=False)
    candidate = d.torch.load(candidate_path, map_location="cpu", weights_only=False)
    assert_same_compact(expected, candidate, d.torch.equal)


def snapshot_sources(output, source_root=ROOT, names=SOURCES):
    destination = Path(output) / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source_root) / name, target)
    (destination / "manifest.json").write_text(json.dumps({"version": VERSION, "sources": list(names)}, indent=2), encoding="utf-8")


def check_source_snapshot(output, source_root=ROOT, names=SOURCES):
    destination = Path(output) / "source_snapshot"
    manifest = read_json(destination / "manifest.json")
    if manifest != {"version": VERSION, "sources": list(names)}:
        raise ValueError("Resume source manifest/version differs")
    for name in names:
        if (Path(source_root) / name).read_bytes() != (destination / name).read_bytes():
            raise ValueError("Scientific source changed since checkpoint: " + name)


def comparable_audit(audit):
    # Paths and mtimes change when users download/copy identical data.
    # No hash or full-dataset duplicate scan is needed for normal relocation.
    ignored = {"window_counts", "csv_mtime_ns", "adj_mtime_ns"}
    return {key: value for key, value in audit.items() if key not in ignored}


def check_audit(recorded, actual):
    if comparable_audit(recorded) != comparable_audit(actual):
        raise ValueError("Dataset content metadata/chronology differs from the checkpoint report")


def relocation_config(config, dataset, data_root, bert_path, specs):
    result = copy.deepcopy(config)
    spec = specs[dataset]
    result.update(root_path=str((data_root / spec["folder"]).resolve()),
                  adjacency_path=spec["adj"], llm_path=str(bert_path.resolve()))
    if result.get("node_num") != spec["nodes"] or result.get("llm_layers") != 12:
        raise ValueError("Checkpoint does not describe this release's corrected 12-layer baseline")
    return result


def restore_rng(state, torch, np):
    required = ("rng", "cuda_rng", "numpy_rng", "python_rng")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("Full last checkpoint missing RNG fields: " + ", ".join(missing))
    torch.set_rng_state(state["rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def check_resume_state(state, stage, completed=None):
    required = {"epoch", "parameters", "buffers", "optimizer", "log", "stale",
                "best_value", "best_epoch", "rng", "cuda_rng", "numpy_rng", "python_rng"}
    if stage == "baseline":
        required.add("scheduler")
    if required.difference(state):
        raise ValueError("Resume requires complete optimizer/RNG state, not only final weights")
    cap = 20 if stage == "baseline" else 5
    if not isinstance(state["epoch"], int) or not 1 <= state["epoch"] <= cap:
        raise ValueError("Resume checkpoint exceeds this release's total epoch budget")
    if len(state["log"]) != state["epoch"] or state["optimizer"] is None:
        raise ValueError("Incomplete resume log/optimizer state")
    if completed is not None and (state["epoch"] != completed["epochs"] or state["log"] != completed["training_log"]):
        raise ValueError("RNG checkpoint does not match the source report's completed training history")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--stage", choices=("baseline", "improved", "evaluate"), default="improved")
    p.add_argument("--seed", type=int, choices=(2021, 2022, 2023), default=2021)
    p.add_argument("--arm", choices=("full", "global_long", "no_utility"), default="full",
                   help="BIKE-Inflow controls; evaluate restores the arm recorded with the weights")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    p.add_argument("--bert-path", type=Path, default=ROOT / "huggingface/google_bert")
    p.add_argument("--baseline", type=Path, help="Completed fixed city baseline; can relocate released weights")
    p.add_argument("--numeric-calibration", type=Path,
                   help="Reuse this city's training-only fit (e.g. seed_2021/numeric_calibration.pt)")
    p.add_argument("--resume", action="store_true", help="Resume complete last checkpoint including optimizer/RNG")
    p.add_argument("--checkpoint-dir", type=Path, help="Original or portable completed run to evaluate")
    p.add_argument("--eval-model", choices=("baseline", "improved"), default="improved")
    p.add_argument("--max-windows", type=int, help="Evaluate only first N test windows; NOT official test metrics")
    p.add_argument("--replay-training-rng", action="store_true",
                   help="Requires *_last.pt: restore post-training RNG and replay final validation before test")
    p.add_argument("--rng-checkpoint", type=Path,
                   help="Optional explicit *_last.pt for full historical test-RNG replay; never needed for weights-only evaluation")
    p.add_argument("--origins", type=int, nargs="+", help="Explicit legal test origins for case export (observed-row index)")
    p.add_argument("--save-predictions", action="store_true", help="Export raw-unit prediction/ground-truth arrays; can be large")
    p.add_argument("--check-data", action="store_true", help="CPU-only data/config check; does not create output")
    return p


def dependencies():
    # Lazy imports keep --help/config tests independent of CUDA and local data.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from research import run_coursework as shared
    from research import run_coursework_extra as extra
    from research.common import seed_all, write_json
    from research.fast_execution import create_fast, create_hourly, training_function
    from research.coursework_data import CourseworkWindows, load_dataset, SPECS
    from models.sthsepnet_h1 import Model
    import numpy as np
    import torch
    return SimpleNamespace(shared=shared, extra=extra, seed_all=seed_all, write_json=write_json,
        create_fast=create_fast, create_hourly=create_hourly, training_function=training_function,
        Windows=CourseworkWindows, load_dataset=load_dataset, specs=SPECS, Model=Model,
        np=np, torch=torch)


def prepare_data(args, d):
    bundle, audit = d.load_dataset(args.data_root, args.dataset)
    short = {role: d.Windows(bundle, role, 48) for role in ("train", "val", "test")}
    long = {role: d.Windows(bundle, role, history_length(args.dataset)) for role in ("train", "val", "test")}
    if args.arm != "full":
        from research.bike_validation_support import BikeMemoryWindows
        long = {role: BikeMemoryWindows(bundle, role, args.arm) for role in ("train", "val", "test")}
    if not d.np.array_equal(short["test"].origins, long["test"].origins):
        raise ValueError("Baseline and improved test origins differ")
    return bundle, audit, short, long


def new_config(args, bundle, d):
    spec = d.specs[args.dataset]
    config = read_json(ROOT / "research/coursework_config.json")
    names = {"pems03": "PEMS03", "metr_la": "METR", "bj500": "BJ500",
             "bike_inflow": "BIKE-Inflow", "bike_outflow": "BIKE-Outflow"}
    config.update(node_num=spec["nodes"], enc_in=spec["nodes"], data=names[args.dataset],
                  d_ff=spec["d_ff"], min_values=bundle.train_min.tolist(), max_values=bundle.train_max.tolist())
    if args.dataset.startswith("bike_"):
        config.update(prompt_domain=1, content=f"{config['data']} bicycle station demand at295 New York stations, sampled hourly. "
            "Local timestamps are calendar/index features, not incident or holiday labels. "
            "The supplied station adjacency supplies spatial connectivity.")
    return relocation_config(config, args.dataset, args.data_root, args.bert_path, d.specs)


def build_base(args, config, d, checkpoint=None, seed=BASELINE_SEED):
    # Compact files omit frozen pretrained weights and some random frozen tensors.
    # Always reconstruct them with the BASELINE seed before loading the compact.
    d.seed_all(seed)
    base = d.Model(SimpleNamespace(**config)).cuda()
    if checkpoint is not None:
        load_complete_compact(base, d.torch.load(checkpoint, map_location="cpu", weights_only=False), d)
    return base


def load_complete_compact(model, state, d):
    """Fail on incomplete/mismatched weights instead of silently random-filling."""
    expected = {"parameters": {name: value for name, value in model.named_parameters() if value.requires_grad},
                "buffers": dict(model.named_buffers())}
    for group, tensors in expected.items():
        if set(state.get(group, {})) != set(tensors):
            raise ValueError(f"Incomplete or incompatible compact checkpoint {group}")
        if any(value.shape != state[group][name].shape for name, value in tensors.items()):
            raise ValueError(f"Incompatible compact checkpoint shapes: {group}")
    d.shared.load_compact(model, state)


def build_student(args, d):
    d.seed_all(args.seed)
    create = d.create_hourly if args.dataset.startswith("bike_") else d.create_fast
    return create(args.bert_path, repair="utility", lora_rank=16, hierarchical_memory=True,
        memory_policy="no_deviation", use_neighbors=False, node_chunk_size=256,
        gradient_checkpointing=False).cuda()


def args_for_training(args, bundle):
    return SimpleNamespace(dataset=args.dataset, seed=args.seed, batch_size=4, target_chunk=64,
                           raw_scale=float(__import__("numpy").median(bundle.std)), resume=args.resume)


def fresh_report(args, audit, config):
    return {"status": "preparing", "process_id": os.getpid(), "configuration": {
        "dataset": args.dataset, "seed": args.seed, "baseline_seed": args.seed if args.stage == "baseline" else BASELINE_SEED,
        "batch_size": 4, "target_chunk": 64, "smoke": False, "arm": args.arm, "bert_path": str(args.bert_path.resolve()),
        "protocol_version": VERSION}, "dataset_audit": audit, "base_model_config": config, "stages": {},
        "protocol": {"model_architecture": MODEL_VERSION, "baseline_input": 48,
            "improved_history": history_length(args.dataset), "prediction_horizon": 48,
            "split": "floor70% train, floor20% test, remaining validation; target-boundary split",
            "normalization": "per-node training segment only", "precision": "BF16", "all_nodes": True,
            "all_training_windows": True, "checkpoint_selection": "validation raw MSE only",
            "early_stop_patience": 5, "adapter_max_epochs": 5, "fresh_baseline_max_epochs": 20,
            "shared_frozen_baseline": args.stage != "baseline", "seed_scope": (
                f"baseline construction/training seed {args.seed}; independent paired-case baseline when seed is 2022/2023"
                if args.stage == "baseline" else "adapter initialization/training; frozen baseline seed recorded separately"),
            "extra_information_disclosed": True, "test_selection": False,
            "budget_note": "Epoch-cap termination is not confirmed convergence. Fresh baseline cap20 need not reproduce historical baselines' original budgets.",
            "ablation": args.arm, "utility_coefficient": 0 if args.arm == "no_utility" else .1,
            "metrics": "unmasked raw-unit MAE/global RMSE/MSE; all 48 horizons; no clipping"}}


def prepare_output(args, report, d):
    path = args.output / "report.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError("Output has a report; use a new directory or explicit --resume")
        previous = read_json(path)
        expected = report["configuration"]
        actual = previous.get("configuration", {})
        if {key: value for key, value in actual.items() if key != "bert_path"} != {
                key: value for key, value in expected.items() if key != "bert_path"}:
            raise ValueError("Resume configuration changed")
        check_audit(previous["dataset_audit"], report["dataset_audit"])
        if not same_model_config(previous["base_model_config"], report["base_model_config"]):
            raise ValueError("Resume scientific model configuration changed")
        prior_provenance = previous.get("baseline_provenance", {})
        provenance = report.get("baseline_provenance", {})
        if {key: value for key, value in prior_provenance.items() if key != "source_report"} != {
                key: value for key, value in provenance.items() if key != "source_report"}:
            raise ValueError("Frozen baseline source changed during resume")
        check_source_snapshot(args.output)
        if args.stage == "improved":
            compare_baseline_files(checkpoint_path(args.output, "baseline"), checkpoint_path(args.baseline, "baseline"), d)
        previous["process_id"] = os.getpid()
        return previous
    if args.resume:
        raise FileNotFoundError("--resume requires an existing portable report")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_sources(args.output)
    report["source_snapshot"] = "source_snapshot/manifest.json"
    d.write_json(path, report)
    return report


def train(args, bundle, audit, short, long, d):
    config = new_config(args, bundle, d)
    baseline_record = None
    if args.stage == "improved":
        baseline_record = read_json(args.baseline / "report.json")
        baseline_stage = check_baseline_record(baseline_record, args.dataset)
        check_audit(baseline_record["dataset_audit"], audit)
        config = relocation_config(baseline_record["base_model_config"], args.dataset, args.data_root, args.bert_path, d.specs)
        checkpoint_path(args.baseline, "baseline")
    report = fresh_report(args, audit, config)
    if baseline_record is not None:
        report["stages"]["baseline"] = copy.deepcopy(baseline_stage)
        report["configuration"]["baseline_seed"] = baseline_seed(baseline_record)
        report["protocol"]["seed_scope"] = f"adapter seed {args.seed}; frozen city backbone seed {baseline_seed(baseline_record)}"
        report["baseline_provenance"] = {"source_report": str((args.baseline / "report.json").resolve()),
                                         "seed": baseline_seed(baseline_record), "shared_frozen_baseline": True}
    report = prepare_output(args, report, d)
    if args.stage in report["stages"]:
        print(json.dumps({"status": "already_completed", "stage": args.stage}))
        return
    if args.resume and not (args.output / f"{args.stage}_last.pt").is_file():
        raise FileNotFoundError("Resume requires a complete *_last.pt with optimizer and all RNG states")
    if args.resume:
        check_resume_state(d.torch.load(args.output / f"{args.stage}_last.pt", map_location="cpu", weights_only=False), args.stage)
    run_args = args_for_training(args, bundle)
    base_path = checkpoint_path(args.baseline, "baseline") if baseline_record is not None else None
    actual_base_seed = baseline_seed(baseline_record) if baseline_record is not None else args.seed
    base = build_base(args, config, d, base_path, actual_base_seed)
    if base_path is not None and not (args.output / "baseline_best.pt").exists():
        shutil.copy2(base_path, args.output / "baseline_best.pt")
    student = None
    datasets = short
    if args.stage == "improved":
        base.eval().requires_grad_(False)
        student = build_student(args, d)
        datasets = long
    with d.extra.sampling_helpers(args.dataset):
        if student is not None:
            numeric_path = args.output / "numeric_calibration.pt"
            if not numeric_path.exists():
                if args.numeric_calibration is not None:
                    source_report = read_json(args.numeric_calibration.parent / "report.json")
                    source_arm = report_arm(source_report)
                    if (args.arm == "global_long") != (source_arm == "global_long"):
                        raise ValueError("Global-long and node-specific history require separate numeric fits")
                    check_baseline_record(source_report, args.dataset)
                    check_audit(source_report["dataset_audit"], audit)
                    if not same_model_config(source_report["base_model_config"], baseline_record["base_model_config"]):
                        raise ValueError("Numeric fit belongs to a different baseline configuration")
                    if baseline_seed(source_report) != actual_base_seed:
                        raise ValueError("Numeric fit belongs to a different baseline seed")
                    # Exact compact-state comparison permits relocation and avoids
                    # trusting an old machine's absolute provenance path.
                    candidate_base = checkpoint_path(args.numeric_calibration.parent, "baseline")
                    compare_baseline_files(base_path, candidate_base, d)
                    shutil.copy2(args.numeric_calibration, numeric_path)
                else:
                    report["status"] = "fitting_train_only_numeric_route"
                    d.write_json(args.output / "report.json", report)
                    state, details = d.shared.fit_numeric(base, long, run_args)
                    d.torch.save({"state": state, "report": details}, numeric_path)
            numeric = d.torch.load(numeric_path, map_location="cpu", weights_only=False)
            student.numeric_calibration.install(numeric["state"])
            report["numeric_fit"] = numeric["report"]
        report["status"] = "training_" + args.stage
        d.write_json(args.output / "report.json", report)
        if student is None:
            train_function = d.shared.train_stage
        else:
            utility = d.shared.utility_training_loss
            if args.arm == "no_utility":
                from research.bike_validation_support import zero_utility_loss
                utility = zero_utility_loss
            # Exactly the canonical ablation's execution-only transform; resolve
            # the hourly teacher AFTER sampling_helpers injected it.
            train_function = d.training_function(d.shared.train_stage, utility)
        row = train_function(base, student, datasets, run_args, report, args.output,
                             20 if student is None else 5)
        row["test"] = d.shared.evaluate(base, student, datasets["test"], run_args,
                                        args.output / f"{args.stage}_test.npz")
    if student is not None:
        reference = report["stages"]["baseline"]["test"]
        row["improvement_percent"] = {key: 100 * (1 - row["test"][key] / reference[key])
                                       for key in ("mae", "rmse", "mse")}
    report["stages"][args.stage] = row
    report["status"] = "baseline_completed" if student is None else "completed_shared_baseline_adapter_seed"
    report.pop("active", None)
    d.write_json(args.output / "report.json", report)


def evaluate_with_predictions(base, student, dataset, args, output, bundle, d):
    """Same scoring as shared.evaluate, plus untouched raw-unit curve exports."""
    import time
    from research.common import loader, move
    np, torch = d.np, d.torch
    base.eval()
    if student is not None:
        student.eval()
    absolute, squared, raw_predictions, raw_truth, origins = [], [], [], [], []
    horizon_abs, horizon_sq, points = np.zeros(48), np.zeros(48), 0
    tick = time.perf_counter()
    with torch.no_grad():
        for cpu in loader(dataset, args.batch_size, False, args.seed):
            batch = move(cpu, "cuda")
            target = batch.pop("target")  # Never pass current future values to a prediction interface.
            count = target.shape[-1]
            if student is None:
                prediction = d.shared.base_prediction(base, batch).float()
            else:
                baseline, info = d.shared.base_prediction(base, batch, True)
                pieces = []
                for start in range(0, count, args.target_chunk):
                    stop = min(count, start + args.target_chunk)
                    current = {"history": batch["history"][:, :, start:stop],
                               "short_state": info["short_state"][:, start:stop].float(),
                               "base": baseline[:, :, start:stop].float(), "origin_step": batch["origin_step"]}
                    pieces.append(current["base"] + d.shared.adapter_delta(student, current).float())
                prediction = torch.cat(pieces, -1)
            error = (prediction - target) * batch["node_std"][:, None]
            if not torch.isfinite(error).all():
                raise RuntimeError("Nonfinite evaluation")
            absolute.extend(error.abs().mean((1, 2)).cpu().tolist())
            squared.extend(error.square().mean((1, 2)).cpu().tolist())
            horizon_abs += error.abs().sum((0, 2)).cpu().numpy()
            horizon_sq += error.square().sum((0, 2)).cpu().numpy()
            points += error.shape[0] * count
            current_origins = cpu["origin_step"].numpy()
            origins.extend(current_origins.tolist())
            raw_predictions.append(prediction.cpu().numpy() * bundle.std[None, None] + bundle.mean[None, None])
            raw_truth.append(np.stack([bundle.raw[int(origin):int(origin) + 48] for origin in current_origins]))
    mse = float(np.mean(squared))
    result = {"mae": float(np.mean(absolute)), "mse": mse, "rmse": mse ** .5,
              "windows": len(dataset), "horizon_mae": (horizon_abs / points).tolist(),
              "horizon_rmse": np.sqrt(horizon_sq / points).tolist(), "seconds": time.perf_counter() - tick,
              "all_nodes": True, "mask": "none, matches unmasked author metrics"}
    np.savez(output / f"{args.eval_model}_test.npz", origin=origins, window_mae=absolute, window_mse=squared)
    origin_array = np.asarray(origins, dtype=np.int64)
    np.savez_compressed(output / f"{args.eval_model}_predictions.npz", origin=origin_array,
        prediction=np.concatenate(raw_predictions), ground_truth=np.concatenate(raw_truth),
        timestamp_ns=np.stack([bundle.timestamps_ns[origin:origin + 48] for origin in origin_array]),
        mean=bundle.mean, std=bundle.std, node_names=np.asarray(bundle.node_names),
        baseline_seed=np.asarray(args.baseline_seed), adapter_seed=np.asarray(args.seed),
        units=np.asarray("original CSV flow units; no clipping/smoothing/rounding"))
    return result


def evaluate(args, bundle, audit, short, long, d):
    source = read_json(args.checkpoint_dir / "report.json")
    if source.get("configuration", {}).get("dataset") != args.dataset:
        raise ValueError("Evaluation dataset differs from checkpoint report")
    check_audit(source["dataset_audit"], audit)
    is_improved = args.eval_model == "improved"
    args.seed = source["configuration"]["seed"]
    args.baseline_seed = baseline_seed(source)
    if is_improved:
        if source.get("protocol", {}).get("model_architecture", MODEL_VERSION) != MODEL_VERSION:
            raise ValueError("Not a fast H1 model checkpoint")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Evaluation writes to a new directory; it never overwrites source results")
    config = relocation_config(source["base_model_config"], args.dataset, args.data_root, args.bert_path, d.specs)
    baseline_dir = args.baseline or args.checkpoint_dir
    # Historical formal adapter outputs contain a baseline_best.pt copy/link.
    if args.baseline is not None:
        supplied = read_json(args.baseline / "report.json")
        check_baseline_record(supplied, args.dataset)
        check_audit(supplied["dataset_audit"], audit)
        if baseline_seed(supplied) != args.baseline_seed or not same_model_config(supplied["base_model_config"], config):
            raise ValueError("Supplied baseline configuration/seed differs from adapter's frozen baseline")
        # Reports alone cannot distinguish two trainings with the same seed and
        # hyperparameters. Require the adapter's preserved compact baseline copy.
        compare_baseline_files(checkpoint_path(args.checkpoint_dir, "baseline"),
                               checkpoint_path(args.baseline, "baseline"), d)
    base = build_base(args, config, d, checkpoint_path(baseline_dir, "baseline"), args.baseline_seed)
    base.eval().requires_grad_(False)
    student = None
    datasets = long if is_improved else short
    run_args = args_for_training(args, bundle)
    if is_improved:
        student = build_student(args, d)
        numeric = d.torch.load(args.checkpoint_dir / "numeric_calibration.pt", map_location="cpu", weights_only=False)
        student.numeric_calibration.install(numeric["state"])
        load_complete_compact(student, d.torch.load(checkpoint_path(args.checkpoint_dir, "improved"), map_location="cpu", weights_only=False), d)
    if args.replay_training_rng or args.rng_checkpoint:
        last = args.rng_checkpoint or args.checkpoint_dir / f"{args.eval_model}_last.pt"
        state = d.torch.load(last, map_location="cpu", weights_only=False)
        check_resume_state(state, args.eval_model, source.get("stages", {}).get(args.eval_model))
        restore_rng(state, d.torch, d.np)
        with d.extra.sampling_helpers(args.dataset):
            # Frozen train_stage evaluates the selected best once after saving last,
            # then the wrapper evaluates test. Replay that exact RNG consumption.
            d.shared.evaluate(base, student, datasets["val"], run_args)
        rng_note = "Restored last checkpoint post-training RNG; replayed selected-best validation before test. Hardware/version equality is not guaranteed."
    else:
        d.seed_all(args.seed)
        rng_note = "Explicit seeded evaluation; original GCN draws top-k tie noise even in eval. Without --replay-training-rng this is not bitwise replay of historical test."
    if args.max_windows is not None:
        datasets["test"].origins = datasets["test"].origins[:args.max_windows]
    elif args.origins:
        legal = set(datasets["test"].origins.tolist())
        if any(origin not in legal for origin in args.origins):
            raise ValueError("Requested origin is outside the legal test target split/history")
        expected_interval = int(86400e9 / d.specs[args.dataset]["period"])
        if any(d.np.any(d.np.diff(bundle.timestamps_ns[origin:origin + 48]) != expected_interval) for origin in args.origins):
            raise ValueError("Case's future 48 timestamps must be continuous")
        datasets["test"].origins = d.np.asarray(args.origins, dtype=d.np.int64)
    args.output.mkdir(parents=True, exist_ok=True)
    with d.extra.sampling_helpers(args.dataset):
        if args.save_predictions:
            run_args.eval_model, run_args.baseline_seed = args.eval_model, args.baseline_seed
            metrics = evaluate_with_predictions(base, student, datasets["test"], run_args, args.output, bundle, d)
        else:
            metrics = d.shared.evaluate(base, student, datasets["test"], run_args,
                                        args.output / f"{args.eval_model}_test.npz")
    status = "smoke_only_not_official_metrics" if args.max_windows else ("completed_selected_cases_not_full_test" if args.origins else "completed_evaluation")
    d.write_json(args.output / "evaluation.json", {"status": status,
        "dataset": args.dataset, "seed": args.seed, "baseline_seed": args.baseline_seed, "arm": args.arm,
        "model": args.eval_model, "metrics": metrics,
        "source_report": str((args.checkpoint_dir / "report.json").resolve()), "evaluation_rng_policy": rng_note,
        "prediction_input": "historical observations only; targets used only after forward for scoring",
        "source_results_modified": False})
    print(json.dumps(metrics))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == "evaluate" and args.checkpoint_dir is not None:
        recorded_arm = report_arm(read_json(args.checkpoint_dir / "report.json"))
        if args.arm != "full" and args.arm != recorded_arm:
            raise ValueError("Requested arm differs from checkpoint report")
        args.arm = recorded_arm
    validate_arguments(args)
    args.output = args.output.resolve()
    d = dependencies()
    bundle, audit, short, long = prepare_data(args, d)
    if args.check_data:
        print(json.dumps({"dataset": args.dataset, "audit": audit, "base_model_config": new_config(args, bundle, d),
            "history": history_length(args.dataset), "window_counts": {
                stage: {role: len(ds) for role, ds in values.items()}
                for stage, values in (("baseline", short), ("improved", long))}, "cuda_used": False}))
        return
    if not d.torch.cuda.is_available():
        raise RuntimeError("Training and evaluation use the released CUDA/BF16 protocol; CUDA is required")
    if not d.torch.cuda.is_bf16_supported():
        raise RuntimeError("This release requires a GPU with BF16 support")
    d.torch.set_num_threads(4)
    if args.stage == "evaluate":
        evaluate(args, bundle, audit, short, long, d)
    else:
        train(args, bundle, audit, short, long, d)


if __name__ == "__main__":
    main()
