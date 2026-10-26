"""Finish the five-dataset, three-seed baseline queue; use --dry-run to preview."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('bike_outflow', 'pems03', 'bike_inflow', 'metr_la', 'bj500')
SEEDS = (2021, 2022, 2023)


def source_files():
    tree = ast.parse((ROOT / 'research/run_coursework.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'SOURCES' for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError('Cannot find the training entry source manifest')


def read_report(destination, dataset, seed):
    report = json.loads((destination / 'report.json').read_text(encoding='utf-8'))
    config = report['configuration']
    if (config.get('dataset') != dataset or config.get('seed') != seed
            or config.get('smoke') is not False):
        raise ValueError(f'Not a matching formal experiment: {destination}')
    return report


def completed(report):
    stage = report.get('stages', {}).get('baseline', {})
    metrics = stage.get('test', {})
    return (stage.get('status') == 'completed_training'
            and all(metrics.get(key) is not None for key in ('mae', 'rmse', 'mse', 'windows')))


def plan_queue(results):
    sources = source_files()
    plan = []
    for dataset in DATASETS:
        for seed in SEEDS:
            destination = results / dataset / f'seed_{seed}'
            command = [sys.executable, '-u', str(ROOT / 'research/run_coursework.py'),
                       '--dataset', dataset, '--seed', str(seed), '--stage', 'baseline',
                       '--output', str(destination)]
            action = 'NEW'
            if destination.exists():
                report = read_report(destination, dataset, seed)
                if completed(report):
                    plan.append(('SKIP', dataset, seed, destination, command))
                    continue
                config = report['configuration']
                if (config.get('protocol_version') != 'coursework_two_v1_patience5'
                        or config.get('early_stop_patience') != 5):
                    raise ValueError(f'Resume protocol mismatch: {destination}')
                for filename in ('baseline_last.pt', 'baseline_best.pt'):
                    checkpoint = destination / filename
                    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                        raise ValueError(f'Cannot resume without checkpoint: {checkpoint}')
                for name in sources:
                    snapshot = destination / 'source_snapshot' / name
                    if not snapshot.is_file() or snapshot.read_bytes() != (ROOT / name).read_bytes():
                        raise ValueError(f'Resume source mismatch: {destination}: {name}')
                command.extend(['--resume', '--batch-size', str(config['batch_size']),
                                '--target-chunk', str(config['target_chunk']),
                                '--bert-path', config['bert_path']])
                action = 'RESUME'
            plan.append((action, dataset, seed, destination, command))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=ROOT / 'results_coursework/run_jinqi_baseline')
    parser.add_argument('--dry-run', action='store_true', help='Check and print the queue without training')
    args = parser.parse_args()
    results = args.results.resolve()
    # Validate the whole queue before launching any expensive training.
    plan = plan_queue(results)
    for action, dataset, seed, _, _ in plan:
        print(f'{action:6} {dataset} seed={seed}', flush=True)
    if args.dry_run:
        print('Preview only; no training or experiment records were created.', flush=True)
        return
    for action, dataset, seed, destination, command in plan:
        if action == 'SKIP':
            continue
        print(f'\n{action} {dataset} seed={seed}', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if not completed(read_report(destination, dataset, seed)):
            raise RuntimeError(f'Missing final baseline test metrics: {destination}')
    print('\nAll 15 baseline results are complete.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nQueue interrupted. Rerun this script to continue.', file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Queue stopped: {exc}', file=sys.stderr)
        sys.exit(1)
