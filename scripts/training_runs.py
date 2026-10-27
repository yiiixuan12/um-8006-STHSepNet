"""Sequential training queues for the documented baseline, H1 and KAN entries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('bike_inflow', 'bike_outflow', 'pems03', 'bj500', 'metr_la')


def invoke(arguments, plan):
    if plan:
        print(json.dumps({'python': sys.executable, 'working_directory': str(ROOT),
                          'arguments': arguments}), flush=True)
    else:
        print(subprocess.list2cmdline([sys.executable, '-u', *arguments]), flush=True)
        subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT, check=True)


def run_stage(arguments, output, stage, dataset, seed, plan, official=False):
    if not plan and output.exists():
        path = output / ('report_0.json' if official else 'report.json')
        if not path.exists():
            raise RuntimeError(f'Existing output has no report; inspect before retrying: {output}')
        report = json.loads(path.read_text(encoding='utf-8'))
        config = report['configuration']
        recorded_dataset = config.get('data') if official else config.get('dataset')
        if recorded_dataset != dataset or config['seed'] != seed or config.get('smoke'):
            raise RuntimeError(f'Output configuration mismatch: {output}')
        if official:
            if report['status'] == 'completed' and bool(config.get('use_kan')) == ('--use_kan' in arguments):
                print(f'SKIP completed {output}', flush=True)
                return
            raise RuntimeError(f'Official-entry run cannot resume automatically: {output}; use a new output root')
        row = report.get('stages', {}).get(stage, {})
        if row.get('status') == 'completed_training' and 'test' in row and (output / f'{stage}_test.npz').is_file():
            print(f'SKIP completed {output}', flush=True)
            return
        if not (output / f'{stage}_last.pt').is_file():
            raise RuntimeError(f'Incomplete run has no checkpoint: {output}')
        arguments += ['--resume']
    invoke(arguments, plan)
    if not plan:
        path = output / ('report_0.json' if official else 'report.json')
        report = json.loads(path.read_text(encoding='utf-8'))
        if official:
            if report['status'] != 'completed':
                raise RuntimeError(f'Run did not complete: {output}')
        elif 'test' not in report.get('stages', {}).get(stage, {}):
            raise RuntimeError(f'Missing final stage metrics: {output}')


def kan_arguments(seed, output, head, smoke=False):
    arguments = ['src/run_main.py', '--task_name', 'long_term_forecast', '--is_training', '1',
        '--root_path', str(ROOT / 'dataset/Bike'), '--data_path', 'inflow.csv',
        '--model_id', 'BIKE_48_48', '--model', 'pool', '--data', 'inflow', '--features', 'M',
        '--seq_len', '48', '--label_len', '48', '--pred_len', '48', '--factor', '3',
        '--des', 'Exp', '--itr', '1', '--d_model', '768', '--d_ff', '32', '--batch_size', '16',
        '--learning_rate', '0.0001', '--llm_model', 'BERT', '--llm_dim', '768', '--llm_layers', '6',
        '--train_epochs', '50', '--model_comment', 'abl', '--gamma', '1.0', '--fusion_gate', 'adaptive',
        '--enc_in', '295', '--dec_in', '295', '--c_out', '295', '--node_num', '295',
        '--scale_hyperedges', '3', '--seed', str(seed), '--num_workers', '0', '--output-dir', str(output)]
    if head == 'kan':
        arguments += ['--use_kan', '--kan_grid_size', '5', '--kan_spline_order', '3']
    if smoke:
        arguments += ['--smoke']
    return arguments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('baseline', 'h1', 'kan', 'smoke'))
    p.add_argument('--datasets', nargs='+', choices=DATASETS)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--heads', choices=('both', 'kan', 'mlp'), default='both')
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    if args.mode == 'kan' and args.datasets:
        p.error('KAN is the BIKE-Inflow study; --datasets is not applicable')
    datasets = args.datasets or (['bike_outflow'] if args.mode == 'smoke' else list(DATASETS))
    seeds = args.seeds or ([2021, 42, 7] if args.mode == 'kan' else [2021, 2022, 2023])
    if args.mode == 'h1' and any(seed not in (2021, 2022, 2023) for seed in seeds):
        p.error('H1 supports seeds 2021, 2022, 2023')
    default = {'baseline': 'results_coursework/run_jinqi_baseline', 'h1': 'runs/h1_queue',
               'kan': 'outputs/kan_queue', 'smoke': 'outputs/training_smoke/' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')}
    output_root = args.output_root or Path(default[args.mode])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root = output_root.resolve()
    if args.mode == 'smoke':
        if output_root.exists() and not args.plan_only:
            raise FileExistsError(f'Choose a new smoke root: {output_root}')
        for dataset in datasets:
            invoke(['research/run_coursework.py', '--dataset', dataset, '--seed', '2021', '--stage', 'baseline',
                    '--smoke', '--output', str(output_root / 'baseline' / dataset)], args.plan_only)
            invoke(['scripts/smoke_h1_training.py', '--dataset', dataset,
                    '--output', str(output_root / 'h1' / dataset)], args.plan_only)
        for head in ('mlp', 'kan'):
            invoke(kan_arguments(2021, output_root / head, head, True), args.plan_only)
        print('Smoke queue complete; these metrics are not accuracy evidence.', file=sys.stderr)
        return
    if args.mode == 'kan':
        for seed in seeds:
            for head in ('mlp', 'kan') if args.heads == 'both' else (args.heads,):
                output = output_root / head / f'seed_{seed}'
                run_stage(kan_arguments(seed, output, head), output, 'baseline', 'inflow', seed,
                          args.plan_only, official=True)
        return
    for dataset in datasets:
        if args.mode == 'h1':
            baseline = output_root / dataset / 'baseline'
            arguments = ['research/run_h1.py', '--dataset', dataset, '--stage', 'baseline', '--seed', '2021',
                         '--output', str(baseline)]
            run_stage(arguments, baseline, 'baseline', dataset, 2021, args.plan_only)
        for seed in seeds:
            output = output_root / dataset / f'seed_{seed}'
            if args.mode == 'baseline':
                arguments = ['research/run_coursework.py', '--dataset', dataset, '--stage', 'baseline',
                             '--seed', str(seed), '--output', str(output)]
                stage = 'baseline'
            else:
                arguments = ['research/run_h1.py', '--dataset', dataset, '--stage', 'improved',
                             '--seed', str(seed), '--baseline', str(baseline), '--output', str(output)]
                stage = 'improved'
            run_stage(arguments, output, stage, dataset, seed, args.plan_only)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
