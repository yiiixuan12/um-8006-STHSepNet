"""Export/check Jinqi's completed baseline evidence without training or GPU.

Use --check to verify the published package without the source experiment tree.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
from pathlib import Path

from check_h1_results import check_record

DATASETS = {'bike_inflow': (5.18, 14.40), 'bike_outflow': (5.33, 14.23),
            'pems03': (21.03, 34.17), 'bj500': (5.58, 9.77), 'metr_la': (9.42, 16.41)}
SEEDS = (2021, 2022, 2023)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if k != 'process_id' and not k.endswith('_mtime_ns')}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str) and re.match(r'^[A-Za-z]:[\\/]', value):
        return 'source://local/' + value.replace('\\', '/').rsplit('/', 1)[-1]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def collect(root):
    rows, aggregates = [], []
    for dataset, paper in DATASETS.items():
        origins = None
        group = []
        for seed in SEEDS:
            report = root / 'reports' / dataset / f'seed_{seed}.json'
            data = json.loads(report.read_text(encoding='utf-8'))
            config = data['configuration']
            if config['dataset'] != dataset or config['seed'] != seed or config.get('smoke'):
                raise ValueError(f'Invalid configuration: {report}')
            if data['status'] != 'baseline_completed':
                raise ValueError(f'Incomplete baseline: {report}')
            origins, record = check_record(root / 'errors' / dataset / f'seed_{seed}_test.npz',
                                           report, 'baseline', origins)
            stage = data['stages']['baseline']
            row = dict(dataset=dataset, seed=seed, method='baseline',
                       **{k: record[k] for k in ('mae', 'rmse', 'mse', 'windows')},
                       epochs=stage['epochs'], best_epoch=stage['best_epoch'],
                       stop_reason=stage['stop_reason'])
            rows.append(row)
            group.append(row)
        item = dict(dataset=dataset, seeds=list(SEEDS))
        for metric in ('mae', 'rmse', 'mse'):
            item[metric + '_mean'] = statistics.mean(x[metric] for x in group)
            item[metric + '_std'] = statistics.stdev(x[metric] for x in group)
        for metric, reference in zip(('mae', 'rmse'), paper):
            item['paper_' + metric] = reference
            item[metric + '_delta_percent'] = (item[metric + '_mean'] / reference - 1) * 100
        aggregates.append(item)
    return dict(status='completed', stage='baseline', runs=15, per_seed=rows, aggregate=aggregates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results_coursework/run_jinqi_baseline'))
    parser.add_argument('--output', type=Path, default=Path('results/baseline_jinqi'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    out = args.output
    if args.check:
        actual = collect(out)
        if actual != json.loads((out / 'summary.json').read_text(encoding='utf-8')):
            raise AssertionError('Summary differs from recomputed results')
        with (out / 'per_seed.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        if rows != [{k: str(v) for k, v in row.items()} for row in actual['per_seed']]:
            raise AssertionError('Per-seed CSV differs from reports')
        manifest = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
        for entry in manifest['files']:
            if digest(out / entry['published']) != entry['published_sha256']:
                raise AssertionError('Published artifact hash mismatch')
        print('PASS: 15 runs, 45 exact metrics, identical test origins across seeds, sample SD, CSV and hashes.')
        return
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}; use --check or a new --output')
    # Validate every original run before creating the export directory.
    for dataset in DATASETS:
        for seed in SEEDS:
            folder = args.source / dataset / f'seed_{seed}'
            check_record(folder / 'baseline_test.npz', folder / 'report.json', 'baseline')
    files = []
    for dataset in DATASETS:
        for seed in SEEDS:
            folder = args.source / dataset / f'seed_{seed}'
            for name, relative in [('report.json', f'reports/{dataset}/seed_{seed}.json'),
                                   ('baseline_test.npz', f'errors/{dataset}/seed_{seed}_test.npz')]:
                source, target = folder / name, out / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith('.json'):
                    write_json(target, clean(json.loads(source.read_text(encoding='utf-8'))))
                else:
                    shutil.copyfile(source, target)
                files.append(dict(source=f'source://run_jinqi_baseline/{dataset}/seed_{seed}/{name}',
                                  source_sha256=digest(source), published=relative,
                                  published_sha256=digest(target)))
    summary = collect(out)
    write_json(out / 'summary.json', summary)
    with (out / 'per_seed.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary['per_seed'][0]))
        writer.writeheader()
        writer.writerows(summary['per_seed'])
    lines = ['# Baseline reproduction — Jinqi Liu', '',
             'Three independent end-to-end seeds (2021, 2022, 2023); mean ± sample standard deviation.', '',
             '| Dataset | MAE | Paper MAE | Δ MAE | RMSE | Paper RMSE | Δ RMSE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for x in summary['aggregate']:
        lines.append(f"| {x['dataset']} | {x['mae_mean']:.4f} ± {x['mae_std']:.4f} | {x['paper_mae']:.2f} | {x['mae_delta_percent']:+.2f}% | {x['rmse_mean']:.4f} ± {x['rmse_std']:.4f} | {x['paper_rmse']:.2f} | {x['rmse_delta_percent']:+.2f}% |")
    lines += ['', 'Paper reference: https://arxiv.org/html/2505.19620v1#S4.T1',
              'Signed relative changes use unrounded means. Positive means higher error.']
    (out / 'table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out / 'provenance.json', dict(
        note='Source hashes identify original files, not training Git commits. No historical commit is inferred. '
             'Absolute machine paths replaced; process IDs and file mtimes removed. Numeric scientific data retained. '
             'source:// references are provenance identifiers, not download URLs.', files=files))
    print(f'Exported 15 completed baseline runs to {out}')


if __name__ == '__main__':
    main()
