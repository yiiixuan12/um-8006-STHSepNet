"""Baseline data, evaluation, training orchestration and publication regressions.

CPU fixtures only: no original CSVs, BERT, downloads or GPU required.
"""
import contextlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from research import coursework_data as data
from research import run_coursework as runner
from research import traffic_data
from research.common import loader

ROOT = Path(__file__).resolve().parents[1]


def fixture(root, name='fixture', nodes=3, period=24, rows=601):
    spec = dict(folder=name, csv='values.csv', adj='adj.csv', nodes=nodes,
                period=period, epochs=30, lr=3e-4, d_ff=32, effective_batch=16)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    values = np.arange(rows * nodes, dtype=np.float32).reshape(rows, nodes) % 37
    frame = pd.DataFrame(values, columns=[f'node_{n}' for n in range(nodes)])
    dates = pd.date_range('2023-01-01', periods=rows, freq=pd.Timedelta(days=1) / period)
    frame.insert(0, 'date', dates)
    frame.to_csv(directory / spec['csv'], index=False)
    pd.DataFrame(np.eye(nodes)).to_csv(directory / spec['adj'], index=False)
    return spec, frame


class BaselineDataTests(unittest.TestCase):
    def test_all_five_dataset_shapes_sampling_and_author_split_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, original in data.SPECS.items():
                with self.subTest(dataset=name):
                    spec, _ = fixture(root, name, original['nodes'], original['period'])
                    with patch.dict(data.SPECS, {name: spec}):
                        bundle, audit = data.load_dataset(root, name)
                    self.assertEqual(bundle.raw.shape, (601, original['nodes']))
                    self.assertEqual(bundle.split_bounds, {'train': (0, 420), 'val': (420, 481), 'test': (481, 601)})
                    self.assertEqual(audit['interval_minutes'], 60 if name.startswith('bike_') else 5)
                    self.assertEqual(audit['timestamp_gap_count'], 0)

    def test_explicit_nanoseconds_for_second_microsecond_and_nanosecond_series(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, frame = fixture(Path(directory))
            path = Path(directory) / spec['folder'] / spec['csv']
            for unit in ('s', 'us', 'ns'):
                dates = pd.Series(frame['date'].to_numpy().astype(f'datetime64[{unit}]'))
                with self.subTest(unit=unit), patch.object(traffic_data.pd, 'to_datetime', return_value=dates):
                    bundle = traffic_data.load_pems03(path, .7, .1)
                    np.testing.assert_array_equal(np.diff(bundle.timestamps_ns), np.full(600, 3600_000_000_000))
                    self.assertEqual(bundle.timestamps_ns[0], pd.Timestamp('2023-01-01').value)

    def test_validation_and_test_values_do_not_affect_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, frame = fixture(Path(directory))
            path = Path(directory) / spec['folder'] / spec['csv']
            frame.iloc[:, -1] = 7.0  # Constant training node must remain finite.
            frame.to_csv(path, index=False)
            before = traffic_data.load_pems03(path, .7, .1)
            expected = frame.iloc[:420, 1:].to_numpy(dtype=np.float32)
            np.testing.assert_allclose(before.mean, expected.mean(0, dtype=np.float64), rtol=1e-6)
            self.assertEqual(before.std[-1], 1.0)
            frame.iloc[420:, 1:] += 10000
            frame.to_csv(path, index=False)
            after = traffic_data.load_pems03(path, .7, .1)
            np.testing.assert_array_equal(before.mean, after.mean)
            np.testing.assert_array_equal(before.std, after.std)
            np.testing.assert_array_equal(before.normalized[:420], after.normalized[:420])
            self.assertTrue(np.isfinite(after.normalized).all())

    def test_gaps_are_preserved_without_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, frame = fixture(root)
            frame = frame.drop(index=300)
            frame.to_csv(root / spec['folder'] / spec['csv'], index=False)
            with patch.dict(data.SPECS, {'fixture': spec}):
                bundle, audit = data.load_dataset(root, 'fixture')
            np.testing.assert_array_equal(bundle.raw, frame.iloc[:, 1:].to_numpy(dtype=np.float32))
            self.assertEqual(audit['rows'], 600)
            self.assertEqual(audit['timestamp_gap_count'], 1)

    def test_duplicate_reversed_and_wrong_interval_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, frame = fixture(root)
            for kind in ('duplicate', 'reversed', 'wrong_interval'):
                bad = frame.copy()
                if kind == 'duplicate':
                    bad.loc[100, 'date'] = bad.loc[99, 'date']
                elif kind == 'reversed':
                    bad['date'] = bad['date'].iloc[::-1].to_numpy()
                else:
                    bad['date'] = pd.date_range('2023-01-01', periods=len(bad), freq='30min')
                bad.to_csv(root / spec['folder'] / spec['csv'], index=False)
                with self.subTest(kind=kind), patch.dict(data.SPECS, {'fixture': spec}):
                    with self.assertRaisesRegex(ValueError, 'timestamps'):
                        data.load_dataset(root, 'fixture')

    def test_nonfinite_values_wrong_node_counts_and_bad_adjacency_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, frame = fixture(root)
            path = root / spec['folder'] / spec['csv']
            for value in (np.nan, np.inf):
                bad = frame.copy()
                bad.iloc[100, 1] = value
                bad.to_csv(path, index=False)
                with self.subTest(value=value), patch.dict(data.SPECS, {'fixture': spec}):
                    with self.assertRaisesRegex(ValueError, 'non-finite'):
                        data.load_dataset(root, 'fixture')
            frame.to_csv(path, index=False)
            with patch.dict(data.SPECS, {'fixture': {**spec, 'nodes': 4}}):
                with self.assertRaisesRegex(ValueError, 'node count'):
                    data.load_dataset(root, 'fixture')
            for adjacency in (np.eye(2), np.full((3, 3), np.nan)):
                pd.DataFrame(adjacency).to_csv(root / spec['folder'] / spec['adj'], index=False)
                with patch.dict(data.SPECS, {'fixture': spec}):
                    with self.assertRaisesRegex(ValueError, 'adjacency'):
                        data.load_dataset(root, 'fixture')

    def test_baseline_48_step_windows_never_cross_target_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = fixture(root)
            with patch.dict(data.SPECS, {'fixture': spec}):
                bundle, _ = data.load_dataset(root, 'fixture')
            for role, (low, high) in bundle.split_bounds.items():
                windows = data.CourseworkWindows(bundle, role, 48)
                self.assertGreaterEqual(windows.origins.min(), max(48, low))
                self.assertEqual(windows.origins[-1] + 48, high)
                first = windows[0]
                origin = int(first['origin_step'])
                np.testing.assert_array_equal(first['history'], bundle.normalized[origin-48:origin])
                np.testing.assert_array_equal(first['target'], bundle.normalized[origin:origin+48])
                self.assertEqual(tuple(first['history'].shape), (48, 3))
                self.assertEqual(tuple(first['target'].shape), (48, 3))


class ErrorWindows(torch.utils.data.Dataset):
    origins = np.arange(100, 105, dtype=np.int64)

    def __len__(self):
        return 5

    def __getitem__(self, index):
        history = torch.tensor([0., 1., 2., 3., 10.][index]).expand(48, 2).clone()
        history[:, 1] *= -2
        return {'history': history, 'target': torch.zeros(48, 2),
                'history_marks': torch.zeros(48, 5), 'future_marks': torch.zeros(48, 5),
                'node_std': torch.tensor([2., 3.]), 'origin_step': torch.tensor(self.origins[index])}


class TinyBaseline(torch.nn.Module):
    """Small stand-in for BERT; exercises the real baseline runner on CPU."""
    def __init__(self, *args):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def cuda(self):
        return self

    def forward(self, history, history_marks, decoder, future_marks, interaction_rank=0, return_features=False):
        if interaction_rank != 0 or return_features:
            raise AssertionError('Unexpected baseline call')
        return self.weight * history


class BaselineEvaluationTests(unittest.TestCase):
    def test_global_raw_unit_metrics_include_zero_targets_and_incomplete_batch(self):
        dataset = ErrorWindows()
        # Independent hand calculation: errors are [2e, -6e] at every horizon.
        # MAE=4*mean(e)=12.8; MSE=20*mean(e^2)=456; RMSE=sqrt(456).
        with tempfile.TemporaryDirectory() as directory:
            for batch_size in (2, 4, 5):
                path = Path(directory) / f'batch_{batch_size}.npz'
                args = SimpleNamespace(batch_size=batch_size, seed=2021, target_chunk=64)
                with patch.object(runner, 'move', side_effect=lambda batch, device: batch), \
                     patch.object(runner, 'autocast', side_effect=lambda device: contextlib.nullcontext()):
                    metrics = runner.evaluate(TinyBaseline(), None, dataset, args, path)
                self.assertAlmostEqual(metrics['mae'], 12.8, places=6)
                self.assertAlmostEqual(metrics['mse'], 456., places=6)
                self.assertAlmostEqual(metrics['rmse'], math.sqrt(456), places=6)
                self.assertEqual(metrics['windows'], 5)
                np.testing.assert_allclose(metrics['horizon_mae'], 12.8)
                np.testing.assert_allclose(metrics['horizon_rmse'], math.sqrt(456))
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['origin'], dataset.origins)
                    self.assertEqual(metrics['mae'], float(saved['window_mae'].mean()))
                    self.assertEqual(metrics['rmse'], float(saved['window_mse'].mean()) ** .5)

    def test_nonfinite_predictions_fail_instead_of_publishing_metrics(self):
        model = TinyBaseline()
        model.weight.data.fill_(float('nan'))
        with patch.object(runner, 'move', side_effect=lambda batch, device: batch), \
             patch.object(runner, 'autocast', side_effect=lambda device: contextlib.nullcontext()):
            with self.assertRaisesRegex(RuntimeError, 'nonfinite evaluation'):
                runner.evaluate(model, None, ErrorWindows(), SimpleNamespace(batch_size=4, seed=1))

    def test_loader_keeps_last_batch_and_seeded_training_order(self):
        dataset = torch.arange(11)
        evaluate_order = torch.cat(list(loader(dataset, 4, False, 2021))).tolist()
        self.assertEqual(evaluate_order, list(range(11)))
        first = torch.cat(list(loader(dataset, 4, True, 2021))).tolist()
        repeat = torch.cat(list(loader(dataset, 4, True, 2021))).tolist()
        other = torch.cat(list(loader(dataset, 4, True, 2022))).tolist()
        self.assertEqual(sorted(first), list(range(11)))
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)


class BaselineIntegrationTests(unittest.TestCase):
    def test_resumed_training_matches_uninterrupted_optimizer_and_rng(self):
        class DropoutBaseline(TinyBaseline):
            def forward(self, history, history_marks, decoder, future_marks, **kwargs):
                return self.weight * torch.nn.functional.dropout(history, p=.2, training=self.training)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, resumed = root / 'full', root / 'resumed'
            full.mkdir()
            resumed.mkdir()
            datasets = {'train': ErrorWindows(), 'val': ErrorWindows()}
            args = SimpleNamespace(dataset='pems03', batch_size=4, seed=2021, resume=False)
            with patch.object(runner, 'move', side_effect=lambda batch, device: batch), \
                 patch.object(runner, 'autocast', side_effect=lambda device: contextlib.nullcontext()), \
                 patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                 patch.object(torch.cuda, 'set_rng_state_all'), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
                torch.manual_seed(2021)
                runner.train_stage(DropoutBaseline(), None, datasets, args, {}, full, 2)
                torch.manual_seed(2021)
                runner.train_stage(DropoutBaseline(), None, datasets, args, {}, resumed, 1)
                torch.manual_seed(999)  # Resume must restore RNG, not use this state.
                args.resume = True
                runner.train_stage(DropoutBaseline(), None, datasets, args, {}, resumed, 2)
            uninterrupted = torch.load(full / 'baseline_last.pt', map_location='cpu', weights_only=False)
            restarted = torch.load(resumed / 'baseline_last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(uninterrupted['epoch'], restarted['epoch'])
            for name in uninterrupted['parameters']:
                torch.testing.assert_close(uninterrupted['parameters'][name], restarted['parameters'][name], atol=0, rtol=0)
            torch.testing.assert_close(uninterrupted['rng'], restarted['rng'], atol=0, rtol=0)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(uninterrupted['optimizer']['state'][0][key],
                                           restarted['optimizer']['state'][0][key], atol=0, rtol=0)

    def test_cpu_training_save_reload_test_and_source_resume_guard(self):
        from models import sthsepnet_h1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = fixture(root / 'dataset', rows=4001)
            (root / 'research').mkdir()
            (root / 'research/coursework_config.json').write_bytes((ROOT / 'research/coursework_config.json').read_bytes())
            source = root / 'fixture_source.py'
            source.write_text('original')
            output = root / 'run'
            args = SimpleNamespace(dataset='pems03', seed=2021, output=output, stage='baseline',
                batch_size=4, target_chunk=64, smoke=True, resume=False, bert_path=root / 'unused_bert')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'SOURCES', ('fixture_source.py',)), \
                 patch.dict(data.SPECS, {'pems03': spec}), \
                 patch.object(sthsepnet_h1, 'Model', TinyBaseline), \
                 patch.object(runner.PeriodicBertV4, '__init__', side_effect=AssertionError('Baseline constructed adapter')), \
                 patch.object(runner, 'move', side_effect=lambda batch, device: batch), \
                 patch.object(runner, 'autocast', side_effect=lambda device: contextlib.nullcontext()), \
                 patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
                runner.run(args)
                report = json.loads((output / 'report.json').read_text())
                self.assertEqual(report['status'], 'baseline_completed')
                self.assertTrue(report['configuration']['smoke'])
                self.assertEqual(set(report['stages']), {'baseline'})
                self.assertEqual(report['stages']['baseline']['epochs'], 1)
                self.assertTrue(math.isfinite(report['stages']['baseline']['test']['rmse']))
                for name in ('baseline_best.pt', 'baseline_last.pt', 'baseline_test.npz'):
                    self.assertTrue((output / name).is_file())
                checkpoint = torch.load(output / 'baseline_last.pt', map_location='cpu', weights_only=False)
                self.assertIn('optimizer', checkpoint)
                self.assertIn('rng', checkpoint)
                self.assertEqual(checkpoint['epoch'], 1)
                with self.assertRaises(FileExistsError):
                    runner.run(args)
                source.write_text('changed')
                args.resume = True
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    runner.run(args)

    def test_cli_accepts_all_five_datasets(self):
        result = subprocess.run([sys.executable, str(ROOT / 'research/run_coursework.py'), '--help'],
                                capture_output=True, text=True, check=True)
        for dataset in data.SPECS:
            self.assertIn(dataset, result.stdout)

    def test_export_aggregates_five_datasets_and_detects_tampered_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source', root / 'published'
            for dataset in data.SPECS:
                for value, seed in enumerate((2021, 2022, 2023), 1):
                    folder = source / dataset / f'seed_{seed}'
                    folder.mkdir(parents=True)
                    report = {'status': 'baseline_completed',
                        'configuration': {'dataset': dataset, 'seed': seed, 'smoke': False},
                        'stages': {'baseline': {'epochs': 2, 'best_epoch': 1, 'stop_reason': 'epoch_cap',
                            'test': {'mae': float(value), 'mse': float(value**2), 'rmse': float(value), 'windows': 2}}}}
                    (folder / 'report.json').write_text(json.dumps(report))
                    np.savez(folder / 'baseline_test.npz', origin=np.array([100, 101]),
                             window_mae=np.full(2, float(value)), window_mse=np.full(2, float(value**2)))
            command = [sys.executable, str(ROOT / 'research/export_baseline_results.py')]
            subprocess.run(command + ['--source', str(source), '--output', str(output)], check=True, capture_output=True)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['runs'], 15)
            self.assertEqual(len(summary['aggregate']), 5)
            for row in summary['aggregate']:
                self.assertEqual(row['mae_mean'], 2.)
                self.assertEqual(row['mae_std'], 1.)
            subprocess.run(command + ['--check', '--output', str(output)], check=True, capture_output=True)
            path = output / 'errors/pems03/seed_2021_test.npz'
            np.savez(path, origin=np.array([100, 101]), window_mae=np.array([99., 99.]), window_mse=np.ones(2))
            result = subprocess.run(command + ['--check', '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('metric mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
