"""Queue plans and output guards; no training or resource downloads."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_runs', ROOT / 'scripts/training_runs.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class TrainingQueueTests(unittest.TestCase):
    def plans(self, mode):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/training_runs.py'), mode,
                                 '--plan-only'], capture_output=True, text=True, check=True)
        return [json.loads(line)['arguments'] for line in result.stdout.splitlines()]

    def test_cohort_counts_seeds_and_control_pairs(self):
        baseline = self.plans('baseline')
        h1 = self.plans('h1')
        kan = self.plans('kan')
        self.assertEqual(len(baseline), 15)
        self.assertEqual(len(h1), 20)
        self.assertEqual(len(kan), 6)
        self.assertEqual({x[x.index('--seed') + 1] for x in baseline}, {'2021', '2022', '2023'})
        self.assertEqual({x[x.index('--seed') + 1] for x in kan}, {'2021', '42', '7'})
        self.assertEqual(sum('--use_kan' in x for x in kan), 3)
        self.assertEqual(sum(x[x.index('--stage') + 1] == 'baseline' for x in h1), 5)
        from research import run_h1
        for args in h1:
            run_h1.validate_arguments(run_h1.parser().parse_args(args[1:]))

    def test_smoke_all_routes_have_separate_paths(self):
        plans = self.plans('smoke')
        self.assertEqual(len(plans), 4)
        self.assertEqual(sum('--smoke' in x for x in plans), 3)
        self.assertIn('scripts/smoke_h1_training.py', plans[1])
        outputs = [x[x.index('--output-dir' if '--output-dir' in x else '--output') + 1] for x in plans]
        self.assertEqual(len(set(outputs)), 4)

    def test_completed_baseline_is_skipped_and_smoke_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = {'configuration': {'dataset': 'pems03', 'seed': 2021, 'smoke': False},
                      'stages': {'baseline': {'status': 'completed_training', 'test': {}}}}
            (root / 'report.json').write_text(json.dumps(report))
            (root / 'baseline_test.npz').touch()
            with patch.object(queue, 'invoke') as invoke:
                queue.run_stage([], root, 'baseline', 'pems03', 2021, False)
                invoke.assert_not_called()
            report['configuration']['smoke'] = True
            (root / 'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, 'configuration mismatch'):
                queue.run_stage([], root, 'baseline', 'pems03', 2021, False)

    @unittest.skipUnless(shutil.which('powershell.exe'), 'PowerShell required')
    def test_all_powershell_entrypoints_plan_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('setup_training', 'smoke_training', 'run_baseline_seeds', 'run_kan_seeds', 'run_h1_seeds'):
                with self.subTest(name=name):
                    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                        '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts' / (name + '.ps1')),
                        '-PythonPath', sys.executable, '-PlanOnly'], cwd=directory,
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(result.stdout.strip())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
