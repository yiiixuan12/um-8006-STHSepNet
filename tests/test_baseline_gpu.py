"""Opt-in real BERT/CUDA integration tests. Never download data or weights."""
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('RUN_BASELINE_GPU') == '1', 'Set RUN_BASELINE_GPU=1 for local GPU integration')
class BaselineGPUIntegrationTests(unittest.TestCase):
    def test_real_hourly_and_five_minute_models_train_save_and_evaluate(self):
        import torch
        self.assertTrue(torch.cuda.is_available(), 'CUDA is required for this explicit GPU test')
        self.assertTrue((ROOT / 'huggingface/google_bert/config.json').is_file(), 'Prepare local BERT first')
        with tempfile.TemporaryDirectory(prefix='baseline_gpu_') as directory:
            for dataset in ('bike_outflow', 'pems03'):
                with self.subTest(dataset=dataset):
                    output = Path(directory) / dataset
                    result = subprocess.run([sys.executable, '-u', str(ROOT / 'research/run_coursework.py'),
                        '--dataset', dataset, '--seed', '2021', '--stage', 'baseline', '--smoke',
                        '--output', str(output)], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads((output / 'report.json').read_text())
                    self.assertTrue(report['configuration']['smoke'])
                    self.assertEqual(report['base_model_config']['llm_layers'], 12)
                    self.assertEqual(report['base_model_config']['llm_control'], 'corrected_lora')
                    self.assertEqual(set(report['stages']), {'baseline'})
                    stage = report['stages']['baseline']
                    self.assertEqual(stage['epochs'], 1)
                    self.assertEqual(stage['test']['windows'], 4)
                    for metric in ('mae', 'rmse', 'mse'):
                        self.assertTrue(math.isfinite(stage['test'][metric]))
                    self.assertTrue((output / 'baseline_best.pt').is_file())
                    self.assertTrue((output / 'baseline_last.pt').is_file())
                    self.assertTrue((output / 'baseline_test.npz').is_file())
                    checkpoint = torch.load(output / 'baseline_best.pt', map_location='cpu', weights_only=False)
                    llm_parameters = [name for name in checkpoint['parameters'] if name.startswith('llm_model.')]
                    self.assertTrue(llm_parameters, 'No trainable BERT LoRA parameters saved')
                    self.assertTrue(all(name.endswith(('weight_A', 'weight_B')) for name in llm_parameters),
                                    'Original BERT parameters should stay frozen')
                    layers = {int(match.group(1)) for name in llm_parameters
                              if (match := re.search(r'encoder\.layer\.(\d+)\.', name))}
                    self.assertEqual(layers, set(range(12)))


if __name__ == '__main__':
    unittest.main()
