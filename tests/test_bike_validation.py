"""Only new risks: information removal, causal case selection and raw units."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from research.bike_validation_support import BikeMemoryWindows, choose_cases, restore_raw, zero_utility_loss
from research.fast_execution import deferred_utility_statistic


def bundle():
    n = 600
    raw = np.arange(n, dtype=np.float32)[:, None]*np.array([[1., 2., 4.]], np.float32)
    mean = raw[:300].mean(0); std = raw[:300].std(0)
    return SimpleNamespace(raw=raw, normalized=(raw-mean)/std, mean=mean, std=std,
        timestamps_ns=np.arange(n, dtype=np.int64)*3_600_000_000_000,
        marks=np.zeros((n, 5), np.float32), split_bounds={'train': (0, 300), 'val': (300, 400), 'test': (400, 600)},
        node_names=('a', 'b', 'c'))


class BikeValidationTests(unittest.TestCase):
    def test_global_is_all_station_raw_mean_and_recent_is_unchanged(self):
        b = bundle(); full = BikeMemoryWindows(b, 'train'); glob = BikeMemoryWindows(b, 'train', 'global_long')
        x, y = full[0], glob[0]
        self.assertTrue(torch.equal(x['history'][-48:], y['history'][-48:]))
        expected = (b.raw[:192].mean(1, dtype=np.float64)[:, None]-b.mean)/b.std
        np.testing.assert_allclose(y['history'][:-48].numpy(), expected, rtol=1e-6)
        for key in ('target', 'node_std', 'history_marks', 'future_marks', 'origin_step'):
            self.assertTrue(torch.equal(x[key], y[key]))
        np.testing.assert_array_equal(full.bundle.normalized[:240], x['history'])

    def test_future_change_does_not_change_history_or_selection(self):
        b = bundle(); first = BikeMemoryWindows(b, 'test', 'global_long')[0]
        cases = choose_cases(b, BikeMemoryWindows(b, 'test').origins)
        b.raw = b.raw.copy(); b.normalized = b.normalized.copy()
        b.raw[400:] = 99999; b.normalized[400:] = -999
        second = BikeMemoryWindows(b, 'test', 'global_long')[0]
        self.assertTrue(torch.equal(first['history'], second['history']))
        self.assertEqual(cases, choose_cases(b, BikeMemoryWindows(b, 'test').origins))

    def test_inverse_is_nodewise_and_preserves_raw_values(self):
        b = bundle()
        np.testing.assert_allclose(restore_raw(b.normalized, b.mean, b.std), b.raw, atol=1e-3)
        with self.assertRaises(ValueError):
            restore_raw(np.zeros((4, 2)), b.mean, b.std)

    def test_removing_auxiliary_loss_preserves_interface_without_labels(self):
        logits = torch.randn(6, 4, 36, requires_grad=True)
        loss, _ = deferred_utility_statistic(zero_utility_loss)(None, {}, logits, loss_scale=1.)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

if __name__ == '__main__':
    unittest.main()
