from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import torch
from torch import nn
import numpy as np

from research.run_periodic_bert_repair import adapter_state, restore_state, train, raw_error, linear_reference
from research.periodic_relation_data import NeighborTable, RelationWindows


class ToyAdapter(nn.Module):
    numeric = False
    def __init__(self):
        super().__init__(); self.bert = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.bert.weight)
        self.head = nn.Parameter(torch.zeros(()))
        self.adapter_parameter_names = tuple(name for name, _ in self.named_parameters())

    def set_training_stage(self, warmup):
        self.bert.weight.requires_grad_(not warmup)

    def forward(self, history, **unused):
        return (self.head+self.bert.weight.sum()).expand(history.shape[0], 48, history.shape[2])


class ToyWindows:
    origins = np.array([2064, 2100])
    def __len__(self): return 2
    def __getitem__(self, index):
        return {"history": torch.ones(2064, 2), "short_state": torch.zeros(2, 128),
                "neighbor_history": torch.zeros(2, 1, 48), "neighbor_valid": torch.zeros(2, 1, dtype=torch.bool),
                "relation_address": torch.zeros(2, 1, 4), "origin_step": torch.tensor(2064+index),
                "base": torch.zeros(48, 2), "target": torch.ones(48, 2), "node_std": torch.tensor([2., 3.])}


class RunnerTests(unittest.TestCase):
    def test_linear_reference_uses_the_same_dataset_node_selection(self):
        rng = np.random.default_rng(9)
        raw = rng.normal(size=(2600, 3)).astype(np.float32)
        bundle = SimpleNamespace(raw=raw, normalized=raw.copy(), std=np.ones(3, np.float32))
        table = NeighborTable(np.array([[1], [0], [1]]), np.ones((3, 1), bool), np.zeros((3, 1, 4), np.float32), {})
        datasets = {}
        for role, origins in {"train": [2064, 2080], "val_a": [2200, 2220], "val_b": [2400, 2420]}.items():
            cache = {"origin": np.array(origins), "base": np.zeros((2, 48, 3), np.float32),
                     "short_state": np.zeros((2, 3, 128), np.float32)}
            datasets[role] = RelationWindows(bundle, origins, cache, table, train_nodes=1 if role == "train" else 0)
        result = linear_reference(datasets)
        self.assertEqual(set(result), {"short_recent", "short_recent_plus_noise", "short_recent_plus_periodic"})
        self.assertTrue(np.isfinite(result["short_recent_plus_periodic"]["val_b"]["mae"]))

    def test_checkpoint_membership_is_independent_of_training_stage(self):
        model = ToyAdapter(); model.set_training_stage(True)
        saved = adapter_state(model)
        self.assertEqual(set(saved), {"head", "bert.weight"})
        with torch.no_grad(): model.bert.weight.fill_(9)
        restore_state(model, saved)
        self.assertEqual(float(model.bert.weight.detach().sum()), 0.)

    def test_raw_error_preserves_per_target_standard_deviation(self):
        batch = {"base": torch.zeros(1, 48, 2), "target": torch.ones(1, 48, 2), "node_std": torch.tensor([[2., 3.]])}
        error = raw_error(batch, torch.zeros(1, 48, 2))
        torch.testing.assert_close(error[0, 0], torch.tensor([-2., -3.]))

    def test_two_stage_training_checkpoint_and_noop_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = SimpleNamespace(interface_lr=.01, backbone_lr=.01, seed=2021, batch_size=1,
                                   eval_batch_size=1, accumulation=2, warmup_epochs=1, min_epochs=2,
                                   epochs=2, patience=3, resume=False)
            model = ToyAdapter(); dataset = ToyWindows()
            result = train(model, dataset, dataset, np.zeros(2, dtype=int), torch.device("cpu"),
                           args, output, "toy", {"runs": []}, time.time()+30)
            self.assertEqual(result["epochs_completed"], 2)
            self.assertEqual(result["training_log"][0]["stage"], "interface_only")
            self.assertEqual(result["training_log"][1]["stage"], "joint")
            self.assertTrue(model.bert.weight.requires_grad)
            checkpoint = torch.load(output/"toy_last.pt", weights_only=False)
            self.assertIn("bert.weight", checkpoint["adapter_state"])
            expected = adapter_state(model); args.resume = True
            restored = ToyAdapter()
            resumed = train(restored, dataset, dataset, np.zeros(2, dtype=int), torch.device("cpu"),
                            args, output, "toy", {"runs": []}, time.time()+30)
            self.assertEqual(resumed["epochs_completed"], 2)
            for name, parameter in adapter_state(restored).items():
                torch.testing.assert_close(parameter, expected[name])


if __name__ == "__main__": unittest.main()
