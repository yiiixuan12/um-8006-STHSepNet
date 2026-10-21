from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd
from transformers import BertConfig, BertModel

from models.periodic_bert_v4 import PeriodicBertV4
from research.periodic_relation_data import RelationWindows, physical_neighbors


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(); cls.path = Path(cls.temp.name)
        config = BertConfig(hidden_size=24, num_hidden_layers=6, num_attention_heads=4,
                            intermediate_size=48, vocab_size=16, hidden_dropout_prob=0.,
                            attention_probs_dropout_prob=0.)
        BertModel(config, add_pooling_layer=False).save_pretrained(cls.path, safe_serialization=False)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def make(self, **kwargs):
        torch.manual_seed(2021)
        return PeriodicBertV4(self.path, dropout=0., node_chunk_size=2, **kwargs).eval()

    def data(self):
        torch.manual_seed(7)
        return {"history": torch.randn(1, 2064, 3), "short_state": torch.randn(1, 3, 128),
                "neighbor_history": torch.randn(1, 3, 2, 48),
                "neighbor_valid": torch.tensor([[[True, False], [True, True], [False, False]]]),
                "relation_address": torch.randn(1, 3, 2, 4), "origin_step": torch.tensor([3000])}

    def open_head(self, model):
        with torch.no_grad(): model.output_head[-1].weight.fill_(.08)

    def test_zero_head_restores_baseline_without_small_gate(self):
        model = self.make(); result = model(**self.data())
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertFalse(model.output_gate.requires_grad)
        self.assertFalse(hasattr(model, "encoder_a"))

    def test_two_optimizer_steps_reach_interface_retrieval_sources_and_lora(self):
        model = self.make(); model.train(); data = self.data()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003)
        for _ in range(2):
            optimizer.zero_grad(); (model(**data)-torch.linspace(-1., 1., 48)[None, :, None]).square().mean().backward(); optimizer.step()
        names = dict(model.named_parameters())
        for name in ("patch_encoder.statistics_projection.weight", "retrieval_query.weight",
                     "retrieval_key.weight", "relation_projection.weight",
                     "bert.encoder.layer.0.attention.self.value.lora_B",
                     "bert.encoder.layer.5.attention.self.value.lora_B"):
            self.assertIsNotNone(names[name].grad, name)
            self.assertTrue(torch.isfinite(names[name].grad).all(), name)
            self.assertGreater(float(names[name].grad.abs().sum()), 0., name)

    def test_attention_mask_blocks_query_feedback_and_target_to_sources(self):
        valid = torch.tensor([[True, True, True, False, True, True]])
        allowed = PeriodicBertV4.joint_allowed(valid, 2, 2)
        self.assertFalse(allowed[0, :4, 4:].any())
        self.assertFalse(allowed[0, 2, :2].any())  # valid source row
        self.assertFalse(allowed[0, :, 3].any())  # padded source never key
        self.assertTrue(allowed.any(-1).all())

    def test_masked_source_cannot_affect_output_and_no_neighbor_is_finite(self):
        model = self.make(); self.open_head(model); data = self.data()
        normal = model(**data)
        changed = {**data, "neighbor_history": data["neighbor_history"].clone()}
        changed["neighbor_history"][~data["neighbor_valid"]] = 1e5
        torch.testing.assert_close(normal, model(**changed), rtol=0, atol=0)
        changed["neighbor_valid"] = torch.zeros_like(data["neighbor_valid"])
        self.assertTrue(torch.isfinite(model(**changed)).all())

    def test_real_source_and_temporal_content_can_affect_forecast(self):
        model = self.make(); self.open_head(model); data = self.data()
        normal = model(**data)
        self.assertGreater(float((normal-model(**data, source_mode="zero")).abs().max()), 1e-8)
        self.assertGreater(float((normal-model(**data, source_mode="time_shuffle")).abs().max()), 1e-8)
        changed = {**data, "history": data["history"].roll(1, 2)}
        self.assertGreater(float((normal-model(**changed)).abs().max()), 1e-8)

    def test_node_neighbor_permutation_and_chunk_invariance(self):
        model = self.make(); self.open_head(model); data = self.data()
        normal = model(**data); order = torch.tensor([2, 0, 1])
        changed = {**data, "history": data["history"][:, :, order], "short_state": data["short_state"][:, order]}
        for name in ("neighbor_history", "neighbor_valid", "relation_address"):
            changed[name] = data[name][:, order].flip(2)
        torch.testing.assert_close(model(**changed), normal[:, :, order], rtol=1e-5, atol=1e-6)
        model.node_chunk_size = 1
        torch.testing.assert_close(model(**data), normal, rtol=1e-5, atol=1e-6)

    def test_pretrained_random_share_interface_and_staging_is_reversible(self):
        pre, random = self.make(), self.make(pretrained=False)
        for name, value in pre.state_dict().items():
            if not name.startswith("bert.") or "lora_" in name:
                self.assertTrue(torch.equal(value, random.state_dict()[name]), name)
        eligible = set(pre.adapter_parameter_names)
        pre.set_training_stage(True)
        self.assertFalse(any(p.requires_grad for name, p in pre.named_parameters() if name.startswith("bert.")))
        pre.set_training_stage(False)
        self.assertEqual({name for name, p in pre.named_parameters() if p.requires_grad}, eligible)
        self.assertTrue(pre.bert.embeddings.LayerNorm.weight.requires_grad)

    def test_retrieval_normalization_and_future_values_are_not_keys(self):
        model = self.make(); data = self.data()
        c = torch.randn(3, 4, 32); a = torch.randn(3, 7, 8, 32)
        rawc = torch.randn(3, 4, 12); rawa = torch.randn(3, 7, 8, 12)
        w = model._retrieval_weights(c, a, rawc, rawa, data["short_state"][0])
        a[:, :, 4:] += 100; rawa[:, :, 4:] += 100
        w2 = model._retrieval_weights(c, a, rawc, rawa, data["short_state"][0])
        torch.testing.assert_close(w, w2, rtol=0, atol=0)
        torch.testing.assert_close(w.sum(-1), torch.ones(3, 4))

    def test_dataset_excludes_target_and_future_and_keeps_unsampled_sources(self):
        graph_path = self.path/"graph.csv"
        pd.DataFrame([[0., .5, 0.], [.5, 0., .8], [0., .8, 0.]]).to_csv(graph_path, index=False)
        table = physical_neighbors(graph_path, 3, 2)
        self.assertFalse(np.any(table.indices == np.arange(3)[:, None]))
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(2300, 3)).astype(np.float32)
        bundle = SimpleNamespace(raw=raw, normalized=raw.copy(), std=np.ones(3, np.float32))
        cache = {"origin": np.array([2200]), "base": np.zeros((1, 48, 3), np.float32),
                 "short_state": np.zeros((1, 3, 128), np.float32)}
        dataset = RelationWindows(bundle, [2200], cache, table, train_nodes=1)
        before = dataset[0]; target_node = dataset.node_indices[0][0]
        for slot, source in enumerate(table.indices[target_node]):
            if source >= 0:
                np.testing.assert_equal(before["neighbor_history"][0, slot].numpy(), raw[2152:2200, source])
        bundle.normalized[2200:2248] += 100
        after = dataset[0]
        for name in ("history", "neighbor_history", "short_state", "base"):
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)

    def test_numeric_backbone_uses_same_interface_and_finite_output(self):
        model = self.make(numeric=True)
        self.assertEqual(model.bert_width, 64)
        self.assertEqual(model.num_layers, 2)
        self.assertTrue(torch.isfinite(model(**self.data())).all())

    def test_upper_source_values_remain_target_excluding(self):
        model = self.make(); data = self.data()
        _, before = model(**data, return_diagnostics=True)
        changed = {**data, "history": data["history"].roll(1, 2)}
        _, after = model(**changed, return_diagnostics=True)
        mask = data["neighbor_valid"].reshape(3, 2, 1).expand(-1, -1, 4).flatten(1)
        torch.testing.assert_close(before["memory_values"][:, 88:][mask],
                                   after["memory_values"][:, 88:][mask], rtol=0, atol=0)

    def test_gated_ablation_has_nonzero_head_and_can_start_learning(self):
        model = self.make(gated_output=True, late_shortcut=True)
        result = model(**self.data())
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        (result-1).square().mean().backward()
        self.assertGreater(float(model.output_gate.grad.abs().sum()), 0.)

    def test_disabling_checkpointing_preserves_forward_and_gradients(self):
        model = self.make(); self.open_head(model); model.train(); data = self.data()
        model.gradient_checkpointing = True
        first = model(**data); first.square().mean().backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True); model.gradient_checkpointing = False
        second = model(**data); second.square().mean().backward()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-5, atol=1e-7)


if __name__ == "__main__": unittest.main()
