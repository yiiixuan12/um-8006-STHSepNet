"""Joint-repair contracts: raw values, train-only labels, shared tokens, no bypass."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertModel

from models.periodic_bert_v4 import PeriodicBertV4
from models.periodic_tokens import UtilityPeriodicRetrieval, utility_teacher, utility_cross_entropy
from research.run_periodic_bert_repair import checkpoint_payload, restore_checkpoint, train
from tests.test_periodic_bert_repair_runner import ToyAdapter, ToyWindows


class UtilityRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        BertModel(BertConfig(hidden_size=24,num_hidden_layers=6,num_attention_heads=4,
            intermediate_size=48,vocab_size=16,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),
            add_pooling_layer=False).save_pretrained(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self, **kwargs):
        torch.manual_seed(2021)
        model = PeriodicBertV4(self.path,repair="utility",node_chunk_size=2,
            gradient_checkpointing=False,dropout=0.,lora_rank=8,**kwargs).eval()
        model.numeric_calibration.install({"mean":torch.zeros(252),"scale":torch.ones(252),
            "weight":torch.zeros(4,252),"bias":torch.tensor([.1,.2,.3,.4])})
        return model

    def data(self):
        torch.manual_seed(7)
        return {"history":torch.randn(1,2064,3),"short_state":torch.randn(1,3,128),
            "baseline_prediction":torch.randn(1,48,3),"origin_step":torch.tensor([3000]),
            "neighbor_history":torch.randn(1,3,2,48),"neighbor_valid":torch.ones(1,3,2,dtype=torch.bool),
            "relation_address":torch.randn(1,3,2,4)}

    def test_zero_heads_preserve_numerical_reference_exactly(self):
        model=self.model(); data=self.data()
        result,aux=model(**data,return_auxiliary=True)
        expected=torch.tensor([.1,.2,.3,.4])[None,:,None].repeat_interleave(12,1).expand(1,48,3)
        torch.testing.assert_close(result,expected,atol=0,rtol=0)
        self.assertEqual(aux["utility_logits"].shape,(3,4,36))

    def test_current_and_candidate_contexts_use_same_patch_count_and_span(self):
        model=self.model(); data=self.data(); calls=[]
        handle=model.patch_encoder.register_forward_pre_hook(lambda module,args:calls.append(tuple(args[0].shape)))
        try:
            model._strict_context_keys(data["history"],data["short_state"][0],torch.tensor([3000,3000,3000]))
        finally: handle.remove()
        self.assertEqual(calls,[(3,4,12),(105,4,12)])

    def test_historical_outcomes_do_not_change_logits(self):
        model=self.model(); data=self.data()
        with torch.no_grad(): model.pattern_gate.fill_(.2)
        first,a=model(**data,return_auxiliary=True)
        changed={**data,"history":data["history"].clone()}
        changed["history"][:,2064-3*288+24:2064-3*288+36]+=3
        second,b=model(**changed,return_auxiliary=True)
        torch.testing.assert_close(a["utility_logits"],b["utility_logits"],atol=0,rtol=0)
        self.assertGreater(float((first-second).detach().abs().max()),1e-6)

    def test_raw_continuations_are_not_rescaled(self):
        operator=UtilityPeriodicRetrieval(32)
        history=torch.randn(2,2064); state=torch.randn(2,32); keys=torch.randn(2,35,32)
        ref=torch.randn(2,48)
        template,info=operator(history,state,candidate_keys=keys,reference_prediction=ref)
        candidate,_,_=operator.candidates(history)
        values=candidate[...,48:].reshape(2,35,4,12).permute(0,2,1,3)
        p=info["pattern_weights"][...,None]
        expected=(p*values).sum(2)+info["abstain_probability"][...,None]*ref.reshape(2,4,12)
        torch.testing.assert_close(template,expected.flatten(1))

    def test_teacher_abstains_when_reference_is_exact(self):
        history=torch.randn(2,2064); truth=torch.randn(2,48)
        teacher=utility_teacher(history,truth,truth,torch.ones(2),loss_scale=1.)
        torch.testing.assert_close(teacher[...,-1],torch.ones(2,4),atol=0,rtol=0)
        self.assertEqual(float(teacher[...,:35].sum()),0.)

    def test_teacher_never_labels_invalid_candidates_and_is_detached(self):
        history=torch.randn(2,2064,requires_grad=True); target=torch.randn(2,48,requires_grad=True)
        teacher=utility_teacher(history,target,target.detach()+10,torch.ones(2),loss_scale=1.)
        _,valid,_=UtilityPeriodicRetrieval.candidates(history)
        self.assertFalse(teacher.requires_grad)
        self.assertEqual(float(teacher[...,:35][:,:,~valid].sum()),0.)
        torch.testing.assert_close(teacher.sum(-1),torch.ones(2,4))

    def test_utility_supervision_reaches_backbone_at_zero_forecast_gate(self):
        model=self.model(); data=self.data()
        _,aux=model(**data,return_auxiliary=True)
        labels=torch.zeros_like(aux["utility_logits"]); labels[:,:,2]=1
        utility_cross_entropy(aux["utility_logits"],labels).backward()
        params=dict(model.named_parameters())
        for name in ("pattern_operator.query.weight","pattern_operator.key.weight",
                     "utility_context_pool.2.weight","bert.encoder.layer.0.attention.self.value.lora_B"):
            grad=params[name].grad
            self.assertIsNotNone(grad,name)
            self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.abs().sum()),0.,name)

    def test_no_direct_short_or_base_bypass_at_main_head(self):
        model=self.model(); data=self.data()
        with torch.no_grad(): model.hourly_readout.weight.normal_(std=.1)
        handle=model.content_read.register_forward_hook(lambda module,args,out:(torch.zeros_like(out[0]),out[1]))
        try:
            a=model(**data)
            b=model(**{**data,"short_state":data["short_state"]*9,"baseline_prediction":data["baseline_prediction"]+5})
        finally: handle.remove()
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_checkpoints_round_trip_and_flags_are_explicit(self):
        model=self.model(); other=self.model(); restore_checkpoint(other,checkpoint_payload(model,0))
        torch.testing.assert_close(model(**self.data()),other(**self.data()),atol=0,rtol=0)
        self.assertTrue(model.metadata["utility_supervision"])
        self.assertEqual(model.metadata["lora_rank"],8)
        with self.assertRaises(ValueError): model(**self.data(),pattern_mode="no_amplitude")

    def test_matched_budget_disables_early_stopping(self):
        args=SimpleNamespace(interface_lr=.01,backbone_lr=.01,seed=2021,batch_size=1,
            eval_batch_size=1,accumulation=1,warmup_epochs=0,min_epochs=1,epochs=3,
            patience=1,resume=False,matched_updates=True)
        values=[({"mae":1.,"rmse":1.},{})]+[({"mae":2.,"rmse":2.},{})]*3
        with TemporaryDirectory() as directory, patch("research.run_periodic_bert_repair.evaluate",side_effect=values):
            result=train(ToyAdapter(),ToyWindows(),ToyWindows(),np.zeros(2,int),torch.device("cpu"),
                args,Path(directory),"toy",{},time.time()+30)
        self.assertEqual(result["epochs_completed"],3)
        self.assertEqual(result["stop_reason"],"epoch_cap")
        self.assertEqual(result["optimizer_updates_completed_epochs"],6)

    def test_zero_weight_never_computes_teacher_and_trains_forecast_only(self):
        class ToyUtilityAdapter(ToyAdapter):
            uses_utility_supervision=True
            def forward(self,history,return_auxiliary=False,**kwargs):
                value=super().forward(history,**kwargs)
                if return_auxiliary:
                    return value,{"utility_logits":history.new_zeros(history.shape[0]*history.shape[2],4,36)}
                return value
        args=SimpleNamespace(interface_lr=.01,backbone_lr=.01,seed=2021,batch_size=1,
            eval_batch_size=1,accumulation=1,warmup_epochs=0,min_epochs=1,epochs=1,
            patience=1,resume=False,matched_updates=True,utility_weight=0.)
        with TemporaryDirectory() as directory,patch("research.run_periodic_bert_repair.utility_training_loss",
                side_effect=AssertionError("teacher must not be computed at zero weight")) as teacher:
            result=train(ToyUtilityAdapter(),ToyWindows(),ToyWindows(),np.zeros(2,int),torch.device("cpu"),
                args,Path(directory),"no_teacher",{},time.time()+30)
        teacher.assert_not_called()
        row=result["training_log"][0]
        self.assertFalse(row["teacher_supervision_enabled"])
        self.assertIsNone(row["utility_cross_entropy"])
        self.assertIsNone(row["teacher_abstain_fraction"])
        self.assertAlmostEqual(row["train_mae"],row["train_objective"],places=6)
        self.assertEqual(result["optimizer_updates_completed_epochs"],2)


if __name__ == "__main__":
    unittest.main()
