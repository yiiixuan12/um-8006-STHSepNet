"""Information boundaries for the supplementary first-contribution ablations."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from models.periodic_memory_policy import (MEMORY_POLICIES, transform_history,
    candidate_memory_enabled, memory_token_mask, mask_numeric_features, history_from_bundle)


class MemoryPolicyTests(unittest.TestCase):
    def test_temporal_policies_remove_older_information_but_preserve_recent(self):
        x=torch.randn(2,2064,3)
        for policy,steps,tokens in (("recent_only",48,16),("day_only",288,24)):
            changed=transform_history(x,policy)
            torch.testing.assert_close(changed[:,-steps:],x[:,-steps:],atol=0,rtol=0)
            self.assertEqual(int(changed[:,:-steps].count_nonzero()),0)
            self.assertEqual(int(memory_token_mask(policy).sum()),tokens)
            self.assertFalse(candidate_memory_enabled(policy))
        self.assertGreater(int(x[:,:-288].count_nonzero()),0)

    def test_global_history_uses_all_roads_not_sampled_target_batch(self):
        raw=np.stack((np.arange(2200),np.arange(2200)*2+10,np.arange(2200)*3+20),1).astype(np.float32)
        mean=np.array([1,2,3],np.float32); std=np.array([2,3,4],np.float32)
        bundle=SimpleNamespace(raw=raw,normalized=(raw-mean)/std,mean=mean,std=std)
        one=history_from_bundle(bundle,2100,np.array([1]),"global_long")
        all_roads=history_from_bundle(bundle,2100,np.arange(3),"global_long")
        np.testing.assert_array_equal(one[:,0],all_roads[:,1])
        np.testing.assert_array_equal(one[-48:,0],bundle.normalized[2052:2100,1])
        expected=((raw[36:2052].mean(1,dtype=np.float64)-mean[1])/std[1]).astype(np.float32)
        np.testing.assert_array_equal(one[:-48,0],expected)

    def test_global_rejects_implicit_target_batch_average(self):
        with self.assertRaises(ValueError): transform_history(torch.randn(1,2064,2),"global_long")

    def test_history_policies_do_not_read_forecast_targets(self):
        rng=np.random.default_rng(1); raw=rng.normal(size=(2300,3)).astype(np.float32)
        for policy in MEMORY_POLICIES:
            a=SimpleNamespace(raw=raw,normalized=raw.copy(),mean=np.zeros(3),std=np.ones(3))
            altered=raw.copy(); altered[2100:]+=10000
            b=SimpleNamespace(raw=altered,normalized=altered.copy(),mean=np.zeros(3),std=np.ones(3))
            np.testing.assert_array_equal(history_from_bundle(a,2100,np.array([0,2]),policy),
                                          history_from_bundle(b,2100,np.array([0,2]),policy))

    def test_feature_masks_match_token_groups_and_preserve_common_controls(self):
        x=torch.arange(252.)[None]
        for policy,start,stop in (("no_repeat",244,248),("no_deviation",248,252)):
            y=mask_numeric_features(x,policy)
            torch.testing.assert_close(y[:,:start],x[:,:start],atol=0,rtol=0)
            torch.testing.assert_close(y[:,stop:],x[:,stop:],atol=0,rtol=0)
            self.assertEqual(int(y[:,start:stop].count_nonzero()),0)
            self.assertEqual(int(memory_token_mask(policy).sum()),84)
        for policy in ("recent_only","day_only"):
            y=mask_numeric_features(x,policy)
            torch.testing.assert_close(y[:,:176],x[:,:176],atol=0,rtol=0)
            torch.testing.assert_close(y[:,184:188],x[:,184:188],atol=0,rtol=0)
            self.assertEqual(int(y[:,188:].count_nonzero()),0)
            np.testing.assert_array_equal(mask_numeric_features(x.numpy(),policy),y.numpy())

    def test_default_policy_has_no_copy_or_effect(self):
        history=torch.randn(1,2064,2); features=torch.randn(2,252)
        self.assertIs(transform_history(history,"full"),history)
        self.assertIs(mask_numeric_features(features,"full"),features)
        self.assertTrue(memory_token_mask("full").all())


if __name__ == "__main__": unittest.main()
