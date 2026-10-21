"""CPU-only control-flow regression: same patience and best restore in both arms."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from research import run_coursework as runner


class EarlyStopTests(unittest.TestCase):
    def test_exact_five_consecutive_nonimproving_epochs(self):
        self.assertEqual(runner.EARLY_STOP_PATIENCE,5)
        self.assertFalse(runner.early_stop_reached(0))
        self.assertFalse(runner.early_stop_reached(4))
        self.assertTrue(runner.early_stop_reached(5))
        self.assertTrue(runner.early_stop_reached(6))

    def test_both_methods_stop_and_restore_best_without_losing_resume_state(self):
        for improved in (False,True):
            with self.subTest(improved=improved), TemporaryDirectory() as directory:
                base=torch.nn.Linear(1,1,bias=False)
                student=torch.nn.Linear(1,1,bias=False) if improved else None
                model=student if improved else base
                batch={'history':torch.ones(1,48,1),'target':torch.zeros(1,48,1),
                    'node_std':torch.ones(1,1),'origin_step':torch.tensor([2200])}
                datasets={'train':[batch],'val':[batch]}
                args=SimpleNamespace(dataset='pems03',batch_size=4,seed=2021,target_chunk=64,raw_scale=1.,resume=False)
                def prediction(module,data,features=False):
                    if features:
                        return module(data['history']),{'short_state':torch.zeros(1,1,128)}
                    return module(data['history'])
                def delta(module,data,aux=False):
                    value=module(data['history'])
                    return (value,{'utility_logits':value}) if aux else value
                values=[{'mse':v,'mae':v,'rmse':v**.5,'seconds':0.} for v in (1.,2.,3.,4.,5.,6.,1.)]
                with patch.object(runner,'loader',return_value=[batch]), \
                     patch.object(runner,'move',side_effect=lambda value,device:value), \
                     patch.object(runner,'base_prediction',side_effect=prediction), \
                     patch.object(runner,'adapter_delta',side_effect=delta), \
                     patch.object(runner,'training_loss',side_effect=lambda data,value,*a,**k:(value-data['target']).square().mean()), \
                     patch.object(runner,'utility_training_loss',side_effect=lambda module,data,logits,**k:(logits.sum()*0,0.)), \
                     patch.object(runner,'evaluate',side_effect=values) as evaluation, \
                     patch.object(torch.cuda,'get_rng_state_all',return_value=[]), \
                     patch.object(torch.cuda,'set_rng_state_all'), \
                     patch.object(torch.cuda,'max_memory_allocated',return_value=0):
                    output=Path(directory)
                    result=runner.train_stage(base,student,datasets,args,{},output,30)
                    self.assertEqual(result['epochs'],6)
                    self.assertEqual(result['best_epoch'],1)
                    self.assertEqual(result['stop_reason'],'early_stop')
                    stage='improved' if improved else 'baseline'
                    best=torch.load(output/f'{stage}_best.pt',weights_only=False)
                    last=torch.load(output/f'{stage}_last.pt',weights_only=False)
                    self.assertEqual(last['stale'],5)
                    self.assertEqual(last['early_stop_patience'],5)
                    self.assertIn('optimizer',last)
                    self.assertIn('rng',last)
                    for name,value in model.named_parameters():
                        torch.testing.assert_close(value,best['parameters'][name],atol=0,rtol=0)
                    # A completed stale counter survives resume, with no extra updates.
                    args.resume=True
                    evaluation.side_effect=[values[-1]]
                    resumed=runner.train_stage(base,student,datasets,args,{},output,30)
                    self.assertEqual(resumed['epochs'],6)
                    self.assertEqual(resumed['best_epoch'],1)
                    self.assertEqual(resumed['stop_reason'],'early_stop')


if __name__=='__main__': unittest.main()
