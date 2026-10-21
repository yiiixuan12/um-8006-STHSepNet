from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
import numpy as np
from research.coursework_data import CourseworkWindows,origins_for,load_dataset
from research.run_coursework import solve_moments,compact_state,load_compact,sliced_adapter_batch
from research.summarize_coursework import summarize
import json


class CourseworkTests(unittest.TestCase):
    def bundle(self):
        raw=np.arange(4000*3,dtype=np.float32).reshape(4000,3)
        return SimpleNamespace(normalized=raw,marks=np.zeros((4000,5),np.float32),std=np.ones(3,np.float32),
            split_bounds={'train':(0,2800),'val':(2800,3200),'test':(3200,4000)})

    def test_target_boundaries_and_common_test_origins(self):
        b=self.bundle()
        for role,(low,high) in b.split_bounds.items():
            values=origins_for(b,role,2064)
            self.assertGreaterEqual(int(values.min()),max(low,2064))
            self.assertEqual(int(values.max())+48,high)
        np.testing.assert_array_equal(origins_for(b,'test',48),origins_for(b,'test',2064))

    def test_future_observations_never_enter_inputs(self):
        b=self.bundle(); data=CourseworkWindows(b,'test',2064); before=data[0]
        b.normalized[3200:]+=999
        after=data[0]
        for name in ('history','history_marks','future_marks','node_std','origin_step'):
            torch.testing.assert_close(before[name],after[name],atol=0,rtol=0)
        self.assertFalse(torch.equal(before['target'],after['target']))

    def test_streaming_ridge_matches_full_standardized_solution(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(100,9)); x[:,-1]=2.; y=rng.normal(size=(100,4)); alpha=3.
        mean,scale,w,b=solve_moments(x.sum(0),y.sum(0),x.T@x,x.T@y,len(x),alpha)
        full=np.column_stack(((x-mean)/scale,np.ones(len(x))))
        penalty=np.eye(10)*alpha; penalty[-1,-1]=0
        expected=np.linalg.solve(full.T@full+penalty,full.T@y)
        np.testing.assert_allclose(np.vstack((w,b)),expected,atol=1e-12)

    def test_compact_checkpoint_preserves_trainable_parameters_and_buffers(self):
        torch.manual_seed(1); first=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4))
        first[0].weight.requires_grad_(False); first.train()(torch.randn(8,3)); state=compact_state(first,2)
        torch.manual_seed(1); other=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4))
        other[0].weight.requires_grad_(False); load_compact(other,state)
        first.eval(); other.eval(); x=torch.randn(3,3)
        torch.testing.assert_close(first(x),other(x),atol=0,rtol=0)

    def test_node_chunks_cover_all_roads_without_sampling(self):
        n=35; batch={'history':torch.randn(2,2064,n),'target':torch.randn(2,48,n),
            'node_std':torch.ones(2,n),'origin_step':torch.tensor([2300,2301])}
        prediction=torch.randn(2,48,n); short=torch.randn(2,n,128); parts=[]
        for start in range(0,n,16):
            part=sliced_adapter_batch(batch,prediction,short,start,min(n,start+16)); parts.append(part['history'])
            self.assertIs(part['origin_step'],batch['origin_step'])
        torch.testing.assert_close(torch.cat(parts,-1),batch['history'],atol=0,rtol=0)

    def test_dataset_audit_survives_json_roundtrip(self):
        root=Path(__file__).resolve().parents[1]/'dataset'
        if not (root/'PEMS03/PEMS03.csv').exists(): self.skipTest('local dataset unavailable')
        _,audit=load_dataset(root,'pems03')
        self.assertEqual(audit,json.loads(json.dumps(audit)))

    def test_summary_never_fills_missing_seeds_with_zero(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); path=root/'pems03/seed_2021'; path.mkdir(parents=True)
            report={'configuration':{'dataset':'pems03','seed':2021,'smoke':False},'stages':{'baseline':{
                'test':{'mae':20.,'rmse':30.,'mse':900.,'windows':100},
                'epochs':3,'best_epoch':2,'stop_reason':'epoch_cap'}}}
            (path/'report.json').write_text(json.dumps(report),encoding='utf-8')
            result=summarize(root)
            self.assertEqual(result['status'],'incomplete'); self.assertEqual(len(result['missing']),11)
            self.assertEqual(result['summary'][0]['mae'],{'mean':20.,'sample_std':None})


if __name__=='__main__': unittest.main()
