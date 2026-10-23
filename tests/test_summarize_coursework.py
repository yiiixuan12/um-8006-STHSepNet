import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research.summarize_coursework import DATASETS,summarize


class SummaryTests(unittest.TestCase):
    def write_report(self,root,dataset,seed,mae,smoke=False):
        path=root/dataset/f'seed_{seed}'
        path.mkdir(parents=True)
        report={'configuration':{'dataset':dataset,'seed':seed,'smoke':smoke},
                'stages':{'baseline':{'test':{'mae':mae,'rmse':mae*2,'mse':(mae*2)**2,'windows':10},
                                      'epochs':3,'best_epoch':2,'stop_reason':'early_stop'}}}
        (path/'report.json').write_text(json.dumps(report),encoding='utf-8')

    def test_all_five_datasets_baseline_mean_and_sample_std(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for dataset in DATASETS:
                for seed,mae in zip((2021,2022,2023),(1.,2.,3.)):
                    self.write_report(root,dataset,seed,mae)
            result=summarize(root,stages=('baseline',))
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['per_seed']),15)
            self.assertEqual(len(result['summary']),5)
            for row in result['summary']:
                self.assertEqual(row['mae'],{'mean':2.,'sample_std':1.})
            paired=summarize(root)
            self.assertEqual(len(paired['missing']),15)
            self.assertTrue(all(name.endswith('/improved') for name in paired['missing']))

    def test_missing_dataset_is_explicit_not_zero(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            self.write_report(root,'bike_outflow',2021,5.)
            result=summarize(root,stages=('baseline',))
            self.assertEqual(len(result['missing']),14)
            empty=next(r for r in result['summary'] if r['dataset']=='bike_inflow')
            self.assertEqual(empty['mae'],{'mean':None,'sample_std':None})
            one=next(r for r in result['summary'] if r['dataset']=='bike_outflow')
            self.assertEqual(one['mae'],{'mean':5.,'sample_std':None})

    def test_smoke_rejected_and_subset_supported(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            self.write_report(root,'bj500',2021,1.,smoke=True)
            with self.assertRaises(ValueError):
                summarize(root,('bj500',),('baseline',))
            result=summarize(root,('metr_la',),('baseline',))
            self.assertEqual(len(result['summary']),1)
            self.assertEqual(len(result['missing']),3)


if __name__=='__main__': unittest.main()
