import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from research.experiment_registry import FIELDS, registered_run


class RegistryTests(unittest.TestCase):
    def exercise(self, error=None, smoke=False):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'dataset/sample').mkdir(parents=True)
            for name in ('data.csv','adj.csv'):
                (root/'dataset/sample'/name).write_text('1,2\n')
            args=SimpleNamespace(dataset='sample',stage='baseline',seed=2021,
                                 smoke=smoke,resume=False,output=root/'output')
            def callback(args):
                if error is not None:
                    raise error
                args.output.mkdir()
                (args.output/'report.json').write_text(json.dumps(
                    {'stages':{'baseline':{'test':{'mae':2.0}}}}))
            def invoke():
                registered_run(args,root,{'folder':'sample','csv':'data.csv','adj':'adj.csv'},
                               (),{'devices':[]},callback)
            if error is None:
                invoke()
            else:
                with self.assertRaises(type(error)):
                    invoke()
            with (root/'experiments/registry.csv').open(newline='',encoding='utf-8') as stream:
                reader=csv.DictReader(stream); rows=list(reader)
                self.assertEqual(reader.fieldnames,list(FIELDS))
            self.assertEqual(len(rows),1)
            self.assertEqual(len(json.loads(rows[0]['data_version'])['csv']),64)
            self.assertFalse(list((root/'experiments/runs').glob('*/started.json')))
            return rows[0]

    def test_success_and_smoke_are_distinct(self):
        self.assertEqual(self.exercise()['status'],'completed')
        row=self.exercise(smoke=True)
        self.assertEqual(row['status'],'completed')
        self.assertTrue(json.loads(row['notes'])['smoke'])

    def test_failure_and_interrupt_are_preserved(self):
        self.assertEqual(self.exercise(ValueError('bad input'))['status'],'failed')
        self.assertEqual(self.exercise(KeyboardInterrupt())['status'],'interrupted')

    def test_missing_data_is_registered(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(dataset='sample',stage='baseline',seed=1,
                                 smoke=False,resume=False,output=root/'output')
            with self.assertRaises(FileNotFoundError):
                registered_run(args,root,{'folder':'missing','csv':'data.csv','adj':'adj.csv'},
                               (),{},lambda _: self.fail('must not train'))
            with (root/'experiments/registry.csv').open(newline='') as stream:
                self.assertEqual([r['status'] for r in csv.DictReader(stream)],['failed'])


if __name__=='__main__': unittest.main()
