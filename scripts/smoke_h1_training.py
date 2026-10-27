"""One-batch H1 baseline + numeric fit + adapter train/save/reload/test.

Uses real local data and BERT. Smoke metrics are not accuracy evidence.
Formal training budgets in research/run_h1.py are unchanged.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from research import run_h1 as h1
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=h1.DATASETS, default='bike_outflow')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bert-path', type=Path, default=ROOT / 'huggingface/google_bert')
    options = p.parse_args()
    if options.output.exists():
        raise FileExistsError('Choose a new smoke output directory')
    args = h1.parser().parse_args(['--dataset', options.dataset, '--stage', 'baseline',
        '--output', str(options.output), '--bert-path', str(options.bert_path)])
    d = h1.dependencies()
    d.torch.set_num_threads(4)
    bundle, audit, short, long = h1.prepare_data(args, d)
    for datasets in (short, long):
        for data in datasets.values():
            data.origins = data.origins[:4]
    config = h1.new_config(args, bundle, d)
    base = h1.build_base(args, config, d)
    args.output.mkdir(parents=True)
    report = h1.fresh_report(args, audit, config)
    report['configuration']['smoke'] = True
    report['protocol'].update(all_training_windows=False, smoke_epochs=1)
    run_args = h1.args_for_training(args, bundle)
    baseline = d.shared.train_stage(base, None, short, run_args, report, args.output, 1)
    baseline['test'] = d.shared.evaluate(base, None, short['test'], run_args, args.output / 'baseline_test.npz')
    report['stages']['baseline'] = baseline
    base.eval().requires_grad_(False)
    student = h1.build_student(args, d)
    with d.extra.sampling_helpers(args.dataset):
        state, details = d.shared.fit_numeric(base, long, run_args)
        d.torch.save({'state': state, 'report': details}, args.output / 'numeric_calibration.pt')
        student.numeric_calibration.install(state)
        train = d.training_function(d.shared.train_stage, d.shared.utility_training_loss)
        row = train(base, student, long, run_args, report, args.output, 1)
        row['test'] = d.shared.evaluate(base, student, long['test'], run_args, args.output / 'improved_test.npz')
    report['stages']['improved'] = row
    report['numeric_fit'] = details
    report['status'] = 'smoke_completed_not_accuracy_evidence'
    report.pop('active', None)
    d.write_json(args.output / 'report.json', report)
    print('PASS: H1 baseline, numeric fit, adapter training, checkpoint reload and test')


if __name__ == '__main__':
    main()
