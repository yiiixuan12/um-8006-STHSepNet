"""Aggregate all five paper datasets; never substitute missing seeds with zero."""
import argparse
import csv
import json
from pathlib import Path
import statistics

DATASETS=('pems03','metr_la','bj500','bike_inflow','bike_outflow')
STAGES=('baseline','improved')


def summarize(root,datasets=DATASETS,stages=STAGES):
    root=Path(root)
    if not root.is_dir():
        raise ValueError(f'results directory does not exist: {root}')
    datasets=tuple(dict.fromkeys(datasets)); stages=tuple(dict.fromkeys(stages))
    if not datasets or any(name not in DATASETS for name in datasets):
        raise ValueError('select supported datasets')
    if not stages or any(name not in STAGES for name in stages):
        raise ValueError('select baseline and/or improved stages')
    rows=[]; missing=[]; summaries=[]
    for dataset in datasets:
        per_stage={stage:[] for stage in stages}
        for seed in (2021,2022,2023):
            path=root/dataset/f'seed_{seed}'/'report.json'
            report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
            for stage in per_stage:
                record=None if report is None else report.get('stages',{}).get(stage)
                if record is None:
                    missing.append(f'{dataset}/{seed}/{stage}'); continue
                config=report['configuration']
                if config['smoke'] or config['dataset']!=dataset or config['seed']!=seed:
                    raise ValueError(f'not a matching formal result: {path}')
                row={'dataset':dataset,'seed':seed,'method':stage,
                    **{metric:record['test'][metric] for metric in ('mae','rmse','mse')},
                    'epochs':record['epochs'],'best_epoch':record['best_epoch'],
                    'stop_reason':record['stop_reason'],'test_windows':record['test']['windows']}
                rows.append(row); per_stage[stage].append(row)
        for stage,values in per_stage.items():
            summary={'dataset':dataset,'method':stage,'n_seeds':len(values),'complete':len(values)==3}
            for metric in ('mae','rmse','mse'):
                series=[row[metric] for row in values]
                summary[metric]={'mean':statistics.mean(series) if series else None,
                    'sample_std':statistics.stdev(series) if len(series)>1 else None}
            summaries.append(summary)
    return {'status':'complete' if not missing else 'incomplete','per_seed':rows,'summary':summaries,'missing':missing,
        'note':'Original paper numbers are separate references, not locally reproduced seeds. Extra periodic history is disclosed.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--stages',nargs='+',choices=STAGES,default=STAGES,
                        help='Use --stages baseline for baseline-only experiments.')
    args=parser.parse_args(); result=summarize(args.results,args.datasets,args.stages)
    (args.results/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.results/'per_seed.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        columns=('dataset','seed','method','mae','rmse','mse','epochs','best_epoch','stop_reason','test_windows')
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(result['per_seed'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
