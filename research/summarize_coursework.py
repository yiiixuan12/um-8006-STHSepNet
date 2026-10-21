"""Aggregate paired full-data runs, never substitute missing seeds with zero."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(root):
    rows=[]; missing=[]; summaries=[]
    for dataset in ('pems03','metr_la'):
        per_stage={stage:[] for stage in ('baseline','improved')}
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
    args=parser.parse_args(); result=summarize(args.results)
    args.results.mkdir(parents=True,exist_ok=True)
    (args.results/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.results/'per_seed.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        columns=('dataset','seed','method','mae','rmse','mse','epochs','best_epoch','stop_reason','test_windows')
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(result['per_seed'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
