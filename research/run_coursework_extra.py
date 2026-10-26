"""Append the other paper datasets without modifying frozen two-dataset sources.

Uses the existing optimizer/training/validation/checkpoint/evaluation functions.
Only dataset construction and the authorized hourly indexing adapter are new.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research import run_coursework as shared
from research.coursework_data import SPECS,load_dataset,CourseworkWindows
from research.common import seed_all,write_json
from models.periodic_bert_v4 import PeriodicBertV4
from models.periodic_hourly import HourlyPeriodicBertV4,hourly_numeric_features,hourly_utility_loss

DATASETS=('bj500','bike_inflow','bike_outflow')
SOURCES=shared.SOURCES+('research/run_coursework_extra.py','models/periodic_hourly.py')

def history_length(name):return 240 if SPECS[name]['period']==24 else 2064

def base_config(name,bundle,bert):
    spec=SPECS[name];config=json.loads((ROOT/'research/coursework_config.json').read_text())
    config.update(root_path=str(ROOT/'dataset'/spec['folder']),adjacency_path=spec['adj'],node_num=spec['nodes'],
        llm_path=str(bert.resolve()),min_values=bundle.train_min.tolist(),max_values=bundle.train_max.tolist(),
        data={'bj500':'BJ500','bike_inflow':'BIKE-Inflow','bike_outflow':'BIKE-Outflow'}[name],
        d_ff=spec['d_ff'],enc_in=spec['nodes'])
    if spec['period']==24:
        # The old default prose incorrectly says every dataset is five-minute.
        config.update(prompt_domain=1,content=f"{config['data']} bicycle station demand at295 New York stations, sampled hourly. "
            'Local timestamps are calendar/index features, not incident or holiday labels. '
            'The supplied station adjacency supplies spatial connectivity.')
    return config

@contextmanager
def sampling_helpers(name):
    # Dependency injection is isolated to THIS new process and restored for tests.
    # Existing running/future PEMS03 and METR-LA processes are unchanged.
    old_features=shared.periodic_numeric_features;old_loss=shared.utility_training_loss
    if SPECS[name]['period']==24:
        shared.periodic_numeric_features=hourly_numeric_features
        shared.utility_training_loss=hourly_utility_loss
    try:yield
    finally:
        shared.periodic_numeric_features=old_features;shared.utility_training_loss=old_loss

def prepare_data(name):
    bundle,audit=load_dataset(ROOT/'dataset',name)
    short={role:CourseworkWindows(bundle,role,48) for role in ('train','val','test')}
    long={role:CourseworkWindows(bundle,role,history_length(name)) for role in ('train','val','test')}
    if not np.array_equal(short['test'].origins,long['test'].origins):
        raise ValueError('paired test origins must be identical')
    audit['window_counts']={stage:{role:len(ds) for role,ds in sets.items()} for stage,sets in (('baseline',short),('improved',long))}
    return bundle,audit,short,long

def run(args):
    bundle,audit,short,long=prepare_data(args.dataset)
    config=base_config(args.dataset,bundle,args.bert_path)
    if args.check_data:
        print(json.dumps({'dataset':args.dataset,'data':audit,'history':history_length(args.dataset),
            'base_model_config':config,'early_stop_patience':shared.EARLY_STOP_PATIENCE,
            'epoch_cap':SPECS[args.dataset]['epochs'],'cuda_used':False}),flush=True)
        return
    output=args.output.resolve();path=output/'report.json';existed=path.exists()
    if output.exists() and not existed and any(output.iterdir()):raise FileExistsError('unrecognized nonempty output')
    if existed and not (args.resume or args.stage=='improved'):raise FileExistsError('explicit resume required')
    configuration={'dataset':args.dataset,'seed':args.seed,'batch_size':args.batch_size,'target_chunk':args.target_chunk,
        'smoke':args.smoke,'bert_path':str(args.bert_path.resolve()),'protocol_version':'coursework_extra_v1_patience5',
        'early_stop_patience':shared.EARLY_STOP_PATIENCE,'period_steps':SPECS[args.dataset]['period'],
        'improved_history':history_length(args.dataset)}
    if existed:
        report=json.loads(path.read_text())
        if report['configuration']!=configuration or report['dataset_audit']!=audit:
            raise ValueError('resume configuration or dataset changed')
        for source in SOURCES:
            if (ROOT/source).read_bytes()!=(output/'source_snapshot'/source).read_bytes():
                raise ValueError('source changed; refuse silent resume')
    else:
        output.mkdir(parents=True,exist_ok=True)
        report={'status':'preparing','configuration':configuration,'stages':{},'dataset_audit':audit,
            'protocol':{'split':'author floor70% train, floor20% test, remainder validation; targets cannot cross boundaries',
                'baseline_input':48,'prediction_horizon':48,'improved_history':history_length(args.dataset),
                'all_windows':not args.smoke,'all_nodes':True,'normalization':'per node, training segment only',
                'checkpoint_selection':'validation raw MSE','early_stop_patience':shared.EARLY_STOP_PATIENCE,
                'extra_information_disclosed':True,'test_selection':False,
                'baseline':'corrected original-structure12-layer BERT/LoRA, not paper-reported numbers',
                'improved':'same6-layer V4 periodic memory and utility retrieval; no neighbor additions',
                'metrics':'unmasked raw-unit MAE, global RMSE and MSE; no clipping',
                'period_steps':SPECS[args.dataset]['period'],
                'sampling_note':('hourly: recent48h/daily24h,7 complete analogue contexts at lags2..8 days, '
                    '48h outcomes,240h observed history,shifts[-2,-1,0,1,2]h; no resampling' if SPECS[args.dataset]['period']==24 else
                    'unchanged five-minute V4; observed-row cycles retained across timestamp gaps')},
            'paper_reference_only':dict(zip(('mae','rmse'),SPECS[args.dataset]['paper']))}
        for source in SOURCES:
            dest=output/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dest)
    report.update(base_model_config=config,process_id=os.getpid());write_json(path,report)
    args.raw_scale=float(np.median(bundle.std));seed_all(args.seed);torch.set_num_threads(4)
    from models.ST_SepNet import Model
    base=Model(SimpleNamespace(**config)).cuda()
    report['baseline_parameters']={'total':sum(p.numel() for p in base.parameters()),
        'trainable':sum(p.numel() for p in base.parameters() if p.requires_grad)}
    if args.smoke:
        for groups in (short,long):
            for ds in groups.values():ds.origins=ds.origins[:args.batch_size]
    epochs=1 if args.smoke else SPECS[args.dataset]['epochs']
    if 'baseline' not in report['stages']:
        if args.stage=='improved':raise ValueError('paired full baseline must finish first')
        report['status']='training_baseline';write_json(path,report)
        row=shared.train_stage(base,None,short,args,report,output,epochs)
        row['test']=shared.evaluate(base,None,short['test'],args,output/'baseline_test.npz')
        report['stages']['baseline']=row;report['status']='baseline_completed';report.pop('active',None);write_json(path,report)
    else:shared.load_compact(base,torch.load(output/'baseline_best.pt',map_location='cpu',weights_only=False))
    if args.stage=='baseline':return
    if 'improved' in report['stages']:return
    base.eval().requires_grad_(False);seed_all(args.seed)
    cls=HourlyPeriodicBertV4 if SPECS[args.dataset]['period']==24 else PeriodicBertV4
    student=cls(args.bert_path,repair='utility',lora_rank=16,hierarchical_memory=True,memory_policy='no_deviation',
        use_neighbors=False,node_chunk_size=args.batch_size*args.target_chunk,gradient_checkpointing=False).cuda()
    report['adapter_parameters']={'total':sum(p.numel() for p in student.parameters()),
        'trainable':sum(p.numel() for p in student.parameters() if p.requires_grad)}
    report['adapter_metadata']=student.metadata
    with sampling_helpers(args.dataset):
        numeric_path=output/'numeric_calibration.pt'
        if numeric_path.exists():numeric=torch.load(numeric_path,map_location='cpu',weights_only=False)
        else:
            report['status']='fitting_train_only_numeric_route';write_json(path,report)
            state,details=shared.fit_numeric(base,long,args);numeric={'state':state,'report':details};torch.save(numeric,numeric_path)
        student.numeric_calibration.install(numeric['state']);report['numeric_fit']=numeric['report']
        report['status']='training_improved';write_json(path,report)
        row=shared.train_stage(base,student,long,args,report,output,epochs)
        row['test']=shared.evaluate(base,student,long['test'],args,output/'improved_test.npz')
    reference=report['stages']['baseline']['test']
    row['improvement_percent']={key:100*(1-row['test'][key]/reference[key]) for key in ('mae','rmse','mse')}
    report['stages']['improved']=row
    report['status']='smoke_completed_not_accuracy_evidence' if args.smoke else 'completed_full_pair'
    report.pop('active',None);write_json(path,report)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('baseline','improved','pair'),default='pair')
    p.add_argument('--bert-path',type=Path,default=ROOT/'huggingface/google_bert')
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--target-chunk',type=int,default=64)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--check-data',action='store_true')
    args=p.parse_args()
    if args.batch_size<1 or args.target_chunk<1 or SPECS[args.dataset]['effective_batch']%args.batch_size:
        p.error('batch must divide effective batch; target chunk must be positive')
    try:run(args)
    except Exception:
        if args.output.exists():write_json(args.output/'failure.json',{'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
