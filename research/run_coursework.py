"""Full chronological two-contribution evaluation, not a pilot.

One training entry for the corrected STH-SepNet48->48 baseline and its frozen
base + per-road periodic/utility adapter. All windows and all target nodes.
No neighbor-recoupling/Value replacement, no test-driven checkpoint selection.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EARLY_STOP_PATIENCE=5
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research.common import seed_all, loader, move, autocast, write_json
from research.coursework_data import SPECS, load_dataset, CourseworkWindows
from research.run_periodic_bert_repair import training_loss, utility_training_loss
from models.periodic_bert_v4 import PeriodicBertV4
from models.periodic_tokens import periodic_numeric_features

SOURCES=('research/run_coursework.py','research/coursework_data.py','research/coursework_config.json',
    'research/common.py','research/traffic_data.py','research/run_periodic_bert_repair.py',
    'models/sthsepnet_h1.py','layers/h1/STHGNN.py','layers/h1/HyperGNN.py','layers/h1/AdaGNN.py',
    'models/periodic_bert_v4.py','models/periodic_tokens.py',
    'models/periodic_memory_policy.py','models/bert_backbone.py','models/spatial_context.py',
    'models/periodic_refinement.py','models/utility_recoupling.py','models/holt_operator.py',
    'research/periodic_relation_data.py','research/traffic_evaluation.py',
    'layers/STHGNN.py','layers/AdaGNN.py','layers/HyperGNN.py','layers/GraphGCN.py','layers/HGCN.py',
    'layers/Embed.py','layers/FusionGate.py','layers/hypergraph.py','layers/StandardNorm.py')


def compact_state(model,epoch):
    # Frozen pretrained weights are recreated from the common local backbone.
    return {'epoch':epoch,'parameters':{n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
        'buffers':{n:v.detach().cpu().clone() for n,v in model.named_buffers()}}


def load_compact(model,state):
    parameters=dict(model.named_parameters()); buffers=dict(model.named_buffers())
    with torch.no_grad():
        for name,value in state['parameters'].items(): parameters[name].copy_(value.to(parameters[name].device))
        for name,value in state['buffers'].items(): buffers[name].copy_(value.to(buffers[name].device))
    if isinstance(model,PeriodicBertV4): model.numeric_calibration.ready=True


def base_prediction(base,batch,features=False):
    with autocast(torch.device('cuda')):
        return base(batch['history'][:,-48:],batch['history_marks'],None,batch['future_marks'],
            interaction_rank=0,return_features=features)


def sliced_adapter_batch(batch,prediction,short,start,stop):
    return {'history':batch['history'][:,:,start:stop], 'short_state':short[:,start:stop].float(),
        'base':prediction[:,:,start:stop].float(),'target':batch['target'][:,:,start:stop],
        'node_std':batch['node_std'][:,start:stop],'origin_step':batch['origin_step']}


def adapter_delta(model,batch,aux=False):
    with autocast(torch.device('cuda')):
        return model(batch['history'],batch['short_state'],baseline_prediction=batch['base'],
            origin_step=batch['origin_step'],return_auxiliary=aux)


def solve_moments(sx,sy,xx,xy,n,alpha):
    mean=sx/n; scale=np.sqrt(np.maximum(np.diag(xx)/n-mean**2,0)).clip(.01)
    gram=(xx-n*np.outer(mean,mean))/np.outer(scale,scale)
    rhs=(xy-mean[:,None]*sy)/scale[:,None]
    return mean,scale,np.linalg.solve(gram+alpha*np.eye(len(sx)),rhs),sy/n


def early_stop_reached(stale):
    return stale>=EARLY_STOP_PATIENCE


@torch.no_grad()
def evaluate(base,student,dataset,args,arrays_path=None):
    base.eval()
    if student is not None: student.eval()
    absolute=[]; squared=[]; horizon_abs=np.zeros(48); horizon_sq=np.zeros(48); points=0
    tick=time.perf_counter()
    for cpu in loader(dataset,args.batch_size,False,args.seed):
        batch=move(cpu,'cuda'); count=batch['target'].shape[-1]
        if student is None: prediction=base_prediction(base,batch).float()
        else:
            prediction,info=base_prediction(base,batch,True); pieces=[]
            for start in range(0,count,args.target_chunk):
                stop=min(count,start+args.target_chunk)
                current=sliced_adapter_batch(batch,prediction,info['short_state'],start,stop)
                pieces.append(current['base']+adapter_delta(student,current).float())
            prediction=torch.cat(pieces,-1)
        error=(prediction-batch['target'])*batch['node_std'][:,None]
        if not torch.isfinite(error).all(): raise RuntimeError('nonfinite evaluation')
        absolute.extend(error.abs().mean((1,2)).cpu().tolist()); squared.extend(error.square().mean((1,2)).cpu().tolist())
        horizon_abs+=error.abs().sum((0,2)).cpu().numpy(); horizon_sq+=error.square().sum((0,2)).cpu().numpy()
        points+=error.shape[0]*count
    mse=float(np.mean(squared))
    result={'mae':float(np.mean(absolute)),'mse':mse,'rmse':mse**.5,'windows':len(dataset),
        'horizon_mae':(horizon_abs/points).tolist(),'horizon_rmse':np.sqrt(horizon_sq/points).tolist(),
        'seconds':time.perf_counter()-tick,'all_nodes':True,'mask':'none, matches unmasked author metrics'}
    if arrays_path is not None: np.savez(arrays_path,origin=dataset.origins,window_mae=absolute,window_mse=squared)
    return result


@torch.no_grad()
def fit_numeric(base,datasets,args):
    """Streaming ridge moments on ALL training road/window pairs; Val selects alpha."""
    n=0; sx=np.zeros(252); sy=np.zeros(4); xx=np.zeros((252,252)); xy=np.zeros((252,4))
    base.eval()
    for step,cpu in enumerate(loader(datasets['train'],args.batch_size,False,args.seed)):
        batch=move(cpu,'cuda'); pred,info=base_prediction(base,batch,True)
        x=periodic_numeric_features(batch['history'],info['short_state'].float(),pred.float(),memory_policy='no_deviation').reshape(-1,252).double().cpu()
        y=(batch['target']-pred.float()).transpose(1,2).reshape(-1,4,12).mean(-1).double().cpu()
        n+=len(x); sx+=x.sum(0).numpy(); sy+=y.sum(0).numpy()
        xx+=(x.T@x).numpy(); xy+=(x.T@y).numpy()
        if (step+1)%128==0: print(json.dumps({'phase':'numeric_fit','batch':step+1,'pairs':n}),flush=True)
    alphas=(.1,1.,10.,100.,1000.,10000.); weights=[]; bias=sy/n
    for alpha in alphas:
        mean,scale,weight,bias=solve_moments(sx,sy,xx,xy,n,alpha); weights.append(weight)
    weights=np.stack(weights); cost=np.zeros(len(alphas)); seen=0
    for cpu in loader(datasets['val'],args.batch_size,False,args.seed):
        batch=move(cpu,'cuda'); pred,info=base_prediction(base,batch,True)
        x=periodic_numeric_features(batch['history'],info['short_state'].float(),pred.float(),memory_policy='no_deviation').reshape(-1,252).cpu().numpy()
        delta=np.einsum('rd,adk->ark',(x-mean)/scale,weights)+bias[None,None]
        true=(batch['target']-pred.float()).transpose(1,2).reshape(-1,4,12).cpu().numpy()
        std=batch['node_std'].flatten().cpu().numpy()
        error=(delta[...,None]-true[None])*std[None,:,None,None]
        cost+=np.square(error).sum((1,2,3)); seen+=error.shape[1]*48
    chosen=int(np.argmin(cost)); state={'mean':torch.tensor(mean,dtype=torch.float32),'scale':torch.tensor(scale,dtype=torch.float32),
        'weight':torch.tensor(weights[chosen].T,dtype=torch.float32),'bias':torch.tensor(bias,dtype=torch.float32)}
    return state,{'train_pairs':n,'alphas':list(alphas),'validation_mse':(cost/seen).tolist(),
        'selected_alpha':alphas[chosen],'selection':'validation raw MSE only; no test','streaming':True}


def train_stage(base,student,datasets,args,report,output,epochs):
    stage='baseline' if student is None else 'improved'; model=base if student is None else student
    if student is None:
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=SPECS[args.dataset]['lr'])
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,20,40,70],gamma=.3)
        accumulation=SPECS[args.dataset]['effective_batch']//args.batch_size
    else:
        groups=[{'params':[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('bert.')==backbone],
            'lr':1e-4 if backbone else 5e-4} for backbone in (False,True)]
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4); scheduler=None; accumulation=1
    last=output/f'{stage}_last.pt'; best=output/f'{stage}_best.pt'
    log=[]; epoch=0; stale=0; best_value=float('inf'); best_epoch=0
    if args.resume and last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False); load_compact(model,state)
        optimizer.load_state_dict(state['optimizer']); epoch=state['epoch']; log=state['log']; stale=state['stale']
        best_value=state['best_value']; best_epoch=state['best_epoch']
        if scheduler is not None: scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    reason='epoch_cap'
    while epoch<epochs:
        if early_stop_reached(stale): reason='early_stop'; break
        model.train()
        if student is not None: base.eval()
        optimizer.zero_grad(set_to_none=True); batches=loader(datasets['train'],args.batch_size,True,args.seed*100+epoch)
        tick=time.perf_counter(); total=0.; seen=0
        for step,cpu in enumerate(batches):
            batch=move(cpu,'cuda'); batch_count=len(batch['history']); group=min(accumulation,len(batches)-(step//accumulation)*accumulation)
            if student is None:
                prediction=base_prediction(base,batch)
                loss=(prediction.float()-batch['target']).square().mean(); (loss/group).backward()
                objective=float(loss.detach())
            else:
                with torch.no_grad(): prediction,info=base_prediction(base,batch,True)
                nodes=batch['target'].shape[-1]; objective=0.
                for start in range(0,nodes,args.target_chunk):
                    stop=min(nodes,start+args.target_chunk); current=sliced_adapter_batch(batch,prediction,info['short_state'],start,stop)
                    delta,aux=adapter_delta(student,current,True)
                    loss=training_loss(current,delta,'step_raw_scaled_mae_mse',scale=args.raw_scale)
                    teacher,_=utility_training_loss(student,current,aux['utility_logits'],loss_scale=args.raw_scale)
                    loss=loss+.1*teacher
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite improved objective')
                    (loss*((stop-start)/nodes)).backward(); objective+=float(loss.detach())*(stop-start)/nodes
            if not np.isfinite(objective): raise RuntimeError('nonfinite objective')
            if (step+1)%accumulation==0 or step+1==len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total+=objective*batch_count; seen+=batch_count
            if (step+1)%64==0:
                progress={'stage':stage,'epoch':epoch+1,'batch':step+1,'batches':len(batches),'train_objective':total/seen,
                    'seconds_so_far':time.perf_counter()-tick,'early_stop_patience':EARLY_STOP_PATIENCE}
                write_json(output/'progress.json',progress); print(json.dumps(progress),flush=True)
        train_seconds=time.perf_counter()-tick
        epoch+=1
        if scheduler is not None: scheduler.step()
        metrics=evaluate(base,student,datasets['val'],args)
        if metrics['mse']<best_value-1e-6:
            best_value=metrics['mse']; best_epoch=epoch; stale=0; torch.save(compact_state(model,epoch),best)
        else: stale+=1
        row={'epoch':epoch,'train_objective':total/seen,'val_mae':metrics['mae'],'val_rmse':metrics['rmse'],
            'val_mse':metrics['mse'],'seconds':time.perf_counter()-tick,'train_seconds':train_seconds,
            'validation_seconds':metrics['seconds'],'best_epoch':best_epoch,
            'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'early_stop_patience':EARLY_STOP_PATIENCE,'stale_epochs':stale}
        log.append(row)
        state={**compact_state(model,epoch),'optimizer':optimizer.state_dict(),'scheduler':None if scheduler is None else scheduler.state_dict(),
            'log':log,'stale':stale,'best_value':best_value,'best_epoch':best_epoch,'rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'early_stop_patience':EARLY_STOP_PATIENCE}
        torch.save(state,last); report['active']={'stage':stage,**row}; write_json(output/'report.json',report)
        print(json.dumps({'stage':stage,**row}),flush=True)
    load_compact(model,torch.load(best,map_location='cpu',weights_only=False))
    return {'status':'completed_training','epochs':epoch,'best_epoch':best_epoch,'stop_reason':reason,
        'early_stop_patience':EARLY_STOP_PATIENCE,
        'convergence_confirmed':reason=='early_stop','training_log':log,'train_windows':len(datasets['train']),
        'all_nodes_in_each_training_window':True,'validation':evaluate(base,student,datasets['val'],args)}


def run(args):
    output=args.output.resolve(); existed=output.exists(); report_path=output/'report.json'
    if existed and not (args.resume or args.stage=='improved'): raise FileExistsError('choose new output or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    configuration={'dataset':args.dataset,'seed':args.seed,'batch_size':args.batch_size,'target_chunk':args.target_chunk,
        'smoke':args.smoke,'bert_path':str(args.bert_path.resolve()),'protocol_version':'coursework_two_v1_patience5',
        'early_stop_patience':EARLY_STOP_PATIENCE}
    if existed:
        report=json.loads(report_path.read_text())
        if report['configuration']!=configuration: raise ValueError('resume protocol mismatch')
        for source in SOURCES:
            if (ROOT/source).read_bytes()!=(output/'source_snapshot'/source).read_bytes(): raise ValueError('source changed; refuse silent resume')
    else:
        report={'status':'preparing','configuration':configuration,'stages':{},'process_id':os.getpid(),
            'protocol':{'split':'author rounding: floor70% train, floor20% test, remainder validation',
                'baseline_input':48,'improved_history':2064,'prediction_horizon':48,'test_selection':False,
                'normalization':'train70% only','all_training_windows':not args.smoke,'all_target_nodes':True,
                'baseline':'local corrected STH-SepNet, real12 pretrained BERT layers; not unmodified buggy author scripts requesting32 layers',
                'improvement':'frozen per-seed baseline + existing6-layer periodic BERT adapter; no neighbor additions, no shape constraint, no explicit deviation features',
                'extra_information_disclosed':True,'checkpoint_selection':'validation raw MSE only',
                'early_stop_patience':EARLY_STOP_PATIENCE,
                'baseline_loss':'normalized MSE, Adam, original multstep5/20/40/70 gamma0.3',
                'student_loss':'existing scaled raw MAE+0.1MSE and0.1 training-only utility CE; AdamW',
                'metrics':'unmasked raw-unit MAE, global RMSE, MSE; no clipping; all windows/48 horizons/all roads',
                'reference_table':'https://arxiv.org/html/2505.19620v1#S4.T1'}}
        for source in SOURCES:
            target=output/'source_snapshot'/source; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/source,target)
    bundle,audit=load_dataset(ROOT/'dataset',args.dataset)
    if existed and audit!=report['dataset_audit']: raise ValueError('dataset file or chronology changed since baseline run')
    report['dataset_audit']=audit
    args.raw_scale=float(np.median(bundle.std)); spec=SPECS[args.dataset]
    config=json.loads((ROOT/'research/coursework_config.json').read_text())
    config.update(root_path=str(ROOT/'dataset'/spec['folder']),adjacency_path=spec['adj'],node_num=spec['nodes'],
        llm_path=str(args.bert_path.resolve()),min_values=bundle.train_min.tolist(),max_values=bundle.train_max.tolist(),
        data={'pems03':'PEMS03','metr_la':'METR','bj500':'BJ',
              'bike_inflow':'inflow','bike_outflow':'outflow'}[args.dataset],d_ff=spec['d_ff'],enc_in=spec['nodes'])
    if spec['period'] != 288:
        config['prompt_domain']=1
        config['content']=(f"{args.dataset} bicycle demand at {spec['nodes']} stations, "
                           'sampled hourly. Timestamps are calendar features, not event labels.')
    report['base_model_config']=config; report['process_id']=os.getpid(); write_json(report_path,report)
    seed_all(args.seed); torch.set_num_threads(4)
    from models.sthsepnet_h1 import Model
    base=Model(SimpleNamespace(**config)).cuda()
    report['baseline_parameters']={'total':sum(p.numel() for p in base.parameters()),
        'trainable':sum(p.numel() for p in base.parameters() if p.requires_grad)}
    datasets={role:CourseworkWindows(bundle,role,48) for role in ('train','val','test')}
    long_sets={role:CourseworkWindows(bundle,role,2064) for role in ('train','val','test')}
    if args.smoke:
        for groups in (datasets,long_sets):
            for ds in groups.values(): ds.origins=ds.origins[:args.batch_size]
    epochs=1 if args.smoke else spec['epochs']
    if 'baseline' not in report['stages']:
        if args.stage=='improved': raise ValueError('train corresponding full baseline first')
        report['status']='training_baseline'; write_json(report_path,report)
        row=train_stage(base,None,datasets,args,report,output,epochs)
        row['test']=evaluate(base,None,datasets['test'],args,output/'baseline_test.npz')
        report['stages']['baseline']=row; report['status']='baseline_completed'; report.pop('active',None); write_json(report_path,report)
    else: load_compact(base,torch.load(output/'baseline_best.pt',map_location='cpu',weights_only=False))
    if args.stage=='baseline': return
    base.eval().requires_grad_(False)
    if 'improved' in report['stages']: return
    seed_all(args.seed)
    student=PeriodicBertV4(args.bert_path,repair='utility',lora_rank=16,hierarchical_memory=True,
        memory_policy='no_deviation',use_neighbors=False,node_chunk_size=args.batch_size*args.target_chunk,
        gradient_checkpointing=False).cuda()
    report['adapter_parameters']={'total':sum(p.numel() for p in student.parameters()),
        'trainable':sum(p.numel() for p in student.parameters() if p.requires_grad)}
    numeric_path=output/'numeric_calibration.pt'
    if numeric_path.exists(): numeric=torch.load(numeric_path,map_location='cpu',weights_only=False)
    else:
        report['status']='fitting_train_only_numeric_route'; write_json(report_path,report)
        state,numeric_report=fit_numeric(base,long_sets,args); numeric={'state':state,'report':numeric_report}; torch.save(numeric,numeric_path)
    student.numeric_calibration.install(numeric['state']); report['numeric_fit']=numeric['report']
    report['status']='training_improved'; write_json(report_path,report)
    row=train_stage(base,student,long_sets,args,report,output,epochs)
    row['test']=evaluate(base,student,long_sets['test'],args,output/'improved_test.npz')
    if not np.array_equal(datasets['test'].origins,long_sets['test'].origins): raise ValueError('test origins must match baseline')
    reference=report['stages']['baseline']['test']
    row['improvement_percent']={key:100*(1-row['test'][key]/reference[key]) for key in ('mae','rmse','mse')}
    report['stages']['improved']=row; report['status']='smoke_completed_not_accuracy_evidence' if args.smoke else 'completed_full_pair'
    report.pop('active',None); write_json(report_path,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=tuple(SPECS),required=True)
    parser.add_argument('--seed',type=int,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('baseline','improved','pair'),default='pair')
    parser.add_argument('--bert-path',type=Path,default=ROOT/'huggingface/google_bert')
    parser.add_argument('--batch-size',type=int,default=4); parser.add_argument('--target-chunk',type=int,default=64)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.batch_size<1 or args.target_chunk<1 or SPECS[args.dataset]['effective_batch']%args.batch_size:
        parser.error('batch must divide baseline effective batch; target chunk must be positive')
    try: run(args)
    except Exception:
        if args.output.exists(): write_json(args.output/'failure.json',{'traceback':traceback.format_exc()})
        raise


if __name__=='__main__': main()
