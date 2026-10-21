"""Paper datasets and chronological 7:1:2 targets; no resampling/imputation."""
from dataclasses import replace
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from research.traffic_data import load_pems03

SPECS={
    'bike_inflow':{'folder':'Bike','csv':'inflow.csv','adj':'adj.csv','nodes':295,'period':24,'epochs':50,'lr':1e-4,'d_ff':32,'effective_batch':16,'paper':[5.18,14.40]},
    'bike_outflow':{'folder':'Bike','csv':'outflow.csv','adj':'adj.csv','nodes':295,'period':24,'epochs':50,'lr':1e-4,'d_ff':32,'effective_batch':16,'paper':[5.33,14.23]},
    'pems03':{'folder':'PEMS03','csv':'PEMS03.csv','adj':'PEMS03_adj.csv','nodes':358,'period':288,'epochs':30,'lr':3e-4,'d_ff':128,'effective_batch':32,'paper':[21.03,34.17]},
    'bj500':{'folder':'BJ','csv':'BJ.csv','adj':'BJ_adj.csv','nodes':500,'period':288,'epochs':30,'lr':3e-4,'d_ff':128,'effective_batch':32,'paper':[5.58,9.77]},
    'metr_la':{'folder':'METR','csv':'METR.csv','adj':'METR_adj.csv','nodes':207,'period':288,'epochs':30,'lr':3e-4,'d_ff':128,'effective_batch':32,'paper':[9.42,16.41]},
}


def load_dataset(root,name):
    spec=SPECS[name]; path=Path(root)/spec['folder']/spec['csv']
    bundle=load_pems03(path,.7,.1)
    if bundle.raw.shape[1]!=spec['nodes']: raise ValueError('paper node count mismatch')
    total=len(bundle.raw); train=int(.7*total); test=int(.2*total)
    # Match author's code rounding, not floor(.1*N) for validation.
    bundle=replace(bundle,split_bounds={'train':(0,train),'val':(train,total-test),'test':(total-test,total)})
    difference=np.diff(bundle.timestamps_ns); expected=int(86400e9/spec['period'])
    if np.any(difference<=0) or int(np.median(difference))!=expected:
        raise ValueError('timestamps must increase at the expected nominal interval')
    graph=pd.read_csv(Path(root)/spec['folder']/spec['adj']).to_numpy(np.float32)
    if graph.shape!=(spec['nodes'],spec['nodes']) or not np.isfinite(graph).all():
        raise ValueError('invalid adjacency matrix')
    audit={'rows':total,'nodes':spec['nodes'],'interval_minutes':1440/spec['period'],
        'csv_bytes':path.stat().st_size,'csv_mtime_ns':path.stat().st_mtime_ns,
        'adj_bytes':(Path(root)/spec['folder']/spec['adj']).stat().st_size,
        'adj_mtime_ns':(Path(root)/spec['folder']/spec['adj']).stat().st_mtime_ns,
        'split_bounds':{key:list(value) for key,value in bundle.split_bounds.items()},'timestamp_gap_count':int((difference!=expected).sum()),
        'start':str(pd.Timestamp(bundle.timestamps_ns[0])),'end':str(pd.Timestamp(bundle.timestamps_ns[-1])),
        'gap_policy':'Original observed-row sequence preserved. Period offsets mean observed daily cycles, not asserted calendar-day lags when dates are missing.',
        'all_values_finite':True,'zeros':int((bundle.raw==0).sum())}
    return bundle,audit


def origins_for(bundle,role,history):
    lower,upper=bundle.split_bounds[role]
    return np.arange(max(lower,history),upper-48+1,dtype=np.int64)


class CourseworkWindows(torch.utils.data.Dataset):
    def __init__(self,bundle,role,history):
        self.bundle=bundle; self.history=history; self.origins=origins_for(bundle,role,history)
        if not len(self.origins): raise ValueError('empty chronological target split')
    def __len__(self): return len(self.origins)
    def __getitem__(self,index):
        origin=int(self.origins[index]); b=self.bundle
        return {'history':torch.from_numpy(b.normalized[origin-self.history:origin].copy()),
            'target':torch.from_numpy(b.normalized[origin:origin+48].copy()),
            'history_marks':torch.from_numpy(b.marks[origin-48:origin].copy()),
            'future_marks':torch.from_numpy(b.marks[origin:origin+48].copy()),
            'origin_step':torch.tensor(origin,dtype=torch.long),
            'node_std':torch.from_numpy(b.std.copy())}
