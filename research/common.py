"""Shared deterministic loading, precision and report I/O. No model imports."""
import contextlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def json_ready(value):
    if isinstance(value,dict):return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_ready(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    return value


def write_json(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_ready(value),ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)


def loader(dataset,batch,shuffle,seed):
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,drop_last=False,num_workers=0,
                      generator=torch.Generator().manual_seed(seed),pin_memory=torch.cuda.is_available())


def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()


def move(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}

