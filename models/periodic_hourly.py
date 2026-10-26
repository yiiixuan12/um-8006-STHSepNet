"""Hourly sampling adapter for the SAME V4 layers, parameter names and losses.

BIKE forecasts48 hourly observations. Its seven complete analogue outcomes must
therefore start at least48 hours ago: use lags2..8 days, not unavailable day1.
No resampling, interpolation, added parameters, neighbors or future inputs.
Five-minute formal runs never import/use this adapter.
"""
from __future__ import annotations
import math
import torch
from models.periodic_bert_v4 import PeriodicBertV4
from models.periodic_tokens import UtilityPeriodicRetrieval,utility_cross_entropy
from models.periodic_memory_policy import mask_numeric_features,memory_token_mask

PERIOD=24
LAGS=tuple(range(2,9))
HISTORY=240  # 8*24 +48 context; contains all zero-shift context/outcome pairs.
SHIFTS=(-2,-1,0,1,2)  # Five distinct observation offsets at hourly resolution.

def hourly_candidates(history):
    if history.ndim!=2 or history.shape[1]!=HISTORY:
        raise ValueError('hourly history must be [roads,240]')
    lag=torch.tensor(LAGS,device=history.device)*PERIOD
    start=HISTORY-lag[:,None]+torch.tensor(SHIFTS,device=history.device)[None]-48
    indices=start[...,None]+torch.arange(96,device=history.device)
    valid=(indices[...,0]>=0)&(indices[...,-1]<HISTORY)
    # Clamping only permits safe masked gathering; invalid candidates cannot
    # contribute K/V weights or teacher labels, and no target is supplied here.
    return history.float()[:,indices.clamp(0,HISTORY-1)].flatten(1,2),valid.flatten(),indices

def hourly_numeric_features(history,short_state,baseline_prediction,*,memory_policy='no_deviation',
                            anchor_shift_steps=0,component_ablation=None):
    b,length,n=history.shape
    if length!=HISTORY or short_state.shape!=(b,n,128) or baseline_prediction.shape!=(b,48,n):
        raise ValueError('invalid hourly numeric interface')
    if anchor_shift_steps or component_ablation is not None or memory_policy!='no_deviation':
        raise ValueError('hourly formal adapter supports the frozen no_deviation policy only')
    values=history.float().transpose(1,2);recent=values[...,-48:]
    day=values[...,-PERIOD:].reshape(b,n,8,3).mean(-1)
    base=baseline_prediction.float().transpose(1,2).reshape(b,n,4,12).mean(-1)
    anchors=torch.stack([values[...,HISTORY-PERIOD*lag-48:HISTORY-PERIOD*lag+48].reshape(b,n,8,12).mean(-1)
                         for lag in LAGS],2)
    context=recent.reshape(b,n,4,12).mean(-1)
    distance=(anchors[...,:4]-context[:,:,None]).square().mean(-1)
    weights=(-distance/.1).softmax(-1)
    repeat=(weights[...,None]*anchors[...,4:]).sum(2)
    deviation=context-(weights[...,None]*anchors[...,:4]).sum(2)
    return mask_numeric_features(torch.cat((short_state.float(),recent,day,base,anchors.flatten(2),repeat,deviation),-1),memory_policy)

@torch.no_grad()
def hourly_teacher(history,target,reference,node_std,*,loss_scale,temperature=.025,improvement_margin=.005):
    if loss_scale<=0 or temperature<=0 or improvement_margin<0:
        raise ValueError('invalid utility teacher calibration')
    if target.shape!=reference.shape or target.shape!=(len(history),48):
        raise ValueError('utility target/reference shape mismatch')
    candidates,valid,_=hourly_candidates(history)
    future=candidates[...,48:].reshape(len(history),35,4,12).permute(0,2,1,3)
    truth=target.float().reshape(len(history),4,12);ref=reference.float().reshape(len(history),4,12)
    errors=(future-truth[:,:,None])*node_std[:,None,None,None]/loss_scale
    costs=errors.abs().mean(-1)+.1*errors.square().mean(-1)
    ref_errors=(ref-truth)*node_std[:,None,None]/loss_scale
    improvement=(ref_errors.abs().mean(-1)+.1*ref_errors.square().mean(-1))[...,None]-costs
    scores=(improvement/temperature).masked_fill(~(valid[None,None]&(improvement>improvement_margin)),-1e4)
    return torch.cat((scores,torch.zeros_like(scores[...,:1])),-1).softmax(-1).detach()

def hourly_utility_loss(model,batch,logits,*,loss_scale):
    with torch.no_grad():
        numeric=model.numeric_calibration(hourly_numeric_features(batch['history'],batch['short_state'],batch['base']))
        reference=batch['base'].float()+numeric.repeat_interleave(12,-1).transpose(1,2)
        labels=hourly_teacher(batch['history'].transpose(1,2).reshape(-1,HISTORY),
            batch['target'].transpose(1,2).reshape(-1,48),reference.transpose(1,2).reshape(-1,48),
            batch['node_std'].reshape(-1),loss_scale=loss_scale)
    return utility_cross_entropy(logits,labels),float(labels[...,-1].mean())

class HourlyUtilityRetrieval(UtilityPeriodicRetrieval):
    candidates=staticmethod(hourly_candidates)
    shifts=SHIFTS

class HourlyPeriodicBertV4(PeriodicBertV4):
    history_length=HISTORY

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.repair!='utility' or self.memory_policy!='no_deviation' or self.use_neighbors:
            raise ValueError('hourly adapter retains only the authorized final utility/no-neighbor configuration')
        # Same tensors/names/order and RNG as the parent. Only indexing changes.
        with torch.random.fork_rng(devices=[]):
            operator=HourlyUtilityRetrieval(32)
            operator.load_state_dict(self.pattern_operator.state_dict())
        self.pattern_operator=operator

    def tokenize(self,history,*,anchor_shift_steps=0,day_permutation=None):
        if history.ndim!=3 or history.shape[1]!=HISTORY or not history.is_floating_point():
            raise ValueError('hourly history must be [B,240,N] floating point')
        if anchor_shift_steps or day_permutation is not None:
            raise ValueError('formal hourly tokenization uses fixed causal daily anchors')
        recent=self._patches(history[:,-48:],16)
        day=self._patches(history[:,-PERIOD:],8)
        current=self._patches(history[:,-48:],4)
        anchors=torch.stack([self._patches(history[:,HISTORY-PERIOD*lag-48:HISTORY-PERIOD*lag+48],8)
                             for lag in LAGS],2)
        return recent,day,current,anchors,None

    def _time(self,relative,spans,origins):
        relative=torch.as_tensor(relative,device=origins.device,dtype=torch.float32)
        spans=torch.as_tensor(spans,device=origins.device,dtype=torch.float32).expand_as(relative)
        absolute=origins.float()[:,None]+relative[None]
        daily=absolute*(2*math.pi/PERIOD);weekly=absolute*(2*math.pi/(7*PERIOD))
        address=torch.stack((relative[None].expand_as(absolute)/HISTORY,
            relative.abs().log1p()[None].expand_as(absolute)/math.log(HISTORY+1),
            spans[None].expand_as(absolute)/PERIOD,daily.sin(),daily.cos(),weekly.sin(),weekly.cos()),-1)
        return self.time_projection(address)

    def _memory(self,history,short,origins,*,anchor_shift_steps=0,component_ablation=None):
        if component_ablation is not None:raise ValueError('unsupported hourly ablation')
        recent,day,current,anchors,_=self.tokenize(history,anchor_shift_steps=anchor_shift_steps)
        count=history.shape[2]
        r=self.patch_encoder(recent).reshape(count,16,32)
        d=self.patch_encoder(day).reshape(count,8,32)
        a=self.patch_encoder(anchors).reshape(count,7,8,32)
        c=self.patch_encoder(current).reshape(count,4,32)
        weights=self._retrieval_weights(c,a,current.reshape(count,4,12),anchors.reshape(count,7,8,12),short)
        repeat=torch.einsum('bhd,bdhc->bhc',weights.to(a.dtype),a[:,:,4:])
        context=torch.einsum('bd,bdhc->bhc',weights.mean(1).to(a.dtype),a[:,:,:4])
        centers=torch.cat((torch.arange(16,device=history.device)*3-46.5,
            torch.arange(8,device=history.device)*3-22.5,
            (torch.arange(8,device=history.device)[None]*12-42-torch.tensor(LAGS,device=history.device)[:,None]*PERIOD).flatten(),
            torch.arange(4,device=history.device)*12+6,torch.arange(4,device=history.device)*12-42))
        spans=torch.tensor([3]*24+[12]*64,device=history.device)
        roles=torch.tensor([0]*16+[1]*8+([2]*4+[3]*4)*7+[4]*4+[5]*4,device=history.device)
        memory=torch.cat((r,d,a.flatten(1,2),repeat,c-context),1)
        memory=memory+self._time(centers,spans,origins)+self.role_embedding(roles)[None]
        return memory.masked_fill(~memory_token_mask(self.memory_policy,device=history.device)[None,:,None],0),weights

    def _strict_context_keys(self,history,short,origins,**kwargs):
        if kwargs:raise ValueError('hourly final model has no spatial source extensions')
        raw=history[0].T;count=len(raw)
        tokens=self.patch_encoder(raw[:,-48:].reshape(count,4,12))
        centers=torch.arange(4,device=raw.device)*12-42
        tokens=tokens+self._time(centers,12,origins)
        q=self.bert.embeddings(inputs_embeds=self.memory_projection(tokens))
        q=self._layers(q,torch.ones(count,4,dtype=torch.bool,device=raw.device),0,self.split_layer)
        query=self.utility_context_pool(self.bert_output_projection(q))
        candidates,valid,_=hourly_candidates(raw)
        keys=self.patch_encoder(candidates[...,:48].reshape(count*35,4,12))
        lag=torch.tensor(LAGS,device=raw.device)*PERIOD
        shifts=torch.tensor(SHIFTS,device=raw.device)
        key_origins=(origins[:,None]-lag.repeat_interleave(5)[None]+shifts.repeat(7)[None]).flatten()
        keys=keys+self._time(centers,12,key_origins)
        chosen=valid[None].expand(count,-1).flatten().nonzero(as_tuple=True)[0]
        hidden=self.bert.embeddings(inputs_embeds=self.memory_projection(keys[chosen]))
        hidden=self._layers(hidden,torch.ones(len(chosen),4,dtype=torch.bool,device=raw.device),0,self.split_layer)
        pooled=self.utility_context_pool(self.bert_output_projection(hidden))
        encoded=query.new_zeros(count*35,32).index_copy(0,chosen,pooled)
        return query,encoded.reshape(count,35,32)

    def forward(self,history,short_state,*,baseline_prediction=None,origin_step=None,
                return_auxiliary=False,return_diagnostics=False):
        if history.ndim!=3 or history.shape[1]!=HISTORY:raise ValueError('expected [B,240,N] hourly history')
        b,_,n=history.shape
        if short_state.shape!=(b,n,128) or baseline_prediction is None or baseline_prediction.shape!=(b,48,n):
            raise ValueError('invalid hourly short/baseline interface')
        if origin_step is None:origin_step=torch.zeros(b,device=history.device)
        if origin_step.shape!=(b,):raise ValueError('one origin per window required')
        numeric=self.numeric_calibration(hourly_numeric_features(history,short_state,baseline_prediction)).repeat_interleave(12,-1).transpose(1,2)
        reference=(baseline_prediction.float()+numeric).transpose(1,2).reshape(b*n,48)
        flat=history.transpose(1,2).reshape(b*n,HISTORY);short=short_state.reshape(b*n,128)
        base=baseline_prediction.transpose(1,2).reshape(b*n,48)
        origins=origin_step[:,None].expand(b,n).reshape(-1)
        predictions=[];diagnostics=[]
        for start in range(0,b*n,self.node_chunk_size):
            stop=min(start+self.node_chunk_size,b*n);count=stop-start
            value,info=self._chunk(flat[start:stop].T[None],short[start:stop],origins[start:stop],
                history.new_zeros(count,1,48),torch.zeros(count,1,dtype=torch.bool,device=history.device),
                history.new_zeros(count,1,4),baseline_prediction=base[start:stop],reference_prediction=reference[start:stop],
                return_diagnostics=return_diagnostics,return_auxiliary=return_auxiliary,pattern_mode=self.pattern_mode)
            predictions.append(value)
            if info is not None:diagnostics.append(info)
        output=torch.cat(predictions).reshape(b,n,48).transpose(1,2).float()+numeric
        if return_auxiliary or return_diagnostics:
            return output,{name:torch.cat([value[name] for value in diagnostics]) for name in diagnostics[0]}
        return output

    @property
    def metadata(self):
        return {**super().metadata,'sampling_adapter':'hourly_v1','period_steps':PERIOD,'history_steps':HISTORY,
                'horizon_steps':48,'analogue_day_lags':list(LAGS),'candidate_shifts':list(SHIFTS),
                'actual_hours':{'recent':48,'daily':24,'forecast':48},'history_resampled':False}
