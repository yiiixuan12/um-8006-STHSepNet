"""Independent architecture pilot: cheap shortlist, compact BERT memory.

Not an equivalent execution optimization and NOT a drop-in formal checkpoint.
No new learned parameters: reuse the original embeddings, LoRA, retrieval and
readout. Keep raw 48-step historical Values and all35 utility/abstain logits.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from models.periodic_bert_v4 import PeriodicBertV4


class FastPeriodicBert(PeriodicBertV4):
    history_length=2064
    def __init__(self,*args,shortlist=4,**kwargs):
        if shortlist<1 or shortlist>33:raise ValueError('shortlist must be in1..33')
        super().__init__(*args,**kwargs)
        if self.repair!='utility' or self.use_neighbors or self.memory_policy!='no_deviation':
            raise ValueError('pilot only supports utility/no_deviation/no-neighbors')
        self.shortlist=shortlist

    @staticmethod
    def compact_memory(memory):
        """Keep16 recent+8 daily, pool adjacent pairs within each day, repeat4.

        Each historical day retains TWO ordered context and TWO ordered outcome
        summaries. No pooling across days, roads, or context/outcome boundary.
        The original raw continuations remain in the separate Value operator.
        """
        count=memory.shape[0]
        anchors=memory[:,24:80].reshape(count,7,4,2,32).mean(3).flatten(1,2)
        return torch.cat((memory[:,:24],anchors,memory[:,80:84]),1)

    def _fast_context_keys(self,history,short,origins):
        raw=history[0].T;count=len(raw)
        candidates,valid,indices=self.pattern_operator.candidates(raw)
        # Only CONTEXT first48 samples: no outcome enters cheap/fine ranking.
        contexts=candidates[...,:48].reshape(count*35,4,12)
        relative=torch.arange(4,device=raw.device)*12-42
        current=self.patch_encoder(raw[:,-48:].reshape(count,4,12))+self._time(relative,12,origins)
        candidate_origins=(origins[:,None]+indices.reshape(35,96)[None,:,0]+48-self.history_length).flatten()
        keys=self.patch_encoder(contexts)+self._time(relative,12,candidate_origins)
        cheap_query=self.utility_context_pool(current)
        cheap_keys=self.utility_context_pool(keys).reshape(count,35,32)
        state=F.layer_norm(cheap_query+self.short_projection(F.layer_norm(short,(128,))),(32,))
        query=F.layer_norm(self.pattern_operator.query(state),(32,))[:,None]+self.pattern_operator.horizons[None]
        key=F.layer_norm(self.pattern_operator.key(cheap_keys),(32,))
        scores=torch.einsum('bhd,bkd->bhk',query.float(),key.float()).mean(1)/math.sqrt(32)
        chosen=scores.masked_fill(~valid[None],-torch.inf).topk(self.shortlist,dim=-1).indices
        tokens=keys.reshape(count,35,4,32)
        selected=tokens.gather(1,chosen[:,:,None,None].expand(-1,-1,4,32))
        # Current and shortlisted contexts use the same lower BERT interface.
        fine_tokens=torch.cat((current[:,None],selected),1).flatten(0,1)
        hidden=self.bert.embeddings(inputs_embeds=self.memory_projection(fine_tokens))
        hidden=self._layers(hidden,torch.ones(hidden.shape[:2],device=raw.device,dtype=torch.bool),0,self.split_layer)
        fine=self.utility_context_pool(self.bert_output_projection(hidden)).reshape(count,self.shortlist+1,32)
        refined_query=F.layer_norm(cheap_query+fine[:,0],(32,))
        refined_keys=cheap_keys.scatter_add(1,chosen[:,:,None].expand(-1,-1,32),fine[:,1:])
        return refined_query,refined_keys,chosen

    def _chunk(self,history,short,origins,neighbor_history,neighbor_valid,relation,*,
            anchor_shift_steps=0,component_ablation=None,return_diagnostics=False,
            baseline_prediction=None,reference_prediction=None,pattern_mode='normal',
            return_auxiliary=False,neighbor_long_history=None):
        if anchor_shift_steps or component_ablation is not None:raise ValueError('unsupported pilot intervention')
        memory,weights=self._memory(history,F.layer_norm(short,(128,)),origins)
        memory=self.compact_memory(memory);count,own_count=memory.shape[:2]
        own=self.bert.embeddings(inputs_embeds=self.memory_projection(memory))
        valid=torch.ones(count,own_count,dtype=torch.bool,device=history.device)
        groups=torch.repeat_interleave(torch.arange(10,device=history.device),
            torch.tensor([16,8]+[4]*7+[4],device=history.device))
        own=self._layers(own,valid,0,self.split_layer,allowed=groups[:,None]==groups[None,:])
        own=self._layers(own,valid,self.split_layer,self.num_layers)
        values=self.bert_output_projection(own)
        def norm(value):return F.layer_norm(value,(32,))
        relative=torch.arange(48,device=history.device)+.5
        address=norm(self.horizon_queries)[None]+norm(self._time(relative,1,origins))
        query=address+norm(self.short_projection(F.layer_norm(short,(128,))))[:,None]
        query=query+norm(self.forecast_query(baseline_prediction[...,None]))
        # Keep all48 horizon queries, but run them in32d, not BERT768d.
        read,attention=self.content_read(query,values,values,need_weights=return_diagnostics)
        state=norm(read)+address
        raw=(state*self.hourly_readout.weight[None]).sum(-1)+self.hourly_readout.bias
        info={};chosen=torch.empty(count,0,dtype=torch.long,device=history.device)
        if pattern_mode!='off':
            context_query,context_keys,chosen=self._fast_context_keys(history,short,origins)
            context_query=norm(context_query+self.short_projection(F.layer_norm(short,(128,))))
            template,info=self.pattern_operator(history[0].T,context_query,candidate_keys=context_keys,
                reference_prediction=reference_prediction,mode=pattern_mode)
            raw=raw.float()+self.pattern_gate.tanh()*(template-reference_prediction.float())
        if return_auxiliary and not return_diagnostics:return raw,{'utility_logits':info['utility_logits']}
        if return_diagnostics:
            return raw,{'retrieval_weights':weights,'read_attention':attention,
                'shortlist_indices':chosen,'memory_values':values,**info}
        return raw,None

    @property
    def metadata(self):
        return {**super().metadata,'interface_version':'periodic_bert_fast_pilot_v1',
            'architecture_changed':True,'bert_memory_tokens':56,'bert_forecast_query_tokens':0,
            'hierarchy_scope':'shared lower BERT within compact groups; upper BERT across56 memory tokens',
            'memory_ablation':{'policy':'no_deviation','raw_memory_slots':88,'encoded_memory_tokens':56,
                'compression':'ordered adjacent pairs within each historical day; no context/outcome mixing'},
            'numeric_attention_queries':48,'fine_candidate_count':self.shortlist,
            'coarse_candidates':35,'utility_classes':36,
            'raw_candidate_outcomes_preserved':True,'no_formal_checkpoint_migration':True}
