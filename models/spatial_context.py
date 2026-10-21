"""Shared current/historical relational context, with explicit content retention.

Only observed context patches enter this module. Candidate continuations and
forecast targets never enter its keys. No sensor/city IDs or direction labels.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialContext(nn.Module):
    def __init__(self,width=32,dropout=.1):
        super().__init__()
        self.relation_key=nn.Linear(4,width,bias=False)
        self.time_key=nn.Parameter(torch.empty(4,width))
        nn.init.normal_(self.time_key,std=.02)
        self.attention=nn.MultiheadAttention(width,4,dropout=dropout,batch_first=True)
        self.spatial_gate=nn.Parameter(torch.tensor(-2.))
        self.bert_gate=nn.Parameter(torch.tensor(-2.))

    def forward(self,target,source,valid,address):
        """target[R,4,D], source[R,K,4,D]; source K/V exclude target roads.

        The SAME function is used before encoding current query and historical
        candidates. Attention is target-patch dependent, not one scalar shared
        over every time patch. Relation/time addresses affect K, not source V.
        """
        r,k,p,d=source.shape
        if target.shape!=(r,4,d) or p!=4 or valid.shape!=(r,k) or address.shape!=(r,k,4):
            raise ValueError("invalid aligned spatial context")
        mask=valid[:,:,None].expand(-1,-1,4).reshape(r,k*4)
        safe=mask.clone(); safe[:,0] |= ~safe.any(-1)
        content=F.layer_norm(source,(d,))
        keys=content+self.relation_key(address)[:,:,None]+self.time_key[None,None]
        query=F.layer_norm(target,(d,))+self.time_key[None]
        read,_=self.attention(query,keys.flatten(1,2),content.flatten(1,2),
            key_padding_mask=~safe,need_weights=False)
        read=read*mask.any(-1)[:,None,None]
        return target+self.spatial_gate.sigmoid()*read

    def preserve(self,encoded,content):
        """Retain observed content; BERT contributes a learned bounded residual.

        This is NOT evidence of language-pretraining utility. Its necessity is
        tested against the same model without this content-retention path.
        """
        return F.layer_norm(content,(content.shape[-1],))+self.bert_gate.sigmoid()*F.layer_norm(encoded,(encoded.shape[-1],))
