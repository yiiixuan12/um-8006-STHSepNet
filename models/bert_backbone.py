"""Local-only BERT construction and zero-delta LoRA for the current V4 model."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        if not isinstance(base, nn.Linear) or isinstance(rank, bool) or int(rank) != rank or rank < 1:
            raise ValueError("LoRA requires a linear layer and positive integer rank")
        self.rank, self.in_features, self.out_features = int(rank), base.in_features, base.out_features
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_A), self.lora_B)


def load_bert(path, *, pretrained, layers=6, lora_rank=4, seed=2021):
    # Isolate both backbone construction and LoRA initialization from interface RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        from transformers import BertConfig, BertModel
        config = BertConfig.from_pretrained(str(path), local_files_only=True)
        if layers > config.num_hidden_layers:
            raise ValueError("requested more BERT layers than the local checkpoint")
        config.num_hidden_layers = layers
        config._attn_implementation = "eager"
        if pretrained:
            bert = BertModel.from_pretrained(str(path), config=config, local_files_only=True,
                                             add_pooling_layer=False, attn_implementation="eager")
        else:
            bert = BertModel(config, add_pooling_layer=False)
        bert.embeddings.word_embeddings = nn.Embedding(1, config.hidden_size)
        for parameter in bert.parameters():
            parameter.requires_grad_(False)
        torch.manual_seed(seed + 1)
        for layer in bert.encoder.layer:
            for name in ("query", "value"):
                attention = layer.attention.self
                setattr(attention, name, LoRALinear(getattr(attention, name), lora_rank))
    return bert
