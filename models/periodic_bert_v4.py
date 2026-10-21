"""V4: content-only periodic readout, learned retrieval and explicit sources.

Historical experiment snapshots/checkpoints remain in pilot_results. No node IDs, inferred road directions,
real-calendar assertions, present forecast values or graph-learned source keys.
The source lower encoder sees each other road independently. Upper source rows
cannot read target tokens/queries; memory rows cannot read conditional queries.
Consequently short state conditions retrieval/attention, never the default
readout values through a late residual or a query-to-memory feedback shortcut.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.bert_backbone import load_bert
from models.spatial_context import SpatialContext
from models.periodic_refinement import daily_block_mask, RelationChangeBias, fine_shape_correction
from models.periodic_memory_policy import (validate_policy, transform_history, memory_token_mask,
    candidate_memory_enabled, metadata as memory_policy_metadata)
from models.periodic_tokens import (
    AlignedPeriodicPattern, FixedPeriodicCalibration, OrderedPatchEncoder, PeriodicTokenization,
    UtilityPeriodicRetrieval, periodic_numeric_features,
)


class PeriodicBertV4(PeriodicTokenization, nn.Module):
    def __init__(self, bert_path: str | Path, *, pretrained=True, numeric=False,
                 retrieval="learned", use_neighbors=True, late_shortcut=False,
                 gated_output=False, adapt_layer_norm=True, node_chunk_size=8,
                 gradient_checkpointing=True, dropout=.1, backbone_seed=2021, repair=None,
                 pattern_mode="normal", lora_rank=4, hierarchical_memory=False,
                 relation_dynamics=False, shape_only_correction=False, refinement_interface=False,
                 memory_policy="full", spatial_repair=False, spatial_context_mode="off",
                 retain_spatial_content=False, retain_context_content=None, retain_source_values=None):
        # Split existing switches only: no new weights/modules. None preserves
        # historical all-or-nothing behavior and existing checkpoint semantics.
        retain_context_content=retain_spatial_content if retain_context_content is None else retain_context_content
        retain_source_values=retain_spatial_content if retain_source_values is None else retain_source_values
        if spatial_context_mode not in ("off","aligned","current_only"):
            raise ValueError("unknown spatial context mode")
        if (spatial_repair and repair!="utility") or ((spatial_context_mode!="off" or retain_spatial_content
                or retain_context_content or retain_source_values) and not spatial_repair):
            raise ValueError("spatial repair requires the utility interface")
        validate_policy(memory_policy)
        if memory_policy != "full" and repair != "utility":
            raise ValueError("memory ablations require utility repair")
        if retrieval not in ("learned", "fixed", "uniform", "day1", "day7"):
            raise ValueError("unknown retrieval control")
        if repair not in (None, "forecast", "query", "full", "fine", "pattern", "strict", "utility"):
            raise ValueError("unknown repair stage")
        if pattern_mode not in ("normal", "off", "no_phase", "no_amplitude", "no_deviation"):
            raise ValueError("unknown pattern intervention")
        if repair is not None and (late_shortcut or gated_output):
            raise ValueError("repair stages do not use legacy output gates/shortcuts")
        if any((hierarchical_memory, relation_dynamics, shape_only_correction, refinement_interface)) and repair != "utility":
            raise ValueError("structured refinements require the utility repair interface")
        super().__init__()
        if node_chunk_size < 1:
            raise ValueError("node_chunk_size must be positive")
        self.pretrained = pretrained
        self.num_layers, self.split_layer, self.lora_rank = 6, 3, lora_rank
        self.node_chunk_size = node_chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.similarity_temperature = .1
        # Keep surviving parameter names AND order for existing optimizer states.
        self.patch_encoder = OrderedPatchEncoder(32)
        self.horizon_queries = nn.Parameter(torch.empty(48, 32))
        self.short_projection = nn.Linear(128, 32)
        self.output_head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
        self.output_gate = nn.Parameter(torch.zeros(48))
        nn.init.normal_(self.horizon_queries, std=.02)
        self.bert = load_bert(bert_path, pretrained=pretrained, seed=backbone_seed, lora_rank=lora_rank)
        self.bert_width = self.bert.config.hidden_size
        self.memory_projection = nn.Linear(32, self.bert_width)
        self.query_projection = nn.Linear(32, self.bert_width)
        self.bert_output_projection = nn.Linear(self.bert_width, 32)
        self.numeric, self.retrieval = numeric, retrieval
        self.use_neighbors, self.late_shortcut = use_neighbors, late_shortcut
        self.gated_output, self.adapt_layer_norm = gated_output, adapt_layer_norm
        if numeric:
            from transformers import BertConfig, BertModel
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(backbone_seed)
                config = BertConfig(hidden_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, intermediate_size=128,
                                    vocab_size=1, max_position_embeddings=512,
                                    hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
                config._attn_implementation = "eager"
                self.bert = BertModel(config, add_pooling_layer=False)
            self.bert.embeddings.word_embeddings.weight.requires_grad_(False)
            self.bert_width, self.num_layers, self.split_layer, self.lora_rank = 64, 2, 1, 0
            self.memory_projection = nn.Linear(32, 64)
            self.query_projection = nn.Linear(32, 64)
            self.bert_output_projection = nn.Linear(64, 32)
        self.role_embedding = nn.Embedding(7, 32)
        nn.init.normal_(self.role_embedding.weight, std=.02)
        self.time_projection = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, 32))
        self.relation_projection = nn.Linear(4, 32, bias=False)
        self.retrieval_query = nn.Linear(4*32, 32)
        self.retrieval_key = nn.Linear(4*32, 32)
        self.retrieval_short = nn.Linear(128, 32, bias=False)
        self.retrieval_horizon = nn.Parameter(torch.empty(4, 32))
        self.retrieval_lag = nn.Parameter(torch.empty(7, 32))
        nn.init.normal_(self.retrieval_horizon, std=.02)
        nn.init.normal_(self.retrieval_lag, std=.02)
        self.content_read = nn.MultiheadAttention(32, 4, dropout=dropout, batch_first=True)
        self.output_gate.requires_grad_(gated_output)
        if not gated_output:
            nn.init.zeros_(self.output_head[-1].weight)
            nn.init.zeros_(self.output_head[-1].bias)
        if adapt_layer_norm and not numeric:
            for module in self.bert.modules():
                if isinstance(module, nn.LayerNorm):
                    for parameter in module.parameters(): parameter.requires_grad_(True)
        # Keep checkpoint membership stable during staged freezing/unfreezing.
        self.adapter_parameter_names = tuple(name for name, p in self.named_parameters() if p.requires_grad)
        self.repair = repair
        self.balanced_query = repair in ("query", "full", "fine", "pattern", "strict", "utility")
        self.fine_query = repair in ("fine", "pattern", "strict", "utility")
        self.uses_numeric_route = repair in ("full", "fine", "pattern", "strict", "utility")
        self.uses_utility_supervision = repair == "utility"
        self.pattern_mode = pattern_mode
        self.memory_policy = memory_policy
        self.hierarchical_memory = hierarchical_memory
        self.relation_dynamics = relation_dynamics
        self.shape_only_correction = shape_only_correction
        self.spatial_repair=spatial_repair
        self.spatial_context_mode=spatial_context_mode
        self.retain_spatial_content=retain_spatial_content
        self.retain_context_content=bool(retain_context_content)
        self.retain_source_values=bool(retain_source_values)
        if repair is not None:
            self.forecast_query = nn.Linear(1, 32)
            self.forecast_calibration = nn.Linear(4, 4)
            nn.init.zeros_(self.forecast_calibration.weight)
            nn.init.zeros_(self.forecast_calibration.bias)
            if self.balanced_query:
                self.hourly_readout = nn.Linear(32, 48 if self.fine_query else 4)
                nn.init.zeros_(self.hourly_readout.weight)
                nn.init.zeros_(self.hourly_readout.bias)
                self.output_head.requires_grad_(False)
            if self.uses_numeric_route:
                self.numeric_calibration = FixedPeriodicCalibration()
                # Its baseline-dependent linear part is already in the fixed route.
                self.forecast_calibration.requires_grad_(False)
            if repair in ("pattern", "strict"):
                self.pattern_operator = AlignedPeriodicPattern(32, context_only=repair == "strict")
                self.pattern_gate = nn.Parameter(torch.zeros(()))
            elif repair == "utility":
                self.pattern_operator = UtilityPeriodicRetrieval(32)
                self.pattern_gate = nn.Parameter(torch.zeros(()))
                # Identical ordered four-token encoding for current/candidates.
                # Flatten instead of averaging away the temporal token order.
                self.utility_context_pool = nn.Sequential(nn.Flatten(1), nn.LayerNorm(128),
                    nn.Linear(128,32), nn.GELU(), nn.LayerNorm(32))
                self.utility_source_read = nn.MultiheadAttention(32,4,dropout=dropout,batch_first=True)
                # Allocate identically for every new structured ablation, even
                # when disabled, so initialization/parameter budgets match.
                if any((hierarchical_memory, relation_dynamics, shape_only_correction, refinement_interface)):
                    self.relation_change_bias = RelationChangeBias()
                if spatial_repair:
                    self.spatial_context=SpatialContext(32,dropout=dropout)
            self.adapter_parameter_names = tuple(name for name, p in self.named_parameters() if p.requires_grad)

    def _layers(self, hidden, valid, start, stop, allowed=None):
        mask = valid[:, None, None, :] if allowed is None else allowed[None, None] & valid[:, None, None, :]
        additive = (~mask).to(hidden.dtype) * torch.finfo(hidden.dtype).min
        for layer in self.bert.encoder.layer[start:stop]:
            def apply(value, block=layer):
                return block(value, attention_mask=additive)[0]
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(apply, hidden, use_reentrant=False)
            else:
                hidden = apply(hidden)
        return hidden

    def set_training_stage(self, adapt_only: bool):
        for name, parameter in self.named_parameters():
            eligible = name in self.adapter_parameter_names
            parameter.requires_grad_(eligible and not (adapt_only and name.startswith("bert.") and not self.numeric))

    def _time(self, relative, spans, origins):
        relative = torch.as_tensor(relative, device=origins.device, dtype=torch.float32)
        spans = torch.as_tensor(spans, device=origins.device, dtype=torch.float32).expand_as(relative)
        absolute = origins.float()[:, None] + relative[None]
        daily, weekly = absolute*(2*math.pi/288), absolute*(2*math.pi/2016)
        address = torch.stack((relative[None].expand_as(absolute)/2016,
                               relative.abs().log1p()[None].expand_as(absolute)/math.log(2065),
                               spans[None].expand_as(absolute)/288,
                               daily.sin(), daily.cos(), weekly.sin(), weekly.cos()), -1)
        return self.time_projection(address)

    def _retrieval_weights(self, current, anchors, current_raw, anchor_raw, short):
        # Keys see historical CONTEXT only. Historical subsequent values are V,
        # never retrieval K. Current forecast targets are not a forward argument.
        def stats(x):
            return torch.stack((x.mean(-1), x.float().std(-1, unbiased=False),
                                x[..., -1]-x[..., 0], x[..., -1]), -1).float()
        distance = (stats(anchor_raw[:, :, :4])-stats(current_raw)[:, None]).square().mean((-1, -2))
        prior = -distance/.1
        if self.retrieval == "learned":
            query = (self.retrieval_query(current.flatten(1)) + self.retrieval_short(short))[:, None]
            query = query+self.retrieval_horizon[None]
            key = self.retrieval_key(anchors[:, :, :4].flatten(2))+self.retrieval_lag[None]
            score = torch.einsum("bhd,bkd->bhk", query.float(), key.float())/math.sqrt(32)
            return (score+prior[:, None]).softmax(-1)
        if self.retrieval == "fixed": return prior.softmax(-1)[:, None].expand(-1, 4, -1)
        weight = torch.full_like(prior, 1/7)
        if self.retrieval in ("day1", "day7"):
            weight.zero_(); weight[:, 0 if self.retrieval == "day1" else 6] = 1
        return weight[:, None].expand(-1, 4, -1)

    def _memory(self, history, short, origins, *, anchor_shift_steps=0,
                component_ablation=None):
        recent, day, current_hour, anchors, _ = self.tokenize(history, anchor_shift_steps=anchor_shift_steps)
        # Here history is [1,T,chunk], hence flattening is target independent.
        count = history.shape[2]
        r, d = self.patch_encoder(recent).reshape(count, 16, 32), self.patch_encoder(day).reshape(count, 8, 32)
        a = self.patch_encoder(anchors).reshape(count, 7, 8, 32)
        c = self.patch_encoder(current_hour).reshape(count, 4, 32)
        weights = self._retrieval_weights(c, a, current_hour.reshape(count, 4, 12),
                                         anchors.reshape(count, 7, 8, 12), short)
        repeat = torch.einsum("bhd,bdhc->bhc", weights.to(a.dtype), a[:, :, 4:])
        context = torch.einsum("bd,bdhc->bhc", weights.mean(1).to(a.dtype), a[:, :, :4])
        deviation = c-context
        if component_ablation == "zero_repeat": repeat = torch.zeros_like(repeat)
        if component_ablation == "zero_deviation": deviation = torch.zeros_like(deviation)
        centers = torch.cat((torch.arange(16, device=history.device)*3-46.5,
                             torch.arange(8, device=history.device)*36-270,
                             (torch.arange(8, device=history.device)[None]*12-42
                              -torch.arange(1, 8, device=history.device)[:, None]*288).flatten(),
                             torch.arange(4, device=history.device)*12+6,
                             torch.arange(4, device=history.device)*12-42))
        spans = torch.tensor([3]*16+[36]*8+[12]*64, device=history.device)
        roles = torch.tensor([0]*16+[1]*8+([2]*4+[3]*4)*7+[4]*4+[5]*4, device=history.device)
        memory = torch.cat((r, d, a.flatten(1, 2), repeat, deviation), 1)
        memory=memory+self._time(centers, spans, origins)+self.role_embedding(roles)[None]
        if self.memory_policy != "full":
            memory=memory.masked_fill(~memory_token_mask(self.memory_policy,device=history.device)[None,:,None],0)
        return memory, weights

    @staticmethod
    def joint_allowed(valid, own_count, source_count):
        """Memory cannot receive Q state; source cannot receive target content."""
        size = valid.shape[1]; memory_count = own_count+source_count
        allowed = valid[:, None, :].expand(-1, size, -1).clone()
        allowed[:, :memory_count, memory_count:] = False
        allowed[:, own_count:memory_count, :own_count] = False
        # A padded source query must still have at least one allowed key, even
        # when there are NO neighbors. Its result is masked from every readout.
        rows = ~allowed.any(-1)
        allowed[:, :, 0] |= rows
        return allowed

    def _joint_layers(self, hidden, valid, own_count, source_count):
        allowed = self.joint_allowed(valid, own_count, source_count)
        additive = (~allowed).to(hidden.dtype)[:, None]*torch.finfo(hidden.dtype).min
        for layer in self.bert.encoder.layer[self.split_layer:]:
            def apply(value, block=layer): return block(value, attention_mask=additive)[0]
            hidden = checkpoint(apply, hidden, use_reentrant=False) if self.training and self.gradient_checkpointing and torch.is_grad_enabled() else apply(hidden)
        return hidden

    def _strict_context_keys(self, history, short, origins, *, neighbor_long_history=None,
                             neighbor_valid=None, relation=None):
        """Separate forward path: neither Q nor K sees analogue outcomes.

        Sharing weights with the main representation does not share activations.
        Candidate contexts are encoded independently, before reading any Value.
        """
        raw = history[0].T
        count = len(raw)
        utility = self.uses_utility_supervision
        pieces, span = (4,12) if utility else (16,3)
        current = raw[:, -48:].reshape(count, pieces, span)
        tokens = self.patch_encoder(current)
        spatial=self.spatial_repair and self.spatial_context_mode!="off"
        if spatial:
            if neighbor_long_history is None or neighbor_valid is None or relation is None:
                raise ValueError("aligned spatial keys require observed neighbor history and addresses")
            k=neighbor_long_history.shape[1]
            source_valid=neighbor_valid & self.use_neighbors
            source_tokens=self.patch_encoder(neighbor_long_history[...,-48:].reshape(count,k,4,12))
            tokens=self.spatial_context(tokens,source_tokens,source_valid,relation)
        tokens = tokens+self._time(torch.arange(pieces, device=raw.device)*span-48+span/2, span, origins)
        if utility:
            q_tokens = tokens
        else:
            short_token = self.short_projection(torch.nn.functional.layer_norm(short, (128,)))[:, None]
            q_tokens = torch.cat((tokens, short_token), 1)
        q_hidden = self.bert.embeddings(inputs_embeds=self.memory_projection(q_tokens))
        q_valid = torch.ones(count, len(q_tokens[0]), dtype=torch.bool, device=raw.device)
        q_hidden = self._layers(q_hidden, q_valid, 0, self.split_layer)
        q_projected = self.bert_output_projection(q_hidden[:, :pieces])
        if self.retain_context_content:
            q_projected=self.spatial_context.preserve(q_projected,tokens)
        query = self.utility_context_pool(q_projected) if utility else q_projected.mean(1)
        candidates, valid, candidate_indices = self.pattern_operator.candidates(raw)
        # Only the first48 samples enter this key encoder. The remaining48
        # historical values are gathered by the operator after weights exist.
        context = candidates[..., :48].reshape(count*35, 4, 12)
        key_tokens = self.patch_encoder(context)
        lag = torch.arange(1, 8, device=raw.device)[:, None]*288
        shifts = torch.tensor(self.pattern_operator.shifts, device=raw.device)[None]
        key_origins = (origins[:, None]-lag.flatten().repeat_interleave(5)[None]
                       +shifts.expand(7, -1).flatten()[None]).flatten()
        key_time=self._time(torch.arange(4, device=raw.device)*12-42, 12, key_origins)
        chosen = valid[None].expand(count, -1).flatten().nonzero(as_tuple=True)[0]
        if spatial and self.spatial_context_mode=="aligned":
            # Context indices ONLY: the candidate's subsequent48 samples must
            # never enter its key, for either target or source roads.
            indices=candidate_indices.reshape(35,96)[:,:48].clamp(0,2063)
            source_context=neighbor_long_history[:,:,indices].permute(0,2,1,3).reshape(count*35,k,48)[chosen]
            source_tokens=self.patch_encoder(source_context.reshape(len(chosen),k,4,12))
            candidate_valid=source_valid[:,None].expand(-1,35,-1).reshape(count*35,k)[chosen]
            candidate_relation=relation[:,None].expand(-1,35,-1,-1).reshape(count*35,k,4)[chosen]
            selected_keys=self.spatial_context(key_tokens[chosen],source_tokens,candidate_valid,candidate_relation)+key_time[chosen]
        else:
            selected_keys=(key_tokens+key_time)[chosen]
        encoded = query.new_zeros(count*35, 32)
        key_hidden = self.bert.embeddings(inputs_embeds=self.memory_projection(selected_keys))
        key_hidden = self._layers(key_hidden, torch.ones(len(chosen), 4, dtype=torch.bool, device=raw.device), 0, self.split_layer)
        projected = self.bert_output_projection(key_hidden)
        if self.retain_context_content:
            projected=self.spatial_context.preserve(projected,selected_keys)
        pooled = self.utility_context_pool(projected) if utility else projected.mean(1)
        encoded = encoded.index_copy(0, chosen, pooled)
        return query, encoded.reshape(count, 35, 32)

    def _chunk(self, history, short, origins, neighbor_history, neighbor_valid, relation,
               *, anchor_shift_steps=0, component_ablation=None, return_diagnostics=False,
               baseline_prediction=None, reference_prediction=None, pattern_mode="normal", return_auxiliary=False,
               neighbor_long_history=None):
        memory_short = torch.nn.functional.layer_norm(short, (128,)) if self.balanced_query else short
        memory, weights = self._memory(history, memory_short, origins,
                                       anchor_shift_steps=anchor_shift_steps,
                                       component_ablation=component_ablation)
        count, own_count = memory.shape[:2]
        own = self.bert.embeddings(inputs_embeds=self.memory_projection(memory))
        own_valid = memory_token_mask(self.memory_policy,device=history.device)[None].expand(count,-1)
        own = self._layers(own, own_valid, 0, self.split_layer,
            allowed=daily_block_mask(history.device) if self.hierarchical_memory else None)
        k = neighbor_history.shape[1]
        source_valid = neighbor_valid & self.use_neighbors
        # Strict independent per-source encoding: no target values in these K/V.
        source_content = self.patch_encoder(neighbor_history.reshape(count, k, 4, 12))
        source=source_content
        centers = torch.arange(4, device=history.device)*12-42
        source = source+self._time(centers, 12, origins)[:, None]+self.role_embedding.weight[6]
        flat = source.reshape(count*k, 4, 32)
        selected = source_valid.flatten().nonzero(as_tuple=True)[0]
        encoded = own.new_zeros(count*k, 4, self.bert_width)
        if len(selected):
            value = self.bert.embeddings(inputs_embeds=self.memory_projection(flat[selected]))
            value = self._layers(value, torch.ones(len(selected), 4, dtype=torch.bool, device=history.device), 0, self.split_layer)
            encoded = encoded.index_copy(0, selected, value)
        encoded = encoded.reshape(count, k, 4, self.bert_width)
        address = self.memory_projection(self.relation_projection(relation)).unsqueeze(2)
        source_hidden = (encoded+address).flatten(1, 2)
        source_mask = source_valid[:, :, None].expand(-1, -1, 4).reshape(count, k*4)
        source_bias = (self.relation_change_bias(history[0].T[:, -48:], neighbor_history).flatten(1)
                       if self.relation_dynamics else None)
        hourly_base = None
        if self.repair is not None:
            hourly_base = baseline_prediction.float().reshape(count, 4, 12).mean(-1)
        if self.balanced_query:
            def normalized(value): return torch.nn.functional.layer_norm(value, (32,))
            horizon = self.horizon_queries if self.fine_query else self.horizon_queries.reshape(4, 12, 32).mean(1)
            query = normalized(horizon)[None]+normalized(self.short_projection(memory_short))[:, None]
            relative = torch.arange(48, device=history.device)+.5 if self.fine_query else torch.arange(4, device=history.device)*12+6
            query = query+normalized(self._time(relative, 1 if self.fine_query else 12, origins))
            forecast = baseline_prediction if self.fine_query else hourly_base
            query = query+normalized(self.forecast_query(forecast[..., None]))
        else:
            query = self.horizon_queries[None]+self.short_projection(short)[:, None]
            query = query+self._time(torch.arange(48, device=history.device)+.5, 1, origins)
            if hourly_base is not None:
                query = query+self.forecast_query(hourly_base[..., None]).repeat_interleave(12, 1)
        query_native = query
        query_count = query.shape[1]
        positions = torch.arange(own_count+k*4, own_count+k*4+query_count, device=history.device)[None].expand(count, -1)
        query = self.bert.embeddings(inputs_embeds=self.query_projection(query), position_ids=positions)
        valid = torch.cat((own_valid, source_mask, torch.ones(count, query_count, dtype=torch.bool, device=history.device)), 1)
        hidden = self._joint_layers(torch.cat((own, source_hidden, query), 1), valid, own_count, k*4)
        # Attention output only: no query residual at the numerical head.
        values = self.bert_output_projection(hidden[:, :own_count+k*4])
        if self.retain_source_values:
            retained=self.spatial_context.preserve(values[:,own_count:],source_content.flatten(1,2))
            values=torch.cat((values[:,:own_count],retained),1)
        queries = self.bert_output_projection(hidden[:, -query_count:])
        read_mask, padding_mask = None, ~valid[:, :own_count+k*4]
        if source_bias is not None:
            bias = torch.cat((source_bias.new_zeros(count, own_count), source_bias), -1)
            bias = bias.masked_fill(padding_mask, float("-inf")).to(queries.dtype)
            read_mask = bias[:, None, None].expand(-1, 4, query_count, -1).reshape(count*4, query_count, -1)
            padding_mask = None
        read, attention = self.content_read(queries, values, values, attn_mask=read_mask,
                                            key_padding_mask=padding_mask,
                                            need_weights=return_diagnostics)
        if self.late_shortcut: read = read+self.short_projection(short)[:, None]
        if self.balanced_query:
            # A dedicated head for each hour cannot be forced into one offset by
            # nearly identical attended memories; keep the native horizon address.
            if self.uses_utility_supervision:
                # Only horizon/time addresses survive directly at the head.
                # Short state/base forecast may condition Q, never bypass memory.
                address_only = normalized(horizon)[None]+normalized(self._time(relative,1,origins))
                state = torch.nn.functional.layer_norm(read, (32,))+address_only
            else:
                state = torch.nn.functional.layer_norm(read, (32,))+query_native
            raw = (state*self.hourly_readout.weight[None]).sum(-1)+self.hourly_readout.bias
            if not self.fine_query:
                raw = raw.repeat_interleave(12, 1)
        else:
            raw = self.output_head(read).squeeze(-1)
        if hourly_base is not None:
            with torch.autocast(history.device.type, enabled=False):
                raw = raw.float()+self.forecast_calibration(hourly_base).repeat_interleave(12, 1)
        if self.gated_output: raw = raw*self.output_gate.tanh()[None]
        if self.shape_only_correction:
            raw = fine_shape_correction(raw)
        pattern_info = {}
        candidates_enabled=candidate_memory_enabled(self.memory_policy)
        if self.uses_utility_supervision and not candidates_enabled:
            # Explicit absent-history control: no candidate encoder/value read,
            # and no fabricated teacher choices from zero-padded past samples.
            neutral=raw.float().sum(-1)[:,None,None]*0
            pattern_info["utility_logits"]=torch.cat((neutral.expand(-1,4,35)-1e4,
                neutral.expand(-1,4,1)), -1)
        if self.repair in ("pattern", "strict", "utility") and pattern_mode != "off" and candidates_enabled:
            if self.repair in ("strict", "utility"):
                if self.spatial_repair and self.spatial_context_mode!="off":
                    context_query,context_keys=self._strict_context_keys(history,short,origins,
                        neighbor_long_history=neighbor_long_history,neighbor_valid=neighbor_valid,relation=relation)
                else:
                    context_query, context_keys = self._strict_context_keys(history, short, origins)
            else:
                context_query = torch.nn.functional.layer_norm(read, (32,)).mean(1)
                context_keys = None
            if reference_prediction is None:
                raise ValueError("pattern correction requires the fixed numerical forecast")
            if self.uses_utility_supervision:
                # Encode sources independently of the target; no analogue outcome
                # or main memory activation is allowed into retrieval Q/K.
                safe_source_mask = source_mask.clone()
                safe_source_mask[:,0] |= ~safe_source_mask.any(-1)
                source_values = values[:,own_count:]
                utility_mask, utility_padding = None, ~safe_source_mask
                if source_bias is not None:
                    bias = source_bias.masked_fill(utility_padding, float("-inf")).to(context_query.dtype)
                    utility_mask = bias[:, None, None].expand(-1,4,1,-1).reshape(count*4,1,-1)
                    utility_padding = None
                conditioned, _ = self.utility_source_read(context_query[:,None], source_values, source_values,
                    key_padding_mask=utility_padding, attn_mask=utility_mask, need_weights=False)
                conditioned = conditioned[:,0]*source_mask.any(-1,keepdim=True)
                context_query = torch.nn.functional.layer_norm(context_query+conditioned+
                    self.short_projection(memory_short), (32,))
                template, pattern_info = self.pattern_operator(history[0].T,context_query,
                    candidate_keys=context_keys,reference_prediction=reference_prediction,mode=pattern_mode)
            else:
                template, pattern_info = self.pattern_operator(history[0].T, context_query,
                    candidate_keys=context_keys, mode=pattern_mode)
            raw = raw.float()+self.pattern_gate.tanh()*(template-reference_prediction.float())
        if return_auxiliary and not return_diagnostics:
            return raw, {"utility_logits": pattern_info["utility_logits"]}
        info = {"retrieval_weights": weights, "read_attention": attention,
                "read_content": read, "memory_values": values, **pattern_info}
        return raw, info if return_diagnostics else None

    def forward(self, history, short_state, *, neighbor_history=None, neighbor_valid=None,
                relation_address=None, origin_step=None, anchor_shift_steps=0,
                component_ablation=None, source_mode="correct", return_diagnostics=False,
                baseline_prediction=None, pattern_mode=None, return_auxiliary=False,neighbor_long_history=None):
        if history.ndim != 3 or history.shape[1] != 2064:
            raise ValueError("expected [B,2064,N] history")
        if self.memory_policy in ("recent_only","day_only"):
            history=transform_history(history,self.memory_policy)
        b, _, n = history.shape
        if short_state.shape != (b, n, 128): raise ValueError("invalid short state")
        if self.repair is not None and (baseline_prediction is None or baseline_prediction.shape != (b, 48, n)):
            raise ValueError("repair requires history-derived baseline predictions [B,48,N]")
        if component_ablation not in (None, "zero_repeat", "zero_deviation"):
            raise ValueError("unknown component ablation")
        if self.repair in ("strict","utility") and (component_ablation is not None or anchor_shift_steps != 0):
            raise ValueError("legacy switches do not cover the strict model; use neural pattern_mode interventions")
        pattern_mode = self.pattern_mode if pattern_mode is None else pattern_mode
        if return_auxiliary and (not self.uses_utility_supervision or pattern_mode == "off"):
            raise ValueError("auxiliary logits require active utility retrieval")
        if self.uses_utility_supervision and pattern_mode not in ("normal","off","no_phase"):
            raise ValueError("utility retrieval preserves raw amplitudes and has no amplitude/deviation switches")
        if pattern_mode not in ("normal", "off", "no_phase", "no_amplitude", "no_deviation"):
            raise ValueError("unknown neural pattern intervention")
        if source_mode not in ("correct", "zero", "source_shuffle", "time_shuffle", "address_shuffle","history_shuffle"):
            raise ValueError("unknown source intervention")
        if neighbor_history is None:
            neighbor_history = history.new_zeros(b, n, 1, 48)
            neighbor_valid = torch.zeros(b, n, 1, dtype=torch.bool, device=history.device)
            relation_address = history.new_zeros(b, n, 1, 4)
        if neighbor_history.shape[:2] != (b, n) or neighbor_history.shape[-1] != 48:
            raise ValueError("invalid neighbor histories")
        k = neighbor_history.shape[2]
        if neighbor_valid.shape != (b, n, k) or relation_address.shape != (b, n, k, 4):
            raise ValueError("invalid neighbor mask/addresses")
        if neighbor_long_history is not None and neighbor_long_history.shape!=(b,n,k,2064):
            raise ValueError("invalid long neighbor history")
        if self.spatial_repair and self.spatial_context_mode!="off" and neighbor_long_history is None:
            raise ValueError("spatial context requires observed long neighbor history")
        if 88+k*4+48 > self.bert.config.max_position_embeddings:
            raise ValueError("source budget exceeds positional capacity")
        if origin_step is None: origin_step = torch.zeros(b, device=history.device)
        if origin_step.shape != (b,): raise ValueError("one origin per window required")
        if source_mode == "zero": neighbor_valid = torch.zeros_like(neighbor_valid)
        if source_mode == "source_shuffle":
            neighbor_history = neighbor_history.roll(1, 1)
            if neighbor_long_history is not None: neighbor_long_history=neighbor_long_history.roll(1,1)
        if source_mode == "time_shuffle":
            neighbor_history = neighbor_history.reshape(b, n, k, 4, 12).flip(-2).reshape(b, n, k, 48)
            if neighbor_long_history is not None:
                neighbor_long_history=neighbor_long_history.reshape(b,n,k,43,4,12).flip(-2).reshape(b,n,k,2064)
        if source_mode == "history_shuffle":
            if neighbor_long_history is None: raise ValueError("historical source shuffle requires long histories")
            changed=neighbor_long_history.clone()
            changed[...,:-48]=neighbor_long_history[...,:-48].roll(1,1)
            neighbor_long_history=changed
        if source_mode == "address_shuffle": relation_address = relation_address.roll(1, 1)
        flat_history = history.transpose(1, 2).reshape(b*n, 2064)
        shorts = short_state.reshape(b*n, 128)
        sources = neighbor_history.reshape(b*n, k, 48)
        long_sources=None if neighbor_long_history is None else neighbor_long_history.reshape(b*n,k,2064)
        masks, addresses = neighbor_valid.reshape(b*n, k), relation_address.reshape(b*n, k, 4)
        origins = origin_step[:, None].expand(b, n).reshape(-1)
        base_flat = None if self.repair is None else baseline_prediction.transpose(1, 2).reshape(b*n, 48)
        numeric_delta = None
        if self.uses_numeric_route:
            numeric_delta = self.numeric_calibration(periodic_numeric_features(
                history, short_state, baseline_prediction, anchor_shift_steps=anchor_shift_steps,
                component_ablation=component_ablation,memory_policy=self.memory_policy)).repeat_interleave(12, -1).transpose(1, 2)
        reference_flat = None if numeric_delta is None else (baseline_prediction.float()+numeric_delta).transpose(1, 2).reshape(b*n, 48)
        predictions, diagnostics = [], []
        for start in range(0, b*n, self.node_chunk_size):
            stop = min(start+self.node_chunk_size, b*n)
            prediction, info = self._chunk(flat_history[start:stop].T[None], shorts[start:stop], origins[start:stop],
                                           sources[start:stop], masks[start:stop], addresses[start:stop],
                                           anchor_shift_steps=anchor_shift_steps,
                                           component_ablation=component_ablation,
                                           return_diagnostics=return_diagnostics,
                                           baseline_prediction=None if base_flat is None else base_flat[start:stop],
                                           reference_prediction=None if reference_flat is None else reference_flat[start:stop],
                                           pattern_mode=pattern_mode,return_auxiliary=return_auxiliary,
                                           neighbor_long_history=None if long_sources is None else long_sources[start:stop])
            predictions.append(prediction)
            if info is not None: diagnostics.append(info)
        output = torch.cat(predictions).reshape(b, n, 48).transpose(1, 2)
        if numeric_delta is not None:
            output = output.float()+numeric_delta
        if return_diagnostics or return_auxiliary:
            return output, {name: torch.cat([value[name] for value in diagnostics]) for name in diagnostics[0]}
        return output

    @property
    def metadata(self):
        return {"interface_version": "periodic_bert_v4", "retrieval": self.retrieval,
                "numeric_backbone": self.numeric, "pretrained": self.pretrained and not self.numeric,
                "bert_layers": self.num_layers, "bert_width": self.bert_width,
                "use_neighbors": self.use_neighbors, "late_shortcut": self.late_shortcut,
                "gated_output": self.gated_output, "adapt_layer_norm": self.adapt_layer_norm,
                "repair_stage": self.repair, "balanced_four_hour_queries": self.balanced_query and not self.fine_query,
                "query_count": 48 if self.fine_query or not self.balanced_query else 4,
                "numeric_route": "frozen train-only ridge, not a PLM contribution" if self.uses_numeric_route else None,
                "fine_grained_48_step_correction": self.fine_query,
                "explicit_pattern_operator": ("raw phase candidates with utility-supervised abstention"
                    if self.uses_utility_supervision else "causal phase candidates, affine amplitude and recent deviation"
                    if self.repair in ("pattern", "strict") else None),
                "context_only_retrieval": self.repair in ("strict","utility"),
                "utility_supervision": self.uses_utility_supervision,
                "memory_ablation": memory_policy_metadata(self.memory_policy),
                "hierarchical_memory": self.hierarchical_memory,
                "hierarchy_scope": ("shared lower BERT per daily snippet; upper BERT across groups; 88 own tokens"
                    if self.hierarchical_memory else None),
                "relation_change_attention_bias": self.relation_dynamics,
                "spatial_context_mode":self.spatial_context_mode,
                "retain_spatial_content":self.retain_spatial_content,
                "retain_context_content":self.retain_context_content,
                "retain_source_values":self.retain_source_values,
                "spatial_source_keys":"observed target and source context only; historical continuations excluded",
                "spatial_repair":self.spatial_repair,
                "shape_only_correction": self.shape_only_correction,
                "fine_shape_bound_standardized": .5 if self.shape_only_correction else None,
                "raw_historical_templates": self.uses_utility_supervision,
                "lora_rank": self.lora_rank,
                "pattern_intervention": self.pattern_mode,
                "pattern_ablation_scope": "additional neural pattern operator; fixed numeric route unchanged",
                "readout": ("memory read plus horizon/time address only; no direct short/base residual"
                    if self.uses_utility_supervision else "memory read plus native horizon query"
                    if self.balanced_query else "memory values only; queries cannot write to memory"),
                "calendar": "relative sampling-index phase, not asserted real weekdays",
                "topology": "provided adjacency, no upstream/downstream or metric-distance assertion",
                "trainable_parameters": sum(p.numel() for name, p in self.named_parameters() if name in self.adapter_parameter_names),
                "total_parameters": sum(p.numel() for p in self.parameters())}
