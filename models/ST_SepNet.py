from math import atanh, sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer,AutoTokenizer, AutoModelForCausalLM, AutoConfig,AutoModel

from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from layers.STHGNN import sthypergnn as STHGNN
from layers.FusionGate import AttentionFusion, LSTMGate
from models.holt_operator import (
    HOLT_DUAL_VARIANTS,
    HOLT_OPERATOR_VARIANTS,
    HOLTCorrectionHead,
    build_binary_undirected_offdiag_aggregation,
)
from models.utility_recoupling import (
    SPATIAL_LLM_MODES,
    HyperedgeStructuralEncoder,
    HyperedgeInnovationTokenizer,
    SparseHyperedgeTemporalRecoupler,
    UtilityRouter,
    build_geometry_matched_random_null,
    build_hyperedge_structural_descriptors,
    build_joint_graph_time_attention_mask,
    build_k_hop_topology_incidence,
    build_metis_incidence,
    build_repeated_time_position_ids,
    build_topology_incidence,
    history_complexity_features,
    hyperedge_structural_descriptor_metadata,
)
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import pandas as pd
import os
transformers.logging.set_verbosity_error()
import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
gc.collect()
torch.cuda.empty_cache()

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        
        self.weight_A = nn.Parameter(torch.randn(self.rank, original_layer.in_features))
        self.weight_B = nn.Parameter(torch.randn(original_layer.out_features, self.rank))
        
        nn.init.kaiming_uniform_(self.weight_A, a=sqrt(5))
        nn.init.zeros_(self.weight_B)
        
    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = F.linear(F.linear(x, self.weight_A), self.weight_B)
        return original_out + lora_out
    



class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)   # From the second layer the input tensor is expanded
        self.linear = nn.Linear(nf, target_window)  # The linear layer transforms the input features into the target output dimension
        self.dropout = nn.Dropout(head_dropout)  # Avoid overfit

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.min_value = configs.min_values
        self.max_value = configs.max_values
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.st_history = configs.st_history
        self.llm_control = configs.llm_control
        self.interaction_rank = configs.interaction_rank
        self.llm_chunk_size = configs.llm_chunk_size
        self.prompt_mode = getattr(configs, 'prompt_mode', 'full')
        self.spatial_llm_mode = getattr(
            configs, 'spatial_llm_mode', 'separate_identity'
        )
        if self.spatial_llm_mode not in SPATIAL_LLM_MODES:
            raise ValueError(f'Unknown spatial_llm_mode: {self.spatial_llm_mode}')
        self.structural_pe_dim = getattr(configs, 'structural_pe_dim', 8)
        self.joint_temporal_radius = getattr(configs, 'joint_temporal_radius', 1)
        if self.structural_pe_dim < 1:
            raise ValueError('structural_pe_dim must be positive')
        if self.joint_temporal_radius < 0:
            raise ValueError('joint_temporal_radius must be non-negative')
        self.uses_llm = self.llm_control != 'graph_only'
        self.uses_graph = self.llm_control != 'llm_only' and configs.use_graph
        self.holt_mode = getattr(configs, 'holt_mode', 'off')
        if self.holt_mode not in {'off', 'fixed', 'learned'}:
            raise ValueError("holt_mode must be one of 'off', 'fixed', or 'learned'")
        self.holt_enabled = self.holt_mode != 'off'
        self.holt_operator_variant = getattr(
            configs, 'holt_operator_variant', 'directed'
        )
        if self.holt_operator_variant not in HOLT_OPERATOR_VARIANTS:
            raise ValueError(
                f'holt_operator_variant must be one of {HOLT_OPERATOR_VARIANTS}'
            )
        if self.holt_operator_variant != 'directed':
            if self.holt_mode != 'fixed':
                raise ValueError(
                    f'{self.holt_operator_variant} qualification control requires '
                    'holt_mode=fixed'
                )
        self.holt_share_hgcn = bool(getattr(configs, 'holt_share_hgcn', False))
        self.holt_basis_training = getattr(configs, 'holt_basis_training', 'fixed')
        if self.holt_basis_training not in {'fixed', 'shared_identity_topology'}:
            raise ValueError(
                'holt_basis_training must be fixed or shared_identity_topology'
            )
        if self.holt_basis_training == 'shared_identity_topology':
            if self.holt_mode != 'fixed':
                raise ValueError(
                    'shared_identity_topology requires holt_mode=fixed'
                )
            if self.holt_share_hgcn:
                raise ValueError(
                    'shared_identity_topology requires holt_share_hgcn=False'
                )
        if (
            self.holt_operator_variant != 'directed'
            and self.holt_basis_training != 'fixed'
        ):
            raise ValueError(
                f'{self.holt_operator_variant} qualification control requires '
                'holt_basis_training=fixed'
            )
        if (
            self.holt_operator_variant != 'directed'
            and self.holt_share_hgcn
        ):
            raise ValueError(
                f'{self.holt_operator_variant} qualification control requires '
                'holt_share_hgcn=False'
            )
        self.holt_shared_secondary = getattr(
            configs, 'holt_shared_secondary', 'topology'
        )
        if self.holt_shared_secondary not in {
            'topology', 'randomized_topology', 'identity'
        }:
            raise ValueError(
                'holt_shared_secondary must be topology, randomized_topology, '
                'or identity'
            )
        if (
            self.holt_basis_training != 'shared_identity_topology'
            and self.holt_shared_secondary != 'topology'
        ):
            raise ValueError(
                'holt_shared_secondary is only configurable for '
                'shared_identity_topology training'
            )
        self.holt_input_mode = getattr(configs, 'holt_input_mode', 'centered')
        if self.holt_input_mode not in {'centered', 'raw'}:
            raise ValueError("holt_input_mode must be either 'centered' or 'raw'")
        self.holt_offdiag_support = getattr(
            configs,
            'holt_offdiag_support',
            'physical_undirected_binary_1hop',
        )
        if self.holt_offdiag_support != 'physical_undirected_binary_1hop':
            raise ValueError(
                'holt_offdiag_support must be physical_undirected_binary_1hop'
            )
        if (
            self.holt_operator_variant in HOLT_DUAL_VARIANTS
            and self.holt_input_mode != 'raw'
        ):
            raise ValueError(
                f'{self.holt_operator_variant} qualification control requires '
                'holt_input_mode=raw so network centering cannot leak target '
                'history into the secondary path'
            )
        self.holt_history_steps_requested = int(
            getattr(configs, 'holt_history_steps', 0)
        )
        if self.holt_history_steps_requested < 0:
            raise ValueError(
                'holt_history_steps must be 0 (full seq_len) or a positive integer'
            )
        if self.holt_history_steps_requested > self.seq_len:
            raise ValueError(
                'holt_history_steps cannot exceed the configured seq_len; got '
                f'{self.holt_history_steps_requested} > {self.seq_len}'
            )
        self.holt_history_steps = (
            self.seq_len
            if self.holt_history_steps_requested == 0
            else self.holt_history_steps_requested
        )
        self.holt_head = None

        adjacency = pd.read_csv(
            os.path.join(configs.root_path, configs.adjacency_path), index_col=False, header=None
        ).values
        if adjacency.shape == (configs.node_num + 1, configs.node_num):
            adjacency = adjacency[1:, :]
        if adjacency.shape != (configs.node_num, configs.node_num):
            raise ValueError(
                f'Expected a {configs.node_num}x{configs.node_num} adjacency matrix, got {adjacency.shape}.'
            )
        self.register_buffer('adj', torch.as_tensor(adjacency, dtype=torch.float32))
        # The CSV stores positive road distances. Graph propagation needs edge
        # support, not a larger message weight for a more distant neighbour.
        self.register_buffer('adj_support', (self.adj != 0).to(torch.float32))
        if self.holt_operator_variant in HOLT_DUAL_VARIANTS:
            holt_physical_offdiag_aggregation = (
                build_binary_undirected_offdiag_aggregation(self.adj)
            )
        else:
            holt_physical_offdiag_aggregation = torch.empty(
                (0, 0), dtype=torch.float32
            )
        self.register_buffer(
            'holt_physical_offdiag_aggregation',
            holt_physical_offdiag_aggregation,
        )
        topology_incidence = build_topology_incidence(self.adj)
        self.register_buffer(
            'holt_topology_incidence', topology_incidence.detach().clone()
        )
        self.register_buffer(
            'holt_identity_incidence',
            torch.eye(
                configs.node_num,
                dtype=topology_incidence.dtype,
                device=topology_incidence.device,
            ),
        )
        self.holt_incidence_mode = getattr(
            configs, 'holt_incidence_mode', 'topology'
        )
        if self.holt_incidence_mode not in {
            'topology', 'randomized_topology', 'identity'
        }:
            raise ValueError(
                'holt_incidence_mode must be topology, randomized_topology, or identity'
            )
        if (
            self.holt_basis_training == 'shared_identity_topology'
            and self.holt_incidence_mode != 'topology'
        ):
            raise ValueError(
                'shared_identity_topology requires holt_incidence_mode=topology'
            )
        if (
            self.holt_operator_variant != 'directed'
            and self.holt_incidence_mode != 'identity'
        ):
            raise ValueError(
                f'{self.holt_operator_variant} qualification control requires '
                'holt_incidence_mode=identity'
            )
        holt_incidence_seed = getattr(configs, 'holt_incidence_seed', None)
        if holt_incidence_seed is None:
            holt_incidence_seed = getattr(configs, 'grouping_seed', 2021)
        self.holt_incidence_seed = int(holt_incidence_seed)
        randomized_topology_incidence, _ = build_geometry_matched_random_null(
            topology_incidence,
            self.adj_support,
            self.holt_incidence_seed,
        )
        self.register_buffer(
            'holt_randomized_topology_incidence',
            randomized_topology_incidence.detach().clone(),
        )
        if self.holt_incidence_mode == 'topology':
            holt_incidence = topology_incidence
        elif self.holt_incidence_mode == 'randomized_topology':
            holt_incidence = self.holt_randomized_topology_incidence
        else:
            holt_incidence = self.holt_identity_incidence
        innovation_node_adjacency = self.adj_support
        if configs.token_grouping == 'topology':
            innovation_incidence = topology_incidence
        elif configs.token_grouping == 'topology_3hop':
            innovation_incidence = build_k_hop_topology_incidence(self.adj, hops=3)
        elif configs.token_grouping == 'random_3hop':
            three_hop_incidence = build_k_hop_topology_incidence(self.adj, hops=3)
            innovation_incidence, innovation_node_adjacency = (
                build_geometry_matched_random_null(
                    three_hop_incidence, self.adj_support, configs.grouping_seed
                )
            )
        elif configs.token_grouping == 'random':
            innovation_incidence, innovation_node_adjacency = (
                build_geometry_matched_random_null(
                    topology_incidence, self.adj_support, configs.grouping_seed
                )
            )
        elif configs.token_grouping == 'metis':
            innovation_incidence = build_metis_incidence(
                self.adj, configs.metis_parts
            )
        else:
            raise ValueError(f'Unknown token grouping: {configs.token_grouping}')
        self.innovation_tokenizer = HyperedgeInnovationTokenizer(
            innovation_incidence,
            selection_policy=getattr(configs, 'selection_policy', 'coherent_diverse'),
            node_adjacency=innovation_node_adjacency,
        )

        if self.uses_graph:
            self.sthgnn = STHGNN(
                gcn_true=configs.gcn_true, hgcn_true=configs.hgcn_true,
                hgat_true=configs.hgat_true,
                buildA_true=configs.graph_source == 'official_adaptive', buildH_true=True,
                gcn_depth=2, num_nodes=configs.node_num, num_hyperedges=configs.node_num,
                device=torch.device('cpu'), adaptive_hyperhgnn=configs.adaptive_hyperhgnn,
                hypergraph_source=configs.hypergraph_source,
                temporl_true=configs.temporl_true, static=configs.static,
                predefined_A=self.adj_support, static_feat=None, dropout=0.3, subgraph_size=20,
                node_dim=40, dilation_exponential=1, conv_channels=32,
                residual_channels=32, skip_channels=64, end_channels=128,
                seq_length=configs.st_history, in_dim=1, out_dim=configs.pred_len,
                layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
                true_adj=self.adj_support, scale_hyperedges=configs.scale_hyperedges,
                alpha=configs.alpha, beta=configs.beta, gamma=configs.gamma, theta=configs.theta,
            )
        self.gate_linear = nn.Linear(configs.node_num * 2, configs.node_num)
        # A selected local correction may only influence its own node through
        # this shared gate.  The legacy dense gate is retained as ``adaptive``
        # for an explicit architecture control.
        self.nodewise_gate = nn.Linear(2, 1)
        self.fusion_gate = configs.fusion_gate
        self.attention_gate = AttentionFusion(configs.node_num)
        self.lstm_gate = LSTMGate(configs.node_num) 
        gate_modules = {
            'adaptive': self.gate_linear,
            'nodewise': self.nodewise_gate,
            'attentiongate': self.attention_gate,
            'lstmgate': self.lstm_gate,
        }
        for gate_name, gate_module in gate_modules.items():
            enabled = self.uses_llm and self.uses_graph and self.fusion_gate == gate_name
            for parameter in gate_module.parameters():
                parameter.requires_grad = enabled
        

        if not self.uses_llm:
            self._initialize_holt(configs, holt_incidence)
            return
        if (
            self.spatial_llm_mode != 'separate_identity'
            and configs.llm_model != 'BERT'
        ):
            raise ValueError('Joint spatial LLM modes currently require BERT.')

        if configs.llm_model == 'LLAMA7b':
            self.llama_config = LlamaConfig.from_pretrained('huggingface/llama7b/')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained(
                'huggingface/llama7b/',
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
            )

            self.tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                'huggingface/llama7b/',
                trust_remote_code=True,
                local_files_only=True
            )

        elif configs.llm_model =='LLAMA1b':
            self.llama_config = AutoConfig.from_pretrained('huggingface/llama1b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained(
                'huggingface/llama1b',
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
                torch_dtype=torch.bfloat16,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                 'huggingface/llama1b',
                trust_remote_code=True,
                local_files_only=True
            )

        elif configs.llm_model =='LLAMA8b':
            self.llama_config = AutoConfig.from_pretrained('huggingface/llama8b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained(
                "huggingface/llama8b",
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
                torch_dtype = torch.float16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "huggingface/llama8b",
                trust_remote_code=True,
                local_files_only=True
            )


        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('huggingface/gpt2')
            
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(
                'huggingface/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
            )

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'huggingface/gpt2',
                trust_remote_code=True,
                local_files_only=True
            )

        elif configs.llm_model == 'GPT3':
            self.gpt2_config = AutoConfig.from_pretrained('huggingface/gpt3')
            
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = AutoModel.from_pretrained(
                'huggingface/gpt3',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
                torch_dtype=torch.float16,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                'huggingface/gpt3',
                trust_remote_code=True,
                local_files_only=True
            )



        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained(configs.llm_path)
            if configs.llm_layers > self.bert_config.num_hidden_layers:
                raise ValueError(
                    f'Checkpoint has {self.bert_config.num_hidden_layers} layers, '
                    f'but --llm_layers={configs.llm_layers}.'
                )
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = False
            self.bert_config.output_hidden_states = False
            if self.spatial_llm_mode != 'separate_identity':
                # Transformers' SDPA shortcut currently assumes a 2-D padding
                # mask.  Eager BERT uses the standard extended-mask path that
                # supports our per-window 3-D graph-time attention mask.
                self.bert_config._attn_implementation = 'eager'
            if configs.llm_control == 'small_transformer':
                self.bert_config.hidden_size = configs.small_transformer_dim
                self.bert_config.num_hidden_layers = configs.small_transformer_layers
                self.bert_config.num_attention_heads = 8
                self.bert_config.intermediate_size = 4 * configs.small_transformer_dim
                self.llm_model = BertModel(self.bert_config)
            elif configs.llm_control == 'random_lora':
                self.llm_model = BertModel(self.bert_config)
            else:
                self.llm_model = BertModel.from_pretrained(
                    configs.llm_path,
                    local_files_only=True,
                    config=self.bert_config,
                )

            self.tokenizer = BertTokenizer.from_pretrained(
                configs.llm_path,
                local_files_only=True,
                )
            self.d_llm = self.llm_model.config.hidden_size


        elif configs.llm_model == 'deepseek2b':
            self.deepseek_config = AutoConfig.from_pretrained('huggingface/deepseek-qwen2b')
            
            self.deepseek_config.num_hidden_layers = configs.llm_layers
            self.deepseek_config.output_attentions = True
            self.deepseek_config.output_hidden_states = True
            self.llm_model = AutoModel.from_pretrained(
                'huggingface/deepseek-qwen2b',
                trust_remote_code=True,
                local_files_only=True,
                config=self.deepseek_config,
                torch_dtype=torch.float16,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                'huggingface/deepseek-qwen2b',
                trust_remote_code=True,
                local_files_only=True
            )

        else:
            raise Exception('LLM model is not defined')


        
        def set_module(model, module_path, new_module):
            parts = module_path.split('.')
            parent = model
            for part in parts[:-1]:
                if part.isdigit():  
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

        
        if configs.llm_control in {'official_frozen', 'corrected_lora', 'random_lora', 'llm_only'}:
            modules_to_replace = []
            for name, module in self.llm_model.named_modules():
                if isinstance(module, nn.Linear):
                    modules_to_replace.append((name, LoRALayer(module, rank=4)))
            for name, new_module in modules_to_replace:
                set_module(self.llm_model, name, new_module)



        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False
        if configs.llm_control == 'small_transformer':
            for param in self.llm_model.parameters():
                param.requires_grad = True
        if configs.llm_control in {'corrected_lora', 'random_lora', 'llm_only'}:
            for module in self.llm_model.modules():
                if isinstance(module, LoRALayer):
                    module.weight_A.requires_grad = True
                    module.weight_B.requires_grad = True
        if getattr(configs, 'llm_gradient_checkpointing', False):
            self.llm_model.gradient_checkpointing_enable()

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = (
                f'{configs.data} traffic-flow dataset with {configs.node_num} road sensors '
                'sampled every five minutes. The local timestamp column is used as a '
                'calendar/index feature only; it is not evidence of incidents or holidays. '
                'The road adjacency matrix supplies spatial connectivity.'
            )
            
            # 'Bike data is bicycle demand data for 295 transit nodes in New York, and bicycle demand usually refers to bicycle use in the area.\
            #     The data contains time series data for 295 nodes, where the demand for day and night shows certain seasonal and cyclical changes, \
            #     the demand data includes time series data for the flow of bicycles into and out of the site nodes, each data node contains the target value "bicycle demand".'
           

            # 'Bike data is bicycle demand data for 295 transit nodes in New York, and bicycle demand usually refers to bicycle use in the area.\
            #     The data contains time series data for 290 nodes, where the demand for day and night shows certain seasonal and cyclical changes, \
            #     the demand data includes time series data for the flow of bicycles into and out of the site nodes, each data node contains the target value "bicycle demand".'
           

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)


        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = configs.num_tokens
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        if self.spatial_llm_mode == 'joint_structural':
            descriptors = build_hyperedge_structural_descriptors(
                innovation_incidence,
                innovation_node_adjacency,
                pe_dim=self.structural_pe_dim,
            )
            self.edge_identity_embedding = None
            self.spatial_structure_encoder = HyperedgeStructuralEncoder(
                descriptors, self.d_llm
            )
            self.spatial_descriptor_metadata = (
                hyperedge_structural_descriptor_metadata(self.structural_pe_dim)
            )
        else:
            self.edge_identity_embedding = nn.Embedding(
                self.innovation_tokenizer.num_edges + 1, self.d_llm, padding_idx=0
            )
            self.spatial_structure_encoder = None
            self.spatial_descriptor_metadata = None
        self.recoupler = SparseHyperedgeTemporalRecoupler(
            self.d_ff, n_heads=configs.recoupler_heads, dropout=configs.dropout
        )
        self.utility_router = UtilityRouter(self.innovation_tokenizer)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, non_norm=True)
        self._initialize_holt(configs, holt_incidence)

    def _initialize_holt(self, configs, topology_incidence):
        """Build the optional history-only HOLT residual after base modules.

        Constructing HOLT last is intentional: with the same seed, enabling a
        Pilot arm cannot perturb the corrected STH-SepNet initialization.
        """
        if not self.holt_enabled:
            return
        if self.holt_share_hgcn:
            if not self.uses_graph:
                raise ValueError('holt_share_hgcn requires an enabled graph branch')
            if not bool(getattr(configs, 'hgcn_true', False)):
                raise ValueError('holt_share_hgcn requires hgcn_true=True')
            if getattr(configs, 'adaptive_hyperhgnn', 'hgcn') != 'hgcn':
                raise ValueError('holt_share_hgcn currently supports only the hgcn operator')

        entropy_weight = float(getattr(configs, 'holt_entropy_weight', 0.0))
        diversity_weight = float(getattr(configs, 'holt_diversity_weight', 0.0))
        escape_weight = float(getattr(configs, 'holt_escape_weight', 0.0))
        min_entropy = float(getattr(configs, 'holt_min_entropy', 0.0))
        if min(entropy_weight, diversity_weight, escape_weight) < 0:
            raise ValueError('HOLT regularization weights must be non-negative')
        if not 0.0 <= min_entropy <= 1.0:
            raise ValueError('holt_min_entropy must lie in [0,1]')
        self.holt_entropy_weight = entropy_weight
        self.holt_diversity_weight = diversity_weight
        self.holt_escape_weight = escape_weight
        self.holt_min_entropy = min_entropy

        escape_slots = getattr(configs, 'holt_escape_slots', None)
        if escape_slots is not None:
            escape_slots = int(escape_slots)
        self.holt_head = HOLTCorrectionHead(
            forecast_horizon=self.pred_len,
            physical_candidate_mask=topology_incidence,
            physical_offdiag_aggregation=(
                self.holt_physical_offdiag_aggregation
                if self.holt_operator_variant in HOLT_DUAL_VARIANTS
                else None
            ),
            num_escape_slots=escape_slots,
            d_model=int(getattr(configs, 'holt_dim', 32)),
            num_operators=int(getattr(configs, 'holt_operators', 8)),
            num_source_heads=int(getattr(configs, 'holt_source_heads', 4)),
            num_temporal_patches=int(getattr(configs, 'holt_temporal_patches', 8)),
            hypergraph_mode=self.holt_mode,
            operator_variant=self.holt_operator_variant,
            temperature=float(getattr(configs, 'holt_temperature', 1.0)),
            initial_blend_rho=float(
                getattr(configs, 'holt_blend_rho_init', 0.1)
            ),
        )
        gate_init = float(getattr(configs, 'holt_gate_init', 0.0))
        if not -1.0 < gate_init < 1.0:
            raise ValueError('holt_gate_init must lie strictly between -1 and 1')
        secondary_gate_init = float(
            getattr(configs, 'holt_secondary_gate_init', gate_init)
        )
        if not -1.0 < secondary_gate_init < 1.0:
            raise ValueError(
                'holt_secondary_gate_init must lie strictly between -1 and 1'
            )
        with torch.no_grad():
            self.holt_head.operator_head.residual_gate.fill_(atanh(gate_init))
            if self.holt_head.secondary_operator_head is not None:
                self.holt_head.secondary_operator_head.residual_gate.fill_(
                    atanh(secondary_gate_init)
                )

    def _resolve_holt_basis_mask(self, holt_basis):
        """Resolve a runtime-only spatial basis without adding arm parameters."""
        if holt_basis is None:
            return None
        if holt_basis not in {
            'identity', 'secondary', 'topology', 'randomized_topology'
        }:
            raise ValueError(
                "holt_basis must be identity, secondary, topology, or "
                "randomized_topology"
            )
        if not self.holt_enabled or self.holt_mode != 'fixed':
            raise ValueError('explicit holt_basis requires fixed HOLT mode')
        if self.holt_share_hgcn:
            raise ValueError(
                'explicit holt_basis is isolated to HOLT and requires '
                'holt_share_hgcn=False'
            )
        if (
            holt_basis == 'secondary'
            and self.holt_basis_training != 'shared_identity_topology'
        ):
            raise ValueError(
                'the secondary runtime basis requires shared_identity_topology training'
            )
        effective_basis = (
            self.holt_shared_secondary if holt_basis == 'secondary' else holt_basis
        )
        return {
            'identity': self.holt_identity_incidence,
            'topology': self.holt_topology_incidence,
            'randomized_topology': self.holt_randomized_topology_incidence,
        }[effective_basis]

    def _holt_components(
        self, x_enc, holt_hypergraph_history=None, holt_basis=None
    ):
        """Return one correction and the exact H optionally shared with HGCN."""
        if not self.holt_enabled:
            zero = x_enc.new_zeros(())
            return None, None, {'regularization_loss': zero}

        if x_enc.ndim != 3 or x_enc.shape[1] < self.holt_history_steps:
            raise ValueError(
                'HOLT input must have shape [batch,time,nodes] with at least '
                f'{self.holt_history_steps} time steps'
            )
        if (
            holt_hypergraph_history is not None
            and holt_hypergraph_history.shape != x_enc.shape
        ):
            raise ValueError(
                'holt_hypergraph_history must have the same shape as x_enc'
            )

        # This is a qualification-only history axis.  The original global-LLM
        # and graph forecasters continue to receive the complete ``x_enc``;
        # only the optional HOLT residual (including its shuffled-H diagnostic)
        # sees the requested causal suffix.
        holt_input = x_enc[:, -self.holt_history_steps:, :]
        shuffled_holt_input = (
            None
            if holt_hypergraph_history is None
            else holt_hypergraph_history[:, -self.holt_history_steps:, :]
        )

        # HOLT is a node-level residual path.  Removing the per-timestep
        # network mean prevents this branch from relearning the global level
        # already represented by the base predictor.  ``raw`` remains an
        # explicit ablation, while every original LLM/graph branch still sees
        # the untouched input supplied to ``forecast``.
        if self.holt_input_mode == 'centered':
            holt_input = holt_input - holt_input.mean(dim=-1, keepdim=True)
            shuffled_holt_input = (
                None
                if shuffled_holt_input is None
                else shuffled_holt_input
                - shuffled_holt_input.mean(dim=-1, keepdim=True)
            )
        basis_mask = self._resolve_holt_basis_mask(holt_basis)
        encoded = self.holt_head.encode_and_build_state(
            holt_input,
            physical_candidate_mask=basis_mask,
            hypergraph_history=shuffled_holt_input,
        )
        correction, operator_diagnostics = self.holt_head.predict_from_state(
            encoded, return_diagnostics=True
        )
        state = encoded.hypergraph
        zero = correction.new_zeros(())
        if self.holt_mode == 'learned':
            entropy_penalty = torch.relu(
                self.holt_min_entropy - state.normalized_entropy
            ).square().mean()
            diversity_penalty = (1.0 - state.edge_diversity).mean()
            escape_penalty = state.escape_mass.mean()
            regularization = (
                self.holt_entropy_weight * entropy_penalty
                + self.holt_diversity_weight * diversity_penalty
                + self.holt_escape_weight * escape_penalty
            )
        else:
            entropy_penalty = zero
            diversity_penalty = zero
            escape_penalty = zero
            regularization = zero

        source_response_overlap = torch.einsum(
            'brqpe,bre->brqp',
            operator_diagnostics['source_weights'].float(),
            operator_diagnostics['response_weights'].float(),
        ).abs().mean(dim=(1, 2, 3))
        edge_mass_entropy = -(
            state.edge_mass.clamp_min(1e-12)
            * state.edge_mass.clamp_min(1e-12).log()
        ).sum(dim=1)
        edge_mass_effective_count = edge_mass_entropy.exp()
        edge_mass_cv = state.edge_mass.std(dim=1, unbiased=False) / (
            state.edge_mass.mean(dim=1).clamp_min(1e-12)
        )
        auxiliary = {
            'regularization_loss': regularization,
            'holt_entropy_penalty': entropy_penalty,
            'holt_diversity_penalty': diversity_penalty,
            'holt_escape_penalty': escape_penalty,
            'holt_normalized_entropy': state.normalized_entropy.mean(),
            'holt_edge_diversity': state.edge_diversity.mean(),
            'holt_escape_mass': state.escape_mass.mean(),
            'holt_blend_rho': state.blend_rho.mean(),
            'holt_node_mass_error': state.node_mass_error.mean(),
            'holt_edge_mass_drift': state.edge_mass_drift.mean(),
            'holt_membership_l1_drift': state.membership_l1_drift.mean(),
            'holt_gate_abs_mean': torch.tanh(
                self.holt_head.operator_head.residual_gate
            ).abs().mean(),
            'holt_correction_abs_mean': correction.abs().mean(),
            'holt_primary_correction_abs_mean': operator_diagnostics.get(
                'primary_correction', correction
            ).abs().mean(),
            'holt_secondary_correction_abs_mean': operator_diagnostics.get(
                'secondary_correction', zero
            ).abs().mean(),
            'holt_secondary_gate_abs_mean': (
                torch.tanh(
                    self.holt_head.secondary_operator_head.residual_gate
                ).abs().mean()
                if self.holt_head.secondary_operator_head is not None
                else zero
            ),
            'holt_input_centered': correction.new_tensor(
                float(self.holt_input_mode == 'centered')
            ),
            'holt_basis_identity': correction.new_tensor(
                float(holt_basis == 'identity')
            ),
            'holt_basis_secondary': correction.new_tensor(
                float(holt_basis == 'secondary')
            ),
            # Per-window arrays are kept separate from the scalar training
            # summaries.  The runner persists these lightweight diagnostics
            # by target_start, never the full N x E membership tensor.
            'holt_window_normalized_entropy': state.normalized_entropy,
            'holt_window_edge_mass_drift': state.edge_mass_drift,
            'holt_window_membership_l1_drift': state.membership_l1_drift,
            'holt_window_edge_mass_effective_count': edge_mass_effective_count,
            'holt_window_edge_mass_cv': edge_mass_cv,
            'holt_window_source_response_overlap': source_response_overlap,
        }
        incidence = state.symmetric_incidence if self.holt_share_hgcn else None
        return correction, incidence, auxiliary


    def forward(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
        interaction_rank=None, return_auxiliary=False,
        holt_hypergraph_history=None, holt_basis=None,
        return_features=False,
    ):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            result = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                interaction_rank=interaction_rank,
                return_auxiliary=return_auxiliary,
                holt_hypergraph_history=holt_hypergraph_history,
                holt_basis=holt_basis,
                return_features=return_features,
            )
            if return_auxiliary or return_features:
                dec_out, auxiliary = result
                return dec_out[:, -self.pred_len:, :], auxiliary
            return result[:, -self.pred_len:, :]
        return None

    def _graph_forecast(self, x_enc, incidence_override=None, return_features=False):
        """Run the short-history graph branch once for a batch."""
        if x_enc.ndim == 4:
            x_enc = x_enc.squeeze(-1)
        graph_input = x_enc[:, -self.st_history:].unsqueeze(-1)
        if return_features:
            prediction, hidden = self.sthgnn(
                graph_input.transpose(1, 3),
                incidence_override=incidence_override,
                return_features=True,
            )
            return prediction.squeeze(-1), hidden
        return self.sthgnn(
            graph_input.transpose(1, 3),
            incidence_override=incidence_override,
        ).squeeze(-1)

    @staticmethod
    def _apply_holt_residual(base_prediction, holt_correction):
        """Apply HOLT once, after the selected base forecast is complete.

        HOLT is defined as an additive residual on the final node forecast,
        not as an input to an existing fusion gate.  Keeping this operation in
        one place also makes ``graph_only`` and every LLM/graph fusion mode
        obey the same ``Y = Y_base + correction`` contract.
        """
        base_prediction = base_prediction.float()
        if holt_correction is None:
            return base_prediction
        return base_prediction + holt_correction.to(
            device=base_prediction.device,
            dtype=base_prediction.dtype,
        )

    def forecast(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec, adj=None,
        interaction_rank=None, innovation_cache=None, graph_cache=None,
        return_auxiliary=False, holt_hypergraph_history=None,
        holt_basis=None,
        return_features=False,
    ):
        if return_features and not self.uses_graph:
            raise ValueError('return_features requires an enabled graph branch')
        if return_features and graph_cache is not None:
            raise ValueError('plain graph_cache does not contain matching hidden features')
        short_hidden = None
        all_min_value = torch.tensor(self.min_value).squeeze().to(x_enc.device)
        all_max_value = torch.tensor(self.max_value).squeeze().to(x_enc.device)
        if len(x_enc.size()) == 4:
            x_enc = x_enc.squeeze(-1)
        B, T, N = x_enc.size()

        holt_correction, holt_incidence, auxiliary = self._holt_components(
            x_enc,
            holt_hypergraph_history=holt_hypergraph_history,
            holt_basis=holt_basis,
        )

        if not self.uses_llm:
            if return_features:
                base_prediction, short_hidden = self._graph_forecast(
                    x_enc, incidence_override=holt_incidence, return_features=True
                )
            else:
                base_prediction = self._graph_forecast(
                    x_enc, incidence_override=holt_incidence
                )
            graph_prediction = self._apply_holt_residual(
                base_prediction, holt_correction
            )
            if return_features:
                features = {'short_state': short_hidden}
                if return_auxiliary:
                    features['auxiliary'] = auxiliary
                return graph_prediction, features
            if return_auxiliary:
                return graph_prediction, auxiliary
            return graph_prediction

        rank = self.interaction_rank if interaction_rank is None else interaction_rank
        if innovation_cache is None:
            global_series, local_series, edge_ids, membership = self.innovation_tokenizer(
                x_enc, rank
            )
        else:
            global_series, local_series, edge_ids, membership = (
                self.innovation_tokenizer.select_precomputed(
                    innovation_cache[0], innovation_cache[1], rank
                )
            )
        numeric_series = torch.cat([global_series, local_series], dim=1)
        series_count = numeric_series.shape[1]
        x_enc_pool = global_series.permute(0, 2, 1)


        
        min_values = torch.min(x_enc_pool, dim=1)[0]
        max_values = torch.max(x_enc_pool, dim=1)[0]
        # Only median values enter the textual prompt.  CUDA median also
        # computes unused tie-dependent indices and rejects strict
        # determinism; stable sorting gives the identical lower median.
        medians = x_enc_pool.sort(dim=1, stable=True).values[:, (x_enc_pool.shape[1] - 1) // 2]
        lags = self.calcute_lags(x_enc_pool)
        trends = x_enc_pool.diff(dim=1).sum(dim=1)
 
        
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: " 
                f'The time stamp information states as [month, day, weekday, hour, minute],'
                f'The input time begins from {str(x_mark_enc[b,0,:].tolist())} to {str(x_mark_enc[b,-1,:].tolist())} '
                f'The prediction time begins from {str(x_mark_dec[b,-self.pred_len,:].tolist())} to {str(x_mark_dec[b,-1,:].tolist())}'
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            if self.prompt_mode == 'no_text':
                prompt_ = '<|start_prompt|><|<end_prompt>|>'
            elif self.prompt_mode == 'no_description':
                prompt_ = (
                    f'<|start_prompt|>Forecast the next {self.pred_len} steps from '
                    f'the previous {self.seq_len} steps. min {min_values_str}, '
                    f'max {max_values_str}, median {median_values_str}, '
                    f'trend {"upward" if trends[b] > 0 else "downward"}, '
                    f'lags {lags_values_str}<|<end_prompt>|>'
                )
            elif self.prompt_mode != 'full':
                raise ValueError(f'Unknown prompt_mode: {self.prompt_mode}')

            prompt.append(prompt_)
        
        x_enc_pool = x_enc_pool.contiguous()
        
        
        tokenized_prompt = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True,
            max_length=min(512, self.llm_model.config.max_position_embeddings - self.patch_nums),
        )
        prompt_ids = tokenized_prompt.input_ids.to(x_enc_pool.device)
        prompt_mask = tokenized_prompt.attention_mask.to(x_enc_pool.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        enc_out, n_vars = self.patch_embedding(numeric_series)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        enc_out = enc_out.reshape(B, series_count, self.patch_nums, self.d_llm)
        if self.spatial_llm_mode == 'joint_structural':
            local_structure = self.spatial_structure_encoder(edge_ids)
            global_structure = local_structure.new_zeros((B, 1, self.d_llm))
            spatial_encoding = torch.cat(
                [global_structure, local_structure], dim=1
            ).unsqueeze(2)
        else:
            identity_ids = torch.cat(
                [
                    torch.zeros((B, 1), dtype=torch.long, device=x_enc.device),
                    edge_ids + 1,
                ],
                dim=1,
            )
            spatial_encoding = self.edge_identity_embedding(identity_ids).unsqueeze(2)
        enc_out = enc_out + spatial_encoding

        # Encode the natural-language prompt once with the global trajectory.
        # Repeating the identical prompt for every selected hyperedge made the
        # original full-coupling control needlessly O(K * prompt_length).
        global_patches = enc_out[:, 0].to(prompt_embeddings.dtype)
        global_input = torch.cat([prompt_embeddings, global_patches], dim=1)
        global_patch_mask = torch.ones(
            (B, self.patch_nums), dtype=prompt_mask.dtype, device=prompt_mask.device
        )
        global_attention_mask = torch.cat([prompt_mask, global_patch_mask], dim=1)
        global_hidden = self.llm_model(
            inputs_embeds=global_input,
            attention_mask=global_attention_mask,
        ).last_hidden_state[:, -self.patch_nums:, :self.d_ff].unsqueeze(1)

        if rank > 0 and self.spatial_llm_mode == 'separate_identity':
            local_input = enc_out[:, 1:].reshape(
                B * rank, self.patch_nums, self.d_llm
            )
            local_attention_mask = torch.ones(
                (B * rank, self.patch_nums),
                dtype=prompt_mask.dtype,
                device=prompt_mask.device,
            )
            chunk_size = self.llm_chunk_size or local_input.shape[0]
            local_chunks = []
            for start in range(0, local_input.shape[0], chunk_size):
                stop = start + chunk_size
                hidden = self.llm_model(
                    inputs_embeds=local_input[start:stop],
                    attention_mask=local_attention_mask[start:stop],
                ).last_hidden_state
                local_chunks.append(hidden[:, :, :self.d_ff])
            local_hidden = torch.cat(local_chunks, dim=0).reshape(
                B, rank, self.patch_nums, self.d_ff
            )
        elif rank > 0:
            # Edge-major flattening is only a storage convention.  Repeating
            # position IDs 0..P-1 for every edge prevents sequence order from
            # masquerading as a spatial coordinate.  Space enters BERT through
            # the per-sample graph-time mask and, in joint_structural mode,
            # through deterministic graph descriptors.
            local_input = enc_out[:, 1:].reshape(
                B, rank * self.patch_nums, self.d_llm
            )
            local_attention_mask = build_joint_graph_time_attention_mask(
                edge_ids,
                self.innovation_tokenizer.edge_adjacency,
                self.patch_nums,
                temporal_radius=self.joint_temporal_radius,
            )
            local_position_ids = build_repeated_time_position_ids(
                B,
                rank,
                self.patch_nums,
                device=x_enc.device,
            )
            chunk_size = self.llm_chunk_size or B
            local_chunks = []
            for start in range(0, B, chunk_size):
                stop = start + chunk_size
                hidden = self.llm_model(
                    inputs_embeds=local_input[start:stop],
                    attention_mask=local_attention_mask[start:stop],
                    position_ids=local_position_ids[start:stop],
                    # Supplying this explicitly avoids BERT's cached
                    # max-position-length token-type buffer.  Position IDs
                    # themselves still repeat only the P valid time indices.
                    token_type_ids=torch.zeros_like(
                        local_position_ids[start:stop]
                    ),
                ).last_hidden_state
                local_chunks.append(hidden[:, :, :self.d_ff])
            local_hidden = torch.cat(local_chunks, dim=0).reshape(
                B, rank, self.patch_nums, self.d_ff
            )
        else:
            local_hidden = global_hidden.new_empty(
                (B, 0, self.patch_nums, self.d_ff)
            )
        if rank > 0:
            local_hidden = self.recoupler(
                global_hidden,
                local_hidden,
                edge_ids,
                self.innovation_tokenizer.edge_adjacency,
            )
        numeric_hidden = torch.cat([global_hidden, local_hidden], dim=1)
        series_predictions = self.output_projection(
            numeric_hidden.permute(0, 1, 3, 2).contiguous()
        )
        global_prediction = series_predictions[:, 0].unsqueeze(-1)
        if rank > 0:
            local_prediction = series_predictions[:, 1:].permute(0, 2, 1)
            membership_count = membership.sum(dim=1).clamp_min(1.0)
            node_delta = torch.einsum('bhk,bkn->bhn', local_prediction, membership)
            node_delta = node_delta / membership_count.unsqueeze(1)
            dec_out = global_prediction.expand(-1, -1, N) + node_delta
        else:
            dec_out = global_prediction.expand(-1, -1, N)

        # Spatio-Temporal Hypergraph Neural Network
        if self.uses_graph:
            if graph_cache is not None and holt_incidence is not None:
                raise ValueError('graph_cache cannot be reused with shared dynamic HOLT incidence')
            if return_features:
                sthgnn_enc, short_hidden = self._graph_forecast(
                    x_enc, incidence_override=holt_incidence, return_features=True
                )
            else:
                sthgnn_enc = graph_cache if graph_cache is not None else self._graph_forecast(
                    x_enc, incidence_override=holt_incidence
                )

        if not self.uses_graph:
            fused_output = dec_out.expand(-1, -1, N)
        elif self.fusion_gate == 'adaptive':
            dec_out_expanded = dec_out.expand(-1, -1, N)  
            combined_input = torch.cat((dec_out_expanded, sthgnn_enc), dim=2)   
            gate_values = torch.sigmoid(self.gate_linear(combined_input))

            fused_output = gate_values * dec_out_expanded + (1 - gate_values) * sthgnn_enc  
            fused_output = fused_output.to(torch.float)
            fused_output = torch.maximum(fused_output, all_min_value.view(1, 1, N))
            fused_output = torch.minimum(fused_output, all_max_value.view(1, 1, N))

        elif self.fusion_gate == 'nodewise':
            dec_out_expanded = dec_out.expand(-1, -1, N)
            combined_input = torch.stack((dec_out_expanded, sthgnn_enc), dim=-1)
            gate_values = torch.sigmoid(self.nodewise_gate(combined_input)).squeeze(-1)
            fused_output = gate_values * dec_out_expanded + (1 - gate_values) * sthgnn_enc
            fused_output = torch.maximum(fused_output.float(), all_min_value.view(1, 1, N))
            fused_output = torch.minimum(fused_output, all_max_value.view(1, 1, N))

        elif self.fusion_gate == 'hyperstgnn':
            # Only adaptively ground higher-order neural networks are considered
            fused_output = sthgnn_enc


        elif self.fusion_gate == 'attentiongate':
            dec_out_expanded = dec_out.expand(-1, -1, N)  

            fused_output = self.attention_gate(dec_out_expanded, sthgnn_enc)
        
        elif self.fusion_gate == 'lstmgate':
            dec_out_expanded = dec_out.expand(-1, -1, N)  

            fused_output = self.lstm_gate(dec_out_expanded, sthgnn_enc)

        else:
            fused_output = dec_out.expand(-1, -1, N)  

        fused_output = self._apply_holt_residual(
            fused_output, holt_correction
        )
        if return_features:
            features = {'short_state': short_hidden}
            if return_auxiliary:
                features['auxiliary'] = auxiliary
            return fused_output, features
        if return_auxiliary:
            return fused_output, auxiliary
        return fused_output

    def forward_routed(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec,
        confidence_threshold=0.5,
    ):
        """Route samples by K and execute only the selected token budget."""
        if self.holt_enabled:
            raise RuntimeError(
                'Legacy K routing is disabled while HOLT is enabled; '
                'Stage-A evaluates a fixed K=0 arm.'
            )
        if not self.uses_llm:
            raise RuntimeError('Utility routing requires an LLM branch.')
        if x_enc.ndim == 4:
            if x_enc.shape[-1] != 1:
                raise ValueError('4D routed input must have a singleton feature axis.')
            x_enc = x_enc.squeeze(-1)
        # The router and the selected forecast share one history-only
        # hyperedge cache, avoiding a second dense node-to-hyperedge reduction.
        innovation_cache = self.innovation_tokenizer.all_innovations(x_enc)
        features = history_complexity_features(
            x_enc, self.innovation_tokenizer, precomputed=innovation_cache
        )
        probabilities = self.utility_router.from_features(features).softmax(dim=-1)
        confidence, indices = probabilities.max(dim=-1)
        ranks = self.utility_router.rank_values[indices]
        ranks = torch.where(
            confidence >= confidence_threshold, ranks, torch.zeros_like(ranks)
        )
        output = x_enc.new_empty((x_enc.shape[0], self.pred_len, x_enc.shape[-1]))
        graph_cache = self._graph_forecast(x_enc) if self.uses_graph else None
        for rank_tensor in torch.unique(ranks, sorted=True):
            rank = int(rank_tensor.item())
            selected = ranks == rank_tensor
            output[selected] = self.forecast(
                x_enc[selected], x_mark_enc[selected], x_dec[selected], x_mark_dec[selected],
                interaction_rank=rank,
                innovation_cache=(
                    innovation_cache[0][selected], innovation_cache[1][selected]
                ),
                graph_cache=None if graph_cache is None else graph_cache[selected],
            )
        return output, ranks, probabilities
    


    
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):

        super(ReprogrammingLayer, self).__init__() 
        # Multi-head Attention Mechanism Approach
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)



    def forward(self, target_embedding, source_embedding, value_embedding):
        
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads   
        # Eigenvalue embedding, information embedding
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        
        out = out.reshape(B, L, -1)


        return self.out_projection(out)
    
    
    def reprogramming(self, target_embedding, source_embedding, value_embedding):

        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
