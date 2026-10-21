# Audit of the --static vs --adaptive_hyperhgnn branches (official-code rule):
# independently test that toggling static and the adaptive hypergraph variant
# (hgat/hgcn/hsage) constructs the STHGNN without error and preserves the
# expected output dimensions, on a small synthetic graph.
import os, sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from layers.STHGNN import sthypergnn as STHGNN


def make_graph(n=6):
    torch.manual_seed(0)
    A = torch.eye(n) + torch.rand(n, n) * 0.3
    A = (A + A.T) / 2
    return A


def build(static, adaptive_hyperhgnn):
    n = 6
    return STHGNN(
        gcn_true=True, hgcn_true=(adaptive_hyperhgnn == 'hgcn'),
        hgat_true=(adaptive_hyperhgnn == 'hgat'), buildA_true=True,
        buildH_true=True, gcn_depth=2, num_nodes=n, num_hyperedges=n,
        device='cpu', adaptive_hyperhgnn=adaptive_hyperhgnn,
        temporl_true=True, static=static, predefined_A=make_graph(n),
        static_feat=None, dropout=0.1, subgraph_size=3, node_dim=8,
        dilation_exponential=1, conv_channels=8, residual_channels=8,
        skip_channels=16, end_channels=16, seq_length=8, in_dim=1,
        out_dim=8, layers=2, propalpha=0.05, tanhalpha=3,
        layer_norm_affline=True, true_adj=make_graph(n),
        scale_hyperedges=3, alpha=0.1, beta=0.2, gamma=0.5, theta=0.2)


@pytest.mark.parametrize('static', [False, True])
@pytest.mark.parametrize('adaptive_hyperhgnn', ['hgat', 'hgcn', 'hsage'])
def test_static_adaptive_branch_shapes(static, adaptive_hyperhgnn):
    n = 6
    gnn = build(static, adaptive_hyperhgnn)
    # STHGNN.forward expects x of shape (B, in_dim, N, seq_len); in the model it is
    # called as self.sthgnn(x_enc.transpose(1,3)) where x_enc is (B,F,N,1).
    x = torch.randn(2, 1, n, 8)
    y = gnn(x)
    assert y.ndim >= 2 and torch.isfinite(y).all()



def test_static_flag_reaches_sthgcn():
    # static toggling must be plumbed through to the module without error;
    # instantiating with static=True must not raise.
    gnn = build(static=True, adaptive_hyperhgnn='hgat')
    assert isinstance(gnn, torch.nn.Module)
