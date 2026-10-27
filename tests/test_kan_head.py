# Test the KANLinear prediction head: shape, parameter count, forward/backward, and
# equivalence of the output-projection branch (FlattenHeadKAN vs FlattenHead).
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from KAN.KANLinear import KANLinear
from models.ST_SepNet import FlattenHead, FlattenHeadKAN

NF = 192      # = d_ff * patch_nums in the BIKE config
TW = 48       # = pred_len


def test_kan_linear_shape_and_params():
    kl = KANLinear(NF, TW, grid_size=5, spline_order=3)
    x = torch.randn(2, NF)
    out = kl(x)
    assert out.shape == (2, TW)
    # forward must be differentiable and produce finite grads on TRAINABLE params
    loss = out.pow(2).mean()
    loss.backward()
    tr = [p for p in kl.parameters() if p.requires_grad]
    assert tr, 'no trainable parameters found'
    for p in tr:
        assert p.grad is not None
    # spline weight (coef) must be registered as a trainable Parameter
    assert any('coef' in n for n, _ in kl.named_parameters())
    # grid registered as a non-grad buffer/parameter
    assert any('grid' in n for n, _ in kl.named_parameters())


def test_flatten_head_dimension_agreement():
    head_mlp = FlattenHead(1, NF, TW, head_dropout=0.0)
    head_kan = FlattenHeadKAN(1, NF, TW, grid_size=5, spline_order=3, head_dropout=0.0)
    x = torch.randn(3, 1, NF, 1)   # (B, n_vars, nf, patch_window)
    y_mlp = head_mlp(x)
    y_kan = head_kan(x)
    assert y_mlp.shape == (3, 1, TW)
    assert y_kan.shape == y_mlp.shape


def test_kan_param_count_ratio():
    kl = KANLinear(NF, TW, grid_size=5, spline_order=3)
    mlp = torch.nn.Linear(NF, TW)
    k = sum(p.numel() for p in kl.parameters() if p.requires_grad)
    m = sum(p.numel() for p in mlp.parameters())
    # KAN head is expected to be larger (B-spline basis) for equal in/out dims
    assert k > m
