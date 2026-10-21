"""Synthetic forward/backward sanity (CPU, no LLM/weights needed).

Run directly:  python -m tests.synthetic_sanity
Used by scripts/smoke_test.sh step 3.

Builds the prediction head (MLP and KAN) on synthetic features, checks output
shapes, runs a backward pass, and prints a compact summary. It does NOT load
transformers or real data, so it runs in a clean, offline environment.
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.ST_SepNet import FlattenHead, FlattenHeadKAN

B, N_VARS, NF, TW = 4, 295, 192, 48


def main():
    torch.manual_seed(0)
    x = torch.randn(B, N_VARS, NF, 1)
    mlp = FlattenHead(N_VARS, NF, TW, head_dropout=0.0)
    kan = FlattenHeadKAN(N_VARS, NF, TW, grid_size=5, spline_order=3, head_dropout=0.0)

    y_mlp = mlp(x)
    y_kan = kan(x)
    assert y_mlp.shape == y_kan.shape == (B, N_VARS, TW), (y_mlp.shape, y_kan.shape)

    loss = y_kan.pow(2).mean()
    loss.backward()
    n_mlp = sum(p.numel() for p in mlp.parameters())
    n_kan = sum(p.numel() for p in kan.parameters())

    print(f'FlattenHead (MLP)      out {tuple(y_mlp.shape)} params {n_mlp}')
    print(f'FlattenHeadKAN          out {tuple(y_kan.shape)} params {n_kan}')
    print('forward + backward OK; shapes match; loss finite =',
          bool(torch.isfinite(loss).item()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
