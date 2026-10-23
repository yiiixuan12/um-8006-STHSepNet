"""KANLinear: wrap the pykan KANLayer (attachment) into a standard single-output
nn.Module with an nn.Linear-like interface (in_features / out_features).

The underlying KANLayer uses learnable B-spline basis functions; it returns
(y, preacts, postacts, postspline); we expose only y so the module can replace
an nn.Linear prediction head transparently.

Dtype handling: the surrounding STH-SepNet pipeline feeds bfloat16 tensors into
the prediction head. B-spline evaluations are numerically safer in float32, so
we compute in float32 and cast the output back to the input dtype.
"""

import torch
import torch.nn as nn

from .KANLayer import KANLayer


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-1., 1.), noise_scale=0.5, base_fun=None,
                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                 grid_eps=0.02, sp_trainable=True, sb_trainable=True,
                 device='cpu'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layer = KANLayer(
            in_dim=in_features,
            out_dim=out_features,
            num=grid_size,
            k=spline_order,
            noise_scale=noise_scale,
            scale_base_mu=scale_base_mu,
            scale_base_sigma=scale_base_sigma,
            scale_sp=scale_sp,
            base_fun=base_fun if base_fun is not None else nn.SiLU(),
            grid_eps=grid_eps,
            grid_range=list(grid_range),
            sp_trainable=sp_trainable,
            sb_trainable=sb_trainable,
            device=device,
        )

    def forward(self, x):
        orig_dtype = x.dtype
        leading_shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        y = self.layer(x2.float())[0]          # (N, out_features)
        return y.to(orig_dtype).reshape(*leading_shape, self.out_features)

    def extra_repr(self):
        return ('in_features={}, out_features={}, grid_size={}, spline_order={}'
                .format(self.in_features, self.out_features,
                        self.grid_size, self.spline_order))
