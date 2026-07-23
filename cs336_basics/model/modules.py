from typing import Any
from math import sqrt

import torch
import torch.nn as nn
import einops

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        W_empty = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = sqrt(2 / (in_features + out_features))
        # a b是前后截断的位置
        self.W = nn.Parameter(torch.nn.init.trunc_normal_(W_empty, mean=0, std=std, a=-3*std, b=3*std))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 单个输入看做行向量  工程实现中，特征维度一般都在最后
        return einops.einsum(self.W, x, 'out in, ... in -> ... out')
