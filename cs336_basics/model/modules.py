from typing import Any
import math

import torch
import torch.nn as nn
import einops

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        # cs336要求按照线性代数表示中列向量优先的方式，y = Wx中x是列向量，根据矩阵乘法，(out, in) *（in, 1）因此W.shape为out * in 
        # 但是工程实现中x通常最后一个维度为in_feature，x.shape==(batch_size, sequence_length, in_features)
        # 把每个向量一层层裹起来的感觉
        W_empty = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        # a b是前后截断的位置
        self.W = nn.Parameter(torch.nn.init.trunc_normal_(W_empty, mean=0, std=std, a=-3*std, b=3*std))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 工程实现中，特征维度(每个特征向量)一般都在最后(一层层包起来)  x.shape==(batch_size, sequence_length, in_features)
        return einops.einsum(self.W, x, 'out in, ... in -> ... out')


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        W_empty = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.W = nn.Parameter(torch.nn.init.trunc_normal_(W_empty, mean=0, std=1, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 会保持token_ids的维度结构，把整数id作为索引，从W中提取嵌入向量  (batch_size, sequence_length) -> (batch_size, sequence_length, d_model)
        return self.W[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        self.eps = eps

        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提高数据精度来避免溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 逐向量计算
        RMS_x = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        result = x * self.g / RMS_x

        return result.to(in_dtype)

        