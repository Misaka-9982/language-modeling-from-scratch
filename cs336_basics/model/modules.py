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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()

        std = math.sqrt(2 / (d_ff + d_model))
        W1_empty = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        W2_empty = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        W3_empty = torch.empty(d_ff, d_model, device=device, dtype=dtype)

        self.W1 = nn.Parameter(torch.nn.init.trunc_normal_(W1_empty, mean=0, std=std, a=-3*std, b=3*std))
        self.W2 = nn.Parameter(torch.nn.init.trunc_normal_(W2_empty, mean=0, std=std, a=-3*std, b=3*std))
        self.W3 = nn.Parameter(torch.nn.init.trunc_normal_(W3_empty, mean=0, std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1x = einops.einsum(self.W1, x, 'out in, ... in -> ... out')
        SiLU_W1x = torch.sigmoid(W1x) * W1x
        W3x = einops.einsum(self.W3, x, 'out in, ... in -> ... out')
        return einops.einsum(self.W2, SiLU_W1x * W3x, 'out in, ... in -> ... out')


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None=None):
        # d_k是传入向量，也就是query和key向量的维度
        super().__init__()

        # 预计算旋转矩阵sin和cos缓存
        # 利用广播机制快速生成theta_ik，避免显式的两层循环
        i = einops.rearrange(torch.arange(max_seq_len), 'max_seq_len -> max_seq_len 1')
        k = einops.rearrange(torch.arange(d_k // 2), 'd_k -> 1 d_k')
        theta_ik = i / (theta ** ((2 * k) / d_k))   # 利用广播机制
        sin_cached = torch.sin(theta_ik)
        cos_cached = torch.cos(theta_ik)
        # 注册成非学习的模型参数
        self.register_buffer('sin_cached', sin_cached, persistent=False)
        self.register_buffer('cos_cached', cos_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # cos_cached是动态挂载的，ide检查会出问题
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # # 对齐特征维度，从[1,2,3]变成[1,1,2,2,3,3,]
        # cos = einops.repeat(cos, '... d -> ... (d repeat)', repeat=2)
        # sin = einops.repeat(sin, '... d -> ... (d repeat)', repeat=2)

        x1 = x[..., 0::2]  # 成对切片取出二维小向量的第一，第二维度
        x2 = x[..., 1::2]  # start:end:step

        x1_out = x1 * cos - x2 * sin   # 旋转矩阵/旋转公式
        x2_out = x1 * sin + x2 * cos

        # 拼回完整的输入x
        x_out = torch.stack([x1_out, x2_out], dim=-1).flatten(start_dim=-2)   # 最后两个维度是行，列  
        # 也可以 x = rearrange(torch.stack((x1, x2), dim=-1), '... d two -> ... (d two)')

        return x_out
    