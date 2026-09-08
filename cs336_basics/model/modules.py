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
        # 注册成非学习的buffer  预声明一下，防止后续ide静态类型检查报错
        self.sin_cached: torch.Tensor
        self.cos_cached: torch.Tensor
        self.register_buffer('sin_cached', sin_cached, persistent=False)
        self.register_buffer('cos_cached', cos_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # cos_cached是动态挂载的
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # # 对齐特征维度，从[1,2,3]变成[1,1,2,2,3,3,]
        # cos = einops.repeat(cos, '... d -> ... (d repeat)', repeat=2)
        # sin = einops.repeat(sin, '... d -> ... (d repeat)', repeat=2)

        # 这样拆出来就是对最内层小向量操作，外层维度全部保持，seq是通过token_positions的形状区分的
        x1 = x[..., 0::2]  # 成对切片取出二维小向量的第一，第二维度
        x2 = x[..., 1::2]  # start:end:step

        x1_out = x1 * cos - x2 * sin   # 旋转矩阵/旋转公式
        x2_out = x1 * sin + x2 * cos

        # 拼回完整的输入x
        x_out = torch.stack([x1_out, x2_out], dim=-1).flatten(start_dim=-2)   # 最后两个维度是行，列  
        # 也可以 x = rearrange(torch.stack((x1, x2), dim=-1), '... d two -> ... (d two)')

        return x_out


def softmax(x: torch.Tensor, dimension: int) -> torch.Tensor:
    max_values = x.max(dim=dimension, keepdim=True).values   # max函数返回的结构是命名元组 (values, indices)
    x_trunc = x - max_values
    out = torch.exp(x_trunc) / torch.sum(torch.exp(x_trunc), dim=dimension, keepdim=True)

    return out
    

# 完整的缩放点积注意力函数，接受QKV和可选的mask  输出为经过注意力加权聚合过的V向量
# mask的语义是：True = 允许 attention；False = 不允许 attention
def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, 
                                 mask: torch.Tensor | None=None) -> torch.Tensor:  # mask里的元素数据类型是bool
    # q @ k.T / sqrt(d_k)
    pre_softmax = einops.einsum(queries, keys, '... q d_k, ... k d_k -> ... q k') / math.sqrt(keys.shape[-1])

    # 应用mask
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))   # 注意masked_fill默认是true的位置填充，所以要用~取反，false填-inf
    
    attention_pattern = softmax(pre_softmax, dimension=-1)   # 逐行计算softmax
    weighted_values = einops.einsum(attention_pattern, values, '... q k, ... k d_v -> ... q d_v')  # 加权乘给v
    
    return weighted_values


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads   # 这里定义d_k = d_v
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

        # 注意这里的W_q W_k W_v 都是多个头直接拼在一起的
        self.q_proj = Linear(d_model, num_heads * self.d_k, device, dtype)   # 严格定义是d_model * (num_heads*d_k)，但是这里定义d_k = d_v = d_model / num_heads
        self.k_proj = Linear(d_model, num_heads * self.d_k, device, dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_v, device, dtype)
        self.o_proj = Linear(num_heads * self.d_v, d_model, device, dtype)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 拆出每个头的结果
        Q_seprated = einops.rearrange(Q, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        K_seprated = einops.rearrange(K, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        V_seprated = einops.rearrange(V, '... seq_len (num_heads d_v) -> ... num_heads seq_len d_v', num_heads=self.num_heads)

        # 对 Q K 逐头施加RoPE
        # 如果positions带batch维度[B, S]，会导致中间缺head维度，rope中会广播失败，所以补一个
        # cos: [B, 1, S, D]
        # Q/K: [B, H, S, D]
        token_positions = einops.rearrange(token_positions, '... seq_len -> ... 1 seq_len')  
        Q_seprated = self.rope(Q_seprated, token_positions)
        K_seprated = self.rope(K_seprated, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device
                                     , dtype=torch.bool))  # 生成下三角矩阵mask

        pre_out = scaled_dot_product_attention(Q_seprated, K_seprated, V_seprated, mask)  # pre_out的形状是(batch heads q d_v)

        # 传给O矩阵之前要把多个头的结果重新拼回去，变回 q * (h*d_v)  把不同头对于同一个token的输出拼成一个长向量
        pre_out = einops.rearrange(pre_out, '... num_heads q d_v -> ... q (num_heads d_v)')
 
        return self.o_proj(pre_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff:int, theta: float, max_seq_len: int, 
                 eps: float = 0.00001, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()

        self.rmsnorm1 = RMSNorm(d_model, eps, device, dtype)
        self.rmsnorm2 = RMSNorm(d_model, eps, device, dtype)
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len,
                                            device=device, dtype=dtype)
        self.swiglu_fnn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        # 生成token_positions的默认值
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
            
        x1 = x + self.mha(self.rmsnorm1(x), token_positions)
        x2 = x1 + self.swiglu_fnn(self.rmsnorm2(x1))

        return x2


class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff:int, rope_theta: float, 
                 vocab_size: int, context_length: int, num_layers: int, rms_eps: float = 0.00001, 
                 device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()

        self.input_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        # nn.ModuleList不能直接调用self.transformer_layers(x) 只是个容器
        self.transformer_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope_theta, 
                                                                  context_length, rms_eps, device, dtype)
                                                                  for _ in range(num_layers)])
        self.last_norm = RMSNorm(d_model, rms_eps, device, dtype)

        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)  # 内层维度映射回词表维度

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        x = self.input_embeddings(x)

        # nn.ModuleList不能直接调用self.transformer_layers(x) 只是个容器
        for layer in self.transformer_layers:
            x = layer(x, token_positions)

        x = self.last_norm(x)
        x = self.lm_head(x)   # 得到未归一化的logits

        return x

