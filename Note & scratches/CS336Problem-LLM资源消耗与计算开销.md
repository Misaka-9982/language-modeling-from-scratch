# Tranformer LM 消耗资源计算

## Problem (a)  
Consider a GPT-2 XL-sized model using our assignment architecture, which has the following
> configuration:
vocab_size: 50,257  
context_length: 1,024  
num_layers: 48  
d_model: 1,600  
num_heads: 25  
d_ff: 4,288 (the nearest multiple of 64 to 8/3 × 1, 600)  

Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?  

从输入开始：  
> Embeddings层：嵌入矩阵W.shape==(num_embeddings, embedding_dim)  
代入：50257 * 1600 == 80,411,200  

> Transformer Block * num_layers  48层：  
> * 2层RMSNome的g参数：2 * (d_model*1)。
代入：2 * 1600 == 3200  
> * MHA的 $W_{q}$  $W_{k}$ $W_{v}$ $W_{o}$ 四个
参数矩阵共d_model * d_model个参数。代入：4 * 1600 * 1600 == 10,240,000
> * FNN的 $W_{1}$  $W_{2}$ $W_{3}$ 共 d_ff * d_model 个参数。代入：4288 * 1600 * 3 == 20,582,400
> * LastRMSNorm：d_model 个，1600
> * Lm_head：d_model * vocab_size个。代入：1600 * 50257 == 80,411,200  

共计：80411200 + 48 * (3200 + 10240000 + 20582400) + 1600 + 80411200 == 1640452800 约1.64B  
用float32存储大约6.56GB  

---
## Problem (b)

Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.  

* Embedding层无矩阵乘法  
* Transformer Block * num_layers  48层：
    * RMSNorm不算
    * MHA：  
        * 计算QKVO：2 * seq_len * d_model * d_k * num_heads * 4 == 2 * 1024 * 1600 * 64 * 25 * 4 == 20971520000  
        * 点积注意力内部：  
        Q @ K.T：2 * seq_len * d_k * num_heads * seq_len == 2 * 1024 * 1024 * 1600 == 3355443200  
        注意力加权 * V：2 * seq_len * seq_len * d_v * num_heads == 2 * 1024 * 1024 * 1600 == 3355443200
    * FNN：
        * W123 * x：2 * d_ff * d_model * seq_len * 3 == 2 * 4288 * 1600 * 1024 * 3 == 42152755200
* LastRMSNorm：无矩阵乘法
* LM-Head：2 * d_model * seq_len * vocab_size == 2 * 1600 * 1024 * 50257 == 164682137600  

总计：(20971520000 + 3355443200 + 3355443200 + 42152755200) * 48 + 164682137600 == 3516769894400  
约3517BFLOPs，3.517TFLOPs  
---
## Problem (c)
Based on your analysis above, which parts of the model require the most FLOPs?  
* Transformer Block的FNN部分计算量最大。
---
## Problem (d)
Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium
(24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As
the model size increases, which parts of the Transformer LM take up proportionally more or
less of the total FLOPs?

* MHA部分，时间复杂度/计算开销为O(T^2 * d)，T为序列长度，d为模型维度d_model；FNN部分，开销为O(T * d^2)。  
* 模型的计算开销也主要由MHA和FNN这两部分构成。