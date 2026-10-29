# Language Modeling from Scratch

基于 Stanford CS336 Assignment 1，从零实现并训练一个 decoder-only Transformer Language Model。

本项目尽量减少对高层封装的依赖，完整实现了从数据处理、模型结构、优化器、训练流程到文本生成的核心模块，并通过一系列 controlled ablation experiments 分析不同架构与训练策略对收敛速度、稳定性和最终性能的影响。

## 项目简介

本项目的主要目标是从底层理解并实现一个完整的 Language Modeling pipeline，而不仅仅是调用现成的 Transformer API。

整体流程包括：

- 训练 BPE tokenizer，并完成数据集 tokenization
- 实现 Multi-Head Self-Attention
- 实现 Rotary Position Embedding（RoPE）
- 实现 RMSNorm 与 Pre-Norm Transformer Block
- 实现 SwiGLU Feed-Forward Network
- 实现 Cross-Entropy Loss、AdamW optimizer 与 Gradient Clipping
- 搭建完整的 training / validation / checkpoint pipeline
- 接入 experiment logging
- 实现基于 temperature 与 nucleus sampling（top-p）的 autoregressive decoding
- 进行 normalization、positional encoding、FFN design 等 controlled ablation experiments

模型主要在 TinyStories 数据集上训练，在单张消费级 GPU 环境下累计处理约 **40M training tokens**，并在此基础上进一步完成 learning rate tuning 与多组架构消融实验。

## 项目亮点

- **从零实现完整 Transformer Language Model**：包括 Attention、RoPE、RMSNorm、SwiGLU、Cross-Entropy Loss、AdamW、Gradient Clipping、Checkpointing 与 Decoding 等核心组件。
- **搭建完整训练流程**：支持 training、validation、checkpoint recovery、experiment logging 与可配置的 hyperparameters。
- **完成 learning rate tuning**：实验表明 `5e-4 ~ 1e-3` 是当前配置下表现较稳定且收敛较快的学习率区间。
- **系统分析训练稳定性**：在移除 RMSNorm、Gradient Clipping 等实验中观察到 gradient / logit explosion，部分配置最终发生 loss divergence 与 NaN。
- **对比 Pre-Norm 与 Post-Norm**：Post-Norm 能够正常训练，但 validation 表现和后期稳定性略差。
- **评估 positional encoding 与 FFN 设计**：移除 RoPE 后模型收敛速度和最终 loss 均明显下降；参数量近似匹配的 SwiGLU 与 SiLU 差异相对较小。
- 详细实验设置、曲线与结论见 [`EXPERIMENTS.md`](./EXPERIMENTS.md)。

## 模型结构与核心实现

本项目实现了一个标准的 decoder-only Transformer Language Model。整体结构可以概括为：

`Token Embedding → N × Transformer Block → RMSNorm → LM Head`

其中每个 Transformer Block 采用 **Pre-Norm** 结构：

`RMSNorm → Multi-Head Self-Attention → Residual → RMSNorm → SwiGLU FFN → Residual`

### 核心模块

- **Token Embedding**  
  将离散 token ID 映射到 `d_model` 维的连续表示。

- **Multi-Head Self-Attention**  
  手动实现 Q/K/V projection、scaled dot-product attention、causal mask、multi-head reshape 与 output projection。

- **Rotary Position Embedding (RoPE)**  
  在 Query 和 Key 上施加 rotary positional encoding，使 attention 能够建模 token 间的相对位置信息。

- **RMSNorm**  
  使用 Root Mean Square 进行特征归一化，并采用 Pre-Norm 方式放置在 Attention 与 FFN 之前。

- **SwiGLU Feed-Forward Network**  
  使用 gated activation 构建 FFN，相比普通两层 MLP 引入额外的 gating branch。

- **Linear / Softmax / Cross-Entropy Loss**  
  实现训练与推理过程中需要的基础算子，并尽量减少对高层 Pytorch neural network API 的直接依赖。

- **AdamW Optimizer**  
  手动实现 AdamW 的 first / second moment 更新、bias correction 与 decoupled weight decay。

- **Gradient Clipping**  
  根据所有参数梯度的 global L2 norm 进行 clipping，用于抑制异常梯度造成的训练不稳定。

### Training Pipeline

在模型实现之外，项目还搭建了完整的训练与评估流程，包括：

- 使用 `np.memmap` / memory-mapped arrays 读取 tokenized dataset
- 随机采样固定长度 context 构造 training batch
- Learning Rate Scheduling
- 定期 validation
- Checkpoint 保存与恢复
- W&B experiment logging
- Autoregressive text generation
- Temperature sampling 与 nucleus sampling (`top-p`)

这些模块共同组成了从 raw text、tokenization、model training 到 text generation 的完整 Language Modeling pipeline。

## Training Setup

主要实验基于 **TinyStories** 数据集进行，模型在单张消费级 GPU 上完成训练与消融实验。

### 模型配置

| 配置项 | 设置 |
|---|---|
| Architecture | Decoder-only Transformer |
| Number of Layers | 4 |
| Number of Heads | 16 |
| Vocabulary Size | 10,000 |
| Context Length | 256 |
| d_model | 512 |
| d_ff | 1344 |
| Parameters | 22.70M total / 5.12M embedding / 17.58M non-embedding |


### 训练配置

| 配置项 | 设置 |
|---|---|
| Dataset | TinyStories |
| Batch Size | 24 |
| Training Steps | 6,500 |
| Training Tokens | ≈ 40M |
| Optimizer | AdamW |
| Gradient Clipping | Enabled |
| Experiment Tracking | Weights & Biases |
| Hardware | RTX 2060 / L4 |

Learning rate 等对结果影响较大的超参数通过独立实验进行调优，具体设置、训练曲线与分析见 [`EXPERIMENTS.md`](./EXPERIMENTS.md)。

## 实验结果摘要

在完成基础训练后，本项目围绕 learning rate、normalization、Transformer normalization placement、positional encoding 与 FFN design 进行了多组 controlled ablation experiments。

### Key Findings

- **Learning Rate**  
  在当前模型与训练规模下，`5e-4 ~ 1e-3` 表现较稳定，其中较高 learning rate 能显著加快前期收敛；当 learning rate 继续增大到 `3e-3` 时，训练出现明显不稳定并最终发散。

- **RMSNorm 与训练稳定性**  
  移除 RMSNorm 后，模型对 learning rate 更敏感。在 `lr=1e-3` 下出现频繁的 loss spikes；降低至 `3e-4` 后能够明显缓解，但整体稳定性仍弱于 baseline。

- **Gradient Clipping**  
  在 `No RMSNorm + lr=3e-4` 的基础上进一步移除 Gradient Clipping 后，训练在约 4,700 steps 后发生彻底发散，loss 上升至 `1e25` 量级并最终变为 NaN，同时伴随 gradient norm 与 logits 爆炸，说明 Gradient Clipping 对异常梯度具有重要的稳定作用。

- **Pre-Norm vs Post-Norm**  
  Post-Norm 可以正常完成训练，但整体 validation loss 略高，后期波动也更明显；相比之下，Pre-Norm 在当前配置下表现出更稳定的优化行为。

- **RoPE**  
  移除 RoPE 后，模型的收敛速度与最终 loss 均明显下降，说明 positional information 对当前 Language Modeling 任务具有显著影响。

- **SwiGLU vs SiLU**  
  在尽量匹配 FFN parameter count 的情况下，SwiGLU 与 SiLU 的最终表现差异相对较小，但 SwiGLU 在后期平均 loss 和最低 loss 上略占优势。

详细实验设置、训练曲线、统计指标与分析见 [`EXPERIMENTS.md`](./EXPERIMENTS.md)。

## Text Generation

训练完成后，使用 autoregressive decoding 进行文本生成。每一步根据当前 context 预测下一个 token，并将新生成的 token 继续加入输入序列，直到达到最大生成长度或遇到结束条件。

推理阶段支持：

- **Temperature Sampling**：通过调整 logits 的分布控制生成结果的随机性
- **Top-p (Nucleus) Sampling**：仅从累计概率达到指定阈值的候选 token 中进行采样
- **Greedy Decoding**：选择当前概率最高的 token 作为下一步输出

实际生成时使用 `temperature=0.5`、`top_p=0.8`，训练后的模型已经能够生成具有基本语法结构和局部连贯性的 TinyStories 风格文本。

### Example

```text
Prompt:
Once upon a time

Generated:
Once upon a time, there was a little girl named Lily. She had a pet cat named Tom. They liked to play together in the park. One day, they saw a big tree with a swing. Lily and Tom wanted to play on the swing.
Lily said to Tom, "Let's go down the tree and play on the swing." Tom agreed, and they went to the swing. They were very happy. But then, something unexpectedhappened. A big wind came and blew the swing away.
Lily and Tom were sad. They looked for the swing, but they could not find it. They were very sad. Then, a kind bird saw them and helped them get the swing back. The bird gave the swing back to Lily and Tom. They all played on the swing together and had a great day.
```

由于模型规模较小、训练 token 数约为 40M，生成质量仍然有限，但能够观察到模型已经学习到 TinyStories 数据中的基本语言结构、常见叙事模式与短距离 token dependencies。

## Repository Structure

```text
.
├── cs336_basics/
│   ├── model/              # Transformer、training、decoding 等核心实现
│   ├── tokenizer/          # BPE tokenizer 与数据预处理
│   └── ...
├── data/                   # Tokenized datasets / preprocessing outputs
├── scripts/                # 实验、训练或辅助脚本
├── figures/                # Training curves 与实验结果图
├── EXPERIMENTS.md          # 完整实验设置、结果与分析
└── README.md
```

核心实现主要位于 `cs336_basics/` 下，包括 tokenizer、Transformer architecture、optimizer、training loop 与 decoding 等模块；实验结果与详细分析集中记录在 [`EXPERIMENTS.md`](./EXPERIMENTS.md) 中。

## Usage

项目使用 `uv` 管理 Python 环境与依赖。

### 1. 安装依赖

```bash
uv sync
```

### 2. 准备数据

完成 BPE tokenizer 训练后，将 TinyStories 数据预处理为 tokenized dataset。

```bash
uv run python <tokenizer_or_preprocessing_script>.py
```

具体数据路径与 tokenizer 配置可根据本地环境进行调整。

### 3. 训练模型

```bash
uv run python -m cs336_basics.model.train \
    --train_data <train_data_path> \
    --val_data <val_data_path> \
    [other arguments...]
```

训练脚本支持 validation、checkpoint 保存与恢复、experiment logging，以及 learning rate 等主要超参数配置。

### 4. 文本生成

训练完成后，可加载 checkpoint 进行 autoregressive text generation：

```bash
uv run python -m cs336_basics.model.infer \
    --checkpoint <checkpoint_path> \
    [other arguments...]
```

推理阶段支持 temperature、top-p 等 decoding 参数。

## Acknowledgements

本项目基于 Stanford **CS336: Language Modeling from Scratch** Assignment 1 的课程框架与实验要求完成。

在课程要求的基础上，从零实现了 Language Modeling pipeline 中的核心组件，并进一步对训练过程中的 learning rate、normalization、gradient clipping、positional encoding 与 FFN architecture 等因素进行了系统实验与分析。

课程主页：[Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)