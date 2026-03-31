# 第三部分：模型结构（Transformer）

## 3.1 为什么是 Transformer？

在 MiniMind 之前，NLP 主流模型是 RNN/LSTM。但它们有致命问题：
- **梯度消失**：长序列训练困难
- **并行困难**：必须顺序计算，训练慢

2017 年，Google 论文《Attention Is All You Need》带来了 Transformer：
- **完全基于 Attention**：并行计算，训练快
- **任意距离建模**：Attention 让任意两个位置直接交互
- **可扩展**：堆叠更多层就能提升效果

> 💡 **类比**：Transformer 就像一个"全连接的社交网络"——每个人（token）都能直接和任何人说话，不需要通过中间人（RNN 的隐藏状态）。

## 3.2 MiniMind 模型整体结构

### 核心代码路径

- **配置文件**：`model/model_minimind.py` → `MiniMindConfig`
- **模型主体**：`model/model_minimind.py` → `MiniMindModel` + `MiniMindForCausalLM`
- **核心组件**：`model/model_minimind.py` → `Attention`, `FeedForward`, `MiniMindBlock`

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MiniMindForCausalLM (完整模型)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   input_ids ──▶ Embedding ──▶ Transformer Blocks ──▶ LM Head ──▶ logits │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                        MiniMindModel                             │  │
│   │  ┌─────────────────────────────────────────────────────────────┐ │  │
│   │  │              MiniMindBlock × num_layers                    │ │  │
│   │  │  ┌────────────────┐    ┌────────────────┐                 │ │  │
│   │  │  │   Attention    │───▶│  FeedForward   │                 │ │  │
│   │  │  │  (Multi-Head)  │    │   (FFN/SwiGLU) │                 │ │  │
│   │  │  │   + RMSNorm    │    │   + RMSNorm    │                 │ │  │
│   │  │  │   + RoPE       │    │                │                 │ │  │
│   │  │  └────────────────┘    └────────────────┘                 │ │  │
│   │  └─────────────────────────────────────────────────────────────┘ │  │
│   │                           │                                       │  │
│   │                           ▼ Final RMSNorm                        │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 配置参数

```python
# 来自 model_minimind.py
class MiniMindConfig(PretrainedConfig):
    def __init__(self, 
        hidden_size=768,           # 隐藏层维度
        num_hidden_layers=8,       # Transformer 层数
        vocab_size=6400,           # 词表大小
        num_attention_heads=8,     # Attention 头数
        num_key_value_heads=4,     # KV 头数 (GQA)
        head_dim=96,               # 每个头的维度
        intermediate_size=12288,   # FFN 中间层维度
        max_position_embeddings=32768,  # 最大位置编码长度
        rope_theta=1e6,            # RoPE 基础频率
        use_moe=False,             # 是否使用 MoE
        # ... 
    ):
```

### 简化版伪代码

```python
class MiniMindForCausalLM(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniMindBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids):
        # 1. Embedding
        x = self.embed_tokens(input_ids)
        
        # 2. Transformer Blocks
        for layer in self.layers:
            x = layer(x)
        
        # 3. Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
```

## 3.3 Attention 实现

### 为什么需要 Attention？

传统 RNN 是"一条线"传递信息，Transformer 是"全连接"——每个 token 都能看到所有其他 token。

**本质**：Attention 就是"加权求和"——给每个位置的 token 分配不同的重要性权重。

### MiniMind 的 Attention 实现

```python
# 来自 model_minimind.py
class Attention(nn.Module):
    def __init__(self, config):
        # QKV 投影
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        # RMSNorm for QK
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
    
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False):
        bsz, seq_len, _ = x.shape
        
        # 1. QKV 投影 + reshape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        # 2. RMSNorm (QK-Norm，这是 MiniMind 的特色！)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        
        # 3. RoPE 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        
        # 4. KV Cache (推理加速)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        # 5. 重复 KV 头 (因为 num_kv_heads < num_heads)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        
        # 6. 计算 Attention
        if self.flash and seq_len > 1:
            # FlashAttention (高效实现)
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        else:
            # 手动实现 (用于推理或长序列)
            scores = xq @ xk.transpose(-2, -1) / sqrt(head_dim)
            scores = scores.masked_fill(~attention_mask, -inf)
            output = softmax(scores) @ xv
        
        # 7. 输出投影
        output = self.o_proj(output)
        return output, past_kv
```

### 关键点解析

#### 1️⃣ GQA (Grouped Query Attention)

```python
# 普通 MHA: num_heads = num_kv_heads
# GQA: num_heads > num_kv_heads (通常 num_heads = 2 * num_kv_heads)

# MiniMind 默认: num_heads=8, num_kv_heads=4
# 这意味着 4 个 KV 头要服务 8 个 Q 头
```

**为什么这样设计？**
- KV Cache 是推理瓶颈，减少 KV 头数可以大大减少显存
- 通过 `repeat_kv` 把一个 KV 复制给多个 Q

#### 2️⃣ QK-Norm (RMSNorm on Q and K)

```python
xq, xk = self.q_norm(xq), self.k_norm(xk)
```

**这是 MiniMind 区别于标准 Transformer 的关键点**：
- 原始 Transformer：没有 QK-Norm
- LLaMA/Qwen：使用了 QK-Norm
- **作用**：提升训练稳定性，防止注意力分数爆炸

> 💡 **直觉**：QK-Norm 就像"温度调节"——把 Query 和 Key 都"标准化"一下，让它们的值域保持稳定，模型更好训练。

#### 3️⃣ FlashAttention

```python
if self.flash:
    output = F.scaled_dot_product_attention(...)
```

- **普通 Attention**：O(N²) 显存（需要显式计算 N×N 矩阵）
- **FlashAttention**：O(N) 显存（分块计算，累加求和）
- **效果**：支持更长的序列，训练更快

## 3.4 RoPE 位置编码

### 为什么需要位置编码？

Attention 本身是"位置无关"的——"我爱你"和"你爱我"在 Attention 看来是一样的。

**解决思路**：给每个位置添加一个"位置信号"，让模型能区分"谁在谁前面"。

### RoPE 的核心思想

**RoPE (Rotary Position Embedding)** 的核心：**旋转**！

```python
# 核心公式
def apply_rotary_pos_emb(q, k, cos, sin):
    # 把位置信息"编织"进 Q 和 K 向量
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    # 旋转 180 度
    return torch.cat([-x[..., dim/2:], x[..., :dim/2]], dim=-1)
```

> 💡 **类比**：RoPE 就像给每个位置发一个"旋转角度"的号码牌。随着位置增加，旋转角度也增加。这样两个位置之间的"相对角度"就编码了它们的相对位置关系。

### MiniMind 的 RoPE 实现

```python
def precompute_freqs_cis(dim, end, rope_base=1e6, rope_scaling=None):
    # 生成频率
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2) / dim))
    
    # 如果使用 YaRN (长度外推)
    if rope_scaling is not None:
        # 应用 YaRN 缩放
        freqs = freqs * (1 - ramp + ramp / factor)
    
    # 生成位置 × 频率 的网格
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    
    # cos 和 sin 就是位置编码
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
```

### YaRN (长度外推)

```python
# MiniMind 支持 YaRN，可以在推理时处理更长的序列
rope_scaling = {
    "type": "yarn",
    "factor": 16,           # 扩展 16 倍
    "original_max_position_embeddings": 2048,
}
```

**YaRN 的作用**：
- 训练时最长 2048 tokens
- 推理时可以外推到 32768 tokens
- 原理：平滑地调整 RoPE 的频率，让模型"泛化"到更长序列

## 3.5 FFN (前馈网络)

### 标准 FFN

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)  # SwiGLU 的 gate
        self.up_proj = nn.Linear(hidden_size, intermediate_size)    # up 投影
        self.down_proj = nn.Linear(intermediate_size, hidden_size)  # down 投影
    
    def forward(self, x):
        # SwiGLU 激活函数
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))
```

### SwiGLU 激活函数

```python
# SwiGLU = SiLU (Swish) + Gated Linear Unit
# SiLU(x) = x * sigmoid(x)
# 效果比 ReLU/GELU 更好，是 LLaMA/Qwen 等模型的标配
```

> 💡 **直觉**：SwiGLU 就像"智能门卫"——`gate_proj` 决定"要不要让这个信息通过"，`up_proj` 提供"具体的内容"，两者相乘就是"有选择地传递信息"。

## 3.6 MoE (混合专家) - 可选模块

### 什么是 MoE？

传统 FFN 每个 token 都经过**所有**参数。MoE 的核心思想：**只激活部分专家**。

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(hidden_size, num_experts)  # 路由器
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(num_experts)])
    
    def forward(self, x):
        # 1. 计算每个专家的"选择分数"
        scores = F.softmax(self.gate(x), dim=-1)
        
        # 2. 选择 top-k 专家
        topk_weight, topk_idx = torch.topk(scores, k=num_experts_per_tok)
        
        # 3. 只计算被选中的专家
        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            y[mask] += expert(x[mask]) * weight[mask]
        
        return y
```

### MiniMind 的 MoE 配置

```python
# MiniMind-MoE: 198M / 64M (激活参数 / 总参数)
# 4 个专家，每次激活 1-2 个
config = MiniMindConfig(
    use_moe=True,
    num_experts=4,
    num_experts_per_tok=1,  # 或 2
)
```

**与真实模型的差异**：
- Mixtral (8×7B)：8 个专家，激活 2 个
- DeepSeek-MoE：更复杂的负载均衡策略
- MiniMind：简化版，但核心思想一样

## 3.7 Forward 过程逐步拆解

### 完整流程

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
    batch_size, seq_len = input_ids.shape
    
    # Step 1: Embedding
    hidden_states = self.embed_tokens(input_ids)
    
    # Step 2: 位置编码
    start_pos = past_key_values[0][0].shape[1] if past_key_values else 0
    cos, sin = self.freqs_cos[start_pos:start_pos+seq_len], self.freqs_sin[start_pos:start_pos+seq_len]
    position_embeddings = (cos, sin)
    
    # Step 3: 堆叠的 Transformer Blocks
    presents = []  # 用于 KV Cache
    for layer in self.layers:
        hidden_states, present = layer(
            hidden_states,
            position_embeddings,
            past_key_value=past_key_values[i] if past_key_values else None,
            use_cache=use_cache
        )
        presents.append(present)
    
    # Step 4: 最终 LayerNorm
    hidden_states = self.norm(hidden_states)
    
    # Step 5: LM Head (与 Embedding 共享权重)
    logits = self.lm_head(hidden_states)
    
    return logits, presents
```

### 数据流动图

```
input_ids: [batch, seq_len]
    │
    ▼
┌─────────────────────────────────────────────┐
│  Embedding Layer                            │
│  [batch, seq_len] → [batch, seq_len, hidden]│
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  MiniMindBlock × N (8 layers)               │
│                                             │
│  For each layer:                            │
│    1. RMSNorm(input)                        │
│    2. Attention(Q, K, V) + 残差             │
│    3. RMSNorm(output)                       │
│    4. FFN(output) + 残差                    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Final RMSNorm                              │
│  [batch, seq_len, hidden]                   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  LM Head (Linear)                           │
│  [batch, seq_len, hidden] → [batch, seq_len, vocab]│
└─────────────────────────────────────────────┘
    │
    ▼
logits: [batch, seq_len, vocab_size]
```

## 3.8 与标准 Transformer 的差异

| 特性 | 标准 Transformer | MiniMind |
|------|-----------------|----------|
| **Position Encoding** | Sinusoidal | RoPE |
| **LayerNorm** | Pre-LN | Pre-LN + RMSNorm |
| **Attention Norm** | 无 | QK-Norm (RMSNorm on Q/K) |
| **FFN 激活** | ReLU/GELU | SwiGLU |
| **FFN 结构** | 2 层 Linear | 3 层 (gate+up+down) |
| **Attention 类型** | MHA | GQA (Grouped Query) |
| **MoE** | 无 | 可选 |

> ⚠️ **重要**：MiniMind 的这些改进（如 QK-Norm、SwiGLU）都是 **GPT-3.5 之后** 主流模型的做法，学到就是赚到！

## 3.9 总结

### 关键 takeaways

1. **Transformer 是 LLM 的"骨骼"**：一切 Transformer 变体的基础
2. **Attention 是"信息枢纽"**：让每个 token 都能访问所有其他 token
3. **RoPE 是"位置记忆"**：用旋转编码位置，比 Sinusoidal 更容易扩展到长序列
4. **GQA + KV Cache 是推理加速的关键**：减少显存占用，加快生成速度
5. **QK-Norm + SwiGLU 是稳定训练的秘密**：让大模型更好训练

### 面试可能问的问题

> **Q1: Transformer 为什么比 RNN 更好？**
> 
> 并行计算效率高（Attention 不依赖顺序）；任意位置直接交互（长距离依赖建模能力强）；可扩展性好（堆叠层数即可提升容量）。

> **Q2: 什么是 GQA？为什么能加速推理？**
> 
> GQA (Grouped Query Attention) 让多个 Query 头共享同一个 Key/Value 头。减少 KV Cache 显存占用，从而支持更长上下文、更快推理速度。

> **Q3: RoPE 的核心思想是什么？为什么适合长文本？**
> 
> RoPE 通过旋转矩阵把位置编码进向量，相對位置关系自然编码在旋转角度中。YaRN 等技术可以平滑外推到更长序列。

> **Q4: SwiGLU 相比 ReLU 有什么优势？**
> 
> SwiGLU = SiLU + 门控机制。门控让模型可以选择性地激活通道，信息流更可控，效果通常比 ReLU/GELU 更好。

---

*文档版本：v1.0*
*项目：MiniMind - 大模型全流程开源实现*
*更新日期：2026-04-01*