# 第七部分：推理（Inference）

## 7.1 推理的目标是什么？

**一句话概括**：把训练好的模型变成"对话机器人"——用户输入文本，模型输出回答。

推理与训练最大的**区别**：
- **训练**：批量并行，一次计算整个序列的 loss
- **推理**：自回归生成，一次只生成**一个** token，然后把这个 token 追加到输入，再生成下一个...

> 💡 **类比**：训练就像"批量生产"，推理就像"流水线"——每个新 token 都要等前一个 token 生成完毕。

## 7.2 自回归生成原理

### 核心流程

```python
# 来自 model_minimind.py
@torch.inference_mode()
def generate(self, inputs, max_new_tokens=8192, ...):
    # inputs: [batch, seq_len] 的 token IDs
    
    for _ in range(max_new_tokens):
        # 1. 只取最后 position 开始预测 (利用 KV Cache)
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = self.forward(
            input_ids[:, past_len:],  # 只输入新 token
            past_key_values=past_key_values,
            use_cache=True
        )
        
        # 2. 获取最后一个位置的 logits
        next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]
        
        # 3. Sampling (采样下一个 token)
        next_token = sampling(next_token_logits)
        
        # 4. 追加到序列
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # 5. 更新 KV Cache
        past_key_values = outputs.past_key_values
        
        # 6. 检查是否遇到 EOS
        if finished.all():
            break
    
    return input_ids
```

### 逐步图解

```
Step 1: 输入 "今天天气"
        tokens: [1, 1234, 5678, 9012]
        │
        ▼ 模型 forward
        预测下一个 token: "很好"
        输出: [1, 1234, 5678, 9012, 3456]

Step 2: 输入 "[包含 KV Cache]"
        只输入: [3456] (新 token)
        │
        ▼ 模型 forward (利用缓存的 K, V)
        预测下一个 token: "。"
        输出: [1, 1234, 5678, 9012, 3456, 2]

Step 3: 输入 "[包含 KV Cache]"
        只输入: [2] (EOS)
        │
        ▼ 检测到 EOS，停止
        完成！
```

## 7.3 KV Cache（关键优化）

### 为什么需要 KV Cache？

**问题**：自回归生成中，每个新 token 都需要重新计算**所有历史 token 的 Attention**。

**例子**：
- 生成第 1 个 token：计算 1 个 token 的 attention
- 生成第 2 个 token：计算 2 个 token 的 attention
- 生成第 100 个 token：计算 100 个 token 的 attention
- **总计算量**：1 + 2 + 3 + ... + N = O(N²)

### KV Cache 的原理

**核心思想**：缓存已经计算过的 Key 和 Value！

```
第一次 forward:
  input: [BOS, "今天"]
  计算: Q1, K1, V1
  缓存: K1, V1
  输出: next_token

第二次 forward:
  input: ["天"]  ← 只输入新 token
  计算: Q2, K2, V2
  缓存: [K1, K2], [V1, V2]
  Attention: Q2 @ [K1, K2]  ← 包含了所有历史！
  输出: next_token

...以此类推
```

### MiniMind 的 KV Cache 实现

```python
# 来自 model_minimind.py
def forward(self, x, past_key_value=None, use_cache=False):
    # 1. 计算当前的 QKV
    xq = self.q_proj(x)  # [batch, 1, hidden]
    xk = self.k_proj(x)
    xv = self.v_proj(x)
    
    # 2. 如果有缓存，拼接历史 K 和 V
    if past_key_value is not None:
        past_k, past_v = past_key_value
        xk = torch.cat([past_k, xk], dim=1)  # 拼接！
        xv = torch.cat([past_v, xv], dim=1)
    
    # 3. 保存当前 K, V 以便下次使用
    if use_cache:
        present = (xk, xv)
    else:
        present = None
    
    # 4. 计算 attention (只需要当前的 Q 和 拼接后的 K, V)
    output = attention(xq, xk, xv)
    
    return output, present
```

### KV Cache 的显存问题

```
假设:
- batch_size = 1
- seq_len = 8192
- num_layers = 32
- num_heads = 8
- head_dim = 128
- 每个 float16 = 2 bytes

KV Cache 大小:
= 2 × batch × layers × seq × (heads × head_dim × 2 for K and V)
= 2 × 1 × 32 × 8192 × (8 × 128 × 2)
≈ 536 MB
```

**优化技术**：
- GQA（减少 KV 头数）
- 量化（fp16 → int8）
- PagedAttention（vLLM 的核心技术）

## 7.4 Sampling（采样策略）

### 核心代码

```python
# 来自 model_minimind.py generate() 函数

# 1. Temperature (温度)
logits = logits / temperature  # 调整概率分布

# 2. Top-K (限制候选词数量)
if top_k > 0:
    top_k_logits = torch.topk(logits, top_k)
    logits[logits < top_k_logits[-1]] = -float('inf')

# 3. Top-P / Nucleus Sampling (核采样)
if top_p < 1.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumsum = torch.cumsum(softmax(sorted_logits), dim=-1)
    # 保留概率之和超过 top_p 的词
    sorted_mask = cumsum > top_p
    logits[sorted_indices[sorted_mask]] = -float('inf')

# 4. Repetition Penalty (重复惩罚)
if repetition_penalty != 1.0:
    logits[input_ids] /= repetition_penalty

# 5. 最终采样
next_token = torch.multinomial(softmax(logits), num_samples=1)
```

### 采样策略对比

| 策略 | 作用 | 效果 |
|------|------|------|
| **Greedy (argmax)** | 总是选概率最高的 | 确定性强，可能重复 |
| **Temperature** | 调整概率分布的"平滑度" | 高温→更随机，低温→更确定 |
| **Top-K** | 只保留概率最高的 K 个 | 过滤低概率词 |
| **Top-P (Nucleus)** | 保留概率之和达 P 的词 | 自适应过滤 |
| **Repetition Penalty** | 惩罚出现过的词 | 减少重复 |

## 7.5 Streaming（流式输出）

### 什么是 Streaming？

普通输出：等模型生成完整个回答 → 一次性返回
流式输出：生成一个 token → 返回一个 token → 用户立刻看到

### MiniMind 的 Streaming 实现

```python
# 来自 eval_llm.py
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 调用 generate 时传入 streamer
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    streamer=streamer,  # ← 关键参数
    ...
)

# TextStreamer 会:
# 1. 每生成一个 token
# 2. 立即 decode 并打印
# 3. 不等待完整输出
```

### 原理图解

```
普通模式:
  用户: "你好" ─────────▶ 模型 ─────────▶ "你好，我是AI很高兴认识你！"
                                           │
                                           └─ 全部生成完才返回

流式模式:
  用户: "你好" ─────────▶ 模型 ──▶ "你好" ──▶ "你好，我是" ──▶ "你好，我是AI" ──▶ ...
                                           │           │              │            │
                                           ▼           ▼              ▼            ▼
                                      立即显示   立即显示      立即显示     立即显示
```

## 7.6 推理代码入口

### 快速测试

```python
# 方式1: 直接运行评估脚本
python eval_llm.py --load_from ./minimind-3 --weight full_sft

# 方式2: 命令行交互
# 会列出预设问题或让你手动输入
```

### 关键参数

```python
# 来自 eval_llm.py
parser.add_argument('--weight', default='full_sft')       # 权重名称
parser.add_argument('--max_new_tokens', default=8192)     # 最大生成长度
parser.add_argument('--temperature', default=0.85)        # 温度
parser.add_argument('--top_p', default=0.95)              # 核采样
parser.add_argument('--open_thinking', default=0)         # 是否开启思考
parser.add_argument('--historys', default=0)              # 携带历史轮数
```

## 7.7 与真实推理系统的差异

| 特性 | MiniMind | 真实大模型 (生产级) |
|------|----------|-------------------|
| **推理引擎** | 原生 PyTorch | vLLM, llama.cpp, TensorRT-LLM |
| **KV Cache** | 基础实现 | PagedAttention, 连续 Batching |
| **量化** | 无 | FP16 → INT8 → INT4 |
| **多并发** | 不支持 | 动态 Batch、Continuous Batching |
| **分布式推理** | 不支持 | 流水线并行、张量并行 |
| **长文本** | 受限 | 64K-1M tokens |

## 7.8 总结

### 关键 takeaways

1. **推理 = 自回归生成**：每次生成一个 token，然后把 token 加入输入，再生成下一个
2. **KV Cache 是核心优化**：避免重复计算历史 token 的 Attention
3. **Sampling 控制多样性**：Temperature、Top-K、Top-P 调整输出风格
4. **Streaming 提升体验**：一个字一个字输出，而不是等全部生成完

### 面试可能问的问题

> **Q1: 为什么推理需要 KV Cache？不用会怎样？**
> 
> 不用 KV Cache 的话，每个新 token 都要重新计算所有历史 token 的 Attention，时间复杂度是 O(N²)。有了 KV Cache，时间复杂度降为 O(N)，推理速度快几十倍。

> **Q2: Temperature 和 Top-P 有什么区别？**
> 
> Temperature 调整整体概率分布的"平滑度"——高温使分布更均匀（更随机），低温使分布更尖锐（更确定）。Top-P 是动态选择候选词集合——只保留累计概率达到 P 的高概率词。

> **Q3: 什么是 Repetition Penalty？为什么需要它？**
> 
> Repetition Penalty 惩罚那些在已生成序列中出现过的 token，降低模型重复同一个词/句子的概率。这让生成结果更加多样和自然。

---

*文档版本：v1.0*
*项目：MiniMind - 大模型全流程开源实现*
*更新日期：2026-04-01*