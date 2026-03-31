# 第四部分：预训练（Pretraining）

## 4.1 预训练的目标是什么？

**一句话概括**：让模型学会 "预测下一个词"（Next Token Prediction）。

这是 LLM 的**基座能力**——模型在预训练阶段学的不是"如何回答问题"，而是"语言本身的规律"：语法、语义、世界知识、推理能力...

> 💡 **类比**：预训练就像"语文老师让学生大量阅读文章"。学生不需要老师教"如何回答问题"，而是书读多了，自然就学会了遣词造句、连词成句。**广泛阅读**就是预训练。

## 4.2 数据是如何进入模型的

### 核心代码路径

- **数据加载**：`dataset/lm_dataset.py` → `PretrainDataset`
- **训练入口**：`trainer/train_pretrain.py`
- **训练循环**：`train_pretrain.py` → `train_epoch()` 函数

### 数据格式

```json
// dataset/pretrain_t2t_mini.jsonl (每行一个 JSON)
{"text": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器..."}
```

### 数据加载流程

```python
# 来自 lm_dataset.py
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        # 直接加载 JSONL 文件
        self.samples = load_dataset('json', data_files=data_path, split='train')
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Step 1: 获取原始文本
        text = str(sample['text'])
        
        # Step 2: Tokenize (添加特殊 token)
        tokens = tokenizer(text, add_special_tokens=False, 
                          max_length=self.max_length - 2, truncation=True).input_ids
        
        # Step 3: 添加 BOS/EOS
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        
        # Step 4: Padding (用 pad_token 补齐到固定长度)
        input_ids = tokens + [tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        # Step 5: 创建 labels (pad_token 位置标记为 -100，不计算 loss)
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100
        
        return torch.tensor(input_ids), torch.tensor(labels)
```

### 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                    原始数据 (JSONL)                              │
│  {"text": "今天天气很好，我想去公园散步"}                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tokenizer                                    │
│  "今天天气很好" → [1234, 5678, 9012, 3456, 7890]               │
│  添加 BOS: [1, 1234, 5678, 9012, 3456, 7890, 2]                │
│  Padding: [1, 1234, 5678, 9012, 3456, 7890, 2, 0, 0, 0, ...]   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    训练数据                                      │
│  input_ids:  [1, 1234, 5678, 9012, 3456, 7890, 2, 0, 0, 0]     │
│  labels:     [1, 1234, 5678, 9012, 3456, 7890, 2, -100, -100]  │
│                                                                  │
│  预测目标：                                                     │
│  token_0 → 预测 token_1 (1234)                                  │
│  token_1 → 预测 token_2 (5678)                                  │
│  ...                                                            │
│  token_5 → 预测 token_6 (2)                                     │
│  token_6 → 预测 token_7 (pad，忽略)                             │
└─────────────────────────────────────────────────────────────────┘
```

## 4.3 Loss 如何计算

### Next Token Prediction Loss

```python
# 来自 model_minimind.py
def forward(self, input_ids, labels=None, ...):
    # 前向传播
    hidden_states = self.model(input_ids)
    logits = self.lm_head(hidden_states)
    
    if labels is not None:
        # Shift: 预测 "下一个" token
        # logits: [batch, seq_len, vocab]
        # labels: [batch, seq_len]
        
        # logits 去掉最后一个 token
        x = logits[:, :-1, :]    # [batch, seq_len-1, vocab]
        # labels 去掉第一个 token
        y = labels[:, 1:]        # [batch, seq_len-1]
        
        # Cross Entropy Loss
        loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
    
    return loss
```

### 计算流程图

```
logits: [batch, seq_len, vocab_size]
    │
    │          ┌─────────────────────────────────────────────────┐
    │          │ 假设 seq_len = 5, vocab_size = 6400             │
    │          │                                                 │
    │          │  input_ids:  [1,  1234, 5678, 9012, 3456]       │
    │          │  labels:      [     1234, 5678, 9012, 3456 ]    │
    │          │                       ↓                         │
    │          │      去掉第一个和最后一个位置                    │
    │          │      只保留中间的有效 token                      │
    └──────────┼─────────────────────────────────────────────────┤
               ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Cross Entropy Loss                                       │
    │                                                          │
    │  for each position i (except first and last):           │
    │    loss_i = -log( softmax(logits[i])[label[i]] )        │
    │                                                          │
    │  最终 loss = mean(loss_i) over all valid positions      │
    │  (忽略 labels == -100 的位置)                            │
    └──────────────────────────────────────────────────────────┘
               │
               ▼
         scalar loss
```

### Auxiliary Loss (MoE 专用)

```python
# 如果使用 MoE，还有一个额外的负载均衡 loss
aux_loss = sum([l.mlp.aux_loss for l in self.layers])
total_loss = loss + aux_loss
```

**为什么需要 aux_loss？**
- MoE 的路由器可能导致"专家负载不均"——某些专家被选太多，某些没人用
- aux_loss 强制让所有专家都有机会被选中

## 4.4 训练循环解析

### 核心代码

```python
# 来自 train_pretrain.py
def train_epoch(epoch, loader, iters, start_step=0):
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 1. 数据移到 GPU
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # 2. 学习率调度 (Cosine Annealing)
        lr = get_lr(epoch * iters + step, total_steps, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 3. 前向传播 (混合精度)
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss  # 主 loss + MoE aux loss
            loss = loss / accumulation_steps  # 梯度累积
        
        # 4. 反向传播
        scaler.scale(loss).backward()
        
        # 5. 梯度裁剪 & 更新
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 6. 日志 & 保存
        if step % log_interval == 0:
            print(f"loss: {loss.item() * accumulation_steps:.4f}")
```

### 关键训练技术

#### 1. 混合精度训练 (AMP)

```python
# 训练使用 bfloat16/float16，推理快，省显存
dtype = torch.bfloat16
with torch.cuda.amp.autocast(dtype=dtype):
    output = model(input_ids)
```

#### 2. 梯度累积 (Gradient Accumulation)

```python
# 物理 batch_size = 32，但 GPU 显存不够
# 模拟 batch_size = 32 * 8 = 256
accumulation_steps = 8
loss = loss / accumulation_steps  # 先累积
```

#### 3. 梯度裁剪 (Gradient Clipping)

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

#### 4. 学习率调度

```python
# MiniMind 使用 Cosine Annealing
def get_lr(step, total_steps, max_lr):
    # 先线性warmup，再余弦衰减
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

## 4.5 与真实 LLM 训练的差异

| 特性 | MiniMind | 真实大模型 |
|------|----------|-----------|
| **数据量** | ~10MB | ~10TB |
| **训练 Token 数** | ~10M | ~1-10T |
| **训练轮数** | 1-3 epochs | 1-2 epochs |
| **Batch Size** | 32 | 4K-32K |
| **优化器** | AdamW | AdamW + 8-bit 压缩 |
| **学习率** | 5e-4 | 1e-4 ~ 3e-4 |
| **Scheduler** | Cosine | Cosine + Warmup |
| **分布式** | DDP | DeepSpeed ZeRO-3 |
| **数据管道** | 简化的 | 高度优化 (prefetch, mmap) |

### 真实预训练的"黑科技"

1. **数据质量过滤**：去重、清洗、低质量过滤
2. **数据混合策略**：不同来源的数据按比例混合
3. **课程学习**：先学简单数据，再学复杂数据
4. **多模态数据**：图文交替训练
5. **持久化数据管道**：避免 IO 成为瓶颈

## 4.6 总结

### 关键 takeaways

1. **预训练 = 广泛阅读**：让模型大量学习"下一个词是什么"
2. **数据格式很简单**：text → tokenize → add BOS/EOS → padding
3. **Loss 就是 Cross Entropy**：预测下一个 token 的概率分布
4. **训练技术很关键**：混合精度、梯度累积、梯度裁剪、lr scheduler
5. **规模决定能力**：更多数据 + 更大模型 = 更强能力

### 面试可能问的问题

> **Q1: 为什么预训练不需要标注数据？**
> 
> 因为任务是"预测下一个词"——这是自然存在的监督信号。互联网上所有文本都可以用作训练数据。

> **Q2: 为什么要添加 BOS 和 EOS token？**
> 
> BOS (Begin of Sequence) 告诉模型"一个序列开始了"；EOS (End of Sequence) 告诉模型"在这里停止"。这帮助模型学习序列边界。

> **Q3: 为什么 labels 中 padding 部分设为 -100？**
> 
> Cross Entropy 的 `ignore_index=-100` 会跳过这些位置的 loss。如果不这样做，模型会学习"预测 pad token"，这是无意义的。

> **Q4: 预训练数据需要清洗吗？**
> 
> 需要。真实训练会去除低质量数据（如 HTML、重复内容、脏话）、去重、可能的话做安全过滤。

---

*文档版本：v1.0*
*项目：MiniMind - 大模型全流程开源实现*
*更新日期：2026-04-01*