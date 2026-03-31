# 第五部分：SFT（监督微调）

## 5.1 SFT 的作用是什么？

**一句话概括**：预训练让模型**会说 话**（语言能力），SFT 让模型**会回答问题**（指令遵循能力）。

### 对比

| 阶段 | 目标 | 数据 | 能力 |
|------|------|------|------|
| **Pretrain** | 预测下一个词 | 任意文本 | 语言建模 |
| **SFT** | 听懂指令并回答 | 对话数据 | 指令遵循 |

> 💡 **类比**：
> - 预训练：就像学生大量阅读小说、新闻、百科全书，学会了语言表达
> - SFT：就像学生去上"新东方厨师学校"，学习"有人问'怎么做饭'时该怎么回答"

## 5.2 SFT 数据格式

### 核心代码路径

- **数据加载**：`dataset/lm_dataset.py` → `SFTDataset`
- **训练入口**：`trainer/train_full_sft.py`

### 数据格式 (JSONL)

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "什么是人工智能？"},
    {"role": "assistant", "content": "人工智能是..."}
  ]
}
```

### 数据预处理流程

```python
# 来自 lm_dataset.py
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def create_chat_prompt(self, conversations):
        # 使用 chat_template 格式化对话
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True  # 关键：添加 generation prompt
        )
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 1. 格式化对话
        prompt = self.create_chat_prompt(conversations)
        
        # 2. Tokenize
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # 3. Padding
        input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))
        
        # 4. 生成 labels (只计算 assistant 回复部分)
        labels = self.generate_labels(input_ids)
        
        return torch.tensor(input_ids), torch.tensor(labels)
```

### Chat Template 格式化结果

```python
# 输入对话
messages = [
    {"role": "system", "content": "你是一个有用的AI助手"},
    {"role": "user", "content": "什么是AI？"},
    {"role": "assistant", "content": "AI 是人工智能的缩写..."}
]

# chat_template 格式化后
"""
<|im_start|>system
你是一个有用的AI助手<|im_end|>
<|im_start|>user
什么是AI？<|im_end|>
<|im_start|>assistant
AI 是人工智能的缩写，它是计算机科学的一个分支...
<|im_end|>
"""
```

### Generation Prompt 的关键作用

```python
# add_generation_prompt=True 在最后添加
# 输出:
"""
<|im_start|>assistant
"""  # ← 告诉模型：接下来该你说了！
```

## 5.3 Labels 生成策略

这是 SFT 和 Pretrain 最关键的**区别**！

### Pretrain 的 Labels

```python
# 来自 PretrainDataset
input_ids:  [1, 1234, 5678, 9012, 3456, 2, 0, 0, 0]
labels:     [1, 1234, 5678, 9012, 3456, 2, -100, -100, -100]
# 所有位置都预测下一个 token
```

### SFT 的 Labels

```python
# 来自 SFTDataset.generate_labels()
input_ids:  [1, 100, 200, 300, 400, 2, ...]
labels:     [-100, -100, -100, -100, 400, 2, ...]
#        ↑  ↑  ↑              ↑
#     system  user       assistant (只计算这部分)
```

### 图解对比

```
Pretrain:
  input:    [BOS] [今] [天] [天] [气] [很] [好] [EOS]
  label:    [今]   [天] [天] [气] [很] [好] [EOS] [PAD]
             ↑     ↑    ↑    ↑    ↑    ↑    ↑
          预测每个"下一个" token

SFT:
  input:    [BOS][system][用户问题...][assistant][模型回答...]
  label:    [-100][-100][-100][-100][模型回答][EOS]
                                    ↑
                            只计算 assistant 部分！
```

> 💡 **关键理解**：SFT 的核心思想是 **"只训练回答部分，忽略问题部分"**。模型只需要学习"如何回答"，不需要学习"如何复述问题"。

## 5.4 SFT 与 Pretrain 的区别

| 方面 | Pretrain | SFT |
|------|----------|-----|
| **目标** | 预测下一个词 | 学会回答问题 |
| **数据格式** | 任意文本 | 对话格式 (user/assistant) |
| **Labels** | 所有位置 | 只有 assistant 部分 |
| **学习率** | 较大 (如 5e-4) | 较小 (如 1e-5) |
| **Epochs** | 1-3 | 2-10 |
| **Loss** | 交叉熵 | 交叉熵 |

### 代码对比

```python
# Pretrain 的 loss 计算
def forward(self, input_ids, labels):
    logits = self.model(input_ids)
    # 预测所有 token
    loss = F.cross_entropy(logits[:, :-1], labels[:, 1:])

# SFT 的 loss 计算 (完全一样！)
def forward(self, input_ids, labels):
    logits = self.model(input_ids)
    # 预测所有 token
    loss = F.cross_entropy(logits[:, :-1], labels[:, 1:])
    # 但 labels 中问题部分是 -100，会被忽略
```

> ⚠️ **重要**：代码完全一样！但因为 **labels 不同**，效果完全不同。

## 5.5 训练配置差异

```python
# pretrain: 大学习率，短训练
train_pretrain.py:
    --learning_rate 5e-4
    --epochs 2
    --batch_size 32

# sft: 小学习率，长训练
train_full_sft.py:
    --learning_rate 1e-5      # 10x smaller!
    --epochs 2
    --batch_size 16
```

**为什么 SFT 学习率更小？**
- 预训练权重已经学到了很多知识
- 大学习率会导致 **灾难性遗忘** (Catastrophic Forgetting)
- 小学习率可以保留预训练能力，同时学习新任务

## 5.6 总结

### 关键 takeaways

1. **SFT = 指令微调**：让模型学会"听懂人话、会回答问题"
2. **Chat Template 是关键**：统一对话格式，让模型知道"谁在说话"
3. **Generation Prompt**：告诉模型"接下来该你回复了"
4. **Labels 只计算回答部分**：这是 SFT 和 pretrain 的核心区别
5. **学习率要小**：避免灾难性遗忘

### 面试可能问的问题

> **Q1: 为什么 SFT 需要比 pretrain 更小的学习率？**
> 
> 预训练阶段模型已经学会了语言能力，如果用大学习率微调，模型可能会"忘记"这些能力（灾难性遗忘）。小学习率可以在学习新任务的同时保留原有能力。

> **Q2: SFT 的 labels 是怎么生成的？为什么只计算 assistant 部分？**
> 
> 通过识别 `<|im_start|>assistant` 和 `<|im_end|>` 标记来确定 assistant 的内容范围。只计算 assistant 部分是因为我们只希望模型学习"如何回答"，而不需要学习"如何重复用户的问题"。

> **Q3: 什么是 Generation Prompt？**
> 
> 在 chat_template 中，add_generation_prompt=True 会在最后添加 `<|im_start|>assistant\n`，这相当于告诉模型"现在轮到你回复了"。在推理时，这个 prompt 引导模型开始生成回答。

> **Q4: SFT 数据不够会怎么样？**
> 
> 如果 SFT 数据太少或质量不佳，模型可能无法很好地遵循指令，可能会过度拟合到特定格式，或者丧失预训练阶段学到的部分能力。

---

*文档版本：v1.0*
*项目：MiniMind - 大模型全流程开源实现*
*更新日期：2026-04-01*