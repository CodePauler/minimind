# 第六部分：RLHF / DPO（对齐训练）

## 6.1 为什么需要对齐？

**一句话概括**：SFT 让模型**会回答**，RLHF/DPO 让模型**回答得更好**——更有帮助、更安全、更符合人类偏好。

### 问题背景

即使经过 SFT，模型仍然可能：
- **有害输出**：回答涉及暴力、歧视、色情等问题
- **无效输出**：答非所问、过于冗长、格式混乱
- **不确定性**：对不确定的问题乱编答案（hallucination）

> 💡 **类比**：
> - SFT 就像教孩子"怎么说人话"
> - RLHF 就像教孩子"怎么做人"——要有礼貌、要有同理心、要做正确的事

## 6.2 RLHF vs DPO

### 两种技术对比

| 特性 | RLHF (PPO) | DPO (Direct Preference Optimization) |
|------|------------|---------------------------------------|
| **复杂度** | 高 (需要训练 Reward Model + PPO) | 低 (只需要偏好数据) |
| **训练时间** | 长 | 短 |
| **效果** | 好 | 足够好 |
| **实现难度** | 复杂 | 简单 |
| **MiniMind 支持** | ✅ (PPO/GRPO) | ✅ (DPO) |

### MiniMind 的对齐算法

```python
# trainer/ 目录下的对齐训练脚本
train_dpo.py      # DPO 对齐训练
train_ppo.py      # PPO 对齐训练
train_grpo.py     # GRPO 对齐训练
```

## 6.3 DPO 详解

### 核心代码路径

- **数据集**：`dataset/lm_dataset.py` → `DPODataset`
- **训练脚本**：`trainer/train_dpo.py`
- **Loss 函数**：`train_dpo.py` → `dpo_loss()`

### 数据格式

```json
{
  "chosen": [
    {"role": "user", "content": "如何煎牛排？"},
    {"role": "assistant", "content": "煎牛排的步骤：1. 牛排提前30分钟从冰箱拿出来..."}
  ],
  "rejected": [
    {"role": "user", "content": "如何煎牛排？"},
    {"role": "assistant", "content": "我不知道。"}
  ]
}
```

**核心思想**：给模型一个"好的回答"和一个"差的回答"，让它学习区分。

### DPO Loss 推导

```python
# 来自 train_dpo.py
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1):
    """
    DPO Loss: -log(sigmoid(beta * (log π(y|x) - log π(y¯|x))))
    
    其中：
    - π(y|x) 是当前(policy)模型对chosen回答的打分
    - π(y¯|x) 是当前模型对rejected回答的打分
    - β 是温度参数，控制偏好强度
    """
    # 1. 分别计算 chosen 和 rejected 的对数概率
    chosen_log_prob = (policy_log_probs[:N] * mask[:N]).sum(dim=-1)
    rejected_log_prob = (policy_log_probs[N:] * mask[N:]).sum(dim=-1)
    
    # 2. 计算两个参考模型的对数概率差
    ref_chosen_log_prob = (ref_log_probs[:N] * mask[:N]).sum(dim=-1)
    ref_rejected_log_prob = (ref_log_probs[N:] * mask[N:]).sum(dim=-1)
    
    # 3. 计算偏好差距
    pi_logratios = chosen_log_prob - rejected_log_prob
    ref_logratios = ref_chosen_log_prob - ref_rejected_log_prob
    logits = pi_logratios - ref_logratios
    
    # 4. Binary Cross Entropy
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

### DPO 训练流程

```python
# 来自 train_dpo.py
def train_epoch(epoch, loader, ref_model, ...):
    for step, batch in enumerate(loader):
        # 1. 获取数据 (chosen + rejected 配对)
        x_chosen, x_rejected = batch['x_chosen'], batch['x_rejected']
        y_chosen, y_rejected = batch['y_chosen'], batch['y_rejected']
        mask_chosen, mask_rejected = batch['mask_chosen'], batch['mask_rejected']
        
        # 2. 拼接 chosen 和 rejected
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        
        # 3. 参考模型 (冻结) 推理
        with torch.no_grad():
            ref_outputs = ref_model(x)
            ref_logits = ref_outputs.logits
        ref_log_probs = logits_to_log_probs(ref_logits, y)
        
        # 4. 策略模型 (可训练) 推理
        outputs = model(x)
        policy_log_probs = logits_to_log_probs(outputs.logits, y)
        
        # 5. 计算 DPO Loss
        loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta)
        
        # 6. 反向传播 (只更新策略模型)
        loss.backward()
```

### 图解 DPO

```
                    参考模型 (冻结)
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   π(y_chosen|x)   π(y_rejected|x)   β参数
        │                │                │
        │         ┌──────┴──────┐         │
        │         │  偏好差距    │         │
        │         │  Δ = logπ(c)│         │
        │         │      - logπ(r)         │
        │         └──────┬──────┘         │
        │                │                │
        ▼                ▼                ▼
   sigmoid(β * Δ)  ──▶  Loss = -log(sigmoid(β * Δ))
   
   目标：让 chosen 的概率远高于 rejected
```

## 6.4 RLHF (PPO) 简述

### PPO 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     RLHF (PPO) 训练流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ SFT 模型      │    │  Reward Model │    │   PPO 训练    │      │
│  │ (Policy)     │───▶│   (打分)       │───▶│  (策略优化)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │              │
│         │                   │                   │              │
│         │           ┌───────┴───────┐           │              │
│         │           │               │           │              │
│         │           ▼               ▼           │              │
│         │     reward(x,y)    baseline(x)        │              │
│         │           │               │           │              │
│         │           └───────┬───────┘           │              │
│         │                   │                   │              │
│         │                   ▼                   │              │
│         │           advantage = r - b           │              │
│         │                   │                   │              │
│         └───────────────────┼───────────────────┘              │
│                             ▼                                   │
│              ┌─────────────────────────────┐                   │
│              │  KL + PPO + Value Loss      │                   │
│              │  最终 loss = reward - kl    │                   │
│              └─────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### MiniMind 的 PPO 实现

```python
# 来自 trainer/train_ppo.py
# MiniMind 实现了完整的 PPO 流程:
# 1. Reward Model (奖励模型)
# 2. Value Model (价值模型)  
# 3. PPO 策略更新
```

## 6.5 与真实对齐训练的差异

| 特性 | MiniMind | 真实大模型 |
|------|----------|-----------|
| **Reward Model** | 不需要 (DPO) 或简化 | 独立训练 Reward Model |
| **数据规模** | 几千条 | 几十万条人类偏好标注 |
| **PPO 实现** | 简化版 | 复杂 (KL penalty, value loss, clipping) |
| **训练稳定** | 一般 | 需要大量调参 |
| **效果** | 基础对齐 | 高度对齐人类偏好 |

## 6.6 总结

### 关键 takeaways

1. **对齐 = 微调 + 偏好学习**：让模型不只是"会回答"，还要"回答得好"
2. **DPO 简单有效**：不需要 Reward Model，直接从偏好数据学习
3. **PPO 效果更好**：但实现复杂，需要更多工程努力
4. **核心思想**：让"好答案"的概率高于"坏答案"

### 面试可能问的问题

> **Q1: RLHF 和 SFT 有什么区别？为什么需要 RLHF？**
> 
> SFT 让模型学习"标准答案"，RLHF 让模型学习"人类偏好"。SFT 可能让模型输出安全但无用的回答，RLHF 可以让模型更有帮助、更安全、更自然。

> **Q2: DPO 相比 PPO 有什么优势？为什么？**
> 
> DPO 更简单，不需要单独训练 Reward Model，不需要复杂的 PPO 训练过程。DPO 直接从成对偏好数据学习，训练更稳定，效果也足够好。

> **Q3: 为什么 DPO 需要参考模型 (ref_model)？**
> 
> 参考模型提供了"没有学习前的基线"，用于计算相对改进。如果没有参考模型，模型可能会简单地提高所有 token 的概率，而不是真正学到偏好。

---

*文档版本：v1.0*
*项目：MiniMind - 大模型全流程开源实现*
*更新日期：2026-04-01*