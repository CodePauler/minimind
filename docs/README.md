# MiniMind 完全指南

> 基于 MiniMind 源码的系统性学习文档，面向具备 NLP 基础和 Python/PyTorch 能力的开发者

## 📚 文档列表

| 序号 | 文档 | 内容概要 |
|:----:|------|----------|
| 1 | [整体架构总览 + Tokenizer](./MiniMind学习指南1_架构与Tokenizer.md) | MiniMind 全流程概览、数据流、Tokenizer 原理与 BPE |
| 2 | [Transformer 模型结构](./MiniMind学习指南2_Transformer模型结构.md) | Attention、RoPE、FFN、GQA、QK-Norm、forward 流程 |
| 3 | [预训练](./MiniMind学习指南3_预训练.md) | 数据加载、Next Token Prediction Loss、训练循环 |
| 4 | [SFT 监督微调](./MiniMind学习指南4_SFT监督微调.md) | Chat Template、Labels 生成、与 pretrain 的区别 |
| 5 | [RLHF / DPO](./MiniMind学习指南5_RLHF与DPO.md) | 对齐训练原理、DPO Loss、PPO 简介 |
| 6 | [推理](./MiniMind学习指南6_推理.md) | 自回归生成、KV Cache、Sampling、Streaming |
| 7 | [扩展指南](./MiniMind学习指南7_扩展指南.md) | RAG、Agent、Memory、二次开发位置 |

## 🎯 学习目标

- 理解 LLM 完整训练流程（Tokenizer → Pretrain → SFT → RLHF → Inference）
- 掌握 MiniMind 核心代码实现，能从源码级别理解模型
- 具备在 MiniMind 基础上进行二次开发的能力

## 📖 阅读前提

- 已学习过 NLP 基础知识（如 CS224N 前半部分）
- 有一定 Python / PyTorch 基础
- 不熟悉完整 LLM pipeline

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/CodePauler/minimind.git
cd minimind

# 安装依赖
pip install -r requirements.txt

# 下载模型
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

# 运行推理
python eval_llm.py --load_from ./minimind-3

# 训练预训练模型
cd trainer && python train_pretrain.py

# 训练 SFT
cd trainer && python train_full_sft.py
```

## 📂 文档结构说明

```
minimind/
├── docs/                          # 学习文档目录
│   ├── MiniMind学习指南1_架构与Tokenizer.md
│   ├── MiniMind学习指南2_Transformer模型结构.md
│   ├── MiniMind学习指南3_预训练.md
│   ├── MiniMind学习指南4_SFT监督微调.md
│   ├── MiniMind学习指南5_RLHF与DPO.md
│   ├── MiniMind学习指南6_推理.md
│   └── MiniMind学习指南7_扩展指南.md
├── model/
│   ├── model_minimind.py          # 核心模型实现
│   └── model_lora.py              # LoRA 实现
├── dataset/
│   └── lm_dataset.py              # 数据集加载
├── trainer/
│   ├── train_pretrain.py          # 预训练
│   ├── train_full_sft.py          # SFT
│   ├── train_dpo.py               # DPO
│   └── ...
├── eval_llm.py                    # 推理入口
└── README.md                      # 项目官方说明
```

## 🔗 相关链接

- [MiniMind GitHub](https://github.com/CodePauler/minimind)
- [HuggingFace 模型](https://huggingface.co/jingyaogong/minimind-3)
- [ModelScope 体验](https://www.modelscope.cn/studios/gongjy/minimind)

---

*文档版本：v1.0 | 更新日期：2026-04-01*