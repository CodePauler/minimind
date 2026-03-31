# 第八部分：如何在此基础上扩展（实战指南）

## 8.1 为什么扩展很重要？

学完 MiniMind 你已经理解了 LLM 的全流程。但这只是起点——**真正的能力是用它来解决实际问题**。

本节会告诉你：
- 在哪些地方扩展最容易
- 具体怎么扩展
- 代码位置精确指引

## 8.2 如何接入 RAG（检索增强生成）

### 什么是 RAG？

**RAG (Retrieval-Augmented Generation)** = 检索 + 生成

**问题**：LLM 的知识是"死"的（训练截止日期之前的数据），无法回答实时问题或私有知识库问题。

**解决**：先从外部知识库检索相关内容，再让 LLM 基于这些内容生成回答。

### 整体架构

```
用户问题 ──▶ 检索系统 (向量数据库) ──▶ 获取相关文档
                                                    │
                                                    ▼
                                         与问题一起发送给 LLM
                                                    │
                                                    ▼
                                         LLM 生成基于检索内容的回答
```

### MiniMind + RAG 扩展方案

#### 方案 A：简单版（不修改模型）

```python
# 只需要修改推理部分
def rag_inference(user_query, knowledge_base, model, tokenizer):
    # 1. 检索相关文档 (用 embedding 模型)
    relevant_docs = retrieve(user_query, knowledge_base, top_k=3)
    
    # 2. 构建提示词
    context = "\n\n".join([doc.content for doc in relevant_docs])
    prompt = f"""基于以下参考资料回答用户问题。

参考资料：
{context}

用户问题：{user_query}

回答："""
    
    # 3. 调用 MiniMind 生成
    response = model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        max_new_tokens=1024
    )
    
    return response
```

#### 方案 B：进阶版（微调模型适配 RAG）

```python
# 扩展数据集格式
class RAGDataset(Dataset):
    def __getitem__(self, index):
        # 新格式：包含 retrieval results
        sample = {
            "query": "什么是 Python？",
            "retrieved_context": [
                "Python 是一种高级编程语言...",
                "Python 由 Guido van Rossum 创建..."
            ],
            "answer": "Python 是一种..."
        }
        
        # 构建输入：问题 + 检索结果
        prompt = f"""你是一个问答助手。请基于给定的参考资料回答问题。

参考资料：
{sample['retrieved_context']}

问题：{sample['query']}

回答："""
        
        return prompt, sample['answer']
```

**关键代码位置**：
- 数据集：`dataset/lm_dataset.py` → 新增 `RAGDataset` 类
- 训练脚本：`trainer/train_full_sft.py` → 修改数据路径

### 推荐工具

| 工具 | 用途 | 特点 |
|------|------|------|
| **LangChain** | RAG 框架 | 简单易用，集成度高 |
| **LangChain-Chatchat** | 本地 RAG | 开源，支持私有部署 |
| **FastGPT** | RAG 平台 | 可视化，无需编码 |
| **Milvus/QDrant** | 向量数据库 | 高效相似度检索 |

## 8.3 如何构建 Agent

### 什么是 Agent？

**Agent = LLM + 工具 + 记忆 + 规划**

与单纯对话不同，Agent 可以：
- **使用工具**：调用 API、搜索网页、执行代码
- **多轮推理**：先思考再行动（ReAct 模式）
- **自我反思**：结果不好，尝试其他方法

### MiniMind Agent 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Loop                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  用户    │───▶│  思考    │───▶│  行动    │───▶│  观察    │ │
│   │  输入    │    │ (LLM)    │    │(工具调用) │    │(结果)    │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        │              │               │               │        │
│        │              │               │               │        │
│        └──────────────┴───────────────┴───────────────┘        │
│                           │                                     │
│                           ▼                                     │
│                  是否完成任务？                                  │
│                     是 → 返回结果                                │
│                     否 → 继续循环                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### MiniMind 已有的 Agent 能力

```python
# 来自 trainer/train_agent.py
# MiniMind 支持:
# 1. Tool Calling (工具调用)
# 2. Agentic RL (Agent 强化学习)
# 3. 多轮对话 + 工具返回
```

### 如何扩展自己的 Agent

#### 步骤 1：定义工具

```python
# 定义可用的工具
AVAILABLE_TOOLS = [
    {
        "name": "search_web",
        "description": "搜索互联网获取信息",
        "parameters": {
            "query": "搜索关键词"
        }
    },
    {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "expression": "数学表达式，如 2+3*4"
        }
    },
    {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
            "city": "城市名称，如 北京"
        }
    }
]
```

#### 步骤 2：解析工具调用

```python
# 从模型输出中解析工具调用
def parse_tool_calls(response_text):
    """解析 <tool_call>...</tool_call> 格式"""
    import re
    pattern = r'<tool_call>\n({"name":.*?"})\n</tool_call>'
    matches = re.findall(pattern, response_text)
    
    tool_calls = []
    for match in matches:
        tool_calls.append(json.loads(match))
    
    return tool_calls
```

#### 步骤 3：执行工具

```python
# 执行工具并返回结果
def execute_tool(tool_name, tool_args):
    if tool_name == "search_web":
        return search_web(tool_args["query"])
    elif tool_name == "calculate":
        return eval(tool_args["expression"])
    elif tool_name == "get_weather":
        return get_weather(tool_args["city"])
    else:
        return f"未知工具: {tool_name}"
```

#### 步骤 4：构建 Agent Loop

```python
def agent_loop(user_query, model, tokenizer, tools):
    conversation = [
        {"role": "system", "content": "你是一个智能助手，可以使用工具来帮助用户。"},
        {"role": "user", "content": user_query}
    ]
    
    for turn in range(10):  # 最多 10 轮
        # 1. 生成回答
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        response_ids = model.generate(tokenizer(prompt))
        response = tokenizer.decode(response_ids[0])
        
        # 2. 检查是否需要工具调用
        tool_calls = parse_tool_calls(response)
        
        if not tool_calls:
            # 没有工具调用，直接返回回答
            conversation.append({"role": "assistant", "content": response})
            return response
        
        # 3. 执行工具
        for tool_call in tool_calls:
            tool_result = execute_tool(tool_call["name"], tool_call["arguments"])
            
            # 4. 把工具结果加入对话
            conversation.append({
                "role": "assistant", 
                "content": response  # 包含 tool_call
            })
            conversation.append({
                "role": "tool",
                "content": f"<tool_response>{tool_result}</tool_response>"
            })
```

**关键代码位置**：
- 工具调用支持：`model_minimind.py` 中的 chat_template
- Agent 训练：`trainer/train_agent.py`
- 推理示例：`scripts/chat_api.py`

## 8.4 如何增加 Memory（记忆）

### 为什么需要 Memory？

默认情况下，LLM **没有记忆**——每次对话都是全新的。用户说"帮我总结上次的会议"，模型根本不知道你在说什么。

### 方案对比

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **上下文窗口** | 把历史对话放在输入里 | 简单 | 受限于 max_seq_len |
| **摘要记忆** | 定期把对话摘要压缩 | 突破长度限制 | 可能丢失细节 |
| **向量记忆** | 把历史存到向量数据库 | 可检索 | 需要额外系统 |
| **外部存储** | 读写外部数据库/文件 | 灵活 | 复杂 |

### MiniMind 实现摘要记忆

```python
class ConversationMemory:
    def __init__(self, max_turns=10):
        self.messages = []
        self.max_turns = max_turns
    
    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        
        # 如果超过最大轮数，压缩历史
        if len(self.messages) > self.max_turns:
            self.compress()
    
    def compress(self):
        """把历史对话压缩成摘要"""
        # 构建压缩 prompt
        compress_prompt = f"""请简洁地总结以下对话的要点：

{self.format_history()}

摘要："""
        
        # 调用 LLM 生成摘要
        summary = llm.generate(compress_prompt)
        
        # 只保留摘要 + 最近几轮
        self.messages = [
            {"role": "system", "content": f"之前对话摘要：{summary}"}
        ] + self.messages[-self.max_turns:]
    
    def format_history(self):
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])
    
    def get_context(self):
        return self.format_history()
```

### 使用示例

```python
memory = ConversationMemory(max_turns=10)

# 对话
memory.add("user", "我喜欢蓝色")
memory.add("assistant", "蓝色是很漂亮的颜色！")
memory.add("user", "我的生日是5月1日")

# ... 几轮之后 ...

# 获取上下文
context = memory.get_context()
# 输出: "user: 我喜欢蓝色\nassistant: 蓝色是很漂亮的颜色！\nuser: 我的生日是5月1日"
```

## 8.5 二次开发最佳位置

### 按难度排序

| 优先级 | 位置 | 难度 | 说明 |
|--------|------|------|------|
| ⭐⭐⭐⭐⭐ | `dataset/lm_dataset.py` | ⭐ | 新增数据集类，适配自己的数据 |
| ⭐⭐⭐⭐⭐ | `eval_llm.py` | ⭐ | 修改推理逻辑，添加后处理 |
| ⭐⭐⭐⭐ | `trainer/train_full_sft.py` | ⭐⭐ | 调整训练超参数，更换数据 |
| ⭐⭐⭐⭐ | `model_minimind.py` | ⭐⭐⭐ | 修改模型结构（增加层数等） |
| ⭐⭐⭐ | `model/model_lora.py` | ⭐⭐⭐ | LoRA 微调实现 |
| ⭐⭐ | `trainer/train_agent.py` | ⭐⭐⭐⭐ | Agent 训练 |
| ⭐ | 各个 trainer/*.py | ⭐⭐⭐⭐ | 添加新的训练算法 |

### 最推荐的新手扩展

1. **自定义数据集**：改 `dataset/lm_dataset.py`，用自己的数据训练
2. **对话模板**：改 tokenizer 的 chat_template，支持新格式
3. **推理后处理**：改 `eval_llm.py`，添加过滤、格式化逻辑
4. **LoRA 微调**：用 `trainer/train_lora.py`，低成本微调

## 8.6 扩展示例：实现一个"私人助理"

### 目标

构建一个可以：
1. 记住用户的偏好
2. 搜索互联网回答实时问题
3. 执行简单计算

### 代码结构

```python
# personal_assistant.py

class PersonalAssistant:
    def __init__(self):
        # 1. 模型
        self.model, self.tokenizer = load_model()
        
        # 2. 工具
        self.tools = [search_web, calculate, get_weather]
        
        # 3. 记忆
        self.memory = ConversationMemory(max_turns=10)
    
    def chat(self, user_input):
        # 1. 获取记忆
        context = self.memory.get_context()
        
        # 2. 构建 prompt
        prompt = self.build_prompt(context, user_input)
        
        # 3. 生成响应 (Agent Loop)
        response = self.agent_loop(prompt)
        
        # 4. 记住对话
        self.memory.add("user", user_input)
        self.memory.add("assistant", response)
        
        return response
```

## 8.7 总结

### 关键 takeaways

1. **RAG = 检索 + 生成**：解决模型知识过时问题
2. **Agent = LLM + 工具 + 循环**：让模型能"动手做事"
3. **Memory = 对话历史**：突破上下文长度限制
4. **二次开发从数据开始**：最简单、效果最明显

### 进一步学习建议

| 方向 | 推荐资源 |
|------|----------|
| **RAG** | LangChain 文档、FastGPT |
| **Agent** | AutoGPT、ReAct 论文、LangChain Agents |
| **生产部署** | vLLM、llama.cpp、TensorRT-LLM |
| **模型优化** | quantization、pruning、distillation |

---

*文档版本：v1.0*
*项目：MiniMind - 大模型全流程开源实现*
*更新日期：2026-04-01*

---

# 🎉 文档完成！

## 学习路径总结

恭喜你完成了 MiniMind 的完整学习！以下是建议的学习路径：

```
第1步: 架构 + Tokenizer (本文第1-2部分)
    ↓
第2步: Transformer 模型 (第3部分)
    ↓
第3步: 预训练 (第4部分)
    ↓
第4步: SFT (第5部分)
    ↓
第5步: DPO/对齐 (第6部分)
    ↓
第6步: 推理 (第7部分)
    ↓
第7步: 实战扩展 (第8部分) ← 你在这里！
```

## 动手实践建议

1. **先跑通 Demo**：按照 README 运行训练和推理
2. **改一个小东西**：比如改数据集、调整参数
3. **尝试扩展**：按照第8部分的指南做自己的项目

---

**祝学习愉快！🚀**