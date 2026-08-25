# `lm_dataset.py` 介绍

对应源码：[`lm_dataset.py`](./lm_dataset.py)

## 文件定位

集中实现 MiniMind 各训练阶段的数据读取、聊天模板处理、序列截断和损失标签构造。

## 预处理函数

- `pre_processing_chat`：Tool Call 数据保持原样；普通对话可按概率补充中英文 system prompt。
- `post_processing_chat`：按概率移除空的 `<think>...</think>` 块，增加模板形式的多样性。

## 数据集类

| 类 | 输入数据 | 返回内容 | 使用方 |
| --- | --- | --- | --- |
| `PretrainDataset` | 含 `text` 的 JSON/JSONL | `input_ids, labels` | `train_pretrain.py` |
| `SFTDataset` | 含 `conversations` 的 JSONL | 输入序列和仅监督 assistant 区域的标签 | Full SFT、LoRA、蒸馏 |
| `DPODataset` | `chosen`/`rejected` 偏好对 | 两组输入、目标和损失掩码 | DPO |
| `RLAIFDataset` | 对话数据 | 待在线生成的 `prompt` | PPO、GRPO |
| `AgentRLDataset` | Agent 对话、工具和 GT | `messages`、`tools`、`gt` | Agent RL |

`PretrainDataset`、`SFTDataset` 和 `DPODataset` 会补齐或截断到固定长度，并用 `-100` 或显式 mask 排除不应参与损失计算的 token。

## 项目脉络

这个文件把磁盘上的 JSON/JSONL 数据转换为训练脚本可直接消费的张量或结构化样本，是“原始语料”和“模型损失”之间的适配层：

```text
JSON / JSONL
  └─> lm_dataset.py
        ├─> 文本清洗与聊天模板
        ├─> Tokenizer 编码
        ├─> 截断 / Padding
        └─> 标签或 Mask
              └─> trainer/train_*.py
                    └─> MiniMindForCausalLM
```

文件加载时会把 `TOKENIZERS_PARALLELISM` 设为 `false`。训练通常已经通过 DataLoader 多进程和 GPU 并行，关闭 Tokenizer 自身并行可以减少线程争用及 fork 相关警告。

## 各数据集的详细数据流

### `PretrainDataset`

预训练样本只需要一个 `text` 字段。单条数据的转换顺序是：

```text
sample['text']
  └─> 不添加 Tokenizer 默认特殊符号的编码
        └─> 截断到 max_length - 2
              └─> 手工添加 BOS / EOS
                    └─> PAD 到 max_length
```

最终返回两个形状均为 `[max_length]` 的 LongTensor：

- `input_ids`：模型输入。
- `labels`：初始复制自输入，padding 位置改为 `-100`。

模型内部会执行一位移位，用前一个 token 预测后一个 token。因此数据集不需要手工构造 `x=input[:-1]` 和 `y=input[1:]`。

### `SFTDataset`

SFT 的关键问题不是“如何编码整段对话”，而是“哪些 token 应该产生损失”。用户问题、system prompt 和工具结果是条件，不应要求模型去复述；只有 assistant 回复是监督目标。

处理流程如下：

1. 用固定 `Features` 描述 `conversations` 中的 `role`、`content`、`reasoning_content`、`tools` 和 `tool_calls` 字段。
2. `create_chat_prompt` 将字符串形式的 `tools`/`tool_calls` 解析为 JSON 对象。
3. 调用 `tokenizer.apply_chat_template` 还原成模型训练时使用的角色标记格式。
4. 查找“assistant 起始 token 序列”和“消息结束 token 序列”。
5. `generate_labels` 默认把所有位置设为 `-100`，只将 assistant 起止标记之间的位置恢复为真实 token id。

这种 label masking 让普通回答、thinking 内容和工具调用只要位于 assistant 消息中，就都可以成为学习目标；用户和工具消息只负责提供上下文。

### `DPODataset`

DPO 样本包含一条更受偏好的 `chosen` 对话和一条较差的 `rejected` 对话。两者分别应用聊天模板、截断并 padding，然后生成 assistant 区域的二值 `loss_mask`。

这里直接返回已经移位的六个张量：

```text
x_chosen, y_chosen, mask_chosen
x_rejected, y_rejected, mask_rejected
```

训练脚本把 chosen 与 rejected 沿 batch 维拼接，只在 mask 为 1 的位置累计序列 logprob。这样 prompt 长度或 padding 不会影响偏好差值。

### `RLAIFDataset`

PPO/GRPO 不使用数据文件里的标准回答进行教师强制训练，而是让当前策略在线生成。因此 `__getitem__` 只返回：

- `prompt`：去掉最后一条对话后，通过聊天模板生成的待续写上下文。
- `answer`：当前固定为空字符串，训练代码没有使用它。

`thinking_ratio` 控制构造 prompt 时开启 thinking 的概率。这里不直接 Tokenize；训练脚本需要对一个 batch 的字符串统一做左 padding，并根据 rollout 需求截断 prompt。

### `AgentRLDataset`

Agent RL 需要保留每条样本不同的工具定义和 ground truth，因此没有使用 Hugging Face `load_dataset` 的固定批处理结构，而是逐行读取 JSONL。

`parse_conversations` 会：

1. 从 system 消息的 `tools` 字段提取工具描述。
2. 返回除最后一条之外的消息作为 rollout 起点。
3. 将样本的 `gt` 原样交给奖励计算，用于验证最终回答。

最后一条对话通常代表数据中的目标答案；在线 RL 不把它直接喂给策略，而是用 `gt` 作为可验证奖励信号。

## 关键技术点解释

### 为什么 Padding 标签使用 `-100`

PyTorch 的 `cross_entropy` 在 `ignore_index=-100` 时跳过这些位置。把无监督 token 标成 `-100`，既能保留完整上下文供注意力读取，又不会要求模型在这些位置拟合目标。

### 聊天模板为何是训练一致性的核心

角色名本身不是模型能够理解的结构。`apply_chat_template` 会把 system、user、assistant、tool 等角色映射为 Tokenizer 配置中的特殊标记。训练与推理若使用不同模板，相同对话会变成不同 token 序列，模型能力会明显下降。

### 随机预处理的作用

普通 SFT 对话会以一定概率增加 system prompt，也会以一定概率保留空 thinking 块。这是在不修改语义的前提下增加格式多样性，降低模型对某一种固定开头的依赖。Tool Call 数据不做 system 随机插入，避免破坏工具协议结构。

### 截断会影响什么

所有固定长度数据集都从尾部或按 Tokenizer 默认方向截断到最大长度。若一条 assistant 回复被截断，监督区域也会随之缩短；若起始标记被截掉，则相应回复可能完全不产生标签。设置 `max_length` 时需要在显存与完整样本比例之间权衡。

## 输入格式概览

| 阶段 | 核心字段 |
| --- | --- |
| 预训练 | `text` |
| SFT | `conversations: [{role, content, reasoning_content, tools, tool_calls}]` |
| DPO | `chosen`、`rejected` |
| RLAIF | `conversations` |
| Agent RL | `conversations`、`gt`，工具通常位于 system 消息的 `tools` |

实际字段细节应以仓库数据文件和 Tokenizer 聊天模板为准。

## 使用注意事项

- `SFTDataset` 使用固定 Features，数据字段应与声明兼容；工具字段如果是字符串，内容必须是合法 JSON。
- `pre_processing_chat` 和 `post_processing_chat` 包含随机行为。训练脚本通过 `setup_seed` 控制复现，但 DataLoader worker 数量也会影响随机调用顺序。
- `RLAIFDataset.max_length` 不在数据集内部截断字符串；PPO/GRPO 会在 Tokenize 后处理长度。
- `AgentRLDataset.max_length` 当前只保存为属性，实际总长度限制由 `train_agent.py` 的 `max_total_len` 执行。
- 文件末尾的 `if __name__ == '__main__': pass` 不提供数据集演示；这些类由训练脚本导入使用。
