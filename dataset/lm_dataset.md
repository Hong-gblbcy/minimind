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

