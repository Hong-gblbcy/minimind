# `model_minimind.py` 介绍

对应源码：[`model_minimind.py`](./model_minimind.py)

## 文件定位

MiniMind 核心语言模型的完整 PyTorch/Transformers 实现，覆盖配置、注意力、前馈网络、MoE、因果语言模型损失和自回归生成。

## 主要组成

| 对象 | 作用 |
| --- | --- |
| `MiniMindConfig` | 定义隐藏维度、层数、词表、注意力头、RoPE 和 MoE 参数 |
| `RMSNorm` | RMS 归一化 |
| `precompute_freqs_cis` | 预计算 RoPE；可选 YaRN 推理外推 |
| `apply_rotary_pos_emb` | 将旋转位置编码应用到 Query/Key |
| `repeat_kv` | 为 GQA 复制 KV 头 |
| `Attention` | GQA、KV cache、Flash Attention 和普通因果注意力 |
| `FeedForward` | SwiGLU 风格稠密前馈层 |
| `MOEFeedForward` | Top-K 路由专家层及路由辅助损失 |
| `MiniMindBlock` | 注意力与前馈层的残差块 |
| `MiniMindModel` | Token Embedding、Transformer 层堆叠、最终归一化和缓存管理 |
| `MiniMindForCausalLM` | Transformers 兼容封装、语言模型头、CE 损失和生成接口 |

## 前向输出

`MiniMindForCausalLM.forward` 返回 `MoeCausalLMOutputWithPast`，其中包含 `loss`、`aux_loss`、`logits`、`past_key_values` 和 `hidden_states`。自定义 `generate` 支持 KV cache、temperature、Top-K、Top-P、重复惩罚、流式输出和多返回序列。

