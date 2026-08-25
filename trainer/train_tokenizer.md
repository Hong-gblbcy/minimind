# train_tokenizer.py 介绍

对应源码：[`train_tokenizer.py`](./train_tokenizer.py)

## 一、文件定位

`train_tokenizer.py` 是 MiniMind 项目中用于**训练 BPE 分词器（Tokenizer，即"词典"）**的脚本。

> ⚠️ **重要提示**（来自脚本头部注释）：
> 不建议重复训练 tokenizer。MiniMind 已自带训练好的 tokenizer，基于不同词典训练的模型会导致输出完全不统一，降低社区模型复用性。**本脚本仅供学习和参考**。

## 二、Tokenzier 是什么？

分词器（Tokenizer）是连接「原始文本」与「模型」的桥梁：

```
原始文本 ──分词器(编码)──▶ token id 序列 ──▶ 模型输入
模型输出(token id) ──分词器(解码)──▶ 原始文本
```

模型本身不认识文字，只认识数字。分词器负责把文本切成一个个 token（词元），并为每个 token 分配一个唯一的整数 id。

## 三、关键配置常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `DATA_PATH` | `../dataset/sft_t2t_mini.jsonl` | 训练语料来源 |
| `TOKENIZER_DIR` | `../model_learn_tokenizer/` | 分词器保存目录 |
| `VOCAB_SIZE` | `6400` | 词表大小（BPE 合并次数上限） |
| `SPECIAL_TOKENS_NUM` | `36` | 特殊 token 总数量 |

## 四、脚本整体结构

```mermaid
flowchart TD
    A[train_tokenizer] --> B[get_texts 读取语料]
    A --> C[构建特殊 token 列表]
    A --> D[BpeTrainer 训练]
    A --> E[保存 tokenizer.json + 生成 config]
    A --> F[eval_tokenizer 评估]
```

### 主要函数

| 函数 | 作用 |
|------|------|
| `get_texts(data_path)` | 从 jsonl 读取对话内容，生成纯文本迭代器（最多 10000 行） |
| `train_tokenizer(...)` | 核心：训练 BPE 分词器并保存 |
| `eval_tokenizer(...)` | 评估：验证编码/解码一致性、计算压缩率 |

## 五、训练流程详解

### 1. 读取语料 `get_texts`

- 逐行读取 `sft_t2t_mini.jsonl`
- 提取每条数据的 `conversations` 字段中所有 `content`
- 用 `\n` 拼接成一段文本
- 最多取前 10000 行（用于测试）

### 2. 构建特殊 token

特殊 token 分为三类：

1. **对话控制 token**：`<|im_start|>`、`<|im_end|>`、`<|endoftext|>`
2. **多模态占位 token**：`<|vision_start|>`、`<|image_pad|>`、`<|audio_pad|>`、`<|video_pad|>` 等
3. **工具调用 / 思考 token**：`<tool_call>`、`</tool_call>`、`<tool_response>`、`<think>`、`</think>`

此外还会**预留 buffer token**（`<|buffer1|>` 等），补齐到 `SPECIAL_TOKENS_NUM` 的数量，为后续扩展留位置。

### 3. BPE 训练核心

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

- **BPE（Byte Pair Encoding）**：统计文本中相邻字节对的出现频率，反复合并最高频的组合，直到达到 `vocab_size`。
- **ByteLevel**：基于字节级别，能处理任意 Unicode 字符（中文、emoji 等都不会产生 `<unk>`）。

```python
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=all_special_tokens
)
```

### 4. 保存与配置生成

保存两样东西：

1. **`tokenizer.json`** — tokenizer 的核心文件（词表 + 合并规则）
2. **`tokenizer_config.json`** — 配置信息，包含：
   - `bos_token` / `eos_token` / `pad_token` / `unk_token` 的映射
   - `chat_template`（聊天模板，Jinja2 语法，控制多轮对话、工具调用、思考内容的格式化）
   - `model_max_length: 131072`（最大长度）
   - 多模态相关 token 配置

## 六、评估函数 `eval_tokenizer` 做什么？

| 测试项 | 说明 |
|--------|------|
| **编码/解码一致性** | `decode(encode(text)) == text` 是否成立 |
| **压缩率测试** | 统计中英文文本的「字符数 / token 数」，衡量分词效率 |
| **流式解码测试** | 模拟流式生成场景，逐个 token 解码，验证字节缓冲处理（处理 `\ufffd` 未完成字节） |

### 压缩率指标

压缩率 = 字符数 ÷ token 数，值越大说明每个 token 承载的信息越多，分词越高效。通常：
- 中文：约 1~1.5（一个 token 约对应 1 个汉字）
- 英文：约 3~4（一个 token 约对应 3~4 个字符）

## 七、运行方式

```python
if __name__ == '__main__':
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
```

直接运行脚本即可完成「训练 → 保存 → 评估」全流程：

脚本中的默认数据与输出路径以 `trainer/` 为当前目录，因此应从该目录运行：

```bash
cd trainer
python train_tokenizer.py
```

## 八、关键设计点总结

1. **ByteLevel BPE** 保证任意字符可编码，无 OOV（out-of-vocabulary）问题。
2. **预置特殊 token** 为多模态、工具调用、思考（R1 风格）预留了位置。
3. **buffer token 预留机制** 为未来扩展词典留出空间，保证词典结构稳定。
4. **chat_template 内嵌** 到 config 中，让训练好的 tokenizer 可直接配合 `apply_chat_template` 使用。
