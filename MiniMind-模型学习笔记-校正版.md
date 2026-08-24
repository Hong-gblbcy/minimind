# MiniMind 模型学习笔记（校正版）

> 基于当前工作区中的 `model/tokenizer_config.json`、`model/tokenizer.json`、`model/model_minimind.py`、聊天模板和训练脚本校正。文中的“默认值”均指源码默认配置，训练命令仍可覆盖它们。

---

## 一、Tokenizer 基础

### 1.1 词表结构

`model/tokenizer.json` 是一个 6400 词 ByteLevel BPE：

| 组成                     | 数量 |
| ------------------------ | ---- |
| 预留/结构 token          | 36   |
| ByteLevel 基础字节 token | 256  |
| BPE merge 结果           | 6108 |
| 合计                     | 6400 |

ID 分区：

```
0—35      协议与预留 token
36—291    完整 256 字节 alphabet
292—6399  BPE merge token
```

### 1.2 36 个预留 token

| ID    | token                                                                | 是否 special |
| ----- | -------------------------------------------------------------------- | ------------ |
| 0     | `<\|endoftext\|>`（同时 PAD/UNK）                                    | 是           |
| 1     | `<\|im_start\|>`（BOS）                                              | 是           |
| 2     | `<\|im_end\|>`（EOS）                                                | 是           |
| 3–8   | object ref、box、quad 起止标记                                       | 是           |
| 9–13  | vision/image/video 标记                                              | 是           |
| 14–16 | audio 标记                                                           | 是           |
| 17–20 | TTS 标记                                                             | 是           |
| 21–24 | `<tool_call>`、`</tool_call>`、`<tool_response>`、`</tool_response>` | **否**       |
| 25–26 | `<think>`、`</think>`                                               | **否**       |
| 27–35 | `<\|buffer1\|>`～`<\|buffer9\|>`                                     | 否           |

### 1.3 Tool/Think 标签：不可再分的 added token，但故意不是 special token

**`added_tokens` 的核心含义**：当输入中精确出现这些字符串时，added-token 匹配会在普通的预分词和 BPE 合并之前识别它们，因此会编码成单个 ID，而不是继续拆成子词。

**`special: false` 的含义**：tokenizer 不把这些 added token 放入需要特殊处理/过滤的集合，因此在解码时不会被自动过滤（`skip_special_tokens=True` 后仍然可见），推理代码可以继续从文本中解析它们。

是否对某个 token 计算训练 loss，与这里的 `special` 属性无关，而由对应位置的 `labels` 是否为 `-100` 决定。即使是 `special: true` 的 token，只要其 label 未被屏蔽，也能参与交叉熵计算。

因此可以概括为：**用 added-token 机制保证结构字符串原子化，用 `special: false` 保证这些结构标签在跳过 special token 的解码中仍然可见。**

| 属性                                | `special: true`（如 `<\|im_start\|>`） | `special: false`（如 `<tool_call>`） |
| ----------------------------------- | -------------------------------------- | ------------------------------------ |
| 不可再分                            | ✅                                     | ✅                                   |
| `skip_special_tokens=True` 时被移除 | ✅ 会被移除                            | ❌ 不会被移除                        |
| tokenizer 分类                      | special added token                    | 普通 added token                     |

---

## 二、Tokenizer 配置

### 2.1 核心配置

```text
model_max_length = 131072
BOS = 1, EOS = 2, PAD = UNK = 0
add_bos_token = false
add_eos_token = false
```

### 2.2 BOS/EOS/PAD/UNK

| 角色 | ID  | 内容              | 含义         |
| ---- | --- | ----------------- | ------------ |
| BOS  | 1   | `<\|im_start\|>`  | 序列开始标记 |
| EOS  | 2   | `<\|im_end\|>`    | 序列结束标记 |
| PAD  | 0   | `<\|endoftext\|>` | 填充 token   |
| UNK  | 0   | `<\|endoftext\|>` | 未知 token   |

BOS 配置指向 `<|im_start|>`。在聊天模板中，它主要充当每条消息的开始标记，而不是只在整个序列开头出现一次。PAD 和 UNK 在 tokenizer 配置层都指向 `<|endoftext|>`（ID=0）。

不过，底层 `tokenizer.json` 中 BPE 模型的 `unk_token` 实际为 `null`，而完整 ByteLevel alphabet 能覆盖任意 UTF-8 输入的字节，所以正常文本不会因为词表只有 6400 项就产生很高的 UNK 率。不能把 PAD/UNK 同 ID 解释成“为高 UNK 率节省词表”。

### 2.3 add_bos_token = false & add_eos_token = false

**当前 tokenizer 不会自动在普通编码结果外添加 BOS/EOS。** 聊天数据由模板显式插入每条消息的 `<|im_start|>` 和 `<|im_end|>`；预训练数据集则在代码中手动添加 BOS/EOS。因此不应再在外层无条件重复添加。

### 2.4 model_max_length = 131072 的真相

**这个 131072 不能证明模型具备 128K 上下文能力。** 从当前源码只能确认它是 tokenizer 的长度元数据；无法仅凭源码断言它具体继承自哪个外部模型。默认模型预计算的 RoPE 缓冲区长度为 32768。

三个长度概念不要混淆：

| 配置                      | 值      | 来源                    | 实际作用                       |
| ------------------------- | ------- | ----------------------- | ------------------------------ |
| `model_max_length`        | 131072  | `tokenizer_config.json` | tokenizer 元数据，控制截断行为 |
| `max_position_embeddings` | 32768   | 模型代码                | 默认实例的 RoPE 缓冲区大小     |
| 训练脚本 `max_seq_len`    | 因脚本而异 | 训练脚本              | 本次数据截断/训练长度          |

`model_max_length` 最直接的作用之一是：调用 `tokenizer(..., truncation=True)` 且不传 `max_length` 时，作为默认截断长度；它也可能用于 tokenizer 的长度检查或警告。

当前主要脚本的默认值并不统一：

| 脚本 | 默认 `max_seq_len` |
| ---- | -------------------- |
| Pretrain / LoRA / Distillation | 340 |
| Full SFT / GRPO / PPO | 768 |
| DPO / Agent | 1024 |

这些参数都可被命令行覆盖。部分推理代码（例如 `eval_llm.py`）使用了 `truncation=True`，但没有传 `max_length`，此时仍会采用 tokenizer 的 131072，而不是自动限制到模型的 32768。由于模型直接切片长度为 32768 的 RoPE 表，默认实例处理超过这一范围的序列会发生维度不匹配；若要改变实现上限，需要同步修改模型配置并重新构造 RoPE 缓冲区。

---

## 三、Chat Template（聊天模板）

### 3.1 模板做了什么

模板把 `<|im_start|>` 和 `<|im_end|>` 当作"对话消息的边界标记"而非"整个序列的边界标记"：

```
<|im_start|>system
系统提示<|im_end|>
<|im_start|>user
用户文本<|im_end|>
<|im_start|>assistant
<think>
思考过程
</think>

回复正文<|im_end|>
```

### 3.2 核心规则

1. 有 `tools` 时，模板建立 system 块，追加固定英文 Tool Calling 说明，工具 JSON 放入 `<tools>...</tools>`
2. 无 tools 时，若首条消息是 system，模板先渲染该 system 块；后续 system 消息也会在普通消息循环中按 system 角色渲染
3. 模板渲染已有 assistant 消息时，会先输出 `<think>...</think>`，之后才是正文和可能的 tool calls；当前词表不存在 ` response` token
4. 优先读取独立的 `reasoning_content`；否则在 content 含 `</think>` 时尝试拆分思考和正文
5. `tool_calls` 同时接受 `{name, arguments}` 和 OpenAI 风格
6. 连续多个 `tool` 消息被合并到一个 user turn
7. `add_generation_prompt=True, open_thinking=True` 时，prompt 结尾是 `<think>` 加换行，模型从思考内容开始续写
8. `open_thinking=False`（或未提供）时，prompt 先放完整空思考块 `<think>\n\n</think>\n\n`，模型直接续写正文

以上描述针对 tokenizer 的 Jinja 模板。SFT 数据进入模型前还会经过 `post_processing_chat`：对于空思考块，当前默认有 80% 概率移除，因此“模板会生成空 think 块”不等于“每条训练样本最终都保留空 think 块”。

---

## 四、模型架构总览

### 4.1 默认配置

| 参数                      | 值    | 含义                        |
| ------------------------- | ----- | --------------------------- |
| `vocab_size`              | 6400  | 与 tokenizer 严格一致       |
| `hidden_size`             | 768   | 隐藏维度                    |
| `num_hidden_layers`       | 8     | Transformer Block 数        |
| `num_attention_heads`     | 8     | Q heads                     |
| `num_key_value_heads`     | 4     | KV heads，GQA 2:1           |
| `head_dim`                | 96    | `768 / 8`                   |
| `intermediate_size`       | 2432  | `ceil(768π/64)×64`          |
| `hidden_act`              | SiLU  | SwiGLU 的激活               |
| `rope_theta`              | 1e6   | RoPE base                   |
| `max_position_embeddings` | 32768 | RoPE buffer                 |
| `tie_word_embeddings`     | true  | embedding 与 LM head 绑权重 |
| `use_moe`                | false | 默认使用 Dense FFN          |
| `num_experts`             | 4     | MoE 专家数                  |
| `num_experts_per_tok`     | 1     | Top-1                       |
| `inference_rope_scaling` | false | 默认不启用 YaRN             |

`num_experts` 等 MoE 参数虽然始终存在于配置对象中，但只有 `use_moe=True` 时才参与模型结构。

### 4.2 一个 Block 的数据流

```
x
├─ RMSNorm → GQA Attention → residual add
└─ RMSNorm → SwiGLU FFN / MoE → residual add
```

### 4.3 参数计算

- Dense：63,912,192 参数，约 63.91M
- MoE：198,416,640 总参数，约 198.42M
- MoE 每 token 的参数路径约 63.94M（包含共享参数、router 与被选中的一个专家），所以可写作 `198M-A64M`

---

## 五、hidden_size — 每个 token 的向量维度

### 5.1 一句话理解

**每个 token 被表示为一个 768 维的向量。** 这个 768 是整个模型内部不变的"通道宽度"：embedding 输出 768 维，每一层 Transformer 的输入输出都是 768 维，最终 LM Head 从 768 维预测下一个 token。

```
token ID: 1234 (整数)
    │
    ▼
Embedding 查表: 6400×768 的矩阵，取第 1234 行
    │
    ▼
向量: [0.023, -0.145, 0.891, ..., 0.332]  ← 768 个数字
```

### 5.2 hidden_size 决定其他维度

| 派生参数     | 计算方式                            | 值             |
| ------------ | ----------------------------------- | -------------- |
| `head_dim`   | `hidden_size / num_attention_heads` | `768 / 8 = 96` |
| Q 投影维度   | `num_attention_heads × head_dim`    | `8 × 96 = 768` |
| K/V 投影维度 | `num_key_value_heads × head_dim`    | `4 × 96 = 384` |
| FFN 中间维度 | `ceil(768 × π / 64) × 64`           | `2432`         |
| Embedding 表 | `vocab_size × hidden_size`          | `6400 × 768`   |

---

## 六、Multi-Head Attention 与 GQA

### 6.1 num_attention_heads = 8

一层有 8 个 Q head。不同 head 使用各自的投影切片，可以学习不同的注意力行为，但它们由同一个训练目标联合优化，输出也会重新拼接并投影，所以不能理解成彼此完全独立，也不能断言一个 head 只能学习一种固定模式。

为了帮助直觉理解，可以把“关注近邻”“关联句法成分”“保留自身信息”等看成可能出现的行为示例，但具体 head 的功能需要通过分析训练后权重和激活来验证，不能由 head 编号预先指定。

8 个 head 各有 96 维，拼接后刚好是 768 维。8 层共有 64 个 Q-head 实例，但它们位于不同层并相互组合，不应称为“64 个独立视角”。

### 6.2 GQA（Grouped Query Attention）

你的模型用了 GQA：

```
num_attention_heads = 8   ← Q 有 8 个 head
num_key_value_heads = 4   ← K/V 只有 4 个 head
```

8 个 Q head 分为 4 组，每组 2 个 Q head 共享 1 个 KV head：

```
Q head 0, Q head 1  →  共享 KV head 0
Q head 2, Q head 3  →  共享 KV head 1
Q head 4, Q head 5  →  共享 KV head 2
Q head 6, Q head 7  →  共享 KV head 3
```

**目的**：减少 K/V 投影与 KV cache 开销。与其他维度相同的 8-KV-head MHA 相比，这里的 K/V head 数量减半，因此 KV cache 中 K/V 张量的主体大小约减半；模型整体显存不会因此直接减半。效果差异取决于模型规模、数据和训练过程，不能仅由源码保证“影响很小”。

### 6.3 head_dim = 96

`head_dim = hidden_size / num_attention_heads = 768 / 8 = 96`。它是每个 attention head 里的向量维度，是 RoPE 旋转操作的基本单位，也是 attention score 缩放中 $\sqrt{d_k}$ 的来源。当前 RoPE 实现把最后一维等分成两半进行配对，因此 `head_dim` 必须是偶数，否则会发生维度不匹配。

---

## 七、RoPE（Rotary Position Embedding）

### 7.1 为什么需要位置编码

Transformer 的 self-attention 本身是位置无关的——它只看 token 之间的内容相似度，不知道"谁在前谁在后"。位置编码的目标：给每个 token 注入它在序列中的位置信息。

### 7.2 核心思想：用旋转编码位置

**不是把一个位置向量直接加到 token 表示上，而是对 Q 和 K 的二维坐标对实施与位置有关的旋转。** 每个频率分量的相位随位置线性变化，并按周期回绕；不同频率组合后，使 Q/K 点积携带相对位置信息。

### 7.3 频率计算

head_dim=96，对应 48 个频率和 48 个二维旋转平面：

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2) / dim))
#       = 1.0 / (1e6 ** (0, 2, 4, ..., 94) / 96)
```

公式中的指数索引是 `0, 2, ..., 94`；也可以用频率索引 `j=0,1,...,47` 表示。当前 `rotate_half` 实现配对的是坐标 `(j, j+48)`，而不是相邻坐标 `(0,1)、(2,3)...`。

**规则：索引越小，频率越高。** 指数索引 i=0 时频率为 1 弧度/步；i=94 时约为 1.33e-6 弧度/步。高频更容易区分较小的位置变化，低频的变化尺度更长，但“某一频率专门负责某种距离关系”只是帮助理解的直觉，不是源码规定的固定分工。

### 7.4 相对位置的自然编码

RoPE 不需要显式计算所有 token 对之间的距离。每个 token 独立旋转 Q 和 K（O(T) 开销），利用旋转的数学性质，attention 点积自动包含相对位置信息：

$$\mathbf{q}_m \cdot \mathbf{k}_n = f(\mathbf{q}_m, \mathbf{k}_n, m - n)$$

更准确地说，点积中的**位置变换部分**只通过相对距离 $m-n$ 出现；点积结果仍然同时取决于 token 内容产生的 Q 和 K。

### 7.5 预计算 cos/sin 表

最终得到 `[32768, 96]` 的 cos 表和 sin 表。对位置 p 的 token，取第 p 行，与 Q/K 向量逐元素旋转。

---

## 八、YaRN（Yet another RoPE extensioN）

### 8.1 问题：RoPE 的外推

RoPE 外推讨论的是：推理长度超过模型训练中充分见过的长度后，位置相位分布发生变化，模型质量可能下降。当前源码中的 `original_max_position_embeddings=2048` 是 YaRN 的参考长度参数，**不能单凭它证明某个权重训练时只见过 0～2047**；实际训练长度由数据和 `max_seq_len` 等训练参数决定。

没有启用 YaRN 时，标准 RoPE 在当前实现中仍可计算到预生成缓冲区的 32768，而不是超过 2048 就发生程序崩溃。超过训练分布可能带来质量退化；超过 32768 才会因默认 RoPE 表长度不足而发生实现层面的尺寸问题。

### 8.2 核心思想：选择性频率缩放

**高频保持不变，低频除以 factor，中间用线性斜坡过渡。** 这保留了较短尺度的相位变化，同时放慢较长尺度频率的旋转：

```python
# 核心公式
freqs = freqs * (1 - ramp + ramp / factor)
```

| 区域              | ramp 值 | 实际缩放 | 效果           |
| ----------------- | ------- | -------- | -------------- |
| 高频（频率索引 0～8）   | 0       | 不变     | 保持较短尺度的相位变化 |
| 中频（频率索引 9～20）  | 0→1     | 渐变     | 平滑过渡               |
| 低频（频率索引 21～47） | 1       | ÷16      | 放慢较长尺度的相位变化 |

### 8.3 分界线计算

`inv_dim` 根据“在原始上下文长度内旋转多少圈”求对应的频率索引：

```python
inv_dim = lambda b: (dim * log(orig_max / (b × 2π))) / (2 × log(rope_base))
```

```
beta_fast = 32  →  inv_dim(32) ≈ 8.064   →  low  = floor(...) = 8
beta_slow = 1   →  inv_dim(1)  ≈ 20.105  →  high = ceil(...)  = 21
```

`beta_fast=32` 表示边界频率在原始 2048-token 区间内约旋转 32 圈，对应波长约 `2048/32=64` token；`beta_slow=1` 对应约旋转 1 圈，波长约 2048 token。因此 beta 不是“波长为 32 或 1 token”。

### 8.4 YaRN 可选配置

```text
factor = 16
original_max_position_embeddings = 2048
beta_fast = 32
beta_slow = 1
attention_factor = 1
```

这些参数只有在 `inference_rope_scaling=True` 时才写入 `rope_scaling` 并生效；默认值为 false。按照参考长度与 factor，名义扩展范围是 `2048 × 16 = 32768`。

YaRN 调整的是位置频率，不能凭空赋予模型没有通过训练获得的长文理解、检索和推理能力。是否改善长上下文效果仍需用对应权重和任务实测。

---

## 九、FFN（Feed-Forward Network）

### 9.1 Attention 和 FFN 的分工（直觉）

```
Attention: 根据内容相关性聚合其他 token 的信息 → token 之间交换信息
FFN:       对聚合后的每个 token 分别做非线性变换 → 逐位置加工信息
```

“Attention 负责看、FFN 负责想”可以作为入门类比，但不是严格的功能边界。推理能力来自多层 Attention、FFN、残差与归一化的共同作用。

### 9.2 FFN 的结构

```python
# SwiGLU FFN
gate = silu(gate_proj(x))   # 768 → 2432，门控
up   = up_proj(x)            # 768 → 2432，激活
hidden = gate * up           # 逐元素乘，引入非线性
output = down_proj(hidden)   # 2432 → 768，压缩
```

Attention 输出包含对 V 的加权求和，但权重由输入相关的 Q/K 点积和 softmax 动态产生，因此整个 Attention 不是固定线性变换。FFN 通过 SiLU 门控、逐元素乘法以及先扩维再压缩提供逐 token 的非线性变换能力。

---

## 十、MoE（Mixture of Experts）

### 10.1 Multi-Head Attention vs MoE

|                   | Multi-Head Attention       | MoE                                 |
| ----------------- | -------------------------- | ----------------------------------- |
| 作用在哪          | Attention 部分             | FFN 部分                            |
| 有几个"分身"      | 8 个 Q head + 4 个 KV head | 4 个专家 FFN                        |
| 每个 token 走几个 | 全部 head 都走             | 只走 1 个专家（Top-1）              |
| 目的              | 聚合 token 间的信息        | 增加容量且避免每个 token 经过所有专家 |

### 10.2 MoE 的工作方式

```python
# 每个 token 经过 router
scores = softmax(gate(x))          # [B, T, 4] — 4 个专家的得分
topk_weight, topk_idx = topk(scores, k=1)  # 只选得分最高的 1 个

# 每个 token 只激活 1 个专家，但模型存储了 4 个专家
```

### 10.3 为什么需要多个专家

**核心思路是增加总参数容量，而不让每个 token 经过所有专家。** 如果直接增加 hidden_size、intermediate_size 或层数，通常会同时增加每 token 计算量；稀疏 MoE 则让每个 token 只经过少数专家。它仍然会增加 router、调度和内存访问开销，所以不能理解成计算量严格不变。

类比：医院分诊制度。病人（token）被路由到不同科室（专家），每个病人只进入少数科室。这个类比解释的是路由方式；专家是否真正形成有效分工，取决于数据、训练和负载均衡。

### 10.4 Dense vs MoE 对比

|                   | Dense 64M | MoE 198M-A64M       |
| ----------------- | --------- | ------------------- |
| 总参数            | 64M       | 198M（3倍）         |
| 每 token 激活参数 | 64M       | ~64M                |
| 每 token 理论主干计算 | 1x     | 接近 1x，另有路由/调度开销 |
| 实际推理速度      | 基准      | 取决于硬件、batch 与实现，可能更慢 |
| 显存占用          | 低        | 高（198M 全要加载） |
| 模型能力          | 取决于训练 | 容量更大，但效果需实测 |

更严谨的说法是：**MoE 用更高的参数存储和更复杂的调度，换取在相近稀疏主干计算量下更大的模型容量。** 它不保证推理速度与 Dense 相同，也不保证在所有训练条件下能力一定更强。显存紧张或稀疏算子效率较低的设备通常更适合 Dense；是否选择 MoE 应以实际吞吐、延迟和效果为准。

---

## 十一、Causal LM Loss

模型内部完成 next-token shift：

```text
x = logits[..., :-1, :]
y = labels[..., 1:]
CE(x, y, ignore_index=-100)
```

所以，对直接调用 `MiniMindForCausalLM(..., labels=labels)` 的 Dataset，应返回等长的 `input_ids` 与 `labels`，不要再额外 shift；不参与监督的位置写成 `-100`。仓库中的 DPO 等自定义目标可能自行构造已 shift 的 `x/y`，不属于这一接口约定。

上述 loss 路径默认假设 `logits_to_keep=0`，即保留完整序列 logits。若在传入完整 labels 的同时只保留部分 logits，当前代码没有同步裁剪 labels，会造成形状不一致。

---

## 十二、数据流完整路径

```
输入: token ID (整数)
  │
  ├─→ Embedding 查表: 6400×768 → [768]
  │
  ├─→ Layer 1~8 (每层):
  │     ├─ RMSNorm → GQA Attention (RoPE + KV Cache) → residual add
  │     └─ RMSNorm → SwiGLU FFN / MoE → residual add
  │
  ├─→ Final RMSNorm → [768]
  │
  └─→ LM Head: 768 → 6400 (预测下一个 token)
```

GQA Attention 内部张量形状：

```
x          [B, T, 768]
Q          [B, T, 8, 96]     ← 8 个 Q head，每个 96 维
K/V        [B, T, 4, 96]     ← 4 个 KV head (GQA 2:1)
缓存 K/V   [B, total_T, 4, 96]
repeat KV  [B, total_T, 8, 96] ← 4 个 KV head 各重复 2 次
scores     [B, 8, T, total_T]  ← 手写 attention 分支中的逻辑形状
output     [B, T, 768]
```

启用 PyTorch scaled-dot-product/Flash attention 路径时，完整的 `scores` 张量不一定会显式物化；上表用于说明注意力计算的逻辑维度。

---

## 十三、快速参考表

| 概念                      | 一句话解释                                              |
| ------------------------- | ------------------------------------------------------- |
| `hidden_size`             | 每个 token 在模型内部的向量维度（768）                  |
| `head_dim`                | 每个 attention head 的向量维度（96 = 768/8）            |
| `num_attention_heads`     | 一层中 Q 的 head 数（8）                                |
| `num_key_value_heads`     | 一层中 KV 的 head 数（4，GQA）                          |
| RoPE                      | 用旋转编码 token 位置，点积自动包含相对位置             |
| YaRN                      | 选择性缩放 RoPE 频率，让模型处理更长序列                |
| FFN                       | Attention 后每个 token 独立的非线性变换（768→2432→768） |
| MoE                       | 4 个 FFN 专家，Top-1 路由；增大容量但带来存储与调度开销 |
| `model_max_length`        | tokenizer 元数据（131072），不等于模型能力              |
| `max_position_embeddings` | 默认 RoPE 缓冲长度（32768），是当前默认实例的实现边界   |
| `inference_rope_scaling`  | 是否启用 YaRN；默认 false                               |

---

## 十四、源码核对入口

- Tokenizer 词表、added token 与 BPE merges：[`model/tokenizer.json`](model/tokenizer.json)
- BOS/EOS、长度元数据与聊天模板：[`model/tokenizer_config.json`](model/tokenizer_config.json)
- 模型配置、RoPE/YaRN、Attention、FFN、MoE 与 loss：[`model/model_minimind.py`](model/model_minimind.py)
- Tokenizer 的生成方式与 special-token 分类：[`trainer/train_tokenizer.py`](trainer/train_tokenizer.py)
- 预训练/SFT 的 token 和 label 构造：[`dataset/lm_dataset.py`](dataset/lm_dataset.py)
