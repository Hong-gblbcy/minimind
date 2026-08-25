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

## 项目脉络

这个文件是仓库的计算核心。数据集、所有训练算法和所有推理入口最终都会调用这里定义的模型：

```text
dataset/lm_dataset.py
  └─> trainer/train_*.py ──────────────┐
                                        ├─> MiniMindForCausalLM
eval_llm.py / API / Web / Tool Call ───┘
                                                │
                                                ├─> loss / logits
                                                └─> generated token ids
```

它继承 Transformers 的配置和模型基类，因此既能保持手写结构的可读性，也能接入 `AutoModelForCausalLM`、`save_pretrained`、Streamer 等生态接口。

## 模型总体数据流

```text
input_ids [B, T]
  └─> Token Embedding [B, T, C]
        └─> Dropout
              └─> N × MiniMindBlock
                    ├─> RMSNorm
                    ├─> GQA Attention + RoPE + Residual
                    └─> RMSNorm
                         └─> Dense FFN 或 MoE FFN + Residual
              └─> Final RMSNorm
                    └─> lm_head [B, T, vocab_size]
                          ├─> 训练：Cross Entropy
                          └─> 推理：采样下一个 token
```

符号含义：`B` 是 batch size，`T` 是序列长度，`C` 是隐藏维度。

## 配置层：`MiniMindConfig`

`MiniMindConfig` 继承 `PretrainedConfig`，`model_type='minimind'` 用于 Transformers 识别模型类型。关键参数分为四组：

### 模型规模

- `vocab_size`：词表大小，默认 6400。
- `hidden_size`：每个 token 的隐藏向量维度。
- `num_hidden_layers`：Transformer Block 数量。
- `intermediate_size`：FFN 中间维度，默认按 `hidden_size × π` 计算后向 64 对齐。

向 64 对齐有利于 GPU 矩阵运算使用更规则的形状。

### 注意力结构

- `num_attention_heads`：Query 头数，默认 8。
- `num_key_value_heads`：Key/Value 头数，默认 4。
- `head_dim`：单头维度，默认 `hidden_size / num_attention_heads`。
- `flash_attn`：允许使用 PyTorch SDPA 快速实现。

Query 头多于 KV 头构成 Grouped Query Attention（GQA）。多个 Query 头共享较少的 Key/Value 头，可以降低 KV cache 显存和带宽开销。

### 位置编码

- `max_position_embeddings`：预计算 RoPE 的最大位置数。
- `rope_theta`：RoPE 基频，默认 `1e6`。
- `inference_rope_scaling`：是否建立 YaRN 缩放配置。

### MoE

- `num_experts`：专家总数。
- `num_experts_per_tok`：每个 token 激活的专家数。
- `moe_intermediate_size`：单专家中间维度。
- `norm_topk_prob`：是否归一化被选中专家的权重。
- `router_aux_loss_coef`：负载均衡辅助损失系数。

## 归一化与位置编码

### `RMSNorm`

RMSNorm 不减去均值，只按均方根缩放：

```text
x_norm = x / sqrt(mean(x²) + eps)
output = weight * x_norm
```

代码先把输入转为 FP32 计算平方和，提升低精度训练时的数值稳定性，再转回原 dtype。与 LayerNorm 相比，它省略均值中心化，结构更简单。

### `precompute_freqs_cis`

该函数预先为每个位置和维度计算 `cos`、`sin` 缓冲区。模型前向时只需切片，不必每层重复计算三角函数。

当启用 `rope_scaling` 且推理长度超过原始最大长度时，YaRN 会在不同频率维度之间使用线性 ramp：低频与高频采用不同程度的缩放，从而比统一缩放更平滑地扩展位置范围。

### `apply_rotary_pos_emb`

RoPE 把 Query/Key 的两半维度看作二维坐标，通过 `cos`、`sin` 做旋转。旋转后，注意力点积能够表达 token 之间的相对位置。Value 不参与旋转，因为位置关系在 Query-Key 相似度中已经体现。

## 注意力：`Attention`

### 张量形状

输入 `x` 的形状是 `[B, T, C]`：

```text
q_proj -> [B, T, num_attention_heads, head_dim]
k_proj -> [B, T, num_kv_heads, head_dim]
v_proj -> [B, T, num_kv_heads, head_dim]
```

Query 和 Key 会先做单头 RMSNorm，再应用 RoPE。若存在历史 KV cache，新 Key/Value 会沿序列维拼到历史后面。

`repeat_kv` 随后把较少的 KV 头复制到 Query 头数量，实现 GQA。转置后进入注意力计算的形状为 `[B, heads, T, head_dim]`。

### Flash Attention 快速路径

满足以下条件时使用 `torch.nn.functional.scaled_dot_product_attention`：

- 当前 PyTorch 提供该接口。
- 当前序列长度大于 1。
- 因果条件和 cache 状态允许直接设置 `is_causal`。
- 没有复杂 attention mask，或 mask 全部有效。

PyTorch 可以根据设备选择 Flash、Memory Efficient 或数学实现，避免显式构造完整注意力矩阵。

### 普通注意力回退

无法走快速路径时，代码显式计算：

```text
scores = Q @ Kᵀ / sqrt(head_dim)
scores += causal mask
scores += padding mask
probabilities = softmax(scores)
output = probabilities @ V
```

softmax 使用 FP32 计算后再转回原类型，减少半精度溢出。最后合并多头，经 `o_proj` 和 residual dropout 返回。

## 前馈网络

### `FeedForward`

稠密 FFN 使用门控结构：

```text
down_proj( activation(gate_proj(x)) * up_proj(x) )
```

默认激活函数是 SiLU。这类 SwiGLU 风格结构让一个分支控制另一个分支的信息流，通常比单一激活层具有更强表达能力。

### `MOEFeedForward`

MoE 为每个 token 计算专家概率，再选择 Top-K 专家：

```text
x
  └─> gate softmax
        └─> top-k experts
              └─> 各专家独立 FFN
                    └─> 按路由权重加权汇总
```

代码只对实际路由到某专家的 token 执行该专家。未被使用的专家在训练时通过零值表达式连接参数，避免分布式训练把它们判断为完全未参与图计算。

路由辅助损失比较专家负载与平均路由概率，鼓励 token 更均匀地分配到专家，防止少数专家过载、其他专家长期得不到训练。

## Block 与主干模型

### `MiniMindBlock`

每层采用 Pre-Norm 残差结构：

```text
h = x + Attention(RMSNorm(x))
out = h + FFN_or_MoE(RMSNorm(h))
```

先归一化再进入子层通常有利于深层网络的梯度稳定性。注意力和 FFN 各自拥有独立 RMSNorm。

### `MiniMindModel`

主干完成以下工作：

1. 把 token id 映射为隐藏向量。
2. 从 KV cache 推导当前序列的起始位置。
3. 切出对应 RoPE `cos/sin`。
4. 逐层传递隐藏状态和每层 cache。
5. 汇总所有 MoE 层的辅助损失。

Transformers 5.x 的 meta-device 初始化可能让非持久化 RoPE buffer 丢失。代码检查 buffer 是否为零，并在需要时重新计算，这是为模型加载生态做的兼容处理。

## 因果语言模型：`MiniMindForCausalLM`

### 权重共享

当 `tie_word_embeddings=True` 时，输入 Embedding 与输出 `lm_head` 共用同一权重矩阵。共享可以减少参数量，并让“token 的输入表示”和“输出分类器”处于同一向量空间。

### Logits 与移位损失

给定长度 T 的输入，位置 `t` 的 logits 用来预测位置 `t+1` 的标签：

```text
logits[..., :-1, :]  对齐  labels[..., 1:]
```

交叉熵设置 `ignore_index=-100`，因此数据集可以用 `-100` 精确控制哪些 token 参与监督。`logits_to_keep` 允许只计算末尾若干位置的输出头，在 rollout 计算逐 token logprob 时减少不必要的词表投影。

### 返回 MoE 兼容输出

使用 `MoeCausalLMOutputWithPast` 能同时返回标准语言模型字段和 `aux_loss`。Dense 模型的辅助损失为零，因此训练脚本可以统一写成“语言模型损失 + 辅助损失”。

## 自定义生成循环

`generate` 没有完全委托给 Transformers 默认实现，而是手写逐 token 循环：

1. 若请求多个返回序列，复制输入 batch。
2. 根据 KV cache 只把尚未计算的新 token 送入模型。
3. 取最后位置 logits，并除以 temperature。
4. 应用重复惩罚、Top-K 和 Top-P 过滤。
5. 采样或贪心选出下一个 token。
6. 追加到输入，更新 attention mask 和 cache。
7. 所有序列都遇到 EOS 后停止。

Streamer 在首轮收到完整 prompt，后续逐 token 收到新结果。`return_kv` 可让调用方同时取得生成 ids 和 cache，便于更高级的推理流程。

## 技术取舍

- 模型结构短小、关键运算直接可读，适合学习 Transformer 全链路。
- GQA 和 KV cache 面向推理效率；MoE 面向参数容量与激活计算量的分离。
- 手写生成更容易理解和修改，但没有覆盖 Transformers 生成框架的全部高级策略。
- RoPE buffer 标记为 `persistent=False`，不会写入权重文件，减少冗余；加载后由配置重新构造。
- `past_key_values` 使用项目自己的 tuple 列表。遇到带 `.layers` 的 Transformers Cache 对象时当前代码会丢弃它，说明主要支持自有缓存格式。

## 输入输出与注意事项

| 场景 | 输入 | 输出 |
| --- | --- | --- |
| 训练 | `input_ids`、可选 `labels`/`attention_mask` | logits、CE loss、MoE aux loss |
| 普通前向 | `input_ids`、可选 cache | hidden states、logits、新 cache |
| 生成 | prompt ids、采样参数 | 拼接 prompt 与 completion 的 token ids |

- `hidden_size` 应能被 `num_attention_heads` 正确划分，KV 头数也应能整除 Query 头数。
- `max_position_embeddings` 必须覆盖实际切片位置，否则 RoPE 缓冲区长度不足。
- 训练权重与 `MiniMindConfig` 必须严格匹配；`.pth` 本身不包含完整结构说明。
- 自定义生成在 `temperature=0` 时会发生除零；若需要确定性输出，应使用 `do_sample=False`，而不是把温度设为零。
- YaRN 只处理位置编码外推，不保证模型在未训练长度上具备可靠语义能力。
