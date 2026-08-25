# `train_dpo.py` 介绍

对应源码：[`train_dpo.py`](./train_dpo.py)

## 文件定位

Direct Preference Optimization（DPO）训练入口，用 chosen/rejected 偏好对直接优化策略。

## 核心函数

- `logits_to_log_probs`：从词表 logits 中提取目标 token 的逐 token 对数概率。
- `dpo_loss`：分别累计 chosen 与 rejected 的有效区域 logprob，比较策略模型和参考模型的偏好差值，再计算 `-logsigmoid(beta * logits)`。
- `train_epoch`：执行参考模型前向、策略模型前向、DPO 损失与可选 MoE 辅助损失的反向传播。

## 模型与数据

- 策略模型参与训练，参考模型从相同基础权重加载后冻结。
- `DPODataset` 返回 chosen/rejected 输入、目标和 assistant 区域掩码。
- 默认数据：`../dataset/dpo.jsonl`。
- 默认基础权重：`full_sft`。
- 默认输出：`../out/dpo_<hidden_size>.pth`。

脚本支持混合精度、梯度累积/裁剪、DDP、`torch.compile`、断点续训和 SwanLab。

```bash
cd trainer
python train_dpo.py
```

## 项目脉络

DPO 属于离线偏好优化。它不在线生成，也不需要显式奖励模型，而是直接学习数据中的“chosen 比 rejected 更好”：

```text
full_sft 策略 ───────────────┐
                             ├─> chosen/rejected 相对 logprob
冻结 full_sft 参考模型 ──────┘
                                      └─> DPO loss
                                            └─> dpo_*.pth
```

参考模型代表训练开始前的行为。DPO 优化的是策略相对参考模型的偏好变化，避免仅仅因为某个回答在所有模型下都更常见，就误把它当成优化进展。

## 数据流

`DPODataset` 为每条样本返回 chosen 和 rejected 两组：

```text
x_*    : 去掉最后一个 token 的输入
y_*    : 去掉第一个 token 的目标
mask_* : assistant 回复区域为 1
```

训练循环把两组沿 batch 维拼接：

```text
[chosen batch; rejected batch]
```

参考模型与策略模型对同一拼接输入分别前向一次，`logits_to_log_probs` 用 `gather` 取出真实目标 token 的 logprob。

## DPO 损失推导

先在 mask 内累计序列 logprob：

```text
log π(y|x) = Σ mask_t × log π(y_t | x, y_<t)
```

分别计算策略与参考模型对 chosen/rejected 的偏好差：

```text
pi_logratios  = logπ(chosen)  - logπ(rejected)
ref_logratios = logref(chosen) - logref(rejected)
logits = pi_logratios - ref_logratios
```

最终：

```text
loss = -log sigmoid(beta × logits)
```

若策略相对参考模型更偏好 chosen，`logits` 增大，loss 降低。`beta` 控制偏好更新强度：值越大，对同一差值的判别越尖锐。

## 为什么只对 Assistant 区域求和

chosen 和 rejected 通常共享 prompt，真正需要比较的是回复。即使 prompt 有差异，system/user token 也不是模型要优化的输出目标。mask 排除 prompt 和 padding，避免序列长度与上下文 token 干扰偏好分数。

## 启动与训练流程

1. 初始化 DDP、随机种子、精度上下文和可选断点。
2. 从 `from_weight` 各加载一份模型。
3. 策略模型可训练；参考模型调用 `eval()` 和 `requires_grad_(False)` 冻结。
4. 创建 `DPODataset`、Sampler、AdamW 和 GradScaler。
5. 恢复策略/优化器/Scaler 状态。
6. 可选 compile，并只对策略模型做 DDP 包装。
7. 每 step 在 `no_grad` 下计算参考 logprob，再计算策略 DPO loss。
8. 加入策略模型的 MoE 辅助损失并更新。

参考模型不包装 DDP，因为每个进程只需在自己的数据分片和设备上做相同的冻结前向，不需要同步梯度。

## 保存与日志

日志区分：

- `dpo_loss`：偏好目标。
- `aux_loss`：MoE 路由正则。
- `loss`：两者之和。

默认推理权重写入 `../out/dpo_<hidden_size>[_moe].pth`，续训断点写入 `../checkpoints/`。

## 技术取舍与注意事项

- DPO 不需要在线 rollout 和奖励模型，流程比 PPO/GRPO 简单、稳定，但能力上限受离线偏好数据覆盖范围限制。
- 策略与参考模型会同时占用显存，是此脚本主要额外成本。
- chosen/rejected 必须使用相同 Tokenizer 和兼容聊天模板，否则 logprob 不可直接比较。
- 学习率默认很低（`4e-8`），用于降低灾难性遗忘风险；不要直接套用 SFT 的较大学习率。
- `beta` 与数据质量、学习率共同影响偏好强度，需要通过实际评测调整。
- 断点只恢复策略训练状态；参考模型始终从 `from_weight` 重新构造，因此恢复时也必须保持该参数不变。
