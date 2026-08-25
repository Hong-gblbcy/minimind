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

