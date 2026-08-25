# `train_grpo.py` 介绍

对应源码：[`train_grpo.py`](./train_grpo.py)

## 文件定位

Group Relative Policy Optimization（GRPO）训练入口，也支持 CISPO 损失变体。

## 训练流程

1. `RLAIFDataset` 为每条对话生成待回答 prompt。
2. Rollout 引擎为每个 prompt 生成多条回答，并记录旧策略逐 token logprob。
3. `calculate_rewards` 综合回答长度、thinking 格式、重复惩罚和外部奖励模型分数。
4. 同一 prompt 的奖励按组计算均值和标准差，标准化为相对优势。
5. 冻结参考模型提供逐 token KL 惩罚，策略模型使用 GRPO 或 CISPO 目标更新。

`loss_type=grpo` 使用上下界裁剪的概率比；`loss_type=cispo` 使用截断后的概率比权重乘当前 logprob，使权重截断后仍保留梯度路径。

## 输入输出

- 默认数据：`../dataset/rlaif.jsonl`。
- 默认基础权重：`full_sft`。
- Rollout：`--rollout_engine torch|sglang`。
- 默认输出：`../out/grpo_<hidden_size>.pth`。
- 支持 DDP、`torch.compile`、断点续训、调试采样和 SwanLab。

```bash
cd trainer
python train_grpo.py
```

