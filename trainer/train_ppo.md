# `train_ppo.py` 介绍

对应源码：[`train_ppo.py`](./train_ppo.py)

## 文件定位

带 Actor-Critic 的 Proximal Policy Optimization（PPO）训练入口。

## 模型组成

- Actor：待优化的 MiniMind 策略模型。
- Reference：冻结的基础策略，用于计算 KL 惩罚。
- `CriticModel`：复用 MiniMind 主干并增加 token 级 value head。
- `LMForRewardModel`：外部奖励模型，为完整回答打分。

## 训练流程

1. Rollout 引擎生成回答和旧策略 logprob。
2. `calculate_rewards` 组合格式、thinking、重复度和奖励模型得分。
3. 将参考策略 KL 惩罚分配为 token 奖励，并在回答末尾加入外部奖励。
4. 使用 GAE 计算优势和回报，再进行多轮随机 minibatch 更新。
5. Actor 使用 PPO 概率比裁剪；Critic 使用 Value 裁剪；近似 KL 超过阈值时提前停止有效更新。

## 输入输出

- 默认数据：`../dataset/rlaif.jsonl`。
- 默认基础权重：`full_sft`。
- Rollout：`--rollout_engine torch|sglang`。
- 默认 Actor 输出：`../out/ppo_actor_<hidden_size>.pth`。
- 完整断点还保存 Critic、两个优化器和两个学习率调度器。

```bash
cd trainer
python train_ppo.py
```

