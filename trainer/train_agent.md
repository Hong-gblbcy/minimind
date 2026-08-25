# `train_agent.py` 介绍

对应源码：[`train_agent.py`](./train_agent.py)

## 文件定位

面向多轮 Tool-Use 的 Agent 强化学习入口，使用组内相对优势，可选择 GRPO 或 CISPO 风格损失。

## 训练流程

1. `AgentRLDataset` 提供初始消息、可用工具和 ground truth。
2. `rollout_single` 执行“模型调用工具 → 注入工具结果 → 模型继续回答”的多轮采样。
3. `rollout_batch` 为每条样本生成多组轨迹，并保存策略 token、观察 token及旧策略 logprob。
4. `calculate_rewards` 根据工具合法性、调用数、GT 命中、是否完成、重复度和奖励模型打分生成奖励。
5. `rl_train_epoch` 做组内奖励标准化，加入参考模型 KL 惩罚，以 GRPO/CISPO 更新策略。

## 工具环境

脚本内置数学、单位换算、天气、时间、汇率和翻译工具，以及确定性的模拟数据。`parse_tool_calls` 解析 XML 标签内的 JSON，`execute_tool` 带一秒超时保护，`validate_gt_in_text` 校验文本或数值答案。

## 输入输出

- 默认数据：`../dataset/agent_rl.jsonl`。
- Rollout：`--rollout_engine torch|sglang`。
- 默认基础权重：`full_sft`。
- 默认输出：`../out/agent_<hidden_size>.pth`。
- 支持 DDP、`torch.compile`、断点续训、调试采样和 SwanLab。

```bash
cd trainer
python train_agent.py
```

