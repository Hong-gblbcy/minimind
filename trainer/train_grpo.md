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

## 项目脉络

GRPO 是在线强化学习：回答不是预先存储的，而是当前策略在训练过程中生成的。

```text
RLAIF prompt
  └─> 当前策略每题生成 G 个回答
        └─> 启发式奖励 + Reward Model
              └─> 同题组内标准化优势
                    └─> GRPO / CISPO 策略更新
                          └─> grpo_*.pth
```

与 PPO 相比，GRPO 不训练 Critic，而是用同一 prompt 多个回答的相对奖励作为基线。这减少一个价值模型，但每个 prompt 必须采样多次。

## 奖励函数

### 重复惩罚 `rep_penalty`

函数把文本切成单词或标点，统计 3-gram 重复数量，并把惩罚限制在 `cap=0.5`。重复段落越多，奖励越低；上限避免该单项完全淹没其他信号。

### `calculate_rewards`

每个回答按以下规则累加：

1. 总长度在 20～800 字符之间：`+0.5`，否则 `-0.5`。
2. 若包含 `</think>`：
   - thinking 长度 20～300：`+1.0`，否则 `-0.5`。
   - 恰好一个闭合标签：`+0.25`，否则 `-0.25`。
3. 对最终答案施加重复惩罚。
4. 把 prompt 解析回 role/content 消息，交给 `LMForRewardModel` 打分。

最终奖励是启发式格式分与外部奖励模型分数之和。这里的格式规则是训练设计，不是通用事实；换数据或目标时需要重新评估是否合理。

## Rollout 与张量对齐

训练先对字符串 prompt 做左 padding，并从右侧保留最多 `max_seq_len` token。左 padding 能让每条样本的真实 prompt 尾部对齐，便于自回归生成。

Rollout 返回：完整 ids、completion ids、文本、旧 logprob、prompt 长度和 mask。代码用：

```text
logp_pos = prompt_len - 1 + completion_position
```

把完整序列 logits 映射到每个生成 token。减 1 是因为位置 t 的 logits 预测 token t+1。

随后查找每条回答的第一个 EOS，只让 EOS 及之前、且非 padding 的位置进入策略损失。

## 组内相对优势

每个 prompt 有 G 个奖励，重排为 `[batch, num_generations]` 后计算：

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-4)
```

优势为正表示该回答优于同题其他回答，为负表示更差。这样不需要 Critic 估计绝对价值，也能消除不同题目天然难度和奖励尺度差异。

若同组所有回答奖励完全相同，标准差接近 0，所有优势也接近 0，这一组几乎不提供策略梯度。因此奖励函数必须能区分同题回答质量。

## 参考模型 KL

冻结参考模型与策略模型对同一生成序列计算逐 token logprob。代码使用：

```text
log_ratio_ref = log π_ref - log π_policy
KL estimator = exp(log_ratio_ref) - log_ratio_ref - 1
```

这是非负的逐样本 KL 估计形式。`beta` 控制惩罚强度，限制策略偏离 `full_sft` 基础行为过快。

## GRPO 与 CISPO 损失

### GRPO

```text
ratio = exp(log π_current - log π_old)
ratio_clipped = clip(ratio, 1-epsilon, 1+epsilon)
policy term = min(ratio × A, ratio_clipped × A)
```

裁剪防止一次更新把新策略推得离生成策略太远。

### CISPO

```text
weight = min(ratio, epsilon_high).detach()
policy term = weight × A × log π_current
```

概率比被截断并 detach，仅作为权重；梯度直接通过当前 logprob 传播。这样即使 ratio 达到上限，策略项仍保留梯度路径。

两种损失最后都减去 `beta × KL`，再在有效 completion token 上按每条回答长度归一化，最后对 batch 求平均。

## 优化与调度

- AdamW 更新策略模型。
- `CosineAnnealingLR` 按实际 optimizer step 调度到初始学习率的 1/10。
- loss 支持梯度累积；到边界后裁剪梯度、step、scheduler step、zero grad。
- epoch 末尾不足累积周期时补做一次更新。
- MoE 模型把路由辅助损失加入策略 loss。

## 模型组成与初始化

训练同时持有：

- Policy：可训练，默认从 `full_sft` 加载。
- Reference：同权重初始化后冻结。
- Reward Model：外部模型，提供标量评分。
- Rollout Engine：Torch 或 SGLang。

恢复断点时会加载 Policy、优化器和 scheduler；Reference 与 Reward Model 每次重新从参数路径加载。

## 策略同步与保存

主进程按 `save_interval` 保存推理权重和完整断点。相同时间点调用 `rollout_engine.update_policy(model)`：

- Torch：更新模型引用，通常仍是同一对象。
- SGLang：导出并通知服务热加载。

在两个保存间隔之间，SGLang 服务可能继续用上次同步的策略生成，旧 logprob 仍与实际生成策略对应，但策略滞后程度会增大。

## 注意事项

- `num_generations` 越大，组内基线越稳定，但生成成本线性增加。
- Reward Model 必须提供项目期望的 `get_score` 接口。
- 奖励模型与启发式规则都可能被策略“钻空子”，应使用 debug 模式定期查看真实样本。
- `thinking_ratio` 只决定 prompt 模板是否打开 thinking，不保证生成一定正确闭合标签。
- Torch 与 SGLang 返回格式不同，核心训练依赖 `RolloutResult` 做统一。
- DDP 下每个进程都会执行自己的 rollout 和奖励计算；外部服务容量要能承受并发请求。
