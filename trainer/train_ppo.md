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
3. 把外部奖励放在回答末尾，并用参考策略 KL 作为 Actor 目标中的独立惩罚项。
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

## 项目脉络

PPO 是仓库中模型组件最多的在线强化学习流程：

```text
RLAIF prompt
  └─> Actor rollout ─> 回答 + old logprob
        ├─> Reward Model ─> 终局奖励
        ├─> Reference ────> KL 约束
        └─> Critic ───────> token value
                              └─> GAE advantage / return
                                    ├─> 更新 Actor
                                    └─> 更新 Critic
```

相比 GRPO，PPO 每个 prompt 默认只生成一条回答，但需要 Critic 学习价值基线。它通过多轮 minibatch 重用同一批 rollout，提高采样利用率。

## 四个模型角色

### Actor

Actor 是可训练的 `MiniMindForCausalLM`，负责生成回答，并通过 PPO 策略损失更新。默认从 `full_sft` 权重开始。

### Reference

Reference 从相同基础权重加载后冻结。它不决定优势，而是在 Actor loss 中提供 KL 惩罚，限制策略偏离基础模型。

### Critic

`CriticModel` 继承 `MiniMindForCausalLM`，复用 Transformer 主干，并增加 `hidden_size -> 1` 的 `value_head`。每个序列位置输出一个标量，表示从该位置继续生成的预期回报。

初始化时先把基础语言模型 state_dict 以 `strict=False` 加载进 Critic，主干继承 Actor 的语言表示，新增 value head 从随机参数开始学习。

### Reward Model

`LMForRewardModel` 对完整回答给出标量质量分。它只参与推理，不反向传播。

## 奖励构造

`calculate_rewards` 与 GRPO 使用相似启发式：

- 回答长度 20～800 字符得到正分，否则扣分。
- thinking 长度和闭合标签格式得到额外分数。
- 3-gram 重复产生最多 0.5 的惩罚。
- 外部奖励模型评价最终答案。

最终得到每条回答一个标量 `rewards[B]`。

## Rollout 与有效区域

Prompt 使用左 padding，并截断到 `max_seq_len`。Rollout 引擎生成最多 `max_gen_len` 个 token，返回旧策略 logprob。

代码根据 prompt 长度建立 `logp_pos`，再结合：

- SGLang/Torch 返回的 completion padding mask。
- 每条回答第一个 EOS 的位置。

构造 `resp_policy_mask` 和 `resp_value_mask`。只有真实回答 token（包括首个 EOS）参与 Actor/Critic 损失，后续 padding 或重复填充 EOS 被排除。

## Value、奖励时序与 GAE

在 `torch.no_grad()` 下，Critic 先对完整序列预测旧 value，Reference 计算回答 token 的 logprob。

外部奖励只加到每条有效回答的最后一个 token：

```text
r_0 = 0, r_1 = 0, ..., r_last = reward_model_score + heuristic
```

参考模型 KL 不写进 `token_rewards`，而是在后续 Actor loss 中单独加入。

GAE 从后向前递推：

```text
delta_t = r_t + gamma × V_{t+1} - V_t
A_t = delta_t + gamma × lambda × A_{t+1}
return_t = A_t + V_t
```

`gamma` 控制未来奖励折扣，`lambda` 在偏差与方差之间折中。优势随后只在有效 token 上计算均值/方差并标准化，改善优化数值尺度。

## PPO Minibatch 更新

同一批 rollout 会重复 `ppo_update_iters` 轮，每轮随机打乱 batch，并按 `mini_batch_size` 切分。

### Actor 概率比

```text
log_ratio = log π_current - log π_old
ratio = exp(log_ratio)
```

策略损失使用未裁剪和裁剪两项中的保守结果：

```text
max(-A × ratio, -A × clip(ratio, 1-eps, 1+eps))
```

它限制同一批数据被重复优化时，当前策略离生成策略变化过大。

### Reference KL

代码用：

```text
exp(log π_ref - log π_current) - (log π_ref - log π_current) - 1
```

构造非负 KL 估计，并乘 `kl_coef` 加到 Actor loss。这里约束的是 Actor 相对基础参考模型的长期漂移。

### Critic Value 裁剪

Critic 同时比较：

- 当前 value 与 return 的平方误差。
- 把当前 value 限制在旧 value ± `cliprange_value` 后的平方误差。

取两者较大值，阻止 Critic 利用大幅 value 跳变得到虚假的小损失。

### 两种 KL 不要混淆

- `approx_kl`：当前 Actor 与 rollout old policy 的近似 KL，用于判断这一批是否更新过头。
- `kl_ref`：当前 Actor 与冻结 Reference 的 KL，作为长期正则项进入 loss。

两者基准不同、用途也不同。

## KL 早停与 DDP

当 `approx_kl > early_stop_kl` 时设置 `stop_ppo`。DDP 下先对各卡 KL 做 all-reduce 平均，确保所有进程作出相同决定。

已经进入当前 minibatch 的各进程不能有的直接 break、有的继续，否则梯度通信会死锁。代码因此仍完成 forward/backward，但把 loss 乘 0；下一轮 PPO epoch 才停止。这个设计优先保证 DDP 通信闭环。

## 优化器、调度器与梯度累积

Actor 和 Critic 各有一个 AdamW 与 CosineAnnealingLR，学习率可分别设置。每个 minibatch 计入 `grad_accum_step`，达到累积边界时：

1. 分别裁剪 Actor/Critic 梯度。
2. 两个优化器 step。
3. 两个 scheduler step。
4. 清空梯度。

每个 rollout batch 结束时若有余数，会立即补做一次更新，因此累积边界主要作用于该批内部的 PPO minibatch。

## 保存与策略同步

推理文件只保存 Actor：

```text
../out/ppo_actor_<hidden_size>[_moe].pth
```

resume 断点另外保存 Critic、Actor/Critic 优化器和调度器。Rollout 引擎在每个外层 step 结束、满足保存间隔时同步 Actor；SGLang 模式下，间隔内可能使用稍旧的远端策略。

## 日志指标解释

- `Reward`：当前 rollout 平均最终奖励。
- `KL_ref`：Actor 与 Reference 的距离。
- `Approx KL`：当前 Actor 与 old policy 的更新幅度。
- `ClipFrac`：概率比超出裁剪区间的 token 比例。
- `Critic Loss`：Value 拟合误差。
- `Avg Response Len`：有效回答 token 平均长度。

当前实现每个外层 rollout step 都输出这些指标，命令行 `log_interval` 参数没有用于包裹该日志分支。

## 资源与注意事项

- 同时持有 Actor、Reference、Critic 和 Reward Model，显存开销显著高于 SFT/GRPO。
- Reward hacking、Critic 发散和 KL 过大都需要结合 debug 样本与指标诊断。
- `debug_log_ratio` 可检查第一次 minibatch 的 ratio 是否接近 1；若偏差很大，说明 rollout 与更新阶段策略或精度不一致。
- Critic 的 value head 没有预训练价值监督，需要足够数据逐步校准。
- SGLang 热更新依赖共享文件系统和兼容服务版本。
- 恢复训练时必须保持基础权重、Critic 结构和优化器配置兼容。
