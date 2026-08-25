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

## 项目脉络

普通 GRPO 只优化“给定 prompt 后的一段回答”。Agent RL 则把工具调用产生的观察插入轨迹，让策略学习多轮行动：

```text
messages + tools
  └─> 模型生成 Action: <tool_call>
        └─> 环境执行工具
              └─> Observation: tool result
                    └─> 模型继续 Action / Final Answer
                          └─> GT 与格式奖励
                                └─> GRPO / CISPO 更新
```

它复用 `rollout_engine.py` 的单次生成能力，但在本文件中额外实现多轮编排、工具环境、轨迹打包和可验证奖励。

## 工具定义与模拟环境

脚本内置六种工具：数学计算、单位换算、天气、时间、汇率和翻译。每个工具有三层信息：

- `TOOLS`：模型可见的 JSON Schema。
- `MOCK_RESULTS`：环境实际执行函数和固定模拟数据。
- `CHECK_ARGS`：奖励阶段判断调用参数是否完整。

`execute_tool` 使用 `signal.alarm(1)` 设置一秒超时。数学表达式的 `eval` 禁用了 builtins，只暴露 `math`，比普通直接 eval 更受限，但仍是仅供本地训练模拟的实现。

天气、时间、汇率和翻译使用固定表，保证同一输入得到确定结果，便于 ground truth 验证和复现实验。

## 工具调用解析

`parse_tool_calls` 查找 `<tool_call>...</tool_call>`，把标签内部解析为 JSON。非法 JSON 会被忽略，不会中断整批训练。

模型生成结构形如：

```json
{"name": "get_current_weather", "arguments": {"location": "北京"}}
```

工具名必须存在于当前样本提供的工具集合，参数还需通过 `CHECK_ARGS`，才能在奖励中算作有效调用。

## 单条多轮轨迹：`rollout_single`

### 初始状态

每条轨迹开始时随机决定一次 `open_thinking`，后续所有 turn 保持相同设置。首次聊天模板编码得到的 token 作为 `prompt_ids`。

### 每个 Turn

1. 对当前 messages 应用带 tools 的聊天模板。
2. 调用 rollout 引擎生成一段 completion。
3. 保存生成 token、旧策略 logprob，并将它们的 `response_mask` 设为 1。
4. 解析工具调用；若没有调用，轨迹结束。
5. 把 assistant 调用写入 messages。
6. 执行每个工具，把 JSON 结果作为 tool 消息写回。
7. 再次应用聊天模板，计算新出现的观察 token。
8. 观察 token 的 `response_mask` 设为 0，旧 logprob 设为 0。

默认最多 3 个 turn。到最后一个 turn 仍在调用工具时，`unfinished=True`，奖励会扣分。

### 为什么区分策略 Token 和观察 Token

工具结果由环境提供，不是策略采样动作。训练时它们必须出现在上下文中，让后续回答能够注意到；但不能对这些 token 计算策略梯度，否则模型会被要求“生成环境观察”。

因此轨迹中：

```text
prompt token      mask=0
assistant/action  mask=1
tool observation  mask=0
next action/answer mask=1
```

这是 Agent 轨迹与普通连续 completion 的核心区别。

## 批量 Rollout：`rollout_batch`

函数按样本、再按 `num_gen` 循环调用 `rollout_single`。每个生成都会复制消息，避免某条轨迹的工具结果污染同题其他采样。

当前实现是顺序多轮采样，不是真正把所有轨迹合成一个大 batch；逻辑直观，但 Tool-Use rollout 会成为主要性能瓶颈。

返回内容不仅包括最终回答，还包括完整上下文、prompt/response ids、策略 mask、旧 logprob、每轮输出和 unfinished 状态。

## 轨迹打包

`rl_train_epoch` 把每条变长轨迹整理为统一张量：

1. 拼接初始 prompt 和后续 response/observation token。
2. prompt mask 补 0，策略生成位置使用 rollout 保存的 mask。
3. 旧 logprob 按“预测下一个 token”的位置对齐。
4. 超过 `max_total_len` 时从左侧截断，只保留末尾。
5. 根据第一个策略 token 重新估算 prompt 长度。
6. 右 padding 到 batch 最大长度。

`full_response_masks[:, 1:]` 最终与 logits 的目标位置对齐，确保策略 loss 只覆盖模型自己生成的动作。

## Ground Truth 校验

`validate_gt_in_text` 同时支持字符串和数值：

- 字符串：忽略大小写，检查是否出现在最终文本。
- 数值：提取回答中的整数/小数，去掉千位逗号后按 `1e-6` 容差比较。

返回命中的 GT 集合，可处理一道题要求多个结果的情况。

## 奖励设计

奖励分为两条路径。

### 没有工具调用

- 回答长度分。
- thinking 长度和标签闭合分。
- 外部 Reward Model 分数。
- 3-gram 重复惩罚。

适用于样本不需要工具、模型直接回答的情况。

### 存在工具调用

1. 统计工具名属于当前可用集合且参数合法的调用数量。
2. 比较有效调用数、实际调用数与 GT 数量，形成 `tool_gap`。
3. 调用数量完全对齐得正分，否则按差距扣分。
4. 在最终非工具文本中验证 GT，按命中比例最多加 2.5。
5. 未完成多轮交互扣分。
6. 对最终答案施加重复惩罚。

两条路径最终都裁剪到 `[-3,3]`，限制极端奖励。

工具路径主要依靠可验证 GT，不调用外部 Reward Model；无工具路径则依靠 Reward Model 评价开放回答。

## 组内优势与策略损失

同一 prompt 的 G 条轨迹按奖励做组内标准化：

```text
A_i = (r_i - mean_group) / (std_group + 1e-4)
```

当前模型和冻结 Reference 对完整打包轨迹计算逐 token logprob，但 completion mask 只选策略 token。KL 估计和 ratio 与普通 GRPO 相同：

```text
ratio = exp(log π_current - log π_old)
KL = exp(log π_ref - log π_current) - (log π_ref - log π_current) - 1
```

根据 `loss_type` 使用 GRPO clip 或 CISPO 截断权重。每条轨迹先按有效策略 token 数归一化，再对有效轨迹求平均。

## 模型、优化与恢复

训练持有 Policy、冻结 Reference、Reward Model 和 Rollout Engine。AdamW 配合 CosineAnnealingLR 更新 Policy，支持梯度累积、裁剪、DDP、compile 和断点恢复。

DataLoader 使用自定义 `collate_fn` 保留每条样本不同长度的 messages、tools 和 gt，避免默认 collate 尝试把嵌套字典堆成张量。

## 保存与策略同步

- 推理权重：`../out/agent_<hidden_size>[_moe].pth`。
- Resume：Policy、优化器、scheduler、epoch、step。
- Rollout 策略：保存间隔到达时调用 `update_policy`；SGLang 会热加载共享目录。

## 注意事项

- `signal.SIGALRM` 是 Unix 风格机制，Windows 环境没有相同支持。
- 工具执行和数据工具 schema 必须匹配；数据可以声明工具，但环境只实现本文件中的六种函数。
- 工具结果截断到 2048 字符，防止异常大结果撑爆上下文。
- 顺序多轮 rollout 成本高，`num_generations`、turn 数和生成长度会相乘。
- 从左截断超长轨迹可能删除早期 system/tool 定义，`max_total_len` 需覆盖关键上下文。
- 固定模拟数据有利于学习流程，但训练出的工具知识不能直接代表真实世界状态。
- Reward 规则可能被模型利用，应结合 `debug_mode` 检查工具调用、完整上下文、mask 和奖励。
