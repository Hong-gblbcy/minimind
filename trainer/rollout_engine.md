# `rollout_engine.py` 介绍

对应源码：[`rollout_engine.py`](./rollout_engine.py)

## 文件定位

PPO、GRPO 和 Agent RL 共用的在线采样抽象层，将“策略生成”和“训练算法”解耦。

## 核心对象

| 对象 | 作用 |
| --- | --- |
| `compute_per_token_logps` | 计算指定输出 token 在模型下的逐 token 对数概率 |
| `RolloutResult` | 统一封装完整输出、completion、旧策略 logprob、文本和 mask |
| `RolloutEngine` | 定义 `rollout` 与 `update_policy` 接口 |
| `TorchRolloutEngine` | 直接使用当前 PyTorch 策略模型生成并计算 logprob |
| `SGLangRolloutEngine` | 通过 SGLang HTTP API 生成和热更新权重 |
| `create_rollout_engine` | 按 `torch` 或 `sglang` 创建实现 |

## 两种引擎

`TorchRolloutEngine` 不需要外部服务，适合直接训练和调试。它重复 prompt、调用模型生成、分离 completion，并返回 padding mask 与旧策略 logprob。

`SGLangRolloutEngine` 调用 `/generate` 获取输出和 logprob。`update_policy` 会把当前策略保存到共享目录，再调用 `/update_weights_from_disk`；此外提供 `/health` 检查和 `/flush_cache` 缓存清理。

使用 SGLang 前需要单独启动服务，源码顶部给出了启动命令示例。

## 项目脉络

在线强化学习包含两个节奏不同的环节：

```text
当前策略生成回答（rollout）
  └─> 记录回答 token 和旧策略 logprob
        └─> 奖励 / 优势 / 策略损失
              └─> 更新当前策略
                    └─> 下一轮 rollout 使用新策略
```

PPO、GRPO 和 Agent RL 都需要这条链路，但算法本身不应关心生成发生在训练进程还是独立推理服务器。`RolloutEngine` 用统一接口隐藏后端差异。

## 为什么必须返回旧策略 Logprob

策略梯度常使用概率比：

```text
ratio = exp(log π_current(token) - log π_old(token))
```

`π_old` 是生成这批回答时的策略，`π_current` 是执行更新时的策略。若只保存文本而不保存旧 logprob，就无法准确计算 PPO/GRPO/CISPO 的重要性采样比率。

## `compute_per_token_logps`

输入包括模型、完整 `input_ids`、需要保留的末尾 token 数 `n_keep` 和可选 attention mask。

函数执行：

1. 解开 DDP 包装。
2. 对 inference tensor 做 clone，避免后续 autograd/inference mode 限制。
3. 请求模型只计算末尾 `n_keep + 1` 个位置的 logits。
4. 去掉最后一个无目标位置。
5. 对 logits 做 `log_softmax`。
6. 用 `gather` 取出实际生成 token 的 logprob。

输出形状为 `[batch, n_keep]`。只投影末尾位置可以减少语言模型头对完整 prompt 做词表投影的成本。

## `RolloutResult` 字段语义

| 字段 | 典型形状 | 含义 |
| --- | --- | --- |
| `output_ids` | `[B×G, P+R]` | prompt 与 completion 拼接后的完整序列 |
| `completion_ids` | `[B×G, R]` | 仅新生成部分 |
| `per_token_logps` | `[B×G, R]` | 生成时策略对 completion 的 logprob |
| `completions` | `List[str]` | 解码后的回答文本 |
| `prompt_lens` | `[B×G]` | 每条样本真实或 padded prompt 长度 |
| `completion_mask` | `[B×G, R]` | completion 中真实 token 为 1，padding 为 0 |

其中 B 是 prompt batch，G 是每个 prompt 的生成数量，P/R 分别是 prompt/completion 长度。

## Torch Rollout 详细流程

`TorchRolloutEngine` 持有当前策略对象和 Tokenizer：

1. `repeat_interleave(num_generations)` 为每个 prompt 复制 G 份输入。
2. 在 `torch.no_grad()` 和可选 autocast 中调用模型自定义 `generate`。
3. 用原 batch 的 padded prompt 宽度切出 completion。
4. 根据 pad id 构造完整 attention mask。
5. 调用 `compute_per_token_logps` 重新计算回答 token 的旧策略概率。
6. 批量解码 completion。

项目自定义生成会让已结束序列继续填 EOS，因此 Torch 引擎初始 `completion_mask` 全为 1。算法层随后查找首个 EOS，重新得到真正有效的策略区域。

`update_policy` 只替换内存中的模型引用。训练脚本通常原地更新同一个模型，这个操作成本很低。

## SGLang Rollout 详细流程

### 请求准备

SGLang 接受每条样本独立的 token 列表。引擎先根据 attention mask 去掉左 padding，再为每个 prompt 复制 G 次，向 `/generate` 发送：

- `input_ids`。
- temperature、最大新 token 数、EOS stop token。
- `return_logprob=True`。

### 响应兼容

不同 SGLang 版本可能把输出 id/logprob 放在顶层或 `meta_info`。代码兼容两种位置，也兼容 logprob 是数字或 tuple/list 的形式。

若 logprob 数量与 completion token 数不同，代码从左补 0 或截取末尾，使张量维度对齐。补 0 是容错手段，不代表真实策略概率；频繁出现时应检查 SGLang 返回协议版本。

### 重新 Padding

HTTP 返回的序列长度不同。引擎把完整输出、completion 和 logprob 分别补齐为张量，并精确构造 `prompt_lens` 与 `completion_mask`，让训练算法可以统一处理变长输出。

## 策略热更新

训练进程更新参数后，独立 SGLang 服务并不会自动知道。`update_policy` 在 rank 0：

1. 解开 DDP/compile 包装。
2. 把当前权重转为 CPU FP16。
3. 以 Transformers 格式保存到共享目录。
4. 保存 Tokenizer。
5. 调用 `/update_weights_from_disk`，要求服务重载该目录。

DDP 下 rank 0 会把成功状态广播给所有进程，并 barrier 对齐；任一更新失败都会抛出异常，避免不同进程在策略版本上继续分叉。

## 工厂函数

`create_rollout_engine` 让训练脚本只通过字符串切换后端。Torch 需要传策略模型和 Tokenizer；SGLang 需要服务 URL、Tokenizer/模型路径和共享权重目录。不支持的类型立即抛出 `ValueError`。

## 技术取舍与注意事项

- Torch 后端简单且策略始终最新，但生成与训练共享 GPU，吞吐可能较低。
- SGLang 可用专门推理引擎加速 rollout，但增加 HTTP、共享存储和权重同步复杂度。
- SGLang 服务必须能够访问训练进程传入的同一个绝对共享路径；若服务在另一台没有共享文件系统的机器上，热更新会失败。
- HTTP 没有认证配置，只应在可信网络使用。
- `timeout` 默认 120 秒，长生成或拥塞时可能需要调整。
- `health` 只检查 `/health` HTTP 状态，不能证明模型版本正确或生成质量正常。
- 训练脚本决定何时调用 `update_policy`；如果更新间隔大于一个优化 step，远端 rollout 使用的可能是滞后策略。
