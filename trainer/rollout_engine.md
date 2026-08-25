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

