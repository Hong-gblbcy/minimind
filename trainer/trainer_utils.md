# `trainer_utils.py` 介绍

对应源码：[`trainer_utils.py`](./trainer_utils.py)

## 文件定位

训练脚本共享的模型初始化、日志、分布式训练、随机种子、断点续训和奖励模型工具。

## 函数与类

| 对象 | 作用 |
| --- | --- |
| `get_model_params` | 统计 Dense/MoE 总参数量与实际激活参数量 |
| `is_main_process` | 判断当前进程是否负责日志与保存 |
| `Logger` | 只在主进程打印 |
| `get_lr` | 计算带最低比例的余弦学习率 |
| `init_distributed_mode` | 根据 `torchrun` 环境变量初始化 NCCL |
| `setup_seed` | 固定 Python、NumPy、PyTorch 和 CUDA 随机状态 |
| `lm_checkpoint` | 保存或读取推理权重和完整续训状态 |
| `init_model` | 创建 MiniMind、加载 Tokenizer 和指定基础权重 |
| `SkipBatchSampler` | 恢复训练时跳过已完成 batch |
| `LMForRewardModel` | 包装外部奖励模型，对候选回复评分 |

## 断点约定

`lm_checkpoint` 会原子写入两个文件：

- `<weight>_<hidden_size>.pth`：半精度模型权重。
- `<weight>_<hidden_size>_resume.pth`：模型、优化器、epoch、step、world size、SwanLab run id 及调用方传入的附加状态。

恢复时如果 GPU 数量变化，会按旧、新 world size 换算已完成 step。`LMForRewardModel` 将对话历史与候选回复整理成奖励模型输入，并把分数裁剪到 `[-3, 3]`。

## 项目脉络

几乎所有训练脚本都遵循相同生命周期：

```text
初始化分布式 / 随机种子
  └─> 创建配置、模型、Tokenizer
        └─> 可选加载断点
              └─> 训练、日志、保存
                    └─> 清理进程组
```

`trainer_utils.py` 把这些横切能力集中起来，使不同算法文件可以聚焦自己的数据和损失。如果修改这里的保存格式、模型加载或 DDP 逻辑，预训练、SFT、DPO、LoRA、蒸馏和 RL 脚本都会受到影响。

## 日志与进程角色

### `is_main_process`

没有初始化 `torch.distributed` 时返回 `True`；DDP 下只有 rank 0 返回 `True`。训练权重保存、普通日志和 SwanLab 初始化都依赖这个判断，避免每张 GPU 重复执行。

### `Logger`

这是对 `print` 的轻量包装。它不写文件、不做日志级别过滤，只负责主进程去重。

## 模型参数统计：`get_model_params`

函数先统计所有参数，再尝试识别不同模型配置可能使用的专家字段名：

- routed expert 数量。
- 每 token 激活专家数。
- shared expert 数量。

它通过参数名中的 `mlp.experts.0.` 估算单专家大小，进而区分：

- Total Params：模型完整存储参数量。
- Active Params：一次 token 前向实际激活的近似参数量。

Dense 模型或无法识别专家时只打印总参数量。

## 学习率：`get_lr`

实现公式为：

```text
lr(step) = lr_base × [0.1 + 0.45 × (1 + cos(π × step / total_steps))]
```

开始约为基础学习率，结束约为 10%。保留非零下限可以避免训练末尾完全停止更新。调用方负责传入全局 step 和总 step。

## 分布式初始化：`init_distributed_mode`

`torchrun` 会设置 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`。函数的行为是：

1. 未发现 `RANK`：返回 0，不初始化 DDP。
2. 使用 NCCL 建立进程组。
3. 把当前进程绑定到 `LOCAL_RANK` GPU。
4. 返回 local rank，供 DDP `device_ids` 使用。

NCCL 面向 NVIDIA GPU；CPU 多进程或其他设备后端不在当前实现范围内。

## 随机种子：`setup_seed`

函数同时设置 Python `random`、NumPy、CPU PyTorch 和所有 CUDA 设备的种子，并要求 cuDNN 使用确定性路径、关闭 benchmark。

这样有利于复现，但确定性内核可能比自动选择的最快内核慢。多进程训练通常在基础种子上加 rank，确保各进程不会产生完全相同的随机序列。

## 断点系统：`lm_checkpoint`

### 保存模式

传入 `model` 时进入保存模式：

1. 创建目标目录。
2. 解开 DDP 与 `torch.compile` 包装。
3. 将模型参数转为 CPU FP16。
4. 先写 `.tmp`，再用 `os.replace` 原子替换正式文件。
5. 收集优化器、epoch、step、world size 和 SwanLab run id。
6. 把额外关键字参数中带 `state_dict` 的对象转换为状态字典，例如 Scaler、Scheduler、Critic。
7. 同样以临时文件方式原子写入 resume 断点。

原子替换降低进程被中断时留下半个权重文件的概率。

### 加载模式

不传 `model` 时尝试读取 `_resume.pth`。若保存时与当前 world size 不同，代码按比例换算 step：

```text
new_step = old_step × old_world_size / new_world_size
```

这样近似保持已经消费的数据量，但 batch 边界会变化，不能保证位级复现。

## 模型初始化：`init_model`

函数统一完成：

1. 从 `tokenizer_path` 加载 Tokenizer。
2. 按 `lm_config` 构造 `MiniMindForCausalLM`。
3. 若 `from_weight != 'none'`，按保存命名规则读取 `.pth`。
4. 用 `strict=False` 装入权重。
5. 打印模型参数与可训练参数量。
6. 移动到目标设备。

使用 `strict=False` 便于某些结构扩展或 LoRA 场景，但也意味着调用者应主动确认日志和配置，避免错误权重被部分加载。

## 恢复采样：`SkipBatchSampler`

普通 Sampler 产生单个样本索引，该类在其外层组装 batch，并跳过指定数量的完整 batch。恢复训练时它能避免真正读取和 Tokenize 已完成样本，比 DataLoader 取出后再 `continue` 更高效。

最后不足一个 batch 的索引仍会返回，`__len__` 使用向上取整计算剩余 batch 数。

## 奖励模型：`LMForRewardModel`

构造函数用外部模型目录加载 Tokenizer 和 `AutoModel`，然后设为 eval 模式。`get_score` 将多轮消息转换为二段结构：

- user：历史对话加最后一个新问题。
- assistant：候选回复。

它调用奖励模型自定义的 `get_score(tokenizer, messages)`，最后把结果裁剪到 `[-3,3]`，避免极端分数支配 RL 更新。

这不是通用 Transformers 标准接口，目标奖励模型必须实现 `get_score`。

## 注意事项

- checkpoint 基于 `torch.save`，只加载可信文件。
- 普通权重和 resume 断点用途不同，不应相互替代。
- `init_model(strict=False)` 可能隐藏结构差异，加载后应结合参数统计和实际评测确认。
- 奖励模型固定使用其自定义接口，换模型前需检查兼容性。
- `setup_seed` 提高复现性，但不同 PyTorch/CUDA 版本、GPU 数量和并行顺序仍可能产生差异。
