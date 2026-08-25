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

