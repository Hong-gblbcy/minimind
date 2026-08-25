# `train_pretrain.py` 介绍

对应源码：[`train_pretrain.py`](./train_pretrain.py)

## 文件定位

MiniMind 的因果语言模型预训练入口，默认从随机初始化模型开始学习纯文本的下一个 token 预测。

## 训练流程

1. `PretrainDataset` 读取 `text` 字段，添加 BOS/EOS，并补齐到固定序列长度。
2. padding 标签设为 `-100`，不参与交叉熵计算。
3. 模型损失由语言模型交叉熵和可选 MoE 路由辅助损失组成。
4. 使用余弦式学习率、混合精度、梯度累积和梯度裁剪更新参数。
5. 定期保存半精度推理权重和完整续训状态。

## 输入输出

- 默认数据：`../dataset/pretrain_t2t_mini.jsonl`。
- 默认 `from_weight=none`，也可指定已有权重继续训练。
- 默认输出：`../out/pretrain_<hidden_size>.pth`。
- 支持 DDP、`torch.compile`、断点续训和 SwanLab。

```bash
cd trainer
python train_pretrain.py
```

## 项目脉络

预训练是 MiniMind 权重演化链的起点：

```text
纯文本 JSONL
  └─> PretrainDataset
        └─> 随机初始化 MiniMind
              └─> next-token prediction
                    └─> pretrain_*.pth
                          └─> Full SFT / 后续训练
```

这一阶段学习的是语言统计规律和基础知识，不使用 user/assistant 角色，也不专门学习指令跟随。后续 `train_full_sft.py` 默认从这里产生的 `pretrain` 权重继续训练。

## 启动阶段的完整流程

### 1. 导入与路径准备

脚本设置 `__package__='trainer'`，并把仓库根目录加入 `sys.path`，从而在 `trainer/` 当前目录下仍能导入 `model` 和 `dataset` 包。提前导入 `datasets` 是对 Windows 下 PyArrow/PyTorch DLL 加载顺序问题的兼容处理。

### 2. 分布式与随机种子

`init_distributed_mode` 检查 `torchrun` 注入的 `RANK`：

- 普通 `python`：返回单进程模式。
- `torchrun`：初始化 NCCL，绑定当前 `LOCAL_RANK` 对应 GPU。

初始种子使用 `42 + rank`，避免不同 GPU 的随机状态完全相同；每个 epoch 又调用 `setup_seed(42 + epoch)`，用于生成可复现的数据顺序。

### 3. 配置与断点

`MiniMindConfig` 由隐藏维度、层数和 MoE 开关构造。`--from_resume 1` 时，`lm_checkpoint` 从 `../checkpoints/` 读取完整训练状态。

“基础权重”和“断点恢复”是两个概念：

- `from_weight`：决定模型初始化时加载哪份普通权重，默认 `none`。
- `from_resume`：恢复模型、优化器、Scaler、epoch 和 step，使训练从中断位置继续。

### 4. 精度上下文

CPU 使用 `nullcontext`；CUDA 根据 `--dtype` 使用 BF16 或 FP16 autocast。`GradScaler` 只在 FP16 时启用，因为 BF16 的指数范围更大，通常不需要动态缩放。

### 5. 模型、数据和优化器

- `init_model` 创建 MiniMind 和 Tokenizer。
- `PretrainDataset` 把文本变成固定长度 token 序列。
- `AdamW` 优化全部模型参数。
- DDP 模式使用 `DistributedSampler` 把样本分到不同进程。

### 6. 编译与 DDP 包装

若开启 `torch.compile`，先编译模型，再包装 `DistributedDataParallel`。这一顺序让每个进程持有编译后的本地模型。

## 单个训练 Step

`train_epoch` 的一次迭代如下：

```text
input_ids / labels -> device
  └─> 计算当前 cosine learning rate
        └─> autocast 前向
              ├─> causal LM loss
              └─> MoE aux loss
                    └─> 除以 accumulation_steps
                          └─> scaled backward
                                └─> 到累积边界时：
                                      unscale -> clip -> step -> update -> zero_grad
```

模型内部将 `logits[..., :-1]` 与 `labels[..., 1:]` 对齐，所以任务是用前文预测下一个 token。Padding 标签为 `-100`，不会进入交叉熵。

若 epoch 最后剩余 step 不满一个累积周期，函数会在循环后补做一次优化器更新，避免丢掉已累积梯度。

## 学习率与梯度累积

`get_lr` 使用余弦曲线从初始学习率逐步下降，但保留 10% 的最低比例。当前全局 step 由 `epoch * iters + step` 估算。

梯度累积把每个 micro-batch 的 loss 除以 `accumulation_steps` 后反向传播，达到指定次数才更新参数。等效 batch size 近似为：

```text
batch_size × accumulation_steps × world_size
```

它能降低单次显存需求，但参数更新频率也相应下降。

## 权重与断点保存

到达 `save_interval` 或 epoch 末尾时，主进程执行两类保存：

1. `../out/pretrain_<hidden_size>[_moe].pth`：仅半精度模型权重，用于推理或后续阶段。
2. `../checkpoints/*_resume.pth`：模型、优化器、Scaler、epoch、step 和实验记录 id，用于续训。

保存前解开 DDP/compile 包装，只让主进程写文件，避免多进程竞争。

## 技术点解释

### 为什么预训练可以从文本本身构造标签

因果语言模型的监督信号就是序列的下一个 token，不需要人工标注。长度为 T 的文本天然提供 T-1 个预测目标，因此可以利用大规模未标注语料。

### 为什么加入 MoE 辅助损失

MoE 主语言模型损失只关心预测是否正确，路由器可能把大量 token 集中到少数专家。辅助损失鼓励负载平衡，使不同专家都有训练机会。

### 为什么使用 AdamW

AdamW 将权重衰减与梯度更新解耦，常用于 Transformer。脚本使用 PyTorch 默认参数，没有额外拆分不衰减参数组，保持实现简洁。

## 注意事项

- 默认相对路径以 `trainer/` 为起点，必须先 `cd trainer`。
- `max_seq_len` 传给数据集控制样本长度，并未改变模型配置中的 `max_position_embeddings`。
- 预训练数据必须含 `text` 字段；聊天格式数据应使用 SFT 脚本。
- CPU 能运行逻辑，但大规模训练速度和低精度算子支持通常不足。
- `use_wandb` 实际导入的是 `swanlab` 并别名为 `wandb`，参数名沿用了通用实验记录习惯。
- 更改 GPU 数量后恢复时，工具会换算 step，但有效 batch size 也会变化，训练轨迹不会与原运行完全一致。
