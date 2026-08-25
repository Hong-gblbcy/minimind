# `train_lora.py` 介绍

对应源码：[`train_lora.py`](./train_lora.py)

## 文件定位

MiniMind 的参数高效 LoRA 微调入口，仅训练挂载到线性层上的低秩增量参数。

## 训练流程

1. 加载基础模型和 Tokenizer。
2. 调用 [`../model/model_lora.py`](../model/model_lora.py) 的 `apply_lora` 注入低秩分支。
3. 冻结全部基础参数，只收集名称含 `lora` 的参数交给优化器。
4. 使用 `SFTDataset` 和因果语言模型损失训练 LoRA 参数。
5. `save_lora` 只保存低秩分支；续训断点另存完整训练状态。

## 输入输出

- 默认数据：`../dataset/lora_medical.jsonl`。
- 默认基础权重：`full_sft`。
- 默认输出：`../out/lora_medical_<hidden_size>.pth`。
- 支持混合精度、DDP、梯度累积、断点续训和 SwanLab。

LoRA 通过 monkey-patch 改写线性层前向函数，与 `torch.compile` 不兼容，因此脚本会自动关闭编译。

```bash
cd trainer
python train_lora.py
```

## 项目脉络

LoRA 复用 Full SFT 的数据和语言模型目标，但改变参数更新范围：

```text
full_sft 基模 + 领域 conversations
  └─> 注入 LoRA A/B
        ├─> 基础参数 requires_grad=False
        └─> LoRA 参数 requires_grad=True
              └─> 领域 LoRA 权重
                    ├─> 与基模组合推理
                    └─> 合并后导出完整模型
```

它适合身份、医疗等垂直数据微调。不同 LoRA 可以共享同一份基模，每个适配器文件只保存自己的增量。

## 启动流程

前四步与其他训练脚本一致：初始化 DDP、随机种子、配置/断点、混合精度和 SwanLab。随后进入 LoRA 特有流程：

1. `init_model` 默认读取 `full_sft` 权重。
2. `apply_lora(model)` 给符合条件的方形线性层注入 A/B 分支。
3. 遍历 `named_parameters()`：名称含 `lora` 的参数保持可训练，其余全部冻结。
4. 统计总参数、LoRA 参数量及占比。
5. AdamW 只接收 `lora_params`，因此不会为基础参数创建优化器状态。
6. `SFTDataset` 提供 assistant-only 监督数据。

## 单个训练 Step

前向仍计算完整模型：基础权重提供原始能力，LoRA 分支提供可学习增量。反向图会经过基础模型，但冻结参数不保存梯度，只有 A/B 得到更新。

梯度裁剪也只作用于 `lora_params`：

```text
loss -> backward
  └─> LoRA grads
        └─> clip_grad_norm_(lora_params)
              └─> optimizer.step()
```

语言模型 loss 与 Full SFT 相同，MoE 路由辅助损失也会加入总 loss。不过基础 MoE 参数若被冻结，辅助项对这些冻结参数不会产生更新；只有它经过的可训练 LoRA 分支可能收到梯度。

## 权重保存与恢复

### 推理权重

`save_lora` 只保存每个已注入层的：

```text
lora.A.weight
lora.B.weight
```

默认文件为：

```text
../out/lora_medical_<hidden_size>[_moe].pth
```

这个文件不能单独构造模型，推理时必须先加载与训练时一致的 `full_sft` 基模。

### 续训断点

`lm_checkpoint` 保存的是当前模型的完整 state_dict、优化器、Scaler 和进度。恢复时用 `strict=False` 加载，因为注入 LoRA 后的模型结构与纯基模不同，而且断点可能包含额外包装前缀。

## 技术点解释

### 参数量为什么显著减少

对 C×C 方形线性层，完整更新需要 C² 个参数；rank 为 r 的 LoRA 只增加 `2Cr`。当 `r << C` 时，训练参数和 AdamW 的一阶/二阶状态都会大幅减少。

### 为什么基础参数仍参与前向

“冻结”只表示不更新，不表示跳过计算。模型仍需要完整基础网络产生表示，LoRA 只是叠加增量。因此 LoRA 主要节省梯度和优化器显存，前向 FLOPs 只减少有限部分，并非把基模计算完全省掉。

### 为什么不使用 `torch.compile`

项目的 `apply_lora` 在运行时把线性层 `forward` 替换为 Python 闭包。编译器难以对这种 monkey patch 建立稳定静态图，所以脚本发现 `use_compile=1` 后会改回 0 并输出说明。

## 注意事项

- 当前 `apply_lora` 只注入输入输出维度相同的线性层，不是所有投影层。
- `lora_name` 只是保存前缀，应同时记录对应基模、数据和配置。
- LoRA rank 在 `apply_lora` 中默认固定为 16，训练 CLI 没有暴露 rank 参数。
- 更换基模尺寸或 Dense/MoE 类型后，旧 LoRA 不能直接复用。
- 默认路径以 `trainer/` 为当前目录。
