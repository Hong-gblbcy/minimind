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

