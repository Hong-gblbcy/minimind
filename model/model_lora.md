# `model_lora.py` 介绍

对应源码：[`model_lora.py`](./model_lora.py)

## 文件定位

不依赖 PEFT 的轻量 LoRA 实现，负责给 MiniMind 注入、保存、加载及合并低秩增量权重。

## 关键接口

- `LoRA`：用两个无偏置线性层表示低秩增量。A 使用高斯初始化，B 使用零初始化。
- `apply_lora(model, rank=16)`：为输入输出维度相同的线性层挂载 LoRA 分支，将前向计算改为“原输出 + 低秩增量”。
- `load_lora(model, path)`：加载单独保存的 LoRA 参数，并兼容带 `module.` 前缀的权重。
- `save_lora(model, path)`：只保存 LoRA 分支，兼容 DDP 和 `torch.compile` 包装。
- `merge_lora(model, lora_path, save_path)`：计算 `B @ A` 并合入基础线性层，输出可独立使用的完整 `.pth` 权重。

训练入口是 [`../trainer/train_lora.py`](../trainer/train_lora.py)，推理时由 [`../eval_llm.py`](../eval_llm.py) 加载 LoRA 权重。

