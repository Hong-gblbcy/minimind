# `train_full_sft.py` 介绍

对应源码：[`train_full_sft.py`](./train_full_sft.py)

## 文件定位

MiniMind 的全参数监督微调（Full SFT）入口，在预训练权重上学习多轮对话与指令跟随能力。

## 训练流程

1. `SFTDataset` 应用聊天模板，并只为 assistant 回复区域生成有效标签。
2. 默认加载 `pretrain` 权重，优化模型全部参数。
3. 前向损失由因果语言模型交叉熵和可选 MoE 路由辅助损失组成。
4. 使用学习率调度、梯度累积、梯度裁剪和混合精度完成更新。
5. 定期保存半精度推理权重及包含优化器、Scaler 和进度的续训断点。

## 输入输出

- 默认数据：`../dataset/sft_t2t_mini.jsonl`。
- 默认基础权重：`pretrain`。
- 默认输出：`../out/full_sft_<hidden_size>.pth`。
- 支持 DDP、`torch.compile`、断点续训和 SwanLab。

```bash
cd trainer
python train_full_sft.py
```

