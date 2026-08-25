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

