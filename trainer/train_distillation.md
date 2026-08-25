# `train_distillation.py` 介绍

对应源码：[`train_distillation.py`](./train_distillation.py)

## 文件定位

MiniMind 的白盒知识蒸馏训练入口，同时加载冻结教师模型和可训练学生模型。

## 损失设计

- `distillation_loss` 对教师 logits 计算软目标概率，对学生 logits 计算对数概率，再用温度缩放 KL 散度匹配分布。
- Ground Truth 部分只在 `SFTDataset` 标记的 assistant token 上计算交叉熵。
- 总损失为 `alpha * CE + (1 - alpha) * KL`。
- 学生为 MoE 时，额外加入路由辅助损失。
- 若教师词表更大，会截取到学生词表大小后再计算蒸馏损失。

## 配置与输出

教师和学生可分别设置隐藏维度、层数、Dense/MoE 结构及初始权重。脚本支持混合精度、梯度累积、DDP、`torch.compile`、断点续训和 SwanLab。

- 默认数据：`../dataset/sft_t2t_mini.jsonl`。
- 默认学生输出：`../out/full_dist_<student_hidden_size>.pth`。

```bash
cd trainer
python train_distillation.py
```

