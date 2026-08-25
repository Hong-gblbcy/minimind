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

## 项目脉络

知识蒸馏让学生不仅学习数据中的标准 token，还学习教师对整个词表的概率判断：

```text
同一 SFT 输入
  ├─> 冻结教师模型 ─> teacher logits ─┐
  └─> 学生模型 ─────> student logits ─┼─> CE + KL
             ground-truth labels ──────┘
                                           └─> full_dist_*.pth
```

仓库注释给出的典型用途是用 MoE 教师蒸馏 Dense 学生，也可以用更宽/更深教师训练较小学生。它属于白盒蒸馏，因为训练直接访问教师 logits，而不是只读取教师生成的文本。

## `distillation_loss` 详解

温度 T 先软化教师和学生分布：

```text
p_teacher = softmax(teacher_logits / T)
log_p_student = log_softmax(student_logits / T)
```

随后计算：

```text
KL(p_teacher || p_student) × T²
```

温度越高，分布越平缓，学生能看到教师对非目标 token 的相对偏好，即所谓“暗知识”。乘 `T²` 用来补偿温度缩放对梯度幅度的影响。

教师 softmax 位于 `torch.no_grad()` 中并显式 detach，不构建教师反向图。

## 三部分损失

### Ground Truth CE

学生 logits 与右移一位的 SFT labels 计算逐 token 交叉熵，只在 `labels != -100` 的 assistant 位置求平均。

### Distillation KL

同一 mask 筛出学生与教师的有效 logits，再计算温度 KL。若教师词表大于学生词表，代码把教师最后一维截到学生 `vocab_size`。

### 总损失

```text
total = alpha × CE + (1 - alpha) × KL
```

学生为 MoE 时，CE 分支中还加入 `aux_loss`。默认 `alpha=0.5`，标准答案和教师分布权重相等。

## 启动流程

1. 初始化 DDP、种子、输出目录和学生/教师配置。
2. 根据学生配置读取可选续训断点。
3. 建立 autocast 和 SwanLab。
4. 用 `from_student_weight` 加载学生。
5. 用 `from_teacher_weight` 加载教师并完全冻结。
6. 使用学生 Tokenizer 创建 `SFTDataset`。
7. AdamW 只优化学生参数。
8. 恢复学生、优化器和 Scaler。
9. 可选 compile，只对学生做 DDP 包装。
10. 逐 step 同时执行学生和教师前向。

教师不需要 DDP 梯度同步，每个进程在本地设备上执行冻结前向即可。

## 为什么使用同一输入和 Mask

蒸馏比较的是教师与学生在完全相同上下文、相同预测位置的词表分布。如果两者聊天模板、Tokenizer 或截断方式不同，位置就不再对齐，KL 没有明确意义。

当前脚本的数据集使用学生加载时返回的 Tokenizer，默认假设教师与学生共享词表，并且教师词表前 `student_vocab_size` 项与学生 token id 一致。仅仅截断 logits 不能解决两个不同词表之间的语义映射问题。

## 资源开销

蒸馏需要学生前向+反向以及教师前向：

- 教师虽不保存梯度，但仍占模型权重和激活计算时间。
- 学生 logits 与教师 logits 都覆盖词表，长序列时显存开销明显。
- 更大教师通常提高监督质量，但增加训练成本。

## 保存与日志

日志包含总 loss、原始 CE、MoE aux、distill loss、学习率和 ETA。默认输出：

```text
../out/full_dist_<student_hidden_size>[_moe].pth
```

续训断点只保存学生训练状态；教师每次按参数重新加载。

## 注意事项

- 教师和学生必须共享 Tokenizer 及 token id 语义。
- `temperature` 过低接近硬标签，过高会让分布过平；默认 1.5 是学习示例，不是所有任务的最优值。
- `alpha=1` 时只剩 CE，`alpha=0` 时只学教师分布；当前参数没有限制范围，调用者应保持在 `[0,1]`。
- 教师若比学生词表小，代码没有补齐逻辑，会产生形状不匹配。
- 默认相对路径以 `trainer/` 为当前目录。
