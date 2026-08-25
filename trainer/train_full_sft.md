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

## 项目脉络

Full SFT 位于预训练和偏好/强化学习之间：

```text
pretrain_*.pth
  + 多轮 conversations JSONL
        └─> SFTDataset
              └─> 全参数监督微调
                    └─> full_sft_*.pth
                          ├─> 直接聊天推理
                          ├─> LoRA
                          ├─> DPO
                          ├─> PPO / GRPO
                          └─> Agent RL
```

预训练模型学会“续写文本”，SFT 则通过聊天模板和 assistant-only labels 教它“根据 system/user/tool 上下文，以 assistant 身份回答”。`full_sft` 因此也是仓库多数后训练算法的默认基础权重。

## 与预训练脚本的共同框架

这个文件与 `train_pretrain.py` 共享同一套工程骨架：

- `torchrun`/NCCL 分布式初始化。
- BF16/FP16 autocast 与 FP16 GradScaler。
- AdamW、梯度累积和梯度裁剪。
- `torch.compile` 可选加速。
- `SkipBatchSampler` 按 step 恢复。
- 主进程日志、SwanLab 和原子断点保存。

差异集中在数据集、初始权重和默认超参数，而不是训练基础设施。

## 数据到损失的脉络

```text
conversations
  └─> pre_processing_chat
        └─> tokenizer.apply_chat_template
              └─> input_ids
                    ├─> system/user/tool token: label = -100
                    └─> assistant token: label = token id
                          └─> MiniMindForCausalLM
                                └─> assistant-only CE + MoE aux
```

完整对话仍作为模型输入，所以 assistant 可以关注 system prompt、用户问题和工具结果；但这些上下文位置的 label 是 `-100`，不会要求模型学习复述它们。

## 启动流程

1. 初始化单进程或 DDP 环境并设置种子。
2. 创建输出目录和 `MiniMindConfig`。
3. 可选读取 `full_sft` 续训断点。
4. 设置 autocast、GradScaler 和 SwanLab。
5. `init_model` 默认加载 `pretrain` 权重。
6. 创建 `SFTDataset`、Sampler 和 AdamW。
7. 恢复模型/优化器/Scaler 状态。
8. 可选 compile，随后 DDP 包装。
9. 逐 epoch 建立带 skip 能力的 DataLoader 并训练。
10. DDP barrier 后销毁进程组。

## 单个训练 Step

前向返回的 `res.loss` 已经是移位后的因果语言模型交叉熵。脚本再加 `res.aux_loss`，除以累积步数后反向：

```text
loss = (language_model_loss + router_aux_loss) / accumulation_steps
```

日志会把总 loss 拆成：

- `logits_loss`：总 loss 减去 aux loss。
- `aux_loss`：MoE 路由辅助项；Dense 模型通常为 0。

这样可以区分“语言建模是否改善”和“路由正则项占多大比例”。

## 断点恢复为何需要跳过 Batch

恢复数据不仅要恢复模型参数，还要避免在当前 epoch 重新训练已经完成的 batch。脚本按 epoch 固定随机种子生成样本顺序，再用 `SkipBatchSampler(skip_batches=start_step)` 跳过前面 batch。

若 DDP world size 变化，`lm_checkpoint` 会换算 step。这个机制力求从相近数据位置继续，但由于 batch 划分改变，不能保证逐样本完全一致。

## 技术点解释

### Full SFT 与 LoRA 的区别

Full SFT 更新模型全部可训练参数，能力迁移更充分，但显存、优化器状态和保存体积更大。LoRA 冻结基模只训练低秩分支，更轻量。两者使用相同 `SFTDataset`，核心差别是“哪些参数参与优化”。

### 为什么学习率远低于预训练

模型已经从预训练获得通用表示。SFT 学习率过高容易快速破坏既有能力，因此默认 `1e-5`，明显低于预训练的 `5e-4`。

### Tool Call 如何进入 SFT

工具定义和调用由 Tokenizer 聊天模板编码。`SFTDataset` 会解析字符串形式的 tools/tool_calls，并把 assistant 生成的工具调用当作监督内容。训练循环无需单独识别工具协议。

## 保存结果

- 推理权重：`../out/full_sft_<hidden_size>[_moe].pth`。
- 续训断点：`../checkpoints/full_sft_<hidden_size>[_moe]_resume.pth`。

推理权重只包含模型；续训断点还包含优化器、Scaler、epoch、step 和实验 id。

## 注意事项

- 必须让基础 `pretrain` 权重与当前模型尺寸、层数和 MoE 配置匹配。
- 默认从 `trainer/` 运行；从其他目录启动会改变所有相对路径。
- 对话字段必须与 `SFTDataset` 固定 schema 兼容，字符串 JSON 必须可解析。
- `history`、thinking 和工具行为来自数据与聊天模板，训练循环本身不理解自然语言角色。
- 训练样本被截断时，末尾 assistant 内容可能缺失，应根据数据长度分布选择 `max_seq_len`。
