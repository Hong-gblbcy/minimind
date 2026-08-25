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

## 项目脉络

LoRA 位于“基础模型”和“领域微调权重”之间：

```text
full_sft 基础权重
  └─> apply_lora
        └─> 冻结基础参数，只训练 A/B
              └─> save_lora
                    ├─> eval_llm: 基模 + LoRA 组合推理
                    └─> merge_lora: 合并为新的完整权重
```

文件本身只提供结构和权重操作，不负责冻结参数、创建优化器或读取数据。这些训练职责放在 `trainer/train_lora.py` 中。这样的分工让相同 LoRA 逻辑可以同时被训练、命令行推理、API 服务和模型转换脚本复用。

## LoRA 原理

标准线性层计算：

```text
y = W x
```

LoRA 不直接更新完整矩阵 `W`，而是学习一个低秩增量：

```text
y = W x + B(Ax)
ΔW = B @ A
```

若原始层形状为 `out_features × in_features`，则：

- A 的形状为 `rank × in_features`。
- B 的形状为 `out_features × rank`。
- `rank` 远小于输入输出维度时，新增参数量从 `out × in` 降为 `rank × (in + out)`。

这使微调只需保存和更新很小的增量参数，同时保留基础模型能力。

## 代码执行脉络

### `LoRA`

构造函数创建 A、B 两个线性层。A 采用均值 0、标准差 0.02 的高斯初始化，B 全零初始化，因此刚注入时：

```text
B(Ax) = 0
```

模型输出在注入前后完全一致，训练可以从基础模型行为平滑开始。随着 B 首先获得梯度，低秩分支逐渐学习有效增量。

### `apply_lora`

函数遍历 `model.named_modules()`，只处理满足下面条件的模块：

```python
isinstance(module, nn.Linear) and module.in_features == module.out_features
```

这意味着当前实现只给方形线性层注入 LoRA。它把新建的 `LoRA` 模块保存为原层的 `lora` 子模块，并保存原始 `forward`，然后把前向替换为：

```python
original_forward(x) + lora(x)
```

闭包通过默认参数显式绑定当前层和当前 LoRA 实例，避免 Python 循环闭包全部引用最后一个模块的问题。

### `save_lora`

保存时先解开可能存在的 `torch.compile` 包装，再遍历所有带 `lora` 属性的模块，只收集 A/B 权重。参数会转到 CPU 并转成 FP16，以减少保存体积。

输出键类似：

```text
model.layers.0.self_attn.q_proj.lora.A.weight
model.layers.0.self_attn.q_proj.lora.B.weight
```

基础权重不会写入 LoRA 文件，所以该文件不能脱离对应基模单独推理。

### `load_lora`

加载时先去掉 DDP 可能添加的 `module.` 前缀，再按模块路径筛出该层的 A/B 参数。调用前必须先执行 `apply_lora`，否则模型中没有接收这些参数的 `lora` 子模块。

### `merge_lora`

合并步骤是：

1. 给模型注入并加载 LoRA。
2. 复制不含 `.lora.` 的基础 `state_dict`。
3. 对每个已注入的线性层计算 `B.weight @ A.weight`。
4. 把结果加到原始 `weight`，保存新的完整模型权重。

合并后推理只需普通 MiniMind 结构，不再执行额外的 A/B 两次线性运算，适合部署和格式转换。

## 技术特点与取舍

### 这是简化版 LoRA

许多 LoRA 实现还包含 `alpha / rank` 缩放、LoRA dropout、目标模块配置和多适配器管理。当前实现没有这些机制，增量直接按 `B(Ax)` 加到原输出，优点是代码短、原理直观，代价是可配置性较少。

### 只注入方形线性层

MiniMind 中注意力投影和部分前馈投影并非全部都是方形矩阵，因此当前筛选条件不会覆盖所有 `nn.Linear`。训练参数量和适配能力都由这条条件决定，不能把它等同于“给模型所有线性层加 LoRA”。

### Monkey Patch 与编译

直接替换模块实例的 `forward` 是一种 monkey patch。它简单有效，但动态图编译器难以稳定追踪这种运行时闭包，所以 `train_lora.py` 会主动关闭 `torch.compile`。

## 输入输出与注意事项

| 操作 | 前置条件 | 输出 |
| --- | --- | --- |
| 注入 | 已构造基础模型 | 带 `lora` 子模块的内存模型 |
| 保存 | 已注入并训练 | 仅包含 A/B 的 `.pth` |
| 加载 | 相同结构基模已注入 | 基模与 LoRA 的组合模型 |
| 合并 | 基模与匹配的 LoRA 文件 | 独立完整 `.pth` |

- 基模的隐藏维度、层数、Dense/MoE 结构必须与 LoRA 训练时一致。
- LoRA 名称本身不携带结构元数据，文件名约定不能替代配置校验。
- `torch.load` 加载的是本地可信权重；不要加载来源不明的 pickle 格式文件。
- 合并输出为 FP16 CPU 权重，后续仍需使用匹配的 `MiniMindConfig` 构造模型。
