# `convert_model.py` 介绍

对应源码：[`convert_model.py`](./convert_model.py)

## 文件定位

模型格式转换、LoRA 合并和聊天模板转换工具集。

## 主要函数

| 函数 | 作用 |
| --- | --- |
| `convert_torch2transformers_minimind` | 将 `.pth` 转为保留 MiniMind 自定义结构的 Transformers 目录 |
| `convert_torch2transformers` | 将原生权重转换为 Qwen3/Qwen3MoE 兼容结构 |
| `convert_transformers2torch` | 从 Transformers 模型提取半精度 `state_dict` |
| `convert_merge_base_lora` | 将基础权重与 LoRA 权重合并为完整 `.pth` |
| `convert_jinja_to_json` | 将 Jinja 模板转义为可写入 JSON 的字符串 |
| `convert_json_to_jinja` | 从 `tokenizer_config.json` 提取 `chat_template` |

转换为 Transformers 格式时会一并保存 Tokenizer，并针对 Transformers 5.x 修正 Tokenizer 类名和 RoPE 配置。MoE 转换还会重排专家权重以匹配 Qwen3MoE 的参数布局。

## 运行说明

文件末尾是需要按实际路径编辑的示例入口。默认配置会把原生 PyTorch 权重转换为 Qwen3 兼容 Transformers 格式：

```bash
cd scripts
python convert_model.py
```

## 项目脉络

训练脚本输出的是轻量 `.pth`，而推理生态常使用 Transformers 目录。这个文件连接两种模型资产形态：

```text
trainer/train_*.py
  └─> out/*.pth
        ├─> MiniMind 自定义 Transformers 结构
        ├─> Qwen3 / Qwen3MoE 兼容结构
        ├─> 合并 LoRA 后的新 .pth
        └─> 从 Transformers 再提取 .pth
```

模型转换不是重新训练，也不改变权重表达的数学功能；它主要调整配置文件、权重键名/布局和保存格式，使不同加载器能够理解同一组参数。

## 两条 PyTorch → Transformers 路线

### 保留 MiniMind 自定义结构

`convert_torch2transformers_minimind` 的流程是：

1. 注册 `MiniMindConfig` 和 `MiniMindForCausalLM` 的 AutoClass。
2. 根据全局 `lm_config` 构造模型。
3. 读取 `.pth` 并加载 `state_dict`。
4. 转换到目标 dtype。
5. 调用 `save_pretrained(..., safe_serialization=False)`。
6. 复制项目 Tokenizer 到目标目录。

这种格式保留自定义 Python 模型代码。加载时通常需要 `trust_remote_code=True`，优点是实现与项目原始结构一致。

### 转为 Qwen3 兼容结构

`convert_torch2transformers` 根据 MiniMind 配置创建 `Qwen3Config` 或 `Qwen3MoeConfig`。MiniMind 的主要参数命名和层结构与目标结构对齐，因此 Dense 权重可以严格加载。

MoE 在 Transformers 5.x 中对专家矩阵采用打包布局，代码会把每层各专家的：

- `gate_proj` 和 `up_proj` 堆叠、拼接为 `gate_up_proj`。
- `down_proj` 堆叠为统一专家张量。

这一步只重排张量，不重新计算训练结果。随后使用 `strict=True` 加载，确保没有漏键或多余键。

## Transformers 5.x 兼容处理

两个导出函数都会在检测到主版本号不小于 5 时修改输出 JSON：

- 把 `tokenizer_class` 设为 `PreTrainedTokenizerFast`。
- 清空 `extra_special_tokens`。
- 显式写回 `rope_theta`。
- 把 `rope_scaling` 设为 `null`。
- 删除生成配置中的 `rope_parameters`。

这些改动是为了让当前项目配置适配不同 Transformers 版本的字段约定。它们直接编辑输出目录中的 JSON，不修改源码模型权重。

## 反向转换与 LoRA 合并

### `convert_transformers2torch`

该函数用 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 恢复模型，然后把 `state_dict` 的每个张量转到 CPU 和 FP16，保存为单个 `.pth`。输出只包含权重，不会自动把 Transformers 配置嵌入 `.pth`。

### `convert_merge_base_lora`

函数根据 `lm_config` 构造基础模型、加载基模权重、注入 LoRA，再调用 `merge_lora` 把 `B @ A` 合到原权重。输出可作为普通完整模型使用，不再需要单独 LoRA 文件。

## 聊天模板转换

Tokenizer 的 `chat_template` 是一段 Jinja 字符串：

- `convert_json_to_jinja` 从 `tokenizer_config.json` 读取该字段并写入独立文件，便于编辑。
- `convert_jinja_to_json` 读取独立模板并用 `json.dumps` 转义，打印可粘贴回 JSON 的键值片段。

前者会写文件，后者只打印结果，不自动修改配置。

## 关键配置与依赖

所有主要转换函数都依赖模块级变量 `lm_config`。这个变量只在 `if __name__ == '__main__'` 中初始化，因此若从其他脚本导入并调用转换函数，调用方需要先设置兼容的 `convert_model.lm_config`，或改造函数签名显式传入配置。

Tokenizer 默认从 `../model/` 读取，路径同样按 `scripts/` 当前目录解析。

## 注意事项

- 转换前必须让 `hidden_size`、层数、Dense/MoE 等配置与源权重完全一致。
- 默认入口中的源路径、目标路径和调用函数都是示例，应先核对再运行，避免覆盖已有模型目录。
- `strict=True` 是重要的结构校验；不要为了“能加载”随意改成宽松模式而忽略缺失权重。
- FP16 能减小体积，但可能不适合所有 CPU 推理后端。
- `trust_remote_code=True` 会执行模型目录中的 Python 代码，只应加载可信来源。
- `.pth` 基于 PyTorch 序列化机制，也只应加载可信文件。
