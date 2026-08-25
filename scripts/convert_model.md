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

